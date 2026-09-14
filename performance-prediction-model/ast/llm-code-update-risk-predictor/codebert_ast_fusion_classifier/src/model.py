"""Partially frozen CodeBERT fused with a relational old-code AST GNN."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel


def _scatter_mean(
    values: torch.Tensor,
    groups: torch.Tensor,
    group_count: int,
) -> torch.Tensor:
    output = values.new_zeros((group_count, values.size(-1)))
    output.index_add_(0, groups, values)
    counts = values.new_zeros((group_count, 1))
    counts.index_add_(0, groups, values.new_ones((values.size(0), 1)))
    return output / counts.clamp_min(1.0)


def _graph_max(
    values: torch.Tensor,
    groups: torch.Tensor,
    graph_count: int,
) -> torch.Tensor:
    return torch.stack(
        [values[groups == index].max(dim=0).values for index in range(graph_count)]
    )


def _encoder_layers(codebert: nn.Module) -> nn.ModuleList:
    encoder = getattr(codebert, "encoder", None)
    layers = getattr(encoder, "layer", None)
    if layers is None:
        raise ValueError(
            "The selected CodeBERT-compatible model does not expose encoder.layer."
        )
    return layers


def configure_codebert_trainability(
    codebert: nn.Module,
    unfrozen_layers: int,
    train_embeddings: bool,
) -> dict:
    """Freeze the encoder, then selectively unfreeze its top layers."""
    layers = _encoder_layers(codebert)
    if not 0 <= unfrozen_layers <= len(layers):
        raise ValueError(
            f"unfrozen_layers must be between 0 and {len(layers)}, "
            f"received {unfrozen_layers}."
        )
    for parameter in codebert.parameters():
        parameter.requires_grad = False
    if unfrozen_layers:
        for layer in layers[-unfrozen_layers:]:
            for parameter in layer.parameters():
                parameter.requires_grad = True

    embeddings = getattr(codebert, "embeddings", None)
    if train_embeddings:
        if embeddings is None:
            raise ValueError("The selected model does not expose embeddings.")
        for parameter in embeddings.parameters():
            parameter.requires_grad = True

    return {
        "total_encoder_layers": len(layers),
        "unfrozen_encoder_layers": unfrozen_layers,
        "frozen_encoder_layers": len(layers) - unfrozen_layers,
        "train_embeddings": train_embeddings,
    }


class RelationalGraphMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.relation_projections = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        message_sum = torch.zeros_like(node_states)
        degree = node_states.new_zeros((node_states.size(0), 1))
        if edge_index.numel() > 0:
            sources, targets = edge_index
            for relation_id, projection in enumerate(self.relation_projections):
                relation_mask = edge_type_ids == relation_id
                if not torch.any(relation_mask):
                    continue
                relation_sources = sources[relation_mask]
                relation_targets = targets[relation_mask]
                messages = projection(node_states[relation_sources])
                message_sum.index_add_(0, relation_targets, messages)
                degree.index_add_(
                    0,
                    relation_targets,
                    messages.new_ones((messages.size(0), 1)),
                )
        neighbor_state = message_sum / degree.clamp_min(1.0)
        updated = self.self_projection(node_states) + neighbor_state
        return self.dropout(F.gelu(self.normalization(updated)))


class AstEncoder(nn.Module):
    def __init__(
        self,
        node_type_vocab_size: int,
        field_vocab_size: int,
        lexical_vocab_size: int,
        hidden_dim: int,
        lexical_dim: int,
        gnn_layers: int,
        dropout: float,
        num_relations: int,
        fusion_dim: int,
    ):
        super().__init__()
        self.node_type_embedding = nn.Embedding(
            node_type_vocab_size,
            hidden_dim,
            padding_idx=0,
        )
        self.field_embedding = nn.Embedding(
            field_vocab_size,
            hidden_dim,
            padding_idx=0,
        )
        self.lexical_embedding = nn.Embedding(
            lexical_vocab_size,
            lexical_dim,
            padding_idx=0,
        )
        self.lexical_projection = nn.Linear(lexical_dim, hidden_dim)
        self.depth_projection = nn.Linear(1, hidden_dim)
        self.input_normalization = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList(
            RelationalGraphMessagePassing(hidden_dim, num_relations, dropout)
            for _ in range(gnn_layers)
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )

    def _node_lexical_states(
        self,
        lexical_ids: torch.Tensor,
        lexical_node_indices: torch.Tensor,
        node_count: int,
    ) -> torch.Tensor:
        states = self.lexical_embedding.weight.new_zeros(
            (node_count, self.lexical_embedding.embedding_dim)
        )
        counts = states.new_zeros((node_count, 1))
        if lexical_ids.numel() > 0:
            embedded = self.lexical_embedding(lexical_ids)
            states.index_add_(0, lexical_node_indices, embedded)
            counts.index_add_(
                0,
                lexical_node_indices,
                embedded.new_ones((embedded.size(0), 1)),
            )
        return states / counts.clamp_min(1.0)

    def forward(
        self,
        node_type_ids: torch.Tensor,
        field_ids: torch.Tensor,
        depths: torch.Tensor,
        graph_batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        lexical_ids: torch.Tensor,
        lexical_node_indices: torch.Tensor,
        graph_count: int,
    ) -> torch.Tensor:
        node_lexical = self._node_lexical_states(
            lexical_ids,
            lexical_node_indices,
            node_type_ids.size(0),
        )
        node_states = self.node_type_embedding(node_type_ids)
        node_states = node_states + self.field_embedding(field_ids)
        node_states = node_states + self.lexical_projection(node_lexical)
        node_states = node_states + self.depth_projection(depths)
        node_states = self.input_normalization(node_states)

        for layer in self.layers:
            node_states = node_states + layer(
                node_states,
                edge_index,
                edge_type_ids,
            )

        graph_mean = _scatter_mean(node_states, graph_batch, graph_count)
        graph_max = _graph_max(node_states, graph_batch, graph_count)
        return self.graph_projection(torch.cat([graph_mean, graph_max], dim=-1))


class CodeBertAstFusionClassifier(nn.Module):
    def __init__(
        self,
        codebert_model: str,
        node_type_vocab_size: int,
        field_vocab_size: int,
        lexical_vocab_size: int,
        hidden_dim: int = 128,
        lexical_dim: int = 64,
        gnn_layers: int = 2,
        fusion_dim: int = 128,
        fusion_hidden_dim: int = 128,
        dropout: float = 0.3,
        num_relations: int = 4,
        num_labels: int = 3,
        unfrozen_codebert_layers: int = 2,
        train_codebert_embeddings: bool = False,
    ):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(codebert_model)
        self.trainability = configure_codebert_trainability(
            self.codebert,
            unfrozen_layers=unfrozen_codebert_layers,
            train_embeddings=train_codebert_embeddings,
        )
        codebert_hidden = int(self.codebert.config.hidden_size)
        self.requirement_projection = nn.Sequential(
            nn.Linear(codebert_hidden, fusion_dim),
            nn.GELU(),
            nn.LayerNorm(fusion_dim),
        )
        self.ast_encoder = AstEncoder(
            node_type_vocab_size=node_type_vocab_size,
            field_vocab_size=field_vocab_size,
            lexical_vocab_size=lexical_vocab_size,
            hidden_dim=hidden_dim,
            lexical_dim=lexical_dim,
            gnn_layers=gnn_layers,
            dropout=dropout,
            num_relations=num_relations,
            fusion_dim=fusion_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, num_labels),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        node_type_ids: torch.Tensor,
        field_ids: torch.Tensor,
        depths: torch.Tensor,
        graph_batch: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type_ids: torch.Tensor,
        lexical_ids: torch.Tensor,
        lexical_node_indices: torch.Tensor,
    ) -> torch.Tensor:
        codebert_output = self.codebert(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # RoBERTa/CodeBERT uses the first <s> token as its sequence representation.
        requirement_embedding = self.requirement_projection(
            codebert_output.last_hidden_state[:, 0, :]
        )
        ast_embedding = self.ast_encoder(
            node_type_ids=node_type_ids,
            field_ids=field_ids,
            depths=depths,
            graph_batch=graph_batch,
            edge_index=edge_index,
            edge_type_ids=edge_type_ids,
            lexical_ids=lexical_ids,
            lexical_node_indices=lexical_node_indices,
            graph_count=input_ids.size(0),
        )
        fused = torch.cat([requirement_embedding, ast_embedding], dim=-1)
        return self.classifier(fused)


def parameter_counts(model: nn.Module) -> dict[str, int]:
    codebert_total = sum(parameter.numel() for parameter in model.codebert.parameters())
    codebert_trainable = sum(
        parameter.numel()
        for parameter in model.codebert.parameters()
        if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
        "codebert_total": codebert_total,
        "codebert_trainable": codebert_trainable,
        "non_codebert_trainable": trainable - codebert_trainable,
    }
