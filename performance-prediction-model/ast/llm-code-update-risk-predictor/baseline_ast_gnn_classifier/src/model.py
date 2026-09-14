"""Message-passing GNN over the old function's AST."""

import torch
from torch import nn
from torch.nn import functional as F


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


class GraphMessagePassing(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_projection = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_projection = nn.Linear(hidden_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        neighbor_sum = torch.zeros_like(node_states)
        degree = node_states.new_zeros((node_states.size(0), 1))
        if edge_index.numel() > 0:
            sources, targets = edge_index
            neighbor_sum.index_add_(0, targets, node_states[sources])
            degree.index_add_(
                0,
                targets,
                node_states.new_ones((targets.size(0), 1)),
            )
        neighbor_mean = neighbor_sum / degree.clamp_min(1.0)
        updated = self.self_projection(node_states)
        updated = updated + self.neighbor_projection(neighbor_mean)
        return self.dropout(F.gelu(self.normalization(updated)))


class AstGnnClassifier(nn.Module):
    def __init__(
        self,
        node_type_vocab_size: int,
        field_vocab_size: int,
        lexical_vocab_size: int,
        hidden_dim: int,
        lexical_dim: int,
        gnn_layers: int,
        dropout: float,
        num_labels: int,
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
        self.gnn_layers = nn.ModuleList(
            GraphMessagePassing(hidden_dim, dropout)
            for _ in range(gnn_layers)
        )
        self.requirement_projection = nn.Linear(lexical_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
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
        lexical_ids: torch.Tensor,
        lexical_node_indices: torch.Tensor,
        requirement_ids: torch.Tensor,
        requirement_batch: torch.Tensor,
    ) -> torch.Tensor:
        graph_count = int(graph_batch.max().item()) + 1
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

        for layer in self.gnn_layers:
            node_states = node_states + layer(node_states, edge_index)

        graph_mean = _scatter_mean(node_states, graph_batch, graph_count)
        graph_max = _graph_max(node_states, graph_batch, graph_count)
        requirement_states = self.lexical_embedding(requirement_ids)
        requirement_mean = _scatter_mean(
            requirement_states,
            requirement_batch,
            graph_count,
        )
        requirement_summary = F.gelu(
            self.requirement_projection(requirement_mean)
        )
        return self.classifier(
            torch.cat([graph_mean, graph_max, requirement_summary], dim=-1)
        )
