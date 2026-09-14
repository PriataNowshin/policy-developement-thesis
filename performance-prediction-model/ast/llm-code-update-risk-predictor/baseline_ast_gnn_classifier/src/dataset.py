"""PyTorch datasets and batching for variable-size AST graphs."""

from collections.abc import Sequence

import torch
from torch.utils.data import Dataset

from .ast_graph import tokenize_text
from .vocabulary import Vocabularies


class AstGraphDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        vocabularies: Vocabularies,
        max_ast_nodes: int = 0,
    ):
        self.examples = examples
        self.vocabularies = vocabularies
        self.max_ast_nodes = max_ast_nodes

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        graph = example["graph"]
        node_count = len(graph["node_types"])
        if self.max_ast_nodes > 0:
            node_count = min(node_count, self.max_ast_nodes)

        edges = [
            edge
            for edge in graph["edges"]
            if edge[0] < node_count and edge[1] < node_count
        ]
        requirement_tokens = tokenize_text(example["requirement"])
        if not requirement_tokens:
            requirement_tokens = ["<unk>"]

        return {
            "example_index": index,
            "node_type_ids": [
                self.vocabularies.node_types.encode(value)
                for value in graph["node_types"][:node_count]
            ],
            "field_ids": [
                self.vocabularies.fields.encode(value)
                for value in graph["fields"][:node_count]
            ],
            "depths": graph["depths"][:node_count],
            "node_lexical_ids": [
                [self.vocabularies.lexical.encode(token) for token in tokens]
                for tokens in graph["lexemes"][:node_count]
            ],
            "edges": edges,
            "requirement_ids": [
                self.vocabularies.lexical.encode(token)
                for token in requirement_tokens
            ],
            "label": example["label_id"],
        }


def collate_graphs(items: Sequence[dict]) -> dict[str, torch.Tensor]:
    node_type_ids: list[int] = []
    field_ids: list[int] = []
    depths: list[float] = []
    graph_batch: list[int] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    lexical_ids: list[int] = []
    lexical_node_indices: list[int] = []
    requirement_ids: list[int] = []
    requirement_batch: list[int] = []
    labels: list[int] = []
    example_indices: list[int] = []

    node_offset = 0
    for graph_index, item in enumerate(items):
        node_count = len(item["node_type_ids"])
        node_type_ids.extend(item["node_type_ids"])
        field_ids.extend(item["field_ids"])
        depths.extend(min(depth, 32) / 32.0 for depth in item["depths"])
        graph_batch.extend([graph_index] * node_count)

        for source, target in item["edges"]:
            edge_sources.append(source + node_offset)
            edge_targets.append(target + node_offset)

        for local_node_index, tokens in enumerate(item["node_lexical_ids"]):
            for token_id in tokens:
                lexical_ids.append(token_id)
                lexical_node_indices.append(node_offset + local_node_index)

        requirement_ids.extend(item["requirement_ids"])
        requirement_batch.extend([graph_index] * len(item["requirement_ids"]))
        labels.append(item["label"])
        example_indices.append(item["example_index"])
        node_offset += node_count

    edge_index = torch.tensor(
        [edge_sources, edge_targets],
        dtype=torch.long,
    )
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    return {
        "example_indices": torch.tensor(example_indices, dtype=torch.long),
        "node_type_ids": torch.tensor(node_type_ids, dtype=torch.long),
        "field_ids": torch.tensor(field_ids, dtype=torch.long),
        "depths": torch.tensor(depths, dtype=torch.float32).unsqueeze(-1),
        "graph_batch": torch.tensor(graph_batch, dtype=torch.long),
        "edge_index": edge_index,
        "lexical_ids": torch.tensor(lexical_ids, dtype=torch.long),
        "lexical_node_indices": torch.tensor(
            lexical_node_indices,
            dtype=torch.long,
        ),
        "requirement_ids": torch.tensor(requirement_ids, dtype=torch.long),
        "requirement_batch": torch.tensor(requirement_batch, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }
