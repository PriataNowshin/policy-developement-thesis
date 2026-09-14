"""Encode and batch complete AST graphs without requirement features."""

import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import EDGE_TYPE_TO_ID
from .vocabulary import Vocabularies


class AstOnlyDataset(Dataset):
    def __init__(self, examples: list[dict], vocabularies: Vocabularies):
        self.examples = examples
        self.vocabularies = vocabularies
        self.node_counts = [
            len(example["graph"]["node_types"]) for example in examples
        ]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        graph = example["graph"]
        return {
            "example_index": index,
            "node_type_ids": [
                self.vocabularies.node_types.encode(value)
                for value in graph["node_types"]
            ],
            "field_ids": [
                self.vocabularies.fields.encode(value)
                for value in graph["fields"]
            ],
            "depths": graph["depths"],
            "node_lexical_ids": [
                [
                    self.vocabularies.lexical.encode(token)
                    for token in node_tokens
                ]
                for node_tokens in graph["lexemes"]
            ],
            "edges": [
                [source, target, EDGE_TYPE_TO_ID[edge_type]]
                for source, target, edge_type in graph["edges"]
            ],
            "label": example["label_id"],
        }


class NodeBudgetBatchSampler(Sampler[list[int]]):
    """Keep graphs complete while limiting total nodes in ordinary batches."""

    def __init__(
        self,
        node_counts: Sequence[int],
        max_graphs: int,
        max_total_nodes: int,
        shuffle: bool,
        random_state: int,
    ):
        if max_graphs < 1 or max_total_nodes < 1:
            raise ValueError("Batch limits must be positive.")
        self.node_counts = list(node_counts)
        self.max_graphs = max_graphs
        self.max_total_nodes = max_total_nodes
        self.shuffle = shuffle
        self.random_state = random_state
        self.epoch = 0
        ordered = sorted(range(len(node_counts)), key=self.node_counts.__getitem__)
        self.base_batches = self._make_batches(ordered)

    def _make_batches(self, indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_nodes = 0
        for index in indices:
            graph_nodes = self.node_counts[index]
            if current and (
                len(current) >= self.max_graphs
                or current_nodes + graph_nodes > self.max_total_nodes
            ):
                batches.append(current)
                current = []
                current_nodes = 0
            current.append(index)
            current_nodes += graph_nodes
            if graph_nodes > self.max_total_nodes:
                batches.append(current)
                current = []
                current_nodes = 0
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = [list(batch) for batch in self.base_batches]
        if self.shuffle:
            rng = random.Random(self.random_state + self.epoch)
            for batch in batches:
                rng.shuffle(batch)
            rng.shuffle(batches)
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        return len(self.base_batches)


def collate_graphs(items: Sequence[dict]) -> dict[str, torch.Tensor]:
    node_type_ids: list[int] = []
    field_ids: list[int] = []
    depths: list[float] = []
    graph_batch: list[int] = []
    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_type_ids: list[int] = []
    lexical_ids: list[int] = []
    lexical_node_indices: list[int] = []
    node_offset = 0

    for graph_index, item in enumerate(items):
        node_count = len(item["node_type_ids"])
        node_type_ids.extend(item["node_type_ids"])
        field_ids.extend(item["field_ids"])
        depths.extend(min(depth, 32) / 32.0 for depth in item["depths"])
        graph_batch.extend([graph_index] * node_count)
        for source, target, edge_type_id in item["edges"]:
            edge_sources.append(source + node_offset)
            edge_targets.append(target + node_offset)
            edge_type_ids.append(edge_type_id)
        for local_node_index, tokens in enumerate(item["node_lexical_ids"]):
            for token_id in tokens:
                lexical_ids.append(token_id)
                lexical_node_indices.append(node_offset + local_node_index)
        node_offset += node_count

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    if edge_index.numel() == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    return {
        "example_indices": torch.tensor(
            [item["example_index"] for item in items],
            dtype=torch.long,
        ),
        "node_type_ids": torch.tensor(node_type_ids, dtype=torch.long),
        "field_ids": torch.tensor(field_ids, dtype=torch.long),
        "depths": torch.tensor(depths, dtype=torch.float32).unsqueeze(-1),
        "graph_batch": torch.tensor(graph_batch, dtype=torch.long),
        "edge_index": edge_index,
        "edge_type_ids": torch.tensor(edge_type_ids, dtype=torch.long),
        "lexical_ids": torch.tensor(lexical_ids, dtype=torch.long),
        "lexical_node_indices": torch.tensor(
            lexical_node_indices,
            dtype=torch.long,
        ),
        "labels": torch.tensor([item["label"] for item in items], dtype=torch.long),
    }


def graph_size_audit(dataset: AstOnlyDataset) -> dict:
    values = dataset.node_counts
    return {
        "minimum": min(values),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "maximum": max(values),
        "truncated_examples": 0,
        "truncation_policy": "none",
    }
