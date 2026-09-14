"""Tokenize requirements and batch complete, untruncated AST graphs."""

import math
import random
from collections.abc import Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .config import EDGE_TYPE_TO_ID
from .vocabulary import Vocabularies


class FusionDataset(Dataset):
    def __init__(
        self,
        examples: list[dict],
        vocabularies: Vocabularies,
        tokenizer,
        max_length: int,
    ):
        self.examples = examples
        self.vocabularies = vocabularies
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.encoded_requirements = []
        self.full_token_lengths = []
        self.node_counts = []

        for example in examples:
            full = tokenizer(
                example["requirement"],
                add_special_tokens=True,
                truncation=False,
            )
            encoded = tokenizer(
                example["requirement"],
                add_special_tokens=True,
                truncation=True,
                max_length=max_length,
            )
            self.full_token_lengths.append(len(full["input_ids"]))
            self.encoded_requirements.append(encoded)
            self.node_counts.append(len(example["graph"]["node_types"]))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        example = self.examples[index]
        graph = example["graph"]
        return {
            "example_index": index,
            "input_ids": self.encoded_requirements[index]["input_ids"],
            "attention_mask": self.encoded_requirements[index]["attention_mask"],
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
    """Batch whole graphs under a total-node budget without truncating them."""

    def __init__(
        self,
        node_counts: Sequence[int],
        max_graphs: int,
        max_total_nodes: int,
        shuffle: bool,
        random_state: int,
    ):
        if max_graphs < 1:
            raise ValueError("max_graphs must be at least 1.")
        if max_total_nodes < 1:
            raise ValueError("max_total_nodes must be at least 1.")
        self.node_counts = list(node_counts)
        self.max_graphs = max_graphs
        self.max_total_nodes = max_total_nodes
        self.shuffle = shuffle
        self.random_state = random_state
        self.epoch = 0

    def _ordered_indices(self, epoch: int) -> list[int]:
        indices = list(range(len(self.node_counts)))
        if not self.shuffle:
            return indices

        rng = random.Random(self.random_state + epoch)
        rng.shuffle(indices)
        # Local sorting reduces padding/size variance without globally ordering labels.
        bucket_size = max(self.max_graphs * 20, 20)
        buckets = [
            indices[start : start + bucket_size]
            for start in range(0, len(indices), bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=self.node_counts.__getitem__)
        rng.shuffle(buckets)
        return [index for bucket in buckets for index in bucket]

    def _batches(self, indices: Sequence[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        current: list[int] = []
        current_nodes = 0
        for index in indices:
            graph_nodes = self.node_counts[index]
            exceeds_budget = current and (
                len(current) >= self.max_graphs
                or current_nodes + graph_nodes > self.max_total_nodes
            )
            if exceeds_budget:
                batches.append(current)
                current = []
                current_nodes = 0
            current.append(index)
            current_nodes += graph_nodes
            # A single oversized graph is retained whole and runs alone.
            if graph_nodes > self.max_total_nodes:
                batches.append(current)
                current = []
                current_nodes = 0
        if current:
            batches.append(current)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        batches = self._batches(self._ordered_indices(self.epoch))
        if self.shuffle:
            self.epoch += 1
        yield from batches

    def __len__(self) -> int:
        # The exact count can vary slightly after shuffled bucket ordering.
        return len(self._batches(self._ordered_indices(0)))


class FusionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items: Sequence[dict]) -> dict[str, torch.Tensor]:
        token_batch = self.tokenizer.pad(
            [
                {
                    "input_ids": item["input_ids"],
                    "attention_mask": item["attention_mask"],
                }
                for item in items
            ],
            padding=True,
            return_tensors="pt",
        )

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
            "input_ids": token_batch["input_ids"],
            "attention_mask": token_batch["attention_mask"],
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
            "labels": torch.tensor(
                [item["label"] for item in items],
                dtype=torch.long,
            ),
        }


def length_audit(dataset: FusionDataset) -> dict:
    def summarize(values: list[int]) -> dict:
        return {
            "minimum": min(values),
            "median": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "maximum": max(values),
        }

    return {
        "requirement_tokens": {
            **summarize(dataset.full_token_lengths),
            "max_length": dataset.max_length,
            "truncated_examples": sum(
                length > dataset.max_length
                for length in dataset.full_token_lengths
            ),
        },
        "ast_nodes": {
            **summarize(dataset.node_counts),
            "truncated_examples": 0,
            "truncation_policy": "none",
        },
    }
