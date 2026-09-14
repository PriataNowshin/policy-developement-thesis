"""Focused tests that do not download or train CodeBERT."""

import sys
import unittest
from pathlib import Path

import torch
from torch import nn

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from src.ast_graph import build_ast_graph
from src.data import split_by_repository, split_overlap_audit
from src.dataset import NodeBudgetBatchSampler
from src.model import configure_codebert_trainability


class _FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList(nn.Linear(4, 4) for _ in range(12))


class _FakeCodeBert(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(20, 4)
        self.encoder = _FakeEncoder()


def _example(repository: str, index: int) -> dict:
    return {
        "source_index": index,
        "repo_full_name": repository,
        "file_path": f"file_{index}.py",
        "function_name": f"function_{index}",
        "requirement": f"Change behavior {index}",
        "old_function_code": f"def function_{index}():\n    return {index}",
        "ast_delta_similarity": index / 19,
        "graph": build_ast_graph(f"def function_{index}():\n    return {index}"),
    }


class AstGraphTests(unittest.TestCase):
    def test_graph_has_typed_bidirectional_edges(self):
        graph = build_ast_graph(
            "def validate(path):\n"
            "    if path == '':\n"
            "        raise ValueError()\n"
        )
        self.assertIn("FunctionDef", graph["node_types"])
        self.assertIn("Raise", graph["node_types"])
        edge_types = {edge[2] for edge in graph["edges"]}
        self.assertIn("ast_parent_to_child", edge_types)
        self.assertIn("ast_child_to_parent", edge_types)


class NodeBudgetTests(unittest.TestCase):
    def test_oversized_graph_is_kept_whole_and_runs_alone(self):
        sampler = NodeBudgetBatchSampler(
            node_counts=[10, 5000, 20],
            max_graphs=8,
            max_total_nodes=100,
            shuffle=False,
            random_state=42,
        )
        batches = list(sampler)
        self.assertEqual(batches, [[0], [1], [2]])
        self.assertEqual(sum(len(batch) for batch in batches), 3)


class FreezingTests(unittest.TestCase):
    def test_top_two_layers_only_are_unfrozen(self):
        model = _FakeCodeBert()
        summary = configure_codebert_trainability(
            model,
            unfrozen_layers=2,
            train_embeddings=False,
        )
        self.assertEqual(summary["frozen_encoder_layers"], 10)
        self.assertFalse(model.embeddings.weight.requires_grad)
        self.assertFalse(model.encoder.layer[9].weight.requires_grad)
        self.assertTrue(model.encoder.layer[10].weight.requires_grad)
        self.assertTrue(model.encoder.layer[11].weight.requires_grad)


class GroupSplitTests(unittest.TestCase):
    def test_repository_split_is_disjoint(self):
        examples = [
            _example(f"owner/repo_{index // 2}", index)
            for index in range(20)
        ]
        splits, thresholds = split_by_repository(examples, random_state=42)
        audit = split_overlap_audit(splits)
        self.assertLessEqual(thresholds["p33"], thresholds["p66"])
        self.assertEqual(
            audit["repository"]["comparisons"]["train_vs_test"][
                "overlapping_unique_values"
            ],
            0,
        )


if __name__ == "__main__":
    unittest.main()
