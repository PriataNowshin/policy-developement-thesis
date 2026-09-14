"""Focused AST-only pipeline tests."""

import sys
import unittest
from pathlib import Path

import torch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT))

from src.ast_graph import build_ast_graph
from src.dataset import AstOnlyDataset, NodeBudgetBatchSampler, collate_graphs
from src.model import AstOnlyGnnClassifier
from src.vocabulary import build_train_vocabularies


def _example(source: str, label: int = 0) -> dict:
    return {
        "source_index": 0,
        "repo_full_name": "owner/repository",
        "file_path": "module.py",
        "function_name": "example",
        "old_function_code": source,
        "ast_delta_similarity": 0.1,
        "similarity_label": "low",
        "label_id": label,
        "graph": build_ast_graph(source),
    }


class AstOnlyTests(unittest.TestCase):
    def test_dataset_has_no_requirement_tensor(self):
        examples = [_example("def example(path):\n    return path")]
        vocabularies = build_train_vocabularies(examples)
        dataset = AstOnlyDataset(examples, vocabularies)
        item = dataset[0]
        self.assertNotIn("requirement", item)
        self.assertNotIn("requirement_ids", item)

    def test_complete_oversized_graph_is_kept(self):
        sampler = NodeBudgetBatchSampler(
            node_counts=[10, 5000, 20],
            max_graphs=16,
            max_total_nodes=100,
            shuffle=False,
            random_state=42,
        )
        self.assertEqual(list(sampler), [[0, 2], [1]])

    def test_model_forward_returns_three_logits(self):
        examples = [
            _example("def example(path):\n    return path"),
            _example("def other(x):\n    if x:\n        return 1\n    return 0", 1),
        ]
        vocabularies = build_train_vocabularies(examples)
        dataset = AstOnlyDataset(examples, vocabularies)
        batch = collate_graphs([dataset[0], dataset[1]])
        model = AstOnlyGnnClassifier(
            node_type_vocab_size=len(vocabularies.node_types),
            field_vocab_size=len(vocabularies.fields),
            lexical_vocab_size=len(vocabularies.lexical),
            hidden_dim=16,
            lexical_dim=8,
            gnn_layers=2,
            dropout=0.0,
            num_relations=4,
            num_labels=3,
        )
        keys = {
            key: value
            for key, value in batch.items()
            if key not in {"example_indices", "labels"}
        }
        self.assertEqual(tuple(model(**keys).shape), (2, 3))


if __name__ == "__main__":
    unittest.main()
