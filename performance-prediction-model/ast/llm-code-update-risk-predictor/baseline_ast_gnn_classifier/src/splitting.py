"""Leakage-resistant splitting and train-only percentile labels."""

import random
from collections import Counter

import numpy as np

from .config import ID_TO_LABEL, LABEL_TO_ID


def split_dataset(
    examples: list[dict],
    random_state: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Randomly split raw examples 70/10/20 before computing labels."""
    if len(examples) < 10:
        raise ValueError("At least 10 valid examples are required.")

    shuffled = list(examples)
    random.Random(random_state).shuffle(shuffled)
    test_start = int(len(shuffled) * 0.8)
    validation_start = int(len(shuffled) * 0.7)
    return (
        shuffled[:validation_start],
        shuffled[validation_start:test_start],
        shuffled[test_start:],
    )


def compute_train_thresholds(train_examples: list[dict]) -> dict[str, float]:
    scores = [example["ast_delta_similarity"] for example in train_examples]
    return {
        "p33": float(np.percentile(scores, 33)),
        "p66": float(np.percentile(scores, 66)),
    }


def assign_labels(examples: list[dict], thresholds: dict[str, float]) -> None:
    """Assign labels in place using thresholds learned from training only."""
    for example in examples:
        score = example["ast_delta_similarity"]
        if score <= thresholds["p33"]:
            label = "low"
        elif score <= thresholds["p66"]:
            label = "mid"
        else:
            label = "high"
        example["similarity_label"] = label
        example["label_id"] = LABEL_TO_ID[label]


def get_class_distribution(examples: list[dict]) -> dict[str, int]:
    counts = Counter(example["label_id"] for example in examples)
    return {
        label: int(counts.get(label_id, 0))
        for label_id, label in ID_TO_LABEL.items()
    }
