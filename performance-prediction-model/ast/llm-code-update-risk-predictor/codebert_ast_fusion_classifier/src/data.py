"""Leakage-aware loading, validation, splitting, and audits."""

import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .ast_graph import build_ast_graph
from .config import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    REQUIRED_FIELDS,
    REQUIRED_FIXED_SPLIT_FIELDS,
)
from .io_utils import load_json


def _validate_common(record: dict, context: str) -> None:
    missing = sorted(REQUIRED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{context} is missing required fields: {missing}")
    for field in ("repo_full_name", "file_path", "function_name", "old_function_code"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{context} has invalid {field}.")
    if not isinstance(record["requirement"], str) or not record["requirement"].strip():
        raise ValueError(f"{context} has an invalid requirement.")
    score = record["ast_delta_similarity"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{context} has non-numeric ast_delta_similarity.")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError(f"{context} has ast_delta_similarity outside [0, 1].")


def _safe_example(record: dict, context: str, source_index: int) -> dict:
    """Copy only inference-time inputs, target, and non-feature metadata."""
    _validate_common(record, context)
    return {
        "source_index": source_index,
        "repo_full_name": record["repo_full_name"].strip(),
        "file_path": record["file_path"].strip(),
        "function_name": record["function_name"].strip(),
        "requirement": record["requirement"].strip(),
        "old_function_code": record["old_function_code"],
        "ast_delta_similarity": float(record["ast_delta_similarity"]),
        "graph": build_ast_graph(record["old_function_code"]),
    }


def _validate_fixed_label(record: dict, context: str) -> tuple[str, int]:
    missing = sorted(REQUIRED_FIXED_SPLIT_FIELDS - set(record))
    if missing:
        raise ValueError(f"{context} is missing fixed-label fields: {missing}")
    label_id = record["label_id"]
    if label_id not in ID_TO_LABEL:
        raise ValueError(f"{context} has invalid label_id {label_id}.")
    expected = ID_TO_LABEL[label_id]
    if record["similarity_label"] != expected:
        raise ValueError(f"{context} label name and ID disagree.")
    return expected, int(label_id)


def _identity(example: dict) -> tuple:
    return (
        example["repo_full_name"],
        example["file_path"],
        example["function_name"],
        example["requirement"],
        example["ast_delta_similarity"],
    )


def load_fixed_splits(split_dir: str | Path) -> tuple[dict[str, list[dict]], dict]:
    """Load the preserved baseline splits and reconstruct old-code ASTs."""
    split_path = Path(split_dir)
    splits: dict[str, list[dict]] = {}
    source_index = 0
    for split_name in ("train", "validation", "test"):
        records = load_json(split_path / f"{split_name}_data.json")
        if not isinstance(records, list):
            raise ValueError(f"{split_name}_data.json must contain a JSON list.")
        examples = []
        for index, record in enumerate(records):
            context = f"{split_name}[{index}]"
            label, label_id = _validate_fixed_label(record, context)
            example = _safe_example(record, context, source_index)
            source_index += 1
            example["similarity_label"] = label
            example["label_id"] = label_id
            examples.append(example)
        splits[split_name] = examples

    identities = {name: {_identity(row) for row in rows} for name, rows in splits.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if identities[left] & identities[right]:
            raise ValueError(f"Exact fixed-split records overlap: {left} and {right}.")

    thresholds = load_json(split_path / "percentile_thresholds.json")
    if set(thresholds) != {"p33", "p66"}:
        raise ValueError("percentile_thresholds.json must contain p33 and p66.")
    return splits, {key: float(value) for key, value in thresholds.items()}


def load_source_examples(input_path: str | Path) -> tuple[list[dict], list[dict]]:
    """Load a raw result file while refusing update-derived fields as features."""
    payload = load_json(input_path)
    records = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError('Input JSON must contain a top-level "results" list.')

    examples: list[dict] = []
    errors: list[dict] = []
    for index, record in enumerate(records):
        try:
            examples.append(_safe_example(record, f"results[{index}]", index))
        except (ValueError, SyntaxError, TypeError) as error:
            errors.append({"source_index": index, "reason": str(error)})
    if len(examples) < 10:
        raise ValueError("At least 10 valid examples are required.")
    return examples, errors


def _compute_thresholds(train_examples: list[dict]) -> dict[str, float]:
    scores = [example["ast_delta_similarity"] for example in train_examples]
    return {
        "p33": float(np.percentile(scores, 33)),
        "p66": float(np.percentile(scores, 66)),
    }


def _assign_labels(examples: list[dict], thresholds: dict[str, float]) -> None:
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


def split_by_repository(
    examples: list[dict],
    random_state: int,
) -> tuple[dict[str, list[dict]], dict]:
    """Create deterministic 70/10/20 splits with disjoint repositories."""
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for example in examples:
        by_repo[example["repo_full_name"]].append(example)

    repositories = sorted(by_repo)
    random.Random(random_state).shuffle(repositories)
    train_target = len(examples) * 0.70
    validation_target = len(examples) * 0.10
    splits = {"train": [], "validation": [], "test": []}

    for repository in repositories:
        group = by_repo[repository]
        if len(splits["train"]) < train_target:
            destination = "train"
        elif len(splits["validation"]) < validation_target:
            destination = "validation"
        else:
            destination = "test"
        splits[destination].extend(group)

    if any(not splits[name] for name in splits):
        raise ValueError("Repository grouping produced an empty split.")

    thresholds = _compute_thresholds(splits["train"])
    for rows in splits.values():
        _assign_labels(rows, thresholds)
    return splits, thresholds


def class_distribution(examples: list[dict]) -> dict[str, int]:
    counts = Counter(example["label_id"] for example in examples)
    return {
        label: int(counts.get(label_id, 0))
        for label_id, label in ID_TO_LABEL.items()
    }


def serializable_manifest(example: dict) -> dict:
    """Save split identity and targets without expanded graphs or update outputs."""
    return {
        "source_index": example["source_index"],
        "repo_full_name": example["repo_full_name"],
        "file_path": example["file_path"],
        "function_name": example["function_name"],
        "requirement": example["requirement"],
        "ast_delta_similarity": example["ast_delta_similarity"],
        "similarity_label": example["similarity_label"],
        "label_id": example["label_id"],
    }


def _normal_requirement(example: dict) -> str:
    return " ".join(example["requirement"].lower().split())


def split_overlap_audit(splits: dict[str, list[dict]]) -> dict:
    """Measure potentially optimistic overlap without calling it label leakage."""
    key_functions = {
        "repository": lambda row: row["repo_full_name"],
        "function_identity": lambda row: (
            row["repo_full_name"],
            row["file_path"],
            row["function_name"],
        ),
        "requirement_text": _normal_requirement,
        "old_function_code": lambda row: row["old_function_code"].strip(),
    }
    audit = {}
    for key_name, key_function in key_functions.items():
        key_sets = {
            split_name: {key_function(row) for row in rows}
            for split_name, rows in splits.items()
        }
        comparisons = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            comparisons[f"{left}_vs_{right}"] = {
                "overlapping_unique_values": len(key_sets[left] & key_sets[right]),
                f"rows_in_{right}_seen_in_{left}": sum(
                    key_function(row) in key_sets[left] for row in splits[right]
                ),
            }
        audit[key_name] = {
            "unique_values": {
                split_name: len(values)
                for split_name, values in key_sets.items()
            },
            "comparisons": comparisons,
        }
    return audit
