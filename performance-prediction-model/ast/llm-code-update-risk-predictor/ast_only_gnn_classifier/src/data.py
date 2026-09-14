"""Load fixed labels while exposing only the old-code AST as a feature."""

from collections import Counter
from pathlib import Path

from .ast_graph import build_ast_graph
from .config import ID_TO_LABEL, REQUIRED_FIXED_FIELDS
from .io_utils import load_json


def _safe_example(record: dict, context: str, source_index: int) -> dict:
    missing = sorted(REQUIRED_FIXED_FIELDS - set(record))
    if missing:
        raise ValueError(f"{context} is missing fields: {missing}")
    for field in ("repo_full_name", "file_path", "function_name", "old_function_code"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise ValueError(f"{context} has invalid {field}.")
    score = record["ast_delta_similarity"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(f"{context} has invalid ast_delta_similarity.")
    if not 0.0 <= float(score) <= 1.0:
        raise ValueError(f"{context} score is outside [0, 1].")
    label_id = record["label_id"]
    if label_id not in ID_TO_LABEL:
        raise ValueError(f"{context} has invalid label_id {label_id}.")
    label = ID_TO_LABEL[label_id]
    if record["similarity_label"] != label:
        raise ValueError(f"{context} label name and ID disagree.")

    # Requirement and every update-derived field are deliberately not copied.
    return {
        "source_index": source_index,
        "repo_full_name": record["repo_full_name"],
        "file_path": record["file_path"],
        "function_name": record["function_name"],
        "old_function_code": record["old_function_code"],
        "ast_delta_similarity": float(score),
        "similarity_label": label,
        "label_id": int(label_id),
        "graph": build_ast_graph(record["old_function_code"]),
    }


def _identity(example: dict) -> tuple:
    return (
        example["repo_full_name"],
        example["file_path"],
        example["function_name"],
        example["ast_delta_similarity"],
    )


def load_fixed_splits(split_dir: str | Path) -> tuple[dict[str, list[dict]], dict]:
    split_path = Path(split_dir)
    splits: dict[str, list[dict]] = {}
    source_index = 0
    for split_name in ("train", "validation", "test"):
        records = load_json(split_path / f"{split_name}_data.json")
        if not isinstance(records, list):
            raise ValueError(f"{split_name}_data.json must contain a JSON list.")
        examples = []
        for index, record in enumerate(records):
            examples.append(
                _safe_example(record, f"{split_name}[{index}]", source_index)
            )
            source_index += 1
        splits[split_name] = examples

    identities = {name: {_identity(row) for row in rows} for name, rows in splits.items()}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        if identities[left] & identities[right]:
            raise ValueError(f"Exact records overlap between {left} and {right}.")

    thresholds = load_json(split_path / "percentile_thresholds.json")
    if set(thresholds) != {"p33", "p66"}:
        raise ValueError("percentile_thresholds.json must contain p33 and p66.")
    return splits, {key: float(value) for key, value in thresholds.items()}


def class_distribution(examples: list[dict]) -> dict[str, int]:
    counts = Counter(example["label_id"] for example in examples)
    return {
        label: int(counts.get(label_id, 0))
        for label_id, label in ID_TO_LABEL.items()
    }


def serializable_manifest(example: dict) -> dict:
    return {
        "source_index": example["source_index"],
        "repo_full_name": example["repo_full_name"],
        "file_path": example["file_path"],
        "function_name": example["function_name"],
        "ast_delta_similarity": example["ast_delta_similarity"],
        "similarity_label": example["similarity_label"],
        "label_id": example["label_id"],
    }


def split_overlap_audit(splits: dict[str, list[dict]]) -> dict:
    key_functions = {
        "repository": lambda row: row["repo_full_name"],
        "function_identity": lambda row: (
            row["repo_full_name"],
            row["file_path"],
            row["function_name"],
        ),
        "old_function_code": lambda row: row["old_function_code"].strip(),
    }
    audit = {}
    for key_name, key_function in key_functions.items():
        values = {
            name: {key_function(row) for row in rows}
            for name, rows in splits.items()
        }
        comparisons = {}
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        ):
            comparisons[f"{left}_vs_{right}"] = {
                "overlapping_unique_values": len(values[left] & values[right]),
                f"rows_in_{right}_seen_in_{left}": sum(
                    key_function(row) in values[left] for row in splits[right]
                ),
            }
        audit[key_name] = {
            "unique_values": {name: len(items) for name, items in values.items()},
            "comparisons": comparisons,
        }
    return audit
