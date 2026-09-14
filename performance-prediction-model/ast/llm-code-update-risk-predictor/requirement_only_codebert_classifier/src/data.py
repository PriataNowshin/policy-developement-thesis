"""Load fixed splits and expose requirement-only model records."""

from pathlib import Path

from .config import ID_TO_LABEL, REQUIRED_SPLIT_FIELDS
from .io_utils import load_json


def _sanitize_record(record: dict, split_name: str, index: int) -> dict:
    missing = sorted(REQUIRED_SPLIT_FIELDS - set(record))
    if missing:
        raise ValueError(
            f"{split_name}[{index}] is missing required fields: {missing}"
        )
    requirement = record["requirement"]
    if not isinstance(requirement, str) or not requirement.strip():
        raise ValueError(f"{split_name}[{index}] has an invalid requirement.")
    label_id = record["label_id"]
    if label_id not in ID_TO_LABEL:
        raise ValueError(f"{split_name}[{index}] has invalid label_id {label_id}.")
    expected_label = ID_TO_LABEL[label_id]
    if record["similarity_label"] != expected_label:
        raise ValueError(
            f"{split_name}[{index}] label name and ID do not agree."
        )

    # old_function_code and all update fields are intentionally not copied.
    return {
        "repo_full_name": record["repo_full_name"],
        "file_path": record["file_path"],
        "function_name": record["function_name"],
        "requirement": requirement.strip(),
        "ast_delta_similarity": float(record["ast_delta_similarity"]),
        "similarity_label": expected_label,
        "label_id": int(label_id),
    }


def _record_identity(record: dict) -> tuple:
    return (
        record["repo_full_name"],
        record["file_path"],
        record["function_name"],
        record["requirement"],
        record["ast_delta_similarity"],
    )


def load_fixed_splits(split_dir: str | Path) -> tuple[dict, dict]:
    split_path = Path(split_dir)
    splits = {}
    for split_name in ("train", "validation", "test"):
        raw_records = load_json(split_path / f"{split_name}_data.json")
        if not isinstance(raw_records, list):
            raise ValueError(f"{split_name}_data.json must contain a JSON list.")
        splits[split_name] = [
            _sanitize_record(record, split_name, index)
            for index, record in enumerate(raw_records)
        ]

    identities = {
        name: {_record_identity(record) for record in records}
        for name, records in splits.items()
    }
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = identities[left] & identities[right]
        if overlap:
            raise ValueError(f"Fixed splits overlap: {left} and {right}.")

    thresholds = load_json(split_path / "percentile_thresholds.json")
    if set(thresholds) != {"p33", "p66"}:
        raise ValueError("percentile_thresholds.json must contain p33 and p66.")
    return splits, {key: float(value) for key, value in thresholds.items()}
