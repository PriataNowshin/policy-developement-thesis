"""Prepare leakage-safe synthetic splits for a separate fusion-model run."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
import random
from pathlib import Path

from src.config import LABEL_TO_ID
from src.io_utils import load_json, save_json


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Synthetic AST evaluation JSON.")
    parser.add_argument("--thresholds", required=True, help="Fixed comparable thresholds JSON.")
    parser.add_argument("--output_dir", required=True, help="New synthetic split directory.")
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def source_group_key(record):
    """Keep every mutation of the same old function in exactly one split."""
    return (
        record.get("repo_full_name", ""),
        record.get("file_path", ""),
        record.get("function_name", ""),
        record["old_function_code"].strip(),
    )


def label_for(score, thresholds):
    if score <= thresholds["p33"]:
        return "low"
    if score <= thresholds["p66"]:
        return "mid"
    return "high"


def grouped_split(records, random_state):
    groups = defaultdict(list)
    for record in records:
        groups[source_group_key(record)].append(record)
    keys = sorted(groups, key=repr)
    random.Random(random_state).shuffle(keys)
    targets = {"train": len(records) * 0.70, "validation": len(records) * 0.10}
    splits = {"train": [], "validation": [], "test": []}
    split_groups = {name: [] for name in splits}
    for key in keys:
        if len(splits["train"]) < targets["train"]:
            destination = "train"
        elif len(splits["validation"]) < targets["validation"]:
            destination = "validation"
        else:
            destination = "test"
        splits[destination].extend(groups[key])
        split_groups[destination].append(key)
    return splits, split_groups


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = load_json(args.input)
    records = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError('Input must contain a top-level "results" list.')
    thresholds = {key: float(value) for key, value in load_json(args.thresholds).items()}
    if set(thresholds) != {"p33", "p66"} or thresholds["p33"] >= thresholds["p66"]:
        raise ValueError("Thresholds must contain distinct ordered p33 and p66 values.")

    prepared = []
    for index, record in enumerate(records):
        required = ("repo_full_name", "file_path", "function_name", "requirement", "old_function_code", "ast_delta_similarity")
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"results[{index}] is missing fields: {missing}")
        label = label_for(float(record["ast_delta_similarity"]), thresholds)
        prepared.append({
            "source_index": index,
            "repo_full_name": record["repo_full_name"],
            "file_path": record["file_path"],
            "function_name": record["function_name"],
            "requirement": record["requirement"],
            "old_function_code": record["old_function_code"],
            "ast_delta_similarity": float(record["ast_delta_similarity"]),
            "similarity_label": label,
            "label_id": LABEL_TO_ID[label],
            "synthetic_id": record.get("id"),
            "mutation_operator": record.get("mutation_operator"),
            "clone_type": record.get("clone_type"),
        })

    splits, split_groups = grouped_split(prepared, args.random_state)
    group_sets = {name: set(keys) for name, keys in split_groups.items()}
    overlap = {
        "train_validation": len(group_sets["train"] & group_sets["validation"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "validation_test": len(group_sets["validation"] & group_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Source-group leakage detected: {overlap}")
    for name, rows in splits.items():
        save_json(rows, output_dir / f"{name}_data.json")
    save_json(thresholds, output_dir / "percentile_thresholds.json")
    audit = {
        "input": str(Path(args.input)),
        "threshold_source": str(Path(args.thresholds)),
        "threshold_policy": "fixed_real_training_thresholds_for_cross-dataset_comparability",
        "random_state": args.random_state,
        "valid_examples": len(prepared),
        "source_groups": len(set(source_group_key(row) for row in prepared)),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "split_source_groups": {name: len(keys) for name, keys in split_groups.items()},
        "class_distribution": {
            name: dict(Counter(row["similarity_label"] for row in rows))
            for name, rows in splits.items()
        },
        "source_group_overlap": overlap,
        "excluded_generation_failures": len(payload.get("errors", [])),
    }
    save_json(audit, output_dir / "split_audit.json")
    print(f"Prepared {len(prepared)} examples from {audit['source_groups']} source groups.")
    print(f"Split sizes: {audit['split_sizes']}")
    print(f"Class distribution: {audit['class_distribution']}")
    print(f"Excluded generation failures: {audit['excluded_generation_failures']}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
