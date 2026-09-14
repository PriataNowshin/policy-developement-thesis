#!/usr/bin/env python3
"""Prepare the fixed split directory for imbalanced synthetic fine-tuning."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--training-variant",
        choices=("imbalanced", "balanced"),
        default="imbalanced",
    )
    return parser.parse_args()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    training_filename = (
        "train_data_imbalanced.json"
        if args.training_variant == "imbalanced"
        else "train_data.json"
    )
    splits = {
        "train": load(args.label_dir / training_filename),
        "validation": load(args.label_dir / "validation_data.json"),
        "test": load(args.label_dir / "test_data.json"),
    }
    source_splits = defaultdict(set)
    for split, rows in splits.items():
        for row in rows:
            source_splits[row["source_index"]].add(split)
    leakage = {key: sorted(value) for key, value in source_splits.items() if len(value) > 1}
    if leakage:
        raise RuntimeError(f"Cross-split source leakage detected for {len(leakage)} sources")
    for split, rows in splits.items():
        save(args.output_dir / f"{split}_data.json", rows)
    thresholds = load(args.label_dir / "percentile_thresholds.json")
    save(args.output_dir / "percentile_thresholds.json", thresholds)
    audit = {
        "experiment": f"fusion_guided_synthetic_{args.training_variant}_training",
        "training_variant": args.training_variant,
        "training_source_file": training_filename,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_distribution": {
            name: dict(Counter(row["similarity_label"] for row in rows))
            for name, rows in splits.items()
        },
        "percentile_thresholds": thresholds,
        "threshold_source": "synthetic_training_tie-controlled_reference",
        "cross_split_source_leakage": 0,
    }
    save(args.output_dir / "split_audit.json", audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
