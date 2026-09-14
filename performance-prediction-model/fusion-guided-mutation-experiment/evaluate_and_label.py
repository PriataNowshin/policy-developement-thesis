#!/usr/bin/env python3
"""Calculate AST-delta similarity and label preserved splits from synthetic train percentiles."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ast" / "function-similarity-checker"))
from delta_similarity.processing import process_item  # noqa: E402

LABEL_TO_ID = {"low": 0, "mid": 1, "high": 2}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--downsample-training-max-ties",
        action="store_true",
        help="Deterministically cap maximum-score training ties before calculating percentiles.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def label(score: float, p33: float, p66: float) -> str:
    if score <= p33:
        return "low"
    if score <= p66:
        return "mid"
    return "high"


def stable_tie_key(row: dict, seed: int) -> tuple[str, str]:
    record_id = str(row.get("id", row.get("index", "")))
    digest = hashlib.sha256(f"synthetic-training-max-tie-v1\0{seed}\0{record_id}".encode()).hexdigest()
    return digest, record_id


def main() -> None:
    args = arguments()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    results, errors = [], []
    for index, row in enumerate(rows):
        if row.get("generation_status") != "success":
            errors.append({"index": index, "id": row.get("id"), "error": "generation_not_successful"})
            continue
        result, error = process_item(index, row)
        if error:
            errors.append(error)
            continue
        result.update({
            "id": row["id"],
            "source_split": row["source_split"],
            "source_index": row["source_index"],
            "mutation_operator": row["mutation_operator"],
        })
        results.append(result)
    raw_train = [row for row in results if row["source_split"] == "train"]
    train_for_labeling = list(raw_train)
    discarded_training = []
    train_scores = [float(row["ast_delta_similarity"]) for row in train_for_labeling]
    if not train_scores:
        raise RuntimeError("No successful synthetic training records")
    p33, p66 = (float(np.percentile(train_scores, percentile)) for percentile in (33, 66))
    if p33 == p66 and args.downsample_training_max_ties:
        maximum = max(train_scores)
        below_max = [row for row in train_for_labeling if float(row["ast_delta_similarity"]) < maximum]
        at_max = [row for row in train_for_labeling if float(row["ast_delta_similarity"]) == maximum]
        if not below_max:
            raise RuntimeError("All synthetic training scores equal the maximum; downsampling cannot create classes")
        keep_max = min(len(at_max), len(below_max) // 2)
        ordered_max = sorted(at_max, key=lambda row: stable_tie_key(row, args.seed))
        kept_max_ids = {row["id"] for row in ordered_max[:keep_max]}
        discarded_training = [row for row in at_max if row["id"] not in kept_max_ids]
        train_for_labeling = below_max + [row for row in at_max if row["id"] in kept_max_ids]
        train_scores = [float(row["ast_delta_similarity"]) for row in train_for_labeling]
        p33, p66 = (float(np.percentile(train_scores, percentile)) for percentile in (33, 66))
    if p33 == p66:
        raise RuntimeError(
            f"Synthetic training percentiles are tied at {p33}; three score-based classes cannot be formed honestly"
        )
    selected_train_ids = {row["id"] for row in train_for_labeling}
    imbalanced_train = []
    for row in raw_train:
        labeled = dict(row)
        name = label(float(labeled["ast_delta_similarity"]), p33, p66)
        labeled["similarity_label"] = name
        labeled["label_id"] = LABEL_TO_ID[name]
        imbalanced_train.append(labeled)
    splits = {name: [] for name in ("train", "validation", "test")}
    for row in results:
        if row["source_split"] == "train" and row["id"] not in selected_train_ids:
            continue
        name = label(float(row["ast_delta_similarity"]), p33, p66)
        row["similarity_label"] = name
        row["label_id"] = LABEL_TO_ID[name]
        splits[row["source_split"]].append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in splits.items():
        save(args.output_dir / f"{split}_data.json", records)
    save(args.output_dir / "train_data_imbalanced.json", imbalanced_train)
    thresholds = {"p33": p33, "p66": p66}
    save(args.output_dir / "percentile_thresholds.json", thresholds)
    save(args.output_dir / "discarded_training_max_ties.json", discarded_training)
    report = {
        "input_records": len(rows), "processed_records": len(results), "errors": errors,
        "threshold_source": "synthetic_training_split_only", "percentile_thresholds": thresholds,
        "raw_training_records": len(raw_train),
        "selected_training_records": len(train_for_labeling),
        "discarded_training_max_ties": len(discarded_training),
        "training_max_tie_policy": (
            "deterministic cap at no more than one maximum-score record per two below-maximum records"
            if args.downsample_training_max_ties else "disabled"
        ),
        "tie_selection_seed": args.seed if args.downsample_training_max_ties else None,
        "split_sizes": {name: len(value) for name, value in splits.items()},
        "class_distribution": {name: dict(Counter(row["similarity_label"] for row in value)) for name, value in splits.items()},
        "imbalanced_training_size": len(imbalanced_train),
        "imbalanced_training_class_distribution": dict(
            Counter(row["similarity_label"] for row in imbalanced_train)
        ),
        "score_ties": dict(Counter(str(row["ast_delta_similarity"]) for row in results).most_common(10)),
    }
    save(args.output_dir / "labeling_audit.json", report)
    save(args.output_dir / "similarity_results.json", {"summary": report, "results": results, "errors": errors})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
