#!/usr/bin/env python3
"""Extract correctly classified Chapter 5 Pathway 3 held-out records."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from verify_study_model import main as verify_study_model

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "experiment_config.json"
OUTPUT_DIR = HERE / "output" / "step1_correct_predictions"
OUTPUT_PATH = OUTPUT_DIR / "correct_predictions.json"
AUDIT_PATH = OUTPUT_DIR / "audit.json"
EXPECTED_CORRECT_BY_LABEL = {"low": 198, "mid": 160, "high": 155}


def resolve(relative_path: str) -> Path:
    return (HERE / relative_path).resolve()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    verify_study_model()
    config = load_json(CONFIG_PATH)
    predictions = load_json(resolve(config["test_predictions"]))
    manifest = load_json(resolve(config["test_manifest"]))
    manifest_by_index = {record["source_index"]: record for record in manifest}
    if len(manifest_by_index) != len(manifest):
        raise RuntimeError("Duplicate source_index values found in the test manifest")

    joined = []
    missing = []
    mismatches = []
    for prediction in predictions:
        source_index = prediction["source_index"]
        source = manifest_by_index.get(source_index)
        if source is None:
            missing.append(source_index)
            continue
        for field in ("repo_full_name", "file_path", "function_name"):
            if prediction.get(field) != source.get(field):
                mismatches.append({"source_index": source_index, "field": field})
        if prediction["true_label"] != source["similarity_label"]:
            mismatches.append({"source_index": source_index, "field": "label"})
        if prediction["predicted_label"] == prediction["true_label"]:
            joined.append({
                "source_index": source_index,
                "repo_full_name": prediction["repo_full_name"],
                "file_path": prediction["file_path"],
                "function_name": prediction["function_name"],
                "requirement": source["requirement"],
                "ast_delta_similarity": prediction["ast_delta_similarity"],
                "true_label": prediction["true_label"],
                "predicted_label": prediction["predicted_label"],
                "probabilities": prediction["probabilities"],
            })

    if missing:
        raise RuntimeError(f"Predictions missing from test manifest: {missing[:10]}")
    if mismatches:
        raise RuntimeError(f"Prediction/manifest mismatches detected: {mismatches[:5]}")
    correct_by_label = dict(Counter(record["true_label"] for record in joined))
    if correct_by_label != EXPECTED_CORRECT_BY_LABEL:
        raise RuntimeError(
            f"Correct-label counts {correct_by_label} do not match "
            f"Chapter 5 counts {EXPECTED_CORRECT_BY_LABEL}"
        )

    audit = {
        "study": config["study"],
        "model_dir": str(resolve(config["model_dir"])),
        "total_test_predictions": len(predictions),
        "correct_predictions": len(joined),
        "incorrect_predictions_excluded": len(predictions) - len(joined),
        "correct_by_label": correct_by_label,
        "selection_rule": "predicted_label == true_label",
        "join_key": "source_index",
        "missing_manifest_records": 0,
        "metadata_mismatches": 0
    }
    save_json(OUTPUT_PATH, joined)
    save_json(AUDIT_PATH, audit)
    print(f"Wrote {len(joined)} correct predictions to {OUTPUT_PATH}")
    print(f"Correct by label: {correct_by_label}")
    print(f"Audit: {AUDIT_PATH}")


if __name__ == "__main__":
    main()
