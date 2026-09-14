#!/usr/bin/env python3
"""Refuse to proceed unless the configured model is the Chapter 5 fusion model."""

from __future__ import annotations

import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "experiment_config.json"


def resolve(relative_path: str) -> Path:
    return (HERE / relative_path).resolve()


def require_close(name: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(
            f"Wrong study model: {name} is {actual!r}, expected {expected!r}"
        )


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    model_dir = resolve(config["model_dir"])
    report_path = resolve(config["evaluation_report"])
    predictions_path = resolve(config["test_predictions"])
    manifest_path = resolve(config["test_manifest"])

    missing = [
        artifact
        for artifact in config["required_model_artifacts"]
        if not (model_dir / artifact).exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing fusion-model artifacts: {missing}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    history = json.loads((model_dir / "training_history.json").read_text(encoding="utf-8"))
    expected = config["expected_metrics"]

    require_close(
        "validation macro-F1",
        report["validation_metrics"]["macro_f1"],
        expected["validation_macro_f1"],
    )
    require_close(
        "test accuracy",
        report["test_metrics"]["accuracy"],
        expected["test_accuracy"],
    )
    require_close(
        "test macro-F1",
        report["test_metrics"]["macro_f1"],
        expected["test_macro_f1"],
    )
    if len(predictions) != report["split_sizes"]["test"]:
        raise RuntimeError("Prediction count does not match the reported test size")
    if len(manifest) != len(predictions):
        raise RuntimeError("Test manifest and prediction counts do not match")
    for key, expected_value in config["expected_model_config"].items():
        actual_value = report["model_config"].get(key)
        if actual_value != expected_value:
            raise RuntimeError(
                f"Wrong study model configuration: {key} is {actual_value!r}, "
                f"expected {expected_value!r}"
            )
    if report["test_metrics"]["confusion_matrix"] != config["expected_test_confusion_matrix"]:
        raise RuntimeError("Test confusion matrix does not match Chapter 5")
    best_epoch = max(history, key=lambda row: row["validation_macro_f1"])["epoch"]
    if best_epoch != config["expected_best_validation_epoch"]:
        raise RuntimeError("Best validation epoch does not match Chapter 5")
    if history[-1]["epoch"] != config["expected_training_stop_epoch"]:
        raise RuntimeError("Training stop epoch does not match Chapter 5")

    print("Verified Chapter 5 Study 3 Pathway 3 fusion model")
    print(f"Model directory: {model_dir}")
    print(f"Test examples: {len(predictions)}")
    print(f"Validation macro-F1: {report['validation_metrics']['macro_f1']:.4f}")
    print(f"Test accuracy: {report['test_metrics']['accuracy']:.4f}")
    print(f"Test macro-F1: {report['test_metrics']['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
