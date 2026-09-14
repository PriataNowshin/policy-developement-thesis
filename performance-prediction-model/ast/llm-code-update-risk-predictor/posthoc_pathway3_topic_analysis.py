"""Post-hoc requirement-topic analysis for the fixed Pathway 3 predictions.

The script does not train or modify the fused capability predictor. It fits an
NMF topic model on training requirements, transforms validation/test
requirements with that fixed model, and evaluates existing held-out Pathway 3
predictions within each requirement topic.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


LABELS = ["low", "mid", "high"]
MANUALLY_REVIEWED_TOPIC_NAMES = {
    8: {
        0: "Function parameters and optional arguments",
        1: "Documentation and comment language",
        2: "String formatting and interpolation",
        3: "Type annotations and docstrings",
        4: "API, method, and deprecation updates",
        5: "Code and documentation formatting",
        6: "File, path, and related error handling",
        7: "Return values and collection handling",
    },
    10: {
        0: "Function parameters and optional arguments",
        1: "Documentation and comment language",
        2: "String formatting and interpolation",
        3: "Type annotations and docstrings",
        4: "API, method, and deprecation updates",
        5: "Code and documentation formatting",
        6: "File, path, and directory operations",
        7: "Return values and collection handling",
        8: "Errors, exceptions, and diagnostic messages",
        9: "Environment variables and CLI configuration",
    }
}
TEMPLATE_STOP_WORDS = {
    "altering",
    "behavior",
    "behaviour",
    "change",
    "code",
    "ensure",
    "existing",
    "function",
    "functionality",
    "preserving",
    "refactor",
    "refactor_change",
    "runtime",
    "update",
    "use",
    "using",
}
PREFIX_RE = re.compile(r"^[A-Z][A-Z_]{2,}:\s*")
BOILERPLATE_PATTERNS = [
    re.compile(
        r"\bwithout (?:altering|changing|affecting) "
        r"(?:the |its )?(?:function(?:'s)? )?"
        r"(?:existing |external |runtime )?"
        r"(?:behavior|behaviour|functionality)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bwhile preserving (?:the |its )?(?:function(?:'s)? )?"
        r"(?:existing |current |same )?"
        r"(?:behavior|behaviour|functionality)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bpreserving (?:the |its )?(?:function(?:'s)? )?"
        r"(?:existing |current |same )?"
        r"(?:behavior|behaviour|functionality)\b",
        re.IGNORECASE,
    ),
]


def parse_args() -> argparse.Namespace:
    base = Path(__file__).resolve().parent
    default_input = base / "output" / "codebert_ast_fusion_top2"
    parser = argparse.ArgumentParser(
        description="Evaluate fixed Pathway 3 predictions by requirement topic."
    )
    parser.add_argument("--input-dir", type=Path, default=default_input)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--topics", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-words", type=int, default=12)
    parser.add_argument("--representative-examples", type=int, default=8)
    parser.add_argument("--min-support", type=int, default=30)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def read_json(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def clean_requirement(text: str) -> str:
    cleaned = PREFIX_RE.sub("", text.strip())
    for pattern in BOILERPLATE_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    return cleaned.strip(" ,.;:") or "empty"


def fit_topics(
    train_documents: list[str],
    topics: int,
    seed: int,
    top_words: int,
):
    stop_words = sorted(set(ENGLISH_STOP_WORDS) | TEMPLATE_STOP_WORDS)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words=stop_words,
        token_pattern=r"(?u)\b[A-Za-z_][A-Za-z0-9_]+\b",
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.90,
        max_features=6000,
        sublinear_tf=True,
    )
    train_matrix = vectorizer.fit_transform(train_documents)
    model = NMF(
        n_components=topics,
        init="nndsvda",
        random_state=seed,
        max_iter=1000,
    )
    train_weights = model.fit_transform(train_matrix)
    terms = np.asarray(vectorizer.get_feature_names_out())
    topic_words = {
        topic_id: terms[row.argsort()[::-1][:top_words]].tolist()
        for topic_id, row in enumerate(model.components_)
    }
    return vectorizer, model, train_weights, topic_words


def transform_topics(vectorizer, model, documents: list[str]):
    weights = model.transform(vectorizer.transform(documents))
    topic_ids = weights.argmax(axis=1).astype(int)
    totals = weights.sum(axis=1)
    strengths = np.divide(
        weights.max(axis=1),
        totals,
        out=np.zeros_like(totals),
        where=totals > 0,
    )
    return weights, topic_ids, strengths


def label_metrics(true_labels: list[str], predicted_labels: list[str]) -> dict:
    per_precision, per_recall, per_f1, support = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predicted_labels,
        labels=LABELS,
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predicted_labels, labels=LABELS)
    non_high = sum(label != "high" for label in true_labels)
    false_high = sum(
        truth != "high" and prediction == "high"
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    true_low = sum(label == "low" for label in true_labels)
    low_predicted_high = sum(
        truth == "low" and prediction == "high"
        for truth, prediction in zip(true_labels, predicted_labels)
    )
    majority = max(Counter(true_labels).values()) / len(true_labels)
    return {
        "support": len(true_labels),
        "label_distribution": {
            label: int(sum(value == label for value in true_labels))
            for label in LABELS
        },
        "majority_baseline_accuracy": float(majority),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "balanced_accuracy": float(
            balanced_accuracy_score(true_labels, predicted_labels)
        ),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(per_precision[index]),
                "recall": float(per_recall[index]),
                "f1": float(per_f1[index]),
                "support": int(support[index]),
            }
            for index, label in enumerate(LABELS)
        },
        "confusion_matrix_labels": LABELS,
        "confusion_matrix": matrix.astype(int).tolist(),
        "false_high_rate_among_non_high": (
            float(false_high / non_high) if non_high else None
        ),
        "low_to_high_error_rate": (
            float(low_predicted_high / true_low) if true_low else None
        ),
    }


def bootstrap_macro_f1(
    true_labels: list[str],
    predicted_labels: list[str],
    samples: int,
    seed: int,
) -> dict | None:
    if not samples or len(true_labels) < 2:
        return None
    rng = np.random.default_rng(seed)
    true_array = np.asarray(true_labels)
    predicted_array = np.asarray(predicted_labels)
    scores = []
    for _ in range(samples):
        indices = rng.integers(0, len(true_array), len(true_array))
        _, _, score, _ = precision_recall_fscore_support(
            true_array[indices],
            predicted_array[indices],
            labels=LABELS,
            average="macro",
            zero_division=0,
        )
        scores.append(float(score))
    lower, upper = np.percentile(scores, [2.5, 97.5])
    return {
        "method": "nonparametric bootstrap over test examples",
        "samples": samples,
        "lower_95": float(lower),
        "upper_95": float(upper),
    }


def representative_examples(
    records: list[dict],
    weights: np.ndarray,
    topic_id: int,
    limit: int,
) -> list[dict]:
    ranked = np.argsort(weights[:, topic_id])[::-1]
    examples = []
    for index in ranked:
        if weights[index, topic_id] <= 0:
            break
        examples.append(
            {
                "source_index": records[index]["source_index"],
                "requirement": records[index]["requirement"],
                "cleaned_requirement": clean_requirement(
                    records[index]["requirement"]
                ),
                "topic_weight": float(weights[index, topic_id]),
            }
        )
        if len(examples) == limit:
            break
    return examples


def distribution(topic_ids: np.ndarray, topics: int) -> dict[str, int]:
    counts = Counter(int(value) for value in topic_ids)
    return {str(topic_id): int(counts.get(topic_id, 0)) for topic_id in range(topics)}


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        args.input_dir.parent / f"pathway3_posthoc_topics_k{args.topics}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    train = read_json(args.input_dir / "train_manifest.json")
    validation = read_json(args.input_dir / "validation_manifest.json")
    test = read_json(args.input_dir / "test_manifest.json")
    predictions = read_json(args.input_dir / "test_predictions.json")

    prediction_by_index = {
        int(record["source_index"]): record for record in predictions
    }
    if len(prediction_by_index) != len(predictions):
        raise ValueError("Pathway 3 predictions contain duplicate source_index values.")
    if {int(record["source_index"]) for record in test} != set(prediction_by_index):
        raise ValueError("Test manifest and predictions do not have identical source_index sets.")

    cleaned = {
        "train": [clean_requirement(record["requirement"]) for record in train],
        "validation": [
            clean_requirement(record["requirement"]) for record in validation
        ],
        "test": [clean_requirement(record["requirement"]) for record in test],
    }
    unique_train_documents = list(dict.fromkeys(cleaned["train"]))
    vectorizer, model, train_weights, topic_words = fit_topics(
        unique_train_documents,
        args.topics,
        args.seed,
        args.top_words,
    )
    train_weights, train_topics, train_strengths = transform_topics(
        vectorizer, model, cleaned["train"]
    )
    validation_weights, validation_topics, validation_strengths = transform_topics(
        vectorizer, model, cleaned["validation"]
    )
    test_weights, test_topics, test_strengths = transform_topics(
        vectorizer, model, cleaned["test"]
    )

    enriched_test = []
    for index, manifest_record in enumerate(test):
        prediction = prediction_by_index[int(manifest_record["source_index"])]
        if prediction["true_label"] != manifest_record["similarity_label"]:
            raise ValueError(
                f"Label mismatch for source_index={manifest_record['source_index']}"
            )
        enriched_test.append(
            {
                **manifest_record,
                "cleaned_requirement": cleaned["test"][index],
                "requirement_topic_id": int(test_topics[index]),
                "requirement_topic_words": topic_words[int(test_topics[index])],
                "topic_assignment_strength": float(test_strengths[index]),
                "true_label": prediction["true_label"],
                "predicted_label": prediction["predicted_label"],
                "probabilities": prediction["probabilities"],
            }
        )

    topic_report = []
    topic_names = MANUALLY_REVIEWED_TOPIC_NAMES.get(args.topics, {})
    for topic_id in range(args.topics):
        group = [
            record
            for record in enriched_test
            if record["requirement_topic_id"] == topic_id
        ]
        true_labels = [record["true_label"] for record in group]
        predicted_labels = [record["predicted_label"] for record in group]
        metrics = label_metrics(true_labels, predicted_labels)
        metrics["macro_f1_confidence_interval"] = bootstrap_macro_f1(
            true_labels,
            predicted_labels,
            args.bootstrap_samples,
            args.seed + topic_id,
        )
        topic_report.append(
            {
                "topic_id": topic_id,
                "topic_name": topic_names.get(
                    topic_id, f"Topic {topic_id} (manual name not assigned)"
                ),
                "topic_words": topic_words[topic_id],
                "minimum_support_met": len(group) >= args.min_support,
                "mean_assignment_strength": float(
                    np.mean(
                        [
                            record["topic_assignment_strength"]
                            for record in group
                        ]
                    )
                )
                if group
                else 0.0,
                **metrics,
                "representative_training_requirements": representative_examples(
                    train,
                    train_weights,
                    topic_id,
                    args.representative_examples,
                ),
                "representative_test_requirements": representative_examples(
                    test,
                    test_weights,
                    topic_id,
                    args.representative_examples,
                ),
            }
        )

    overall_true = [record["true_label"] for record in enriched_test]
    overall_predicted = [record["predicted_label"] for record in enriched_test]
    report = {
        "analysis": "post-hoc requirement-topic stratification of fixed Pathway 3 predictions",
        "predictor_retrained": False,
        "input_dir": str(args.input_dir),
        "topic_fit_split": "train",
        "topic_transform_splits": ["validation", "test"],
        "topic_model": {
            "method": "TF-IDF + NMF",
            "topics": args.topics,
            "seed": args.seed,
            "top_words": args.top_words,
            "template_prefix_removed": True,
            "boilerplate_phrases_removed": True,
            "topic_words": {str(key): value for key, value in topic_words.items()},
            "topic_names": {
                str(topic_id): topic_names.get(
                    topic_id, f"Topic {topic_id} (manual name not assigned)"
                )
                for topic_id in range(args.topics)
            },
        },
        "split_sizes": {
            "train": len(train),
            "unique_cleaned_train_requirements_used_for_topic_fit": len(
                unique_train_documents
            ),
            "validation": len(validation),
            "test": len(test),
        },
        "topic_distributions": {
            "train": distribution(train_topics, args.topics),
            "validation": distribution(validation_topics, args.topics),
            "test": distribution(test_topics, args.topics),
        },
        "mean_assignment_strength": {
            "train": float(np.mean(train_strengths)),
            "validation": float(np.mean(validation_strengths)),
            "test": float(np.mean(test_strengths)),
        },
        "overall_pathway3_metrics": label_metrics(
            overall_true, overall_predicted
        ),
        "topics": topic_report,
        "interpretation_limits": [
            "Topics are post-hoc grouping variables, not predictor inputs.",
            "Topic names require manual validation from representative requirements.",
            "Within-topic performance does not establish that fusion adds value within that topic.",
            "Topics below the minimum support threshold are descriptive only.",
            "Test results must not be used iteratively to tune the predictor or oversight rule.",
        ],
    }
    write_json(output_dir / "topic_analysis_report.json", report)
    write_json(output_dir / "test_predictions_with_topics.json", enriched_test)

    with (output_dir / "topic_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "topic_id",
            "topic_name",
            "topic_words",
            "support",
            "minimum_support_met",
            "majority_baseline_accuracy",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "macro_f1_ci_lower_95",
            "macro_f1_ci_upper_95",
            "low_recall",
            "mid_recall",
            "high_recall",
            "false_high_rate_among_non_high",
            "low_to_high_error_rate",
            "mean_assignment_strength",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for entry in topic_report:
            interval = entry["macro_f1_confidence_interval"] or {}
            writer.writerow(
                {
                    "topic_id": entry["topic_id"],
                    "topic_name": entry["topic_name"],
                    "topic_words": " | ".join(entry["topic_words"]),
                    "support": entry["support"],
                    "minimum_support_met": entry["minimum_support_met"],
                    "majority_baseline_accuracy": entry[
                        "majority_baseline_accuracy"
                    ],
                    "accuracy": entry["accuracy"],
                    "balanced_accuracy": entry["balanced_accuracy"],
                    "macro_f1": entry["macro_f1"],
                    "macro_f1_ci_lower_95": interval.get("lower_95"),
                    "macro_f1_ci_upper_95": interval.get("upper_95"),
                    "low_recall": entry["per_class"]["low"]["recall"],
                    "mid_recall": entry["per_class"]["mid"]["recall"],
                    "high_recall": entry["per_class"]["high"]["recall"],
                    "false_high_rate_among_non_high": entry[
                        "false_high_rate_among_non_high"
                    ],
                    "low_to_high_error_rate": entry[
                        "low_to_high_error_rate"
                    ],
                    "mean_assignment_strength": entry[
                        "mean_assignment_strength"
                    ],
                }
            )

    print(f"Wrote post-hoc Pathway 3 topic analysis to {output_dir}")


if __name__ == "__main__":
    main()
