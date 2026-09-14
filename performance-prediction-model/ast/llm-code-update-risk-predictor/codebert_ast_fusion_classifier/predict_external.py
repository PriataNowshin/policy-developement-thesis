"""Evaluate a saved fusion model on a new, labeled external result file."""

from argparse import ArgumentParser
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import ID_TO_LABEL, LABEL_TO_ID
from src.data import _safe_example
from src.dataset import FusionCollator, FusionDataset, NodeBudgetBatchSampler, length_audit
from src.io_utils import load_json, save_json
from src.metrics import evaluate_predictions
from src.model import CodeBertAstFusionClassifier
from src.training import load_trainable_state_dict, predict
from src.vocabulary import Vocabularies, Vocabulary


def parse_args():
    parser = ArgumentParser(
        description="Test a trained CodeBERT+AST fusion model without retraining."
    )
    parser.add_argument("--input", required=True, help="AST evaluation JSON.")
    parser.add_argument("--model_dir", required=True, help="Saved fusion output folder.")
    parser.add_argument("--output_dir", required=True, help="External-test output folder.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_graphs_per_batch", type=int, default=8)
    parser.add_argument("--max_total_nodes_per_batch", type=int, default=4096)
    return parser.parse_args()


def choose_device(requested):
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_source(model_dir: Path, source: str) -> Path:
    candidate = Path(source)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    predictor_root = model_dir.resolve().parents[1]
    resolved = predictor_root / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"Saved CodeBERT source does not exist: {resolved}")
    return resolved


def assign_label(score, thresholds):
    if score <= thresholds["p33"]:
        return "low"
    if score <= thresholds["p66"]:
        return "mid"
    return "high"


def load_external(input_path, thresholds):
    payload = load_json(input_path)
    records = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        raise ValueError('Input must contain a top-level "results" list.')
    examples, metadata, errors = [], [], []
    for index, record in enumerate(records):
        try:
            example = _safe_example(record, f"results[{index}]", index)
            label = assign_label(example["ast_delta_similarity"], thresholds)
            example["similarity_label"] = label
            example["label_id"] = LABEL_TO_ID[label]
            examples.append(example)
            metadata.append({
                "id": record.get("id"),
                "mutation_operator": record.get("mutation_operator"),
                "mutation_operator_name": record.get("mutation_operator_name"),
                "clone_type": record.get("clone_type"),
                "exact_reference_match": record.get("exact_reference_match"),
                "text_delta_similarity": record.get("text_delta_similarity"),
            })
        except (ValueError, SyntaxError, TypeError) as error:
            errors.append({"source_index": index, "reason": str(error)})
    if not examples:
        raise ValueError("No valid external examples were found.")
    return examples, metadata, errors, payload.get("errors", [])


def grouped_metrics(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[str(row.get(field, "unknown"))].append(row)
    output = {}
    for name, group in sorted(groups.items()):
        output[name] = {
            "count": len(group),
            **evaluate_predictions(
                [row["true_label_id"] for row in group],
                [row["predicted_label_id"] for row in group],
            ),
        }
    return output


def main():
    args = parse_args()
    model_dir, output_dir = Path(args.model_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_dir / "model_trainable.pt", map_location="cpu", weights_only=True)
    thresholds = {key: float(value) for key, value in checkpoint["percentile_thresholds"].items()}
    if checkpoint["label_mapping"] != LABEL_TO_ID:
        raise ValueError("Checkpoint label mapping differs from the implementation mapping.")

    examples, metadata, load_errors, generation_errors = load_external(args.input, thresholds)
    vocab_data = checkpoint["vocabularies"]
    vocabularies = Vocabularies(
        Vocabulary.from_dict(vocab_data["node_types"]),
        Vocabulary.from_dict(vocab_data["fields"]),
        Vocabulary.from_dict(vocab_data["lexical"]),
    )
    tokenizer_dir = model_dir / "tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    max_length = int(load_json(model_dir / "run_config.json")["max_length"])
    dataset = FusionDataset(examples, vocabularies, tokenizer, max_length)
    sampler = NodeBudgetBatchSampler(
        dataset.node_counts,
        max_graphs=args.max_graphs_per_batch,
        max_total_nodes=args.max_total_nodes_per_batch,
        shuffle=False,
        random_state=0,
    )
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=FusionCollator(tokenizer), num_workers=0)

    model_config = dict(checkpoint["model_config"])
    model_config["codebert_model"] = str(resolve_source(model_dir, checkpoint["codebert_source"]))
    device = choose_device(args.device)
    model = CodeBertAstFusionClassifier(**model_config).to(device)
    load_trainable_state_dict(model, checkpoint["trainable_state_dict"])
    indices, true_ids, predicted_ids, probabilities = predict(model, loader, device)

    rows = []
    for index, true_id, predicted_id, probability in zip(indices, true_ids, predicted_ids, probabilities):
        example, extra = examples[index], metadata[index]
        rows.append({
            **extra,
            "source_index": example["source_index"],
            "repo_full_name": example["repo_full_name"],
            "file_path": example["file_path"],
            "function_name": example["function_name"],
            "ast_delta_similarity": example["ast_delta_similarity"],
            "true_label": ID_TO_LABEL[true_id],
            "true_label_id": true_id,
            "predicted_label": ID_TO_LABEL[predicted_id],
            "predicted_label_id": predicted_id,
            "correct": true_id == predicted_id,
            "probabilities": {ID_TO_LABEL[i]: float(value) for i, value in enumerate(probability)},
        })
    rows.sort(key=lambda row: row["source_index"])
    report = {
        "experiment": "synthetic_external_test_of_saved_codebert_ast_fusion",
        "training_performed": False,
        "model_dir": str(model_dir),
        "input": str(Path(args.input)),
        "device": str(device),
        "threshold_source": str(model_dir / "percentile_thresholds.json"),
        "percentile_thresholds": thresholds,
        "model_inputs": ["requirement", "old_function_code_ast"],
        "evaluated_examples": len(rows),
        "generation_errors_reported_by_input": len(generation_errors),
        "external_load_errors": len(load_errors),
        "true_class_distribution": dict(Counter(row["true_label"] for row in rows)),
        "predicted_class_distribution": dict(Counter(row["predicted_label"] for row in rows)),
        "overall_metrics": evaluate_predictions(true_ids, predicted_ids),
        "by_clone_type": grouped_metrics(rows, "clone_type"),
        "by_mutation_operator": grouped_metrics(rows, "mutation_operator"),
        "length_audit": length_audit(dataset),
    }
    save_json(report, output_dir / "evaluation_report.json")
    save_json(rows, output_dir / "predictions.json")
    save_json(load_errors, output_dir / "load_errors.json")
    save_json(generation_errors, output_dir / "generation_errors.json")
    print(f"Evaluated {len(rows)} external synthetic examples; no training performed.")
    print(f"Accuracy: {report['overall_metrics']['accuracy']:.4f}")
    print(f"Macro-F1: {report['overall_metrics']['macro_f1']:.4f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
