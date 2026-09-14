"""Fine-tune CodeBERT using requirement text only."""

from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config import DEFAULT_RANDOM_STATE, ID_TO_LABEL, LABEL_TO_ID, MODEL_NAME
from src.data import load_fixed_splits
from src.dataset import RequirementCollator, RequirementDataset
from src.io_utils import ensure_output_dir, save_json
from src.metrics import evaluate_predictions
from src.training import predict, set_random_seed, train_model


def parse_args():
    parser = ArgumentParser(
        description="Requirement-only CodeBERT ablation using fixed AST-label splits."
    )
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default=MODEL_NAME)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_distribution(records: list[dict]) -> dict[str, int]:
    counts = Counter(record["label_id"] for record in records)
    return {
        label: int(counts.get(label_id, 0))
        for label_id, label in ID_TO_LABEL.items()
    }


def build_loader(records, tokenizer, max_length, batch_size, shuffle):
    dataset = RequirementDataset(records, tokenizer, max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=RequirementCollator(tokenizer),
        num_workers=0,
    )


def build_predictions(records, predicted_ids, probabilities) -> list[dict]:
    return [
        {
            "repo_full_name": record["repo_full_name"],
            "file_path": record["file_path"],
            "function_name": record["function_name"],
            "ast_delta_similarity": record["ast_delta_similarity"],
            "true_label": record["similarity_label"],
            "predicted_label": ID_TO_LABEL[predicted_id],
            "probabilities": {
                ID_TO_LABEL[index]: float(value)
                for index, value in enumerate(probability)
            },
        }
        for record, predicted_id, probability in zip(
            records,
            predicted_ids,
            probabilities,
        )
    ]


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)
    output_dir = ensure_output_dir(args.output_dir)
    device = choose_device(args.device)
    splits, thresholds = load_fixed_splits(args.split_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABEL_TO_ID),
        label2id=LABEL_TO_ID,
        id2label=ID_TO_LABEL,
    ).to(device)
    loaders = {
        name: build_loader(
            records,
            tokenizer,
            args.max_length,
            args.batch_size,
            shuffle=name == "train",
        )
        for name, records in splits.items()
    }

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        checkpoint_path=output_dir / ".best_model_state.pt",
    )

    evaluations = {}
    predictions = {}
    for split_name in ("validation", "test"):
        true_ids, predicted_ids, probabilities = predict(
            model,
            loaders[split_name],
            device,
        )
        evaluations[split_name] = evaluate_predictions(true_ids, predicted_ids)
        predictions[split_name] = (predicted_ids, probabilities)

    model_dir = output_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    save_json(history, output_dir / "training_history.json")
    save_json(LABEL_TO_ID, output_dir / "label_mapping.json")
    save_json(thresholds, output_dir / "percentile_thresholds.json")
    test_ids, test_probabilities = predictions["test"]
    save_json(
        build_predictions(splits["test"], test_ids, test_probabilities),
        output_dir / "test_predictions.json",
    )

    run_config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": args.max_length,
        "patience": args.patience,
        "random_state": args.random_state,
        "device": str(device),
    }
    save_json(run_config, output_dir / "run_config.json")
    report = {
        "experiment": "requirement_only_codebert_ablation",
        "input_contract": {
            "model_inputs": ["requirement"],
            "excluded_model_inputs": [
                "old_function_code",
                "new_function_code",
                "new_function_code_by_llm",
                "human_delta",
                "llm_delta",
                "ast_edit_scripts",
                "ast_delta_similarity",
            ],
            "label_source": "ast_delta_similarity",
            "threshold_source": "fixed_baseline_training_split_only",
        },
        "source_split_dir": str(Path(args.split_dir)),
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "class_distribution": {
            name: class_distribution(records)
            for name, records in splits.items()
        },
        "percentile_thresholds": thresholds,
        "validation_metrics": evaluations["validation"],
        "test_metrics": evaluations["test"],
        "run_config": run_config,
    }
    save_json(report, output_dir / "evaluation_report.json")

    print("\nFinal summary")
    print(f"Device: {device}")
    print(f"Fixed split sizes: {report['split_sizes']}")
    print(f"Fixed train-only thresholds: {thresholds}")
    print(f"Test macro F1: {evaluations['test']['macro_f1']:.4f}")
    print(f"Test accuracy: {evaluations['test']['accuracy']:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
