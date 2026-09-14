"""Fine-tune the saved Chapter 5 Pathway 2 model on fixed synthetic splits."""

import time
from argparse import ArgumentParser
from pathlib import Path

import torch

from train import (
    build_loader,
    build_predictions,
    choose_device,
    shuffled_ast_metrics,
)
from src.config import EDGE_TYPE_TO_ID, LABEL_TO_ID, PAD_TOKEN, UNKNOWN_TOKEN
from src.data import (
    class_distribution,
    load_fixed_splits,
    serializable_manifest,
    split_overlap_audit,
)
from src.dataset import AstOnlyDataset, graph_size_audit
from src.io_utils import ensure_new_output_dir, save_json
from src.metrics import evaluate_predictions
from src.model import AstOnlyGnnClassifier, parameter_counts
from src.training import predict, set_random_seed, train_model
from src.vocabulary import Vocabularies, Vocabulary


def parse_args():
    parser = ArgumentParser(
        description=(
            "Fine-tune the saved AST-only Pathway 2 GNN on fixed synthetic splits."
        )
    )
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--initial_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_graphs_per_batch", type=int, default=16)
    parser.add_argument("--max_total_nodes_per_batch", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def vocabulary_from_dict(values: dict[str, int]) -> Vocabulary:
    if values.get(PAD_TOKEN) != 0 or values.get(UNKNOWN_TOKEN) != 1:
        raise ValueError("Saved vocabulary has an incompatible PAD/UNKNOWN mapping.")
    if sorted(values.values()) != list(range(len(values))):
        raise ValueError("Saved vocabulary IDs must be contiguous from zero.")
    vocabulary = Vocabulary()
    vocabulary.value_to_id = dict(values)
    return vocabulary


def vocabularies_from_dict(values: dict) -> Vocabularies:
    if set(values) != {"node_types", "fields", "lexical"}:
        raise ValueError("Saved checkpoint has incomplete AST vocabularies.")
    return Vocabularies(
        node_types=vocabulary_from_dict(values["node_types"]),
        fields=vocabulary_from_dict(values["fields"]),
        lexical=vocabulary_from_dict(values["lexical"]),
    )


def load_historical_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "true_ast_only_relational_gnn":
        raise ValueError("Checkpoint is not the Chapter 5 Pathway 2 model.")
    if checkpoint.get("label_mapping") != LABEL_TO_ID:
        raise ValueError("Checkpoint label mapping does not match Pathway 2.")
    if checkpoint.get("edge_type_mapping") != EDGE_TYPE_TO_ID:
        raise ValueError("Checkpoint edge mapping does not match Pathway 2.")

    vocabularies = vocabularies_from_dict(checkpoint["vocabularies"])
    model_config = checkpoint["model_config"]
    expected_sizes = {
        "node_type_vocab_size": len(vocabularies.node_types),
        "field_vocab_size": len(vocabularies.fields),
        "lexical_vocab_size": len(vocabularies.lexical),
    }
    for key, expected in expected_sizes.items():
        if model_config.get(key) != expected:
            raise ValueError(f"Checkpoint {key} does not match its vocabulary.")

    model = AstOnlyGnnClassifier(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, vocabularies, model_config


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)
    output_dir = ensure_new_output_dir(args.output_dir)
    device = choose_device(args.device)
    start = time.perf_counter()

    checkpoint_path = Path(args.initial_checkpoint)
    model, vocabularies, model_config = load_historical_model(
        checkpoint_path, device
    )
    splits, thresholds = load_fixed_splits(args.split_dir)
    datasets = {
        name: AstOnlyDataset(rows, vocabularies)
        for name, rows in splits.items()
    }
    loaders = {
        name: build_loader(dataset, args, shuffle=name == "train")
        for name, dataset in datasets.items()
    }

    counts = parameter_counts(model)
    print(f"Device: {device}")
    print(f"Initial checkpoint: {checkpoint_path}")
    print(f"Trainable parameters: {counts['trainable']:,}")
    print("Model inputs: old_function_code AST only")
    print("Vocabulary: fixed historical Pathway 2 vocabulary")

    history, best_epoch = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        gradient_clip_norm=args.gradient_clip_norm,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        checkpoint_path=output_dir / ".best_state.pt",
    )

    evaluations = {}
    test_predictions = None
    for split_name in ("validation", "test"):
        indices, true_ids, predicted_ids, probabilities = predict(
            model, loaders[split_name], device
        )
        evaluations[split_name] = evaluate_predictions(true_ids, predicted_ids)
        if split_name == "test":
            test_predictions = build_predictions(
                splits["test"], indices, predicted_ids, probabilities
            )

    shuffled_metrics = shuffled_ast_metrics(
        model, splits["test"], vocabularies, args, device
    )
    run_config = {
        **vars(args),
        "device": str(device),
        "initial_checkpoint": str(checkpoint_path),
        "vocabulary_source": str(checkpoint_path),
    }
    report = {
        "experiment": "synthetic_transfer_ast_only_relational_gnn",
        "input_contract": {
            "model_inputs": ["old_function_code_ast"],
            "excluded_model_inputs": [
                "requirement",
                "new_function_code",
                "new_function_code_by_llm",
                "human_delta",
                "llm_delta",
                "human_ast_edit_script",
                "llm_ast_edit_script",
                "ast_delta_similarity",
            ],
            "label_source": "ast_delta_similarity",
            "threshold_source": "fixed_synthetic_training_split_only",
        },
        "initialization": {
            "mode": "historical_pathway2_transfer",
            "checkpoint": str(checkpoint_path),
            "vocabulary": str(checkpoint_path),
        },
        "source_split_dir": str(Path(args.split_dir)),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_distribution": {
            name: class_distribution(rows) for name, rows in splits.items()
        },
        "percentile_thresholds": thresholds,
        "split_overlap_audit": split_overlap_audit(splits),
        "graph_size_audit": {
            name: graph_size_audit(dataset) for name, dataset in datasets.items()
        },
        "validation_metrics": evaluations["validation"],
        "test_metrics": evaluations["test"],
        "shuffled_ast_test_metrics": shuffled_metrics,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": time.perf_counter() - start,
        "model_config": model_config,
        "parameter_counts": counts,
        "run_config": run_config,
    }

    torch.save(
        {
            "architecture": "true_ast_only_relational_gnn",
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "vocabularies": vocabularies.to_dict(),
            "percentile_thresholds": thresholds,
            "label_mapping": LABEL_TO_ID,
            "edge_type_mapping": EDGE_TYPE_TO_ID,
            "initial_checkpoint": str(checkpoint_path),
        },
        output_dir / "model.pt",
    )
    save_json(vocabularies.to_dict(), output_dir / "vocabularies.json")
    save_json(LABEL_TO_ID, output_dir / "label_mapping.json")
    save_json(EDGE_TYPE_TO_ID, output_dir / "edge_type_mapping.json")
    save_json(thresholds, output_dir / "percentile_thresholds.json")
    save_json(history, output_dir / "training_history.json")
    save_json(run_config, output_dir / "run_config.json")
    save_json(report, output_dir / "evaluation_report.json")
    save_json(test_predictions, output_dir / "test_predictions.json")
    for split_name, rows in splits.items():
        save_json(
            [serializable_manifest(row) for row in rows],
            output_dir / f"{split_name}_manifest.json",
        )

    print("\nFinal summary")
    print(f"Best epoch: {best_epoch}")
    print(f"Test macro F1: {evaluations['test']['macro_f1']:.4f}")
    print(f"Test accuracy: {evaluations['test']['accuracy']:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
