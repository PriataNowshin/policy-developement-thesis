"""Train a GNN to predict AST-delta similarity groups."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_RANDOM_STATE,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from src.dataset import AstGraphDataset, collate_graphs
from src.io_utils import ensure_output_dir, load_json, save_json
from src.metrics import evaluate_predictions
from src.model import AstGnnClassifier
from src.preprocessing import (
    extract_results,
    prepare_examples,
    serializable_example,
)
from src.splitting import (
    assign_labels,
    compute_train_thresholds,
    get_class_distribution,
    split_dataset,
)
from src.training import (
    compute_class_weights,
    predict,
    set_random_seed,
    train_model,
)
from src.vocabulary import build_train_vocabularies


def parse_args():
    parser = ArgumentParser(
        description=(
            "Train an AST-GNN using only old_function_code and requirement."
        )
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lexical_dim", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max_ast_nodes", type=int, default=0)
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


def build_loader(dataset, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def save_splits(output_dir: Path, split_examples: dict[str, list[dict]]) -> None:
    for split_name, examples in split_examples.items():
        save_json(
            [serializable_example(example) for example in examples],
            output_dir / f"{split_name}_data.json",
        )


def build_predictions(examples, predicted_ids, probabilities) -> list[dict]:
    predictions = []
    for example, predicted_id, probability in zip(
        examples,
        predicted_ids,
        probabilities,
    ):
        predictions.append(
            {
                "repo_full_name": example["repo_full_name"],
                "file_path": example["file_path"],
                "function_name": example["function_name"],
                "ast_delta_similarity": example["ast_delta_similarity"],
                "true_label": example["similarity_label"],
                "predicted_label": ID_TO_LABEL[predicted_id],
                "probabilities": {
                    ID_TO_LABEL[index]: float(value)
                    for index, value in enumerate(probability)
                },
            }
        )
    return predictions


def main() -> None:
    args = parse_args()
    set_random_seed(args.random_state)
    output_dir = ensure_output_dir(args.output_dir)
    device = choose_device(args.device)

    input_data = load_json(args.input)
    examples, errors = prepare_examples(extract_results(input_data))
    train_examples, validation_examples, test_examples = split_dataset(
        examples,
        args.random_state,
    )

    thresholds = compute_train_thresholds(train_examples)
    for split in (train_examples, validation_examples, test_examples):
        assign_labels(split, thresholds)

    vocabularies = build_train_vocabularies(train_examples)
    split_examples = {
        "train": train_examples,
        "validation": validation_examples,
        "test": test_examples,
    }
    datasets = {
        name: AstGraphDataset(
            split,
            vocabularies,
            max_ast_nodes=args.max_ast_nodes,
        )
        for name, split in split_examples.items()
    }
    loaders = {
        name: build_loader(
            dataset,
            args.batch_size,
            shuffle=name == "train",
        )
        for name, dataset in datasets.items()
    }

    model_config = {
        "node_type_vocab_size": len(vocabularies.node_types),
        "field_vocab_size": len(vocabularies.fields),
        "lexical_vocab_size": len(vocabularies.lexical),
        "hidden_dim": args.hidden_dim,
        "lexical_dim": args.lexical_dim,
        "gnn_layers": args.gnn_layers,
        "dropout": args.dropout,
        "num_labels": len(LABEL_TO_ID),
    }
    model = AstGnnClassifier(**model_config).to(device)
    class_weights = compute_class_weights(
        train_examples,
        len(LABEL_TO_ID),
    )
    _, history = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=class_weights,
    )

    evaluation = {}
    prediction_data = {}
    for split_name in ("validation", "test"):
        true_ids, predicted_ids, probabilities = predict(
            model,
            loaders[split_name],
            device,
        )
        evaluation[split_name] = evaluate_predictions(
            true_ids,
            predicted_ids,
        )
        prediction_data[split_name] = (predicted_ids, probabilities)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "vocabularies": vocabularies.to_dict(),
            "percentile_thresholds": thresholds,
            "label_mapping": LABEL_TO_ID,
        },
        output_dir / "model.pt",
    )
    save_json(vocabularies.to_dict(), output_dir / "vocabularies.json")
    save_json(LABEL_TO_ID, output_dir / "label_mapping.json")
    save_json(thresholds, output_dir / "percentile_thresholds.json")
    save_json(history, output_dir / "training_history.json")
    save_json(errors, output_dir / "errors.json")
    save_splits(output_dir, split_examples)
    test_predicted, test_probabilities = prediction_data["test"]
    save_json(
        build_predictions(
            test_examples,
            test_predicted,
            test_probabilities,
        ),
        output_dir / "test_predictions.json",
    )

    report = {
        "input_contract": {
            "model_inputs": ["old_function_code_ast", "requirement"],
            "label_source": "ast_delta_similarity",
            "threshold_source": "training_split_only",
        },
        "device": str(device),
        "valid_examples": len(examples),
        "skipped_examples": len(errors),
        "split_sizes": {
            name: len(split) for name, split in split_examples.items()
        },
        "class_distribution": {
            name: get_class_distribution(split)
            for name, split in split_examples.items()
        },
        "percentile_thresholds": thresholds,
        "validation_metrics": evaluation["validation"],
        "test_metrics": evaluation["test"],
        "model_config": model_config,
        "max_ast_nodes": args.max_ast_nodes,
    }
    save_json(report, output_dir / "evaluation_report.json")

    print("\nFinal summary")
    print(f"Device: {device}")
    print(f"Valid examples: {len(examples)}")
    print(f"Skipped examples: {len(errors)}")
    print(f"Train-only thresholds: {thresholds}")
    print(f"Test macro F1: {evaluation['test']['macro_f1']:.4f}")
    print(f"Test accuracy: {evaluation['test']['accuracy']:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
