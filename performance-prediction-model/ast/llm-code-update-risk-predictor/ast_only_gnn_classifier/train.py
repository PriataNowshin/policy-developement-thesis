"""Train a genuine AST-only GNN on the preserved fixed splits."""

import random
import time
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.config import (
    DEFAULT_RANDOM_STATE,
    EDGE_TYPE_TO_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from src.data import (
    class_distribution,
    load_fixed_splits,
    serializable_manifest,
    split_overlap_audit,
)
from src.dataset import (
    AstOnlyDataset,
    NodeBudgetBatchSampler,
    collate_graphs,
    graph_size_audit,
)
from src.io_utils import ensure_new_output_dir, save_json
from src.metrics import evaluate_predictions
from src.model import AstOnlyGnnClassifier, parameter_counts
from src.training import predict, set_random_seed, train_model
from src.vocabulary import build_train_vocabularies


def parse_args():
    parser = ArgumentParser(
        description="Predict AST-edit similarity groups using old-code AST only."
    )
    parser.add_argument("--split_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lexical_dim", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_graphs_per_batch", type=int, default=16)
    parser.add_argument("--max_total_nodes_per_batch", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--random_state", type=int, default=DEFAULT_RANDOM_STATE)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    if args.epochs < 1 or args.gnn_layers < 1:
        raise ValueError("epochs and gnn_layers must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if args.hidden_dim < 1 or args.lexical_dim < 1:
        raise ValueError("Embedding dimensions must be positive.")


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_loader(dataset, args, shuffle: bool) -> DataLoader:
    sampler = NodeBudgetBatchSampler(
        node_counts=dataset.node_counts,
        max_graphs=args.max_graphs_per_batch,
        max_total_nodes=args.max_total_nodes_per_batch,
        shuffle=shuffle,
        random_state=args.random_state,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_graphs,
        num_workers=0,
    )


def build_predictions(
    examples: list[dict],
    indices: list[int],
    predicted_ids: list[int],
    probabilities: list[list[float]],
) -> list[dict]:
    rows = []
    for index, predicted_id, probability in zip(
        indices,
        predicted_ids,
        probabilities,
    ):
        example = examples[index]
        rows.append(
            {
                "source_index": example["source_index"],
                "repo_full_name": example["repo_full_name"],
                "file_path": example["file_path"],
                "function_name": example["function_name"],
                "ast_delta_similarity": example["ast_delta_similarity"],
                "true_label": example["similarity_label"],
                "predicted_label": ID_TO_LABEL[predicted_id],
                "probabilities": {
                    ID_TO_LABEL[class_id]: float(value)
                    for class_id, value in enumerate(probability)
                },
            }
        )
    return sorted(rows, key=lambda row: row["source_index"])


def shuffled_ast_metrics(
    model,
    test_examples,
    vocabularies,
    args,
    device,
) -> dict:
    graphs = [example["graph"] for example in test_examples]
    random.Random(args.random_state + 1000).shuffle(graphs)
    shuffled_examples = []
    for example, graph in zip(test_examples, graphs):
        shuffled = dict(example)
        shuffled["graph"] = graph
        shuffled_examples.append(shuffled)
    dataset = AstOnlyDataset(shuffled_examples, vocabularies)
    loader = build_loader(dataset, args, shuffle=False)
    _, true_ids, predicted_ids, _ = predict(model, loader, device)
    return evaluate_predictions(true_ids, predicted_ids)


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seed(args.random_state)
    output_dir = ensure_new_output_dir(args.output_dir)
    device = choose_device(args.device)
    experiment_start = time.perf_counter()

    splits, thresholds = load_fixed_splits(args.split_dir)
    vocabularies = build_train_vocabularies(splits["train"])
    datasets = {
        name: AstOnlyDataset(rows, vocabularies)
        for name, rows in splits.items()
    }
    loaders = {
        name: build_loader(dataset, args, shuffle=name == "train")
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
        "num_relations": len(EDGE_TYPE_TO_ID),
        "num_labels": len(LABEL_TO_ID),
    }
    model = AstOnlyGnnClassifier(**model_config).to(device)
    counts = parameter_counts(model)
    print(f"Device: {device}")
    print(f"Trainable parameters: {counts['trainable']:,}")
    print("Model inputs: old_function_code AST only")
    print("AST policy: no node truncation")

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
            model,
            loaders[split_name],
            device,
        )
        evaluations[split_name] = evaluate_predictions(true_ids, predicted_ids)
        if split_name == "test":
            test_predictions = build_predictions(
                splits["test"],
                indices,
                predicted_ids,
                probabilities,
            )

    shuffled_metrics = shuffled_ast_metrics(
        model,
        splits["test"],
        vocabularies,
        args,
        device,
    )
    total_seconds = time.perf_counter() - experiment_start
    run_config = {**vars(args), "device": str(device)}
    report = {
        "experiment": "true_ast_only_relational_gnn",
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
            "threshold_source": "fixed_baseline_training_split_only",
        },
        "source_split_dir": str(Path(args.split_dir)),
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_distribution": {
            name: class_distribution(rows) for name, rows in splits.items()
        },
        "percentile_thresholds": thresholds,
        "split_overlap_audit": split_overlap_audit(splits),
        "graph_size_audit": {
            name: graph_size_audit(dataset)
            for name, dataset in datasets.items()
        },
        "validation_metrics": evaluations["validation"],
        "test_metrics": evaluations["test"],
        "shuffled_ast_test_metrics": shuffled_metrics,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_seconds": total_seconds,
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
    print(f"Shuffled-AST accuracy: {shuffled_metrics['accuracy']:.4f}")
    print(f"Runtime seconds: {total_seconds:.1f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
