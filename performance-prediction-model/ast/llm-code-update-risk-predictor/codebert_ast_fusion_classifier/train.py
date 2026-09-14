"""Train CodeBERT-requirement and old-code-AST fusion classifier."""

from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.config import (
    DEFAULT_RANDOM_STATE,
    EDGE_TYPE_TO_ID,
    ID_TO_LABEL,
    LABEL_TO_ID,
)
from src.data import (
    class_distribution,
    load_fixed_splits,
    load_source_examples,
    serializable_manifest,
    split_by_repository,
    split_overlap_audit,
)
from src.dataset import (
    FusionCollator,
    FusionDataset,
    NodeBudgetBatchSampler,
    length_audit,
)
from src.io_utils import ensure_new_output_dir, load_json, save_json
from src.metrics import evaluate_predictions
from src.model import CodeBertAstFusionClassifier, parameter_counts
from src.training import (
    load_trainable_state_dict,
    predict,
    set_random_seed,
    train_model,
    trainable_state_dict,
)
from src.vocabulary import Vocabularies, Vocabulary, build_train_vocabularies


def parse_args():
    parser = ArgumentParser(
        description=(
            "Fuse CodeBERT(requirement) with a relational GNN(AST(old function)) "
            "to predict AST-delta similarity groups."
        )
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--split_mode",
        choices=("fixed", "repo_grouped"),
        default="fixed",
        help="Use fixed baseline splits or a repository-disjoint robustness split.",
    )
    parser.add_argument(
        "--split_dir",
        default="output/ast_gnn_classifier_output",
        help="Fixed split directory used when --split_mode fixed.",
    )
    parser.add_argument(
        "--threshold_source_description",
        default="fixed_baseline_training_split_only",
        help="Audit label describing where fixed-split thresholds came from.",
    )
    parser.add_argument(
        "--input",
        default="data/ast_delta_similarity_results.json",
        help="Raw AST result file used when --split_mode repo_grouped.",
    )
    parser.add_argument(
        "--codebert_model",
        default="output/requirement_only_codebert_output/model",
        help="Requirement-only checkpoint or microsoft/codebert-base.",
    )
    parser.add_argument(
        "--initial_model_dir",
        default=None,
        help=(
            "Saved fusion model to fine-tune. Reuses its tokenizer, AST "
            "vocabularies, architecture, and trainable weights."
        ),
    )
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--unfrozen_codebert_layers", type=int, default=2)
    parser.add_argument("--train_codebert_embeddings", action="store_true")
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lexical_dim", type=int, default=64)
    parser.add_argument("--gnn_layers", type=int, default=2)
    parser.add_argument("--fusion_dim", type=int, default=128)
    parser.add_argument("--fusion_hidden_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_graphs_per_batch", type=int, default=8)
    parser.add_argument("--max_total_nodes_per_batch", type=int, default=4096)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--codebert_learning_rate", type=float, default=5e-6)
    parser.add_argument("--new_layers_learning_rate", type=float, default=3e-4)
    parser.add_argument("--codebert_weight_decay", type=float, default=0.01)
    parser.add_argument("--new_layers_weight_decay", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
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


def validate_args(args) -> None:
    if not 1 <= args.max_length <= 512:
        raise ValueError("CodeBERT max_length must be between 1 and 512.")
    if not 0 <= args.unfrozen_codebert_layers <= 12:
        raise ValueError("unfrozen_codebert_layers must be between 0 and 12.")
    if args.gnn_layers < 1:
        raise ValueError("gnn_layers must be at least 1.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1).")
    if args.epochs < 1:
        raise ValueError("epochs must be at least 1.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1).")


def resolve_saved_source(model_dir: Path, source: str) -> Path:
    candidate = Path(source)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    predictor_root = model_dir.resolve().parents[1]
    resolved = predictor_root / candidate
    if not resolved.exists():
        raise FileNotFoundError(f"Saved CodeBERT source does not exist: {resolved}")
    return resolved


def build_loader(
    dataset: FusionDataset,
    tokenizer,
    args,
    shuffle: bool,
) -> DataLoader:
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
        collate_fn=FusionCollator(tokenizer),
        num_workers=0,
    )


def build_predictions(
    examples: list[dict],
    example_indices: list[int],
    predicted_ids: list[int],
    probabilities: list[list[float]],
) -> list[dict]:
    rows = []
    for index, predicted_id, probability in zip(
        example_indices,
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


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_random_seed(args.random_state)
    output_dir = ensure_new_output_dir(args.output_dir)
    device = choose_device(args.device)

    if args.split_mode == "fixed":
        splits, thresholds = load_fixed_splits(args.split_dir)
        errors = []
        split_source = str(Path(args.split_dir))
    else:
        examples, errors = load_source_examples(args.input)
        splits, thresholds = split_by_repository(
            examples,
            random_state=args.random_state,
        )
        split_source = str(Path(args.input))

    initialization = {"mode": "fresh", "initial_model_dir": None}
    if args.initial_model_dir:
        initial_model_dir = Path(args.initial_model_dir)
        checkpoint = torch.load(
            initial_model_dir / "model_trainable.pt",
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint["label_mapping"] != LABEL_TO_ID:
            raise ValueError("Initial checkpoint label mapping differs from implementation mapping.")
        vocab_data = checkpoint["vocabularies"]
        vocabularies = Vocabularies(
            Vocabulary.from_dict(vocab_data["node_types"]),
            Vocabulary.from_dict(vocab_data["fields"]),
            Vocabulary.from_dict(vocab_data["lexical"]),
        )
        tokenizer = AutoTokenizer.from_pretrained(initial_model_dir / "tokenizer")
        model_config = dict(checkpoint["model_config"])
        saved_codebert_source = str(
            resolve_saved_source(initial_model_dir, checkpoint["codebert_source"])
        )
        model_config["codebert_model"] = saved_codebert_source
        initialization = {
            "mode": "fine_tune_saved_fusion",
            "initial_model_dir": str(initial_model_dir),
            "initial_checkpoint": str(initial_model_dir / "model_trainable.pt"),
        }
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.codebert_model)
        vocabularies = build_train_vocabularies(splits["train"])
        saved_codebert_source = args.codebert_model
        model_config = {
            "codebert_model": args.codebert_model,
            "node_type_vocab_size": len(vocabularies.node_types),
            "field_vocab_size": len(vocabularies.fields),
            "lexical_vocab_size": len(vocabularies.lexical),
            "hidden_dim": args.hidden_dim,
            "lexical_dim": args.lexical_dim,
            "gnn_layers": args.gnn_layers,
            "fusion_dim": args.fusion_dim,
            "fusion_hidden_dim": args.fusion_hidden_dim,
            "dropout": args.dropout,
            "num_relations": len(EDGE_TYPE_TO_ID),
            "num_labels": len(LABEL_TO_ID),
            "unfrozen_codebert_layers": args.unfrozen_codebert_layers,
            "train_codebert_embeddings": args.train_codebert_embeddings,
        }
    datasets = {
        name: FusionDataset(
            examples=examples,
            vocabularies=vocabularies,
            tokenizer=tokenizer,
            max_length=args.max_length,
        )
        for name, examples in splits.items()
    }
    loaders = {
        name: build_loader(
            dataset,
            tokenizer,
            args,
            shuffle=name == "train",
        )
        for name, dataset in datasets.items()
    }

    model = CodeBertAstFusionClassifier(**model_config).to(device)
    if args.initial_model_dir:
        load_trainable_state_dict(model, checkpoint["trainable_state_dict"])
    counts = parameter_counts(model)
    print(f"Device: {device}")
    print(
        f"Trainable parameters: {counts['trainable']:,} / {counts['total']:,} "
        f"({counts['trainable'] / counts['total']:.2%})"
    )
    print(
        "AST policy: no node truncation; "
        f"up to {args.max_total_nodes_per_batch} total nodes per normal batch"
    )

    history = train_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        codebert_learning_rate=args.codebert_learning_rate,
        new_layers_learning_rate=args.new_layers_learning_rate,
        codebert_weight_decay=args.codebert_weight_decay,
        new_layers_weight_decay=args.new_layers_weight_decay,
        warmup_ratio=args.warmup_ratio,
        patience=args.patience,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_clip_norm=args.gradient_clip_norm,
        checkpoint_path=output_dir / ".best_trainable_state.pt",
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

    run_config = {
        **vars(args),
        "device": str(device),
        "split_source": split_source,
    }
    report = {
        "experiment": "codebert_requirement_ast_gnn_fusion",
        "input_contract": {
            "model_inputs": ["requirement", "old_function_code_ast"],
            "excluded_model_inputs": [
                "new_function_code",
                "new_function_code_by_llm",
                "human_delta",
                "llm_delta",
                "human_ast_edit_script",
                "llm_ast_edit_script",
                "ast_delta_similarity",
            ],
            "label_source": "ast_delta_similarity",
            "threshold_source": (
                args.threshold_source_description
                if args.split_mode == "fixed"
                else "repo_grouped_training_split_only"
            ),
        },
        "split_mode": args.split_mode,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "class_distribution": {
            name: class_distribution(rows) for name, rows in splits.items()
        },
        "percentile_thresholds": thresholds,
        "split_overlap_audit": split_overlap_audit(splits),
        "length_audit": {
            name: length_audit(dataset)
            for name, dataset in datasets.items()
        },
        "validation_metrics": evaluations["validation"],
        "test_metrics": evaluations["test"],
        "model_config": model_config,
        "codebert_trainability": model.trainability,
        "parameter_counts": counts,
        "run_config": run_config,
        "initialization": initialization,
    }

    tokenizer.save_pretrained(output_dir / "tokenizer")
    save_json(vocabularies.to_dict(), output_dir / "vocabularies.json")
    save_json(LABEL_TO_ID, output_dir / "label_mapping.json")
    save_json(EDGE_TYPE_TO_ID, output_dir / "edge_type_mapping.json")
    save_json(thresholds, output_dir / "percentile_thresholds.json")
    save_json(history, output_dir / "training_history.json")
    save_json(errors, output_dir / "errors.json")
    save_json(run_config, output_dir / "run_config.json")
    save_json(report, output_dir / "evaluation_report.json")
    save_json(test_predictions, output_dir / "test_predictions.json")
    for split_name, rows in splits.items():
        save_json(
            [serializable_manifest(row) for row in rows],
            output_dir / f"{split_name}_manifest.json",
        )
    torch.save(
        {
            "architecture": "codebert_requirement_relational_ast_gnn_concat_fusion",
            "model_config": model_config,
            "trainable_state_dict": trainable_state_dict(model),
            "codebert_source": saved_codebert_source,
            "vocabularies": vocabularies.to_dict(),
            "percentile_thresholds": thresholds,
            "label_mapping": LABEL_TO_ID,
            "edge_type_mapping": EDGE_TYPE_TO_ID,
        },
        output_dir / "model_trainable.pt",
    )
    print("\nFinal summary")
    print(f"Validation macro F1: {evaluations['validation']['macro_f1']:.4f}")
    print(f"Test macro F1: {evaluations['test']['macro_f1']:.4f}")
    print(f"Test accuracy: {evaluations['test']['accuracy']:.4f}")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
