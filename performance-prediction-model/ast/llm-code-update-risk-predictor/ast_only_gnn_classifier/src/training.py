"""Training and prediction loops for the true AST-only GNN."""

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .metrics import evaluate_predictions

MODEL_INPUT_KEYS = (
    "node_type_ids",
    "field_ids",
    "depths",
    "graph_batch",
    "edge_index",
    "edge_type_ids",
    "lexical_ids",
    "lexical_node_indices",
)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _model_inputs(batch: dict) -> dict:
    return {key: batch[key] for key in MODEL_INPUT_KEYS}


def predict(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[list[int], list[int], list[int], list[list[float]]]:
    model.eval()
    indices: list[int] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = _to_device(batch, device)
            logits = model(**_model_inputs(batch))
            batch_probabilities = torch.softmax(logits, dim=-1)
            indices.extend(batch["example_indices"].cpu().tolist())
            true_labels.extend(batch["labels"].cpu().tolist())
            predicted_labels.extend(batch_probabilities.argmax(dim=-1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return indices, true_labels, predicted_labels, probabilities


def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    gradient_clip_norm: float,
    gradient_accumulation_steps: int,
    checkpoint_path: str | Path,
) -> tuple[list[dict], int]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(checkpoint_path)
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        total_loss = 0.0
        total_examples = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            batch = _to_device(batch, device)
            logits = model(**_model_inputs(batch))
            unscaled_loss = criterion(logits, batch["labels"])
            (unscaled_loss / gradient_accumulation_steps).backward()
            batch_size = batch["labels"].size(0)
            total_loss += float(unscaled_loss.item()) * batch_size
            total_examples += batch_size

            if batch_index % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=gradient_clip_norm,
                )
                optimizer.step()
                optimizer.zero_grad()

        if batch_index % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        _, true_labels, predicted_labels, _ = predict(
            model,
            validation_loader,
            device,
        )
        validation_metrics = evaluate_predictions(true_labels, predicted_labels)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
            "epoch_seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | "
            f"val_macro_f1={record['validation_macro_f1']:.4f} | "
            f"val_accuracy={record['validation_accuracy']:.4f} | "
            f"seconds={record['epoch_seconds']:.1f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    checkpoint_path.unlink(missing_ok=True)
    return history, best_epoch
