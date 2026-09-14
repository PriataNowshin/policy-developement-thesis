"""Training and prediction loops for the AST-GNN model."""

import copy
import random

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
    "lexical_ids",
    "lexical_node_indices",
    "requirement_ids",
    "requirement_batch",
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


def predict(model, data_loader, device) -> tuple[list[int], list[int], list[list[float]]]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[list[float]] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = _to_device(batch, device)
            logits = model(**_model_inputs(batch))
            batch_probabilities = torch.softmax(logits, dim=-1)
            true_labels.extend(batch["labels"].cpu().tolist())
            predicted_labels.extend(batch_probabilities.argmax(dim=-1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return true_labels, predicted_labels, probabilities


def train_model(
    model,
    train_loader,
    validation_loader,
    device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    patience: int,
    class_weights: torch.Tensor,
) -> tuple[dict, list[dict]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    best_state = copy.deepcopy(model.state_dict())
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            batch = _to_device(batch, device)
            optimizer.zero_grad()
            logits = model(**_model_inputs(batch))
            loss = criterion(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item()) * batch["labels"].size(0)
            total_examples += batch["labels"].size(0)

        true_labels, predicted_labels, _ = predict(
            model,
            validation_loader,
            device,
        )
        validation_metrics = evaluate_predictions(
            true_labels,
            predicted_labels,
        )
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_macro_f1": validation_metrics["macro_f1"],
            "validation_accuracy": validation_metrics["accuracy"],
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch:03d} | "
            f"loss={epoch_record['train_loss']:.4f} | "
            f"val_macro_f1={epoch_record['validation_macro_f1']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return best_state, history


def compute_class_weights(examples: list[dict], num_labels: int) -> torch.Tensor:
    counts = torch.bincount(
        torch.tensor([example["label_id"] for example in examples]),
        minlength=num_labels,
    ).float()
    weights = counts.sum() / (num_labels * counts.clamp_min(1.0))
    return weights
