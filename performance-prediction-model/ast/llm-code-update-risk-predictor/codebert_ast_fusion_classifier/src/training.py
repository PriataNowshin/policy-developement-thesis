"""Differential-rate training and evaluation for the fusion model."""

import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import get_linear_schedule_with_warmup

from .metrics import evaluate_predictions

MODEL_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
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


def trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def load_trainable_state_dict(
    model: nn.Module,
    state: dict[str, torch.Tensor],
) -> None:
    expected_trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    absent = expected_trainable - set(state)
    if absent:
        raise RuntimeError(f"Trainable checkpoint is missing keys: {sorted(absent)}")
    _, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"Trainable checkpoint has unexpected keys: {unexpected}")


def build_optimizer(
    model: nn.Module,
    codebert_learning_rate: float,
    new_layers_learning_rate: float,
    codebert_weight_decay: float,
    new_layers_weight_decay: float,
) -> torch.optim.AdamW:
    """Use small updates for pretrained layers and larger updates for new layers."""
    groups = {
        ("codebert", "decay"): [],
        ("codebert", "no_decay"): [],
        ("new", "decay"): [],
        ("new", "no_decay"): [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = "codebert" if name.startswith("codebert.") else "new"
        no_decay = name.endswith(".bias") or "normalization" in name.lower()
        no_decay = no_decay or "layernorm" in name.lower() or "layer_norm" in name.lower()
        groups[(family, "no_decay" if no_decay else "decay")].append(parameter)

    parameter_groups = []
    for (family, decay_kind), parameters in groups.items():
        if not parameters:
            continue
        parameter_groups.append(
            {
                "params": parameters,
                "lr": (
                    codebert_learning_rate
                    if family == "codebert"
                    else new_layers_learning_rate
                ),
                "weight_decay": (
                    0.0
                    if decay_kind == "no_decay"
                    else (
                        codebert_weight_decay
                        if family == "codebert"
                        else new_layers_weight_decay
                    )
                ),
            }
        )
    return torch.optim.AdamW(parameter_groups)


def predict(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[list[int], list[int], list[int], list[list[float]]]:
    model.eval()
    example_indices: list[int] = []
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[list[float]] = []

    with torch.no_grad():
        for batch in data_loader:
            batch = _to_device(batch, device)
            logits = model(**_model_inputs(batch))
            batch_probabilities = torch.softmax(logits, dim=-1)
            example_indices.extend(batch["example_indices"].cpu().tolist())
            true_labels.extend(batch["labels"].cpu().tolist())
            predicted_labels.extend(batch_probabilities.argmax(dim=-1).cpu().tolist())
            probabilities.extend(batch_probabilities.cpu().tolist())
    return example_indices, true_labels, predicted_labels, probabilities


def train_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    codebert_learning_rate: float,
    new_layers_learning_rate: float,
    codebert_weight_decay: float,
    new_layers_weight_decay: float,
    warmup_ratio: float,
    patience: int,
    gradient_accumulation_steps: int,
    gradient_clip_norm: float,
    checkpoint_path: str | Path,
) -> list[dict]:
    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1.")
    optimizer = build_optimizer(
        model,
        codebert_learning_rate=codebert_learning_rate,
        new_layers_learning_rate=new_layers_learning_rate,
        codebert_weight_decay=codebert_weight_decay,
        new_layers_weight_decay=new_layers_weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / gradient_accumulation_steps
    )
    total_steps = max(optimizer_steps_per_epoch * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint_path = Path(checkpoint_path)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
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

            should_step = batch_index % gradient_accumulation_steps == 0
            if should_step:
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=gradient_clip_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # Shuffled node-budget batches can vary slightly in count. Always flush
        # the final partial accumulation group instead of trusting __len__.
        if batch_index % gradient_accumulation_steps != 0:
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                max_norm=gradient_clip_norm,
            )
            optimizer.step()
            scheduler.step()
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
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | "
            f"val_macro_f1={record['validation_macro_f1']:.4f} | "
            f"val_accuracy={record['validation_accuracy']:.4f}"
        )

        if validation_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = validation_metrics["macro_f1"]
            torch.save(trainable_state_dict(model), checkpoint_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if patience > 0 and epochs_without_improvement >= patience:
                print(f"Early stopping after epoch {epoch}.")
                break

    best_state = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    load_trainable_state_dict(model, best_state)
    checkpoint_path.unlink(missing_ok=True)
    return history
