"""Fine-tuning and prediction loops for requirement-only CodeBERT."""

import random
from pathlib import Path

import numpy as np
import torch
from transformers import get_linear_schedule_with_warmup

from .metrics import evaluate_predictions


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def predict(model, data_loader, device) -> tuple[list[int], list[int], list[list[float]]]:
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for batch in data_loader:
            batch = _move_batch(batch, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            batch_probabilities = torch.softmax(outputs.logits, dim=-1)
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
    warmup_ratio: float,
    patience: int,
    checkpoint_path: str | Path,
) -> list[dict]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    total_steps = max(len(train_loader) * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * warmup_ratio),
        num_training_steps=total_steps,
    )
    checkpoint_path = Path(checkpoint_path)
    best_macro_f1 = -1.0
    epochs_without_improvement = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in train_loader:
            batch = _move_batch(batch, device)
            optimizer.zero_grad()
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            batch_size = batch["labels"].size(0)
            total_loss += float(outputs.loss.item()) * batch_size
            total_examples += batch_size

        true_labels, predicted_labels, _ = predict(
            model,
            validation_loader,
            device,
        )
        metrics = evaluate_predictions(true_labels, predicted_labels)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_macro_f1": metrics["macro_f1"],
            "validation_accuracy": metrics["accuracy"],
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d} | loss={record['train_loss']:.4f} | "
            f"val_macro_f1={record['validation_macro_f1']:.4f}"
        )

        if metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = metrics["macro_f1"]
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
    return history
