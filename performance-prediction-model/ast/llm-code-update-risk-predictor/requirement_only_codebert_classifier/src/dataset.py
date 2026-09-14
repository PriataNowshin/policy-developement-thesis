"""Requirement-only tokenization and dynamic padding."""

import torch
from torch.utils.data import Dataset


class RequirementDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        encoded = self.tokenizer(
            record["requirement"],
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "label": record["label_id"],
        }


class RequirementCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        labels = torch.tensor([item["label"] for item in items], dtype=torch.long)
        features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item["attention_mask"],
            }
            for item in items
        ]
        batch = self.tokenizer.pad(features, padding=True, return_tensors="pt")
        return {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": labels,
        }
