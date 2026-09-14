"""Input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_json_data(input_path: Path) -> List[Dict[str, Any]]:
    """Load dataset items from a JSON file.

    Expected input format: a JSON array of objects.
    """

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list of items")

    items: List[Dict[str, Any]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item at index {idx} is not an object")
        items.append(item)

    return items


def save_json_output(output_path: Path, output_data: Dict[str, Any]) -> None:
    """Save output data to a JSON file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
