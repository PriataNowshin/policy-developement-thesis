from __future__ import annotations

import json
from pathlib import Path
import os
from typing import Any, Dict, List


def save_data_as_json(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    """Write an iterable of records to a JSON file.

    Writes a list of JSON objects and overwrites any existing file at `path`.
    If `path` includes a parent directory, it is created automatically.

    Args:
        path: Output file path (e.g., "output/function_changes.json").
        rows: List of JSON-serializable mapping objects.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(rows, file, ensure_ascii=False, indent=2)