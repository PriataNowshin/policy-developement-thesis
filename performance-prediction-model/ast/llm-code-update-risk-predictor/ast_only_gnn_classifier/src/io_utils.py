"""JSON and output-directory helpers."""

import json
from pathlib import Path
from typing import Any


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: Any, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def ensure_new_output_dir(path: str | Path) -> Path:
    output_path = Path(path)
    if output_path.exists() and any(output_path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_path}. "
            "Use a new directory for this experiment."
        )
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path
