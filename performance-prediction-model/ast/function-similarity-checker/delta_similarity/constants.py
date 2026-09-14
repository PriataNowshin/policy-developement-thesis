"""Project-wide constants."""

from __future__ import annotations

from typing import Tuple

REQUIRED_FIELDS: Tuple[str, ...] = (
    "repo_full_name",
    "file_path",
    "function_name",
    "old_function_code",
    "new_function_code",
    "new_function_code_by_llm",
    "requirement",
)
