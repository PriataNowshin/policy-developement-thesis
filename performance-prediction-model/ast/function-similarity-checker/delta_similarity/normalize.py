"""Code normalization logic."""

from __future__ import annotations

import textwrap
from typing import List, Tuple


def _split_newline_suffix(line: str) -> Tuple[str, str]:
    if line.endswith("\r\n"):
        return line[:-2], "\r\n"
    if line.endswith("\n"):
        return line[:-1], "\n"
    if line.endswith("\r"):
        return line[:-1], "\r"
    return line, ""


def normalize_code(code: str) -> str:
    """Lightly normalize code before diffing.

    - Use textwrap.dedent
    - Remove trailing whitespace from each line
    - Preserve line structure
    - Do not force-add or force-remove a final trailing newline
    """

    dedented = textwrap.dedent(code)

    # Keep line endings so we preserve whether the last line had a newline.
    lines = dedented.splitlines(keepends=True)
    if not lines:
        return dedented

    normalized_lines: List[str] = []
    for line in lines:
        body, newline = _split_newline_suffix(line)
        body = body.rstrip(" \t")
        normalized_lines.append(body + newline)

    return "".join(normalized_lines)
