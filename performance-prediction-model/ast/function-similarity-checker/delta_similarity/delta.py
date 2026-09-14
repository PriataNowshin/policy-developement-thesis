"""Changed-line-only delta extraction."""

from __future__ import annotations

import difflib
from typing import List


def extract_changed_delta(old_code: str, updated_code: str) -> str:
    """Extract a changed-line-only delta between two code strings.

    Diff rules (line-based):
      - removed lines prefixed with "- "
      - added lines prefixed with "+ "
      - no unchanged lines
      - no metadata
      - for "replace" edits, list all removed lines first, followed by all added lines

    Blank changed lines are preserved as:
      - removed blank line => "- "
      - added blank line   => "+ "

    The returned delta is joined with "\n" and contains no extra leading/trailing newline.
    """

    old_lines = old_code.splitlines(keepends=False)
    updated_lines = updated_code.splitlines(keepends=False)

    matcher = difflib.SequenceMatcher(a=old_lines, b=updated_lines, autojunk=False)

    delta_lines: List[str] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue

        if tag in {"delete", "replace"}:
            for line in old_lines[i1:i2]:
                delta_lines.append(f"- {line}")

        if tag in {"insert", "replace"}:
            for line in updated_lines[j1:j2]:
                delta_lines.append(f"+ {line}")

    return "\n".join(delta_lines)
