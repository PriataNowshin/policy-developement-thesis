#!/usr/bin/env python3
"""Merge tracked LLM batches and verify them against the complete mutation cohort."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def save(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def digest(ids: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def main() -> None:
    args = arguments()
    expected = load(args.expected)
    expected_ids = [row["id"] for row in expected]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Expected cohort contains duplicate IDs")

    batches = [load(path) for path in args.batch]
    merged_rows = [row for batch in batches for row in batch]
    merged_ids = [row["id"] for row in merged_rows]
    duplicate_ids = sorted(row_id for row_id, count in Counter(merged_ids).items() if count > 1)
    missing_ids = sorted(set(expected_ids) - set(merged_ids))
    unexpected_ids = sorted(set(merged_ids) - set(expected_ids))
    if duplicate_ids or missing_ids or unexpected_ids:
        raise RuntimeError(
            f"Invalid merge: duplicates={len(duplicate_ids)}, missing={len(missing_ids)}, "
            f"unexpected={len(unexpected_ids)}"
        )

    by_id = {row["id"]: row for row in merged_rows}
    merged = [by_id[row_id] for row_id in expected_ids]
    save(args.output, merged)
    status_counts = Counter(row.get("generation_status") for row in merged)
    failed = [
        {
            "id": row["id"],
            "source_split": row["source_split"],
            "mutation_operator": row["mutation_operator"],
            "error": row.get("generation_error", ""),
        }
        for row in merged
        if row.get("generation_status") != "success"
    ]
    audit = {
        "expected_records": len(expected),
        "batch_files": [str(path) for path in args.batch],
        "batch_sizes": [len(batch) for batch in batches],
        "merged_records": len(merged),
        "unique_merged_ids": len(set(merged_ids)),
        "duplicate_ids": 0,
        "missing_ids": 0,
        "unexpected_ids": 0,
        "expected_id_sha256": digest(set(expected_ids)),
        "merged_id_sha256": digest(set(merged_ids)),
        "status_distribution": dict(status_counts),
        "prompt_version_distribution": dict(Counter(row.get("prompt_version") for row in merged)),
        "model_distribution": dict(Counter(row.get("llm_model") for row in merged)),
        "failed_records": failed,
        "llm_calls_made_during_merge": 0,
    }
    audit_path = args.output.with_name(args.output.stem + "_merge_audit.json")
    save(audit_path, audit)
    print(json.dumps(audit, indent=2))
    print(f"Output: {args.output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
