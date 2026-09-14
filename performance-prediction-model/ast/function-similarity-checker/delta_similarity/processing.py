"""Per-item processing, validation, and AST similarity summary building."""

from __future__ import annotations

import statistics
from typing import Any, Dict, Optional, Sequence, Tuple

from .ast_similarity import compare_ast_deltas
from .constants import REQUIRED_FIELDS
from .delta import extract_changed_delta
from .normalize import normalize_code


def validate_item(item: Dict[str, Any], required_fields: Sequence[str] = REQUIRED_FIELDS) -> Tuple[bool, str]:
    """Validate that the item contains all required fields."""

    for field in required_fields:
        if field not in item:
            return False, f"Missing required field: {field}"
        if item[field] is None:
            return False, f"Field is None: {field}"
        if not isinstance(item[field], str):
            return False, f"Field is not a string: {field}"

    return True, ""


def process_item(
    index: int,
    item: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Process a single dataset item with AST edit-script similarity.

    Returns:
      (result_record, None) on success
      (None, error_record) on failure
    """

    ok, message = validate_item(item)
    if not ok:
        return None, {
            "index": index,
            "repo_full_name": item.get("repo_full_name", ""),
            "file_path": item.get("file_path", ""),
            "function_name": item.get("function_name", ""),
            "parse_success": False,
            "error": message,
        }

    try:
        old_raw = item["old_function_code"]
        new_raw = item["new_function_code"]
        llm_raw = item["new_function_code_by_llm"]

        old_norm = normalize_code(old_raw)
        new_norm = normalize_code(new_raw)
        llm_norm = normalize_code(llm_raw)

        human_delta = extract_changed_delta(old_norm, new_norm)
        llm_delta = extract_changed_delta(old_norm, llm_norm)
        ast_scores = compare_ast_deltas(old_norm, new_norm, llm_norm)

        result_record = {
            "index": index,
            "id": item.get("id", ""),
            "repo_full_name": item["repo_full_name"],
            "file_path": item["file_path"],
            "function_name": item["function_name"],
            "old_function_code": old_raw,
            "new_function_code": new_raw,
            "new_function_code_by_llm": llm_raw,
            "requirement": item["requirement"],
            "human_delta": human_delta,
            "llm_delta": llm_delta,
            "parse_success": True,
            **ast_scores,
        }

        return result_record, None
    except Exception as exc:  # noqa: BLE001 - record and continue
        return None, {
            "index": index,
            "id": item.get("id", ""),
            "repo_full_name": item.get("repo_full_name", ""),
            "file_path": item.get("file_path", ""),
            "function_name": item.get("function_name", ""),
            "parse_success": False,
            "error": f"Failed processing item: {type(exc).__name__}: {exc}",
        }


def build_summary(
    results: Sequence[Dict[str, Any]],
    errors: Sequence[Dict[str, Any]],
    total_examples: int,
) -> Dict[str, Any]:
    """Build summary stats over processed examples only."""

    processed = len(results)
    skipped = len(errors)

    def metric_summary(field: str) -> Dict[str, float]:
        values = [float(r[field]) for r in results]
        if not values:
            return {"average": 0.0, "minimum": 0.0, "maximum": 0.0}
        return {
            "average": float(statistics.mean(values)),
            "minimum": float(min(values)),
            "maximum": float(max(values)),
        }

    return {
        "total_examples": total_examples,
        "processed_examples": processed,
        "skipped_examples": skipped,
        "ast_delta_similarity": metric_summary("ast_delta_similarity"),
        "operation_precision": metric_summary("operation_precision"),
        "operation_recall": metric_summary("operation_recall"),
        "final_ast_similarity": metric_summary("final_ast_similarity"),
    }
