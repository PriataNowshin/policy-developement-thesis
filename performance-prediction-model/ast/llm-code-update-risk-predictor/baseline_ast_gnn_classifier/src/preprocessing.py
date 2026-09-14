"""Validate AST-similarity records and build old-code graphs."""

from typing import Any

from .ast_graph import build_ast_graph
from .config import REQUIRED_FIELDS


def extract_results(input_data: dict) -> list[dict]:
    results = input_data.get("results")
    if not isinstance(results, list):
        raise ValueError('Input JSON must contain a top-level "results" list.')
    return results


def _validate_item(item: dict, index: int) -> dict | None:
    missing = [field for field in REQUIRED_FIELDS if field not in item]
    if missing:
        return {
            "index": index,
            "reason": "missing_required_fields",
            "missing_fields": missing,
        }

    for field in ("old_function_code", "requirement"):
        if not isinstance(item[field], str) or not item[field].strip():
            return {"index": index, "reason": f"invalid_{field}"}

    score = item["ast_delta_similarity"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        return {"index": index, "reason": "non_numeric_ast_delta_similarity"}
    if not 0.0 <= float(score) <= 1.0:
        return {"index": index, "reason": "score_outside_zero_one"}
    return None


def _metadata(item: dict[str, Any]) -> dict:
    return {
        "repo_full_name": item["repo_full_name"],
        "file_path": item["file_path"],
        "function_name": item["function_name"],
        "requirement": item["requirement"],
        "old_function_code": item["old_function_code"],
        "ast_delta_similarity": float(item["ast_delta_similarity"]),
    }


def prepare_examples(results: list[dict]) -> tuple[list[dict], list[dict]]:
    """Parse only old_function_code; updated code is never copied or read."""
    examples: list[dict] = []
    errors: list[dict] = []

    for index, item in enumerate(results):
        error = _validate_item(item, index)
        if error is not None:
            errors.append(error)
            continue

        try:
            graph = build_ast_graph(item["old_function_code"])
        except (SyntaxError, ValueError, TypeError) as parse_error:
            errors.append(
                {
                    "index": index,
                    "reason": "old_function_ast_parse_error",
                    "message": str(parse_error),
                }
            )
            continue

        example = _metadata(item)
        example["graph"] = graph
        examples.append(example)

    return examples, errors


def serializable_example(example: dict) -> dict:
    """Return an audit record without the expanded graph representation."""
    return {key: value for key, value in example.items() if key != "graph"}
