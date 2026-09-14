#!/usr/bin/env python3
"""CLI entrypoint for AST edit-script delta similarity.

The score compares the structural edit script for:
  - human update: old_function_code -> new_function_code
  - LLM update:   old_function_code -> new_function_code_by_llm
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

from delta_similarity.io_utils import load_json_data, save_json_output
from delta_similarity.processing import build_summary, process_item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute AST edit-script similarity between human and LLM code changes."
    )
    parser.add_argument("--input", required=True, help="Path to input JSON file (dataset list).")
    parser.add_argument(
        "--output",
        default="output_ast/ast_delta_similarity_results.json",
        help="Path to output JSON file. Defaults to output_ast/ast_delta_similarity_results.json.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of items to iterate. Use 0 for all items.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    items = load_json_data(input_path)
    total_examples = len(items)
    if args.limit > 0:
        items = items[: args.limit]

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    for idx, item in enumerate(items):
        if idx > 0 and idx % 100 == 0:
            print(
                "Progress: "
                f"iterated={idx}/{len(items)} "
                f"processed={len(results)} "
                f"skipped={len(errors)}"
            )

        result, error = process_item(idx, item)
        if error is not None:
            errors.append(error)
            continue
        if result is not None:
            results.append(result)

    summary = build_summary(results, errors, total_examples if args.limit == 0 else len(items))
    output_data = {"summary": summary, "results": results, "errors": errors}
    save_json_output(output_path, output_data)

    ast_summary = summary["ast_delta_similarity"]

    print("\nDone.")
    print(f"Total examples: {summary['total_examples']}")
    print(f"Processed examples: {summary['processed_examples']}")
    print(f"Skipped/error examples: {summary['skipped_examples']}")
    print(f"Average AST delta similarity: {ast_summary['average']:.6f}")
    print(f"Minimum AST delta similarity: {ast_summary['minimum']:.6f}")
    print(f"Maximum AST delta similarity: {ast_summary['maximum']:.6f}")
    print(f"Output written to: {output_path}")


if __name__ == "__main__":
    main()
