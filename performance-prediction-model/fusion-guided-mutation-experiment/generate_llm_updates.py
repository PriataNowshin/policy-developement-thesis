#!/usr/bin/env python3
"""Resumably ask an LLM to implement each controlled mutation requirement."""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llm-function-generation"))
from src.constants import MODEL_MAP  # noqa: E402
from src.llm_client import build_config, generate_function  # noqa: E402
from src.prompts import FUNCTION_GENERATION_PROMPT_VERSION  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--llm", required=True, help="Alias from MODEL_MAP or an explicit provider model slug")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--record_retries", type=int, default=4)
    parser.add_argument("--retry_base_seconds", type=float, default=5.0)
    return parser.parse_args()


def save(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def clean_code(text: str) -> str:
    value = text.strip()
    fenced = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", value, flags=re.DOTALL | re.IGNORECASE)
    return fenced.group(1).strip() if fenced else value


def validate_generated_code(code: str) -> None:
    """Reject instruction markup and syntactically invalid model output."""
    if "<deleted>" in code:
        raise ValueError("LLM copied the deletion instruction marker into generated code")
    ast.parse(code)


def main() -> None:
    args = arguments()
    source = json.loads(args.input.read_text(encoding="utf-8"))
    if args.limit:
        source = source[:args.limit]
    model_slug = MODEL_MAP.get(args.llm, args.llm)
    config = build_config(model_slug)
    output = json.loads(args.output.read_text(encoding="utf-8")) if args.resume and args.output.exists() else []
    by_id = {row["id"]: row for row in output}
    failures = []
    for position, row in enumerate(source, 1):
        if row["id"] in by_id and by_id[row["id"]].get("generation_status") == "success":
            continue
        error = None
        generated = ""
        for attempt in range(args.record_retries + 1):
            try:
                generated = clean_code(generate_function(row["old_function_code"], row["requirement"], config))
                validate_generated_code(generated)
                error = None
                break
            except Exception as exc:  # preserve progress across provider and parse failures
                error = f"{type(exc).__name__}: {exc}"
                if attempt < args.record_retries:
                    time.sleep(min(120.0, args.retry_base_seconds * (2 ** attempt)))
        result = dict(row)
        result.update({
            "new_function_code": row["expected_mutated_function"],
            "new_function_code_by_llm": generated,
            "llm_generated_function": generated,
            "llm_model": model_slug,
            "llm_model_alias": args.llm,
            "prompt_version": FUNCTION_GENERATION_PROMPT_VERSION,
            "generation_status": "success" if error is None else "failed",
            "generation_error": error or "",
        })
        by_id[row["id"]] = result
        output = [by_id[item["id"]] for item in source if item["id"] in by_id]
        save(args.output, output)
        if error:
            failures.append({"id": row["id"], "error": error})
        print(f"[{position}/{len(source)}] {row['id']}: {result['generation_status']}", flush=True)
    audit = {
        "requested_records": len(source),
        "written_records": len(output),
        "status_distribution": dict(Counter(row["generation_status"] for row in output)),
        "model": model_slug,
        "failed_records": failures,
    }
    save(args.output.with_name(args.output.stem + "_audit.json"), audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
