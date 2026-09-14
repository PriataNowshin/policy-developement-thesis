#!/usr/bin/env python3
"""Select a fixed source cohort, then attempt every controlled operator on each source."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from controlled_mutation import OPERATOR_SPECS, apply_operator
from generate_mutation_dataset import (
    _ast_changed,
    _parses,
    _requirement_is_faithful,
    load_fixed_sources,
    save_json,
    taxonomy_support,
)
from verify_study_model import main as verify_study_model

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output/step6_full_cohort/mutations_1000_sources.json"


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-sources", type=int, default=700)
    parser.add_argument("--validation-sources", type=int, default=140)
    parser.add_argument("--test-sources", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def selected_sources(rows: list[dict], split: str, count: int, seed: int) -> list[dict]:
    if count < 1 or count > len(rows):
        raise ValueError(f"{split}: requested {count} sources from {len(rows)} available")
    ordered = list(rows)
    material = f"fusion-guided-full-cohort-v1\0{seed}\0{split}".encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.Random(local_seed).shuffle(ordered)
    return ordered[:count]


def record_for(source: dict, split: str, operator_id: str, support_count: int):
    original = textwrap.dedent(source["old_function_code"]).strip("\n") + "\n"
    parsed = ast.parse(original)
    old_code = ast.unparse(parsed).strip() + "\n"
    mutation = apply_operator(old_code, operator_id)
    if mutation is None:
        return None
    return {
        "id": f"fgm_cohort_v1_{split}_{source['source_index']}_{operator_id}",
        "source_index": source["source_index"],
        "source_split": split,
        "repo_full_name": source["repo_full_name"],
        "file_path": source["file_path"],
        "function_name": source["function_name"],
        "original_old_function_code": original,
        "old_function_code": old_code,
        "expected_mutated_function": mutation.code,
        "requirement": mutation.requirement,
        "mutation_operator": mutation.operator_id,
        "taxonomy_category": mutation.taxonomy_category,
        "taxonomy_support_count": support_count,
        "mutation_location": mutation.location,
        "old_fragment": mutation.old_fragment,
        "new_fragment": mutation.new_fragment,
        "llm_generated_function": "",
        "llm_model": "",
        "prompt_version": "",
        "generation_status": "mutation_only",
    }


def main() -> None:
    args = arguments()
    verify_study_model()
    available = load_fixed_sources()
    support = taxonomy_support()
    requested = {
        "train": args.train_sources,
        "validation": args.validation_sources,
        "test": args.test_sources,
    }
    cohort = {
        split: selected_sources(available[split], split, count, args.seed)
        for split, count in requested.items()
    }
    source_ids = [row["source_index"] for rows in cohort.values() for row in rows]
    if len(source_ids) != len(set(source_ids)):
        raise RuntimeError("Selected source functions are not unique")

    records = []
    attempts = Counter()
    applicable = Counter()
    failures = []
    per_source = Counter()
    for split, sources in cohort.items():
        for source in sources:
            for operator_id, spec in OPERATOR_SPECS.items():
                attempts[(split, operator_id)] += 1
                try:
                    record = record_for(source, split, operator_id, support[spec.taxonomy_category])
                except (SyntaxError, TypeError, ValueError) as error:
                    failures.append({
                        "source_index": source["source_index"],
                        "source_split": split,
                        "mutation_operator": operator_id,
                        "reason": f"{type(error).__name__}: {error}",
                    })
                    continue
                if record is None:
                    continue
                records.append(record)
                applicable[(split, operator_id)] += 1
                per_source[source["source_index"]] += 1

    if len({row["id"] for row in records}) != len(records):
        raise RuntimeError("Duplicate record IDs detected")
    if len({(row["source_index"], row["mutation_operator"]) for row in records}) != len(records):
        raise RuntimeError("Duplicate source/operator pairs detected")
    if not all(_requirement_is_faithful(row) for row in records):
        raise RuntimeError("At least one requirement does not embed its exact fragments")
    if not all(_parses(row["old_function_code"]) and _parses(row["expected_mutated_function"]) for row in records):
        raise RuntimeError("At least one generated record does not parse")
    if not all(_ast_changed(row) for row in records):
        raise RuntimeError("At least one mutation is not AST-visible")

    split_by_source = defaultdict(set)
    for split, rows in cohort.items():
        for row in rows:
            split_by_source[row["source_index"]].add(split)
    if any(len(splits) != 1 for splits in split_by_source.values()):
        raise RuntimeError("Source cohort leakage detected")

    records.sort(key=lambda row: (row["source_split"], row["source_index"], row["mutation_operator"]))
    save_json(args.output, records)
    for split in requested:
        save_json(args.output.with_name(f"{args.output.stem}_{split}.json"), [row for row in records if row["source_split"] == split])
    save_json(args.output.with_name("operator_specifications.json"), [asdict(spec) for spec in OPERATOR_SPECS.values()])
    save_json(args.output.with_name("selected_source_cohort.json"), {
        split: [{key: row[key] for key in ("source_index", "repo_full_name", "file_path", "function_name")} for row in rows]
        for split, rows in cohort.items()
    })

    operator_attempts = {
        operator: sum(attempts[(split, operator)] for split in requested)
        for operator in OPERATOR_SPECS
    }
    operator_valid = {
        operator: sum(applicable[(split, operator)] for split in requested)
        for operator in OPERATOR_SPECS
    }
    audit = {
        "experiment": "fusion_guided_full_source_cohort_v1",
        "llm_calls_made": 0,
        "seed": args.seed,
        "selected_unique_source_functions": len(source_ids),
        "selected_sources_by_split": requested,
        "operator_count": len(OPERATOR_SPECS),
        "attempted_source_operator_combinations": len(source_ids) * len(OPERATOR_SPECS),
        "valid_mutation_records": len(records),
        "inapplicable_combinations": sum(operator_attempts.values()) - len(records) - len(failures),
        "generation_failures": len(failures),
        "failure_details": failures,
        "valid_mutations_by_split": dict(Counter(row["source_split"] for row in records)),
        "attempts_by_operator": operator_attempts,
        "valid_mutations_by_operator": operator_valid,
        "inapplicable_by_operator": {operator: operator_attempts[operator] - operator_valid[operator] for operator in OPERATOR_SPECS},
        "sources_with_at_least_one_mutation": len(per_source),
        "sources_with_no_applicable_mutation": len(source_ids) - len(per_source),
        "mutations_per_source_distribution": dict(sorted(Counter(per_source.values()).items())),
        "cross_split_source_leakage": 0,
        "all_old_functions_parse": True,
        "all_mutated_functions_parse": True,
        "all_mutations_ast_visible": True,
        "all_requirements_embed_exact_fragments": True,
    }
    audit_path = args.output.with_name(f"{args.output.stem}_audit.json")
    save_json(audit_path, audit)
    print(json.dumps(audit, indent=2))
    print(f"Output: {args.output}")
    print(f"Audit: {audit_path}")


if __name__ == "__main__":
    main()
