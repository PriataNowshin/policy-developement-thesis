#!/usr/bin/env python3
"""Generate 50 controlled mutations per operator while preserving Study 3 splits."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import random
import textwrap
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from controlled_mutation import OPERATOR_SPECS, apply_operator
from verify_study_model import main as verify_study_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
FUSION_DIR = ROOT / "ast/llm-code-update-risk-predictor/output/codebert_ast_fusion_top2"
FIXED_SPLIT_DIR = ROOT / "ast/llm-code-update-risk-predictor/output/ast_gnn_classifier_output"
TAXONOMY = HERE / "output/step3_author_validation/author_validated_taxonomy.csv"
DEFAULT_OUTPUT = HERE / "output/step4_mutation_generation/mutations_500.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-per-operator", type=int, default=35)
    parser.add_argument("--validation-per-operator", type=int, default=7)
    parser.add_argument("--test-per-operator", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def identity(record: dict) -> tuple:
    return (
        record["repo_full_name"].strip(), record["file_path"].strip(),
        record["function_name"].strip(), " ".join(record["requirement"].split()),
        round(float(record["ast_delta_similarity"]), 12),
    )


def load_fixed_sources() -> dict[str, list[dict]]:
    splits = {}
    for split in ("train", "validation", "test"):
        source_rows = load_json(FIXED_SPLIT_DIR / f"{split}_data.json")
        manifest_rows = load_json(FUSION_DIR / f"{split}_manifest.json")
        manifest_by_identity = {identity(row): row for row in manifest_rows}
        if len(manifest_by_identity) != len(manifest_rows):
            raise RuntimeError(f"Duplicate identities in {split} manifest")
        joined = []
        for source in source_rows:
            manifest = manifest_by_identity.get(identity(source))
            if manifest is None:
                raise RuntimeError(f"Could not join a {split} source to the fusion manifest")
            item = dict(source)
            item["source_index"] = manifest["source_index"]
            joined.append(item)
        if len(joined) != len(manifest_rows):
            raise RuntimeError(f"{split} source and manifest sizes differ")
        splits[split] = joined
    return splits


def taxonomy_support() -> Counter:
    with TAXONOMY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    retained = [row for row in rows if row["author_decision"] in {"accept", "correct"}]
    counts = Counter(row["author_final_category"] for row in retained)
    missing = sorted({spec.taxonomy_category for spec in OPERATOR_SPECS.values()} - set(counts))
    if missing:
        raise RuntimeError(f"Operators lack retained taxonomy support: {missing}")
    return counts


def ordered_sources(rows: list[dict], operator_id: str, split: str, seed: int) -> list[dict]:
    ordered = list(rows)
    material = f"fusion-guided-dataset-v1\0{seed}\0{split}\0{operator_id}".encode()
    local_seed = int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
    random.Random(local_seed).shuffle(ordered)
    return ordered


def generate_for(
    rows: list[dict], operator_id: str, split: str, quota: int,
    seed: int, support_count: int,
) -> tuple[list[dict], dict]:
    generated = []
    inspected = 0
    inapplicable = 0
    failures = []
    for source in ordered_sources(rows, operator_id, split, seed):
        if len(generated) >= quota:
            break
        inspected += 1
        original_old_code = textwrap.dedent(source["old_function_code"]).strip("\n") + "\n"
        try:
            parsed_old = ast.parse(original_old_code)
            old_code = ast.unparse(parsed_old).strip() + "\n"
            mutation = apply_operator(old_code, operator_id)
        except (SyntaxError, TypeError, ValueError) as error:
            failures.append({"source_index": source["source_index"], "reason": str(error)})
            continue
        if mutation is None:
            inapplicable += 1
            continue
        example_id = f"fgm_v1_{split}_{operator_id}_{source['source_index']}"
        generated.append({
            "id": example_id,
            "source_index": source["source_index"],
            "source_split": split,
            "repo_full_name": source["repo_full_name"],
            "file_path": source["file_path"],
            "function_name": source["function_name"],
            "original_old_function_code": original_old_code,
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
        })
    if len(generated) != quota:
        raise RuntimeError(
            f"{operator_id}/{split}: generated {len(generated)} of required {quota}; "
            f"inspected {inspected}, inapplicable {inapplicable}, failures {len(failures)}"
        )
    return generated, {
        "quota": quota,
        "generated": len(generated),
        "source_records_inspected": inspected,
        "inapplicable": inapplicable,
        "failures": failures,
    }


def main() -> None:
    args = parse_args()
    if min(args.train_per_operator, args.validation_per_operator, args.test_per_operator) < 0:
        raise ValueError("Per-operator quotas cannot be negative")
    verify_study_model()
    sources = load_fixed_sources()
    support = taxonomy_support()
    quotas = {
        "train": args.train_per_operator,
        "validation": args.validation_per_operator,
        "test": args.test_per_operator,
    }
    records = []
    generation_audit = defaultdict(dict)
    for operator_id, spec in OPERATOR_SPECS.items():
        for split, quota in quotas.items():
            generated, audit = generate_for(
                sources[split], operator_id, split, quota, args.seed,
                support[spec.taxonomy_category],
            )
            records.extend(generated)
            generation_audit[operator_id][split] = audit

    expected_total = len(OPERATOR_SPECS) * sum(quotas.values())
    if len(records) != expected_total:
        raise RuntimeError("Generated total does not match the requested balanced design")
    if len({record["id"] for record in records}) != len(records):
        raise RuntimeError("Duplicate example IDs detected")
    malformed_requirements = [
        record["id"] for record in records
        if not _requirement_is_faithful(record)
    ]
    if malformed_requirements:
        raise RuntimeError(
            "Requirements do not preserve unambiguous full before/after fragments: "
            f"{malformed_requirements[:5]}"
        )
    pair_count = Counter((record["source_index"], record["mutation_operator"]) for record in records)
    if any(count != 1 for count in pair_count.values()):
        raise RuntimeError("Duplicate source/operator pairs detected")
    split_by_source = defaultdict(set)
    for record in records:
        split_by_source[record["source_index"]].add(record["source_split"])
    leaking = {key: sorted(value) for key, value in split_by_source.items() if len(value) > 1}
    if leaking:
        raise RuntimeError(f"Source functions cross synthetic splits: {list(leaking.items())[:5]}")

    records.sort(key=lambda row: (row["source_split"], row["mutation_operator"], row["source_index"]))
    save_json(args.output, records)
    for split in quotas:
        save_json(
            args.output.with_name(f"{args.output.stem}_{split}.json"),
            [record for record in records if record["source_split"] == split],
        )
    specs_path = args.output.with_name("operator_specifications.json")
    save_json(specs_path, [asdict(spec) for spec in OPERATOR_SPECS.values()])
    audit = {
        "experiment": "fusion_guided_controlled_mutation_v1",
        "llm_calls_made": 0,
        "seed": args.seed,
        "operator_count": len(OPERATOR_SPECS),
        "per_operator_quotas": quotas,
        "expected_total": expected_total,
        "generated_total": len(records),
        "counts_by_split": dict(Counter(record["source_split"] for record in records)),
        "counts_by_operator": dict(Counter(record["mutation_operator"] for record in records)),
        "unique_source_functions": len(split_by_source),
        "duplicate_ids": 0,
        "duplicate_source_operator_pairs": 0,
        "cross_split_source_leakage": 0,
        "all_old_functions_parse": all(_parses(record["old_function_code"]) for record in records),
        "all_mutated_functions_parse": all(_parses(record["expected_mutated_function"]) for record in records),
        "all_mutations_ast_visible": all(_ast_changed(record) for record in records),
        "all_requirements_embed_exact_fragments": True,
        "taxonomy_support_counts": dict(support),
        "generation_details": generation_audit,
    }
    audit_path = args.output.with_name(f"{args.output.stem}_audit.json")
    save_json(audit_path, audit)
    print(f"Wrote {len(records)} mutation-only examples to {args.output}")
    print(f"Splits: {audit['counts_by_split']}")
    print(f"Audit: {audit_path}")


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _ast_changed(record: dict) -> bool:
    return ast.dump(ast.parse(record["old_function_code"]), include_attributes=False) != ast.dump(
        ast.parse(record["expected_mutated_function"]), include_attributes=False
    )


def _requirement_is_faithful(record: dict) -> bool:
    requirement = record["requirement"]
    old_fragment = record["old_fragment"]
    new_fragment = record["new_fragment"]
    delimiters = ("<before>", "</before>", "<after>", "</after>")
    if any(delimiter in old_fragment or delimiter in new_fragment for delimiter in delimiters):
        return False
    if any(requirement.count(delimiter) != 1 for delimiter in delimiters):
        return False
    return (
        f"<before>{old_fragment}</before>" in requirement
        and f"<after>{new_fragment}</after>" in requirement
    )


if __name__ == "__main__":
    main()
