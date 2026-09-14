#!/usr/bin/env python3
"""Create a reproducible, operator-aware LLM batch and its exact complement."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


SPLITS = ("train", "validation", "test")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train", type=int, default=3500)
    parser.add_argument("--validation", type=int, default=500)
    parser.add_argument("--test", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_key(row: dict, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"fusion-guided-llm-batch-v1\0{seed}\0{row['id']}".encode()).hexdigest()
    return digest, row["id"]


def balanced_quotas(available: dict[str, int], target: int) -> dict[str, int]:
    """Water-fill a target as evenly as availability permits."""
    if target < 0 or target > sum(available.values()):
        raise ValueError(f"Cannot select {target} records from {sum(available.values())} available")
    quotas = {operator: 0 for operator in sorted(available)}
    remaining = target
    active = {operator for operator, count in available.items() if count > 0}
    while remaining and active:
        share, extra = divmod(remaining, len(active))
        progressed = False
        for index, operator in enumerate(sorted(active)):
            request = share + (1 if index < extra else 0)
            capacity = available[operator] - quotas[operator]
            take = min(request, capacity)
            quotas[operator] += take
            remaining -= take
            progressed = progressed or take > 0
        active = {operator for operator in active if quotas[operator] < available[operator]}
        if not progressed:
            break
    if remaining:
        raise RuntimeError(f"Quota allocation left {remaining} records unassigned")
    return quotas


def distribution(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts = Counter((row["source_split"], row["mutation_operator"]) for row in rows)
    return {
        split: {
            operator: counts[(split, operator)]
            for operator in sorted({row["mutation_operator"] for row in rows})
            if counts[(split, operator)]
        }
        for split in SPLITS
    }


def id_digest(rows: list[dict]) -> str:
    material = "\n".join(sorted(row["id"] for row in rows)).encode()
    return hashlib.sha256(material).hexdigest()


def main() -> None:
    args = arguments()
    rows = json.loads(args.input.read_text(encoding="utf-8"))
    if len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("Input contains duplicate record IDs")
    targets = {"train": args.train, "validation": args.validation, "test": args.test}
    by_split_operator: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        split = row.get("source_split")
        if split not in SPLITS:
            raise RuntimeError(f"Unexpected source split: {split!r}")
        by_split_operator[split][row["mutation_operator"]].append(row)

    selected_ids: set[str] = set()
    quotas_by_split: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        groups = by_split_operator[split]
        available = {operator: len(group) for operator, group in groups.items()}
        quotas = balanced_quotas(available, targets[split])
        quotas_by_split[split] = quotas
        for operator, quota in quotas.items():
            ordered = sorted(groups[operator], key=lambda row: stable_key(row, args.seed))
            selected_ids.update(row["id"] for row in ordered[:quota])

    batch = [row for row in rows if row["id"] in selected_ids]
    remainder = [row for row in rows if row["id"] not in selected_ids]
    order = {name: index for index, name in enumerate(SPLITS)}
    batch.sort(key=lambda row: (order[row["source_split"]], stable_key(row, args.seed)))
    remainder.sort(key=lambda row: (order[row["source_split"]], stable_key(row, args.seed)))

    if len(batch) != sum(targets.values()):
        raise RuntimeError("Selected batch does not match requested total")
    if set(row["id"] for row in batch) & set(row["id"] for row in remainder):
        raise RuntimeError("Batch and remainder overlap")
    if {row["id"] for row in batch + remainder} != {row["id"] for row in rows}:
        raise RuntimeError("Batch and remainder do not reconstruct the input")
    batch_split_counts = Counter(row["source_split"] for row in batch)
    if dict(batch_split_counts) != targets:
        raise RuntimeError(f"Split targets were not preserved: {dict(batch_split_counts)}")

    save(args.output_dir / "mutations_5000.json", batch)
    save(args.output_dir / "mutations_remaining.json", remainder)
    for split in SPLITS:
        save(args.output_dir / f"mutations_5000_{split}.json", [row for row in batch if row["source_split"] == split])
        save(args.output_dir / f"mutations_remaining_{split}.json", [row for row in remainder if row["source_split"] == split])

    manifest = {
        "selection_policy": "operator-aware water-filled quotas within immutable source splits",
        "seed": args.seed,
        "input_file": str(args.input),
        "input_records": len(rows),
        "requested_batch_by_split": targets,
        "selected_records": len(batch),
        "remaining_records": len(remainder),
        "selected_by_split": dict(batch_split_counts),
        "selected_operator_distribution": distribution(batch),
        "remaining_operator_distribution": distribution(remainder),
        "allocated_quotas_by_split": quotas_by_split,
        "selected_id_sha256": id_digest(batch),
        "remaining_id_sha256": id_digest(remainder),
        "integrity": {
            "unique_input_ids": True,
            "batch_remainder_disjoint": True,
            "batch_plus_remainder_equals_input": True,
            "source_splits_unchanged": True,
            "llm_calls_made": 0,
        },
    }
    save(args.output_dir / "selection_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
