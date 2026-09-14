#!/usr/bin/env python3
"""Build a reproducible, mutation-oriented taxonomy of correct requirements."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "output/step1_correct_predictions/correct_predictions.json"
OUTPUT_DIR = HERE / "output/step2_requirement_taxonomy"

# Ordered from narrow, easily identifiable edits to broader behavioural changes.
RULES = [
    ("documentation_or_comment", r"\b(docstring|documentation|comment|comments|wording|phrasing|spelling|typo)\b"),
    ("identifier_rename", r"\b(rename|renamed|variable name|underscores? to separate words)\b"),
    ("type_annotation", r"\b((?:add|update|remove|change|include|use).*?(?:type annotation|type annotations|type hint|type hints|typing annotation))\b"),
    ("function_signature_or_parameter", r"\b(function signature|method signature|constructor.*accept|accepts? (?:an?|the|optional|new)|add (?:an? )?(?:optional )?(?:argument|parameter|option)|command[ -]line option)\b"),
    ("exception_handling", r"\b(raise[sd]?|exception|error handling|catch|caught|propagate|try/except|silently ignoring)\b"),
    ("logging_or_user_output", r"\b(log|logging|print|display|console|standard output|standard error|warning message)\b"),
    ("file_or_serialization_io", r"\b(read|write|load|save|export|file|directory|path|json|yaml|csv|stream|serialize|encoding)\b"),
    ("return_value", r"\b(return|returned|returns|returning)\b"),
    ("condition_or_validation", r"\b(condition|only when|otherwise|if |when |unless|validat|invalid|check whether|handle cases?)\b"),
    ("function_call_or_dependency", r"\b(use|call|invoke|helper|method|api|instead of|replace .* with)\b"),
    ("literal_or_configuration_value", r"\b(constant|literal|setting|configuration|config|environment variable|flag|value from|path should|directory.*named)\b"),
    ("statement_addition_or_removal", r"\b(add|insert|include|remove|delete|unused|unnecessary)\b"),
    ("control_flow_or_iteration", r"\b(loop|iteration|for each|continue|break|resume|pause|skip|branch)\b"),
    ("computation_or_data_transformation", r"\b(calculate|compute|convert|transform|concatenat|sort|order|filter|merge|parse|shape|tensor|array|dataframe)\b"),
]

DETAILS = {
    "documentation_or_comment": (True, "Edit a docstring or comment without changing executable AST behaviour."),
    "identifier_rename": (True, "Rename a selected Python identifier consistently."),
    "type_annotation": (True, "Add or modify parameter, return, or assignment annotations."),
    "function_signature_or_parameter": (True, "Add, remove, rename, or default a function parameter."),
    "exception_handling": (True, "Add, remove, or change a raise/try/except construct."),
    "logging_or_user_output": (True, "Add, remove, or modify a logging or print statement."),
    "file_or_serialization_io": (False, "Often depends on project APIs, files, and external state."),
    "return_value": (True, "Add, remove, or modify a return statement or returned expression."),
    "condition_or_validation": (True, "Add or modify a comparison, Boolean expression, or guard."),
    "function_call_or_dependency": (True, "Replace a callable or modify call arguments when locally resolvable."),
    "literal_or_configuration_value": (True, "Replace a string, numeric, Boolean, or None literal."),
    "statement_addition_or_removal": (True, "Insert or delete a locally valid statement."),
    "control_flow_or_iteration": (False, "General loop/control-flow edits need stronger semantic constraints."),
    "computation_or_data_transformation": (False, "Usually project-specific and not safely derivable from syntax alone."),
    "complex_or_project_specific": (False, "No single safe local mutation represents the requirement."),
}


def classify(text: str):
    normalized = " ".join(text.lower().split())
    matches = [name for name, pattern in RULES if re.search(pattern, normalized)]
    primary = matches[0] if matches else "complex_or_project_specific"
    # Long requirements with several signals are explicitly marked complex, except
    # concise refactors whose first category describes the dominant requested edit.
    word_count = len(normalized.split())
    refactor = normalized.startswith("refactor_change:")
    if not refactor and (word_count > 65 or len(matches) >= 5):
        primary = "complex_or_project_specific"
    return primary, matches, word_count


def save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    records = json.loads(INPUT.read_text(encoding="utf-8"))
    classified = []
    counts = Counter()
    by_label = defaultdict(Counter)
    examples = defaultdict(lambda: defaultdict(list))
    for record in records:
        primary, signals, word_count = classify(record["requirement"])
        feasible, reason = DETAILS[primary]
        item = dict(record)
        item.update({
            "taxonomy_category": primary,
            "taxonomy_signals": signals,
            "requirement_word_count": word_count,
            "candidate_for_local_python_mutation": feasible,
            "mutation_feasibility_note": reason,
            "taxonomy_status": "candidate_requires_manual_review"
        })
        classified.append(item)
        counts[primary] += 1
        by_label[primary][record["true_label"]] += 1
        if len(examples[primary][record["true_label"]]) < 3:
            examples[primary][record["true_label"]].append({
                "source_index": record["source_index"],
                "requirement": record["requirement"]
            })

    categories = []
    for category, count in counts.most_common():
        feasible, note = DETAILS[category]
        categories.append({
            "category": category,
            "count": count,
            "label_distribution": {label: by_label[category].get(label, 0) for label in ("low", "mid", "high")},
            "candidate_for_local_python_mutation": feasible,
            "feasibility_note": note,
            "representative_examples": dict(examples[category])
        })
    summary = {
        "input_records": len(records),
        "taxonomy_method": "deterministic ordered semantic rules with explicit complex-category fallback",
        "status": "candidate taxonomy; manual review required before implementing operators",
        "category_count": len(categories),
        "mutation_candidate_records": sum(counts[c] for c, d in DETAILS.items() if d[0]),
        "unsupported_or_complex_records": sum(counts[c] for c, d in DETAILS.items() if not d[0]),
        "categories": categories
    }
    save(OUTPUT_DIR / "classified_requirements.json", classified)
    save(OUTPUT_DIR / "taxonomy_summary.json", summary)
    print(f"Classified {len(records)} requirements into {len(categories)} categories")
    for row in categories:
        print(f"{row['category']}: {row['count']} {row['label_distribution']}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
