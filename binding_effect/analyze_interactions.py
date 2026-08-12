#!/usr/bin/env python3
"""Compute paper-aligned interaction statistics from detection CSV files."""

import argparse
import csv
import json
import sys
from pathlib import Path

from binding_stats import paper_labels, permutation_test, validate_interaction_spec


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "predicted_gender" not in rows[0]:
        raise ValueError(f"{path}: missing predicted_gender column")
    return paper_labels(row["predicted_gender"] for row in rows)


def main():
    parser = argparse.ArgumentParser(description="Compute equations 5/6 with 10,000 constrained-null randomizations")
    parser.add_argument("--spec", required=True, help="Interaction JSON specification")
    parser.add_argument("--output", required=True, help="Output CSV")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    errors = validate_interaction_spec(spec)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 2

    output_rows = []
    spec_dir = Path(args.spec).resolve().parent
    for item in spec["interactions"]:
        groups, adult_counts, child_counts, exclusions = [], [], [], []
        for condition in item["conditions"]:
            condition_path = Path(condition["csv"])
            if not condition_path.is_absolute():
                condition_path = spec_dir / condition_path
            try:
                labels, adult, counts, excluded = load_labels(condition_path)
            except (OSError, ValueError) as exc:
                print(f"[ERROR] incomplete condition {condition['name']}: {exc}", file=sys.stderr)
                return 2
            groups.append(labels)
            adult_counts.append(len(adult))
            child_counts.append(counts["child"])
            exclusions.append(sum(excluded.values()))
        coefficients = [1, -1, -1] if item["order"] == 2 else [1, -1, -1, -1, 1, 1, 1]
        value, p_value, significance = permutation_test(
            groups, coefficients, args.iterations, args.seed
        )
        output_rows.append({
            "name": item["name"], "order": item["order"], "interaction": value,
            "p_value": p_value, "significance": significance,
            "classified_n": sum(map(len, groups)), "adult_n": sum(adult_counts),
            "child_n": sum(child_counts), "unknown_or_other_n": sum(exclusions),
        })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_rows[0].keys())
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
