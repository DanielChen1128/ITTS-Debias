#!/usr/bin/env python3
"""Summarize completed unified constant-steering model conditions."""

import argparse
import csv
import json
from pathlib import Path


STRATA = (
    ("status", "stage1/status"),
    ("career", "stage1/career"),
    ("persona", "stage1/persona"),
    ("two-axis", "stage2/two-axis"),
    ("three-axis", "stage2/three-axis"),
)
CONDITIONS = (
    ("parler-mini", "constant-steering-2x"),
    ("parler-large", "constant-steering-2x"),
    ("voxinstruct", "constant-steering-2x"),
    ("voxinstruct", "ar6-nar2"),
)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def gender_summary(root, method):
    by_stratum = {}
    all_rows = []
    for name, relative in STRATA:
        rows = read_csv(root / method / relative / "detection_results.csv")
        binary = [row for row in rows if row["predicted_gender"] in ("female", "male")]
        female = sum(row["predicted_gender"] == "female" for row in binary)
        by_stratum[name] = {
            "binary_n": len(binary),
            "female_rate": female / len(binary) if binary else None,
        }
        all_rows.extend(binary)
    female = sum(row["predicted_gender"] == "female" for row in all_rows)
    return {
        "binary_n": len(all_rows),
        "female_rate": female / len(all_rows) if all_rows else None,
        "strata": by_stratum,
    }


def interaction_summary(path):
    rows = read_csv(path)
    values = [abs(float(row["interaction"])) for row in rows]
    counts = {
        label: sum(row["significance"] == label for row in rows)
        for label in ("strong", "moderate", "not_significant")
    }
    return {"cells": len(rows), "mean_absolute": sum(values) / len(values), **counts}


def summarize(root, model, condition):
    analysis = root / "runs/analysis" / model
    comparison = analysis / "comparison" / condition
    methods = {}
    for method in ("original", condition):
        methods[method] = {
            "gender": gender_summary(analysis, method),
            "interactions": interaction_summary(
                comparison / "interactions" / method / "interactions_10000.csv"
            ),
        }
    quality = json.loads(
        (comparison / "quality/quality_v1.json").read_text(encoding="utf-8")
    )
    return {"model": model, "condition": condition, "methods": methods, "quality": {
        "n": quality["n"],
        "means": quality["means"],
        "paired_deltas": quality["paired_deltas"],
        "noninferiority_passed": quality["noninferiority_passed"],
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "protocol_id": "unified-model-local-constant-steering-v1",
        "conditions": [summarize(args.root, *item) for item in CONDITIONS],
        "prompt_count_per_condition": 13300,
        "interaction_cells_per_condition": 64,
        "quality_pairs_per_condition": 500,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(report['conditions'])} condition summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
