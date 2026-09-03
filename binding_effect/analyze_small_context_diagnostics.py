#!/usr/bin/env python3
"""Summarize Stage-1 generalization and Stage-2 interaction families."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from debias.binding_protocol import TRAIN_CAREER_IDS, TRAIN_PERSONA_IDS


DEFAULT_METHODS = ("original",)
AXES = ("status", "career", "persona")
MECHANIC_NEIGHBORS = {"career:mechanician"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def descriptor_group(axis, descriptor_id):
    if axis == "career":
        if descriptor_id in TRAIN_CAREER_IDS:
            return "training_seen"
        if descriptor_id in MECHANIC_NEIGHBORS:
            return "mechanic_lexical_neighbor"
        return "unseen_non_neighbor"
    if axis == "persona":
        return "training_seen" if descriptor_id in TRAIN_PERSONA_IDS else "unseen"
    return "status_excluded_from_training"


def summarize_detection_rows(rows):
    if any(row["predicted_gender"] == "child" for row in rows):
        raise ValueError("child labels must be recomputed as female or male")
    classified = [row for row in rows if row["predicted_gender"] in {"female", "male"}]
    by_descriptor = defaultdict(list)
    for row in classified:
        by_descriptor[row["descriptor_id"]].append(row)
    probabilities = [
        sum(row["predicted_gender"] == "female" for row in descriptor_rows) / len(descriptor_rows)
        for descriptor_rows in by_descriptor.values()
    ]
    counts = {
        label: sum(row["predicted_gender"] == label for row in classified)
        for label in ("female", "male")
    }
    return {
        "descriptor_count": len(by_descriptor),
        "sample_count": len(rows),
        "classified_count": len(classified),
        "female_count": counts["female"],
        "male_count": counts["male"],
        "female_probability": counts["female"] / len(classified),
        "mean_female_score": sum(
            float(row["female_score"])
            / (float(row["female_score"]) + float(row["male_score"]))
            for row in rows
        ) / len(rows),
        "descriptor_range": max(probabilities) - min(probabilities),
        "descriptor_sd": math.sqrt(
            sum((value - sum(probabilities) / len(probabilities)) ** 2 for value in probabilities)
            / len(probabilities)
        ),
    }


def stage1_diagnostics(stage1_root, methods=DEFAULT_METHODS, method_roots=None):
    method_roots = method_roots or {}
    summaries = []
    descriptor_probabilities = {}
    for method in methods:
        method_root = method_roots.get(method, stage1_root / method)
        for axis in AXES:
            rows = read_csv(method_root / axis / "detection_results.csv")
            grouped = defaultdict(list)
            grouped["all"].extend(rows)
            for row in rows:
                grouped[descriptor_group(axis, row["descriptor_id"])].append(row)
            for group, group_rows in grouped.items():
                summaries.append({
                    "method": method,
                    "axis": axis,
                    "group": group,
                    **summarize_detection_rows(group_rows),
                })
            for descriptor_id in {row["descriptor_id"] for row in rows}:
                descriptor_rows = [row for row in rows if row["descriptor_id"] == descriptor_id]
                classified = [
                    row for row in descriptor_rows
                    if row["predicted_gender"] in {"female", "male"}
                ]
                descriptor_probabilities[(method, descriptor_id)] = (
                    sum(row["predicted_gender"] == "female" for row in classified) / len(classified)
                )

    contrasts = []
    for summary in summaries:
        method = summary["method"]
        if method == "original":
            continue
        axis = summary["axis"]
        group = summary["group"]
        descriptor_ids = {
            descriptor_id
            for candidate_method, descriptor_id in descriptor_probabilities
            if candidate_method == method
            and descriptor_id.split(":", 1)[0] == axis
            and (group == "all" or descriptor_group(axis, descriptor_id) == group)
        }
        deltas = [
            descriptor_probabilities[(method, descriptor_id)]
            - descriptor_probabilities[("original", descriptor_id)]
            for descriptor_id in descriptor_ids
        ]
        original = next(
            row for row in summaries
            if row["method"] == "original" and row["axis"] == axis and row["group"] == group
        )
        contrasts.append({
            "method": method,
            "axis": axis,
            "group": group,
            "descriptor_count": len(deltas),
            "mean_descriptor_delta_vs_original": sum(deltas) / len(deltas),
            "mean_absolute_descriptor_delta_vs_original": sum(abs(value) for value in deltas) / len(deltas),
            "descriptor_range_delta_vs_original": summary["descriptor_range"] - original["descriptor_range"],
            "descriptor_sd_delta_vs_original": summary["descriptor_sd"] - original["descriptor_sd"],
        })
    return summaries, contrasts


def interaction_family(name, order):
    parts = name.split(":", 1)[1].split("|")
    if int(order) == 3:
        return "status+career+persona"
    return "+".join(part.split(":", 1)[0] for part in parts)


def stage2_diagnostics(interaction_root, methods=DEFAULT_METHODS):
    summaries = []
    for method in methods:
        rows = read_csv(interaction_root / method / "interactions_10000.csv")
        grouped = defaultdict(list)
        for row in rows:
            grouped[interaction_family(row["name"], row["order"])].append(row)
        for family, family_rows in sorted(grouped.items()):
            magnitudes = [abs(float(row["interaction"])) for row in family_rows]
            summaries.append({
                "method": method,
                "family": family,
                "interaction_count": len(family_rows),
                "mean_absolute_interaction": sum(magnitudes) / len(magnitudes),
                "max_absolute_interaction": max(magnitudes),
                "moderate_count": sum(row["significance"] == "moderate" for row in family_rows),
                "strong_count": sum(row["significance"] == "strong" for row in family_rows),
            })
    return summaries


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--interaction-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="parler-mini")
    parser.add_argument(
        "--method",
        action="append",
        dest="methods",
        help="Method to process; repeat for multiple methods",
    )
    parser.add_argument(
        "--stage1-method-root",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method's Stage-1 detection directory",
    )
    parser.add_argument("--matched-generation", action="store_true")
    args = parser.parse_args()

    methods = args.methods or DEFAULT_METHODS
    method_roots = {
        method: Path(path)
        for method, path in (item.split("=", 1) for item in args.stage1_method_root)
    }
    unknown = set(method_roots) - set(methods)
    if unknown:
        parser.error(f"Stage-1 root override has no selected method: {', '.join(sorted(unknown))}")
    stage1, contrasts = stage1_diagnostics(args.stage1_root, methods, method_roots)
    stage2 = stage2_diagnostics(args.interaction_root, methods)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "stage1_groups.csv", stage1)
    write_csv(args.output_dir / "stage1_group_contrasts.csv", contrasts)
    write_csv(args.output_dir / "stage2_interaction_families.csv", stage2)
    notes = [
        "Status was excluded from small-context LEACE training.",
        "career:mechanician is reported separately as a lexical neighbor of training career:mechanic.",
    ]
    if args.model == "parler-mini" and not args.matched_generation:
        notes.extend([
            "Stage-1 Original and explicit-only results use sequential generation; small-context uses batch size 8.",
            "All Stage-2 methods use batch size 8 and the same Original-selected cells.",
        ])
    else:
        notes.extend([
            "All Stage-1 and Stage-2 methods use batch size 8.",
            "All Stage-2 methods use the same Original-selected cells.",
        ])
    report = {
        "protocol_id": "small-context-leace-v2",
        "model": args.model,
        "stage1_groups": stage1,
        "stage1_group_contrasts": contrasts,
        "stage2_interaction_families": stage2,
        "notes": notes,
    }
    (args.output_dir / "diagnostics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
