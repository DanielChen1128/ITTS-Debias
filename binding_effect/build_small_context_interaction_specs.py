#!/usr/bin/env python3
"""Materialize complete Stage-1/2 condition CSVs and interaction specs."""

import argparse
import csv
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path


DEFAULT_METHODS = ("original",)
AXES = ("status", "career", "persona")


def _read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _parts(descriptor_id):
    return tuple(descriptor_id.split("|"))


def _key(parts):
    return "|".join(sorted(parts, key=lambda part: AXES.index(part.split(":", 1)[0])))


def build_method(
    stage1_root,
    stage2_root,
    output_root,
    method,
    matched_generation=False,
    stage1_method_root=None,
    stage2_method_root=None,
):
    stage1_root = Path(stage1_method_root) if stage1_method_root else Path(stage1_root) / method
    stage2_root = Path(stage2_method_root) if stage2_method_root else Path(stage2_root) / method
    output_root = Path(output_root) / method
    condition_dir = output_root / "conditions"
    condition_rows = {}
    for axis in AXES:
        for row in _read_rows(stage1_root / axis / "detection_results.csv"):
            condition_rows.setdefault(row["descriptor_id"], []).append(row)
    two_rows = _read_rows(stage2_root / "two-axis" / "detection_results.csv")
    three_rows = _read_rows(stage2_root / "three-axis" / "detection_results.csv")
    for row in two_rows:
        condition_rows.setdefault(_key(_parts(row["descriptor_id"])), []).append(row)
    for row in three_rows:
        condition_rows.setdefault(_key(_parts(row["descriptor_id"])), []).append(row)

    paths = {}
    for condition, rows in condition_rows.items():
        slug = condition.replace(":", "_").replace("|", "__")
        path = condition_dir / f"{slug}.csv"
        _write_rows(path, rows)
        paths[condition] = path.relative_to(output_root).as_posix()

    two_cells = sorted({_key(_parts(row["descriptor_id"])) for row in two_rows})
    three_cells = sorted({_key(_parts(row["descriptor_id"])) for row in three_rows})
    interactions = []
    for cell in two_cells:
        parts = _parts(cell)
        interactions.append({
            "name": f"two:{cell}",
            "order": 2,
            "conditions": [
                {"name": "joint", "csv": paths[cell]},
                {"name": parts[0], "csv": paths[parts[0]]},
                {"name": parts[1], "csv": paths[parts[1]]},
            ],
        })
    for cell in three_cells:
        parts = _parts(cell)
        pairs = [_key(pair) for pair in combinations(parts, 2)]
        interactions.append({
            "name": f"three:{cell}",
            "order": 3,
            "conditions": [
                {"name": "triple", "csv": paths[cell]},
                *[{"name": pair, "csv": paths[pair]} for pair in pairs],
                *[{"name": part, "csv": paths[part]} for part in parts],
            ],
        })
    spec = {
        "protocol_id": "small-context-leace-v2",
        "method": method,
        "provisional": not matched_generation,
        "interaction_count": len(interactions),
        "interactions": interactions,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "interaction_spec.json").write_text(
        json.dumps(spec, indent=2) + "\n", encoding="utf-8"
    )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--matched-generation",
        action="store_true",
        help="Mark all methods non-provisional when Stage-1 and Stage-2 generation settings match",
    )
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
    parser.add_argument(
        "--stage2-method-root",
        action="append",
        default=[],
        metavar="METHOD=PATH",
        help="Override a method's Stage-2 detection directory",
    )
    args = parser.parse_args()
    overrides = dict(item.split("=", 1) for item in args.stage1_method_root)
    stage2_overrides = dict(item.split("=", 1) for item in args.stage2_method_root)
    methods = args.methods or DEFAULT_METHODS
    unknown = (set(overrides) | set(stage2_overrides)) - set(methods)
    if unknown:
        parser.error(f"Stage-1 root override has no selected method: {', '.join(sorted(unknown))}")
    for method in methods:
        spec = build_method(
            args.stage1_root,
            args.stage2_root,
            args.output_root,
            method,
            matched_generation=args.matched_generation,
            stage1_method_root=overrides.get(method),
            stage2_method_root=stage2_overrides.get(method),
        )
        print(f"{method}: wrote {spec['interaction_count']} interactions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
