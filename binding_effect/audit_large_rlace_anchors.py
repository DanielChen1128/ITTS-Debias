#!/usr/bin/env python3
"""Audit balance, pairing, coverage, and holdout isolation for Large anchors."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_GENDER_SPAN = re.compile(r"\b(?:female|male|woman|man)(?:'s)?(?:-sounding)?\b", re.IGNORECASE)


def audit(records, held_out_sets=()):
    errors = []
    groups = defaultdict(list)
    descriptions = set()
    for row in records:
        groups[(row["context_id"], row["template_id"])].append(row)
        if row["description"] in descriptions:
            errors.append(f"duplicate description: {row['id']}")
        descriptions.add(row["description"])
        if "status" in row["axes"]:
            errors.append(f"status leaked into training: {row['id']}")
    for key, pair in groups.items():
        if Counter(row["gender_label"] for row in pair) != Counter({"female": 1, "male": 1}):
            errors.append(f"unmatched gender pair: {key}")
        elif len({_GENDER_SPAN.sub("<GENDER>", row["description"]) for row in pair}) != 1:
            errors.append(f"pair differs beyond gender span: {key}")
    contexts = {row["context_id"] for row in records}
    templates = {row["template_id"] for row in records}
    if len(records) != 5040 or len(contexts) != 180 or templates != set(range(14)):
        errors.append("expected 5040 records, 180 contexts, and templates 0..13")
    descriptor_ids = {item for row in records for item in row["descriptor_ids"]}
    if len(descriptor_ids) != 67:
        errors.append(f"expected all 67 career/persona descriptors, found {len(descriptor_ids)}")
    for name, held_out in held_out_sets:
        overlap = descriptions & {row["description"] for row in held_out}
        if overlap:
            errors.append(f"{name} description leakage: {len(overlap)} rows")
    return {
        "passed": not errors,
        "records": len(records),
        "contexts": len(contexts),
        "templates": len(templates),
        "pairs": len(groups),
        "gender_counts": dict(Counter(row["gender_label"] for row in records)),
        "descriptor_coverage": len(descriptor_ids),
        "status_records": sum("status" in row["axes"] for row in records),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--held-out", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.anchors.read_text(encoding="utf-8"))
    held_out = [(path.stem, json.loads(path.read_text(encoding="utf-8"))) for path in args.held_out]
    report = audit(records, held_out)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        for error in report["errors"]:
            print(f"[ERROR] {error}")
        return 1
    print(f"Large anchor audit passed for {report['records']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
