#!/usr/bin/env python3
"""Audit the fixed small-context LEACE anchor set."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path


_GENDER_SPAN = re.compile(r"\b(?:female|male|woman|man)\b", re.IGNORECASE)


def audit(records):
    errors = []
    groups = defaultdict(list)
    context_axes = {}
    context_templates = defaultdict(set)
    descriptions = set()
    for row in records:
        groups[(row["context_id"], row["template_id"])].append(row)
        axes = tuple(row["axes"])
        previous_axes = context_axes.setdefault(row["context_id"], axes)
        if previous_axes != axes:
            errors.append(f"inconsistent axes for context {row['context_id']}")
        context_templates[row["context_id"]].add(row["template_id"])
        if row["description"] in descriptions:
            errors.append(f"duplicate description: {row['description']}")
        descriptions.add(row["description"])
        if "status" in row["axes"]:
            errors.append(f"status leaked into training anchor {row['id']}")
    for key, pair in groups.items():
        if Counter(row["gender_label"] for row in pair) != Counter({"female": 1, "male": 1}):
            errors.append(f"unmatched gender pair: {key}")
            continue
        normalized = {_GENDER_SPAN.sub("<GENDER>", row["description"]) for row in pair}
        if len(normalized) != 1:
            errors.append(f"pair differs beyond gender span: {key}")
    if len(records) != 540:
        errors.append(f"expected 540 records, found {len(records)}")
    for context_id, template_ids in context_templates.items():
        if template_ids != set(range(6)):
            errors.append(f"unexpected templates for context {context_id}: {sorted(template_ids)}")
    axes = Counter("+".join(value) or "neutral" for value in context_axes.values())
    expected_axes = Counter({"neutral": 1, "career": 4, "persona": 8, "career+persona": 32})
    if axes != expected_axes:
        errors.append(f"unexpected context topology: {dict(axes)}")
    return {
        "passed": not errors,
        "records": len(records),
        "contexts": len({row["context_id"] for row in records}),
        "pairs": len(groups),
        "status_records": sum("status" in row["axes"] for row in records),
        "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads(args.anchors.read_text(encoding="utf-8"))
    report = audit(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        for error in report["errors"]:
            print(f"[ERROR] {error}")
        return 1
    print(f"Anchor audit passed for {report['records']} records")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
