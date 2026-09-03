#!/usr/bin/env python3
"""Rebuild a lightweight selected-ID manifest from a retained quality report."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    baseline = report["baseline"]
    candidate = report["candidate"]
    baseline_rows = report["per_item"][baseline]
    candidate_rows = report["per_item"][candidate]
    baseline_ids = [row["id"] for row in baseline_rows]
    candidate_ids = [row["id"] for row in candidate_rows]
    if baseline_ids != candidate_ids:
        parser.error("quality report is not paired by ordered prompt ID")

    prompt_rows = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompt_ids = {row["id"] for row in prompt_rows}
    missing = set(baseline_ids) - prompt_ids
    if missing:
        parser.error(f"{len(missing)} selected IDs are absent from prompts")

    manifest = {
        "schema_version": 1,
        "protocol_id": report["protocol"],
        "historical_manifest_sha256": report["prompt_manifest_sha256"],
        "quality_report": str(args.report),
        "prompt_path": str(args.prompts),
        "prompt_sha256": hashlib.sha256(args.prompts.read_bytes()).hexdigest(),
        "baseline": baseline,
        "candidate": candidate,
        "paired_records": len(baseline_ids),
        "selected_ids": baseline_ids,
        "generated_wavs_retained": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(baseline_ids)} paired IDs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
