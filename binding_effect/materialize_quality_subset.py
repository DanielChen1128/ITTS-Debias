#!/usr/bin/env python3
"""Materialize the selected prompts from a frozen quality manifest."""

import argparse
import json
from pathlib import Path

from evaluate_screen_quality import load_prompt_spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stratum", action="append", dest="strata")
    parser.add_argument(
        "--preserve-source-batches", type=int, metavar="SIZE",
        help="Include whole source batches containing selected IDs, preserving source order",
    )
    args = parser.parse_args()
    rows = load_prompt_spec(args.manifest, args.root)
    if args.strata:
        rows = [row for row in rows if row["stratum"] in set(args.strata)]
    if args.preserve_source_batches:
        if args.preserve_source_batches < 1:
            parser.error("--preserve-source-batches must be at least 1")
        selected_ids = {row["id"] for row in rows}
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        source_paths = {
            args.root / source["prompt_path"] for source in manifest["sources"].values()
        }
        if len(source_paths) != 1:
            parser.error("--preserve-source-batches requires one shared source prompt file")
        source_rows = json.loads(source_paths.pop().read_text(encoding="utf-8"))
        rows = [
            row
            for start in range(0, len(source_rows), args.preserve_source_batches)
            for batch in [source_rows[start:start + args.preserve_source_batches]]
            if any(item["id"] in selected_ids for item in batch)
            for row in batch
        ]
    rows = [{key: value for key, value in row.items() if key != "stratum"} for row in rows]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
