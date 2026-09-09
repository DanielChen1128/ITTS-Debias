#!/usr/bin/env python3
"""Build a deterministic 500-prompt stratified extension screen."""

import argparse
import hashlib
import json
from pathlib import Path


STAGE1_SOURCES = {
    "status": Path("descriptions/descriptions_status_bias.json"),
    "career": Path("descriptions/description_career_bias.json"),
    "persona": Path("descriptions/descriptions_persona_bias.json"),
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_key(seed, row):
    identifier = row.get("id") or row.get("prompt_id")
    if identifier is None:
        raise ValueError("every prompt must have id or prompt_id")
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("parler-mini", "parler-large", "voxinstruct"), required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/extension-screen-v1"))
    parser.add_argument("--per-stratum", type=int, default=100)
    parser.add_argument("--seed", default="b200-extension-screen-v1")
    args = parser.parse_args()

    output_dir = args.output_root / args.model
    output_dir.mkdir(parents=True, exist_ok=True)
    stage2_model = "parler-mini" if args.model == "parler-mini" else "parler-large"
    sources = {
        **STAGE1_SOURCES,
        "two-axis": Path(f"data/{stage2_model}/stage2/descriptions_two_axis.json"),
        "three-axis": Path(f"data/{stage2_model}/stage2/descriptions_multi_axis.json"),
    }
    manifest = {
        "protocol_id": "b200-extension-screen-v1",
        "model": args.model,
        "stage2_prompt_protocol": stage2_model,
        "seed": args.seed,
        "per_stratum": args.per_stratum,
        "strata": {},
    }
    for stratum, source in sources.items():
        rows = json.loads(source.read_text(encoding="utf-8"))
        if len(rows) < args.per_stratum:
            parser.error(f"{source} has fewer than {args.per_stratum} prompts")
        selected = sorted(rows, key=lambda row: row_key(args.seed, row))[:args.per_stratum]
        output = output_dir / f"{stratum}.json"
        output.write_text(json.dumps(selected, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
        manifest["strata"][stratum] = {
            "source": str(source),
            "source_sha256": digest(source),
            "output": str(output),
            "output_sha256": digest(output),
            "prompts": len(selected),
        }
    manifest["total_prompts"] = sum(item["prompts"] for item in manifest["strata"].values())
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
