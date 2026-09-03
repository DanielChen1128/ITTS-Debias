#!/usr/bin/env python3
"""Freeze a five-stratum paired-quality subset for a full model evaluation."""

import argparse
import json
from pathlib import Path

from evaluate_screen_quality import select_stratified, sha256_file


STRATA = {
    "stage1-status": ("descriptions/descriptions_status_bias.json", "stage1", "status"),
    "stage1-career": ("descriptions/description_career_bias.json", "stage1", "career"),
    "stage1-persona": ("descriptions/descriptions_persona_bias.json", "stage1", "persona"),
    "stage2-two-axis": (None, "stage2", "two-axis"),
    "stage2-three-axis": (None, "stage2", "three-axis"),
}


def build_manifest(root, *, model, candidate, pairs_per_stratum=100, seed=None):
    model_name = {
        "mini": "parler-mini",
        "large": "parler-large",
        "voxinstruct": "voxinstruct",
    }.get(model)
    seed = seed or "parler-constant-steering-full-quality-v1"
    sources = {}
    selections = {}

    if model_name is None:
        raise ValueError(f"unsupported model: {model}")
    original_root = f"results/{model_name}/original"
    candidate_root = f"results/{model_name}/{candidate}"

    method_directories = {"original": {}, candidate: {}}
    for stratum, (prompt_path, stage, axis) in STRATA.items():
        if prompt_path is None:
            filename = "descriptions_two_axis.json" if axis == "two-axis" else "descriptions_multi_axis.json"
            prompt_model = "parler-large" if model == "voxinstruct" else model_name
            prompt_path = f"data/{prompt_model}/stage2/{filename}"
        absolute_prompt_path = root / prompt_path
        rows = json.loads(absolute_prompt_path.read_text(encoding="utf-8"))
        selected = select_stratified({stratum: rows}, pairs_per_stratum, seed=seed)
        sources[stratum] = {
            "prompt_path": prompt_path,
            "sha256": sha256_file(absolute_prompt_path),
            "population": len(rows),
        }
        selections[stratum] = [row["id"] for row in selected]
        method_directories["original"][stratum] = f"{original_root}/{stage}/{axis}"
        method_directories[candidate][stratum] = f"{candidate_root}/{stage}/{axis}"

    total_pairs = pairs_per_stratum * len(STRATA)
    return {
        "schema_version": 1,
        "protocol_id": f"{model_name}-{candidate}-full-paired-quality-v1",
        "description": f"Frozen five-stratum paired Original vs {candidate} UTMOS and Whisper subset.",
        "selection_rule": {
            "algorithm": "Within each stratum, sort by SHA-256(seed + NUL + stratum + NUL + prompt id), then id.",
            "seed": seed,
            "pairs_per_stratum": pairs_per_stratum,
            "total_pairs": total_pairs,
            "total_wavs": total_pairs * 2,
        },
        "sources": sources,
        "method_directories": method_directories,
        "selection": selections,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--model", choices=("mini", "large", "voxinstruct"), required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--pairs-per-stratum", type=int, default=100)
    parser.add_argument("--seed", default="parler-constant-steering-full-quality-v1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs_per_stratum < 1:
        parser.error("--pairs-per-stratum must be at least 1")

    manifest = build_manifest(
        args.root,
        model=args.model,
        candidate=args.candidate,
        pairs_per_stratum=args.pairs_per_stratum,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest['selection_rule']['total_pairs']} pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
