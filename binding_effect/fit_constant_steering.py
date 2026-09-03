#!/usr/bin/env python3
"""Fit a directed centroid offset from cached pooled conditioning states."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from debias.steering import ConstantSteering


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--activations-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-label", default="female")
    parser.add_argument("--target-label", default="male")
    parser.add_argument("--label-field", default="gender_label")
    args = parser.parse_args()

    rows = json.loads(args.anchors.read_text(encoding="utf-8"))
    anchor_sha = hashlib.sha256(args.anchors.read_bytes()).hexdigest()
    cached = torch.load(args.activations_cache, map_location="cpu", weights_only=True)
    if cached.get("provenance", {}).get("training_sha256") != anchor_sha:
        parser.error("activation cache does not match --anchors")
    features = cached["activations"].float()
    if features.ndim != 2 or len(rows) != len(features):
        parser.error("activation cache shape does not match --anchors")
    labels = [row[args.label_field] for row in rows]
    source = features[[label == args.source_label for label in labels]]
    target = features[[label == args.target_label for label in labels]]
    if not len(source) or not len(target):
        parser.error("source and target labels must both occur in --anchors")

    cache_provenance = cached.get("provenance", {})
    metadata = {
        "method": "constant-centroid-steering",
        "representation": cache_provenance.get(
            "representation", "projected_final_encoder_masked_mean"
        ),
        "direction": f"{args.source_label}-to-{args.target_label}",
        "source_label": args.source_label,
        "target_label": args.target_label,
        "source_samples": len(source),
        "target_samples": len(target),
        "anchor_sha256": anchor_sha,
        "activation_cache_sha256": hashlib.sha256(args.activations_cache.read_bytes()).hexdigest(),
        "model_id": cache_provenance.get("model_id"),
        "model_revision": cache_provenance.get("model_revision"),
    }
    for key in ("stage", "checkpoint", "checkpoint_sha256", "text_field", "prompt_text_field"):
        if cache_provenance.get(key) is not None:
            metadata[key] = cache_provenance[key]
    steering = ConstantSteering(target.mean(0) - source.mean(0), metadata)
    steering.save(args.output)
    print(json.dumps({**metadata, "offset_norm": steering.offset.norm().item()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
