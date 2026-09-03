#!/usr/bin/env python3
"""Fit an RLACE artifact from labeled Parler-TTS voice descriptions."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from debias.rlace import fit_rlace
from debias.parler import collect_pooled_activations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="JSON list containing descriptions and labels")
    parser.add_argument("--output", help="Output .pt RLACE artifact")
    parser.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    parser.add_argument("--model-revision", help="Immutable Hugging Face revision for Parler")
    parser.add_argument("--text-field", default="description")
    parser.add_argument("--label-field", default="gender_label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Subspace rank removed by RLACE",
    )
    parser.add_argument("--seed", type=int, default=0, help="Optimization or random-projection seed")
    parser.add_argument("--rlace-epochs", type=int, default=100)
    parser.add_argument("--rlace-adversary-steps", type=int, default=5)
    parser.add_argument("--rlace-adversary-lr", type=float, default=5e-2)
    parser.add_argument("--rlace-projector-lr", type=float, default=1e-3)
    parser.add_argument("--rlace-l2", type=float, default=1e-4)
    parser.add_argument(
        "--activations-cache", type=Path,
        help="Reuse pooled encoder activations when training and model provenance match",
    )
    parser.add_argument(
        "--extract-only", action="store_true",
        help="Populate or validate --activations-cache without fitting an eraser",
    )
    parser.add_argument(
        "--training-manifest", type=Path,
        help="Optional small-context training manifest recorded in artifact provenance",
    )
    args = parser.parse_args()
    if args.extract_only and not args.activations_cache:
        parser.error("--extract-only requires --activations-cache")
    if not args.extract_only and not args.output:
        parser.error("--output is required unless --extract-only is used")

    records = json.loads(Path(args.json).read_text(encoding="utf-8"))
    if not isinstance(records, list) or len(records) < 2:
        parser.error("--json must contain a list with at least two records")
    try:
        texts = [record[args.text_field] for record in records]
        raw_labels = [record[args.label_field] for record in records]
    except (KeyError, TypeError) as exc:
        parser.error(f"invalid training record: {exc}")
    label_names = sorted(set(raw_labels))
    if len(label_names) < 2:
        parser.error("at least two distinct labels are required")
    label_ids = torch.tensor([label_names.index(label) for label in raw_labels])

    training_sha256 = hashlib.sha256(Path(args.json).read_bytes()).hexdigest()
    cache_provenance = {
        "training_sha256": training_sha256,
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "text_field": args.text_field,
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.activations_cache and args.activations_cache.exists():
        cached = torch.load(args.activations_cache, map_location="cpu", weights_only=True)
        if cached.get("provenance") != cache_provenance:
            parser.error("--activations-cache provenance does not match this fit")
        activations = cached["activations"]
        if activations.ndim != 2 or activations.shape[0] != len(records):
            parser.error("--activations-cache has an invalid activation shape")
        print(f"Loaded activations from {args.activations_cache}")
    else:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        dtype = torch.float16 if device == "cuda" else torch.float32
        revision_kwargs = {"revision": args.model_revision} if args.model_revision else {}
        model = ParlerTTSForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, **revision_kwargs
        ).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, **revision_kwargs)
        activations = collect_pooled_activations(
            model, tokenizer, texts, batch_size=args.batch_size, device=device
        ).cpu()
        if args.activations_cache:
            args.activations_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"provenance": cache_provenance, "activations": activations}, args.activations_cache)
            print(f"Saved activations to {args.activations_cache}")
    if args.extract_only:
        print(f"Validated {activations.shape[0]} pooled activations with {activations.shape[1]} features")
        return 0
    base_metadata = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "representation": "projected_final_encoder_masked_mean",
        "labels": label_names,
        "samples": len(records),
        "method": "rlace",
        "training_sha256": training_sha256,
        "protocol_ids": sorted({
            record.get("protocol_id") for record in records if record.get("protocol_id")
        }),
        "context_count": len({
            record.get("context_id") for record in records if record.get("context_id")
        }),
        "duplicate_policy": "exact_description_duplicates_rejected_by_anchor_audit",
    }
    if args.training_manifest:
        manifest_bytes = args.training_manifest.read_bytes()
        training_manifest = json.loads(manifest_bytes)
        base_metadata.update({
            "training_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
            "training_protocol_id": training_manifest.get("protocol_id"),
            "status_in_training": training_manifest.get("status_in_training"),
        })
    eraser = fit_rlace(
        activations,
        label_ids,
        rank=args.rank,
        epochs=args.rlace_epochs,
        adversary_steps=args.rlace_adversary_steps,
        adversary_lr=args.rlace_adversary_lr,
        projector_lr=args.rlace_projector_lr,
        l2=args.rlace_l2,
        seed=args.seed,
        device=device,
        metadata=base_metadata,
    )
    eraser.metadata["fitted_rank"] = eraser.concept_rank
    eraser.save(args.output)
    print(f"Saved rank-{eraser.concept_rank} RLACE artifact to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
