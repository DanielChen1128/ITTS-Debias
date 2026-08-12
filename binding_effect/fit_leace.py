#!/usr/bin/env python3
"""Fit a LEACE artifact from labeled Parler-TTS voice descriptions."""

import argparse
import json
from pathlib import Path

import torch

from debias.comparators import fit_gender_direction, fit_random_projection
from debias.leace import LeaceEraser
from debias.parler import collect_pooled_activations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", required=True, help="JSON list containing descriptions and labels")
    parser.add_argument("--output", required=True, help="Output .pt LEACE artifact")
    parser.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    parser.add_argument("--text-field", default="description")
    parser.add_argument("--label-field", default="gender_label")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--method",
        default="leace",
        choices=("leace", "gender-direction", "random"),
        help="Erasure method: full LEACE, rank-1 gender-direction, or random projection",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=1,
        help="Subspace rank removed by the random-projection baseline",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed for the random-projection baseline")
    args = parser.parse_args()

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

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    activations = collect_pooled_activations(
        model, tokenizer, texts, batch_size=args.batch_size, device=device
    )
    base_metadata = {
        "model_id": args.model_id,
        "representation": "projected_final_encoder_masked_mean",
        "labels": label_names,
        "samples": len(records),
        "method": args.method,
    }
    if args.method == "leace":
        eraser = LeaceEraser.fit(activations, label_ids, metadata=base_metadata)
    elif args.method == "gender-direction":
        eraser = fit_gender_direction(activations, label_ids, metadata=base_metadata)
    else:
        eraser = fit_random_projection(
            activations, args.rank, seed=args.seed, metadata=base_metadata
        )
    eraser.save(args.output)
    print(f"Saved rank-{eraser.concept_rank} {args.method} artifact to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
