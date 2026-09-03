#!/usr/bin/env python3
"""Extract separate AR and NAR VoxInstruct projected conditioning caches."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

from debias.voxinstruct import REPRESENTATIONS, collect_pooled_activations, format_instruction


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--model-id", type=Path, required=True)
    parser.add_argument("--backend-path", type=Path, required=True)
    parser.add_argument("--ar-output", type=Path, required=True)
    parser.add_argument("--nar-output", type=Path, required=True)
    parser.add_argument("--text-field", default="description")
    parser.add_argument("--prompt-text-field")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = json.loads(args.json.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or len(rows) < 2:
        parser.error("--json must contain a list with at least two records")
    try:
        texts = [
            format_instruction(
                row[args.text_field],
                row.get(args.prompt_text_field) if args.prompt_text_field else None,
            )
            for row in rows
        ]
    except (KeyError, TypeError) as exc:
        parser.error(f"invalid activation record: {exc}")

    backend_path = args.backend_path.resolve()
    model_id = args.model_id.resolve()
    checkpoint_dir = model_id / "voxinstruct-sft-checkpoint"
    sys.path.insert(0, str(backend_path))
    from model.ar import VoxInstructAR
    from model.nar import VoxInstructNAR
    from transformers import AutoTokenizer
    from utils.utils import get_config_from_file

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    mt5_path = model_id / "google-mt5-base-checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(mt5_path, local_files_only=True)
    training_sha256 = hashlib.sha256(args.json.read_bytes()).hexdigest()

    stages = (
        ("ar", VoxInstructAR, backend_path / "configs/train_ar.yaml", checkpoint_dir / "ar_1800k.pyt", args.ar_output),
        ("nar", VoxInstructNAR, backend_path / "configs/train_nar.yaml", checkpoint_dir / "nar_1800k.pyt", args.nar_output),
    )
    for stage, model_class, config_path, checkpoint_path, output_path in stages:
        hp = get_config_from_file(str(config_path)).hparams
        hp.mt5_path = str(mt5_path)
        model = model_class(hp=hp)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        del checkpoint
        model.to(dtype).eval()
        model.to(device)
        activations = collect_pooled_activations(
            model, tokenizer, texts, max_length=hp.max_text_len,
            batch_size=args.batch_size, device=device,
        )
        provenance = {
            "training_sha256": training_sha256,
            "model_id": str(model_id),
            "model_revision": None,
            "stage": stage,
            "representation": REPRESENTATIONS[stage],
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": file_sha256(checkpoint_path),
            "text_field": args.text_field,
            "prompt_text_field": args.prompt_text_field,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"provenance": provenance, "activations": activations}, output_path)
        print(f"Saved {tuple(activations.shape)} {stage.upper()} activations to {output_path}")
        del model, activations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
