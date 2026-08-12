#!/usr/bin/env python3
"""Stage 2: screen the Parler encoder for linear gender decodability.

Runs entirely on encoder activations (no audio generation). It measures:
  * separability   - can a probe read gender from explicit anchors?
  * spillover      - does that anchor direction predict the stereotype gender
                     of held-out implicit descriptors?
  * erasure        - do the above collapse after unconditional LEACE?
  * axis retention - does LEACE preserve non-gender axis information?

Grounded-prior labels only (male/female) are used for the spillover set.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from debias.data import DATASET_PROFILES, load_descriptors, validate_split_manifest
from debias.leace import LeaceEraser
from debias.probe import LinearProbe


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _gender_tensor(labels):
    return torch.tensor([1 if g == "female" else 0 for g in labels], dtype=torch.long)


def _select_spillover(records, assignment, split):
    seen = {}
    for record in records:
        if (
            record["axis"] == "career"
            and record["stereotype_gender"] in ("male", "female")
            and assignment.get(record["lemma"]) == split
        ):
            seen.setdefault(record["description"], record)
    return list(seen.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="parler-tts/parler-tts-mini-v1")
    parser.add_argument("--anchors", default="datasets/legacy-5900-v1/data/anchors.json")
    parser.add_argument("--splits", default="datasets/legacy-5900-v1/data/splits.json")
    parser.add_argument(
        "--descriptions", default="datasets/legacy-5900-v1/descriptions"
    )
    parser.add_argument(
        "--dataset-profile", default="legacy-5900-v1", choices=sorted(DATASET_PROFILES)
    )
    parser.add_argument(
        "--spillover-split", default="train",
        choices=("train", "dev", "test"),
        help="Descriptor split used for the implicit-career spillover screen",
    )
    parser.add_argument("--output", default="results/screening/report.json")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    from debias.parler import collect_pooled_activations

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = (
        ParlerTTSForConditionalGeneration.from_pretrained(
            args.model_id, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, local_files_only=True)

    def encode(texts):
        return collect_pooled_activations(
            model, tokenizer, texts, batch_size=args.batch_size, device=device
        )

    # --- Explicit anchors: fit vs held-out control ---
    anchors = _load_json(args.anchors)
    fit = [a for a in anchors if a["anchor_split"] == "fit"]
    control = [a for a in anchors if a["anchor_split"] == "control"]
    fit_x = encode([a["description"] for a in fit])
    fit_y = _gender_tensor([a["gender_label"] for a in fit])
    control_x = encode([a["description"] for a in control])
    control_y = _gender_tensor([a["gender_label"] for a in control])

    # --- Implicit spillover set: career-axis descriptors with clean grounded
    #     priors from the requested held-out split, deduplicated by description.
    #     Composite files are excluded because their gender annotation conflates
    #     axes. ---
    split_manifest = _load_json(args.splits)
    validate_split_manifest(split_manifest, args.descriptions, args.dataset_profile)
    assignment = split_manifest["lemma_assignment"]
    descriptors = _select_spillover(
        load_descriptors(args.descriptions, args.dataset_profile),
        assignment,
        args.spillover_split,
    )
    if not descriptors:
        raise ValueError(
            f"no grounded career descriptors are assigned to {args.spillover_split!r}"
        )
    spill_x = encode([r["description"] for r in descriptors])
    spill_y = _gender_tensor([r["stereotype_gender"] for r in descriptors])
    spill_lemma = [r["lemma"] for r in descriptors]

    # --- Fit LEACE (unconditional) on anchor fit activations ---
    eraser = LeaceEraser.fit(fit_x, fit_y, metadata={"model_id": args.model_id})

    def probe_report(train_x, train_y, sets):
        probe = LinearProbe.fit(train_x, train_y, seed=args.seed)
        return {name: probe.evaluate(x, y) for name, (x, y) in sets.items()}

    def per_lemma_auc(probe, x, y, lemmas):
        groups = defaultdict(list)
        for index, lemma in enumerate(lemmas):
            groups[lemma].append(index)
        out = {}
        for lemma, idx in groups.items():
            sub_y = y[idx]
            if len(set(sub_y.tolist())) < 2:
                continue
            from debias.probe import roc_auc

            out[lemma] = roc_auc(probe.scores(x[idx]), sub_y)
        return out

    results = {
        "model_id": args.model_id,
        "dataset_profile": args.dataset_profile,
        "spillover_split": args.spillover_split,
        "spillover_descriptions": len(descriptors),
        "spillover_lemmas": len(set(spill_lemma)),
    }

    for phase, transform in (("original", None), ("leace", eraser)):
        if transform is None:
            f_x, c_x, s_x = fit_x, control_x, spill_x
        else:
            f_x, c_x, s_x = transform(fit_x), transform(control_x), transform(spill_x)

        gender_probe = LinearProbe.fit(f_x, fit_y, seed=args.seed)
        results[phase] = {
            "separability_control": gender_probe.evaluate(c_x, control_y),
            "spillover": gender_probe.evaluate(s_x, spill_y),
            "spillover_per_lemma": per_lemma_auc(gender_probe, s_x, spill_y, spill_lemma),
        }

    # --- Axis-retention: probe a non-gender axis contrast before/after LEACE ---
    axis_pairs = [
        r
        for r in load_descriptors(args.descriptions, args.dataset_profile)
        if r["axis"] == "status" and assignment.get(r["lemma"]) in ("train", "test", "dev")
    ]
    if axis_pairs:
        axis_x = encode([r["description"] for r in axis_pairs])
        axis_y = torch.tensor(
            [1 if "high" in r["lemma"] else 0 for r in axis_pairs], dtype=torch.long
        )
        if len(set(axis_y.tolist())) == 2:
            results["axis_retention_status"] = {
                "original": LinearProbe.fit(axis_x, axis_y, seed=args.seed).evaluate(axis_x, axis_y),
                "leace": LinearProbe.fit(eraser(axis_x), axis_y, seed=args.seed).evaluate(
                    eraser(axis_x), axis_y
                ),
            }

    results["leace_rank"] = eraser.concept_rank
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[screening] LEACE rank={eraser.concept_rank}")
    for phase in ("original", "leace"):
        sep = results[phase]["separability_control"]["auc"]
        spill = results[phase]["spillover"]["auc"]
        print(f"  {phase:8s} separability AUC={sep:.3f}  spillover AUC={spill:.3f}")
    print(f"[screening] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
