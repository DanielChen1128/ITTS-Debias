#!/usr/bin/env python3
"""Build the Stage-4 Go/No-Go pilot prompt set (small smoke slice).

The pilot exercises the full generate -> evaluate pipeline on a deliberately
small set of prompts and four prompt *kinds*:

  * grounded  - implicit descriptors whose career carries a data-grounded gender
                prior. Drawn from the TRAIN split only (never the LEACE fit
                labels, never the held-out test set) so we get a directional
                fairness signal without test leakage. dev has no grounded
                careers, hence train.
  * dev       - implicit dev-split descriptors with no grounded gender label;
                used to gauge the *magnitude* of the gender shift relative to
                the neutral baseline and its reduction after debiasing.
  * neutral   - no-descriptor reference voices; the baseline P(female | t) that
                the fairness metric is measured against.
  * anchor    - explicit male/female control prompts (anchors control split);
                used for explicit-control accuracy and bypass verification.

Each record carries id/description/prompt_text (consumed by generate_wav.py)
plus evaluation metadata (kind/expected_gender/axis/lemma/transcript_id/explicit).
"""

import argparse
import json
from pathlib import Path

from debias.data import DATASET_PROFILES, load_descriptors, validate_split_manifest
from debias.parler import has_explicit_gender_command


def _dedup_by_description(records):
    seen = {}
    for r in records:
        seen.setdefault(r["description"], r)
    return list(seen.values())


def build(descriptions_root, splits_path, neutral_path, anchors_path, *,
          transcripts, grounded_per_lemma, dev_lemmas_per_axis, dev_per_lemma,
          neutral_descriptions, anchor_per_gender, grounded_split="train",
          dataset_profile="legacy-5900-v1"):
    split_manifest = json.loads(Path(splits_path).read_text(encoding="utf-8"))
    validate_split_manifest(split_manifest, descriptions_root, dataset_profile)
    assignment = split_manifest["lemma_assignment"]
    records = load_descriptors(descriptions_root, dataset_profile)
    items = []
    counter = 0

    def emit(description, prompt_text, transcript_id, kind, expected_gender, axis, lemma):
        nonlocal counter
        counter += 1
        items.append({
            "id": f"{counter:04d}",
            "description": description,
            "prompt_text": prompt_text,
            "kind": kind,
            "expected_gender": expected_gender,
            "axis": axis,
            "lemma": lemma,
            "transcript_id": transcript_id,
            "explicit": bool(has_explicit_gender_command(description)),
        })

    # --- grounded implicit (directional prior; split configurable) ---
    grounded = [
        r for r in records
        if r["axis"] == "career"
        and r["stereotype_gender"] in ("male", "female")
        and assignment.get(r["lemma"]) == grounded_split
    ]
    by_lemma = {}
    for r in grounded:
        by_lemma.setdefault(r["lemma"], []).append(r)
    for lemma in sorted(by_lemma):
        chosen = _dedup_by_description(by_lemma[lemma])[:grounded_per_lemma]
        for r in chosen:
            for tid in transcripts:
                emit(r["description"], transcripts[tid], tid, "grounded",
                     r["stereotype_gender"], r["axis"], lemma)

    # --- dev implicit (no grounded label; magnitude probe) ---
    for axis in ("career", "persona"):
        dev = [
            r for r in records
            if r["axis"] == axis and assignment.get(r["lemma"]) == "dev"
        ]
        lemmas = sorted({r["lemma"] for r in dev})[:dev_lemmas_per_axis]
        for lemma in lemmas:
            chosen = _dedup_by_description([r for r in dev if r["lemma"] == lemma])[:dev_per_lemma]
            for r in chosen:
                for tid in transcripts:
                    emit(r["description"], transcripts[tid], tid, "dev",
                         None, axis, lemma)

    # --- neutral baseline ---
    neutral = json.loads(Path(neutral_path).read_text(encoding="utf-8"))
    neutral_descs = []
    for r in neutral:
        if r["description"] not in neutral_descs:
            neutral_descs.append(r["description"])
    for description in neutral_descs[:neutral_descriptions]:
        for tid in transcripts:
            emit(description, transcripts[tid], tid, "neutral", None, "neutral",
                 f"neutral:{neutral_descs.index(description)}")

    # --- explicit anchor control ---
    anchors = json.loads(Path(anchors_path).read_text(encoding="utf-8"))
    control = [a for a in anchors if a["anchor_split"] == "control"]
    for gender in ("female", "male"):
        descs = _dedup_by_description(
            [a for a in control if a["gender_label"] == gender]
        )[:anchor_per_gender]
        for a in descs:
            for tid in transcripts:
                emit(a["description"], transcripts[tid], tid, "anchor", gender,
                     "anchor", f"anchor:{gender}")

    return items


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--descriptions", default="datasets/legacy-5900-v1/descriptions"
    )
    parser.add_argument(
        "--dataset-profile", default="legacy-5900-v1", choices=sorted(DATASET_PROFILES)
    )
    parser.add_argument("--splits", default="datasets/legacy-5900-v1/data/splits.json")
    parser.add_argument("--neutral", default="datasets/legacy-5900-v1/data/neutral.json")
    parser.add_argument("--anchors", default="datasets/legacy-5900-v1/data/anchors.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grounded-split", default="train",
                        choices=["train", "dev", "test"],
                        help="split whose grounded careers seed the fairness signal "
                             "(train for pilot, test for the main experiment)")
    parser.add_argument("--transcript-ids", default="0,1")
    parser.add_argument("--grounded-per-lemma", type=int, default=1)
    parser.add_argument("--dev-lemmas-per-axis", type=int, default=3)
    parser.add_argument("--dev-per-lemma", type=int, default=1)
    parser.add_argument("--neutral-descriptions", type=int, default=3)
    parser.add_argument("--anchor-per-gender", type=int, default=2)
    args = parser.parse_args()

    from debias.data import TRANSCRIPTS_BY_PROFILE
    tids = [int(x) for x in args.transcript_ids.split(",") if x.strip() != ""]
    transcript_catalog = TRANSCRIPTS_BY_PROFILE[args.dataset_profile]
    transcripts = {tid: transcript_catalog[tid] for tid in tids}

    items = build(
        args.descriptions, args.splits, args.neutral, args.anchors,
        transcripts=transcripts,
        grounded_per_lemma=args.grounded_per_lemma,
        dev_lemmas_per_axis=args.dev_lemmas_per_axis,
        dev_per_lemma=args.dev_per_lemma,
        neutral_descriptions=args.neutral_descriptions,
        anchor_per_gender=args.anchor_per_gender,
        grounded_split=args.grounded_split,
        dataset_profile=args.dataset_profile,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    from collections import Counter
    kinds = Counter(i["kind"] for i in items)
    print(f"Wrote {len(items)} pilot prompts to {out}")
    for kind, n in sorted(kinds.items()):
        print(f"  {kind:9s} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
