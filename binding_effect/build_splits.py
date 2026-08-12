#!/usr/bin/env python3
"""Produce a leakage-safe train/dev/test split of held-out descriptors.

Splitting is performed at the descriptor-lemma level, independently within each
axis. Axes with at least three lemmas retain train/dev/test items; the two-lemma
status axis cannot populate all three splits without descriptor leakage.
Because a lemma (e.g. a specific career or persona word) lives entirely in one
split, no stereotype cue used to tune the intervention reappears at test time.
The neutral baseline and explicit anchors are defined separately and are never
part of these evaluation splits.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from debias.data import DATASET_PROFILES, assign_splits, dataset_hashes, load_descriptors


def build(root, seed, dev_frac, test_frac, profile, *, verify_dataset=True):
    records = load_descriptors(root, profile, verify=verify_dataset)
    source_sha256 = dataset_hashes(root, profile) if verify_dataset else None

    lemmas_by_axis = defaultdict(set)
    for record in records:
        lemmas_by_axis[record["axis"]].add(record["lemma"])

    assignment = {}
    for axis, lemmas in lemmas_by_axis.items():
        assignment.update(
            assign_splits(lemmas, seed=seed, dev_frac=dev_frac, test_frac=test_frac)
        )

    record_counts = defaultdict(lambda: defaultdict(int))
    lemma_counts = defaultdict(lambda: defaultdict(int))
    for record in records:
        split = assignment[record["lemma"]]
        record_counts[record["axis"]][split] += 1
    for lemma, split in assignment.items():
        axis = lemma.split(":", 1)[0]
        # Normalise composite axis keys back to their axis name.
        axis = "two_axis" if axis == "descriptions_two_axis" else axis
        axis = "multi_axis" if axis == "descriptions_multi_axis" else axis
        lemma_counts[axis][split] += 1

    manifest = {
        "dataset_profile": profile,
        "source_sha256": source_sha256,
        "seed": seed,
        "dev_frac": dev_frac,
        "test_frac": test_frac,
        "split_unit": "descriptor_lemma",
        "n_records": len(records),
        "n_lemmas": len(assignment),
        "record_counts_by_axis": {k: dict(v) for k, v in record_counts.items()},
        "lemma_counts_by_axis": {k: dict(v) for k, v in lemma_counts.items()},
        "lemma_assignment": assignment,
    }
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=sorted(DATASET_PROFILES))
    parser.add_argument("--root", help="Descriptor root; defaults from --profile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--dev-frac", type=float, default=0.2)
    parser.add_argument("--test-frac", type=float, default=0.4)
    args = parser.parse_args()

    root = args.root or (
        "descriptions"
        if args.profile == "canonical-stage1-6900-v1"
        else "datasets/legacy-5900-v1/descriptions"
    )
    manifest = build(root, args.seed, args.dev_frac, args.test_frac, args.profile)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote split for {manifest['n_lemmas']} lemmas / {manifest['n_records']} records to {out}")
    for axis, counts in sorted(manifest["record_counts_by_axis"].items()):
        print(f"  {axis}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
