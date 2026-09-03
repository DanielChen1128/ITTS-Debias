#!/usr/bin/env python3
"""Build the frozen 5,040-anchor Parler Large RLACE training set."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_prompts import source_catalog
from debias.data import dataset_hashes
from debias.large_anchor_protocol import (
    BASE_SEED, PROTOCOL_ID, TRAIN_TEMPLATES, expanded_contexts, render_large_anchor,
)


def build(catalog):
    records = []
    for context in expanded_contexts(catalog):
        for template_id, template_pair in enumerate(TRAIN_TEMPLATES):
            for gender in ("female", "male"):
                records.append({
                    "id": f"large-anchor-{len(records) + 1:05d}",
                    "description": render_large_anchor(context, template_pair, gender),
                    "gender_label": gender,
                    "context_id": context["context_id"],
                    "axes": context["axes"],
                    "descriptor_ids": context["descriptor_ids"],
                    "template_id": template_id,
                    "protocol_id": PROTOCOL_ID,
                })
    if len(records) != 5040:
        raise AssertionError(f"expected 5040 anchors, found {len(records)}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("descriptions"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    records = build(source_catalog(args.source_dir))
    payload = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    context_counts = Counter("+".join(row["axes"]) or "neutral" for row in records[::28])
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "seed": BASE_SEED,
        "training_only": True,
        "status_in_training": False,
        "adult_gender_terms_only": True,
        "context_counts": dict(sorted(context_counts.items())),
        "contexts": 180,
        "templates": len(TRAIN_TEMPLATES),
        "genders": 2,
        "anchors": len(records),
        "anchor_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "held_out_template_sets": ["gate_dev", "gate_test"],
        "source_sha256": {
            name: digest for name, digest in dataset_hashes(
                args.source_dir, "canonical-stage1-6900-v1"
            ).items() if name in {"description_career_bias.json", "descriptions_persona_bias.json"}
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} Large RLACE anchors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
