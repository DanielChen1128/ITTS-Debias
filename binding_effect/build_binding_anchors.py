#!/usr/bin/env python3
"""Build the fixed 540-anchor small-context LEACE training set."""

import argparse
import hashlib
import json
from pathlib import Path

from build_prompts import source_catalog
from debias.binding_protocol import (
    ANCHOR_TEMPLATES,
    BASE_SEED,
    GENDER_TERMS,
    PROTOCOL_ID,
    TRAIN_CAREER_IDS,
    TRAIN_PERSONA_IDS,
    render_anchor,
    training_contexts,
)
from debias.data import dataset_hashes


def build(catalog):
    records = []
    for context in training_contexts(catalog):
        for template_id, template in enumerate(ANCHOR_TEMPLATES):
            for gender in GENDER_TERMS:
                records.append({
                    "id": f"anchor-{len(records) + 1:04d}",
                    "description": render_anchor(context, template, gender),
                    "gender_label": gender,
                    "context_id": context["context_id"],
                    "axes": context["axes"],
                    "descriptor_ids": context["descriptor_ids"],
                    "template_id": template_id,
                    "protocol_id": PROTOCOL_ID,
                })
    if len(records) != 540:
        raise AssertionError(f"expected 540 anchors, found {len(records)}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("descriptions"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    catalog = source_catalog(args.source_dir)
    records = build(catalog)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(records, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "seed": BASE_SEED,
        "training_only": True,
        "status_in_training": False,
        "career_descriptor_ids": list(TRAIN_CAREER_IDS),
        "persona_descriptor_ids": list(TRAIN_PERSONA_IDS),
        "context_counts": {"neutral": 1, "career": 4, "persona": 8, "career+persona": 32},
        "contexts": 45,
        "templates": 6,
        "genders": 2,
        "anchors": len(records),
        "anchor_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_sha256": {
            name: digest
            for name, digest in dataset_hashes(
                args.source_dir, "canonical-stage1-6900-v1"
            ).items()
            if name in {"description_career_bias.json", "descriptions_persona_bias.json"}
        },
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(records)} training anchors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
