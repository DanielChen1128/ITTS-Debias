#!/usr/bin/env python3
"""Build matched male/female explicit anchor prompts for LEACE fitting.

Anchors provide clean gender-labelled activations. Each carrier template is
rendered once per gender so that the only systematic difference between a
matched pair is the explicit gender term. Templates are partitioned into a
``fit`` group used to estimate the eraser and a held-out ``control`` group used
to check that explicit gender control survives the intervention.
"""

import argparse
import json
from pathlib import Path

from debias.data import TRANSCRIPTS

# Carrier templates with matched adjective/noun gender terms. ``{adj}`` takes
# female/male and ``{noun}`` takes woman/man so both lexical forms are covered.
TEMPLATES = (
    "A {adj} speaker with a clear, natural voice.",
    "Speak with a distinctly {adj} voice.",
    "The voice belongs to a {noun}.",
    "Use the voice of a {noun} speaking calmly.",
    "This is a {adj} narrator reading aloud.",
    "Please read this as a {noun} would.",
    "A {adj} presenter delivers the line evenly.",
    "The recording features a {noun}'s voice.",
)

GENDER_TERMS = {
    "female": {"adj": "female", "noun": "woman"},
    "male": {"adj": "male", "noun": "man"},
}


def build(control_templates):
    records = []
    counter = 0
    for template_id, template in enumerate(TEMPLATES):
        split = "control" if template_id in control_templates else "fit"
        for gender, terms in GENDER_TERMS.items():
            description = template.format(**terms)
            for transcript_id, transcript in enumerate(TRANSCRIPTS):
                counter += 1
                records.append(
                    {
                        "id": f"{counter:04d}",
                        "description": description,
                        "gender_label": gender,
                        "template_id": template_id,
                        "transcript_id": transcript_id,
                        "prompt_text": transcript,
                        "anchor_split": split,
                    }
                )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/anchors.json")
    parser.add_argument(
        "--control-templates",
        default="6,7",
        help="Comma-separated template ids held out for explicit-control evaluation",
    )
    args = parser.parse_args()

    control = {int(x) for x in args.control_templates.split(",") if x.strip() != ""}
    records = build(control)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    fit = sum(1 for r in records if r["anchor_split"] == "fit")
    print(f"Wrote {len(records)} anchors ({fit} fit / {len(records) - fit} control) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
