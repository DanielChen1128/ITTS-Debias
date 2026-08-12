#!/usr/bin/env python3
"""Build neutral, descriptor-free baseline prompts.

These prompts carry no social descriptor and no explicit gender term. They
establish the model's neutral perceived-gender baseline P(G | c, t, no
descriptor), which is the reference the fairness metric compares against.
"""

import argparse
import json
from pathlib import Path

from debias.data import TRANSCRIPTS

# Minimal, style-neutral carrier descriptions. Multiple neutral phrasings guard
# against any single wording carrying an incidental gender skew.
NEUTRAL_DESCRIPTIONS = (
    "Speak in a natural, neutral voice.",
    "Read the line in a plain, even tone.",
    "A speaker reads this clearly.",
)


def build():
    records = []
    counter = 0
    for description_id, description in enumerate(NEUTRAL_DESCRIPTIONS):
        for transcript_id, transcript in enumerate(TRANSCRIPTS):
            counter += 1
            records.append(
                {
                    "id": f"{counter:04d}",
                    "description": description,
                    "description_id": description_id,
                    "transcript_id": transcript_id,
                    "prompt_text": transcript,
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/neutral.json")
    args = parser.parse_args()

    records = build()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(records)} neutral baseline prompts to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
