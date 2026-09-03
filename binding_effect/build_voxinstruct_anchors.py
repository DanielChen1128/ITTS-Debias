#!/usr/bin/env python3
"""Add pair-matched neutral transcripts to constant-steering anchors."""

import argparse
import hashlib
import json
from pathlib import Path


TRANSCRIPTS = (
    "The meeting begins at nine o'clock tomorrow morning.",
    "Please place the package beside the front entrance.",
    "The library recently added several books to its collection.",
    "We can review the updated schedule after lunch.",
    "A light breeze moved through the trees near the river.",
    "The train is expected to arrive at the central station soon.",
    "Remember to bring the documents needed for the appointment.",
    "The community center offers classes throughout the week.",
    "Fresh coffee and tea are available in the next room.",
    "Our next discussion will cover the remaining questions.",
    "The weather should remain clear for most of the afternoon.",
    "Please check the final report before sending it to the team.",
    "The museum opens its new exhibition on Saturday.",
    "We have enough time to consider each option carefully.",
)


def build(rows):
    if len(rows) % 2:
        raise ValueError("anchors must contain complete female/male pairs")
    output = []
    for pair_index in range(0, len(rows), 2):
        pair = rows[pair_index:pair_index + 2]
        labels = {row.get("gender_label") for row in pair}
        if labels != {"female", "male"}:
            raise ValueError(f"anchor pair {pair_index // 2} is not female/male matched")
        comparison_fields = ("context_id", "axes", "descriptor_ids", "template_id")
        if any(pair[0].get(field) != pair[1].get(field) for field in comparison_fields):
            raise ValueError(f"anchor pair {pair_index // 2} has mismatched context metadata")
        transcript = TRANSCRIPTS[(pair_index // 2) % len(TRANSCRIPTS)]
        for row in pair:
            enriched = dict(row)
            enriched["prompt_text"] = transcript
            enriched["protocol_id"] = "voxinstruct-balanced-paired-transcript-anchors-v1"
            output.append(enriched)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    rows = json.loads(args.input.read_text(encoding="utf-8"))
    output = build(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(output, indent=2, ensure_ascii=True) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    manifest = {
        "protocol_id": "voxinstruct-balanced-paired-transcript-anchors-v1",
        "source": str(args.input),
        "source_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "anchor_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "records": len(output),
        "pairs": len(output) // 2,
        "transcripts": len(TRANSCRIPTS),
        "pair_matched_transcript": True,
        "evaluation_prompts_used": False,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
