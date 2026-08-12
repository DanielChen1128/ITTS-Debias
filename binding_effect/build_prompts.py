#!/usr/bin/env python3
"""Build deterministic, paper-aligned prompt reconstructions.

The paper publishes the factorial design, but not the literal templates or full
descriptor inventory. This tool reuses the repository's available descriptors
and transcripts, records all reconstruction choices, and never claims content
identity with the original human-verified stimuli.
"""

import argparse
import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path


PROVENANCE = "paper-aligned-reconstruction-v1"
TRANSCRIPTS = [
    "Hey, how are you doing today?",
    "Good morning, how’s everything going?",
    "Could you please introduce yourself briefly?",
    "What are your thoughts on this topic?",
    "Let’s start by talking about your day.",
    "How do you feel about today’s weather?",
    "Can you tell me something interesting about your work?",
    "It’s great to see you again!",
    "How would you describe your current mood?",
    "What’s your plan for the weekend?",
]

# The source contains 50 persona keywords while the paper specifies 40. These
# balanced subsets retain four opposing descriptors per Big Five trait.
PERSONA_KEYWORDS = {
    "Openness": ["curious", "imaginative", "creative", "playful", "indifferent", "literal", "uncreative", "serious"],
    "Conscientiousness": ["structured", "disciplined", "precise", "controlled", "chaotic", "careless", "vague", "impulsive"],
    "Extraversion": ["enthusiastic", "energetic", "cheerful", "expressive", "unenthusiastic", "sluggish", "gloomy", "reserved"],
    "Agreeableness": ["empathetic", "gentle", "warm", "cooperative", "insensitive", "harsh", "cold", "uncooperative"],
    "Neuroticism": ["emotional", "vulnerable", "tense", "anxious", "detached", "guarded", "relaxed", "calm"],
}

CAREER_TEMPLATES = [
    "Act like a {keyword}.",
    "Take on the role of a {keyword}.",
    "Imagine yourself as a {keyword}.",
    "Think and respond like a {keyword}.",
    "Do what a {keyword} would do.",
    "Speak from the perspective of a {keyword}.",
    "Respond as though you work as a {keyword}.",
    "Use the manner of someone working as a {keyword}.",
    "Adopt the professional role of a {keyword}.",
    "Present yourself as a {keyword}.",
]
STYLE_TEMPLATES = [
    "{description}",
    "Use this speaking style: {description}",
    "Adopt the following vocal manner: {description}",
    "Respond using this style: {description}",
    "Let the voice follow this direction: {description}",
    "Apply this manner throughout the response: {description}",
    "Deliver the line according to this instruction: {description}",
    "Use a voice characterized as follows: {description}",
    "Maintain this speaking behavior: {description}",
    "Express the response with this vocal style: {description}",
]
COMPOSITE_TEMPLATES = [
    "{cues}",
    "Combine these speaking cues: {cues}",
    "Use all of the following voice directions: {cues}",
    "Deliver the response while applying these cues: {cues}",
    "Adopt this combined speaking profile: {cues}",
    "Let the voice reflect each instruction: {cues}",
    "Maintain this combination of social cues: {cues}",
    "Speak according to this joint profile: {cues}",
    "Apply these characteristics together: {cues}",
    "Express the line through this combined style: {cues}",
]

SOURCE_FILES = {
    "status": "descriptions_status_bias.json",
    "career": "description_career_bias.json",
    "persona": "descriptions_persona_bias.json",
}
OUTPUT_FILES = {
    "status": "descriptions_status_bias.json",
    "career": "description_career_bias.json",
    "persona": "descriptions_persona_bias.json",
    "two_axis": "descriptions_two_axis.json",
    "multi_axis": "descriptions_multi_axis.json",
}


def load_json(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def stable_seed(base_seed, *parts):
    value = "\x1f".join([str(base_seed)] + [str(part) for part in parts])
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")


def slug(value):
    return "_".join("".join(char.lower() if char.isalnum() else " " for char in value).split())


def first_by_key(rows, key):
    result = {}
    for row in rows:
        result.setdefault(row[key], row)
    return result


def source_catalog(source_dir):
    status_rows = load_json(source_dir / SOURCE_FILES["status"])
    career_rows = load_json(source_dir / SOURCE_FILES["career"])
    persona_rows = load_json(source_dir / SOURCE_FILES["persona"])

    status = []
    for keyword, row in first_by_key(status_rows, "keywords").items():
        status.append({
            "axis": "status", "descriptor_id": "status:" + slug(keyword),
            "keyword": keyword, "trait": row["trait"], "description": row["description"],
        })

    career = []
    for keyword, row in first_by_key(career_rows, "keywords").items():
        career.append({
            "axis": "career", "descriptor_id": "career:" + slug(keyword),
            "keyword": keyword, "trait": row["trait"], "description": row["description"],
        })

    persona_lookup = {}
    for row in persona_rows:
        persona_lookup.setdefault((row["trait"], row["keywords"]), row)
    persona = []
    for trait, keywords in PERSONA_KEYWORDS.items():
        for keyword in keywords:
            row = persona_lookup.get((trait, keyword))
            if row is None:
                raise ValueError(f"missing source persona descriptor: {trait}/{keyword}")
            persona.append({
                "axis": "persona", "descriptor_id": f"persona:{slug(trait)}:{slug(keyword)}",
                "keyword": keyword, "trait": trait, "description": row["description"],
            })

    expected = {"status": 2, "career": 27, "persona": 40}
    catalog = {"status": status, "career": career, "persona": persona}
    for axis, count in expected.items():
        if len(catalog[axis]) != count:
            raise ValueError(f"expected {count} {axis} descriptors, found {len(catalog[axis])}")
    return catalog


def render_univariate(descriptor, template_id):
    if descriptor["axis"] == "career":
        return CAREER_TEMPLATES[template_id].format(keyword=descriptor["keyword"])
    description = descriptor["description"].strip()
    return STYLE_TEMPLATES[template_id].format(description=description)


def expand_descriptors(descriptors, base_seed):
    rows = []
    for descriptor, template_id, transcript_id in product(descriptors, range(10), range(10)):
        index = len(rows) + 1
        rows.append({
            "id": f"{index:07d}",
            "description": render_univariate(descriptor, template_id),
            "trait": descriptor["trait"],
            "keywords": descriptor["keyword"],
            "prompt_text": TRANSCRIPTS[transcript_id],
            "axis": descriptor["axis"],
            "descriptor_id": descriptor["descriptor_id"],
            "template_id": template_id,
            "transcript_id": transcript_id,
            "cell_id": descriptor["descriptor_id"],
            "seed": stable_seed(base_seed, descriptor["descriptor_id"], template_id, transcript_id),
            "provenance": PROVENANCE,
        })
    return rows


def build_stage1(source_dir, output_dir, base_seed):
    catalog = source_catalog(source_dir)
    for axis in ("status", "career", "persona"):
        write_json(output_dir / OUTPUT_FILES[axis], expand_descriptors(catalog[axis], base_seed))
    write_json(output_dir / "_protocol" / "reconstruction.json", {
        "schema_version": 1,
        "provenance": PROVENANCE,
        "source_directory": str(source_dir),
        "base_seed": base_seed,
        "paper_content_identity": False,
        "notes": [
            "The paper's literal templates, complete descriptors, ordering, and seeds are unavailable.",
            "The canonical Stage 1 reconstruction originates from the repository's legacy files.",
            "The 40-persona subset and all added templates are explicit reconstruction choices.",
        ],
        "persona_keywords": PERSONA_KEYWORDS,
        "transcripts": TRANSCRIPTS,
    })
    return catalog


def load_stage1_catalog(stage1_dir):
    catalog = {}
    for axis in ("status", "career", "persona"):
        rows = load_json(stage1_dir / OUTPUT_FILES[axis])
        for row in rows:
            catalog.setdefault(row["descriptor_id"], {
                "axis": row["axis"], "descriptor_id": row["descriptor_id"],
                "keyword": row["keywords"], "trait": row["trait"],
                "description": row["description"] if row["template_id"] == 0 else None,
            })
    return catalog


def rank_descriptors(detection_csvs, catalog, model_name):
    totals = {}
    observed_pairs = {}
    required_fields = {
        "descriptor_id", "predicted_gender", "template_id", "transcript_id",
        "axis", "provenance", "model_name",
    }
    for detection_csv in detection_csvs:
        with open(detection_csv, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            missing_fields = required_fields - set(reader.fieldnames or [])
            if missing_fields:
                raise ValueError(f"{detection_csv}: missing fields: {', '.join(sorted(missing_fields))}")
            for row in reader:
                descriptor_id = row.get("descriptor_id")
                if descriptor_id not in catalog:
                    raise ValueError(f"{detection_csv}: unknown descriptor_id {descriptor_id}")
                descriptor = catalog[descriptor_id]
                if row["model_name"] != model_name:
                    raise ValueError(f"{detection_csv}: expected model_name {model_name}, found {row['model_name']}")
                if row["axis"] != descriptor["axis"] or row["provenance"] != PROVENANCE:
                    raise ValueError(f"{detection_csv}: metadata mismatch for {descriptor_id}")
                try:
                    pair = (int(row["template_id"]), int(row["transcript_id"]))
                except ValueError as exc:
                    raise ValueError(f"{detection_csv}: non-integer template/transcript ID") from exc
                pairs = observed_pairs.setdefault(descriptor_id, set())
                if pair in pairs:
                    raise ValueError(f"{detection_csv}: duplicate outcome for {descriptor_id} {pair}")
                pairs.add(pair)
                label = row.get("predicted_gender", "").lower()
                if label not in ("female", "male", "child"):
                    continue
                female, total = totals.get(descriptor_id, (0, 0))
                totals[descriptor_id] = (female + (label == "female"), total + 1)
    expected_pairs = set(product(range(10), range(10)))
    incomplete = [
        descriptor_id for descriptor_id in catalog
        if observed_pairs.get(descriptor_id) != expected_pairs
    ]
    if incomplete:
        raise ValueError(f"detections lack a complete 10 x 10 grid for {len(incomplete)} descriptors")
    unclassified = sorted(set(catalog) - set(totals))
    if unclassified:
        raise ValueError(f"detections lack classified outcomes for {len(unclassified)} descriptors")
    return {
        descriptor_id: female / total
        for descriptor_id, (female, total) in totals.items()
    }


def select_stage2(catalog, scores):
    selected = {"status": [], "career": [], "persona": []}
    for descriptor in catalog.values():
        selected[descriptor["axis"]].append(descriptor)
    selected["status"].sort(key=lambda item: item["descriptor_id"])
    for axis in ("career", "persona"):
        ranked = sorted(selected[axis], key=lambda item: (scores[item["descriptor_id"]], item["descriptor_id"]))
        if len(ranked) < 4:
            raise ValueError(f"at least four {axis} descriptors are required")
        selected[axis] = ranked[:2] + ranked[-2:]
    return selected


def composite_description(descriptors, template_id):
    cues = " ".join(item["description"].strip() for item in descriptors)
    return COMPOSITE_TEMPLATES[template_id].format(cues=cues)


def expand_cells(cells, base_seed):
    rows = []
    for descriptors, template_id, transcript_id in product(cells, range(10), range(10)):
        cell_id = "|".join(item["descriptor_id"] for item in descriptors)
        index = len(rows) + 1
        rows.append({
            "id": f"{index:07d}",
            "description": composite_description(descriptors, template_id),
            "trait": "+".join(item["trait"] for item in descriptors),
            "keywords": ";".join(f"{item['axis']}={item['keyword']}" for item in descriptors),
            "prompt_text": TRANSCRIPTS[transcript_id],
            "axis": "+".join(item["axis"] for item in descriptors),
            "descriptor_id": cell_id,
            "template_id": template_id,
            "transcript_id": transcript_id,
            "cell_id": cell_id,
            "seed": stable_seed(base_seed, cell_id, template_id, transcript_id),
            "provenance": PROVENANCE,
        })
    return rows


def build_stage2(stage1_dir, detection_csvs, output_dir, model_name, base_seed):
    catalog = load_stage1_catalog(stage1_dir)
    scores = rank_descriptors(detection_csvs, catalog, model_name)
    selected = select_stage2(catalog, scores)
    two_axis = []
    for left, right in (("status", "career"), ("status", "persona"), ("career", "persona")):
        two_axis.extend(tuple(items) for items in product(selected[left], selected[right]))
    multi_axis = [tuple(items) for items in product(selected["status"], selected["career"], selected["persona"])]
    if len(two_axis) != 32 or len(multi_axis) != 32:
        raise AssertionError("paper Stage 2 must contain 32 two-axis and 32 three-axis cells")
    for axis in ("status", "career", "persona"):
        write_json(output_dir / OUTPUT_FILES[axis], load_json(stage1_dir / OUTPUT_FILES[axis]))
    write_json(output_dir / OUTPUT_FILES["two_axis"], expand_cells(two_axis, base_seed))
    write_json(output_dir / OUTPUT_FILES["multi_axis"], expand_cells(multi_axis, base_seed))
    selection = {
        axis: [{**item, "female_probability": scores[item["descriptor_id"]]} for item in items]
        for axis, items in selected.items()
    }
    write_json(output_dir / "_protocol" / "stage2_selection.json", {
        "schema_version": 1, "provenance": PROVENANCE,
        "model_name": model_name, "detection_csvs": [str(path) for path in detection_csvs],
        "base_seed": base_seed, "paper_content_identity": False,
        "selection_rule": "two lowest and two highest classified female probabilities; descriptor_id breaks ties",
        "selected": selection,
    })


def main():
    parser = argparse.ArgumentParser(description="Build deterministic paper-aligned prompt reconstructions")
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage1 = subparsers.add_parser("stage1", help="build the 6,900 univariate prompts")
    stage1.add_argument("--source-dir", type=Path, default=Path("descriptions"))
    stage1.add_argument("--output-dir", type=Path, required=True)
    stage1.add_argument("--base-seed", type=int, default=0)
    stage2 = subparsers.add_parser("stage2", help="build 6,400 model-specific compositional prompts")
    stage2.add_argument("--stage1-dir", type=Path, required=True)
    stage2.add_argument("--detections", type=Path, nargs="+", required=True,
                        help="Stage 1 detection CSVs for status, career, and persona")
    stage2.add_argument("--output-dir", type=Path, required=True)
    stage2.add_argument("--model-name", required=True)
    stage2.add_argument("--base-seed", type=int, default=0)
    args = parser.parse_args()
    try:
        if args.command == "stage1":
            build_stage1(args.source_dir, args.output_dir, args.base_seed)
        else:
            build_stage2(args.stage1_dir, args.detections, args.output_dir, args.model_name, args.base_seed)
    except (OSError, csv.Error, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
