#!/usr/bin/env python3
"""Audit prompt schemas and emit a reproducibility manifest."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

PAPER_EXPECTED = {
    "description_career_bias.json": 2700,
    "descriptions_persona_bias.json": 4000,
    "descriptions_status_bias.json": 200,
    "descriptions_two_axis.json": 3200,
    "descriptions_multi_axis.json": 3200,
}
STAGE1_EXPECTED = {
    "description_career_bias.json": 2700,
    "descriptions_persona_bias.json": 4000,
    "descriptions_status_bias.json": 200,
}
CANONICAL_STAGE1_SHA256 = {
    "description_career_bias.json": "faffa2b10ce4cdc201054fe6470270028c2a9f373d5c9ba60fd521ff45ead1fd",
    "descriptions_persona_bias.json": "ee973d75bc12118f0165a658c64d6ff77ea49561b5bbbd0383254381a067469f",
    "descriptions_status_bias.json": "ff729a1bace361d8bed732a1d27ad58bd0646eaf270eaadd69670f86ed6f88b9",
}
REQUIRED_KEYS = {"id", "description", "trait", "keywords", "prompt_text"}
RECONSTRUCTION_KEYS = {
    "axis", "descriptor_id", "template_id", "transcript_id", "cell_id", "seed", "provenance",
}
STRUCTURE = {
    "description_career_bias.json": (27, 100),
    "descriptions_persona_bias.json": (40, 100),
    "descriptions_status_bias.json": (2, 100),
    "descriptions_two_axis.json": (32, 100),
    "descriptions_multi_axis.json": (32, 100),
}
EXPECTED_GRID = {(str(template), str(transcript)) for template in range(10) for transcript in range(10)}
RECONSTRUCTION_PROVENANCE = "paper-aligned-reconstruction-v1"
EXPECTED_AXES = {
    "description_career_bias.json": {"career"},
    "descriptions_persona_bias.json": {"persona"},
    "descriptions_status_bias.json": {"status"},
    "descriptions_two_axis.json": {"status+career", "status+persona", "career+persona"},
    "descriptions_multi_axis.json": {"status+career+persona"},
}
EXPECTED_CELL_AXES = {
    "description_career_bias.json": {"career": 27},
    "descriptions_persona_bias.json": {"persona": 40},
    "descriptions_status_bias.json": {"status": 2},
    "descriptions_two_axis.json": {"status+career": 8, "status+persona": 8, "career+persona": 16},
    "descriptions_multi_axis.json": {"status+career+persona": 32},
}


def audit_file(path):
    raw = path.read_bytes()
    errors = []
    protocol_errors = []
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        return {"file": str(path), "count": 0, "sha256": hashlib.sha256(raw).hexdigest(), "errors": [str(exc)]}
    if not isinstance(data, list):
        errors.append("root must be a list")
        data = []
    ids = []
    tuples = []
    cell_pairs = {}
    cell_axes = {}
    missing_metadata_count = 0
    invalid_metadata_count = 0
    known_protocol_file = path.name in STRUCTURE
    for index, item in enumerate(data):
        missing = REQUIRED_KEYS - set(item) if isinstance(item, dict) else REQUIRED_KEYS
        if missing:
            errors.append(f"item {index} missing: {', '.join(sorted(missing))}")
        elif not all(str(item[key]).strip() for key in REQUIRED_KEYS):
            errors.append(f"item {index} contains an empty required value")
        if isinstance(item, dict):
            ids.append(str(item.get("id", "")))
            tuples.append((str(item.get("description", "")), str(item.get("prompt_text", ""))))
            if known_protocol_file:
                missing_metadata = RECONSTRUCTION_KEYS - set(item)
                if missing_metadata:
                    missing_metadata_count += 1
                else:
                    cell_id = str(item["cell_id"])
                    pair = (str(item["template_id"]), str(item["transcript_id"]))
                    cell_pairs.setdefault(cell_id, []).append(pair)
                    cell_axes.setdefault(cell_id, set()).add(item["axis"])
                    if (
                        not cell_id or not str(item["descriptor_id"])
                        or cell_id != str(item["descriptor_id"])
                        or item["provenance"] != RECONSTRUCTION_PROVENANCE
                        or item["axis"] not in EXPECTED_AXES[path.name]
                        or type(item["template_id"]) is not int
                        or type(item["transcript_id"]) is not int
                        or type(item["seed"]) is not int
                        or not 0 <= item["template_id"] < 10
                        or not 0 <= item["transcript_id"] < 10
                    ):
                        invalid_metadata_count += 1
    if len(ids) != len(set(ids)):
        errors.append("duplicate ids")
    if missing_metadata_count:
        protocol_errors.append(f"{missing_metadata_count} items lack reconstruction metadata")
    if invalid_metadata_count:
        protocol_errors.append(f"{invalid_metadata_count} items contain invalid reconstruction metadata")
    duplicate_tuples = len(tuples) - len(set(tuples))
    structure = None
    if known_protocol_file and cell_pairs:
        expected_cells, expected_per_cell = STRUCTURE[path.name]
        incomplete = {
            cell_id: len(set(pairs)) for cell_id, pairs in cell_pairs.items()
            if len(pairs) != expected_per_cell or set(pairs) != EXPECTED_GRID
        }
        if len(cell_pairs) != expected_cells:
            protocol_errors.append(f"expected {expected_cells} cells, found {len(cell_pairs)}")
        if incomplete:
            protocol_errors.append(f"{len(incomplete)} cells lack a unique 10 x 10 template/transcript product")
        actual_cell_axes = {}
        for axes in cell_axes.values():
            if len(axes) != 1:
                protocol_errors.append("a cell contains mixed axis metadata")
                continue
            axis = next(iter(axes))
            actual_cell_axes[axis] = actual_cell_axes.get(axis, 0) + 1
        if actual_cell_axes != EXPECTED_CELL_AXES[path.name]:
            protocol_errors.append(
                f"expected cell topology {EXPECTED_CELL_AXES[path.name]}, found {actual_cell_axes}"
            )
        structure = {
            "expected_cells": expected_cells,
            "actual_cells": len(cell_pairs),
            "expected_prompts_per_cell": expected_per_cell,
            "actual_cells_by_axis": actual_cell_axes,
            "incomplete_cells": incomplete,
        }
    return {
        "file": str(path), "count": len(data), "sha256": hashlib.sha256(raw).hexdigest(),
        "duplicate_description_transcript_tuples": duplicate_tuples,
        "structure": structure, "errors": errors, "protocol_errors": protocol_errors,
    }


def build_manifest(description_dir):
    files = [audit_file(path) for path in sorted(description_dir.glob("*.json"))]
    actual = {Path(item["file"]).name: item["count"] for item in files}
    missing = {name: max(0, expected - actual.get(name, 0)) for name, expected in PAPER_EXPECTED.items()}
    missing = {name: count for name, count in missing.items() if count}
    count_mismatches = {
        name: {"expected": expected, "actual": actual.get(name, 0)}
        for name, expected in PAPER_EXPECTED.items()
        if actual.get(name, 0) != expected
    }
    unexpected = sorted(set(actual) - set(PAPER_EXPECTED))
    schema_valid = all(not item["errors"] for item in files)
    structure_valid = all(not item.get("protocol_errors") for item in files)
    count_alignment = not count_mismatches and not unexpected
    stage1_files = [item for item in files if Path(item["file"]).name in STAGE1_EXPECTED]
    stage1_count_alignment = all(actual.get(name) == expected for name, expected in STAGE1_EXPECTED.items())
    stage1_structure_valid = all(not item["errors"] and not item.get("protocol_errors") for item in stage1_files)
    stage1_alignment = stage1_count_alignment and stage1_structure_valid
    actual_hashes = {Path(item["file"]).name: item["sha256"] for item in stage1_files}
    canonical_hashes_match = actual_hashes == CANONICAL_STAGE1_SHA256
    # The paper provides counts but no reference hashes for identity checking.
    # Equal counts alone therefore cannot establish exact prompt reproduction.
    content_identity_verified = False
    return {
        "schema_version": 5,
        "paper_expected_total": sum(PAPER_EXPECTED.values()),
        "paper_stage1_expected_total": sum(STAGE1_EXPECTED.values()),
        "repository_total": sum(actual.values()),
        "missing_from_complete_protocol": sum(missing.values()),
        "stage1_reconstruction_alignment": stage1_alignment,
        "canonical_stage1_hashes_match": canonical_hashes_match,
        "canonical_stage1_alignment": (
            stage1_alignment and set(actual) == set(STAGE1_EXPECTED) and canonical_hashes_match
        ),
        "paper_count_alignment": count_alignment,
        "reconstruction_protocol_alignment": count_alignment and schema_valid and structure_valid,
        "content_identity_verified": content_identity_verified,
        "exact_paper_reproduction_available": False,
        "missing_by_file": missing,
        "count_mismatches": count_mismatches,
        "unexpected_json_files": unexpected,
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="Audit prompt files and write a hash manifest")
    parser.add_argument("--descriptions", default="descriptions")
    parser.add_argument("--output", default="prompt_manifest.json")
    parser.add_argument("--strict-paper", action="store_true", help="Fail when original paper content identity is unverifiable")
    parser.add_argument("--strict-stage1", action="store_true",
                        help="Fail unless the canonical 6,900-prompt Stage 1 reconstruction aligns")
    parser.add_argument("--strict-reconstruction", action="store_true",
                        help="Fail unless reconstructed counts, metadata, and Cartesian structure align")
    args = parser.parse_args()
    manifest = build_manifest(Path(args.descriptions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    schema_errors = sum(bool(item["errors"]) for item in manifest["files"])
    print(f"Audited {manifest['repository_total']} prompts; missing {manifest['missing_from_complete_protocol']} from complete protocol")
    if schema_errors:
        print(f"[ERROR] {schema_errors} files failed schema validation", file=sys.stderr)
        return 1
    if args.strict_paper and not manifest["exact_paper_reproduction_available"]:
        print("[ERROR] exact paper reproduction is unavailable", file=sys.stderr)
        return 2
    if args.strict_stage1 and not manifest["canonical_stage1_alignment"]:
        print("[ERROR] canonical Stage 1 alignment failed", file=sys.stderr)
        return 2
    if args.strict_reconstruction and not manifest["reconstruction_protocol_alignment"]:
        print("[ERROR] reconstruction protocol alignment failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
