#!/usr/bin/env python3
"""Build deterministic nested female/male anchor-pair subsets and caches."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch


PAIR_FIELDS = ("context_id", "axes", "descriptor_ids", "template_id")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_pairs(rows):
    if len(rows) % 2:
        raise ValueError("anchor records must form complete pairs")
    pairs = []
    for start in range(0, len(rows), 2):
        pair = rows[start:start + 2]
        if {row.get("gender_label") for row in pair} != {"female", "male"}:
            raise ValueError(f"records {start} and {start + 1} are not a female/male pair")
        if any(pair[0].get(field) != pair[1].get(field) for field in PAIR_FIELDS):
            raise ValueError(f"records {start} and {start + 1} have mismatched pair metadata")
        pairs.append(pair)
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument(
        "--activation-cache", action="append", default=[], metavar="NAME=PATH",
        help="Aligned full-pool cache to subset, e.g. ar=full-ar.pt",
    )
    args = parser.parse_args()

    rows = json.loads(args.anchors.read_text(encoding="utf-8"))
    pairs = validate_pairs(rows)
    counts = sorted(set(args.pair_count))
    if counts[0] < 1 or counts[-1] > len(pairs):
        parser.error(f"pair counts must be between 1 and {len(pairs)}")

    order = list(range(len(pairs)))
    random.Random(args.seed).shuffle(order)
    caches = {}
    for specification in args.activation_cache:
        if "=" not in specification:
            parser.error("--activation-cache must use NAME=PATH")
        name, raw_path = specification.split("=", 1)
        cache_path = Path(raw_path)
        state = torch.load(cache_path, map_location="cpu", weights_only=True)
        activations = state.get("activations")
        if activations is None or activations.ndim != 2 or len(activations) != len(rows):
            parser.error(f"activation cache is not aligned with anchors: {cache_path}")
        if state.get("provenance", {}).get("training_sha256") != sha256(args.anchors):
            parser.error(f"activation cache provenance does not match anchors: {cache_path}")
        caches[name] = (cache_path, state)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol_id": "nested-anchor-pair-scaling-v1",
        "source": str(args.anchors),
        "source_sha256": sha256(args.anchors),
        "available_pairs": len(pairs),
        "pair_counts": counts,
        "seed": args.seed,
        "nested": True,
        "subsets": {},
    }
    for count in counts:
        selected_indices = order[:count]
        selected_rows = [row for index in selected_indices for row in pairs[index]]
        anchor_path = args.output_dir / f"pairs-{count}.json"
        payload = json.dumps(selected_rows, indent=2, ensure_ascii=True) + "\n"
        anchor_path.write_text(payload, encoding="utf-8")
        anchor_hash = sha256(anchor_path)
        subset = {
            "records": len(selected_rows),
            "pairs": count,
            "anchor_path": str(anchor_path),
            "anchor_sha256": anchor_hash,
            "source_pair_indices": selected_indices,
            "activation_caches": {},
        }
        row_indices = torch.tensor(
            [row_index for pair_index in selected_indices
             for row_index in (2 * pair_index, 2 * pair_index + 1)],
            dtype=torch.long,
        )
        for name, (cache_path, state) in caches.items():
            provenance = dict(state.get("provenance", {}))
            provenance.update({
                "training_sha256": anchor_hash,
                "parent_training_sha256": manifest["source_sha256"],
                "parent_activation_cache_sha256": sha256(cache_path),
                "anchor_pair_count": count,
                "anchor_subset_seed": args.seed,
            })
            output_cache = args.output_dir / f"pairs-{count}-{name}-activations.pt"
            torch.save({
                "provenance": provenance,
                "activations": state["activations"].index_select(0, row_indices),
            }, output_cache)
            subset["activation_caches"][name] = {
                "path": str(output_cache),
                "sha256": sha256(output_cache),
            }
        manifest["subsets"][str(count)] = subset

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "manifest": str(manifest_path),
        "pair_counts": counts,
        "available_pairs": len(pairs),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
