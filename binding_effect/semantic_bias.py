#!/usr/bin/env python3
"""Optional text-encoder semantic bias (paper equation 7 and Cohen's d)."""

import argparse
import json
import math
import sys
from pathlib import Path


def cohens_d(first, second):
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Cohen's d requires at least two values per group")
    mean_a, mean_b = sum(first) / len(first), sum(second) / len(second)
    var_a = sum((x - mean_a) ** 2 for x in first) / (len(first) - 1)
    var_b = sum((x - mean_b) ** 2 for x in second) / (len(second) - 1)
    pooled = math.sqrt(((len(first) - 1) * var_a + (len(second) - 1) * var_b) / (len(first) + len(second) - 2))
    if pooled == 0:
        raise ValueError("Cohen's d is undefined for zero pooled variance")
    return (mean_a - mean_b) / pooled


def write_result(path, result):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Compute embedding gender bias and optional Cohen's d")
    parser.add_argument("--spec", required=True, help="JSON containing model_id, traits, and anchor lists")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        print("[ERROR] semantic analysis requires: pip install sentence-transformers", file=sys.stderr)
        return 2
    with open(args.spec, encoding="utf-8") as handle:
        spec = json.load(handle)
    required = ("model_id", "traits", "female_anchors", "male_anchors")
    if any(not spec.get(key) for key in required):
        print(f"[ERROR] semantic spec requires {', '.join(required)}", file=sys.stderr)
        return 2
    model = SentenceTransformer(spec["model_id"])
    anchors = spec["female_anchors"] + spec["male_anchors"]
    anchor_vectors = model.encode(anchors, normalize_embeddings=True)
    split = len(spec["female_anchors"])
    rows = []
    group_values = {}
    for trait in spec["traits"]:
        vector = model.encode([trait["text"]], normalize_embeddings=True)[0]
        delta = float(np.mean(anchor_vectors[:split] @ vector) - np.mean(anchor_vectors[split:] @ vector))
        rows.append({"text": trait["text"], "group": trait.get("group", ""), "delta": delta})
        group_values.setdefault(trait.get("group", ""), []).append(delta)
    result = {"traits": rows, "cohens_d": {}}
    for comparison in spec.get("comparisons", []):
        a, b = comparison
        result["cohens_d"][f"{a}_vs_{b}"] = cohens_d(group_values[a], group_values[b])
    write_result(args.output, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
