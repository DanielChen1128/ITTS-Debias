#!/usr/bin/env python3
"""Bootstrap 95% CIs for the pilot metrics from evaluate_pilot per-item output."""

import argparse
import json
from pathlib import Path

import numpy as np


def boot_ci(values, stat=np.mean, n=5000, seed=0):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n, len(values)))
    dist = stat(values[idx], axis=1)
    return float(stat(values)), float(np.percentile(dist, 2.5)), float(np.percentile(dist, 97.5))


def method_metrics(per_item):
    by = {}
    for r in per_item:
        by.setdefault(r["kind"], []).append(r)
    neutral_pf = np.mean([r["p_female"] for r in by.get("neutral", [])]) if by.get("neutral") else float("nan")

    stereo, gacc = [], []
    for r in by.get("grounded", []):
        sign = 1.0 if r["expected_gender"] == "female" else -1.0
        stereo.append(sign * (r["p_female"] - neutral_pf))
        gacc.append(float((r["p_female"] > 0.5) == (r["expected_gender"] == "female")))
    dev_shift = [abs(r["p_female"] - neutral_pf) for r in by.get("dev", [])]
    anchor = [float((r["p_female"] > 0.5) == (r["expected_gender"] == "female")) for r in by.get("anchor", [])]
    utmos = [r["utmos"] for r in per_item]
    wer = [r["wer"] for r in per_item if r["kind"] != "anchor"]
    return {
        "stereotyped_shift": stereo,
        "grounded_acc": gacc,
        "dev_abs_shift": dev_shift,
        "anchor_control_acc": anchor,
        "utmos": utmos,
        "wer": wer,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", default="results/pilot_std/evaluation.json")
    parser.add_argument("--output", default="results/pilot_std/bootstrap.json")
    args = parser.parse_args()

    report = json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
    metrics = ["stereotyped_shift", "grounded_acc", "dev_abs_shift",
               "anchor_control_acc", "utmos", "wer"]
    out = {}
    print(f"{'method':16s} " + " ".join(f"{m[:16]:>22s}" for m in metrics))
    for method, per_item in report["per_item"].items():
        mm = method_metrics(per_item)
        out[method] = {}
        cells = []
        for m in metrics:
            mean, lo, hi = boot_ci(mm[m])
            out[method][m] = {"mean": mean, "ci95": [lo, hi], "n": len(mm[m])}
            cells.append(f"{mean:6.3f}[{lo:5.2f},{hi:5.2f}]")
        print(f"{method:16s} " + " ".join(f"{c:>22s}" for c in cells))

    Path(args.output).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[bootstrap] wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
