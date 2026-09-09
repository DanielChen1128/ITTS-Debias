#!/usr/bin/env python3
"""Rank B200 extension screen conditions from gender detection CSV files."""

import argparse
import csv
import json
from pathlib import Path


STRATA = ("status", "career", "persona", "two-axis", "three-axis")


def female_rate(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty detection file: {path}")
    female = 0
    for row in rows:
        label = (row.get("binary_gender") or row.get("predicted_gender") or row.get("gender") or "").lower()
        if label not in {"female", "male"}:
            raise ValueError(f"missing binary female/male label in {path}")
        female += label == "female"
    return female / len(rows), len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summaries = []
    for condition in sorted(path for path in args.analysis_root.iterdir() if path.is_dir()):
        rates = {}
        counts = {}
        missing = []
        for stratum in STRATA:
            path = condition / "screen" / ("stage1" if stratum in STRATA[:3] else "stage2") / stratum / "detection_results.csv"
            if not path.is_file():
                missing.append(str(path))
                continue
            rates[stratum], counts[stratum] = female_rate(path)
        if missing:
            continue
        total = sum(counts.values())
        global_rate = sum(rates[name] * counts[name] for name in STRATA) / total
        errors = [abs(rates[name] - 0.5) for name in STRATA]
        summaries.append({
            "condition": condition.name,
            "prompts": total,
            "female_rate": global_rate,
            "global_calibration_error": abs(global_rate - 0.5),
            "mean_stratum_error": sum(errors) / len(errors),
            "worst_stratum_error": max(errors),
            "selection_score": abs(global_rate - 0.5) + sum(errors) / len(errors),
            "saturated": global_rate in {0.0, 1.0},
            "stratum_female_rates": rates,
        })
    summaries.sort(key=lambda item: (item["saturated"], item["selection_score"], item["condition"]))
    payload = {"conditions": len(summaries), "ranking": summaries}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
