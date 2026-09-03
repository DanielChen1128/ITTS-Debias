#!/usr/bin/env python3
"""Recompute saved detections with the female/male binary decision rule."""

import argparse
from pathlib import Path

import pandas as pd

import analyze_gender


analyze_gender.pd = pd


def recompute_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"predicted_gender", "female_score", "male_score"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required columns: {', '.join(sorted(missing))}")

    result = frame.copy()
    female = pd.to_numeric(result["female_score"], errors="coerce")
    male = pd.to_numeric(result["male_score"], errors="coerce")
    valid = female.notna() & male.notna()
    result.loc[~valid, "predicted_gender"] = "unknown"
    result.loc[valid, "predicted_gender"] = female[valid].ge(male[valid]).map(
        {True: "female", False: "male"}
    )
    return result


def detection_files(paths):
    files = set()
    for path in paths:
        if path.is_file():
            files.add(path)
        elif path.is_dir():
            files.update(path.rglob("detection_results.csv"))
        else:
            raise FileNotFoundError(path)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Detection CSVs or directories")
    args = parser.parse_args()

    files = detection_files(args.paths)
    if not files:
        parser.error("no detection_results.csv files found")

    for path in files:
        frame = recompute_frame(pd.read_csv(path, dtype={"id": str}))
        frame.to_csv(path, index=False)
        analyze_gender.save_statistics(analyze_gender.compute_statistics(frame), path.parent)
        counts = frame["predicted_gender"].value_counts().to_dict()
        print(f"[INFO] Recomputed {path}: {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
