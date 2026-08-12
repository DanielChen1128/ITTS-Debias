#!/usr/bin/env python3
"""Verify frozen dataset and completed-experiment provenance hashes."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_hashes(base, expected, *, optional=False):
    errors = []
    for relative, digest in expected.items():
        path = base / relative
        if not path.is_file():
            if not optional:
                errors.append(f"missing: {path}")
            continue
        actual = _sha256(path)
        if actual != digest:
            errors.append(f"hash mismatch: {path} ({actual} != {digest})")
    return errors


def _check_wav_manifest(records_root, root, spec, *, require_binaries=False):
    errors = []
    path = records_root / spec["file"]
    if not path.is_file():
        return [f"missing: {path}"]
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != spec["wav_count"]:
        errors.append(f"WAV manifest count mismatch: {len(lines)} != {spec['wav_count']}")
    for line_number, line in enumerate(lines, 1):
        parts = line.split(None, 1)
        if len(parts) != 2 or len(parts[0]) != 64:
            errors.append(f"invalid WAV manifest line {line_number}: {path}")
            continue
        if not require_binaries:
            continue
        digest, relative = parts
        wav_path = root / relative
        if not wav_path.is_file():
            errors.append(f"missing: {wav_path}")
        elif _sha256(wav_path) != digest:
            errors.append(f"hash mismatch: {wav_path}")
    return errors


def verify(require_binaries=False):
    errors = []

    legacy_root = ROOT / "datasets" / "legacy-5900-v1"
    legacy = json.loads((legacy_root / "manifest.json").read_text(encoding="utf-8"))
    errors.extend(_check_hashes(legacy_root, legacy["files"]))

    description_files = [
        relative for relative in legacy["files"] if relative.startswith("descriptions/")
    ]
    record_count = 0
    for relative in description_files:
        record_count += len(json.loads((legacy_root / relative).read_text(encoding="utf-8")))
    if record_count != legacy["record_count"]:
        errors.append(
            f"legacy record count mismatch: {record_count} != {legacy['record_count']}"
        )

    # The compatibility inputs used by existing scripts must remain byte-identical
    # to the frozen experiment snapshot.
    for relative, digest in legacy["files"].items():
        if relative.startswith("data/"):
            errors.extend(_check_hashes(ROOT, {relative: digest}))

    records_root = ROOT / "experiment_records"
    experiment = json.loads((records_root / "manifest.json").read_text(encoding="utf-8"))
    errors.extend(_check_hashes(ROOT, experiment["input_sha256"]))
    errors.extend(_check_hashes(records_root, experiment["record_sha256"]))
    errors.extend(
        _check_hashes(
            ROOT,
            experiment["artifact_sha256"],
            optional=not require_binaries,
        )
    )
    errors.extend(
        _check_wav_manifest(
            records_root,
            ROOT,
            experiment["wav_manifest"],
            require_binaries=require_binaries,
        )
    )

    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-binaries",
        action="store_true",
        help="Fail when locally excluded artifact files are unavailable",
    )
    args = parser.parse_args()
    errors = verify(require_binaries=args.require_binaries)
    if errors:
        for error in errors:
            print(f"[ERROR] {error}")
        return 1
    print("[OK] frozen dataset and experiment provenance verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
