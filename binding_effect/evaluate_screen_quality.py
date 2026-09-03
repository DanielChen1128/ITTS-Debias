#!/usr/bin/env python3
"""Evaluate paired UTMOS and Whisper WER for frozen-screen finalists."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


class UTMOS:
    def __init__(self, device):
        self.device = device
        self.model = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True,
        ).eval().to(device)
        self.resamplers = {}

    @torch.no_grad()
    def score(self, wav, sample_rate):
        import torchaudio

        audio = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if audio.ndim > 1:
            audio = audio.mean(-1)
        if sample_rate != 16000:
            self.resamplers.setdefault(
                sample_rate, torchaudio.transforms.Resample(sample_rate, 16000),
            )
            audio = self.resamplers[sample_rate](audio)
        return float(self.model(audio.unsqueeze(0).to(self.device), 16000))


class WhisperASR:
    def __init__(self, model_id, device):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer

        self.device = device
        self.processor = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).eval().to(device)
        self.normalize = BasicTextNormalizer()
        self.resamplers = {}

    @torch.no_grad()
    def wer(self, wav, sample_rate, reference):
        import torchaudio

        audio = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if audio.ndim > 1:
            audio = audio.mean(-1)
        if sample_rate != 16000:
            self.resamplers.setdefault(
                sample_rate, torchaudio.transforms.Resample(sample_rate, 16000),
            )
            audio = self.resamplers[sample_rate](audio)
        inputs = self.processor(
            audio.numpy(), sampling_rate=16000, return_tensors="pt",
            return_attention_mask=True,
        )
        token_ids = self.model.generate(
            inputs.input_features.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            max_new_tokens=128,
        )
        hypothesis = self.processor.batch_decode(token_ids, skip_special_tokens=True)[0]
        reference_words = self.normalize(reference).split()
        hypothesis_words = self.normalize(hypothesis).split()
        if not reference_words:
            return 0.0
        previous = list(range(len(hypothesis_words) + 1))
        for row, reference_word in enumerate(reference_words, 1):
            current = [row]
            for column, hypothesis_word in enumerate(hypothesis_words, 1):
                current.append(min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                ))
            previous = current
        return previous[-1] / len(reference_words)


def paired_interval(candidate, baseline, *, seed=20260818, samples=10000):
    """Return a percentile bootstrap interval for a paired mean difference."""
    differences = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("paired samples must be non-empty one-dimensional arrays")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, differences.size, size=(samples, differences.size))
    means = differences[indices].mean(axis=1)
    return {
        "mean": float(differences.mean()),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def select_stratified(rows_by_stratum, per_stratum, *, seed):
    """Select an equal, order-independent subset using stable hash ranks."""
    selected = []
    for stratum in sorted(rows_by_stratum):
        rows = rows_by_stratum[stratum]
        if len(rows) < per_stratum:
            raise ValueError(f"stratum {stratum!r} has only {len(rows)} rows")
        ranked = sorted(
            rows,
            key=lambda row: (
                hashlib.sha256(
                    f"{seed}\0{stratum}\0{row['id']}".encode("utf-8")
                ).hexdigest(),
                row["id"],
            ),
        )
        selected.extend({**row, "stratum": stratum} for row in ranked[:per_stratum])
    return selected


def load_prompt_spec(path, root):
    """Load a legacy prompt list or a hash-verified stratified manifest."""
    spec = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(spec, list):
        return spec
    if not isinstance(spec, dict) or "sources" not in spec or "selection" not in spec:
        raise ValueError("prompts must be a JSON list or a quality manifest")

    prompts = []
    for stratum, source in spec["sources"].items():
        source_path = root / source["prompt_path"]
        actual_hash = sha256_file(source_path)
        if actual_hash != source["sha256"]:
            raise ValueError(
                f"source hash mismatch for {source_path}: "
                f"expected {source['sha256']}, got {actual_hash}"
            )
        rows = json.loads(source_path.read_text(encoding="utf-8"))
        if "population" in source and len(rows) != source["population"]:
            raise ValueError(
                f"source population mismatch for {source_path}: "
                f"expected {source['population']}, got {len(rows)}"
            )
        source_filter = source.get("filter", {})
        if not isinstance(source_filter, dict):
            raise ValueError(f"source filter for {stratum!r} must be an object")
        rows = [
            row for row in rows
            if all(row.get(field) == value for field, value in source_filter.items())
        ]
        if "filtered_population" in source and len(rows) != source["filtered_population"]:
            raise ValueError(
                f"filtered source population mismatch for {source_path}: "
                f"expected {source['filtered_population']}, got {len(rows)}"
            )
        by_id = {row["id"]: row for row in rows}
        selected_ids = spec["selection"][stratum]
        if len(selected_ids) != len(set(selected_ids)):
            raise ValueError(f"duplicate selected id in stratum {stratum!r}")
        missing = [item_id for item_id in selected_ids if item_id not in by_id]
        if missing:
            raise ValueError(f"unknown ids in stratum {stratum!r}: {missing}")
        rule = spec.get("selection_rule")
        if rule:
            per_stratum = rule.get("pairs_by_stratum", {}).get(
                stratum, rule.get("pairs_per_stratum"),
            )
            if per_stratum is None:
                raise ValueError(f"selection rule has no pair count for {stratum!r}")
            expected = select_stratified(
                {stratum: rows}, per_stratum, seed=rule["seed"],
            )
            if selected_ids != [row["id"] for row in expected]:
                raise ValueError(f"selection does not match frozen rule for {stratum!r}")
        prompts.extend({**by_id[item_id], "stratum": stratum} for item_id in selected_ids)
    return prompts


def resolve_wav_path(root, method, prompt, method_directories=None):
    """Resolve manifest paths while retaining the original flat layout."""
    if method_directories is not None:
        try:
            directory = method_directories[method][prompt["stratum"]]
        except KeyError as error:
            raise ValueError(
                f"no WAV directory for method={method!r}, "
                f"stratum={prompt.get('stratum')!r}"
            ) from error
        return root / directory / f"{prompt['id']}.wav"
    return root / method / f"{prompt['id']}.wav"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--baseline", default="original")
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--asr-model", default="openai/whisper-tiny.en")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    prompt_spec = json.loads(args.prompts.read_text(encoding="utf-8"))
    method_directories = (
        prompt_spec.get("method_directories") if isinstance(prompt_spec, dict) else None
    )
    prompts = load_prompt_spec(args.prompts, args.root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utmos = UTMOS(device)
    asr = WhisperASR(args.asr_model, device)

    import soundfile as sf

    per_method = {}
    for method in (args.baseline, args.candidate):
        rows = []
        for prompt in prompts:
            wav_path = resolve_wav_path(args.root, method, prompt, method_directories)
            if not wav_path.is_file():
                raise FileNotFoundError(wav_path)
            wav, sample_rate = sf.read(str(wav_path), dtype="float32")
            rows.append({
                "id": prompt["id"],
                **({"stratum": prompt["stratum"]} if "stratum" in prompt else {}),
                "utmos": utmos.score(wav, sample_rate),
                "wer": asr.wer(wav, sample_rate, prompt["prompt_text"]),
            })
        per_method[method] = rows

    baseline = per_method[args.baseline]
    candidate = per_method[args.candidate]
    utmos_delta = paired_interval(
        [row["utmos"] for row in candidate], [row["utmos"] for row in baseline],
    )
    wer_delta = paired_interval(
        [row["wer"] for row in candidate], [row["wer"] for row in baseline],
    )
    report = {
        "protocol": (
            prompt_spec.get("protocol_id", "mini-algorithm-screen-v1")
            if isinstance(prompt_spec, dict)
            else "mini-algorithm-screen-v1"
        ),
        "prompt_manifest_sha256": sha256_file(args.prompts),
        "baseline": args.baseline,
        "candidate": args.candidate,
        "n": len(prompts),
        "means": {
            method: {
                metric: float(np.mean([row[metric] for row in rows]))
                for metric in ("utmos", "wer")
            }
            for method, rows in per_method.items()
        },
        "paired_deltas": {"utmos": utmos_delta, "wer": wer_delta},
        "margins": {"utmos": -0.10, "wer": 0.03},
        "noninferiority_passed": {
            "utmos": utmos_delta["ci95"][0] >= -0.10,
            "wer": wer_delta["ci95"][1] <= 0.03,
        },
        "per_item": per_method,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "per_item"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
