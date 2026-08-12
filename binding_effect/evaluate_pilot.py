#!/usr/bin/env python3
"""Stage 4 Go/No-Go pilot evaluation over generated wavs.

Scores every method directory under --root against three axes plus explicit
control, reusing existing checkpoints:

  * gender fairness  - audEERING w2v2 age-gender ONNX (analyze_gender.GenderDetector)
  * audio quality    - torch.hub tarepan/SpeechMOS utmos22_strong
  * semantic fidelity- Whisper ASR word error rate vs the prompt transcript
  * explicit control - anchor gender accuracy (should survive the main method)

Fairness is measured as the *stereotyped shift* of P(female) relative to that
method's own neutral baseline, so debiasing should shrink it toward zero without
harming UTMOS/WER or the explicit-anchor control accuracy.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def binary_p_female(female_score, male_score):
    denom = female_score + male_score
    return 0.5 if denom <= 0 else female_score / denom


def gender_scores(detector, wav, sr):
    """Return (p_female_binary, child_prob) from the audEERING ONNX model.

    Called directly (not via GenderDetector.detect_gender) because that method
    references numpy without importing it at module scope.
    """
    import librosa
    sig = np.asarray(wav, dtype=np.float32)
    if sig.ndim > 1:
        sig = sig.mean(axis=1)
    if sr != detector.sampling_rate:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=detector.sampling_rate)
    logits = np.asarray(detector.model(sig, detector.sampling_rate)["logits_gender"][0])
    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    female, male, child = float(probs[0]), float(probs[1]), float(probs[2])
    return binary_p_female(female, male), child


class UTMOS:
    def __init__(self, device):
        self.device = device
        self.m = torch.hub.load(
            "tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True
        ).eval().to(device)
        self._rs = {}

    @torch.no_grad()
    def score(self, wav, sr):
        import torchaudio
        w = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if w.ndim > 1:
            w = w.mean(-1)
        if sr != 16000:
            if sr not in self._rs:
                self._rs[sr] = torchaudio.transforms.Resample(sr, 16000)
            w = self._rs[sr](w)
        return float(self.m(w.unsqueeze(0).to(self.device), 16000))


class ASR:
    def __init__(self, model_id, device):
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        from transformers.models.whisper.english_normalizer import BasicTextNormalizer
        self.device = device
        self.proc = WhisperProcessor.from_pretrained(model_id)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_id).eval().to(device)
        self.norm = BasicTextNormalizer()
        self._rs = {}

    @torch.no_grad()
    def wer(self, wav, sr, ref):
        import torchaudio
        w = torch.from_numpy(np.asarray(wav, dtype=np.float32))
        if w.ndim > 1:
            w = w.mean(-1)
        if sr != 16000:
            if sr not in self._rs:
                self._rs[sr] = torchaudio.transforms.Resample(sr, 16000)
            w = self._rs[sr](w)
        inputs = self.proc(w.numpy(), sampling_rate=16000, return_tensors="pt",
                           return_attention_mask=True)
        ids = self.model.generate(
            inputs.input_features.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
            max_new_tokens=128,
        )
        hyp = self.proc.batch_decode(ids, skip_special_tokens=True)[0]
        r, h = self.norm(ref), self.norm(hyp)
        if not r.strip():
            return 0.0
        rw, hw = r.split(), h.split()
        prev = list(range(len(hw) + 1))
        for i, rwi in enumerate(rw, 1):
            cur = [i]
            for j, hwj in enumerate(hw, 1):
                cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (rwi != hwj)))
            prev = cur
        return prev[-1] / max(1, len(rw))


def score_method(method_dir, items, gender, utmos, asr):
    import soundfile as sf
    per_item = []
    for it in items:
        wav_path = Path(method_dir) / f"{it['id']}.wav"
        if not wav_path.is_file():
            continue
        wav, sr = sf.read(str(wav_path), dtype="float32")
        pf, child = gender_scores(gender, wav, sr)
        per_item.append({
            "id": it["id"], "kind": it["kind"],
            "expected_gender": it["expected_gender"], "explicit": it["explicit"],
            "p_female": pf, "child": child,
            "utmos": utmos.score(wav, sr),
            "wer": asr.wer(wav, sr, it["prompt_text"]),
        })
    return per_item


def aggregate(per_item):
    by_kind = defaultdict(list)
    for r in per_item:
        by_kind[r["kind"]].append(r)

    neutral = by_kind.get("neutral", [])
    neutral_pf = float(np.mean([r["p_female"] for r in neutral])) if neutral else float("nan")

    # Fairness: stereotyped shift of grounded items vs neutral baseline.
    grounded = by_kind.get("grounded", [])
    stereo_shifts, grounded_correct = [], []
    for r in grounded:
        sign = 1.0 if r["expected_gender"] == "female" else -1.0
        stereo_shifts.append(sign * (r["p_female"] - neutral_pf))
        grounded_correct.append(int((r["p_female"] > 0.5) == (r["expected_gender"] == "female")))

    dev = by_kind.get("dev", [])
    dev_abs_shift = [abs(r["p_female"] - neutral_pf) for r in dev]

    anchors = by_kind.get("anchor", [])
    anchor_correct = [
        int((r["p_female"] > 0.5) == (r["expected_gender"] == "female")) for r in anchors
    ]

    non_anchor = [r for r in per_item if r["kind"] != "anchor"]
    return {
        "n": len(per_item),
        "neutral_p_female": neutral_pf,
        "stereotyped_shift_mean": float(np.mean(stereo_shifts)) if stereo_shifts else float("nan"),
        "grounded_stereotype_acc": float(np.mean(grounded_correct)) if grounded_correct else float("nan"),
        "dev_abs_shift_mean": float(np.mean(dev_abs_shift)) if dev_abs_shift else float("nan"),
        "anchor_control_acc": float(np.mean(anchor_correct)) if anchor_correct else float("nan"),
        "utmos_mean": float(np.mean([r["utmos"] for r in per_item])),
        "wer_mean": float(np.mean([r["wer"] for r in non_anchor])) if non_anchor else float("nan"),
        "child_frac": float(np.mean([r["child"] > 0.5 for r in per_item])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", default="data/pilot.json")
    parser.add_argument("--root", default="results/pilot")
    parser.add_argument("--methods", nargs="+",
                        default=["original", "random", "gender_direction",
                                 "leace_bypass", "leace_nobypass"])
    parser.add_argument("--asr-model", default="openai/whisper-tiny.en")
    parser.add_argument("--output", default="results/pilot/evaluation.json")
    args = parser.parse_args()

    items = json.loads(Path(args.pilot).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    from analyze_gender import GenderDetector
    gender = GenderDetector()
    utmos = UTMOS(device)
    asr = ASR(args.asr_model, device)

    report = {"per_method": {}, "per_item": {}}
    for method in args.methods:
        mdir = Path(args.root) / method
        if not mdir.is_dir():
            print(f"[skip] missing {mdir}")
            continue
        per_item = score_method(mdir, items, gender, utmos, asr)
        report["per_item"][method] = per_item
        report["per_method"][method] = aggregate(per_item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    cols = ["stereotyped_shift_mean", "grounded_stereotype_acc", "dev_abs_shift_mean",
            "anchor_control_acc", "utmos_mean", "wer_mean", "neutral_p_female"]
    print(f"\n{'method':16s} " + " ".join(f"{c[:12]:>12s}" for c in cols))
    for method, agg in report["per_method"].items():
        print(f"{method:16s} " + " ".join(f"{agg[c]:12.3f}" for c in cols))
    print(f"\n[pilot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
