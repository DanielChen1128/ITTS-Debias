#!/usr/bin/env python3
"""
Unified TTS Generation Script for BindingBias
The Binding Effect: Multi-Dimensional Gender Bias in Instruction TTS

Supports 4 TTS models:
  - parler-large: Parler-TTS Large model
  - parler-mini: Parler-TTS Mini model
  - promptttspp: PromptTTS++
  - voxinstruct: VoxInstruct

Usage:
  python generate_wav.py --model parler-large --json descriptions/descriptions_persona_bias.json --output results/parler_large/
  python generate_wav.py --model parler-mini --json descriptions/descriptions_multi_axis.json --output results/parler_mini/
  python generate_wav.py --model promptttspp --json descriptions/descriptions_two_axis.json --output results/promptttspp/
  python generate_wav.py --model voxinstruct --json descriptions/descriptions_sdo_bias.json --output results/voxinstruct/

Input JSON format:
  [
    {
      "id": "0001",
      "description": "Style/voice description...",
      "trait": "trait type",
      "keywords": "keywords",
      "prompt_text": "Text to synthesize"
    },
    ...
  ]

Output:
  - WAV files named by ID: 0001.wav, 0002.wav, etc.
"""

import os
import sys
import json
import argparse
import hashlib
import re
import random
import importlib.util
from pathlib import Path
from typing import List, Dict

import numpy as np

SCRIPT_DIR = Path(__file__).parent


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "parler-large": {
        "model_id": "parler-tts/parler-tts-large-v1",
        # The Large decoder supports only 4096 positions; cap short-form
        # synthesis well below that limit to avoid overrun.
        "max_new_tokens": 2600,
        "temperature": 0.8,
    },
    "parler-mini": {
        "model_id": "parler-tts/parler-tts-mini-v1",
        "max_new_tokens": 2048,
        "temperature": 0.8,
    },
    "promptttspp": {
        "model_id": None,
        "noise_scale": 0.5,
    },
    "voxinstruct": {
        "model_id": None,
        "max_length": 1000,
        "temperature": 1.0,
    }
}

ENV_MODEL_IDS = {
    "parler-large": "BINDING_PARLER_LARGE_MODEL",
    "parler-mini": "BINDING_PARLER_MINI_MODEL",
    "promptttspp": "BINDING_PROMPTTTSPP_MODEL",
    "voxinstruct": "BINDING_VOXINSTRUCT_MODEL",
}
ENV_BACKENDS = {
    "promptttspp": "BINDING_PROMPTTTSPP_PATH",
    "voxinstruct": "BINDING_VOXINSTRUCT_PATH",
}


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_voxinstruct_steering(
    steering, stage, representation, width, model_dir, checkpoint_path,
):
    metadata = steering.metadata
    if metadata.get("representation") != representation:
        raise ValueError(f"VoxInstruct {stage.upper()} steering representation is incompatible")
    if steering.offset.numel() != width:
        raise ValueError(
            f"VoxInstruct {stage.upper()} steering width differs from conditioning width"
        )
    artifact_checkpoint_sha256 = metadata.get("checkpoint_sha256")
    if artifact_checkpoint_sha256:
        if _file_sha256(checkpoint_path) != artifact_checkpoint_sha256:
            raise ValueError(
                f"VoxInstruct {stage.upper()} artifact checkpoint does not match the loaded checkpoint"
            )
    else:
        artifact_model = metadata.get("model_id")
        if artifact_model and Path(artifact_model).resolve() != Path(model_dir).resolve():
            raise ValueError(
                f"VoxInstruct {stage.upper()} artifact was fit for {artifact_model}, not {model_dir}"
            )


def resolve_model_config(model_name, model_id=None, backend_path=None, config_path=None, model_revision=None):
    """Resolve CLI, environment, config-file, then public defaults."""
    file_config = {}
    if config_path:
        with open(config_path, encoding="utf-8") as handle:
            all_config = json.load(handle)
        if not isinstance(all_config, dict):
            raise ValueError("config root must be an object")
        models = all_config.get("models", all_config)
        if not isinstance(models, dict) or not isinstance(models.get(model_name, {}), dict):
            raise ValueError(f"config for {model_name} must be an object")
        file_config = models.get(model_name, {})
    config = dict(MODEL_CONFIGS[model_name])
    config["model_id"] = (
        model_id or os.getenv(ENV_MODEL_IDS[model_name])
        or file_config.get("model_id") or config.get("model_id")
    )
    if model_name in ENV_BACKENDS:
        config["backend_path"] = (
            backend_path or os.getenv(ENV_BACKENDS[model_name])
            or file_config.get("backend_path")
        )
    if model_name in ("parler-large", "parler-mini"):
        config["revision"] = model_revision or file_config.get("revision")
    return config


def preflight(model_name, config, json_path=None):
    errors = []
    if json_path:
        try:
            with open(json_path, encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list) or any(
                not isinstance(item, dict)
                or not all(item.get(key) for key in ("id", "description", "prompt_text"))
                for item in data
            ):
                errors.append("input JSON must be a list with non-empty id, description, and prompt_text")
            elif len({str(item["id"]) for item in data}) != len(data):
                errors.append("input JSON contains duplicate ids that would overwrite WAV files")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read input JSON: {exc}")
    if not config.get("model_id"):
        errors.append(
            f"model checkpoint is required via --model-id, {ENV_MODEL_IDS[model_name]}, or --config"
        )
    elif model_name not in ("parler-large", "parler-mini") and not Path(config["model_id"]).is_dir():
        errors.append(f"external checkpoint root is not a directory: {config['model_id']}")
    backend_path = None
    if model_name in ENV_BACKENDS:
        backend = config.get("backend_path")
        if not backend:
            errors.append(
                f"backend checkout is required via --backend-path, {ENV_BACKENDS[model_name]}, or --config"
            )
        elif not Path(backend).is_dir():
            errors.append(f"backend checkout does not exist: {backend}")
        else:
            backend_path = Path(backend)
    required_packages = {
        "parler-large": ("torch", "transformers", "parler_tts", "numpy", "soundfile", "tqdm"),
        "parler-mini": ("torch", "transformers", "parler_tts", "numpy", "soundfile", "tqdm"),
        "promptttspp": ("torch", "hydra", "omegaconf", "nltk", "g2p_en", "torchaudio", "numpy", "tqdm"),
        "voxinstruct": ("torch", "transformers", "vocos", "encodec", "torchaudio", "fairseq", "numpy", "tqdm"),
    }
    for package in required_packages[model_name]:
        try:
            available = importlib.util.find_spec(package) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            available = False
        if not available:
            errors.append(f"Python dependency '{package}' is not installed")
    required_paths = []
    model_path = Path(config["model_id"]) if config.get("model_id") else None
    if model_name == "promptttspp" and backend_path and model_path:
        required_paths = [
            backend_path / "promptttspp" / "text" / "eng.py",
            backend_path / "promptttspp" / "utils" / "model.py",
            backend_path / "egs" / "proposed" / "bin" / "conf" / "demo.yaml",
            model_path / "checkpoint" / "proposed" / "last.ckpt",
            model_path / "checkpoint" / "bigvgan_f0_full" / "last.ckpt",
            model_path / "checkpoint" / "stats.yaml",
        ]
    elif model_name == "voxinstruct" and backend_path and model_path:
        required_paths = [
            backend_path / "model" / "ar.py", backend_path / "model" / "nar.py",
            backend_path / "utils" / "utils.py", backend_path / "utils" / "extract_hubert.py",
            backend_path / "configs" / "train_ar.yaml", backend_path / "configs" / "train_nar.yaml",
            model_path / "voxinstruct-sft-checkpoint" / "ar_1800k.pyt",
            model_path / "voxinstruct-sft-checkpoint" / "nar_1800k.pyt",
            model_path / "google-mt5-base-checkpoint",
            model_path / "vocos-encodec-24khz" / "config.yaml",
            model_path / "vocos-encodec-24khz" / "pytorch_model.bin",
            model_path / "hubert-base-checkpoint" / "hubert_base_ls960.pt",
            model_path / "hubert-base-checkpoint" / "hubert_base_ls960_L9_km500.bin",
        ]
    for path in required_paths:
        if not path.exists():
            errors.append(f"required backend asset is missing: {path}")
    return errors


# ============================================================
# Helper Functions
# ============================================================

def _strip_control_chars(s: str) -> str:
    """Remove control characters from string"""
    return re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", " ", s)


def _is_valid_wav(path: str, min_seconds: float = 0.08) -> bool:
    """Check if WAV file exists and has valid audio"""
    if not os.path.isfile(path):
        return False
    try:
        import soundfile as snd
        info = snd.info(path)
        if info.samplerate is None or info.frames is None or info.frames <= 0:
            return False
        dur = info.frames / float(info.samplerate)
        return dur >= min_seconds
    except Exception:
        return False


def _trim_batched_audio_padding(audio):
    """Remove exact right padding returned when Parler generates a mixed-length batch."""
    nonzero = np.flatnonzero(audio != 0)
    return audio[: nonzero[-1] + 1] if len(nonzero) else audio


def _seed_item(item: Dict) -> int:
    """Seed stochastic backends from prompt metadata before one generation."""
    seed = item.get("seed")
    if seed is None:
        serialized = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        seed = int.from_bytes(hashlib.sha256(serialized.encode("utf-8")).digest()[:4], "big")
    seed = int(seed) % (2 ** 32)
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    return seed


def _seed_batch(items: List[Dict]) -> int:
    """Seed a batched generation deterministically from its ordered items."""
    serialized = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    seed = int.from_bytes(hashlib.sha256(serialized.encode("utf-8")).digest()[:4], "big")
    _seed_item({"seed": seed})
    return seed


def _voxinstruct_language_token(hparams, language="en"):
    """Translate a Vox source language ID to its offset model token."""
    return int(hparams.lang_mapping[language]) + 1


def _prepare_generation_manifest(output_dir, manifest, skip_existing, expected_wav_names=None):
    """Reject stale WAV reuse, then record a non-destructive run state."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "generation_manifest.json"
    run_state_path = output_dir / "generation_manifest.in_progress.json"
    wav_names = {path.name for path in output_dir.glob("*.wav")}
    expected_names = wav_names if expected_wav_names is None else set(expected_wav_names)
    unexpected_wavs = wav_names - expected_names
    if unexpected_wavs:
        examples = ", ".join(sorted(unexpected_wavs)[:3])
        raise ValueError(f"output contains WAVs outside the current prompt set: {examples}; use another directory")
    wavs_exist = bool(wav_names)
    if skip_existing and wavs_exist:
        provenance_path = manifest_path if manifest_path.exists() else run_state_path
        if not provenance_path.exists():
            raise ValueError(
                "output contains WAVs without generation_manifest.json or "
                "generation_manifest.in_progress.json; use --no-skip or another directory"
            )
        try:
            existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot validate existing generation manifest: {exc}") from exc
        identity_keys = (
            "model", "model_id", "resolved_config", "input_sha256", "prompt_count",
            "intervention", "batch_size", "batch_seed_strategy",
        )
        # Manifests created before batch support are unambiguously sequential B=1.
        defaults = {
            "batch_size": 1,
            "batch_seed_strategy": "per-item metadata seed; canonical prompt hash fallback",
        }
        existing = {**defaults, **existing}
        expected = {**defaults, **manifest}
        # Full-strength LEACE manifests predate the explicit strength field.
        for candidate in (existing, expected):
            intervention = candidate.get("intervention")
            if intervention and intervention.get("method") == "leace":
                candidate["intervention"] = {"strength": 1.0, **intervention}
            intervention = candidate.get("intervention")
            if intervention:
                intervention = dict(intervention)
                if intervention.get("artifact_sha256"):
                    intervention.pop("artifact_path", None)
                artifacts = intervention.get("artifacts")
                if artifacts:
                    intervention["artifacts"] = {
                        stage: {
                            key: value for key, value in dict(artifact).items()
                            if key != "artifact_path" or not artifact.get("artifact_sha256")
                        }
                        for stage, artifact in artifacts.items()
                    }
                candidate["intervention"] = intervention
        mismatches = [key for key in identity_keys if existing.get(key) != expected.get(key)]
        if mismatches:
            raise ValueError(f"existing WAV provenance differs in: {', '.join(mismatches)}; use --no-skip or another directory")
    _write_json_atomic(run_state_path, manifest)
    return manifest_path, run_state_path


def _write_json_atomic(path, payload):
    """Replace a JSON record only after its complete content is on disk."""
    path = Path(path)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    temporary.replace(path)


def _finish_generation_manifest(manifest_path, run_state_path, manifest, failed):
    """Publish only successful runs, preserving any prior complete manifest."""
    manifest["status"] = "complete" if not failed else "incomplete"
    manifest["failed_count"] = failed
    _write_json_atomic(run_state_path, manifest)
    if not failed:
        _write_json_atomic(manifest_path, manifest)
        Path(run_state_path).unlink(missing_ok=True)


# ============================================================
# Parler-TTS Generator
# ============================================================

class ParlerTTSGenerator:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        config,
        skip_existing: bool = True,
        leace_artifact: str = None,
        bypass_explicit: bool = True,
        batch_size: int = 1,
        intervention_strength: float = 1.0,
        intervention_mode: str = "token",
        steering_artifact: str = None,
    ):
        global torch, np, sf, tqdm
        import torch
        import numpy as np
        import soundfile as sf
        from tqdm import tqdm
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bypass_explicit = bypass_explicit
        self.batch_size = batch_size
        self.intervention_strength = intervention_strength
        self.intervention_mode = intervention_mode
        self.leace_artifact_sha256 = None
        self.steering_artifact_sha256 = None
        self.intervention_method = None
        self.intervention_records = {}

        # Get config
        self.config = config
        self.model_dir = self.config["model_id"]

        print(f"[INFO] Loading {model_name} from {self.model_dir}")
        print(f"[INFO] Using device: {self.device}")

        # Load model
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer

        revision_kwargs = {"revision": self.config["revision"]} if self.config.get("revision") else {}

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(
            self.model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            **revision_kwargs,
        ).to(self.device)

        try:
            self.model.set_default_attn_implementation("sdpa")
        except Exception:
            pass

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, **revision_kwargs)
        if leace_artifact and steering_artifact:
            raise ValueError("intervention artifacts are mutually exclusive")
        self.leace_eraser = None
        if leace_artifact:
            from debias.leace import LeaceEraser

            self.leace_eraser = LeaceEraser.load(leace_artifact, map_location=self.device)
            self.intervention_method = "leace"
            self.leace_artifact_sha256 = hashlib.sha256(Path(leace_artifact).read_bytes()).hexdigest()
            artifact_model = self.leace_eraser.metadata.get("model_id")
            if artifact_model and artifact_model != self.model_dir:
                raise ValueError(
                    f"LEACE artifact was fit for {artifact_model}, not {self.model_dir}"
                )
        elif steering_artifact:
            from debias.steering import ConstantSteering

            if intervention_mode != "pooled-shift":
                raise ValueError("constant steering requires intervention_mode='pooled-shift'")
            self.leace_eraser = ConstantSteering.load(steering_artifact, map_location=self.device)
            self.intervention_method = "constant-steering"
            self.steering_artifact_sha256 = hashlib.sha256(
                Path(steering_artifact).read_bytes()
            ).hexdigest()
            metadata = self.leace_eraser.metadata
            if metadata.get("representation") != "projected_final_encoder_masked_mean":
                raise ValueError("steering representation is incompatible with generation")
            if self.leace_eraser.offset.numel() != self.model.decoder.config.hidden_size:
                raise ValueError("steering feature width differs from decoder conditioning width")
        item_manifest = self.output_dir / "intervention_items.json"
        if item_manifest.exists():
            existing = json.loads(item_manifest.read_text(encoding="utf-8"))
            self.intervention_records = {str(row["id"]): row for row in existing}

        # Resize embeddings if needed
        emb = self.model.get_input_embeddings()
        emb_vocab = int(emb.num_embeddings)
        tk_vocab = getattr(self.tokenizer, "vocab_size", None)
        if tk_vocab and tk_vocab > emb_vocab:
            print(f"[INFO] Resizing embeddings: {emb_vocab} -> {tk_vocab}")
            self.model.resize_token_embeddings(tk_vocab)

        print(f"[INFO] {model_name} loaded successfully")

    def safe_tokenize(self, text: str, is_description: bool = False):
        """Clean and tokenize text, clip out-of-range IDs"""
        text = (text or "").strip()
        text = _strip_control_chars(text)

        tok = self.tokenizer(
            text,
            return_tensors="pt",
            padding=False,
            return_attention_mask=True,
            truncation=True,
            max_length=getattr(self.tokenizer, "model_max_length", 4096),
        )

        ids = tok.input_ids
        attn = tok.attention_mask
        vocab_size = int(self.model.get_input_embeddings().num_embeddings)

        # Clip out-of-range tokens to UNK
        unk_id = getattr(self.tokenizer, "unk_token_id", 0)
        mask_bad = (ids < 0) | (ids >= vocab_size)
        ids = torch.where(mask_bad, torch.tensor(unk_id, dtype=ids.dtype), ids)

        return ids.to(self.device), attn.to(self.device)

    def safe_tokenize_batch(self, texts: List[str]):
        texts = [_strip_control_chars((text or "").strip()) for text in texts]
        tok = self.tokenizer(
            texts, return_tensors="pt", padding=True, return_attention_mask=True,
            truncation=True, max_length=getattr(self.tokenizer, "model_max_length", 4096),
        )
        ids = tok.input_ids
        vocab_size = int(self.model.get_input_embeddings().num_embeddings)
        unk_id = getattr(self.tokenizer, "unk_token_id", 0)
        ids = torch.where((ids < 0) | (ids >= vocab_size), torch.tensor(unk_id, dtype=ids.dtype), ids)
        return ids.to(self.device), tok.attention_mask.to(self.device)

    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"

        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True

        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")

        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False

        try:
            _seed_item(item)
            # Tokenize
            desc_ids, desc_attn = self.safe_tokenize(description, is_description=True)
            prompt_ids, prompt_attn = self.safe_tokenize(prompt_text, is_description=False)

            # Generate
            with torch.no_grad():
                generation_kwargs = {
                    "prompt_input_ids": prompt_ids,
                    "prompt_attention_mask": prompt_attn,
                    "max_new_tokens": self.config["max_new_tokens"],
                    "temperature": self.config["temperature"],
                    "do_sample": True,
                }
                if self.leace_eraser is None:
                    generation = self.model.generate(
                        input_ids=desc_ids,
                        attention_mask=desc_attn,
                        **generation_kwargs,
                    )
                else:
                    from debias.parler import generate_with_eraser

                    generation, intervention = generate_with_eraser(
                        self.model,
                        self.leace_eraser,
                        description,
                        desc_ids,
                        desc_attn,
                        bypass_explicit=self.bypass_explicit,
                        intervention_strength=self.intervention_strength,
                        intervention_mode=self.intervention_mode,
                        return_intervention=True,
                        **generation_kwargs,
                    )
                    self.intervention_records[str(item_id)] = {
                        "id": str(item_id),
                        **intervention,
                        "leace_artifact_sha256": self.leace_artifact_sha256,
                        "steering_artifact_sha256": self.steering_artifact_sha256,
                        "intervention_method": self.intervention_method,
                    }

            # Save - convert to float32 for soundfile compatibility
            audio_arr = generation.cpu().numpy().squeeze()
            if audio_arr.dtype == np.float16:
                audio_arr = audio_arr.astype(np.float32)
            sf.write(str(output_path), audio_arr, self.model.config.sampling_rate)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            return False

    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        if getattr(self, "batch_size", 1) > 1:
            return self._batch_generate_batched(data)
        success_count = 0
        skip_count = 0
        fail_count = 0

        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"

            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue

            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1

        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")
        if getattr(self, "intervention_records", None):
            _write_json_atomic(
                self.output_dir / "intervention_items.json",
                [self.intervention_records[key] for key in sorted(self.intervention_records)],
            )
        return fail_count

    def _batch_generate_batched(self, data: List[Dict]):
        """Generate groups of prompts; batch RNG is intentionally batch-dependent."""
        pending = []
        skip_count = 0
        for item in data:
            output_path = self.output_dir / f"{item.get('id', 'unknown')}.wav"
            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
            else:
                pending.append(item)
        success_count = 0
        fail_count = 0
        for start in tqdm(range(0, len(pending), self.batch_size), desc=f"Generating with {self.model_name}"):
            items = pending[start : start + self.batch_size]
            if any(not item.get("description") or not item.get("prompt_text") for item in items):
                for item in items:
                    if self.generate_single(item):
                        success_count += 1
                    else:
                        fail_count += 1
                continue
            try:
                _seed_batch(items)
                descriptions = [item["description"] for item in items]
                desc_ids, desc_attn = self.safe_tokenize_batch(descriptions)
                prompt_ids, prompt_attn = self.safe_tokenize_batch([item["prompt_text"] for item in items])
                kwargs = {
                    "prompt_input_ids": prompt_ids,
                    "prompt_attention_mask": prompt_attn,
                    "max_new_tokens": self.config["max_new_tokens"],
                    "temperature": self.config["temperature"],
                    "do_sample": True,
                }
                with torch.no_grad():
                    if self.leace_eraser is None:
                        generation = self.model.generate(
                            input_ids=desc_ids, attention_mask=desc_attn, **kwargs,
                        )
                    else:
                        from debias.parler import generate_with_eraser_batch
                        generation, interventions = generate_with_eraser_batch(
                            self.model, self.leace_eraser, descriptions, desc_ids, desc_attn,
                            bypass_explicit=self.bypass_explicit,
                            intervention_strength=self.intervention_strength,
                            intervention_mode=self.intervention_mode,
                            return_intervention=True, **kwargs,
                        )
                        for item, intervention in zip(items, interventions):
                            self.intervention_records[str(item["id"])] = {
                                "id": str(item["id"]), **intervention,
                                "leace_artifact_sha256": self.leace_artifact_sha256,
                                "steering_artifact_sha256": self.steering_artifact_sha256,
                                "intervention_method": self.intervention_method,
                            }
                if generation.ndim != 2 or generation.shape[0] != len(items):
                    raise ValueError(f"expected batched audio [{len(items)}, samples], got {tuple(generation.shape)}")
                for item, audio in zip(items, generation.cpu().numpy()):
                    audio = _trim_batched_audio_padding(audio)
                    if audio.dtype == np.float16:
                        audio = audio.astype(np.float32)
                    sf.write(str(self.output_dir / f"{item['id']}.wav"), audio, self.model.config.sampling_rate)
                    success_count += 1
            except Exception as exc:
                print(f"[WARN] Batch starting at {start} failed ({exc}); retrying items sequentially")
                for item in items:
                    if self.generate_single(item):
                        success_count += 1
                    else:
                        fail_count += 1
        print(f"\n[SUMMARY]\n  Success: {success_count}\n  Skipped: {skip_count}\n  Failed:  {fail_count}\n  Total:   {len(data)}")
        if self.intervention_records:
            _write_json_atomic(
                self.output_dir / "intervention_items.json",
                [self.intervention_records[key] for key in sorted(self.intervention_records)],
            )
        return fail_count


# ============================================================
# PromptTTS++ Generator
# ============================================================

class PromptTTSPPGenerator:
    def __init__(self, model_name: str, output_dir: str, config, skip_existing: bool = True):
        global torch, tqdm
        import torch
        from tqdm import tqdm
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.config = config
        model_dir = self.config['model_id']

        print(f"[INFO] Loading {model_name} from {model_dir}")
        print(f"[INFO] Using device: {self.device}")

        # Import dependencies
        import hydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        import nltk
        from g2p_en import G2p
        import torchaudio

        # Add promptttspp to path
        promptttspp_dir = Path(self.config["backend_path"]).resolve()
        if str(promptttspp_dir) not in sys.path:
            sys.path.insert(0, str(promptttspp_dir))

        from promptttspp.text.eng import symbols, text_to_sequence
        from promptttspp.utils.model import lowpass_filter

        # Download NLTK data
        nltk.download('averaged_perceptron_tagger_eng', quiet=True)

        # Initialize G2P and utilities
        self.g2p = G2p()
        self.symbols = symbols
        self.text_to_sequence = text_to_sequence
        self.lowpass_filter = lowpass_filter

        # Load configs using hydra - use initialize_config_dir for absolute paths
        from hydra import initialize_config_dir, compose
        config_dir = str(promptttspp_dir / "egs" / "proposed" / "bin" / "conf")
        initialize_config_dir(config_dir=config_dir, version_base=None)
        cfg = compose(config_name="demo")

        # Override checkpoint paths
        cfg.model_ckpt_path = os.path.join(model_dir, "checkpoint", "proposed", "last.ckpt")
        cfg.vocoder_ckpt_path = os.path.join(model_dir, "checkpoint", "bigvgan_f0_full", "last.ckpt")
        cfg.mel_stats_file = os.path.join(model_dir, "checkpoint", "stats.yaml")

        # Load model and vocoder using the helper function from original code
        self.model, self.vocoder = self._load_model(
            cfg.model, cfg.model_ckpt_path,
            cfg.vocoder, cfg.vocoder_ckpt_path
        )

        # Load mel transform and stats
        self.to_mel = instantiate(cfg.transforms)
        self.sample_rate = self.to_mel.sample_rate
        self.mel_stats = OmegaConf.load(cfg.mel_stats_file)

        print(f"[INFO] {model_name} loaded successfully")

    def _load_model(self, model_cfg, model_ckpt_path, vocoder_cfg, vocoder_ckpt_path):
        """Load model and vocoder (from original main.py)"""
        from hydra.utils import instantiate

        model = instantiate(model_cfg)
        model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu")["model"])
        model = model.to(self.device).eval()

        vocoder = instantiate(vocoder_cfg)
        vocoder.load_state_dict(torch.load(vocoder_ckpt_path, map_location="cpu")["generator"])
        vocoder = vocoder.to(self.device).eval()

        return model, vocoder

    def _synthesize_single(self, content_prompt: str, style_prompt: str):
        """Core synthesis function (from original main.py)"""
        # Convert text to phonemes
        with torch.no_grad():
            return self._synthesize_single_impl(content_prompt, style_prompt)

    def _synthesize_single_impl(self, content_prompt: str, style_prompt: str):
        phonemes = self.g2p(content_prompt)
        phonemes = [p if p not in [",", "."] else "sil" for p in phonemes]
        phonemes = [p for p in phonemes if p in self.symbols]
        phoneme_ids = self.text_to_sequence(" ".join(phonemes))
        phoneme_ids = torch.LongTensor(phoneme_ids)[None, :].to(self.device)

        # Generate mel-spectrogram with style prompt
        dec, log_cf0, vuv = self.model.infer(
            phoneme_ids,
            style_prompt=style_prompt,
            use_max=True,
            noise_scale=self.config.get('noise_scale', 0.5),
            return_f0=True,
        )

        # Post-process f0
        modfs = int(1.0 / (10 * 0.001))
        log_cf0 = self.lowpass_filter(log_cf0, modfs, cutoff=20)
        f0 = log_cf0.exp()
        f0[vuv < 0.5] = 0

        # Denormalize mel-spectrogram
        dec = dec * self.mel_stats["std"] + self.mel_stats["mean"]

        # Vocoder to waveform
        wav = self.vocoder(dec, f0).squeeze(1).cpu()
        return wav

    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"

        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True

        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")

        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False

        try:
            _seed_item(item)
            # Synthesize
            wav = self._synthesize_single(prompt_text, description)

            # Save wav file
            import torchaudio
            torchaudio.save(str(output_path), wav, sample_rate=self.sample_rate)
            return True

        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        success_count = 0
        skip_count = 0
        fail_count = 0

        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"

            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue

            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1

        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")
        return fail_count


# ============================================================
# VoxInstruct Generator
# ============================================================

class VoxInstructGenerator:
    def __init__(
        self, model_name: str, output_dir: str, config, skip_existing: bool = True,
        ar_steering_artifact: str = None, nar_steering_artifact: str = None,
        bypass_explicit: bool = True, intervention_strength: float = 1.0,
        ar_intervention_strength: float = None, nar_intervention_strength: float = None,
    ):
        global torch, tqdm
        import torch
        from tqdm import tqdm
        # Inference only: disable autograd globally so the AR generation loop
        # does not accumulate a computation graph (which caused CUDA OOM on
        # samples that never emit EOS and run to max_length).
        torch.set_grad_enabled(False)
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bypass_explicit = bypass_explicit
        self.intervention_strength = intervention_strength
        self.ar_intervention_strength = (
            intervention_strength if ar_intervention_strength is None else ar_intervention_strength
        )
        self.nar_intervention_strength = (
            intervention_strength if nar_intervention_strength is None else nar_intervention_strength
        )
        self.ar_steering = None
        self.nar_steering = None
        self.steering_artifact_sha256 = None
        self.intervention_records = {}

        self.config = config
        model_dir = self.config['model_id']
        backend_dir = Path(self.config["backend_path"]).resolve()
        sys.path.insert(0, str(backend_dir))

        print(f"[INFO] Loading {model_name} from {model_dir}")
        print(f"[INFO] Using device: {self.device}")

        # Import VoxInstruct dependencies
        from model.ar import VoxInstructAR
        from model.nar import VoxInstructNAR
        from utils.utils import get_config_from_file, sequence_mask, to_device, save_wav, top_k_top_p_filtering, convert_audio
        from utils.extract_hubert import HubertWithKmeans
        from transformers import AutoTokenizer
        from vocos import Vocos
        from encodec import EncodecModel
        import torchaudio

        # Save utility functions
        self.sequence_mask = sequence_mask
        self.to_device = to_device
        self.save_wav = save_wav
        self.top_k_top_p_filtering = top_k_top_p_filtering
        self.convert_audio = convert_audio

        # Load AR config and fix paths
        ar_config_path = str(backend_dir / "configs" / "train_ar.yaml")
        ar_hp = get_config_from_file(ar_config_path).hparams
        self.ar_hp = ar_hp

        # Fix relative paths to absolute paths
        checkpoint_dir = os.path.join(model_dir, "voxinstruct-sft-checkpoint")
        ar_hp.mt5_path = os.path.join(model_dir, "google-mt5-base-checkpoint")
        ar_hp.vocos_path = os.path.join(model_dir, "vocos-encodec-24khz")
        ar_hp.encodec_path = os.path.join(model_dir, "encodec-checkpoint")
        ar_hp.hubert_path = os.path.join(model_dir, "hubert-base-checkpoint")

        # Load AR model
        self.ar_model = VoxInstructAR(hp=ar_hp).to(self.device)
        ar_checkpoint_path = os.path.join(checkpoint_dir, "ar_1800k.pyt")
        ar_ckpt = torch.load(
            ar_checkpoint_path, map_location=self.device, weights_only=False,
        )
        self.ar_model.load_state_dict(ar_ckpt['model'], strict=True)
        self.ar_model.to(torch.bfloat16).eval()

        # Load NAR config and fix paths
        nar_config_path = str(backend_dir / "configs" / "train_nar.yaml")
        nar_hp = get_config_from_file(nar_config_path).hparams
        self.nar_hp = nar_hp

        # Fix relative paths to absolute paths
        nar_hp.mt5_path = os.path.join(model_dir, "google-mt5-base-checkpoint")
        nar_hp.vocos_path = ar_hp.vocos_path

        # Load NAR model
        self.nar_model = VoxInstructNAR(hp=nar_hp).to(self.device)
        nar_checkpoint_path = os.path.join(checkpoint_dir, "nar_1800k.pyt")
        nar_ckpt = torch.load(
            nar_checkpoint_path, map_location=self.device, weights_only=False,
        )
        self.nar_model.load_state_dict(nar_ckpt['model'], strict=True)
        self.nar_model.to(torch.bfloat16).eval()

        if ar_steering_artifact:
            from debias.steering import ConstantSteering
            from debias.voxinstruct import REPRESENTATIONS

            self.ar_steering = ConstantSteering.load(
                ar_steering_artifact, map_location=self.device,
            )
            _validate_voxinstruct_steering(
                self.ar_steering, "ar", REPRESENTATIONS["ar"], self.ar_hp.hidden_dim,
                model_dir, ar_checkpoint_path,
            )
        if nar_steering_artifact:
            from debias.steering import ConstantSteering
            from debias.voxinstruct import REPRESENTATIONS

            self.nar_steering = ConstantSteering.load(
                nar_steering_artifact, map_location=self.device,
            )
            _validate_voxinstruct_steering(
                self.nar_steering, "nar", REPRESENTATIONS["nar"], self.nar_hp.hidden_dim,
                model_dir, nar_checkpoint_path,
            )
        if ar_steering_artifact or nar_steering_artifact:
            self.steering_artifact_sha256 = {
                stage: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                for stage, path in (("ar", ar_steering_artifact), ("nar", nar_steering_artifact))
                if path
            }
        item_manifest = self.output_dir / "intervention_items.json"
        if item_manifest.exists():
            existing = json.loads(item_manifest.read_text(encoding="utf-8"))
            self.intervention_records = {str(row["id"]): row for row in existing}

        # Load Vocos vocoder
        vocos_path = ar_hp.vocos_path
        self.vocos = Vocos.from_hparams(f"{vocos_path}/config.yaml")
        state_dict = torch.load(
            f"{vocos_path}/pytorch_model.bin", map_location="cpu", weights_only=False,
        )
        encodec_parameters = {
            "feature_extractor.encodec." + key: value
            for key, value in self.vocos.feature_extractor.encodec.state_dict().items()
        }
        state_dict.update(encodec_parameters)
        self.vocos.load_state_dict(state_dict)
        self.vocos.to(self.device).eval()

        # Load Encodec - use default pretrained if local not available
        encodec_path = ar_hp.encodec_path
        if os.path.exists(encodec_path) and os.path.isdir(encodec_path):
            self.encodec = EncodecModel.encodec_model_24khz(pretrained=True, repository=Path(encodec_path)).to(self.device)
        else:
            print(f"[INFO] Local encodec not found, using default pretrained model")
            self.encodec = EncodecModel.encodec_model_24khz(pretrained=True).to(self.device)
        self.encodec.overlap = 0
        self.encodec.set_target_bandwidth(bandwidth=6.0)

        # Load HuBERT
        hubert_path = ar_hp.hubert_path
        self.hubert = HubertWithKmeans(
            checkpoint_path=f'{hubert_path}/hubert_base_ls960.pt',
            kmeans_path=f'{hubert_path}/hubert_base_ls960_L9_km500.bin',
            target_sample_hz=16000,
            seq_len_multiple_of=320
        )
        self.hubert.eval()
        self.hubert.to(self.device)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(ar_hp.mt5_path, local_files_only=True)

        # CFG parameters (from original inference)
        self.cfg_st_on_text = 1.5
        self.cfg_at_on_text = 3.0
        self.cfg_at_on_st = 1.5
        self.nar_iter_steps = 8

        print(f"[INFO] {model_name} loaded successfully")

    def generate_single(self, item: Dict) -> bool:
        """Generate a single WAV file from JSON item"""
        item_id = item.get("id", "unknown")
        output_path = self.output_dir / f"{item_id}.wav"

        # Skip if exists
        if self.skip_existing and _is_valid_wav(str(output_path)):
            return True

        description = item.get("description", "")
        prompt_text = item.get("prompt_text", "")

        if not description or not prompt_text:
            print(f"[WARN] Skipping {item_id}: missing description or prompt_text")
            return False

        try:
            _seed_item(item)
            # Prepare text instruction
            text = f"{description}. \"{prompt_text}\""
            text = text.strip().capitalize()

            # Tokenize text
            text_id = self.tokenizer(text, return_tensors="pt").input_ids.squeeze()
            actual_len = text_id.shape[0]
            if actual_len >= self.ar_hp.max_text_len:
                text_id = text_id[:self.ar_hp.max_text_len]
                text_id[self.ar_hp.max_text_len - 1] = 1  # <eos> for mt5
                actual_len = self.ar_hp.max_text_len

            # Pad text_id to max_text_len
            import torch.nn.functional as F
            text_id = F.pad(text_id, (0, self.ar_hp.max_text_len - actual_len), value=0)

            text_ids = text_id.unsqueeze(0).to(self.device)
            text_id_lens = torch.tensor([actual_len], device=self.device)
            text_attn_mask = self.sequence_mask(text_id_lens, max_len=self.ar_hp.max_text_len, device=self.device)

            # Source language ID 0 is English; token IDs reserve 0 for padding.
            lang_id = _voxinstruct_language_token(self.ar_hp, "en")
            seqs = torch.tensor([[self.ar_hp.bos_id, lang_id]], device=self.device)
            segment_ids = torch.tensor([[1, 1]], device=self.device)

            # AR inference
            pred_st_flag = True
            past_key_values_base = None
            past_key_values_free = None
            past_key_values_mask_st = None
            ar_text_encode = self.ar_model.encode_text(text_ids, text_attn_mask)
            bypassed = False
            if self.ar_steering is not None or self.nar_steering is not None:
                from debias.voxinstruct import has_explicit_gender_command

                bypassed = self.bypass_explicit and has_explicit_gender_command(description)
            if self.ar_steering is not None:
                from debias.voxinstruct import steer_text_states

                if not bypassed:
                    ar_text_encode = steer_text_states(
                        ar_text_encode, text_attn_mask, self.ar_steering,
                        self.ar_intervention_strength,
                    )
            free_text_encode = torch.zeros([1, self.ar_hp.max_text_len, self.ar_hp.hidden_dim], device=self.device).to(torch.bfloat16)

            for j in range(self.config['max_length']):
                ar_outputs_base, ar_text_encode = self.ar_model.predict(
                    input_ids=seqs,
                    segment_ids=segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    past_key_values=past_key_values_base,
                    text_encode=ar_text_encode,
                )
                cond_logits = ar_outputs_base['logits']
                past_key_values_base = ar_outputs_base['past_key_values']

                ar_outputs_free, _ = self.ar_model.predict(
                    input_ids=seqs,
                    segment_ids=segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    past_key_values=past_key_values_free,
                    text_encode=free_text_encode,
                )
                free_logits = ar_outputs_free['logits']
                past_key_values_free = ar_outputs_free['past_key_values']

                if pred_st_flag:
                    logits = free_logits + (
                        cond_logits - free_logits
                    ) * self.cfg_st_on_text
                    logits[:, :, self.ar_hp.bos_id] = -1e5
                    logits[:, :, self.ar_hp.st_token_num + 1:self.ar_hp.eos_id] = -1e5
                    logits[:, :, self.ar_hp.eos_id + 1] = -1e5
                    filtered_logits = self.top_k_top_p_filtering(logits[0, -1, :], top_k=5, top_p=0.95, temperature=self.config.get('temperature', 0.8))
                else:
                    text_guided_logits = free_logits + (
                        cond_logits - free_logits
                    ) * self.cfg_at_on_text
                    ar_outputs_mask_st, _ = self.ar_model.predict(
                        input_ids=seqs,
                        segment_ids=segment_ids,
                        text_ids=text_ids,
                        text_attn_mask=text_attn_mask,
                        past_key_values=past_key_values_mask_st,
                        text_encode=ar_text_encode,
                        mask_st=True,
                    )
                    mask_st_logits = ar_outputs_mask_st['logits']
                    past_key_values_mask_st = ar_outputs_mask_st['past_key_values']
                    logits = mask_st_logits + (
                        text_guided_logits - mask_st_logits
                    ) * self.cfg_at_on_st
                    logits[:, :, self.ar_hp.bos_id] = -1e5
                    logits[:, :, 0:self.ar_hp.st_token_num + 1] = -1e5
                    logits[:, :, self.ar_hp.eos_id] = -1e5
                    filtered_logits = self.top_k_top_p_filtering(logits[0, -1, :], top_k=50, top_p=0.95, temperature=self.config.get('temperature', 0.8))

                probs = filtered_logits.softmax(dim=-1)
                samples = torch.multinomial(probs, 1).unsqueeze(1).to(self.device)

                seqs = torch.cat([seqs, samples], dim=1)

                if pred_st_flag:
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 1], dim=1)
                else:
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)

                # Switch from ST to AT
                if samples.item() == self.ar_hp.eos_id:
                    pred_st_flag = False
                    st_len = (segment_ids == 1).sum(dim=1)
                    past_key_values_base = None
                    past_key_values_free = None
                    past_key_values_mask_st = None

                if samples.item() == self.ar_hp.eos_id + 1:
                    break
                elif j == self.config['max_length'] - 1:
                    samples[:, :] = self.ar_hp.eos_id + 1
                    seqs = torch.cat([seqs, samples], dim=1)
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)
                    break

            # NAR inference
            b, t = seqs.shape
            from debias.voxinstruct import initialize_nar_sequences

            full_seqs = initialize_nar_sequences(
                seqs, st_len, self.ar_hp.at_res_num,
            )
            layer_index = torch.ones(size=[b,], device=self.device)
            nar_text_encode = self.nar_model.encode_text(text_ids, text_attn_mask)
            if self.nar_steering is not None and not bypassed:
                from debias.voxinstruct import steer_text_states

                nar_text_encode = steer_text_states(
                    nar_text_encode, text_attn_mask, self.nar_steering,
                    self.nar_intervention_strength,
                )

            for layer_idx in range(1, self.ar_hp.at_res_num):
                full_seqs = self.nar_model.predict(
                    full_seqs,
                    0,  # at_prompt_len (no audio prompt)
                    segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    layer_index=layer_idx,
                    iter_step=self.nar_iter_steps,
                    text_encode=nar_text_encode,
                )

            # Extract acoustic tokens
            st_len_val = st_len.item()
            full_seq = full_seqs[:, :, st_len_val:]
            full_seq = (full_seq - self.ar_hp.st_token_num - self.ar_hp.lang_num - 1)[:, :, :-1].clamp(0, 1023)

            # Decode with Vocos
            features = self.vocos.codes_to_features(full_seq[0])
            bandwidth_id = torch.tensor([2], device=self.device)
            wav = self.vocos.decode(features, bandwidth_id=bandwidth_id)
            wav = wav.cpu().squeeze().numpy()

            # Save
            import torchaudio
            torchaudio.save(str(output_path), torch.from_numpy(wav).unsqueeze(0), sample_rate=24000)
            if self.ar_steering is not None or self.nar_steering is not None:
                self.intervention_records[str(item_id)] = {
                    "id": str(item_id),
                    "bypass_mode": "regex" if self.bypass_explicit else "disabled",
                    "bypassed": bool(bypassed),
                    "eraser_applied": not bypassed,
                    "intervention_strength": {
                        "ar": self.ar_intervention_strength if self.ar_steering else None,
                        "nar": self.nar_intervention_strength if self.nar_steering else None,
                    },
                    "intervention_mode": "pooled-shift",
                    "intervention_method": "constant-steering",
                    "steering_artifact_sha256": self.steering_artifact_sha256,
                }
            return True

        except Exception as e:
            print(f"[ERROR] Failed to generate {item_id}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def batch_generate(self, data: List[Dict]):
        """Generate WAV files for all items in data"""
        success_count = 0
        skip_count = 0
        fail_count = 0

        for item in tqdm(data, desc=f"Generating with {self.model_name}"):
            item_id = item.get("id", "unknown")
            output_path = self.output_dir / f"{item_id}.wav"

            if self.skip_existing and _is_valid_wav(str(output_path)):
                skip_count += 1
                continue

            if self.generate_single(item):
                success_count += 1
            else:
                fail_count += 1

        print(f"\n[SUMMARY]")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count}")
        print(f"  Failed:  {fail_count}")
        print(f"  Total:   {len(data)}")
        if self.intervention_records:
            _write_json_atomic(
                self.output_dir / "intervention_items.json",
                [self.intervention_records[key] for key in sorted(self.intervention_records)],
            )
        return fail_count


# ============================================================
# Main Function
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Unified TTS Generation Script for CoP_bias',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parler-TTS Large
  python generate_wav.py --model parler-large --json descriptions/descriptions_sdo_bias.json --output

  # Parler-TTS Mini
  python generate_wav.py --model parler-mini --json descriptions/descriptions_sdo_bias.json  --output

  # PromptTTS++
  python generate_wav.py --model promptttspp --json descriptions/descriptions_sdo_bias.json --output

  # VoxInstruct
  python generate_wav.py --model voxinstruct --json descriptions/descriptions_sdo_bias.json --output
        """
    )

    parser.add_argument('--model', type=str, required=True,
                       choices=['parler-large', 'parler-mini', 'promptttspp', 'voxinstruct'],
                       help='TTS model to use')
    parser.add_argument('--json', type=str,
                       help='Input JSON file with generation data')
    parser.add_argument('--output', type=str,
                       help='Output directory for generated WAV files')
    parser.add_argument('--model-id', help='Hugging Face model ID (Parler) or external checkpoint root')
    parser.add_argument('--model-revision', help='Immutable Hugging Face commit revision for Parler')
    parser.add_argument('--backend-path', help='External PromptTTS++ or VoxInstruct source checkout')
    parser.add_argument('--config', help='JSON configuration file with a models mapping')
    parser.add_argument('--check', action='store_true', help='Validate configuration without loading models')
    parser.add_argument('--leace-artifact', help='LEACE .pt artifact for Parler description states')
    parser.add_argument('--steering-artifact',
                        help='Constant steering .pt artifact for pooled Parler description states')
    parser.add_argument('--voxinstruct-ar-steering-artifact',
                        help='Constant steering artifact for VoxInstruct AR projected mT5 states')
    parser.add_argument('--voxinstruct-nar-steering-artifact',
                        help='Constant steering artifact for VoxInstruct NAR projected mT5 states')
    parser.add_argument('--intervention-strength', type=float, default=1.0,
                          help='Non-negative intervention strength; LEACE is limited to 1 (default: 1)')
    parser.add_argument('--voxinstruct-ar-strength', type=float,
                        help='VoxInstruct AR strength (defaults to --intervention-strength)')
    parser.add_argument('--voxinstruct-nar-strength', type=float,
                        help='VoxInstruct NAR strength (defaults to --intervention-strength)')
    parser.add_argument('--intervention-mode', choices=('token', 'pooled-shift'), default='token',
                        help='Apply the eraser per token or as a pooled-mean shift')
    parser.add_argument('--no-bypass', action='store_true',
                         help='Apply the intervention to explicit gender prompts as well')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Parler prompts per generation call; batch RNG differs from sequential mode')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip existing WAV files (default: True)')
    parser.add_argument('--no-skip', action='store_false', dest='skip_existing',
                       help='Regenerate all files even if they exist')

    args = parser.parse_args()

    if not args.check and (not args.json or not args.output):
        parser.error('--json and --output are required unless --check is used')
    try:
        config = resolve_model_config(args.model, args.model_id, args.backend_path, args.config, args.model_revision)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[ERROR] invalid configuration: {exc}", file=sys.stderr)
        return 2
    errors = preflight(args.model, config, args.json)
    vox_steering_artifacts = (
        args.voxinstruct_ar_steering_artifact,
        args.voxinstruct_nar_steering_artifact,
    )
    vox_steering = any(vox_steering_artifacts)
    intervention_artifact = args.leace_artifact or args.steering_artifact or vox_steering
    artifact_families = sum(bool(value) for value in (
        args.leace_artifact, args.steering_artifact, vox_steering,
    ))
    if artifact_families > 1:
        errors.append('intervention artifacts are mutually exclusive')
    if args.leace_artifact and args.model not in ('parler-large', 'parler-mini'):
        errors.append('--leace-artifact is only supported for Parler-TTS models')
    if args.leace_artifact and not Path(args.leace_artifact).is_file():
        errors.append(f'LEACE artifact does not exist: {args.leace_artifact}')
    if args.steering_artifact and args.model not in ('parler-large', 'parler-mini'):
        errors.append('--steering-artifact is only supported for Parler-TTS models')
    if args.steering_artifact and not Path(args.steering_artifact).is_file():
        errors.append(f'steering artifact does not exist: {args.steering_artifact}')
    if args.steering_artifact and args.intervention_mode != 'pooled-shift':
        errors.append('--steering-artifact requires --intervention-mode pooled-shift')
    if vox_steering and args.model != 'voxinstruct':
        errors.append('--voxinstruct-*-steering-artifact requires --model voxinstruct')
    for artifact in filter(None, vox_steering_artifacts):
        if not Path(artifact).is_file():
            errors.append(f'steering artifact does not exist: {artifact}')
    if vox_steering and args.intervention_mode != 'pooled-shift':
        errors.append('VoxInstruct steering requires --intervention-mode pooled-shift')
    if args.no_bypass and not intervention_artifact:
        errors.append('--no-bypass requires an intervention artifact')
    if not args.intervention_strength >= 0.0:
        errors.append('--intervention-strength must be non-negative')
    if args.voxinstruct_ar_strength is not None and args.voxinstruct_ar_strength < 0.0:
        errors.append('--voxinstruct-ar-strength must be non-negative')
    if args.voxinstruct_nar_strength is not None and args.voxinstruct_nar_strength < 0.0:
        errors.append('--voxinstruct-nar-strength must be non-negative')
    if args.voxinstruct_ar_strength is not None and not args.voxinstruct_ar_steering_artifact:
        errors.append('--voxinstruct-ar-strength requires an AR steering artifact')
    if args.voxinstruct_nar_strength is not None and not args.voxinstruct_nar_steering_artifact:
        errors.append('--voxinstruct-nar-strength requires a NAR steering artifact')
    if args.leace_artifact and args.intervention_strength > 1.0:
        errors.append('--leace-artifact limits --intervention-strength to 1')
    if args.intervention_strength != 1.0 and not intervention_artifact:
        errors.append('--intervention-strength requires an intervention artifact')
    if args.batch_size < 1:
        errors.append('--batch-size must be at least 1')
    if args.batch_size > 1 and args.model not in ('parler-large', 'parler-mini'):
        errors.append('--batch-size > 1 is currently supported only for Parler-TTS models')
    if errors:
        for error in errors:
            print(f"[ERROR] {error}", file=sys.stderr)
        return 2
    print(f"[OK] {args.model}: model={config['model_id']}")
    if config.get("backend_path"):
        print(f"[OK] backend={config['backend_path']}")
    if args.check:
        return 0

    # Load JSON data
    print(f"[INFO] Loading data from {args.json}")
    with open(args.json, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[ERROR] JSON must contain a list of items")
        sys.exit(1)

    print(f"[INFO] Loaded {len(data)} items")
    intervention = None
    if vox_steering:
        artifacts = {
            stage: {
                "artifact_path": str(Path(path).resolve()),
                "artifact_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            }
            for stage, path in zip(("ar", "nar"), vox_steering_artifacts)
            if path
        }
        intervention = {
            "method": "constant-steering",
            "artifacts": artifacts,
            "strength": args.intervention_strength,
            "stage_strengths": {
                "ar": (
                    args.voxinstruct_ar_strength
                    if args.voxinstruct_ar_strength is not None
                    else args.intervention_strength
                ) if args.voxinstruct_ar_steering_artifact else None,
                "nar": (
                    args.voxinstruct_nar_strength
                    if args.voxinstruct_nar_strength is not None
                    else args.intervention_strength
                ) if args.voxinstruct_nar_steering_artifact else None,
            },
            "mode": args.intervention_mode,
            "bypass_explicit": not args.no_bypass,
            "bypass": "disabled" if args.no_bypass else "regex",
        }
    elif intervention_artifact:
        artifact_path = Path(intervention_artifact)
        intervention = {
            "method": (
                "leace" if args.leace_artifact else "constant-steering"
            ),
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "strength": args.intervention_strength,
            "mode": args.intervention_mode,
            "bypass_explicit": not args.no_bypass,
        }
        intervention["bypass"] = "disabled" if args.no_bypass else "regex"
    manifest = {
        "schema_version": 1,
        "model": args.model,
        "model_id": config["model_id"],
        "resolved_config": config,
        "input_json": str(Path(args.json).resolve()),
        "input_sha256": hashlib.sha256(Path(args.json).read_bytes()).hexdigest(),
        "prompt_count": len(data),
        "explicit_seed_count": sum(isinstance(item, dict) and "seed" in item for item in data),
        "seed_strategy": "per-item metadata seed; canonical prompt hash fallback",
        "batch_size": args.batch_size,
        "batch_seed_strategy": (
            "per-item metadata seed; canonical prompt hash fallback"
            if args.batch_size == 1 else "ordered-batch content hash; not byte-identical to sequential mode"
        ),
        "intervention": intervention,
        "status": "in_progress",
        "failed_count": None,
    }
    try:
        expected_wavs = {f"{item['id']}.wav" for item in data}
        manifest_path, run_state_path = _prepare_generation_manifest(
            args.output, manifest, args.skip_existing, expected_wav_names=expected_wavs,
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    # Create generator
    if args.model in ['parler-large', 'parler-mini']:
        generator = ParlerTTSGenerator(
            args.model,
            args.output,
            config,
            args.skip_existing,
            args.leace_artifact,
            bypass_explicit=not args.no_bypass,
            batch_size=args.batch_size,
            intervention_strength=args.intervention_strength,
            intervention_mode=args.intervention_mode,
            steering_artifact=args.steering_artifact,
        )
    elif args.model == 'promptttspp':
        generator = PromptTTSPPGenerator(args.model, args.output, config, args.skip_existing)
    elif args.model == 'voxinstruct':
        generator = VoxInstructGenerator(
            args.model, args.output, config, args.skip_existing,
            ar_steering_artifact=args.voxinstruct_ar_steering_artifact,
            nar_steering_artifact=args.voxinstruct_nar_steering_artifact,
            bypass_explicit=not args.no_bypass,
            intervention_strength=args.intervention_strength,
            ar_intervention_strength=args.voxinstruct_ar_strength,
            nar_intervention_strength=args.voxinstruct_nar_strength,
        )
    else:
        print(f"[ERROR] Unknown model: {args.model}")
        sys.exit(1)

    # Generate
    failed = generator.batch_generate(data)
    _finish_generation_manifest(manifest_path, run_state_path, manifest, failed)
    if failed:
        print(f"[ERROR] Generation incomplete: {failed} item(s) failed", file=sys.stderr)
        return 1

    print(f"\n[INFO] Generation complete!")
    print(f"[INFO] Output directory: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
