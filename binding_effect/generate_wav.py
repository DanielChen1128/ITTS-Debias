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

SCRIPT_DIR = Path(__file__).parent


# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "parler-large": {
        "model_id": "parler-tts/parler-tts-large-v1",
        "max_new_tokens": 8000,
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
        if not manifest_path.exists():
            raise ValueError("output contains WAVs without generation_manifest.json; use --no-skip or another directory")
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot validate existing generation manifest: {exc}") from exc
        identity_keys = (
            "model", "model_id", "resolved_config", "input_sha256", "prompt_count",
            "intervention",
        )
        mismatches = [key for key in identity_keys if existing.get(key) != manifest.get(key)]
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
        self.leace_eraser = None
        if leace_artifact:
            from debias.leace import LeaceEraser

            self.leace_eraser = LeaceEraser.load(leace_artifact, map_location=self.device)
            artifact_model = self.leace_eraser.metadata.get("model_id")
            if artifact_model and artifact_model != self.model_dir:
                raise ValueError(
                    f"LEACE artifact was fit for {artifact_model}, not {self.model_dir}"
                )

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

                    generation = generate_with_eraser(
                        self.model,
                        self.leace_eraser,
                        description,
                        desc_ids,
                        desc_attn,
                        bypass_explicit=self.bypass_explicit,
                        **generation_kwargs,
                    )

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
    def __init__(self, model_name: str, output_dir: str, config, skip_existing: bool = True):
        global torch, tqdm
        import torch
        from tqdm import tqdm
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.skip_existing = skip_existing
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
        ar_ckpt = torch.load(os.path.join(checkpoint_dir, "ar_1800k.pyt"), map_location=self.device)
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
        nar_ckpt = torch.load(os.path.join(checkpoint_dir, "nar_1800k.pyt"), map_location=self.device)
        self.nar_model.load_state_dict(nar_ckpt['model'], strict=True)
        self.nar_model.to(torch.bfloat16).eval()

        # Load Vocos vocoder
        vocos_path = ar_hp.vocos_path
        self.vocos = Vocos.from_hparams(f"{vocos_path}/config.yaml")
        state_dict = torch.load(f"{vocos_path}/pytorch_model.bin", map_location="cpu")
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

            # Initialize sequences (no audio prompt, lang_id=1 for English)
            lang_id = 1 + 1  # offset
            seqs = torch.tensor([[self.ar_hp.bos_id, lang_id]], device=self.device)
            segment_ids = torch.tensor([[1, 1]], device=self.device)

            # AR inference
            pred_st_flag = True
            past_key_values_base = None
            text_encode = None
            free_text_encode = torch.zeros([1, self.ar_hp.max_text_len, self.ar_hp.hidden_dim], device=self.device).to(torch.bfloat16)

            for j in range(self.config['max_length']):
                ar_outputs_base, text_encode = self.ar_model.predict(
                    input_ids=seqs,
                    segment_ids=segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    past_key_values=past_key_values_base,
                    text_encode=text_encode,
                )
                cond_logits = ar_outputs_base['logits']
                past_key_values_base = ar_outputs_base['past_key_values']

                if pred_st_flag:
                    logits = cond_logits
                    logits[:, :, self.ar_hp.bos_id] = -1e5
                    logits[:, :, self.ar_hp.st_token_num + 1:self.ar_hp.eos_id] = -1e5
                    logits[:, :, self.ar_hp.eos_id + 1] = -1e5
                    filtered_logits = self.top_k_top_p_filtering(logits[0, -1, :], top_k=5, top_p=0.95, temperature=self.config.get('temperature', 0.8))
                else:
                    logits = cond_logits
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

                if samples.item() == self.ar_hp.eos_id + 1:
                    break
                elif j == self.config['max_length'] - 1:
                    samples[:, :] = self.ar_hp.eos_id + 1
                    seqs = torch.cat([seqs, samples], dim=1)
                    segment_ids = torch.cat([segment_ids, torch.zeros_like(segment_ids[:, -1:]) + 2], dim=1)
                    break

            # NAR inference
            b, t = seqs.shape
            full_seqs = torch.stack([seqs] * self.ar_hp.at_res_num, dim=1)
            layer_index = torch.ones(size=[b,], device=self.device)

            for layer_idx in range(1, self.ar_hp.at_res_num):
                full_seqs = self.nar_model.predict(
                    full_seqs,
                    0,  # at_prompt_len (no audio prompt)
                    segment_ids,
                    text_ids=text_ids,
                    text_attn_mask=text_attn_mask,
                    layer_index=layer_idx,
                    iter_step=self.nar_iter_steps,
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
    parser.add_argument('--no-bypass', action='store_true',
                        help='Apply the eraser to explicit gender prompts as well')
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
    if args.leace_artifact and args.model not in ('parler-large', 'parler-mini'):
        errors.append('--leace-artifact is only supported for Parler-TTS models')
    if args.leace_artifact and not Path(args.leace_artifact).is_file():
        errors.append(f'LEACE artifact does not exist: {args.leace_artifact}')
    if args.no_bypass and not args.leace_artifact:
        errors.append('--no-bypass requires --leace-artifact')
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
    if args.leace_artifact:
        artifact_path = Path(args.leace_artifact)
        intervention = {
            "method": "leace",
            "artifact_path": str(artifact_path.resolve()),
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "bypass_explicit": not args.no_bypass,
        }
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
        )
    elif args.model == 'promptttspp':
        generator = PromptTTSPPGenerator(args.model, args.output, config, args.skip_existing)
    elif args.model == 'voxinstruct':
        generator = VoxInstructGenerator(args.model, args.output, config, args.skip_existing)
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
