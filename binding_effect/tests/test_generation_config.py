import hashlib
import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import generate_wav
from generate_wav import (
    ParlerTTSGenerator,
    _finish_generation_manifest,
    _validate_voxinstruct_steering,
    _prepare_generation_manifest,
    _seed_batch,
    _trim_batched_audio_padding,
    _seed_item,
    _voxinstruct_language_token,
    preflight,
    resolve_model_config,
)


class GenerationConfigTests(unittest.TestCase):
    def test_skip_allows_artifact_relocation_with_same_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"wav")
            base = {
                "model": "voxinstruct", "model_id": "model",
                "resolved_config": {"model_id": "model"},
                "input_sha256": "input", "prompt_count": 1,
                "batch_size": 1,
                "batch_seed_strategy": "per-item metadata seed; canonical prompt hash fallback",
                "intervention": {
                    "method": "constant-steering",
                    "artifacts": {"ar": {"artifact_path": "/old/ar.pt", "artifact_sha256": "same"}},
                },
            }
            (output / "generation_manifest.in_progress.json").write_text(
                json.dumps(base), encoding="utf-8",
            )
            current = json.loads(json.dumps(base))
            current["intervention"]["artifacts"]["ar"]["artifact_path"] = "/new/ar.pt"
            _prepare_generation_manifest(
                output, current, True, expected_wav_names={"0001.wav"},
            )

    def test_public_parler_default(self):
        with patch.dict(os.environ, {}, clear=True):
            config = resolve_model_config("parler-mini")
        self.assertEqual(config["model_id"], "parler-tts/parler-tts-mini-v1")

    def test_cli_precedes_environment_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"models": {"parler-mini": {"model_id": "file"}}}), encoding="utf-8")
            with patch.dict(os.environ, {"BINDING_PARLER_MINI_MODEL": "environment"}):
                config = resolve_model_config("parler-mini", model_id="cli", config_path=str(path))
        self.assertEqual(config["model_id"], "cli")

    def test_parler_revision_is_resolved(self):
        config = resolve_model_config("parler-mini", model_revision="commit-sha")
        self.assertEqual(config["revision"], "commit-sha")

    def test_prompt_hash_fallback_seed_is_content_specific(self):
        fake_torch = Mock()
        fake_torch.cuda.is_available.return_value = False
        with patch.object(generate_wav, "torch", fake_torch, create=True):
            first = _seed_item({"id": "1", "description": "calm", "prompt_text": "hello"})
            repeated = _seed_item({"id": "1", "description": "calm", "prompt_text": "hello"})
            changed = _seed_item({"id": "1", "description": "warm", "prompt_text": "hello"})
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)

    def test_batch_seed_is_stable_and_order_specific(self):
        fake_torch = Mock()
        fake_torch.cuda.is_available.return_value = False
        with patch.object(generate_wav, "torch", fake_torch, create=True):
            first = _seed_batch([{"id": "1"}, {"id": "2"}])
            repeated = _seed_batch([{"id": "1"}, {"id": "2"}])
            reversed_items = _seed_batch([{"id": "2"}, {"id": "1"}])
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, reversed_items)

    def test_batched_audio_padding_is_trimmed(self):
        audio = generate_wav.np.array([0.2, -0.1, 0.0, 0.0])
        trimmed = _trim_batched_audio_padding(audio)
        self.assertEqual(trimmed.tolist(), [0.2, -0.1])

    def test_voxinstruct_english_language_token_uses_source_id_zero(self):
        hp = SimpleNamespace(lang_mapping={"en": 0, "zh": 1})
        self.assertEqual(_voxinstruct_language_token(hp, "en"), 1)
        self.assertEqual(_voxinstruct_language_token(hp, "zh"), 2)

    def test_voxinstruct_steering_allows_ar_only_ablation(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "ar.pt"
            artifact.write_bytes(b"artifact")
            argv = [
                "generate_wav.py", "--model", "voxinstruct", "--check",
                "--voxinstruct-ar-steering-artifact", str(artifact),
                "--intervention-mode", "pooled-shift",
            ]
            with patch.object(sys, "argv", argv), \
                    patch("generate_wav.resolve_model_config", return_value={"model_id": "model", "backend_path": "backend"}), \
                    patch("generate_wav.preflight", return_value=[]):
                result = generate_wav.main()
        self.assertEqual(result, 0)

    def test_voxinstruct_artifact_identity_uses_checkpoint_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "ar.pyt"
            checkpoint.write_bytes(b"same checkpoint")
            steering = SimpleNamespace(
                offset=SimpleNamespace(numel=lambda: 1024),
                metadata={
                    "representation": "ar-representation",
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "model_id": "/a/different/host/path",
                },
            )
            _validate_voxinstruct_steering(
                steering, "ar", "ar-representation", 1024,
                "/current/model/path", checkpoint,
            )

    def test_voxinstruct_artifact_rejects_different_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "nar.pyt"
            checkpoint.write_bytes(b"current checkpoint")
            steering = SimpleNamespace(
                offset=SimpleNamespace(numel=lambda: 1024),
                metadata={
                    "representation": "nar-representation",
                    "checkpoint_sha256": hashlib.sha256(b"different checkpoint").hexdigest(),
                },
            )
            with self.assertRaisesRegex(ValueError, "checkpoint does not match"):
                _validate_voxinstruct_steering(
                    steering, "nar", "nar-representation", 1024,
                    directory, checkpoint,
                )

    def test_preflight_reports_concrete_external_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            backend = root / "backend"
            model = root / "model"
            backend.mkdir()
            model.mkdir()
            with patch("generate_wav.importlib.util.find_spec", return_value=object()):
                errors = preflight("promptttspp", {
                    "model_id": str(model), "backend_path": str(backend),
                })
        self.assertTrue(any("demo.yaml" in error for error in errors))
        self.assertTrue(any("last.ckpt" in error for error in errors))

    def test_batch_reports_item_failures(self):
        generator = ParlerTTSGenerator.__new__(ParlerTTSGenerator)
        generator.model_name = "parler-mini"
        generator.output_dir = Path("unused")
        generator.skip_existing = False
        generator.generate_single = lambda item: item["id"] != "bad"
        with patch.object(generate_wav, "tqdm", side_effect=lambda data, **kwargs: data, create=True):
            failed = generator.batch_generate([{"id": "ok"}, {"id": "bad"}])
        self.assertEqual(failed, 1)

    def test_main_returns_nonzero_for_partial_generation(self):
        with tempfile.TemporaryDirectory() as directory:
            prompts = Path(directory) / "prompts.json"
            prompts.write_text(json.dumps([{
                "id": "1", "description": "voice", "prompt_text": "hello",
            }]), encoding="utf-8")
            generator = Mock()
            generator.batch_generate.return_value = 1
            argv = ["generate_wav.py", "--model", "parler-mini", "--json", str(prompts), "--output", directory]
            with patch.object(sys, "argv", argv), \
                    patch("generate_wav.preflight", return_value=[]), \
                    patch("generate_wav.ParlerTTSGenerator", return_value=generator):
                result = generate_wav.main()
        self.assertEqual(result, 1)

    def test_skip_rejects_wavs_without_matching_manifest(self):
        manifest = {"model": "parler-mini", "model_id": "checkpoint", "input_sha256": "abc", "prompt_count": 1}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "without generation_manifest"):
                _prepare_generation_manifest(output, manifest, skip_existing=True)

    def test_skip_rejects_changed_input_provenance(self):
        old = {"model": "parler-mini", "model_id": "checkpoint", "input_sha256": "old", "prompt_count": 1}
        new = dict(old, input_sha256="new")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            (output / "generation_manifest.json").write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                _prepare_generation_manifest(output, new, skip_existing=True)

    def test_skip_resumes_matching_in_progress_manifest(self):
        manifest = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "same", "prompt_count": 2, "intervention": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"partial")
            (output / "generation_manifest.in_progress.json").write_text(
                json.dumps(manifest), encoding="utf-8",
            )
            completed, in_progress = _prepare_generation_manifest(
                output, manifest, skip_existing=True,
                expected_wav_names={"0001.wav", "0002.wav"},
            )
            self.assertEqual(completed, output / "generation_manifest.json")
            self.assertEqual(in_progress, output / "generation_manifest.in_progress.json")

    def test_skip_rejects_changed_in_progress_provenance(self):
        old = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "old", "prompt_count": 1, "intervention": None,
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"partial")
            (output / "generation_manifest.in_progress.json").write_text(
                json.dumps(old), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                _prepare_generation_manifest(
                    output, dict(old, input_sha256="new"), skip_existing=True,
                )

    def test_skip_rejects_changed_intervention_provenance(self):
        old = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "same", "prompt_count": 1, "intervention": None,
        }
        new = dict(old, intervention={"method": "leace", "artifact_sha256": "abc"})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            (output / "generation_manifest.json").write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "intervention"):
                _prepare_generation_manifest(output, new, skip_existing=True)

    def test_legacy_full_strength_intervention_manifest_resumes(self):
        old = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "same", "prompt_count": 1,
            "intervention": {"method": "leace", "artifact_sha256": "abc"},
        }
        new = dict(old, intervention={**old["intervention"], "strength": 1.0})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            (output / "generation_manifest.json").write_text(json.dumps(old), encoding="utf-8")
            _prepare_generation_manifest(output, new, skip_existing=True)

    def test_skip_rejects_changed_batch_provenance(self):
        old = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "same", "prompt_count": 1, "intervention": None,
            "batch_size": 1, "batch_seed_strategy": "per-item",
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            (output / "generation_manifest.json").write_text(json.dumps(old), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "batch_size"):
                _prepare_generation_manifest(
                    output, dict(old, batch_size=4), skip_existing=True,
                )

    def test_legacy_manifest_resumes_as_sequential_batch_one(self):
        manifest = {
            "model": "parler-mini", "model_id": "checkpoint", "resolved_config": {},
            "input_sha256": "same", "prompt_count": 1, "intervention": None,
            "batch_size": 1,
            "batch_seed_strategy": "per-item metadata seed; canonical prompt hash fallback",
        }
        legacy = {key: value for key, value in manifest.items() if not key.startswith("batch_")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "0001.wav").write_bytes(b"stale")
            (output / "generation_manifest.json").write_text(json.dumps(legacy), encoding="utf-8")
            _prepare_generation_manifest(output, manifest, skip_existing=True)

    def test_rejects_wavs_outside_current_prompt_set(self):
        manifest = {"model": "parler-mini"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "stale.wav").write_bytes(b"stale")
            with self.assertRaisesRegex(ValueError, "outside the current prompt set"):
                _prepare_generation_manifest(
                    output, manifest, skip_existing=False, expected_wav_names={"current.wav"},
                )

    def test_failed_run_preserves_completed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed = root / "generation_manifest.json"
            in_progress = root / "generation_manifest.in_progress.json"
            completed.write_text('{"status": "complete"}\n', encoding="utf-8")
            manifest = {"status": "in_progress", "failed_count": None}
            _finish_generation_manifest(completed, in_progress, manifest, failed=1)
            self.assertEqual(
                json.loads(completed.read_text(encoding="utf-8")),
                {"status": "complete"},
            )
            self.assertEqual(
                json.loads(in_progress.read_text(encoding="utf-8"))["status"],
                "incomplete",
            )


if __name__ == "__main__":
    unittest.main()
