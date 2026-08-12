import json
import os
import tempfile
import unittest
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import generate_wav
from generate_wav import (
    ParlerTTSGenerator,
    _finish_generation_manifest,
    _prepare_generation_manifest,
    _seed_item,
    preflight,
    resolve_model_config,
)


class GenerationConfigTests(unittest.TestCase):
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
