import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from evaluate_screen_quality import (
    load_prompt_spec,
    paired_interval,
    resolve_wav_path,
    select_stratified,
)
from build_full_quality_manifest import build_manifest


class ScreenQualityTests(unittest.TestCase):
    def test_full_quality_manifest_supports_voxinstruct(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sources = {
                "descriptions/descriptions_status_bias.json": 2,
                "descriptions/description_career_bias.json": 2,
                "descriptions/descriptions_persona_bias.json": 2,
                "data/parler-large/stage2/descriptions_two_axis.json": 2,
                "data/parler-large/stage2/descriptions_multi_axis.json": 2,
            }
            for relative, count in sources.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps([
                    {"id": f"{relative}-{index}", "prompt_text": "text"}
                    for index in range(count)
                ]), encoding="utf-8")

            manifest = build_manifest(
                root, model="voxinstruct", candidate="constant-steering-2x",
                pairs_per_stratum=1, seed="fixed",
            )

            self.assertEqual(
                manifest["method_directories"]["constant-steering-2x"]["stage1-status"],
                "results/voxinstruct/constant-steering-2x/stage1/status",
            )
            self.assertEqual(
                manifest["sources"]["stage2-two-axis"]["prompt_path"],
                "data/parler-large/stage2/descriptions_two_axis.json",
            )

    def test_paired_interval_reports_candidate_minus_baseline(self):
        result = paired_interval([1.1, 2.2, 3.3], [1.0, 2.0, 3.0], samples=1000)
        self.assertAlmostEqual(result["mean"], 0.2)
        self.assertLessEqual(result["ci95"][0], result["mean"])
        self.assertGreaterEqual(result["ci95"][1], result["mean"])

    def test_paired_interval_rejects_empty_input(self):
        with self.assertRaisesRegex(ValueError, "non-empty"):
            paired_interval([], [])

    def test_stratified_selection_is_deterministic_and_order_independent(self):
        rows = [{"id": str(index), "prompt_text": "text"} for index in range(10)]
        forward = select_stratified({"status": rows}, 4, seed="frozen-v1")
        reverse = select_stratified({"status": list(reversed(rows))}, 4, seed="frozen-v1")
        self.assertEqual([row["id"] for row in forward], [row["id"] for row in reverse])
        self.assertTrue(all(row["stratum"] == "status" for row in forward))

    def test_manifest_loads_selected_rows_and_resolves_nested_wavs(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "prompts.json"
            source_bytes = b'[{"id": "a", "prompt_text": "A"}, {"id": "b", "prompt_text": "B"}]\n'
            source_path.write_bytes(source_bytes)
            manifest_path = root / "quality.json"
            manifest_path.write_text(json.dumps({
                "sources": {
                    "stage1-status": {
                        "prompt_path": "prompts.json",
                        "sha256": hashlib.sha256(source_bytes).hexdigest(),
                    },
                },
                "selection": {"stage1-status": ["b"]},
                "method_directories": {
                    "original": {"stage1-status": "stage1/original/status"},
                },
            }), encoding="utf-8")

            prompts = load_prompt_spec(manifest_path, root)
            self.assertEqual(prompts[0]["prompt_text"], "B")
            self.assertEqual(
                resolve_wav_path(
                    root, "original", prompts[0],
                    {"original": {"stage1-status": "stage1/original/status"}},
                ),
                root / "stage1/original/status/b.wav",
            )

    def test_manifest_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "prompts.json").write_text("[]", encoding="utf-8")
            manifest_path = root / "quality.json"
            manifest_path.write_text(json.dumps({
                "sources": {
                    "status": {"prompt_path": "prompts.json", "sha256": "0" * 64},
                },
                "selection": {"status": []},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                load_prompt_spec(manifest_path, root)

    def test_manifest_filters_before_validating_frozen_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "prompts.json"
            source_path.write_text(json.dumps([
                {"id": "i", "prompt_text": "I", "partition": "implicit"},
                {"id": "e", "prompt_text": "E", "partition": "explicit"},
            ]), encoding="utf-8")
            manifest_path = root / "quality.json"
            manifest_path.write_text(json.dumps({
                "sources": {"explicit": {
                    "prompt_path": "prompts.json",
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "population": 2,
                    "filter": {"partition": "explicit"},
                    "filtered_population": 1,
                }},
                "selection_rule": {"pairs_per_stratum": 1, "seed": "fixed"},
                "selection": {"explicit": ["e"]},
            }), encoding="utf-8")

            prompts = load_prompt_spec(manifest_path, root)

            self.assertEqual([row["id"] for row in prompts], ["e"])

    def test_manifest_accepts_per_stratum_pair_counts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "prompts.json"
            source_path.write_text(json.dumps([
                {"id": "a", "prompt_text": "A"},
                {"id": "b", "prompt_text": "B"},
            ]), encoding="utf-8")
            selected = select_stratified({"small": json.loads(source_path.read_text())}, 1, seed="fixed")
            manifest_path = root / "quality.json"
            manifest_path.write_text(json.dumps({
                "sources": {"small": {
                    "prompt_path": "prompts.json",
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                }},
                "selection_rule": {"pairs_by_stratum": {"small": 1}, "seed": "fixed"},
                "selection": {"small": [selected[0]["id"]]},
            }), encoding="utf-8")

            self.assertEqual(len(load_prompt_spec(manifest_path, root)), 1)

    def test_manifest_rejects_selection_not_matching_frozen_rule(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "prompts.json"
            source_path.write_text(json.dumps([
                {"id": "a", "prompt_text": "A"},
                {"id": "b", "prompt_text": "B"},
            ]), encoding="utf-8")
            rows = json.loads(source_path.read_text(encoding="utf-8"))
            ranked_id = select_stratified({"status": rows}, 1, seed="fixed")[0]["id"]
            wrong_id = "b" if ranked_id == "a" else "a"
            manifest_path = root / "quality.json"
            manifest_path.write_text(json.dumps({
                "sources": {"status": {
                    "prompt_path": "prompts.json",
                    "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "population": 2,
                }},
                "selection_rule": {"pairs_per_stratum": 1, "seed": "fixed"},
                "selection": {"status": [wrong_id]},
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "frozen rule"):
                load_prompt_spec(manifest_path, root)

    def test_legacy_flat_wav_path_is_preserved(self):
        self.assertEqual(
            resolve_wav_path(Path("audio"), "original", {"id": "7"}),
            Path("audio/original/7.wav"),
        )


if __name__ == "__main__":
    unittest.main()
