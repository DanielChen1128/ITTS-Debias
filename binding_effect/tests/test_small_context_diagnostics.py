import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from analyze_small_context_diagnostics import (
    descriptor_group,
    interaction_family,
    summarize_detection_rows,
)
from build_small_context_interaction_specs import build_method


class SmallContextDiagnosticsTests(unittest.TestCase):
    def test_descriptor_groups_distinguish_training_and_neighbor(self):
        self.assertEqual(descriptor_group("career", "career:mechanic"), "training_seen")
        self.assertEqual(
            descriptor_group("career", "career:mechanician"),
            "mechanic_lexical_neighbor",
        )
        self.assertEqual(descriptor_group("persona", "persona:openness:curious"), "training_seen")
        self.assertEqual(descriptor_group("status", "status:high"), "status_excluded_from_training")

    def test_detection_summary_uses_binary_gender_denominator(self):
        rows = [
            {"descriptor_id": "career:a", "predicted_gender": "female", "female_score": "0.9", "male_score": "0.1"},
            {"descriptor_id": "career:a", "predicted_gender": "male", "female_score": "0.1", "male_score": "0.2"},
            {"descriptor_id": "career:b", "predicted_gender": "male", "female_score": "0.2", "male_score": "0.8"},
            {"descriptor_id": "career:b", "predicted_gender": "female", "female_score": "0.8", "male_score": "0.2"},
        ]
        summary = summarize_detection_rows(rows)
        self.assertEqual(summary["descriptor_count"], 2)
        self.assertEqual(summary["female_probability"], 0.5)
        self.assertEqual(summary["descriptor_range"], 0.0)

    def test_detection_summary_rejects_stale_child_labels(self):
        with self.assertRaisesRegex(ValueError, "child labels must be recomputed"):
            summarize_detection_rows([{
                "descriptor_id": "career:a",
                "predicted_gender": "child",
                "female_score": "0.2",
                "male_score": "0.1",
            }])

    def test_interaction_family_preserves_axis_order(self):
        self.assertEqual(
            interaction_family("two:status:low|career:nurse", "2"),
            "status+career",
        )
        self.assertEqual(
            interaction_family("three:status:low|career:nurse|persona:warm", "3"),
            "status+career+persona",
        )

    def test_interaction_builder_accepts_stage1_method_root_override(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            stage1_method = root / "matched-original"
            stage2 = root / "model-first" / "original" / "stage2"
            fields = "descriptor_id,predicted_gender,female_score,male_score\n"
            for axis in ("status", "career", "persona"):
                path = stage1_method / axis / "detection_results.csv"
                path.parent.mkdir(parents=True)
                path.write_text(fields + f"{axis}:a,female,0.9,0.1\n", encoding="utf-8")
            for axis, descriptors in (
                (
                    "two-axis",
                    (
                        "status:a|career:a",
                        "status:a|persona:a",
                        "career:a|persona:a",
                    ),
                ),
                ("three-axis", ("status:a|career:a|persona:a",)),
            ):
                path = stage2 / axis / "detection_results.csv"
                path.parent.mkdir(parents=True)
                rows = "".join(f"{descriptor},female,0.9,0.1\n" for descriptor in descriptors)
                path.write_text(fields + rows, encoding="utf-8")

            spec = build_method(
                root / "unused",
                root / "unused-stage2",
                root / "output",
                "original",
                matched_generation=True,
                stage1_method_root=stage1_method,
                stage2_method_root=stage2,
            )

            self.assertFalse(spec["provisional"])
            self.assertEqual(spec["interaction_count"], 4)


if __name__ == "__main__":
    unittest.main()
