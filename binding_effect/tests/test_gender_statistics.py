import unittest
import pandas as pd

import analyze_gender
import recompute_binary_gender


class GenderStatisticsTests(unittest.TestCase):
    def test_binary_gender_ignores_child_score(self):
        self.assertEqual(analyze_gender.binary_gender_label(0.2, 0.1), "female")
        self.assertEqual(analyze_gender.binary_gender_label(0.1, 0.2), "male")
        self.assertEqual(analyze_gender.binary_gender_label(0.1, 0.1), "female")

    def test_binary_gender_denominator_excludes_unknown_labels(self):
        analyze_gender.pd = pd
        frame = pd.DataFrame({
            "predicted_gender": ["female", "male", "unknown"],
            "female_score": [0.6, 0.2, 0.0],
            "male_score": [0.3, 0.6, 0.0],
        })
        overall = analyze_gender.compute_statistics(frame)["overall"]
        self.assertEqual(overall["total"], 2)
        self.assertAlmostEqual(overall["female_probability"], 1 / 2)
        self.assertAlmostEqual(overall["conditional_female_score"], (2 / 3 + 1 / 4) / 2)
        self.assertEqual(overall["excluded"], {"unknown": 1})

    def test_statistics_reject_stale_child_labels(self):
        analyze_gender.pd = pd
        frame = pd.DataFrame({
            "predicted_gender": ["child"],
            "female_score": [0.2],
            "male_score": [0.1],
        })
        with self.assertRaisesRegex(ValueError, "child labels must be recomputed"):
            analyze_gender.compute_statistics(frame)

    def test_recompute_binary_gender_replaces_child_and_preserves_unknown(self):
        frame = pd.DataFrame({
            "predicted_gender": ["child", "child", "unknown"],
            "female_score": [0.7, 0.2, float("nan")],
            "male_score": [0.1, 0.8, 0.4],
        })
        result = recompute_binary_gender.recompute_frame(frame)
        self.assertEqual(result["predicted_gender"].tolist(), ["female", "male", "unknown"])

    def test_merge_preserves_protocol_metadata(self):
        analyze_gender.pd = pd
        metadata = pd.DataFrame([{
            "id": "0001", "trait": "Career", "descriptor_id": "career:nurse", "seed": 7,
        }])
        merged = analyze_gender.merge_with_metadata([{
            "id": "0001", "predicted_gender": "female",
        }], metadata)
        self.assertEqual(merged.loc[0, "descriptor_id"], "career:nurse")
        self.assertEqual(merged.loc[0, "seed"], 7)

    def test_merge_rejects_incomplete_wav_set(self):
        analyze_gender.pd = pd
        metadata = pd.DataFrame([{"id": "0001"}, {"id": "0002"}])
        with self.assertRaisesRegex(ValueError, "1 missing WAVs"):
            analyze_gender.merge_with_metadata([{"id": "0001"}], metadata)

    def test_summary_accepts_mixed_trait_metadata(self):
        stats = {
            "overall": {
                "total": 2, "excluded": {},
                "female_probability": 0.5, "conditional_female_score": 0.5,
                "counts": {"female": 1, "male": 1},
                "percentages": {"female": 50.0, "male": 50.0},
            },
            "by_trait": {
                "Career": {"total": 1, "counts": {"female": 1},
                           "percentages": {"female": 100.0}, "female_male_ratio": float("inf")},
                float("nan"): {"total": 1, "counts": {"male": 1},
                               "percentages": {"male": 100.0}, "female_male_ratio": 0.0},
            },
        }
        analyze_gender.print_summary(stats)


if __name__ == "__main__":
    unittest.main()
