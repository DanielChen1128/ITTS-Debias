import unittest
import pandas as pd

import analyze_gender


class GenderStatisticsTests(unittest.TestCase):
    def test_paper_and_adult_denominators_are_both_reported(self):
        analyze_gender.pd = pd
        frame = pd.DataFrame({
            "predicted_gender": ["female", "male", "child", "unknown"],
        })
        overall = analyze_gender.compute_statistics(frame)["overall"]
        self.assertEqual(overall["total"], 3)
        self.assertEqual(overall["adult_total"], 2)
        self.assertAlmostEqual(overall["female_probability"], 1 / 3)
        self.assertAlmostEqual(overall["adult_female_probability"], 1 / 2)
        self.assertEqual(overall["excluded"], {"unknown": 1})

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


if __name__ == "__main__":
    unittest.main()
