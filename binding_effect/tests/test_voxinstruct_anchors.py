import unittest

from build_voxinstruct_anchors import TRANSCRIPTS, build


class VoxInstructAnchorTests(unittest.TestCase):
    def test_build_adds_pair_matched_transcripts(self):
        rows = [
            {"gender_label": "female", "context_id": "x", "axes": [],
             "descriptor_ids": [], "template_id": 1},
            {"gender_label": "male", "context_id": "x", "axes": [],
             "descriptor_ids": [], "template_id": 1},
        ]
        output = build(rows)
        self.assertEqual(output[0]["prompt_text"], output[1]["prompt_text"])
        self.assertIn(output[0]["prompt_text"], TRANSCRIPTS)

    def test_build_rejects_unmatched_pair(self):
        rows = [
            {"gender_label": "female", "context_id": "x"},
            {"gender_label": "female", "context_id": "x"},
        ]
        with self.assertRaisesRegex(ValueError, "female/male"):
            build(rows)


if __name__ == "__main__":
    unittest.main()
