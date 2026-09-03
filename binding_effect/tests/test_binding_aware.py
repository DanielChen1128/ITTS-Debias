import unittest
from collections import Counter
from pathlib import Path

from audit_binding_anchors import audit
from audit_large_rlace_anchors import audit as audit_large
from build_binding_anchors import build as build_anchors
from build_large_rlace_anchors import build as build_large_anchors
from build_prompts import source_catalog
from debias.binding_protocol import (
    TRAIN_CAREER_IDS,
    TRAIN_PERSONA_IDS,
    training_contexts,
)


class SmallContextProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).parents[1]
        cls.catalog = source_catalog(cls.root / "descriptions")

    def test_training_inventory_is_small_fixed_and_status_free(self):
        self.assertEqual(len(TRAIN_CAREER_IDS), 4)
        self.assertEqual(len(TRAIN_PERSONA_IDS), 8)
        contexts = training_contexts(self.catalog)
        self.assertEqual(len(contexts), 45)
        self.assertEqual(Counter("+".join(row["axes"]) or "neutral" for row in contexts), Counter({
            "career+persona": 32, "persona": 8, "career": 4, "neutral": 1,
        }))
        self.assertFalse(any("status" in row["axes"] for row in contexts))

    def test_anchor_set_has_540_matched_unique_records(self):
        anchors = build_anchors(self.catalog)
        self.assertEqual(len(anchors), 540)
        self.assertEqual(Counter(row["gender_label"] for row in anchors), Counter({
            "female": 270, "male": 270,
        }))
        report = audit(anchors)
        self.assertTrue(report["passed"], report["errors"][:3])
        self.assertEqual(report["contexts"], 45)
        self.assertEqual(report["status_records"], 0)

    def test_large_anchor_set_is_balanced_and_complete(self):
        anchors = build_large_anchors(self.catalog)
        report = audit_large(anchors)
        self.assertTrue(report["passed"], report["errors"][:3])
        self.assertEqual(report["records"], 5040)
        self.assertEqual(report["contexts"], 180)
        self.assertEqual(report["templates"], 14)
        self.assertEqual(report["descriptor_coverage"], 67)
        self.assertEqual(report["gender_counts"], {"female": 2520, "male": 2520})
        self.assertEqual(report["status_records"], 0)


if __name__ == "__main__":
    unittest.main()
