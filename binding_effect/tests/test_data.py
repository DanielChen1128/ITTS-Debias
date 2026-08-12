import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from debias.data import (
    CAREER_GENDER_PRIOR,
    TRANSCRIPTS,
    assign_splits,
    dataset_hashes,
    load_descriptors,
    parse_keywords,
    validate_split_manifest,
)

import build_anchors
import build_neutral
import build_splits


class DataUtilTests(unittest.TestCase):
    def test_parse_keywords(self):
        fields = parse_keywords("SDO=high;career=nanny;persona_trait=none")
        self.assertEqual(fields["SDO"], "high")
        self.assertEqual(fields["career"], "nanny")
        self.assertEqual(fields["persona_trait"], "none")

    def test_parse_keywords_handles_plain_label(self):
        self.assertEqual(parse_keywords("Fisherman"), {})

    def test_assign_splits_is_disjoint_and_deterministic(self):
        lemmas = [f"l{i}" for i in range(20)]
        first = assign_splits(lemmas, seed=1)
        second = assign_splits(lemmas, seed=1)
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), {"train", "dev", "test"})
        # A different seed should not produce an identical assignment.
        self.assertNotEqual(assign_splits(lemmas, seed=2), first)

    def test_assign_splits_partition_sizes(self):
        lemmas = [f"l{i}" for i in range(10)]
        assignment = assign_splits(lemmas, seed=3, dev_frac=0.2, test_frac=0.4)
        counts = {s: sum(v == s for v in assignment.values()) for s in ("train", "dev", "test")}
        self.assertEqual(counts, {"test": 4, "dev": 2, "train": 4})


class LoadDescriptorTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        root = Path(self.dir.name)
        (root / "description_career_bias.json").write_text(
            json.dumps(
                [{"id": "1", "description": "Act like a Fisherman.", "trait": "Career",
                  "keywords": "Fisherman", "prompt_text": TRANSCRIPTS[0]}]
            ),
            encoding="utf-8",
        )
        (root / "descriptions_persona_bias.json").write_text(
            json.dumps(
                [{"id": "1", "description": "curious phrasing", "trait": "Openness",
                  "keywords": "curious", "prompt_text": TRANSCRIPTS[1]}]
            ),
            encoding="utf-8",
        )
        (root / "descriptions_status_bias.json").write_text(
            json.dumps(
                [{"id": "1", "description": "high status", "trait": "SDO",
                  "keywords": "high_status_speaker", "prompt_text": TRANSCRIPTS[2]}]
            ),
            encoding="utf-8",
        )
        (root / "descriptions_two_axis.json").write_text(
            json.dumps(
                [{"id": "1", "description": "composite", "trait": "none",
                  "keywords": "SDO=high;SDO_bias=female;career_gender=female;career=nanny;"
                              "persona_gender=none;persona_trait=none;persona_words=none",
                  "prompt_text": TRANSCRIPTS[3]}]
            ),
            encoding="utf-8",
        )
        (root / "descriptions_multi_axis.json").write_text(
            json.dumps(
                [{"id": "1", "description": "composite", "trait": "Openness",
                  "keywords": "SDO=low;SDO_bias=male;career_gender=male;career=mechanic;"
                              "persona_gender=male;persona_trait=Openness;persona_words=vague",
                  "prompt_text": TRANSCRIPTS[4]}]
            ),
            encoding="utf-8",
        )
        self.root = root

    def tearDown(self):
        self.dir.cleanup()

    def test_axis_tagging_and_stereotype(self):
        records = {
            r["axis"]: r
            for r in load_descriptors(self.root, "legacy-5900-v1", verify=False)
        }
        self.assertEqual(records["career"]["lemma"], "career:fisherman")
        self.assertEqual(records["career"]["stereotype_gender"], CAREER_GENDER_PRIOR["fisherman"])
        self.assertEqual(records["persona"]["lemma"], "persona:curious")
        self.assertEqual(records["persona"]["stereotype_gender"], "unspecified")
        self.assertEqual(records["two_axis"]["stereotype_gender"], "female")
        self.assertEqual(records["multi_axis"]["stereotype_gender"], "male")
        self.assertIn("status=low", records["multi_axis"]["lemma"])

    def test_canonical_profile_excludes_legacy_composites(self):
        records = load_descriptors(
            self.root, "canonical-stage1-6900-v1", verify=False
        )
        self.assertEqual({record["axis"] for record in records}, {"career", "persona", "status"})

    def test_unknown_profile_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown dataset profile"):
            load_descriptors(self.root, "latest", verify=False)

    def test_dataset_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "description_career_bias.json",
                "descriptions_persona_bias.json",
                "descriptions_status_bias.json",
            ):
                (root / name).write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match the frozen"):
                dataset_hashes(root, "canonical-stage1-6900-v1")

    def test_split_profile_mismatch_is_rejected(self):
        manifest = {
            "dataset_profile": "canonical-stage1-6900-v1",
            "source_sha256": {"file": "digest"},
        }
        with patch("debias.data.dataset_hashes", return_value={"file": "digest"}):
            with self.assertRaisesRegex(ValueError, "does not match dataset profile"):
                validate_split_manifest(manifest, self.root, "legacy-5900-v1")

    def test_split_source_hash_mismatch_is_rejected(self):
        manifest = {
            "dataset_profile": "legacy-5900-v1",
            "source_sha256": {"file": "old"},
        }
        with patch("debias.data.dataset_hashes", return_value={"file": "current"}):
            with self.assertRaisesRegex(ValueError, "source hashes"):
                validate_split_manifest(manifest, self.root, "legacy-5900-v1")


class BuilderTests(unittest.TestCase):
    def test_anchor_pairs_are_gender_matched(self):
        records = build_anchors.build(control_templates={6, 7})
        females = [r for r in records if r["gender_label"] == "female"]
        males = [r for r in records if r["gender_label"] == "male"]
        self.assertEqual(len(females), len(males))
        self.assertEqual({r["anchor_split"] for r in records}, {"fit", "control"})
        # Every template renders both genders over all transcripts.
        self.assertEqual(len(records), len(build_anchors.TEMPLATES) * 2 * len(TRANSCRIPTS))

    def test_neutral_prompts_have_no_gender_terms(self):
        records = build_neutral.build()
        joined = " ".join(r["description"].lower() for r in records)
        for term in ("female", "male", "woman", "man", "girl", "boy"):
            self.assertNotIn(term, joined)

    def test_split_manifest_has_disjoint_lemmas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, payload in _MINI_CORPUS.items():
                (root / name).write_text(json.dumps(payload), encoding="utf-8")
            manifest = build_splits.build(
                str(root), seed=7, dev_frac=0.2, test_frac=0.4,
                profile="legacy-5900-v1", verify_dataset=False,
            )
        self.assertEqual(manifest["split_unit"], "descriptor_lemma")
        self.assertEqual(len(manifest["lemma_assignment"]), manifest["n_lemmas"])


_MINI_CORPUS = {
    "description_career_bias.json": [
        {"id": str(i), "description": "c", "trait": "Career", "keywords": f"Career{i}",
         "prompt_text": TRANSCRIPTS[i % len(TRANSCRIPTS)]}
        for i in range(10)
    ],
    "descriptions_persona_bias.json": [
        {"id": str(i), "description": "p", "trait": "Openness", "keywords": f"word{i}",
         "prompt_text": TRANSCRIPTS[i % len(TRANSCRIPTS)]}
        for i in range(10)
    ],
    "descriptions_status_bias.json": [
        {"id": "1", "description": "s", "trait": "SDO", "keywords": "high_status_speaker",
         "prompt_text": TRANSCRIPTS[0]},
        {"id": "2", "description": "s", "trait": "SDO", "keywords": "low_status_speaker",
         "prompt_text": TRANSCRIPTS[1]},
    ],
    "descriptions_two_axis.json": [
        {"id": "1", "description": "t", "trait": "none",
         "keywords": "SDO=high;career=nanny;persona_trait=none", "prompt_text": TRANSCRIPTS[0]},
    ],
    "descriptions_multi_axis.json": [
        {"id": "1", "description": "m", "trait": "Openness",
         "keywords": "SDO=low;career=mechanic;persona_trait=Openness", "prompt_text": TRANSCRIPTS[0]},
    ],
}


if __name__ == "__main__":
    unittest.main()
