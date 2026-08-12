import json
import tempfile
import unittest
from pathlib import Path

from unittest.mock import patch

from build_prompts import expand_descriptors
from prompt_audit import audit_file, build_manifest


class PromptAuditTests(unittest.TestCase):
    def test_tracked_descriptions_are_canonical_stage1(self):
        root = Path(__file__).parents[1]
        manifest = build_manifest(root / "descriptions")
        self.assertEqual(manifest["repository_total"], 6900)
        self.assertEqual(manifest["missing_from_complete_protocol"], 6400)
        self.assertTrue(manifest["canonical_stage1_alignment"])
        self.assertEqual(
            {Path(item["file"]).name for item in manifest["files"]},
            {"description_career_bias.json", "descriptions_persona_bias.json", "descriptions_status_bias.json"},
        )

    def test_valid_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            path.write_text(json.dumps([{
                "id": "1", "description": "voice", "trait": "status",
                "keywords": "high", "prompt_text": "hello",
            }]), encoding="utf-8")
            self.assertEqual(audit_file(path)["errors"], [])

    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prompts.json"
            item = {"id": "1", "description": "voice", "trait": "status", "keywords": "high", "prompt_text": "hello"}
            path.write_text(json.dumps([item, item]), encoding="utf-8")
            self.assertIn("duplicate ids", audit_file(path)["errors"])

    def test_overcount_is_not_exact_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            item = {"id": "1", "description": "voice", "trait": "status", "keywords": "high", "prompt_text": "hello"}
            path.write_text(json.dumps([item, dict(item, id="2")]), encoding="utf-8")
            with patch("prompt_audit.PAPER_EXPECTED", {"expected.json": 1}):
                manifest = build_manifest(Path(directory))
            self.assertFalse(manifest["paper_count_alignment"])
            self.assertEqual(manifest["count_mismatches"]["expected.json"], {"expected": 1, "actual": 2})
            self.assertFalse(manifest["exact_paper_reproduction_available"])

    def test_canonical_alignment_rejects_changed_content_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "expected.json"
            item = {"id": "1", "description": "voice", "trait": "status", "keywords": "high", "prompt_text": "hello"}
            path.write_text(json.dumps([item]), encoding="utf-8")
            with patch("prompt_audit.PAPER_EXPECTED", {"expected.json": 1}), \
                    patch("prompt_audit.STAGE1_EXPECTED", {"expected.json": 1}), \
                    patch("prompt_audit.CANONICAL_STAGE1_SHA256", {"expected.json": "different"}):
                manifest = build_manifest(Path(directory))
            self.assertFalse(manifest["canonical_stage1_hashes_match"])
            self.assertFalse(manifest["canonical_stage1_alignment"])

    def test_reconstructed_status_cartesian_product(self):
        descriptors = [
            {"axis": "status", "descriptor_id": "status:high", "keyword": "high", "trait": "SDO", "description": "High style."},
            {"axis": "status", "descriptor_id": "status:low", "keyword": "low", "trait": "SDO", "description": "Low style."},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions_status_bias.json"
            path.write_text(json.dumps(expand_descriptors(descriptors, 0)), encoding="utf-8")
            result = audit_file(path)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["protocol_errors"], [])
        self.assertEqual(result["structure"]["actual_cells"], 2)

    def test_rejects_non_cartesian_identifier_ranges(self):
        descriptors = [
            {"axis": "status", "descriptor_id": "status:high", "keyword": "high", "trait": "SDO", "description": "High style."},
            {"axis": "status", "descriptor_id": "status:low", "keyword": "low", "trait": "SDO", "description": "Low style."},
        ]
        rows = expand_descriptors(descriptors, 0)
        for index, row in enumerate(rows):
            row["template_id"] = index % 100
            row["transcript_id"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions_status_bias.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            result = audit_file(path)
        self.assertTrue(any("10 x 10" in error for error in result["protocol_errors"]))

    def test_rejects_string_identifier_metadata(self):
        descriptors = [
            {"axis": "status", "descriptor_id": "status:high", "keyword": "high", "trait": "SDO", "description": "High style."},
            {"axis": "status", "descriptor_id": "status:low", "keyword": "low", "trait": "SDO", "description": "Low style."},
        ]
        rows = expand_descriptors(descriptors, 0)
        rows[0]["template_id"] = "0"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions_status_bias.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            result = audit_file(path)
        self.assertTrue(any("invalid reconstruction metadata" in error for error in result["protocol_errors"]))

    def test_rejects_wrong_two_axis_topology(self):
        descriptor = {
            "axis": "status", "descriptor_id": "status:high", "keyword": "high",
            "trait": "SDO", "description": "High style.",
        }
        rows = expand_descriptors([descriptor] * 32, 0)
        for cell_index in range(32):
            for row in rows[cell_index * 100:(cell_index + 1) * 100]:
                row["axis"] = "status+career"
                row["cell_id"] = row["descriptor_id"] = f"status:high|career:{cell_index}"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "descriptions_two_axis.json"
            path.write_text(json.dumps(rows), encoding="utf-8")
            result = audit_file(path)
        self.assertTrue(any("cell topology" in error for error in result["protocol_errors"]))


if __name__ == "__main__":
    unittest.main()
