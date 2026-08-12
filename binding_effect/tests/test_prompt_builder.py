import csv
import tempfile
import unittest
from pathlib import Path

from build_prompts import (
    build_stage1,
    build_stage2,
    expand_cells,
    expand_descriptors,
    select_stage2,
    source_catalog,
    stable_seed,
)
from prompt_audit import build_manifest


class PromptBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = source_catalog(Path(__file__).parents[1] / "descriptions")

    def test_stage1_catalog_matches_paper_cardinalities(self):
        self.assertEqual({axis: len(items) for axis, items in self.catalog.items()}, {
            "status": 2, "career": 27, "persona": 40,
        })

    def test_stage1_is_unique_cartesian_product(self):
        for axis, descriptors in self.catalog.items():
            rows = expand_descriptors(descriptors, base_seed=7)
            self.assertEqual(len(rows), len(descriptors) * 100)
            tuples = {(row["descriptor_id"], row["template_id"], row["transcript_id"]) for row in rows}
            self.assertEqual(len(tuples), len(rows), axis)

    def test_stage2_selects_two_probability_extremes(self):
        flat = {item["descriptor_id"]: item for items in self.catalog.values() for item in items}
        scores = {descriptor_id: index / len(flat) for index, descriptor_id in enumerate(sorted(flat))}
        selected = select_stage2(flat, scores)
        self.assertEqual(len(selected["status"]), 2)
        self.assertEqual(len(selected["career"]), 4)
        self.assertEqual(len(selected["persona"]), 4)
        for axis in ("career", "persona"):
            ranked = sorted(self.catalog[axis], key=lambda item: (scores[item["descriptor_id"]], item["descriptor_id"]))
            self.assertEqual(
                [item["descriptor_id"] for item in selected[axis]],
                [item["descriptor_id"] for item in ranked[:2] + ranked[-2:]],
            )

    def test_composite_cells_have_unique_cartesian_products(self):
        cells = [
            (self.catalog["status"][0], self.catalog["career"][0]),
            (self.catalog["status"][1], self.catalog["career"][1]),
        ]
        rows = expand_cells(cells, base_seed=3)
        self.assertEqual(len(rows), 200)
        tuples = {(row["cell_id"], row["template_id"], row["transcript_id"]) for row in rows}
        self.assertEqual(len(tuples), 200)

    def test_seed_is_stable_and_item_specific(self):
        self.assertEqual(stable_seed(1, "a", 2), stable_seed(1, "a", 2))
        self.assertNotEqual(stable_seed(1, "a", 2), stable_seed(1, "a", 3))

    def test_full_two_stage_reconstruction_passes_protocol_audit(self):
        source = Path(__file__).parents[1] / "descriptions"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage1 = root / "stage1"
            complete = root / "complete"
            detections = root / "detections.csv"
            catalog = build_stage1(source, stage1, base_seed=11)
            with open(detections, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "descriptor_id", "predicted_gender", "template_id", "transcript_id",
                    "axis", "provenance", "model_name",
                ])
                writer.writeheader()
                for index, descriptor in enumerate(item for items in catalog.values() for item in items):
                    for template_id in range(10):
                        for transcript_id in range(10):
                            writer.writerow({
                                "descriptor_id": descriptor["descriptor_id"],
                                "predicted_gender": "female" if index % 3 else "male",
                                "template_id": template_id, "transcript_id": transcript_id,
                                "axis": descriptor["axis"],
                                "provenance": "paper-aligned-reconstruction-v1",
                                "model_name": "synthetic-test-model",
                            })
            build_stage2(stage1, [detections], complete, "synthetic-test-model", base_seed=11)
            manifest = build_manifest(complete)
        self.assertEqual(manifest["repository_total"], 13300)
        self.assertTrue(manifest["reconstruction_protocol_alignment"])
        self.assertFalse(manifest["exact_paper_reproduction_available"])


if __name__ == "__main__":
    unittest.main()
