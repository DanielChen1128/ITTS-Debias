import math
import unittest

from binding_stats import (
    binary_labels, interaction, log_odds, paper_labels, permutation_test,
    validate_interaction_spec,
)
from semantic_bias import cohens_d, write_result


class StatisticsTests(unittest.TestCase):
    def test_pairwise_equation(self):
        joint = [1] * 8 + [0] * 2
        first = [1] * 6 + [0] * 4
        second = [1] * 5 + [0] * 5
        expected = math.log(4) - math.log(1.5) - 0.0
        self.assertAlmostEqual(interaction([joint, first, second], [1, -1, -1]), expected)

    def test_three_way_expanded_equation(self):
        groups = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0],
                  [1, 1, 1, 0], [1, 1, 1, 0], [1, 1, 1, 0]]
        self.assertAlmostEqual(interaction(groups, [1, -1, -1, -1, 1, 1, 1]), 4 * math.log(3))

    def test_permutation_is_reproducible(self):
        groups = [[1] * 9 + [0], [1] + [0] * 9, [1] * 5 + [0] * 5]
        first = permutation_test(groups, [1, -1, -1], iterations=200, seed=7)
        self.assertEqual(first, permutation_test(groups, [1, -1, -1], iterations=200, seed=7))

    def test_identical_point_eight_groups_are_non_additive(self):
        groups = [[1] * 80 + [0] * 20 for _ in range(3)]
        value, p_value, significance = permutation_test(
            groups, [1, -1, -1], iterations=1000, seed=3
        )
        self.assertAlmostEqual(value, -math.log(4))
        self.assertLess(p_value, 0.05)
        self.assertEqual(significance, "moderate")

    def test_boundary_correction_uses_half_count(self):
        self.assertAlmostEqual(log_odds([1] * 10), math.log(10.5 / 0.5))
        self.assertAlmostEqual(log_odds([0] * 10), math.log(0.5 / 10.5))
        self.assertAlmostEqual(log_odds([1] * 8 + [0] * 2), math.log(4))

    def test_classifier_exclusions(self):
        values, excluded = binary_labels(["female", "male", "child", "unknown", "bad"])
        self.assertEqual(values, [1, 0])
        self.assertEqual(excluded, {"child": 1, "unknown": 1, "other": 1})

    def test_paper_denominator_includes_child_as_non_female(self):
        values, adult, counts, excluded = paper_labels(
            ["female", "male", "child", "unknown"]
        )
        self.assertEqual(values, [1, 0, 0])
        self.assertEqual(adult, [1, 0])
        self.assertEqual(counts, {"female": 1, "male": 1, "child": 1})
        self.assertEqual(excluded, {"unknown": 1, "other": 0})

    def test_cohens_d(self):
        self.assertGreater(cohens_d([2, 3, 4], [0, 1, 2]), 1)

    def test_semantic_writer_creates_parent(self):
        import json
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "result.json"
            write_result(output, {"value": 1})
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), {"value": 1})


class SchemaTests(unittest.TestCase):
    def test_complete_pair_spec(self):
        spec = {"interactions": [{"name": "pair", "order": 2, "conditions": [
            {"name": str(i), "csv": f"{i}.csv"} for i in range(3)
        ]}]}
        self.assertEqual(validate_interaction_spec(spec), [])

    def test_incomplete_triple_rejected(self):
        spec = {"interactions": [{"name": "triple", "order": 3, "conditions": []}]}
        self.assertIn("exactly 7", validate_interaction_spec(spec)[0])

    def test_missing_interaction_name_rejected(self):
        spec = {"interactions": [{"name": " ", "order": 2, "conditions": [
            {"name": str(i), "csv": f"{i}.csv"} for i in range(3)
        ]}]}
        self.assertTrue(any("name must" in error for error in validate_interaction_spec(spec)))


if __name__ == "__main__":
    unittest.main()
