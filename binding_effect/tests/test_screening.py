import unittest

import torch

from debias.probe import LinearProbe, roc_auc
from screen_encoder import _select_spillover


class RocAucTests(unittest.TestCase):
    def test_perfect_separation(self):
        scores = torch.tensor([0.1, 0.2, 0.8, 0.9])
        labels = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(scores, labels), 1.0, places=6)

    def test_inverted_separation(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        labels = torch.tensor([0, 0, 1, 1])
        self.assertAlmostEqual(roc_auc(scores, labels), 0.0, places=6)

    def test_ties_give_half(self):
        scores = torch.tensor([0.5, 0.5, 0.5, 0.5])
        labels = torch.tensor([0, 1, 0, 1])
        self.assertAlmostEqual(roc_auc(scores, labels), 0.5, places=6)

    def test_requires_both_classes(self):
        with self.assertRaises(ValueError):
            roc_auc(torch.tensor([0.1, 0.2]), torch.tensor([1, 1]))


class LinearProbeTests(unittest.TestCase):
    def test_separable_data_is_learned(self):
        generator = torch.Generator().manual_seed(0)
        labels = torch.tensor([0, 1]).repeat(100)
        signal = (labels.double() * 2 - 1).unsqueeze(1)
        features = torch.randn(200, 4, generator=generator).double() + signal @ torch.tensor([[4.0, 0, 0, 0]]).double()
        probe = LinearProbe.fit(features, labels)
        metrics = probe.evaluate(features, labels)
        self.assertGreater(metrics["auc"], 0.95)
        self.assertGreater(metrics["accuracy"], 0.9)

    def test_pure_noise_is_near_chance(self):
        generator = torch.Generator().manual_seed(1)
        labels = torch.tensor([0, 1]).repeat(150)
        features = torch.randn(300, 6, generator=generator)
        probe = LinearProbe.fit(features, labels)
        held_out = torch.randn(300, 6, generator=generator)
        metrics = probe.evaluate(held_out, labels)
        self.assertLess(abs(metrics["auc"] - 0.5), 0.15)

    def test_rejects_non_binary_labels(self):
        with self.assertRaises(ValueError):
            LinearProbe.fit(torch.randn(6, 3), torch.tensor([0, 1, 2, 0, 1, 2]))


class SpilloverSelectionTests(unittest.TestCase):
    def test_filters_requested_split_and_deduplicates_descriptions(self):
        records = [
            {"axis": "career", "stereotype_gender": "female", "lemma": "career:a", "description": "a"},
            {"axis": "career", "stereotype_gender": "female", "lemma": "career:a", "description": "a"},
            {"axis": "career", "stereotype_gender": "male", "lemma": "career:b", "description": "b"},
            {"axis": "persona", "stereotype_gender": "female", "lemma": "persona:c", "description": "c"},
        ]
        selected = _select_spillover(
            records, {"career:a": "dev", "career:b": "test", "persona:c": "dev"}, "dev"
        )
        self.assertEqual([record["description"] for record in selected], ["a"])


if __name__ == "__main__":
    unittest.main()
