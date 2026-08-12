import tempfile
import unittest
from pathlib import Path

import torch

from debias.comparators import fit_gender_direction, fit_random_projection
from debias.leace import LeaceEraser
from debias.probe import LinearProbe


def _gendered_activations(seed=3):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0, 1] * 100)
    concept = (labels.float() * 2 - 1).unsqueeze(1)
    noise = torch.randn(200, 5, generator=generator)
    direction = torch.tensor([[3.0, -2.0, 1.0, 0.5, -1.0]])
    return noise + concept @ direction, labels


class ComparatorTests(unittest.TestCase):
    def test_gender_direction_removes_mean_difference(self):
        activations, labels = _gendered_activations()
        eraser = fit_gender_direction(activations, labels)
        erased = eraser(activations)
        mean_gap = erased[labels == 1].mean(dim=0) - erased[labels == 0].mean(dim=0)

        self.assertEqual(eraser.concept_rank, 1)
        self.assertLess(mean_gap.abs().max().item(), 1e-4)

    def test_gender_direction_matrix_is_a_projection(self):
        activations, labels = _gendered_activations()
        eraser = fit_gender_direction(activations, labels)
        matrix = eraser.matrix
        torch.testing.assert_close(matrix @ matrix, matrix, atol=1e-6, rtol=1e-6)
        self.assertEqual(eraser.metadata["method"], "gender_direction")

    def test_gender_direction_requires_two_classes(self):
        activations, _ = _gendered_activations()
        with self.assertRaises(ValueError):
            fit_gender_direction(activations, torch.zeros(200, dtype=torch.long))

    def test_random_projection_rank_and_determinism(self):
        activations, _ = _gendered_activations()
        a = fit_random_projection(activations, rank=2, seed=11)
        b = fit_random_projection(activations, rank=2, seed=11)
        c = fit_random_projection(activations, rank=2, seed=12)

        self.assertEqual(a.concept_rank, 2)
        torch.testing.assert_close(a.matrix, b.matrix)
        self.assertGreater((a.matrix - c.matrix).abs().max().item(), 1e-6)

    def test_random_projection_preserves_gender_more_than_leace(self):
        activations, labels = _gendered_activations()
        leace = LeaceEraser.fit(activations, labels)
        random = fit_random_projection(activations, rank=leace.concept_rank, seed=5)

        def auc(eraser):
            x = eraser(activations)
            return LinearProbe.fit(x, labels, seed=0).evaluate(x, labels)["auc"]

        self.assertLess(auc(leace), 0.6)
        self.assertGreater(auc(random), auc(leace))

    def test_comparator_artifacts_round_trip_as_leace_v1(self):
        activations, labels = _gendered_activations()
        for eraser in (
            fit_gender_direction(activations, labels, metadata={"model_id": "t"}),
            fit_random_projection(activations, rank=1, seed=0, metadata={"model_id": "t"}),
        ):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "eraser.pt"
                eraser.save(path)
                restored = LeaceEraser.load(path)
            torch.testing.assert_close(restored(activations), eraser(activations))
            self.assertEqual(restored.metadata["model_id"], "t")


if __name__ == "__main__":
    unittest.main()
