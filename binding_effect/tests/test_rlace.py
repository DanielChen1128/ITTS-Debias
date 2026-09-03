import tempfile
import unittest
from pathlib import Path

import torch

from debias.rlace import (
    _project_fantope,
    fit_rlace,
)
from debias.leace import LeaceEraser


def _gendered_activations(seed=3):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.tensor([0, 1] * 100)
    concept = (labels.float() * 2 - 1).unsqueeze(1)
    noise = torch.randn(200, 5, generator=generator)
    direction = torch.tensor([[3.0, -2.0, 1.0, 0.5, -1.0]])
    return noise + concept @ direction, labels


class RlaceTests(unittest.TestCase):
    def test_fantope_projection_has_bounded_spectrum_and_requested_trace(self):
        matrix = torch.tensor([
            [2.0, 0.4, 0.0],
            [0.4, -1.0, 0.2],
            [0.0, 0.2, 0.5],
        ], dtype=torch.float64)
        projected = _project_fantope(matrix, rank=2)
        eigenvalues = torch.linalg.eigvalsh(projected)

        torch.testing.assert_close(projected, projected.T)
        self.assertGreaterEqual(eigenvalues.min().item(), -1e-10)
        self.assertLessEqual(eigenvalues.max().item(), 1 + 1e-10)
        self.assertAlmostEqual(projected.trace().item(), 2.0, places=8)

    def test_rlace_returns_deterministic_rank_projection(self):
        activations, labels = _gendered_activations()
        kwargs = {
            "rank": 2,
            "epochs": 8,
            "adversary_steps": 2,
            "seed": 13,
            "device": "cpu",
        }
        a = fit_rlace(activations, labels, **kwargs)
        b = fit_rlace(activations, labels, **kwargs)

        self.assertEqual(a.concept_rank, 2)
        self.assertEqual(a.metadata["method"], "rlace")
        torch.testing.assert_close(a.matrix, b.matrix)
        torch.testing.assert_close(a.matrix @ a.matrix, a.matrix, atol=1e-6, rtol=1e-6)
        self.assertAlmostEqual(torch.trace(torch.eye(5) - a.matrix).item(), 2.0, places=6)

    def test_rlace_requires_two_classes(self):
        activations, _ = _gendered_activations()
        with self.assertRaisesRegex(ValueError, "exactly two classes"):
            fit_rlace(activations, torch.zeros(200, dtype=torch.long), rank=1, epochs=1)

    def test_artifact_round_trip_as_leace_v1(self):
        activations, labels = _gendered_activations()
        eraser = fit_rlace(
            activations, labels, rank=1, epochs=2, adversary_steps=1,
            seed=0, device="cpu", metadata={"model_id": "t"},
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eraser.pt"
            eraser.save(path)
            restored = LeaceEraser.load(path)
        torch.testing.assert_close(restored(activations), eraser(activations))
        self.assertEqual(restored.metadata["model_id"], "t")


if __name__ == "__main__":
    unittest.main()
