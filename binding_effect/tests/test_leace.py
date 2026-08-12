import tempfile
import unittest
from pathlib import Path

import torch

from debias.leace import LeaceEraser
from debias.parler import (
    encode_description_states,
    generate_with_eraser,
    has_explicit_gender_command,
    masked_mean,
)


class _Config:
    def __init__(self, hidden_size, cross_attention_hidden_size=None):
        self.hidden_size = hidden_size
        self.cross_attention_hidden_size = cross_attention_hidden_size


class _Output:
    def __init__(self, states):
        self.last_hidden_state = states


class _Encoder:
    config = _Config(2)

    def __call__(self, input_ids, attention_mask, return_dict):
        return _Output(input_ids.float().unsqueeze(-1).repeat(1, 1, 2))


class _Decoder:
    config = _Config(2)


class _Model:
    decoder = _Decoder()

    def __init__(self):
        self.encoder = _Encoder()
        self.generated_states = None

    def get_text_encoder(self):
        return self.encoder

    def generate(self, encoder_outputs, **kwargs):
        self.generated_states = encoder_outputs.last_hidden_state
        return self.generated_states


class LeaceTests(unittest.TestCase):
    def test_removes_fitted_linear_cross_covariance(self):
        generator = torch.Generator().manual_seed(7)
        labels = torch.tensor([0, 1] * 100)
        concept = (labels.float() * 2 - 1).unsqueeze(1)
        noise = torch.randn(200, 5, generator=generator)
        activations = noise + concept @ torch.tensor([[3.0, -2.0, 1.0, 0.5, -1.0]])

        eraser = LeaceEraser.fit(activations, labels)
        erased = eraser(activations)
        centered_x = erased - erased.mean(dim=0)
        centered_y = concept - concept.mean(dim=0)

        self.assertEqual(eraser.concept_rank, 1)
        self.assertLess((centered_x.T @ centered_y).abs().max().item(), 1e-4)

    def test_zero_cross_covariance_is_identity(self):
        activations = torch.tensor([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
        labels = torch.tensor([0, 0, 1, 1])
        eraser = LeaceEraser.fit(activations, labels)
        self.assertEqual(eraser.concept_rank, 0)
        torch.testing.assert_close(eraser(activations), activations)

    def test_artifact_round_trip(self):
        activations = torch.tensor([[-1.0, 2.0], [1.0, -2.0], [-2.0, 1.0], [2.0, -1.0]])
        eraser = LeaceEraser.fit(activations, torch.tensor([0, 1, 0, 1]), metadata={"model_id": "test"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eraser.pt"
            eraser.save(path)
            restored = LeaceEraser.load(path)
        torch.testing.assert_close(restored(activations), eraser(activations))
        self.assertEqual(restored.metadata["model_id"], "test")

    def test_masked_mean_ignores_padding(self):
        states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        result = masked_mean(states, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(result, torch.tensor([[2.0, 3.0]]))

    def test_explicit_gender_detection(self):
        self.assertTrue(has_explicit_gender_command("Use a clearly female voice."))
        self.assertTrue(has_explicit_gender_command("A masculine speaker"))
        self.assertFalse(has_explicit_gender_command("Speak carefully and confidently."))
        self.assertFalse(has_explicit_gender_command("Describe the human condition."))

    def test_encoder_states_remain_masked(self):
        model = _Model()
        states = encode_description_states(model, torch.tensor([[2, 9]]), torch.tensor([[1, 0]]))
        torch.testing.assert_close(states, torch.tensor([[[2.0, 2.0], [0.0, 0.0]]]))

    def test_generation_erases_implicit_but_bypasses_explicit(self):
        model = _Model()
        eraser = lambda states: states + 10
        ids = torch.tensor([[2, 0]])
        mask = torch.tensor([[1, 0]])
        generate_with_eraser(model, eraser, "A calm speaker", ids, mask)
        torch.testing.assert_close(
            model.generated_states, torch.tensor([[[12.0, 12.0], [0.0, 0.0]]])
        )
        generate_with_eraser(model, eraser, "A female speaker", ids, mask)
        torch.testing.assert_close(
            model.generated_states, torch.tensor([[[2.0, 2.0], [0.0, 0.0]]])
        )


if __name__ == "__main__":
    unittest.main()
