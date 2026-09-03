import unittest

import torch

from debias.steering import ConstantSteering
from debias.voxinstruct import (
    format_instruction,
    initialize_nar_sequences,
    steer_text_states,
)


class VoxInstructSteeringTests(unittest.TestCase):
    def test_instruction_format_matches_generation(self):
        self.assertEqual(
            format_instruction("A calm voice", "hello there"),
            'A calm voice. "hello there"',
        )

    def test_steering_shifts_valid_tokens_and_preserves_padding(self):
        states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [0.0, 0.0]]])
        mask = torch.tensor([[1, 1, 0]])
        steering = ConstantSteering(torch.tensor([0.5, -1.0]))
        shifted = steer_text_states(states, mask, steering, strength=2.0)
        expected = torch.tensor([[[2.0, 0.0], [4.0, 2.0], [0.0, 0.0]]])
        torch.testing.assert_close(shifted, expected)

    def test_negative_strength_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            steer_text_states(
                torch.zeros(1, 1, 2), torch.ones(1, 1),
                ConstantSteering(torch.ones(2)), strength=-1,
            )

    def test_nar_initialization_keeps_semantics_and_first_codebook(self):
        seqs = torch.tensor([[10, 11, 12, 13, 14]])
        initialized = initialize_nar_sequences(
            seqs, torch.tensor([3]), codebooks=3,
        )
        expected = torch.tensor([[[10, 11, 12, 13, 14],
                                  [10, 11, 12, 0, 0],
                                  [10, 11, 12, 0, 0]]])
        torch.testing.assert_close(initialized, expected)


if __name__ == "__main__":
    unittest.main()
