import tempfile
import unittest
from pathlib import Path

import torch

from debias.steering import ConstantSteering


class ConstantSteeringTests(unittest.TestCase):
    def test_applies_offset_and_preserves_dtype(self):
        steering = ConstantSteering(torch.tensor([1.0, -2.0]))
        values = torch.tensor([[3.0, 4.0]], dtype=torch.float16)
        result = steering(values)
        torch.testing.assert_close(result, torch.tensor([[4.0, 2.0]], dtype=torch.float16))
        self.assertEqual(result.dtype, values.dtype)

    def test_round_trip(self):
        steering = ConstantSteering(torch.tensor([1.0, 2.0]), {"direction": "female-to-male"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "steering.pt"
            steering.save(path)
            loaded = ConstantSteering.load(path)
        torch.testing.assert_close(loaded.offset, steering.offset)
        self.assertEqual(loaded.metadata, steering.metadata)

    def test_rejects_wrong_width(self):
        steering = ConstantSteering(torch.ones(2))
        with self.assertRaisesRegex(ValueError, "expected activation width 2"):
            steering(torch.ones(1, 3))


if __name__ == "__main__":
    unittest.main()
