"""Constant steering offsets for pooled conditioning states."""

from pathlib import Path

import torch


class ConstantSteering:
    def __init__(self, offset, metadata=None):
        if offset.ndim != 1:
            raise ValueError("offset must have shape [features]")
        self.offset = offset.detach()
        self.metadata = dict(metadata or {})

    def __call__(self, features):
        if features.shape[-1] != self.offset.numel():
            raise ValueError(
                f"expected activation width {self.offset.numel()}, got {features.shape[-1]}"
            )
        return features + self.offset.to(device=features.device, dtype=features.dtype)

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "constant-steering-v1",
            "offset": self.offset.cpu(),
            "metadata": self.metadata,
        }, path)

    @classmethod
    def load(cls, path, *, map_location="cpu"):
        state = torch.load(path, map_location=map_location, weights_only=True)
        if state.get("format") != "constant-steering-v1":
            raise ValueError("unsupported constant steering format")
        return cls(state["offset"], state.get("metadata"))
