"""Linear concept erasure using the closed-form LEACE intervention."""

from pathlib import Path

import torch


class LeaceEraser:
    """Affine map that removes linear information about fitted labels."""

    def __init__(self, mean, matrix, concept_rank, metadata=None):
        if mean.ndim != 1 or matrix.shape != (mean.numel(), mean.numel()):
            raise ValueError("mean and matrix dimensions do not agree")
        self.mean = mean.detach()
        self.matrix = matrix.detach()
        self.concept_rank = int(concept_rank)
        self.metadata = dict(metadata or {})

    @classmethod
    def fit(cls, activations, labels, *, rtol=1e-7, metadata=None):
        """Fit LEACE in float64 from `[samples, features]` activations."""
        if activations.ndim != 2 or activations.shape[0] < 2:
            raise ValueError("activations must have shape [samples, features] with at least two samples")
        if labels.shape[0] != activations.shape[0]:
            raise ValueError("labels and activations must contain the same number of samples")

        x = activations.detach().to(dtype=torch.float64)
        if labels.ndim == 1:
            _, encoded = torch.unique(labels.detach(), sorted=True, return_inverse=True)
            z = torch.nn.functional.one_hot(encoded).to(dtype=torch.float64)
        elif labels.ndim == 2:
            z = labels.detach().to(dtype=torch.float64)
        else:
            raise ValueError("labels must have shape [samples] or [samples, label_features]")

        mean = x.mean(dim=0)
        xc = x - mean
        zc = z - z.mean(dim=0)
        covariance = xc.T @ xc / x.shape[0]
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
        maximum = eigenvalues.max().clamp_min(0)
        keep = eigenvalues > rtol * maximum
        if not torch.any(keep):
            raise ValueError("activation covariance has no nonzero directions")

        basis = eigenvectors[:, keep]
        values = eigenvalues[keep]
        whitener = (basis / values.sqrt()) @ basis.T
        dewhitener = (basis * values.sqrt()) @ basis.T
        cross_covariance = whitener @ (xc.T @ zc / x.shape[0])
        left, singular_values, _ = torch.linalg.svd(cross_covariance, full_matrices=False)
        if singular_values.numel() == 0 or singular_values[0] == 0:
            rank = 0
        else:
            rank = int((singular_values > rtol * singular_values[0]).sum().item())

        identity = torch.eye(x.shape[1], dtype=x.dtype, device=x.device)
        if rank:
            concept_projection = left[:, :rank] @ left[:, :rank].T
            matrix = identity - dewhitener @ concept_projection @ whitener
        else:
            matrix = identity
        return cls(mean, matrix, rank, metadata)

    def __call__(self, activations):
        if activations.shape[-1] != self.mean.numel():
            raise ValueError(
                f"expected activation width {self.mean.numel()}, got {activations.shape[-1]}"
            )
        mean = self.mean.to(device=activations.device, dtype=activations.dtype)
        matrix = self.matrix.to(device=activations.device, dtype=activations.dtype)
        return (activations - mean) @ matrix.T + mean

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "leace-v1",
                "mean": self.mean.cpu(),
                "matrix": self.matrix.cpu(),
                "concept_rank": self.concept_rank,
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path, *, map_location="cpu"):
        state = torch.load(path, map_location=map_location, weights_only=True)
        if state.get("format") != "leace-v1":
            raise ValueError("unsupported LEACE artifact format")
        return cls(state["mean"], state["matrix"], state["concept_rank"], state.get("metadata"))
