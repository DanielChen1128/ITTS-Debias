"""Baseline comparison erasers that share the LEACE affine-map interface.

Each builder returns a :class:`~debias.leace.LeaceEraser` so the artifacts save,
load, and flow through ``generate_with_eraser`` identically to the main method.
An eraser removes a subspace spanned by orthonormal columns ``U`` via the affine
map ``x -> (x - mean) @ (I - U U^T)^T + mean`` (``matrix`` is symmetric here).
"""

import torch

from debias.leace import LeaceEraser


def _projection_eraser(mean, basis, concept_rank, metadata):
    """Build an eraser that removes the span of ``basis`` (orthonormal columns)."""
    features = mean.numel()
    identity = torch.eye(features, dtype=mean.dtype, device=mean.device)
    if basis.numel() == 0:
        matrix = identity
    else:
        matrix = identity - basis @ basis.T
    return LeaceEraser(mean, matrix, concept_rank, metadata)


def fit_gender_direction(activations, labels, *, metadata=None):
    """Rank-1 eraser removing the class mean-difference direction.

    A deliberately naive baseline: it erases only the single direction that
    separates the two label means, ignoring covariance whitening.
    """
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must be [samples, features] with >=2 samples")
    if labels.shape[0] != activations.shape[0]:
        raise ValueError("labels and activations must have matching sample counts")
    x = activations.detach().to(dtype=torch.float64)
    classes = torch.unique(labels.detach(), sorted=True)
    if classes.numel() != 2:
        raise ValueError("gender-direction baseline requires exactly two classes")
    mask = labels.detach() == classes[1]
    direction = x[mask].mean(dim=0) - x[~mask].mean(dim=0)
    norm = torch.linalg.norm(direction)
    if norm == 0:
        raise ValueError("class means coincide; no gender direction to remove")
    unit = (direction / norm).unsqueeze(1)
    meta = dict(metadata or {})
    meta.setdefault("method", "gender_direction")
    return _projection_eraser(x.mean(dim=0), unit, 1, meta)


def fit_random_projection(activations, rank, *, seed=0, metadata=None):
    """Eraser removing a random orthonormal subspace of the given rank.

    Rank-matched null control: it destroys the same number of directions as the
    real method but chosen independently of the gender labels.
    """
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must be [samples, features] with >=2 samples")
    features = activations.shape[1]
    rank = int(rank)
    if not 0 <= rank <= features:
        raise ValueError(f"rank must be in [0, {features}]")
    x = activations.detach().to(dtype=torch.float64)
    mean = x.mean(dim=0)
    meta = dict(metadata or {})
    meta.setdefault("method", "random_projection")
    meta.setdefault("seed", int(seed))
    if rank == 0:
        return _projection_eraser(mean, x.new_zeros((features, 0)), 0, meta)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    gaussian = torch.randn(features, rank, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(gaussian, mode="reduced")
    return _projection_eraser(mean, basis, rank, meta)
