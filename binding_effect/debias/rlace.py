"""RLACE erasers that share the LEACE affine-map artifact interface.

Each builder returns a :class:`~debias.leace.LeaceEraser` so the artifacts save,
load, and flow through ``generate_with_eraser`` identically to the main method.
An eraser removes a subspace spanned by orthonormal columns ``U`` via the affine
map ``x -> (x - mean) @ (I - U U^T)^T + mean`` (``matrix`` is symmetric here).
"""

import torch
import torch.nn.functional as F

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


def _project_fantope(matrix, rank, *, iterations=60):
    """Project a symmetric matrix onto ``0 <= A <= I, trace(A) = rank``."""
    matrix = (matrix + matrix.T) / 2
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    features = eigenvalues.numel()
    rank = int(rank)
    if not 0 <= rank <= features:
        raise ValueError(f"rank must be in [0, {features}]")
    if rank == 0:
        projected_values = torch.zeros_like(eigenvalues)
    elif rank == features:
        projected_values = torch.ones_like(eigenvalues)
    else:
        lower = eigenvalues.min() - 1
        upper = eigenvalues.max()
        for _ in range(iterations):
            shift = (lower + upper) / 2
            projected_values = (eigenvalues - shift).clamp(0, 1)
            if projected_values.sum() > rank:
                lower = shift
            else:
                upper = shift
        projected_values = (eigenvalues - (lower + upper) / 2).clamp(0, 1)
    return (eigenvectors * projected_values) @ eigenvectors.T


def fit_rlace(
    activations,
    labels,
    *,
    rank,
    epochs=100,
    adversary_steps=5,
    adversary_lr=5e-2,
    projector_lr=1e-3,
    l2=1e-4,
    seed=0,
    device=None,
    metadata=None,
):
    """Fit a relaxed adversarial concept eraser and harden it to rank ``rank``.

    The learned removed-subspace matrix is constrained to the Fantope during
    optimization. The returned artifact removes its top-k eigenspace, yielding
    the same orthogonal affine-map interface as the other comparators.
    """
    if activations.ndim != 2 or activations.shape[0] < 2:
        raise ValueError("activations must be [samples, features] with >=2 samples")
    if labels.shape[0] != activations.shape[0]:
        raise ValueError("labels and activations must have matching sample counts")
    classes = torch.unique(labels.detach(), sorted=True)
    if classes.numel() != 2:
        raise ValueError("RLACE requires exactly two classes")

    features = activations.shape[1]
    rank = int(rank)
    if not 1 <= rank <= features:
        raise ValueError(f"rank must be in [1, {features}]")
    if int(epochs) < 1 or int(adversary_steps) < 1:
        raise ValueError("epochs and adversary_steps must be positive")

    optimization_device = torch.device(device or activations.device)
    dtype = torch.float32 if optimization_device.type == "cuda" else torch.float64
    x = activations.detach().to(device=optimization_device, dtype=dtype)
    y = (labels.detach() == classes[1]).to(device=optimization_device, dtype=dtype)
    mean = x.mean(dim=0)
    centered = x - mean
    scale = centered.square().mean().sqrt().clamp_min(1e-8)
    normalized = centered / scale

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    class_gap = (
        activations.detach()[labels.detach() == classes[1]].to(dtype=torch.float64).mean(dim=0)
        - activations.detach()[labels.detach() == classes[0]].to(dtype=torch.float64).mean(dim=0)
    )
    if torch.linalg.norm(class_gap) > 1e-10:
        initial = torch.cat((
            class_gap.unsqueeze(1),
            torch.randn(features, rank - 1, generator=generator, dtype=torch.float64),
        ), dim=1)
    else:
        initial = torch.randn(features, rank, generator=generator, dtype=torch.float64)
    initial_basis, _ = torch.linalg.qr(initial, mode="reduced")
    removed = (initial_basis @ initial_basis.T).to(optimization_device, dtype=dtype)
    removed.requires_grad_(True)
    weight = torch.zeros(features, device=optimization_device, dtype=dtype, requires_grad=True)
    bias = torch.zeros(1, device=optimization_device, dtype=dtype, requires_grad=True)
    adversary_optimizer = torch.optim.Adam((weight, bias), lr=float(adversary_lr))
    projector_optimizer = torch.optim.Adam((removed,), lr=float(projector_lr))

    diagnostics = []
    report_every = max(1, int(epochs) // 10)
    for epoch in range(int(epochs)):
        for _ in range(int(adversary_steps)):
            adversary_optimizer.zero_grad()
            erased = normalized - normalized @ removed.detach()
            logits = erased @ weight + bias
            adversary_loss = F.binary_cross_entropy_with_logits(logits, y)
            (adversary_loss + float(l2) * weight.square().sum()).backward()
            adversary_optimizer.step()

        projector_optimizer.zero_grad()
        erased = normalized - normalized @ removed
        projector_loss = F.binary_cross_entropy_with_logits(
            erased @ weight.detach() + bias.detach(), y,
        )
        (-projector_loss).backward()
        projector_optimizer.step()
        with torch.no_grad():
            removed.copy_(_project_fantope(removed, rank))
        if epoch == 0 or (epoch + 1) % report_every == 0 or epoch + 1 == int(epochs):
            diagnostics.append({
                "epoch": epoch + 1,
                "adversary_loss": float(adversary_loss.detach().cpu()),
                "projector_loss": float(projector_loss.detach().cpu()),
            })

    relaxed = ((removed.detach() + removed.detach().T) / 2).to(dtype=torch.float64).cpu()
    eigenvalues, eigenvectors = torch.linalg.eigh(relaxed)
    basis = eigenvectors[:, -rank:]
    meta = dict(metadata or {})
    meta.update({
        "method": "rlace",
        "requested_rank": rank,
        "epochs": int(epochs),
        "adversary_steps": int(adversary_steps),
        "adversary_lr": float(adversary_lr),
        "projector_lr": float(projector_lr),
        "l2": float(l2),
        "seed": int(seed),
        "initialization": "class_mean_direction_plus_random",
        "optimization_device": str(optimization_device),
        "training_diagnostics": diagnostics,
        "relaxed_top_eigenvalues": [float(value) for value in eigenvalues[-rank:]],
    })
    return _projection_eraser(mean.to(dtype=torch.float64).cpu(), basis, rank, meta)
