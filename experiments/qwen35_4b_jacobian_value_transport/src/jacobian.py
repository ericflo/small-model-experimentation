"""Pure Jacobian-coordinate operations, independent of the production model.

The tensor convention is explicit: ``J[target_dim, source_dim]`` maps a column
perturbation at the source residual to a target residual perturbation. A target
covector ``u`` therefore pulls back to ``J.T @ u`` at the source.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch


def pullback_direction(jacobian: torch.Tensor, target_covector: torch.Tensor) -> torch.Tensor:
    """Return the source direction associated with a target-layer covector."""
    if jacobian.ndim != 2 or target_covector.ndim != 1:
        raise ValueError("expected J[d_target,d_source] and u[d_target]")
    if jacobian.shape[0] != target_covector.shape[0]:
        raise ValueError("target dimensions do not match")
    return jacobian.T @ target_covector


def normalized_columns(directions: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a ``[d, k]`` direction matrix column-wise."""
    if directions.ndim != 2:
        raise ValueError("directions must have shape [d,k]")
    norms = directions.norm(dim=0, keepdim=True)
    if bool((norms <= eps).any()):
        raise ValueError("zero or near-zero direction")
    return directions / norms


def read_coordinates(residual: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Least-squares coordinates of row-vector residuals in normalized directions."""
    v = normalized_columns(directions.float().to(residual.device))
    pinv = torch.linalg.pinv(v)
    return residual.float() @ pinv.T


def swap_coordinates(
    residual: torch.Tensor,
    source_direction: torch.Tensor,
    target_direction: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Swap two least-squares coordinates and return ``(patched, delta)``.

    Both dictionary vectors are unit-normalized before coordinate reading. This
    prevents unequal vector norms from becoming an unregistered intervention scale.
    """
    if residual.shape[-1] != source_direction.numel() or source_direction.shape != target_direction.shape:
        raise ValueError("residual and direction dimensions do not match")
    v = torch.stack([source_direction, target_direction], dim=1).float().to(residual.device)
    v = normalized_columns(v)
    coordinates = residual.float() @ torch.linalg.pinv(v).T
    swapped = coordinates.flip(-1)
    delta = alpha * ((swapped - coordinates) @ v.T)
    return residual + delta.to(residual.dtype), delta


def swap_coordinates_batched(
    residual: torch.Tensor,
    source_directions: torch.Tensor,
    target_directions: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-example coordinate swaps for ``residual[B,T,d]`` and directions ``[B,d]``."""
    if residual.ndim != 3 or source_directions.ndim != 2 or target_directions.ndim != 2:
        raise ValueError("expected residual[B,T,d] and direction batches[B,d]")
    if source_directions.shape != target_directions.shape:
        raise ValueError("source and target direction batches disagree")
    if residual.shape[0] != source_directions.shape[0] or residual.shape[2] != source_directions.shape[1]:
        raise ValueError("residual and direction batch dimensions do not match")
    vectors = torch.stack([source_directions, target_directions], dim=-1).float().to(residual.device)
    norms = vectors.norm(dim=1, keepdim=True)
    if bool((norms <= 1e-8).any()):
        raise ValueError("zero or near-zero batched direction")
    vectors = vectors / norms
    pseudoinverse = torch.linalg.pinv(vectors)
    coordinates = torch.einsum("btd,bkd->btk", residual.float(), pseudoinverse)
    swapped = coordinates.flip(-1)
    delta = alpha * torch.einsum("btk,bdk->btd", swapped - coordinates, vectors)
    return residual + delta.to(residual.dtype), delta


def replace_coordinates(
    residual: torch.Tensor,
    directions: torch.Tensor,
    replacement: torch.Tensor,
    *,
    alpha: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set selected coordinates toward ``replacement`` by fraction ``alpha``."""
    v = normalized_columns(directions.float().to(residual.device))
    current = residual.float() @ torch.linalg.pinv(v).T
    desired = replacement.float().to(residual.device).expand_as(current)
    delta = alpha * ((desired - current) @ v.T)
    return residual + delta.to(residual.dtype), delta


def project_onto_span(vector: torch.Tensor, directions: torch.Tensor) -> torch.Tensor:
    """Orthogonal projection onto the span of normalized direction columns."""
    v = normalized_columns(directions.float().to(vector.device))
    projected = (vector.float() @ torch.linalg.pinv(v).T) @ v.T
    return projected.to(vector.dtype)


def orthogonal_norm_matched(
    reference: torch.Tensor,
    directions: torch.Tensor,
    *,
    generator: torch.Generator,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Random vector orthogonal to ``directions`` with ``reference`` norm."""
    random = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    random = random.to(reference)
    random = random - project_onto_span(random, directions)
    norm = random.norm()
    if float(norm) <= eps:
        raise RuntimeError("random orthogonal control collapsed")
    return random * (reference.norm() / norm)


def equal_pair_average(
    pair_jacobians: Iterable[tuple[int, int, torch.Tensor]],
    *,
    require_causal: bool = True,
) -> torch.Tensor:
    """Average explicit source/target Jacobians with equal pair weight.

    Each item is ``(source_position, target_position, J)``. This utility pins the
    estimator convention that differs from summing all future targets first and
    then averaging source positions.
    """
    matrices: list[torch.Tensor] = []
    shape: tuple[int, ...] | None = None
    for source, target, matrix in pair_jacobians:
        if require_causal and target < source:
            raise ValueError("target position precedes source position")
        if matrix.ndim != 2:
            raise ValueError("pair Jacobian must be a matrix")
        if shape is None:
            shape = tuple(matrix.shape)
        elif tuple(matrix.shape) != shape:
            raise ValueError("pair Jacobians disagree on shape")
        matrices.append(matrix.float())
    if not matrices:
        raise ValueError("at least one source/target pair is required")
    return torch.stack(matrices).mean(dim=0)


def greedy_sparse_nonnegative(
    vector: torch.Tensor,
    dictionary: torch.Tensor,
    *,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Small deterministic nonnegative pursuit for a ``[d,n]`` dictionary.

    This is used for controls, not as evidence that the entire residual is in a
    proper linear subspace. Coefficients are refit by least squares and clipped
    nonnegative after every greedy selection.
    """
    if vector.ndim != 1 or dictionary.ndim != 2 or dictionary.shape[0] != vector.numel():
        raise ValueError("expected vector[d] and dictionary[d,n]")
    if not 1 <= k <= dictionary.shape[1]:
        raise ValueError("k must be between 1 and dictionary width")
    d = normalized_columns(dictionary.to(vector))
    residual = vector.clone()
    selected: list[int] = []
    coefficients = torch.empty(0, device=vector.device, dtype=vector.dtype)
    reconstruction = torch.zeros_like(vector)
    for _ in range(k):
        scores = d.T @ residual
        for index in selected:
            scores[index] = -torch.inf
        index = int(torch.argmax(scores).item())
        if not torch.isfinite(scores[index]) or float(scores[index]) <= 0:
            break
        selected.append(index)
        active = d[:, selected]
        coefficients = torch.linalg.lstsq(active, vector[:, None]).solution[:, 0].clamp_min(0)
        reconstruction = active @ coefficients
        residual = vector - reconstruction
    return reconstruction, residual, selected
