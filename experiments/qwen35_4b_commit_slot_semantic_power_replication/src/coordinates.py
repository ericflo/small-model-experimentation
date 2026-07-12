"""Frozen-lens coordinate readout contracts."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DictionaryStats:
    singular_values: tuple[float, ...]
    effective_rank: int
    condition_number: float


def normalized_dictionary(
    directions: torch.Tensor, *, eps: float = 1e-12
) -> torch.Tensor:
    if directions.ndim != 2:
        raise ValueError("directions must have shape [d_model,n_coordinates]")
    value = directions.float()
    norms = value.norm(dim=0, keepdim=True)
    if bool((norms <= eps).any()) or not bool(torch.isfinite(value).all()):
        raise ValueError("dictionary contains zero or non-finite directions")
    return value / norms


def dictionary_stats(directions: torch.Tensor, *, rtol: float) -> DictionaryStats:
    if not 0.0 < rtol < 1.0:
        raise ValueError("rtol must be between zero and one")
    singular = torch.linalg.svdvals(normalized_dictionary(directions))
    threshold = singular[0] * rtol
    rank = int((singular > threshold).sum().item())
    smallest = singular[rank - 1] if rank else torch.tensor(0.0)
    condition = float((singular[0] / smallest).item()) if rank else float("inf")
    return DictionaryStats(
        singular_values=tuple(float(value) for value in singular.tolist()),
        effective_rank=rank,
        condition_number=condition,
    )


def pseudoinverse(
    directions: torch.Tensor, *, rtol: float
) -> tuple[torch.Tensor, torch.Tensor]:
    dictionary = normalized_dictionary(directions)
    return dictionary, torch.linalg.pinv(dictionary, rtol=rtol)


def read_coordinates(
    residual: torch.Tensor, directions: torch.Tensor, *, rtol: float
) -> torch.Tensor:
    _dictionary, inverse = pseudoinverse(
        directions.to(residual.device), rtol=rtol
    )
    return residual.float() @ inverse.T


def non_j_random_dictionary(
    directions: torch.Tensor,
    *,
    width: int,
    seed: int,
    rtol: float,
    max_span_projection: float,
) -> tuple[torch.Tensor, float]:
    """Return deterministic orthonormal columns outside the frozen J span."""
    if width < 1 or width > directions.shape[0] - directions.shape[1]:
        raise ValueError("invalid non-J random dictionary width")
    dictionary, inverse = pseudoinverse(directions.cpu(), rtol=rtol)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(
        directions.shape[0], width, generator=generator, dtype=torch.float32
    )
    projected = dictionary @ (inverse @ random)
    orthogonal = random - projected
    basis, _triangular = torch.linalg.qr(orthogonal, mode="reduced")
    residual_projection = dictionary @ (inverse @ basis)
    fraction = float(
        residual_projection.norm(dim=0).max()
        / basis.norm(dim=0).clamp_min(1e-12).min()
    )
    if not bool(torch.isfinite(basis).all()) or fraction > max_span_projection:
        raise RuntimeError(
            f"non-J random dictionary leaked into J span: {fraction}"
        )
    return basis, fraction
