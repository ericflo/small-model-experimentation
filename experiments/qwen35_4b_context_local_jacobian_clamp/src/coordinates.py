"""Targeted coordinate clamping and exact perturbation controls."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class DictionaryStats:
    singular_values: tuple[float, ...]
    effective_rank: int
    condition_number: float


def normalized_dictionary(directions: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if directions.ndim != 2:
        raise ValueError("directions must have shape [d_model, n_coordinates]")
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


def pseudoinverse(directions: torch.Tensor, *, rtol: float) -> tuple[torch.Tensor, torch.Tensor]:
    dictionary = normalized_dictionary(directions)
    return dictionary, torch.linalg.pinv(dictionary, rtol=rtol)


def read_coordinates(residual: torch.Tensor, directions: torch.Tensor, *, rtol: float) -> torch.Tensor:
    dictionary, inverse = pseudoinverse(directions.to(residual.device), rtol=rtol)
    del dictionary
    return residual.float() @ inverse.T


def replace_coordinates(
    residual: torch.Tensor,
    directions: torch.Tensor,
    desired: torch.Tensor,
    *,
    rtol: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Set row-vector residual coordinates to fixed desired values."""
    dictionary, inverse = pseudoinverse(directions.to(residual.device), rtol=rtol)
    current = residual.float() @ inverse.T
    target = desired.float().to(residual.device)
    if target.shape != current.shape:
        raise ValueError(f"desired coordinate shape {target.shape} != current {current.shape}")
    delta = (target - current) @ dictionary.T
    return residual + delta.to(residual.dtype), delta


def orthogonal_norm_matched(
    reference_delta: torch.Tensor,
    directions: torch.Tensor,
    *,
    generator: torch.Generator,
    rtol: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Return row-wise random controls orthogonal to the dictionary span."""
    if reference_delta.ndim != 2:
        raise ValueError("reference_delta must have shape [batch,d_model]")
    dictionary, inverse = pseudoinverse(directions.to(reference_delta.device), rtol=rtol)
    random = torch.randn(
        reference_delta.shape,
        generator=generator,
        dtype=torch.float32,
        device=reference_delta.device,
    )
    projected = (random @ inverse.T) @ dictionary.T
    orthogonal = random - projected
    norms = orthogonal.norm(dim=-1, keepdim=True)
    targets = reference_delta.float().norm(dim=-1, keepdim=True)
    if bool((norms <= eps).any()):
        raise RuntimeError("orthogonal random control collapsed")
    return orthogonal * (targets / norms)


def relative_norm_error(reference: torch.Tensor, candidate: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    reference_norm = reference.float().norm(dim=-1)
    candidate_norm = candidate.float().norm(dim=-1)
    return (candidate_norm - reference_norm).abs() / reference_norm.clamp_min(eps)
