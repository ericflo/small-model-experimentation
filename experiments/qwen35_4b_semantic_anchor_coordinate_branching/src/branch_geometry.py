"""Balanced J branches and exact-Gram J-orthogonal controls."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BranchStats:
    width: int
    rank: int
    rms_norm: float
    maximum_sum_abs: float
    gram_frobenius: float


def normalized_dictionary(directions: torch.Tensor, *, eps: float = 1e-12) -> torch.Tensor:
    if directions.ndim != 2:
        raise ValueError("directions must be [d_model,n_concepts]")
    value = directions.float()
    norms = value.norm(dim=0, keepdim=True)
    if not bool(torch.isfinite(value).all()) or bool((norms <= eps).any()):
        raise ValueError("invalid direction dictionary")
    return value / norms


def branch_stats(branches: torch.Tensor, *, rtol: float = 1e-5) -> BranchStats:
    if branches.ndim != 2:
        raise ValueError("branches must be [d_model,n_branches]")
    singular = torch.linalg.svdvals(branches.float())
    rank = int((singular > singular[0] * rtol).sum().item()) if singular.numel() else 0
    norms = branches.float().norm(dim=0)
    return BranchStats(
        width=int(branches.shape[1]),
        rank=rank,
        rms_norm=float(torch.sqrt(torch.mean(norms.square())).item()),
        maximum_sum_abs=float(branches.float().sum(dim=1).abs().max().item()),
        gram_frobenius=float(torch.linalg.norm(branches.float().T @ branches.float()).item()),
    )


def balanced_j_branches(
    directions: torch.Tensor,
    *,
    public_concepts: int,
    target_rms_norm: float,
) -> torch.Tensor:
    if public_concepts < 2 or public_concepts > directions.shape[1]:
        raise ValueError("invalid public concept count")
    if target_rms_norm <= 0:
        raise ValueError("target RMS norm must be positive")
    public = normalized_dictionary(directions)[:, :public_concepts]
    centered = public - public.mean(dim=1, keepdim=True)
    rms = torch.sqrt(torch.mean(centered.norm(dim=0).square()))
    if float(rms) <= 1e-12:
        raise ValueError("centered public dictionary collapsed")
    return centered * (float(target_rms_norm) / rms)


def gram_matched_non_j(
    full_directions: torch.Tensor,
    j_branches: torch.Tensor,
    *,
    seed: int,
    rtol: float,
) -> torch.Tensor:
    """Rotate branch left singular vectors outside the complete J span."""

    dictionary = normalized_dictionary(full_directions)
    inverse = torch.linalg.pinv(dictionary, rtol=rtol)
    u, singular, vh = torch.linalg.svd(j_branches.float(), full_matrices=False)
    threshold = singular[0] * rtol
    rank = int((singular > threshold).sum().item())
    if rank < 1 or rank > full_directions.shape[0] - full_directions.shape[1]:
        raise ValueError("invalid branch rank for non-J rotation")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    random = torch.randn(
        full_directions.shape[0], rank, generator=generator, dtype=torch.float32
    )
    outside = random - dictionary @ (inverse @ random)
    q, _r = torch.linalg.qr(outside, mode="reduced")
    control = q @ (singular[:rank, None] * vh[:rank, :])
    return control


def geometry_receipt(
    full_directions: torch.Tensor,
    j_branches: torch.Tensor,
    non_j_branches: torch.Tensor,
    *,
    rtol: float,
) -> dict[str, object]:
    dictionary = normalized_dictionary(full_directions)
    inverse = torch.linalg.pinv(dictionary, rtol=rtol)
    j_gram = j_branches.float().T @ j_branches.float()
    non_j_gram = non_j_branches.float().T @ non_j_branches.float()
    gram_denominator = float(torch.linalg.norm(j_gram).item())
    return {
        "j": asdict(branch_stats(j_branches, rtol=rtol)),
        "non_j": asdict(branch_stats(non_j_branches, rtol=rtol)),
        "gram_relative_error": float(
            torch.linalg.norm(j_gram - non_j_gram).item()
            / max(gram_denominator, 1e-12)
        ),
        "non_j_max_span_projection_fraction": float(
            (dictionary @ (inverse @ non_j_branches.float())).norm(dim=0).max().item()
            / non_j_branches.float().norm(dim=0).clamp_min(1e-12).min().item()
        ),
    }
