from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coordinates import (  # noqa: E402
    dictionary_stats,
    normalized_dictionary,
    orthogonal_norm_matched,
    read_coordinates,
    relative_norm_error,
    replace_coordinates,
)


def test_fixed_coordinate_replacement_is_exact_and_idempotent() -> None:
    generator = torch.Generator().manual_seed(7)
    directions = torch.randn(32, 6, generator=generator)
    residual = torch.randn(4, 32, generator=generator)
    desired = torch.randn(4, 6, generator=generator)
    patched, first_delta = replace_coordinates(residual, directions, desired, rtol=1e-6)
    observed = read_coordinates(patched, directions, rtol=1e-6)
    assert torch.allclose(observed, desired, atol=2e-5, rtol=2e-5)
    patched_twice, second_delta = replace_coordinates(patched, directions, desired, rtol=1e-6)
    assert torch.allclose(patched_twice, patched, atol=2e-5, rtol=2e-5)
    assert first_delta.norm() > 0
    assert second_delta.norm() < 1e-4


def test_random_control_is_span_orthogonal_and_rowwise_norm_matched() -> None:
    generator = torch.Generator().manual_seed(11)
    directions = torch.randn(40, 8, generator=generator)
    reference = torch.randn(5, 40, generator=generator)
    control = orthogonal_norm_matched(
        reference,
        directions,
        generator=torch.Generator().manual_seed(12),
        rtol=1e-6,
    )
    dictionary = normalized_dictionary(directions)
    assert float((control @ dictionary).abs().max()) < 2e-5
    assert float(relative_norm_error(reference, control).max()) < 1e-6


def test_dictionary_stats_detect_rank() -> None:
    generator = torch.Generator().manual_seed(17)
    full = torch.randn(30, 7, generator=generator)
    assert dictionary_stats(full, rtol=1e-6).effective_rank == 7
    deficient = torch.cat([full[:, :6], full[:, :1]], dim=1)
    assert dictionary_stats(deficient, rtol=1e-6).effective_rank == 6
