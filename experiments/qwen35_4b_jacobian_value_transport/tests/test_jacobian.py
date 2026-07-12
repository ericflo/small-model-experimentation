from __future__ import annotations

import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from jacobian import (  # noqa: E402
    equal_pair_average,
    greedy_sparse_nonnegative,
    orthogonal_norm_matched,
    pullback_direction,
    read_coordinates,
    swap_coordinates,
)


def test_pullback_orientation_matches_autograd() -> None:
    weight = torch.tensor([[2.0, -1.0], [0.5, 3.0]])
    source = torch.tensor([0.3, -0.4], requires_grad=True)
    covector = torch.tensor([1.2, -0.7])
    target = weight @ source
    (gradient,) = torch.autograd.grad(covector @ target, source)
    torch.testing.assert_close(pullback_direction(weight, covector), gradient)


def test_coordinate_swap_exchanges_read_coordinates() -> None:
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 2.0, 0.0])
    residual = torch.tensor([3.0, 1.0, 5.0])
    before = read_coordinates(residual, torch.stack([source, target], dim=1))
    patched, delta = swap_coordinates(residual, source, target)
    after = read_coordinates(patched, torch.stack([source, target], dim=1))
    torch.testing.assert_close(after, before.flip(-1))
    assert float(delta[2]) == 0.0


def test_coordinate_swap_supports_bfloat16_residual() -> None:
    source = torch.tensor([1.0, 0.0, 0.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    residual = torch.tensor([2.0, -1.0, 4.0], dtype=torch.bfloat16)
    patched, delta = swap_coordinates(residual, source, target)
    assert patched.dtype == torch.bfloat16
    assert delta.dtype == torch.float32
    torch.testing.assert_close(patched.float(), torch.tensor([-1.0, 2.0, 4.0]))


def test_random_control_is_orthogonal_and_norm_matched() -> None:
    directions = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    reference = torch.tensor([0.0, 0.0, 4.0])
    generator = torch.Generator().manual_seed(9)
    control = orthogonal_norm_matched(reference, directions, generator=generator)
    torch.testing.assert_close(directions.T @ control, torch.zeros(2), atol=1e-6, rtol=0)
    torch.testing.assert_close(control.norm(), reference.norm())


def test_equal_pair_average_does_not_source_weight() -> None:
    rows = [
        (0, 0, torch.eye(2)),
        (0, 1, 3 * torch.eye(2)),
        (1, 1, 8 * torch.eye(2)),
    ]
    torch.testing.assert_close(equal_pair_average(rows), 4 * torch.eye(2))


def test_sparse_nonnegative_separates_remainder() -> None:
    dictionary = torch.eye(4)
    vector = torch.tensor([3.0, 0.0, 2.0, -4.0])
    reconstruction, remainder, selected = greedy_sparse_nonnegative(vector, dictionary, k=2)
    assert selected == [0, 2]
    torch.testing.assert_close(reconstruction, torch.tensor([3.0, 0.0, 2.0, 0.0]))
    torch.testing.assert_close(remainder, torch.tensor([0.0, 0.0, 0.0, -4.0]))
