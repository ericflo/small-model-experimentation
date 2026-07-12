from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_ops import (  # noqa: E402
    ActivationRecorder,
    AddDeltaPatcher,
    ContextLens,
    CoordinateClampPatcher,
    FullActivationPatcher,
    NormMatchedDeltaPatcher,
)


class Block(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.linear(value)


def test_activation_recorder_roots_frozen_graph() -> None:
    layers = torch.nn.ModuleList([Block(4), Block(4), Block(4)])
    for parameter in layers.parameters():
        parameter.requires_grad_(False)
    value = torch.randn(1, 5, 4)
    with ActivationRecorder(layers, at=[0, 2], start_graph_at=0) as recorder:
        output = value
        for layer in layers:
            output = layer(output)
        (gradient,) = torch.autograd.grad(output.sum(), recorder.activations[0])
    assert gradient.shape == value.shape
    assert torch.isfinite(gradient).all()


def test_patchers_modify_only_registered_position() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity(), torch.nn.Identity()])
    value = torch.zeros(1, 3, 6)
    donor = {0: torch.arange(6, dtype=torch.float32)}
    with FullActivationPatcher(layers, 1, donor):
        output = layers[0](value)
    assert torch.equal(output[0, 0], value[0, 0])
    assert torch.equal(output[0, 2], value[0, 2])
    assert torch.equal(output[0, 1], donor[0])

    delta = {1: torch.ones(1, 6)}
    with AddDeltaPatcher(layers, 2, delta):
        output = layers[1](value)
    assert torch.equal(output[0, 2], torch.ones(6))
    assert output[0, :2].abs().sum() == 0


def test_coordinate_patcher_sets_fixed_donor_coordinates() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity()])
    generator = torch.Generator().manual_seed(5)
    directions = {0: torch.randn(10, 3, generator=generator)}
    desired = {0: torch.tensor([1.0, -2.0, 0.5])}
    value = torch.randn(1, 2, 10, generator=generator)
    patcher = CoordinateClampPatcher(layers, 0, directions, desired, rtol=1e-6)
    with patcher:
        output = layers[0](value)
    from coordinates import read_coordinates

    observed = read_coordinates(output[:, 0], directions[0], rtol=1e-6)
    torch.testing.assert_close(observed[0], desired[0], atol=2e-5, rtol=2e-5)
    assert 0 in patcher.deltas


def test_context_lens_round_trip(tmp_path: Path) -> None:
    lens = ContextLens(
        concepts=("cat", "dog"),
        token_ids=(7, 9),
        source_layers=(1, 2),
        directions={1: torch.randn(4, 2), 2: torch.randn(4, 2)},
        n_prompts=5,
    )
    path = tmp_path / "lens.pt"
    torch.save(lens.state_dict(), path)
    loaded = ContextLens.load(str(path))
    assert loaded.concepts == lens.concepts
    assert loaded.token_ids == lens.token_ids
    assert loaded.estimator == "mean_direct_logit_pullback_at_selected_token"
    for layer in lens.source_layers:
        torch.testing.assert_close(loaded.directions[layer], lens.directions[layer], atol=2e-3, rtol=0)


def test_norm_matched_patcher_meets_realized_bfloat16_norm() -> None:
    layers = torch.nn.ModuleList([torch.nn.Identity()])
    generator = torch.Generator().manual_seed(29)
    value = torch.randn(1, 2, 128, generator=generator).to(torch.bfloat16)
    base = torch.randn(1, 128, generator=generator)
    base = base * (0.73 / base.norm())
    patcher = NormMatchedDeltaPatcher(
        layers,
        1,
        {0: base},
        {0: 0.73},
        search_steps=64,
    )
    with patcher:
        layers[0](value)
    assert patcher.relative_errors[0] < 1e-3
    assert abs(float(patcher.deltas[0].norm()) - 0.73) / 0.73 < 1e-3
