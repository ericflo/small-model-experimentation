from __future__ import annotations

import sys
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from model_ops import ActivationRecorder, TargetedLens  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.linear = torch.nn.Linear(width, width, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.linear(value)


def test_activation_recorder_roots_frozen_graph() -> None:
    layers = torch.nn.ModuleList([Block(3), Block(3), Block(3)])
    for parameter in layers.parameters():
        parameter.requires_grad_(False)
    value = torch.randn(1, 4, 3)
    with ActivationRecorder(layers, at=[0, 2], start_graph_at=0) as recorder:
        output = value
        for layer in layers:
            output = layer(output)
        (gradient,) = torch.autograd.grad(output.sum(), recorder.activations[0])
    assert gradient.shape == value.shape
    assert torch.isfinite(gradient).all()


def test_targeted_lens_round_trip(tmp_path: Path) -> None:
    lens = TargetedLens(
        concepts=("cat", "dog"),
        token_ids=(7, 9),
        source_layers=(1, 2),
        target_layer=3,
        directions={1: torch.randn(2, 4), 2: torch.randn(2, 4)},
        n_prompts=5,
    )
    path = tmp_path / "lens.pt"
    torch.save(lens.state_dict(), path)
    loaded = TargetedLens.load(str(path))
    assert loaded.concepts == lens.concepts
    assert loaded.token_ids == lens.token_ids
    assert loaded.source_layers == lens.source_layers
    assert loaded.pair_weighting == "equal_valid_causal_source_target_pairs"
    for layer in lens.source_layers:
        torch.testing.assert_close(loaded.directions[layer], lens.directions[layer], atol=2e-3, rtol=0)
