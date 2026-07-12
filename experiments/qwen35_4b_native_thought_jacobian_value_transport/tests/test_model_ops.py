from __future__ import annotations

import sys
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from model_ops import QwenThinkModel  # noqa: E402


def test_sample_token_is_seeded_and_respects_top_k() -> None:
    logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
    first = QwenThinkModel._sample_token(
        logits,
        generator=torch.Generator().manual_seed(17),
        temperature=0.6,
        top_p=0.95,
        top_k=2,
    )
    second = QwenThinkModel._sample_token(
        logits,
        generator=torch.Generator().manual_seed(17),
        temperature=0.6,
        top_p=0.95,
        top_k=2,
    )
    assert first == second
    assert first in {2, 3}
