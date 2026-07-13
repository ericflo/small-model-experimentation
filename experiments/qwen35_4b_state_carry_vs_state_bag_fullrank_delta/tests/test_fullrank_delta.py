from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.state_loop_model import FullRankDeltaBank  # noqa: E402


class TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.linear(inputs)


class FullRankDeltaBankTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.base = TinyBase()
        self.base.requires_grad_(False)
        self.bank = FullRankDeltaBank(
            self.base, ["linear"], dropout=0.0, scale=2.0
        )
        self.inputs = torch.tensor([[1.0, -2.0, 0.5]])

    def test_zero_delta_and_disabled_paths_are_exact(self) -> None:
        direct = self.base(self.inputs)
        with self.bank.enabled(True):
            enabled = self.base(self.inputs)
        self.assertTrue(torch.equal(direct, enabled))
        self.assertEqual(self.bank.active_call_count, 1)
        self.assertFalse(self.bank.is_enabled)
        receipt = self.bank.zero_receipt()
        self.assertEqual(receipt, {"nonzero": 0, "max_abs": 0.0})
        manifest = self.bank.target_manifest()
        self.assertEqual(manifest[0]["target"], "linear")
        self.assertEqual(manifest[0]["shape"], [2, 3])
        self.assertEqual(manifest[0]["dtype"], "torch.float32")

    def test_only_enabled_delta_changes_output_and_receives_gradient(self) -> None:
        with torch.no_grad():
            self.bank.deltas["d000"].weight.fill_(0.25)
        direct = self.base(self.inputs)
        with self.bank.enabled(True):
            enabled = self.base(self.inputs)
        with self.bank.suspended(), self.bank.enabled(True):
            suspended = self.base(self.inputs)
        with self.bank.enabled(True), self.bank.suspended():
            reverse_nested_suspended = self.base(self.inputs)
        self.assertFalse(torch.equal(direct, enabled))
        self.assertTrue(torch.equal(direct, suspended))
        self.assertTrue(torch.equal(direct, reverse_nested_suspended))

        enabled.sum().backward()
        self.assertIsNotNone(self.bank.deltas["d000"].weight.grad)
        self.assertTrue(
            torch.count_nonzero(self.bank.deltas["d000"].weight.grad).item() > 0
        )
        self.assertIsNone(self.base.linear.weight.grad)

    def test_state_roundtrip_and_context_exception_cleanup(self) -> None:
        with torch.no_grad():
            self.bank.deltas["d000"].weight.copy_(
                torch.arange(6, dtype=torch.float32).reshape(2, 3)
            )
        payload = {
            key: value.detach().clone() for key, value in self.bank.state_dict().items()
        }
        with torch.no_grad():
            self.bank.deltas["d000"].weight.zero_()
        self.bank.load_state_dict(payload, strict=True)
        self.assertTrue(
            torch.equal(self.bank.deltas["d000"].weight, payload["deltas.d000.weight"])
        )
        with self.assertRaisesRegex(RuntimeError, "sentinel"):
            with self.bank.enabled(True):
                raise RuntimeError("sentinel")
        self.assertFalse(self.bank.is_enabled)


if __name__ == "__main__":
    unittest.main()
