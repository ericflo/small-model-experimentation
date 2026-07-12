from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from analyze_locality import _row_metrics  # noqa: E402


class LocalityMetricTests(unittest.TestCase):
    def test_identical_logits_have_zero_drift_and_loss_delta(self):
        logits = torch.tensor([0.3, -0.2, 1.1, 0.7, -1.0])
        target = torch.tensor([2, 3], dtype=torch.long)
        teacher = torch.log_softmax(logits, dim=-1)[target]
        row = _row_metrics(logits, logits.clone(), target, teacher)
        self.assertEqual(row["median_centered_non_target_logit_drift"], 0.0)
        self.assertAlmostEqual(row["target_loss_before"], row["target_loss_after"])
        self.assertAlmostEqual(row["entropy_before"], row["entropy_after"])

    def test_softmax_invariant_global_shift_has_zero_centered_drift(self):
        before = torch.linspace(-2.0, 2.0, 11)
        after = before + 17.0
        target = torch.tensor([9, 10], dtype=torch.long)
        teacher = torch.log_softmax(before, dim=-1)[target]
        row = _row_metrics(before, after, target, teacher)
        self.assertLess(row["median_centered_non_target_logit_drift"], 3e-6)
        self.assertAlmostEqual(row["target_loss_before"], row["target_loss_after"], places=6)


if __name__ == "__main__":
    unittest.main()
