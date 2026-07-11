#!/usr/bin/env python3

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import train_sparse  # noqa: E402


CFG = {
    "demote_margin_logits": 2.0,
    "uplift_gain_logits": 0.5,
    "lambda_mse": 0.4,
    "lambda_mse_target": 0.05,
    "tau_mse_target": 0.5,
}
ROW = {"chosen_ids": [2], "rejected_id": 1}


class ObjectiveTests(unittest.TestCase):
    def test_uplift_raises_chosen_without_direct_rejected_gradient(self):
        z = torch.zeros((1, 5), requires_grad=True)
        ref = torch.zeros_like(z)
        loss, metrics = train_sparse.objective(z, ref, ROW, "uplift", CFG, z.device)
        loss.backward()
        self.assertLess(float(z.grad[0, 2]), 0.0)  # descent raises chosen
        self.assertAlmostEqual(float(z.grad[0, 1]), 0.0, places=7)
        self.assertEqual(metrics["hit"], 0.0)

    def test_demote_pushes_chosen_up_and_rejected_down(self):
        z = torch.zeros((1, 5), requires_grad=True)
        ref = torch.zeros_like(z)
        loss, _metrics = train_sparse.objective(z, ref, ROW, "demote", CFG, z.device)
        loss.backward()
        self.assertLess(float(z.grad[0, 2]), 0.0)
        self.assertGreater(float(z.grad[0, 1]), 0.0)

    def test_uplift_deactivates_at_target_inside_tether_deadzone(self):
        z = torch.zeros((1, 5), requires_grad=True)
        z.data[0, 2] = 0.5
        ref = torch.zeros_like(z)
        loss, metrics = train_sparse.objective(z, ref, ROW, "uplift", CFG, z.device)
        self.assertAlmostEqual(float(loss.detach()), 0.0, places=7)
        self.assertEqual(metrics["hit"], 1.0)


if __name__ == "__main__":
    unittest.main()
