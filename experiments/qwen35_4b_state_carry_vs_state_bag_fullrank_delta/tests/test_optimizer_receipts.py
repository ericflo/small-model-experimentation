from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimizer_receipts import optimizer_state_receipt  # noqa: E402


class OptimizerReceiptTests(unittest.TestCase):
    def _stepped_optimizer(
        self,
    ) -> tuple[torch.optim.AdamW, list[torch.nn.Parameter]]:
        parameters = [
            torch.nn.Parameter(torch.ones(2, 3, dtype=torch.float32)),
            torch.nn.Parameter(torch.ones(4, dtype=torch.float32)),
        ]
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        sum(parameter.sum() for parameter in parameters).backward()
        optimizer.step()
        return optimizer, parameters

    def test_every_delta_requires_two_matching_finite_fp32_moments(self) -> None:
        optimizer, parameters = self._stepped_optimizer()
        receipt = optimizer_state_receipt(
            optimizer, delta_parameters=parameters
        )
        self.assertEqual(receipt["delta_parameters_audited"], 2)
        self.assertEqual(receipt["delta_moment_tensors"], 4)
        self.assertTrue(receipt["delta_states_complete"])

        del optimizer.state[parameters[1]]["exp_avg_sq"]
        with self.assertRaisesRegex(RuntimeError, "lacks Adam exp_avg_sq"):
            optimizer_state_receipt(optimizer, delta_parameters=parameters)

    def test_wrong_shape_dtype_and_nonfinite_moment_fail_closed(self) -> None:
        for mutation, message in (
            (lambda state: state.__setitem__("exp_avg", torch.zeros(1)), "dtype/shape"),
            (
                lambda state: state.__setitem__(
                    "exp_avg", state["exp_avg"].to(torch.float64)
                ),
                "dtype/shape",
            ),
            (
                lambda state: state["exp_avg"].view(-1).__setitem__(0, float("nan")),
                "nonfinite",
            ),
        ):
            optimizer, parameters = self._stepped_optimizer()
            mutation(optimizer.state[parameters[0]])
            with self.assertRaisesRegex(RuntimeError, message):
                optimizer_state_receipt(optimizer, delta_parameters=parameters)


if __name__ == "__main__":
    unittest.main()
