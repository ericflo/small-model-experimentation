from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.optimizer_receipts import optimizer_state_receipt  # noqa: E402
from src.gpu_runner import _build_optimizer  # noqa: E402


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

    def test_adaptation_and_common_state_are_distinct_optimizer_groups(self) -> None:
        adaptation = [torch.nn.Parameter(torch.ones(2, 3))]
        common = [torch.nn.Parameter(torch.ones(4))]
        optimizer = _build_optimizer(
            adaptation,
            common,
            learning_rate=2e-4,
            weight_decay=0.01,
        )
        self.assertEqual(
            [group["group_name"] for group in optimizer.param_groups],
            ["adaptation", "common_state"],
        )
        self.assertIs(optimizer.param_groups[0]["params"][0], adaptation[0])
        self.assertIs(optimizer.param_groups[1]["params"][0], common[0])
        with self.assertRaisesRegex(RuntimeError, "disjoint"):
            _build_optimizer(
                adaptation,
                adaptation,
                learning_rate=2e-4,
                weight_decay=0.01,
            )

    def test_missing_state_exemption_is_registered_once_and_really_missing(self) -> None:
        active = torch.nn.Parameter(torch.ones(2, dtype=torch.float32))
        missing = torch.nn.Parameter(torch.ones(3, dtype=torch.float32))
        optimizer = torch.optim.AdamW([active, missing], lr=1e-3)
        active.sum().backward()
        optimizer.step()

        receipt = optimizer_state_receipt(
            optimizer,
            delta_parameters=[active],
            allowed_missing_parameters=[missing],
        )
        self.assertEqual(receipt["registered_missing_state_exemptions"], 1)
        self.assertEqual(
            receipt["groups"][0]["registered_missing_state_exemptions"], 1
        )

        optimizer.param_groups[0]["params"].append(missing)
        with self.assertRaisesRegex(RuntimeError, "duplicate tensors"):
            optimizer_state_receipt(
                optimizer,
                delta_parameters=[active],
                allowed_missing_parameters=[missing],
            )

    def test_missing_state_exemption_rejects_existing_state_or_ungrouped_tensor(self) -> None:
        optimizer, parameters = self._stepped_optimizer()
        with self.assertRaisesRegex(RuntimeError, "unexpectedly has Adam state"):
            optimizer_state_receipt(
                optimizer,
                delta_parameters=[parameters[0]],
                allowed_missing_parameters=[parameters[1]],
            )

        foreign = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
        with self.assertRaisesRegex(RuntimeError, "not a registered optimizer parameter"):
            optimizer_state_receipt(
                optimizer,
                delta_parameters=parameters,
                allowed_missing_parameters=[foreign],
            )


if __name__ == "__main__":
    unittest.main()
