from __future__ import annotations

import sys
import inspect
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adaptation import (  # noqa: E402
    AdaptationBank,
    capacity_seed,
    microbatch_dropout_seed,
)


class TinyBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(3, 4, bias=False)
        self.second = nn.Linear(4, 2, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.tanh(self.first(value)))


def adaptation_config(dropout: float = 0.0) -> dict:
    return {
        "lora": {"rank": 2, "dropout": dropout, "scale": 2.0},
        "fullrank": {"dropout": dropout, "scale": 2.0},
    }


def make_bank(capacity: str, *, seed: int = 7411, dropout: float = 0.0):
    torch.manual_seed(17)
    base = TinyBase()
    base.requires_grad_(False)
    bank = AdaptationBank(
        base,
        ["first", "second"],
        capacity=capacity,
        model_seed=seed,
        config=adaptation_config(dropout),
    )
    return base, bank


class AdaptationBankTests(unittest.TestCase):
    def test_lora_and_fullrank_have_expected_toy_target_counts_and_zero_function(self) -> None:
        for capacity, expected_parameters in (("lora", 26), ("fullrank", 20)):
            with self.subTest(capacity=capacity):
                base, bank = make_bank(capacity)
                try:
                    manifest = bank.target_manifest()
                    self.assertEqual([item["target"] for item in manifest], ["first", "second"])
                    self.assertEqual(sum(item["parameters"] for item in manifest), expected_parameters)
                    self.assertTrue(all(item["dtype"] == "torch.float32" for item in manifest))
                    self.assertEqual(
                        bank.zero_function_receipt(),
                        {"nonzero_output_weights": 0, "max_abs_output_weight": 0.0},
                    )
                    value = torch.randn(5, 3)
                    direct = base(value)
                    with bank.enabled(True):
                        enabled = base(value)
                    self.assertTrue(torch.equal(direct, enabled))
                    self.assertEqual(bank.active_call_count, 2)
                finally:
                    bank.close()

    def test_capacity_initialization_seed_is_deterministic_and_rng_isolation_holds(self) -> None:
        self.assertEqual(capacity_seed(7411, "lora"), capacity_seed(7411, "lora"))
        self.assertNotEqual(capacity_seed(7411, "lora"), capacity_seed(7412, "lora"))
        self.assertNotEqual(capacity_seed(7411, "lora"), capacity_seed(7411, "fullrank"))

        _, first = make_bank("lora", seed=7411)
        _, second = make_bank("lora", seed=7411)
        _, third = make_bank("lora", seed=7412)
        try:
            first_down = first.down["d000"].weight.detach()
            self.assertTrue(torch.equal(first_down, second.down["d000"].weight.detach()))
            self.assertFalse(torch.equal(first_down, third.down["d000"].weight.detach()))
        finally:
            first.close()
            second.close()
            third.close()

        # Verify isolation around the constructor itself, after base-model
        # initialization has consumed exactly the same global draws.
        torch.manual_seed(992)
        TinyBase()
        expected_after_base = torch.rand(7)
        torch.manual_seed(992)
        observed_base = TinyBase()
        bank = AdaptationBank(
            observed_base,
            ["first", "second"],
            capacity="lora",
            model_seed=7411,
            config=adaptation_config(),
        )
        try:
            self.assertTrue(torch.equal(torch.rand(7), expected_after_base))
        finally:
            bank.close()

    def test_enabled_and_suspended_contexts_are_exact_and_exception_safe(self) -> None:
        value = torch.randn(8, 3)
        for capacity in ("lora", "fullrank"):
            base, bank = make_bank(capacity)
            try:
                with torch.no_grad():
                    if capacity == "lora":
                        bank.up["d000"].weight.fill_(0.2)
                        bank.up["d001"].weight.fill_(0.2)
                    else:
                        bank.deltas["d000"].weight.fill_(0.2)
                        bank.deltas["d001"].weight.fill_(0.2)
                direct = base(value)
                with bank.enabled(True):
                    enabled = base(value)
                with bank.enabled(True), bank.suspended():
                    suspended = base(value)
                with bank.suspended(), bank.enabled(True):
                    reverse_suspended = base(value)
                self.assertFalse(torch.equal(direct, enabled))
                self.assertTrue(torch.equal(direct, suspended))
                self.assertTrue(torch.equal(direct, reverse_suspended))
                with self.assertRaisesRegex(RuntimeError, "sentinel"):
                    with bank.enabled(True):
                        raise RuntimeError("sentinel")
                self.assertFalse(bank.is_enabled)
            finally:
                bank.close()

    def test_zero_initialized_lora_has_expected_two_step_gradient_onset(self) -> None:
        base, bank = make_bank("lora")
        value = torch.randn(16, 3)
        target = torch.ones(16, 2)
        optimizer = torch.optim.SGD(bank.parameters(), lr=0.1)
        try:
            optimizer.zero_grad(set_to_none=True)
            with bank.enabled(True):
                first_output = base(value)
            torch.nn.functional.mse_loss(first_output, target).backward()
            self.assertTrue(
                all(torch.count_nonzero(module.weight.grad).item() > 0 for module in bank.up.values())
            )
            self.assertTrue(
                all(torch.count_nonzero(module.weight.grad).item() == 0 for module in bank.down.values())
            )
            optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            with bank.enabled(True):
                second_output = base(value)
            torch.nn.functional.mse_loss(second_output, target).backward()
            self.assertTrue(
                all(torch.count_nonzero(module.weight.grad).item() > 0 for module in bank.up.values())
            )
            self.assertTrue(
                all(torch.count_nonzero(module.weight.grad).item() > 0 for module in bank.down.values())
            )
        finally:
            bank.close()

    def test_microbatch_seed_and_native_dropout_masks_are_reproducible(self) -> None:
        self.assertEqual(
            list(inspect.signature(microbatch_dropout_seed).parameters),
            ["model_seed", "microbatch_index", "row_id", "k"],
        )
        seed = microbatch_dropout_seed(7411, 19, "row-7", 4)
        self.assertEqual(seed, microbatch_dropout_seed(7411, 19, "row-7", 4))
        self.assertNotEqual(seed, microbatch_dropout_seed(7411, 20, "row-7", 4))
        self.assertNotEqual(seed, microbatch_dropout_seed(7411, 19, "row-7", 5))

        base, bank = make_bank("lora", dropout=0.5)
        bank.train()
        with torch.no_grad():
            bank.up["d000"].weight.fill_(0.2)
            bank.up["d001"].weight.fill_(0.2)
        value = torch.randn(64, 3)

        def run_once(dropout_seed: int) -> tuple[torch.Tensor, dict]:
            # Exercise the CUDA-only controller contract on CPU by replacing only
            # the device seed primitive; native_dropout and hook code remain real.
            with mock.patch("src.adaptation.torch.cuda.is_available", return_value=True), mock.patch(
                "src.adaptation.torch.cuda.manual_seed_all",
                side_effect=lambda item: torch.random.default_generator.manual_seed(int(item)),
            ):
                bank.begin_microbatch(dropout_seed, capture_masks=True)
                with bank.enabled(True):
                    output = base(value)
                return output.detach().clone(), bank.end_microbatch()

        try:
            first_output, first_receipt = run_once(seed)
            second_output, second_receipt = run_once(seed)
            third_output, third_receipt = run_once(seed + 1)
            self.assertTrue(torch.equal(first_output, second_output))
            self.assertEqual(first_receipt, second_receipt)
            self.assertEqual(first_receipt["calls"], 2)
            self.assertIsNotNone(first_receipt["mask_sha256"])
            self.assertFalse(torch.equal(first_output, third_output))
            self.assertNotEqual(first_receipt["mask_sha256"], third_receipt["mask_sha256"])
        finally:
            bank.close()

    def test_lora_and_fullrank_share_the_exact_dropout_mask_schedule(self) -> None:
        seed = microbatch_dropout_seed(7411, 3, "matched-row", 4)
        value = torch.arange(96, dtype=torch.float32).reshape(32, 3) / 100
        receipts = {}
        for capacity in ("lora", "fullrank"):
            base, bank = make_bank(capacity, dropout=0.5)
            bank.train()
            try:
                with mock.patch(
                    "src.adaptation.torch.cuda.is_available", return_value=True
                ), mock.patch(
                    "src.adaptation.torch.cuda.manual_seed_all",
                    side_effect=lambda item: torch.random.default_generator.manual_seed(int(item)),
                ):
                    bank.begin_microbatch(seed, capture_masks=True)
                    with bank.enabled(True):
                        base(value)
                    receipts[capacity] = bank.end_microbatch()
            finally:
                bank.close()
        self.assertEqual(receipts["lora"]["calls"], 2)
        self.assertEqual(receipts["fullrank"]["calls"], 2)
        self.assertEqual(
            receipts["lora"]["call_manifest_sha256"],
            receipts["fullrank"]["call_manifest_sha256"],
        )
        self.assertEqual(
            receipts["lora"]["mask_sha256"],
            receipts["fullrank"]["mask_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
