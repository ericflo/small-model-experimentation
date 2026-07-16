"""Unit tests for the write-ahead one-seed ledger and gateway-receipt gating.

The ledger contract is stricter than the reference pilots': the seed is
spent when the event OPENS (a write-ahead ``opened`` record lands before
the first gateway call), and it closes with a ``closed`` record after the
summary. Any closed (or legacy/malformed) record refuses a new event
forever; a lone matching opened record is a crashed event that may only
continue under an explicit ``--resume``. Receipt authentication is
fail-closed: a receipt with any drifted field (seed, tier, budget,
backend, model, family set, merge-receipt hash) or any non-finite /
out-of-range score must be rejected — a NaN compares unequal to
everything and would otherwise silently drop its family from the
strict-win partition. Budget-probe twist versus the reference cell:
``within_budget`` must be a strict bool but FALSE IS ACCEPTED (the
budget_integrity reading scopes the paired comparison; scores are still
recorded), and ``wall_seconds`` must be a finite non-negative number.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_benchmark as cb  # noqa: E402
import run_benchmark as rb  # noqa: E402


def opened_record() -> dict:
    return {
        "name": rb.FROZEN_NAME,
        "phase": "opened",
        "seed": rb.FROZEN_SEED,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "tier": rb.FROZEN_TIER,
    }


def closed_record() -> dict:
    return {
        "name": rb.FROZEN_NAME,
        "phase": "closed",
        "tier": rb.FROZEN_TIER,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "seed": rb.FROZEN_SEED,
        "summary": "runs/benchmark/x/summary.json",
        "summary_sha256": "0" * 64,
    }


def write_ledger(scratch: str, rows: list[dict] | None) -> Path:
    ledger = Path(scratch) / "benchmark_events.jsonl"
    if rows is not None:
        ledger.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
    return ledger


class TestLedgerRefusal(unittest.TestCase):
    def test_missing_ledger_allows_a_fresh_event(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, None)
            rb.require_unconsumed_ledger(ledger, opened_record(), resume=False)

    def test_empty_ledger_allows_a_fresh_event(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [])
            rb.require_unconsumed_ledger(ledger, opened_record(), resume=False)
            ledger.write_text("\n  \n", encoding="utf-8")
            rb.require_unconsumed_ledger(ledger, opened_record(), resume=False)

    def test_closed_record_refuses_forever_even_with_resume(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [opened_record(), closed_record()])
            for resume in (False, True):
                with self.assertRaises(ValueError):
                    rb.require_unconsumed_ledger(
                        ledger, opened_record(), resume=resume
                    )

    def test_legacy_record_without_phase_counts_as_closed(self) -> None:
        legacy = {key: value for key, value in closed_record().items() if key != "phase"}
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [legacy])
            with self.assertRaises(ValueError):
                rb.require_unconsumed_ledger(ledger, opened_record(), resume=True)

    def test_crashed_opened_record_refuses_without_resume(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [opened_record()])
            with self.assertRaises(ValueError):
                rb.require_unconsumed_ledger(ledger, opened_record(), resume=False)

    def test_crashed_opened_record_continues_only_under_resume(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [opened_record()])
            rb.require_unconsumed_ledger(ledger, opened_record(), resume=True)

    def test_mismatched_opened_record_refuses_even_with_resume(self) -> None:
        for drift in (
            {"seed": rb.FROZEN_SEED + 1},
            {"tier": "quick"},
            {"think_budget": 1024},
            {"name": "other"},
        ):
            with tempfile.TemporaryDirectory() as scratch:
                ledger = write_ledger(scratch, [{**opened_record(), **drift}])
                with self.assertRaises(ValueError, msg=str(drift)):
                    rb.require_unconsumed_ledger(
                        ledger, opened_record(), resume=True
                    )

    def test_duplicate_opened_records_refuse(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            ledger = write_ledger(scratch, [opened_record(), opened_record()])
            with self.assertRaises(ValueError):
                rb.require_unconsumed_ledger(ledger, opened_record(), resume=True)


def receipt_payload(model: Path) -> dict:
    merge_receipt = model / "merge_receipt.json"
    return {
        "schema_version": 1,
        "stage": "menagerie_aggregate_gateway",
        "tier": rb.FROZEN_TIER,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "seed": rb.FROZEN_SEED,
        "backend": "qwen_vllm",
        "model": str(model),
        "model_merge_receipt_sha256": hashlib.sha256(
            merge_receipt.read_bytes()
        ).hexdigest(),
        "benchmark_runner_sha256": "a" * 64,
        "benchmark_source_inventory_sha256": "b" * 64,
        "benchmark_source_file_count": 100,
        "aggregate": 0.25,
        "per_family": {family: 0.25 for family in sorted(rb.PUBLIC_FAMILIES)},
        "within_budget": True,
        "wall_seconds": 1.0,
    }


def make_model(scratch: str) -> Path:
    model = Path(scratch) / "merged" / "arm"
    model.mkdir(parents=True)
    (model / "merge_receipt.json").write_text('{"name": "arm"}\n', encoding="utf-8")
    return model


def write_receipt(scratch: str, payload: dict) -> Path:
    receipt = Path(scratch) / "arm.json"
    receipt.write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt


class TestReceiptAuthentication(unittest.TestCase):
    def test_valid_receipt_loads(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            receipt = write_receipt(scratch, receipt_payload(model))
            loaded = rb.load_event(receipt, model)
            self.assertEqual(loaded["seed"], rb.FROZEN_SEED)
            self.assertEqual(loaded["think_budget"], 8192)

    def test_drifted_fields_are_rejected(self) -> None:
        drifts = (
            {"seed": rb.FROZEN_SEED + 1},
            {"tier": "quick"},
            {"think_budget": 1024},
            {"backend": "other"},
            {"model_merge_receipt_sha256": "c" * 64},
            {"stage": "something_else"},
        )
        for drift in drifts:
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                receipt = write_receipt(scratch, {**receipt_payload(model), **drift})
                with self.assertRaises(ValueError, msg=str(drift)):
                    rb.load_event(receipt, model)

    def test_over_budget_receipt_is_accepted_and_recorded(self) -> None:
        # Budget-probe contract: within_budget false never rejects the
        # receipt; the budget_integrity reading scopes the comparison.
        for module in (rb, cb):
            with tempfile.TemporaryDirectory() as scratch:
                model = make_model(scratch)
                payload = {**receipt_payload(model), "within_budget": False}
                receipt = write_receipt(scratch, payload)
                loaded = module.load_event(receipt, model)
                self.assertIs(loaded["within_budget"], False, module.__name__)

    def test_non_bool_within_budget_is_rejected(self) -> None:
        for bad in (None, "true", "yes", 1, 0.0):
            for module in (rb, cb):
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = {**receipt_payload(model), "within_budget": bad}
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, bad)
                    ):
                        module.load_event(receipt, model)

    def test_wall_seconds_must_be_finite_and_non_negative(self) -> None:
        for bad in (float("nan"), float("inf"), -1.0, True, "12.5", None):
            for module in (rb, cb):
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = {**receipt_payload(model), "wall_seconds": bad}
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, bad)
                    ):
                        module.load_event(receipt, model)
        # No upper cap: tb8192 arms may run long; record what the gateway
        # returns.
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = {**receipt_payload(model), "wall_seconds": 86_400.0}
            receipt = write_receipt(scratch, payload)
            self.assertEqual(
                rb.load_event(receipt, model)["wall_seconds"], 86_400.0
            )

    def test_missing_family_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = receipt_payload(model)
            payload["per_family"].pop("menders")
            receipt = write_receipt(scratch, payload)
            with self.assertRaises(ValueError):
                rb.load_event(receipt, model)

    def test_extra_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            model = make_model(scratch)
            payload = receipt_payload(model)
            payload["surprise"] = True
            receipt = write_receipt(scratch, payload)
            with self.assertRaises(ValueError):
                rb.load_event(receipt, model)


class TestFinitenessGuards(unittest.TestCase):
    def test_valid_score_predicate(self) -> None:
        for module in (rb, cb):
            for good in (0.0, 0.5, 1.0, 1):
                self.assertTrue(module._valid_score(good), (module.__name__, good))
            for bad in (
                float("nan"), float("inf"), float("-inf"),
                -0.001, 1.001, True, False, None, "0.5",
            ):
                self.assertFalse(module._valid_score(bad), (module.__name__, bad))

    def test_valid_wall_seconds_predicate(self) -> None:
        for module in (rb, cb):
            for good in (0.0, 1, 230.5, 86_400.0):
                self.assertTrue(
                    module._valid_wall_seconds(good), (module.__name__, good)
                )
            for bad in (
                float("nan"), float("inf"), -0.001, True, False, None, "1.0",
            ):
                self.assertFalse(
                    module._valid_wall_seconds(bad), (module.__name__, bad)
                )

    def test_non_finite_or_out_of_range_scores_rejected_by_both_loaders(self) -> None:
        corruptions = (
            ("aggregate_nan", "aggregate", float("nan")),
            ("aggregate_above_one", "aggregate", 1.5),
            ("family_nan", "menders", float("nan")),
            ("family_infinite", "sirens", float("inf")),
            ("family_negative", "warren", -0.1),
            ("family_bool", "mirage", True),
        )
        for module in (rb, cb):
            for name, field, value in corruptions:
                with tempfile.TemporaryDirectory() as scratch:
                    model = make_model(scratch)
                    payload = receipt_payload(model)
                    if field == "aggregate":
                        payload["aggregate"] = value
                    else:
                        payload["per_family"][field] = value
                    receipt = write_receipt(scratch, payload)
                    with self.assertRaises(
                        ValueError, msg=(module.__name__, name)
                    ):
                        module.load_event(receipt, model)


if __name__ == "__main__":
    unittest.main()
