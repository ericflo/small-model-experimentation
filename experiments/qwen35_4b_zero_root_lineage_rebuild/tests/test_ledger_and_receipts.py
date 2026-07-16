"""Single-seed receipt-pinned write-ahead ledger and gateway-receipt guards.

The only valid ledger history is a prefix of opened(78159),
closed(78159): a fresh ledger runs, a crashed one resumes only under an
explicit --resume, a closed one refuses forever, and everything else
fails closed. Gateway receipts must be finite, in-range, and pinned to
the frozen seed/tier/budget.
"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_benchmark as cb  # noqa: E402
import run_benchmark as rb  # noqa: E402


def closed_record(**overrides) -> dict:
    record = {
        "name": rb.FROZEN_NAME,
        "phase": "closed",
        "tier": rb.FROZEN_TIER,
        "think_budget": rb.FROZEN_THINK_BUDGET,
        "seed": rb.FROZEN_SEED,
        "summary": str(rb.EVENT_DIR / "summary.json"),
        "summary_sha256": "a" * 64,
        "receipts": {label: "b" * 64 for label in rb.MODEL_ORDER},
    }
    record.update(overrides)
    return record


class TestLedgerPlan(unittest.TestCase):
    def test_empty_ledger_is_fresh(self):
        self.assertEqual(
            rb.ledger_plan([], resume=False), {"status": "fresh", "closed": None}
        )

    def test_crashed_ledger_requires_explicit_resume(self):
        rows = [rb.opened_record()]
        with self.assertRaises(ValueError):
            rb.ledger_plan(rows, resume=False)
        self.assertEqual(
            rb.ledger_plan(rows, resume=True),
            {"status": "crashed", "closed": None},
        )

    def test_closed_ledger_refuses_forever_resume_or_not(self):
        rows = [rb.opened_record(), closed_record()]
        for resume in (False, True):
            with self.assertRaises(ValueError):
                rb.ledger_plan(rows, resume=resume)

    def test_malformed_first_row_fails_closed(self):
        with self.assertRaises(ValueError):
            rb.ledger_plan([{"phase": "opened", "seed": 12345}], resume=True)

    def test_wrong_seed_in_opened_record_fails_closed(self):
        bad = dict(rb.opened_record(), seed=78158)
        with self.assertRaises(ValueError):
            rb.ledger_plan([bad], resume=True)

    def test_trailing_rows_fail_closed(self):
        rows = [rb.opened_record(), closed_record(), rb.opened_record()]
        with self.assertRaises(ValueError):
            rb.ledger_plan(rows, resume=True)

    def test_malformed_closed_record_fails_closed(self):
        for tampered in (
            closed_record(receipts={"base": "b" * 64}),  # missing arms
            closed_record(summary_sha256="not-a-sha"),
            closed_record(tier="quick"),
            {**closed_record(), "extra": 1},
        ):
            with self.assertRaises(ValueError):
                rb.ledger_plan([rb.opened_record(), tampered], resume=True)


class TestReadoutLedgerAuthentication(unittest.TestCase):
    def test_complete_ledger_returns_the_closed_record(self):
        record = closed_record()
        self.assertEqual(
            cb.authenticate_ledger([cb.opened_record(), record]), record
        )

    def test_empty_ledger_refuses(self):
        with self.assertRaises(ValueError):
            cb.authenticate_ledger([])

    def test_crashed_ledger_refuses(self):
        with self.assertRaises(ValueError):
            cb.authenticate_ledger([cb.opened_record()])

    def test_trailing_rows_refuse(self):
        with self.assertRaises(ValueError):
            cb.authenticate_ledger(
                [cb.opened_record(), closed_record(), {"x": 1}]
            )

    def test_closed_record_shape_agrees_between_runner_and_checker(self):
        record = closed_record()
        self.assertTrue(rb.is_closed_record(record))
        self.assertTrue(cb.is_closed_record(record))
        self.assertEqual(rb.opened_record(), cb.opened_record())


class TestGatewayReceiptGuards(unittest.TestCase):
    def _receipt(self, model: Path, **overrides) -> dict:
        payload = {
            "schema_version": 1,
            "stage": "menagerie_aggregate_gateway",
            "tier": rb.FROZEN_TIER,
            "think_budget": rb.FROZEN_THINK_BUDGET,
            "seed": rb.FROZEN_SEED,
            "backend": "qwen_vllm",
            "model": str(model),
            "model_merge_receipt_sha256": None,  # filled by caller
            "benchmark_runner_sha256": "c" * 64,
            "benchmark_source_inventory_sha256": "d" * 64,
            "benchmark_source_file_count": 56,
            "aggregate": 0.5,
            "per_family": {family: 0.5 for family in cb.FAMILIES},
            "within_budget": True,
            "wall_seconds": 100.0,
        }
        payload.update(overrides)
        return payload

    def _write_and_load(self, tamper: dict) -> dict:
        with tempfile.TemporaryDirectory() as scratch:
            model = Path(scratch) / "model"
            model.mkdir()
            (model / "merge_receipt.json").write_text("{}", encoding="utf-8")
            receipt = self._receipt(model, **tamper)
            if receipt["model_merge_receipt_sha256"] is None:
                receipt["model_merge_receipt_sha256"] = rb.sha256_file(
                    model / "merge_receipt.json"
                )
            path = Path(scratch) / "arm.json"
            path.write_text(json.dumps(receipt), encoding="utf-8")
            return rb.load_event(path, model)

    def test_valid_receipt_loads(self):
        payload = self._write_and_load({})
        self.assertEqual(payload["seed"], 78159)

    def test_nan_aggregate_fails_closed(self):
        with self.assertRaises(ValueError):
            self._write_and_load({"aggregate": math.nan})

    def test_nan_family_score_fails_closed(self):
        table = {family: 0.5 for family in cb.FAMILIES}
        table["menders"] = math.nan
        with self.assertRaises(ValueError):
            self._write_and_load({"per_family": table})

    def test_out_of_range_score_fails_closed(self):
        with self.assertRaises(ValueError):
            self._write_and_load({"aggregate": 1.5})

    def test_wrong_seed_fails_closed(self):
        with self.assertRaises(ValueError):
            self._write_and_load({"seed": 78154})

    def test_non_bool_within_budget_fails_closed(self):
        with self.assertRaises(ValueError):
            self._write_and_load({"within_budget": 1})

    def test_missing_family_fails_closed(self):
        table = {family: 0.5 for family in cb.FAMILIES if family != "warren"}
        with self.assertRaises(ValueError):
            self._write_and_load({"per_family": table})

    def test_infinite_wall_seconds_fails_closed(self):
        with self.assertRaises(ValueError):
            self._write_and_load({"wall_seconds": math.inf})

    def test_over_budget_arm_is_recorded_not_rejected(self):
        payload = self._write_and_load({"within_budget": False})
        self.assertFalse(payload["within_budget"])


class TestCleanSlateAndRecovery(unittest.TestCase):
    def test_stale_event_files_detects_preexisting_artifacts(self):
        with tempfile.TemporaryDirectory() as scratch:
            directory = Path(scratch)
            self.assertEqual(rb.stale_event_files(directory), [])
            (directory / "summary.json").write_text("{}", encoding="utf-8")
            (directory / "base.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                rb.stale_event_files(directory), ["base.json", "summary.json"]
            )

    def test_crashed_summary_reconciliation_requires_byte_equality(self):
        with tempfile.TemporaryDirectory() as scratch:
            summary = Path(scratch) / "summary.json"
            summary.write_bytes(b'{"a": 1}\n')
            rb.reconcile_crashed_summary(summary, b'{"a": 1}\n')  # no raise
            with self.assertRaises(ValueError):
                rb.reconcile_crashed_summary(summary, b'{"a": 2}\n')


class TestTodoPins(unittest.TestCase):
    def test_unfilled_todo_pins_refuse_fail_closed(self):
        """Pre-fill state: all three zero-root pins are None and the runner
        must refuse to consume the seed. Post-fill state: all three are
        sha256 hexes. Any mixed state is a bug."""
        pins = (
            rb.FROZEN_TREE_SHA256[rb.ZERO_ROOT_ARM],
            rb.FROZEN_WEIGHTS_SHA256[rb.ZERO_ROOT_ARM],
            rb.ZERO_ROOT_MERGE_RECEIPT_SHA256,
        )
        states = {pin is None for pin in pins}
        self.assertEqual(len(states), 1, "TODO-pins must be filled together")
        if pins[0] is None:
            with self.assertRaises(ValueError) as caught:
                rb.require_todo_pins_filled()
            self.assertIn("TODO-PIN", str(caught.exception))
        else:
            import re

            for pin in pins:
                self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", pin))
            rb.require_todo_pins_filled()  # must not raise


if __name__ == "__main__":
    unittest.main()
