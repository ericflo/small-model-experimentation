import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module("stc_run_benchmark_ledger", "run_benchmark.py")

SEEDS = BENCH.SEED_ORDER
ARMS = BENCH.MODEL_ORDER
SHA = "0" * 64


def closed_record(seed: int) -> dict:
    return {
        "name": BENCH.FROZEN_NAME,
        "phase": "closed",
        "tier": BENCH.FROZEN_TIER,
        "think_budget": BENCH.FROZEN_THINK_BUDGET,
        "seed": seed,
        "summary": str(BENCH.EVENT_DIRS[seed] / "summary.json"),
        "summary_sha256": SHA,
        "receipts": {label: SHA for label in ARMS},
    }


def history(n_closed: int, crashed: bool = False) -> list[dict]:
    rows: list[dict] = []
    for seed in SEEDS[:n_closed]:
        rows.append(BENCH.opened_record(seed))
        rows.append(closed_record(seed))
    if crashed:
        rows.append(BENCH.opened_record(SEEDS[n_closed]))
    return rows


class LedgerPlanTests(unittest.TestCase):
    def test_empty_ledger_is_all_fresh(self) -> None:
        plan = BENCH.ledger_plan([], resume=False)
        self.assertEqual(
            {seed: entry["status"] for seed, entry in plan.items()},
            {seed: "fresh" for seed in SEEDS},
        )

    def test_prior_records_require_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "--resume"):
            BENCH.ledger_plan(history(1), resume=False)
        plan = BENCH.ledger_plan(history(1), resume=True)
        self.assertEqual(plan[SEEDS[0]]["status"], "closed")
        self.assertEqual(plan[SEEDS[1]]["status"], "fresh")

    def test_crashed_seed_is_recovered_only_under_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "--resume"):
            BENCH.ledger_plan(history(1, crashed=True), resume=False)
        plan = BENCH.ledger_plan(history(1, crashed=True), resume=True)
        self.assertEqual(plan[SEEDS[0]]["status"], "closed")
        self.assertEqual(plan[SEEDS[1]]["status"], "crashed")

    def test_all_six_closed_is_spent_forever(self) -> None:
        for resume in (False, True):
            with self.assertRaisesRegex(ValueError, "budget is spent"):
                BENCH.ledger_plan(history(6), resume=resume)

    def test_double_consume_of_a_closed_row_refuses(self) -> None:
        rows = history(1) + [BENCH.opened_record(SEEDS[0])]
        for resume in (False, True):
            with self.assertRaisesRegex(ValueError, "does not match the frozen"):
                BENCH.ledger_plan(rows, resume=resume)

    def test_out_of_order_seed_refuses(self) -> None:
        rows = [BENCH.opened_record(SEEDS[1])]
        with self.assertRaisesRegex(ValueError, "does not match the frozen"):
            BENCH.ledger_plan(rows, resume=True)

    def test_off_list_seed_refuses(self) -> None:
        # The prior/off-list seed 78169 can never open a confirmation event.
        rows = [{**BENCH.opened_record(SEEDS[0]), "seed": 78169}]
        with self.assertRaisesRegex(ValueError, "does not match the frozen"):
            BENCH.ledger_plan(rows, resume=True)

    def test_malformed_closed_record_refuses(self) -> None:
        record = closed_record(SEEDS[0])
        del record["receipts"]["state_track"]
        rows = [BENCH.opened_record(SEEDS[0]), record]
        with self.assertRaisesRegex(ValueError, "not the closed record"):
            BENCH.ledger_plan(rows, resume=True)

    def test_trailing_extra_rows_refuse(self) -> None:
        rows = history(6) + [{"phase": "extra"}]
        with self.assertRaisesRegex(ValueError, "beyond the frozen"):
            BENCH.ledger_plan(rows, resume=True)


class CompleteLedgerTests(unittest.TestCase):
    def test_complete_ledger_returns_all_closed_records(self) -> None:
        closed = BENCH.authenticate_complete_ledger(history(6))
        self.assertEqual(set(closed), set(SEEDS))

    def test_empty_ledger_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            BENCH.authenticate_complete_ledger([])

    def test_incomplete_ledger_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "incomplete or corrupt"):
            BENCH.authenticate_complete_ledger(history(3))

    def test_trailing_crashed_record_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "crashed opened record"):
            BENCH.authenticate_complete_ledger(history(5, crashed=True))

    def test_rows_beyond_the_event_refuse(self) -> None:
        rows = history(6) + [BENCH.opened_record(SEEDS[0])]
        with self.assertRaisesRegex(ValueError, "beyond the frozen"):
            BENCH.authenticate_complete_ledger(rows)


class ClosedRecordShapeTests(unittest.TestCase):
    def test_well_formed_record_passes(self) -> None:
        self.assertTrue(BENCH.is_closed_record(closed_record(SEEDS[0]), SEEDS[0]))

    def test_wrong_seed_summary_or_receipts_fail(self) -> None:
        record = closed_record(SEEDS[0])
        self.assertFalse(BENCH.is_closed_record(record, SEEDS[1]))
        tampered = dict(record)
        tampered["summary"] = "/tmp/elsewhere/summary.json"
        self.assertFalse(BENCH.is_closed_record(tampered, SEEDS[0]))
        tampered = dict(record)
        tampered["receipts"] = {label: SHA for label in ARMS[:-1]}
        self.assertFalse(BENCH.is_closed_record(tampered, SEEDS[0]))
        tampered = dict(record)
        tampered["summary_sha256"] = "zz" * 32
        self.assertFalse(BENCH.is_closed_record(tampered, SEEDS[0]))
        tampered = dict(record)
        tampered["extra"] = 1
        self.assertFalse(BENCH.is_closed_record(tampered, SEEDS[0]))


class CrashReconciliationTests(unittest.TestCase):
    def test_byte_identical_regeneration_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            rendered = b'{"seed": 78170}\n'
            summary.write_bytes(rendered)
            BENCH.reconcile_crashed_summary(SEEDS[0], summary, rendered)

    def test_divergent_summary_refuses_with_both_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            existing = b'{"seed": 78170, "tampered": true}\n'
            rendered = b'{"seed": 78170}\n'
            summary.write_bytes(existing)
            with self.assertRaisesRegex(ValueError, "does not match") as ctx:
                BENCH.reconcile_crashed_summary(SEEDS[0], summary, rendered)
            message = str(ctx.exception)
            self.assertIn(hashlib.sha256(existing).hexdigest(), message)
            self.assertIn(hashlib.sha256(rendered).hexdigest(), message)


class StaleEventFileTests(unittest.TestCase):
    def test_missing_directory_is_clean(self) -> None:
        self.assertEqual(
            BENCH.stale_event_files(Path("/nonexistent/event/dir")), []
        )

    def test_receipt_failure_and_summary_files_are_stale(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            (event_dir / "count_walk.json").write_text("{}", encoding="utf-8")
            (event_dir / "state_track.failure.json").write_text("{}", encoding="utf-8")
            (event_dir / "summary.json").write_text("{}", encoding="utf-8")
            (event_dir / "unrelated.txt").write_text("x", encoding="utf-8")
            self.assertEqual(
                BENCH.stale_event_files(event_dir),
                ["count_walk.json", "state_track.failure.json", "summary.json"],
            )


if __name__ == "__main__":
    unittest.main()
