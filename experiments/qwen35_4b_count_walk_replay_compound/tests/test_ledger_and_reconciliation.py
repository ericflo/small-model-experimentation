"""The one-seed write-ahead ledger: open / close / reconcile / double-consume.

The ledger function refuses fail-closed on every path that could silently
re-consume the sealed seed; the byte-equal crash reconciliation contract
(summary bytes recomputed from receipts must equal the preserved file
before the closed record is appended) is enforced as a source contract on
the frozen runner.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import run_benchmark as rb  # noqa: E402


OPENED = {
    "name": rb.FROZEN_NAME,
    "phase": "opened",
    "seed": rb.FROZEN_SEED,
    "think_budget": rb.FROZEN_THINK_BUDGET,
    "tier": rb.FROZEN_TIER,
}
CLOSED = {
    "name": rb.FROZEN_NAME,
    "phase": "closed",
    "tier": rb.FROZEN_TIER,
    "think_budget": rb.FROZEN_THINK_BUDGET,
    "seed": rb.FROZEN_SEED,
    "summary": "summary.json",
    "summary_sha256": "ab" * 32,
}


def write_ledger(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


class TestLedger(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = Path(self._tmp.name) / "benchmark_events.jsonl"

    def test_absent_ledger_is_unconsumed(self):
        rb.require_unconsumed_ledger(self.ledger, OPENED, False)

    def test_opened_without_resume_refuses(self):
        write_ledger(self.ledger, [OPENED])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, False)

    def test_opened_with_resume_continues(self):
        write_ledger(self.ledger, [OPENED])
        rb.require_unconsumed_ledger(self.ledger, OPENED, True)

    def test_closed_refuses_forever_even_with_resume(self):
        write_ledger(self.ledger, [OPENED, CLOSED])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, True)
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, False)

    def test_double_consume_of_a_closed_only_ledger_refuses(self):
        write_ledger(self.ledger, [CLOSED])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, True)

    def test_malformed_row_counts_as_closed(self):
        write_ledger(self.ledger, [{"phase": "unknown"}])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, True)

    def test_mismatched_opened_record_refuses(self):
        drifted = dict(OPENED, seed=rb.FROZEN_SEED + 1)
        write_ledger(self.ledger, [drifted])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, True)

    def test_two_opened_records_refuse(self):
        write_ledger(self.ledger, [OPENED, OPENED])
        with self.assertRaises(ValueError):
            rb.require_unconsumed_ledger(self.ledger, OPENED, True)


class TestByteEqualReconciliation(unittest.TestCase):
    """The reconciliation is a source contract on the normalized-frozen
    runner: a preserved summary must reconcile byte-identically with the
    recomputed payload before the closed record may be appended, and the
    payload must be a pure function of the receipts (no wall-clock)."""

    @classmethod
    def setUpClass(cls):
        cls.text = (EXP / "scripts" / "run_benchmark.py").read_text(
            encoding="utf-8"
        )

    def test_reconciliation_guard_is_present(self):
        self.assertIn(
            "if result.read_bytes() != rendered:", self.text
        )
        self.assertIn(
            "preserved event summary does not reconcile byte-identically",
            self.text,
        )

    def test_summary_payload_carries_no_wall_clock(self):
        # wall_seconds lives only inside the per-arm GATEWAY receipts (their
        # schema requires the key); the summary payload itself must never
        # copy it or read a clock.
        payload_start = self.text.index("payload = {")
        payload_end = self.text.index("rendered = (")
        payload_source = self.text[payload_start:payload_end]
        self.assertNotIn("wall_seconds", payload_source)
        self.assertNotIn("time.", payload_source)
        self.assertNotIn("datetime", self.text)

    def test_closed_record_pins_the_summary_sha(self):
        self.assertIn('"summary_sha256": sha256_file(result)', self.text)


if __name__ == "__main__":
    unittest.main()
