"""The local write-ahead ledger: open / refuse / complete.

eval_local_vllm appends an ``opened`` record to
``runs/local/local_events.jsonl`` BEFORE every engine event launches and a
matching ``receipts`` record (sha-pinning the run's raw artifacts) after
validation. ``require_local_ledger_reconciled`` must: pass on an absent or
fully reconciled ledger, refuse any opened record without its receipts (a
torn attempt), refuse receipts whose pinned artifacts no longer verify on
disk (a discarded attempt), and fail closed on malformed rows — a new
local pass may never silently re-roll over either.
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

import eval_local_vllm as ev  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class TestLocalLedger(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)
        self.ledger = self.tmp / "local_events.jsonl"

    def paths_for(self, label: str, seed: int) -> dict[str, Path]:
        stem = self.tmp / f"{label}_seed{seed}"
        return {
            "output": stem.with_suffix(".jsonl"),
            "metadata": Path(str(stem) + ".meta.json"),
            "log": stem.with_suffix(".log"),
        }

    def opened(self, index: int, label: str = "count_walk", seed: int = 88063) -> dict:
        return {
            "schema_version": 1,
            "phase": "opened",
            "index": index,
            "arm": label,
            "seed": seed,
            "screen_seeds": [88063, 88064, 88065],
            "design_receipt_sha256": "ab" * 32,
        }

    def receipts(self, index: int, label: str = "count_walk", seed: int = 88063) -> dict:
        paths = self.paths_for(label, seed)
        record = {
            "schema_version": 1,
            "phase": "receipts",
            "index": index,
            "arm": label,
            "seed": seed,
        }
        for key, path in paths.items():
            content = f"{key} {label} {seed} {index}".encode()
            path.write_bytes(content)
            record[f"{key}_sha256"] = sha256_bytes(content)
        return record

    def reconcile(self) -> int:
        return ev.require_local_ledger_reconciled(
            self.ledger, paths_for=self.paths_for
        )

    def test_absent_ledger_reconciles_to_index_zero(self):
        self.assertEqual(self.reconcile(), 0)

    def test_open_then_complete_reconciles(self):
        ev.append_local_ledger(self.ledger, self.opened(0))
        ev.append_local_ledger(self.ledger, self.receipts(0))
        self.assertEqual(self.reconcile(), 1)

    def test_six_completed_events_reconcile_in_order(self):
        index = 0
        for label in ("count_walk", "state_track"):
            for seed in (88063, 88064, 88065):
                ev.append_local_ledger(self.ledger, self.opened(index, label, seed))
                ev.append_local_ledger(self.ledger, self.receipts(index, label, seed))
                index += 1
        self.assertEqual(self.reconcile(), 6)

    def test_torn_opened_without_receipts_refuses(self):
        ev.append_local_ledger(self.ledger, self.opened(0))
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_torn_second_event_refuses_after_a_complete_first(self):
        ev.append_local_ledger(self.ledger, self.opened(0))
        ev.append_local_ledger(self.ledger, self.receipts(0))
        ev.append_local_ledger(self.ledger, self.opened(1, seed=88064))
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_discarded_artifacts_refuse(self):
        ev.append_local_ledger(self.ledger, self.opened(0))
        record = self.receipts(0)
        ev.append_local_ledger(self.ledger, record)
        self.paths_for("count_walk", 88063)["output"].unlink()
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_tampered_artifact_refuses(self):
        ev.append_local_ledger(self.ledger, self.opened(0))
        ev.append_local_ledger(self.ledger, self.receipts(0))
        self.paths_for("count_walk", 88063)["log"].write_bytes(b"rewritten")
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_receipts_without_opened_refuses(self):
        ev.append_local_ledger(self.ledger, self.receipts(0))
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_mismatched_arm_refuses(self):
        ev.append_local_ledger(self.ledger, self.opened(0, label="count_walk"))
        record = self.receipts(0, label="state_track")
        ev.append_local_ledger(self.ledger, record)
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_non_monotonic_opened_index_refuses(self):
        ev.append_local_ledger(self.ledger, self.opened(1))
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_malformed_row_fails_closed(self):
        self.ledger.write_text(
            json.dumps({"phase": "unknown"}) + "\n", encoding="utf-8"
        )
        with self.assertRaises(ValueError):
            self.reconcile()

    def test_ledger_path_is_frozen_beside_the_local_receipts(self):
        self.assertEqual(
            ev.LOCAL_LEDGER,
            EXP / "runs" / "local" / "local_events.jsonl",
        )

    def test_opened_record_source_precedes_the_engine_launch(self):
        # Source contract on the (normalized-pinned) evaluator: the opened
        # append sits before the Popen launch inside the arm loop.
        text = (EXP / "scripts" / "eval_local_vllm.py").read_text(
            encoding="utf-8"
        )
        opened_at = text.index('"phase": "opened"')
        launch_at = text.index("subprocess.Popen")
        receipts_at = text.index('"phase": "receipts"')
        self.assertLess(opened_at, launch_at)
        self.assertLess(launch_at, receipts_at)


if __name__ == "__main__":
    unittest.main()
