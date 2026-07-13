from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


RUN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run.py"
SPEC = importlib.util.spec_from_file_location("fresh_replication_scaffold", RUN_PATH)
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


class ScaffoldReceiptTests(unittest.TestCase):
    def test_receipt_freezes_new_identity_without_model_authorization(self) -> None:
        value = run.receipt()
        self.assertEqual(
            value["decision"], "SCAFFOLD_IDENTITY_RESERVED_NO_MODEL_AUTHORIZATION"
        )
        self.assertEqual(value["fresh_seed_block"], list(range(2026072700, 2026072710)))
        self.assertEqual(value["fresh_task_seed_domains"], 1)
        self.assertEqual(value["fresh_sampling_seed_domains"], 1)
        self.assertEqual(value["model_loads"], 0)
        self.assertEqual(value["model_calls"], 0)
        self.assertEqual(value["benchmark_files_read"], [])
        self.assertNotIn("2026072602", str(value))

    def test_parent_terminal_transaction_is_directly_bound(self) -> None:
        value = run.receipt()
        self.assertEqual(
            value["parent_terminal_started_sha256"],
            "f6aa447b1936fac397a353fc13183f008e31884b5006ed7fc50ac78deed3387a",
        )
        self.assertEqual(
            value["request_identity_namespace"],
            "materialized-residual-fresh-replication-v1",
        )


if __name__ == "__main__":
    unittest.main()
