from __future__ import annotations

import json
import unittest
from pathlib import Path


RECEIPT = (
    Path(__file__).resolve().parents[1] / "runs" / "scaffold" / "summary.json"
)


def receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text())


class ScaffoldReceiptTests(unittest.TestCase):
    def test_receipt_freezes_new_identity_without_model_authorization(self) -> None:
        value = receipt()
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
        value = receipt()
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
