from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import mechanics_transactions as tx  # noqa: E402


class MechanicsTransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.prepared = self.root / "prepared.jsonl"
        self.lock = self.root / "lock.json"
        self.preflight = self.root / "preflight.json"
        self.runner = self.root / "runner.py"
        rows = [
            {
                "id": f"row-{index}",
                "messages": [{"role": "user", "content": str(index)}],
                "meta": {"index": index},
            }
            for index in range(2)
        ]
        self.prepared.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        self.lock.write_text('{\n  "lock": true\n}\n')
        self.preflight.write_text('{\n  "preflight": true\n}\n')
        self.runner.write_text("# mechanics runner\n")
        self.calls = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def generate(self, rows, sampling):
        self.calls += 1
        generated = [
            {
                "id": row["id"],
                "meta": row["meta"],
                "outputs": [{"text": row["id"]}],
            }
            for row in rows
        ]
        return generated, {
            "model": tx.MODEL_ID,
            "model_revision": tx.MODEL_REVISION,
            "runner_sha256": tx.sha256_file(self.runner),
            "counts": {"requests": len(rows), "completions": len(rows)},
            "sampling": dict(sampling),
        }

    def fail_generate(self, rows, sampling):
        raise AssertionError("durable recovery attempted a model call")

    def registration(self) -> dict[str, object]:
        return {
            "prepared_path": self.prepared,
            "expected_rows": 2,
            "implementation_lock_path": self.lock,
            "live_preflight_path": self.preflight,
            "runner_path": self.runner,
            "sampling": {"thinking": "off", "max_tokens": 24},
            "authorization_paths": {},
        }

    def run_tx(self, *, crash_after=None, generate=None):
        return tx.run_transaction(
            raw_dir=self.raw,
            invocation="direct",
            invocation_order=("direct",),
            generate=self.generate if generate is None else generate,
            crash_after=crash_after,
            **self.registration(),
        )

    @staticmethod
    def rewrite(path: Path, value: object) -> None:
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")

    def test_started_boolean_schema_alias_is_terminal(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after STARTED"):
            self.run_tx(crash_after="started")
        path = tx.artifact_paths(self.raw, "direct")["started"]
        value = tx.read_canonical(path)
        value["schema_version"] = True
        self.rewrite(path, value)
        with self.assertRaisesRegex(RuntimeError, "STARTED receipt differs"):
            self.run_tx(generate=self.fail_generate)
        self.assertEqual(self.calls, 0)

    def test_bundle_boolean_schema_alias_cannot_promote(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after bundle"):
            self.run_tx(crash_after="bundle")
        path = tx.artifact_paths(self.raw, "direct")["bundle"]
        value = tx.read_canonical(path)
        value["schema_version"] = True
        self.rewrite(path, value)
        with self.assertRaisesRegex(RuntimeError, "identity/order"):
            self.run_tx(generate=self.fail_generate)
        self.assertEqual(self.calls, 1)

    def test_receipt_boolean_schema_alias_fails_chain_authentication(self) -> None:
        self.run_tx()
        path = tx.artifact_paths(self.raw, "direct")["generated"]
        value = tx.read_canonical(path)
        value["schema_version"] = True
        self.rewrite(path, value)
        with self.assertRaisesRegex(RuntimeError, "GENERATED receipt changed"):
            tx.authenticate_registered_complete_chain(
                raw_dir=self.raw,
                invocation_order=("direct",),
                registrations={"direct": self.registration()},
            )

    def test_registered_boolean_expected_rows_is_rejected(self) -> None:
        self.run_tx()
        registration = copy.deepcopy(self.registration())
        registration["expected_rows"] = True
        with self.assertRaisesRegex(ValueError, "values changed"):
            tx.authenticate_registered_complete_chain(
                raw_dir=self.raw,
                invocation_order=("direct",),
                registrations={"direct": registration},
            )


if __name__ == "__main__":
    unittest.main()
