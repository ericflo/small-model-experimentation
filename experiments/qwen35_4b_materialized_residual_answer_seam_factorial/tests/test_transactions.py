from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from transactions import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    authenticate_complete_chain,
    inventory_state,
    json_bytes,
    run_transaction,
    sha256_file,
)


class TransactionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.lock = self.root / "lock.json"
        self.preflight = self.root / "preflight.json"
        self.runner = self.root / "runner.py"
        self.lock.write_bytes(json_bytes({"lock": True}))
        self.preflight.write_bytes(json_bytes({"preflight": True}))
        self.runner.write_text("# frozen runner\n")
        self.calls = 0

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def prepared(self, invocation: str, count: int = 2) -> Path:
        path = self.root / f"{invocation}.jsonl"
        rows = [
            {
                "id": f"{invocation}-{index}",
                "messages": [{"role": "user", "content": f"prompt-{index}"}],
                "meta": {"index": index},
            }
            for index in range(count)
        ]
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
        return path

    def generate(self, rows, sampling):
        self.calls += 1
        outputs = [
            {
                "id": row["id"],
                "meta": row["meta"],
                "outputs": [{"text": f"answer-{row['id']}"}],
            }
            for row in rows
        ]
        return outputs, {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": sha256_file(self.runner),
            "counts": {"requests": len(rows)},
            "sampling": dict(sampling),
        }

    def fail_generate(self, rows, sampling):
        raise AssertionError("recovery attempted a forbidden model call")

    def run_tx(
        self,
        invocation: str = "a",
        *,
        order=("a",),
        crash_after: str | None = None,
        generate=None,
    ):
        return run_transaction(
            raw_dir=self.raw,
            invocation=invocation,
            invocation_order=order,
            prepared_path=self.prepared(invocation),
            expected_rows=2,
            implementation_lock_path=self.lock,
            live_preflight_path=self.preflight,
            runner_path=self.runner,
            sampling={"thinking": "off", "max_tokens": 24},
            generate=self.generate if generate is None else generate,
            crash_after=crash_after,
        )

    def test_normal_transactions_chain_in_frozen_order(self) -> None:
        first = self.run_tx("a", order=("a", "b"))
        second = self.run_tx("b", order=("a", "b"))
        self.assertEqual(first["state"], "COMPLETE")
        self.assertEqual(
            second["predecessor_complete_sha256"],
            sha256_file(artifact_paths(self.raw, "a")["complete"]),
        )
        receipt = authenticate_complete_chain(
            raw_dir=self.raw, invocation_order=("a", "b")
        )
        self.assertEqual(receipt["rows"], 4)
        self.assertEqual(receipt["sampled_outputs"], 4)
        self.assertEqual(self.calls, 2)

    def test_started_only_is_terminal_and_never_resampled(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after STARTED"):
            self.run_tx(crash_after="started")
        self.assertEqual(self.calls, 0)
        with self.assertRaisesRegex(RuntimeError, "refusing to resample"):
            self.run_tx(generate=self.fail_generate)
        self.assertEqual(inventory_state(self.raw, "a"), "started_only")
        self.assertEqual(self.calls, 0)

    def test_bundle_recovery_promotes_without_model_call(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after bundle"):
            self.run_tx(crash_after="bundle")
        self.assertEqual(self.calls, 1)
        complete = self.run_tx(generate=self.fail_generate)
        self.assertEqual(complete["state"], "COMPLETE")
        self.assertEqual(self.calls, 1)

    def test_generated_recovery_promotes_without_model_call(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after GENERATED"):
            self.run_tx(crash_after="generated")
        self.assertEqual(self.calls, 1)
        complete = self.run_tx(generate=self.fail_generate)
        self.assertEqual(complete["state"], "COMPLETE")
        self.assertEqual(self.calls, 1)

    def test_complete_restart_is_read_only(self) -> None:
        first = self.run_tx()
        before = {
            name: path.read_bytes()
            for name, path in artifact_paths(self.raw, "a").items()
        }
        second = self.run_tx(generate=self.fail_generate)
        after = {
            name: path.read_bytes()
            for name, path in artifact_paths(self.raw, "a").items()
        }
        self.assertEqual(first, second)
        self.assertEqual(before, after)
        self.assertEqual(self.calls, 1)

    def test_complete_crash_is_recoverable_without_model_call(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after COMPLETE"):
            self.run_tx(crash_after="complete")
        self.assertEqual(inventory_state(self.raw, "a"), "complete")
        self.run_tx(generate=self.fail_generate)
        self.assertEqual(self.calls, 1)

    def test_unknown_inventory_and_symlinks_fail_closed(self) -> None:
        self.raw.mkdir()
        (self.raw / "unknown.bin").write_bytes(b"x")
        with self.assertRaisesRegex(RuntimeError, "unknown transaction inventory"):
            self.run_tx()
        (self.raw / "unknown.bin").unlink()
        paths = artifact_paths(self.raw, "a")
        paths["started"].symlink_to(self.lock)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.run_tx()

    def test_invalid_partial_state_fails_closed(self) -> None:
        self.raw.mkdir()
        artifact_paths(self.raw, "a")["bundle"].write_bytes(json_bytes({"bad": True}))
        with self.assertRaisesRegex(RuntimeError, "invalid append-only"):
            self.run_tx()

    def test_tampered_bundle_refuses_recovery(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after bundle"):
            self.run_tx(crash_after="bundle")
        path = artifact_paths(self.raw, "a")["bundle"]
        value = json.loads(path.read_text())
        value["rows"].reverse()
        path.write_bytes(json_bytes(value))
        with self.assertRaisesRegex(RuntimeError, "identity/order"):
            self.run_tx(generate=self.fail_generate)

    def test_authentication_refuses_incomplete_chain(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "after bundle"):
            self.run_tx(crash_after="bundle")
        with self.assertRaisesRegex(RuntimeError, "requires COMPLETE"):
            authenticate_complete_chain(raw_dir=self.raw, invocation_order=("a",))

    def test_later_invocation_cannot_precede_earlier(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "predecessor invocation"):
            self.run_tx("b", order=("a", "b"))

    def test_predecessor_hash_tamper_is_detected(self) -> None:
        self.run_tx("a", order=("a", "b"))
        complete = artifact_paths(self.raw, "a")["complete"]
        value = json.loads(complete.read_text())
        value["chain_sha256"] = "0" * 64
        complete.write_bytes(json_bytes(value))
        with self.assertRaisesRegex(RuntimeError, "COMPLETE receipt changed"):
            authenticate_complete_chain(
                raw_dir=self.raw, invocation_order=("a", "b")
            )


if __name__ == "__main__":
    unittest.main()
