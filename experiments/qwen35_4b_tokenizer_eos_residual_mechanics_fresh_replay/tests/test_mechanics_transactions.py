from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def run_tx(self, *, crash_after=None, generate=None, sampling=None):
        registration = self.registration()
        if sampling is not None:
            registration["sampling"] = sampling
        return tx.run_transaction(
            raw_dir=self.raw,
            invocation="direct",
            invocation_order=("direct",),
            generate=self.generate if generate is None else generate,
            crash_after=crash_after,
            **registration,
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

    def test_production_sampling_tuple_is_normalized_before_fresh_write(self) -> None:
        sampling = {
            "thinking": "off",
            "max_tokens": 24,
            "logprob_token_ids": (248044, 248046),
        }
        self.assertIs(type(sampling["logprob_token_ids"]), tuple)
        complete = self.run_tx(sampling=sampling)
        self.assertEqual(complete["state"], "COMPLETE")
        paths = tx.artifact_paths(self.raw, "direct")
        started = tx.read_canonical(paths["started"])
        bundle = tx.read_canonical(paths["bundle"])
        self.assertIs(type(started["sampling"]["logprob_token_ids"]), list)
        self.assertIs(
            type(
                bundle["runner_metadata"]["sampling"]["logprob_token_ids"]
            ),
            list,
        )
        self.assertEqual(self.calls, 1)

    def test_full_chain_precedes_historical_prefix_replay(self) -> None:
        order = ("transport", "direct", "materialized", "name", "shuffled")
        registrations = {name: self.registration() for name in order}
        tx.run_transaction(
            raw_dir=self.raw,
            invocation=order[0],
            invocation_order=order,
            generate=self.generate,
            **registrations[order[0]],
        )
        initial = tx.authenticate_registered_complete_prefix(
            raw_dir=self.raw,
            invocation_order=order,
            registrations=registrations,
            through="transport",
        )
        self.assertEqual(initial["registered_invocations"], ["transport"])
        for name in order[1:]:
            tx.run_transaction(
                raw_dir=self.raw,
                invocation=name,
                invocation_order=order,
                generate=self.generate,
                **registrations[name],
            )
        self.assertEqual(self.calls, 5)
        with self.assertRaisesRegex(RuntimeError, "later invocation to be absent"):
            tx.authenticate_registered_complete_prefix(
                raw_dir=self.raw,
                invocation_order=order,
                registrations=registrations,
                through="transport",
            )
        chain = tx.authenticate_registered_complete_chain(
            raw_dir=self.raw,
            invocation_order=order,
            registrations=registrations,
        )
        historical = tx.authenticate_registered_historical_prefix(
            raw_dir=self.raw,
            invocation_order=order,
            registrations=registrations,
            through="transport",
            authenticated_chain=chain,
        )
        self.assertEqual(
            historical["decision"],
            "REGISTERED_HISTORICAL_PREFIX_AUTHENTICATED",
        )
        self.assertEqual(historical["registered_invocations"], ["transport"])
        self.assertEqual(self.calls, 5)

        forged_chain = copy.deepcopy(chain)
        forged_chain["schema_version"] = True
        with self.assertRaisesRegex(RuntimeError, "caller complete chain differs"):
            tx.authenticate_registered_historical_prefix(
                raw_dir=self.raw,
                invocation_order=order,
                registrations=registrations,
                through="transport",
                authenticated_chain=forged_chain,
            )

    def test_fresh_successor_authenticates_full_predecessor_before_call(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        self.assertEqual(self.calls, 1)
        complete_path = tx.artifact_paths(self.raw, "first")["complete"]
        complete = tx.read_canonical(complete_path)
        complete["schema_version"] = True
        self.rewrite(complete_path, complete)
        with self.assertRaisesRegex(RuntimeError, "COMPLETE receipt changed"):
            tx.run_transaction(
                raw_dir=self.raw,
                invocation="second",
                invocation_order=("first", "second"),
                generate=self.fail_generate,
                **common,
            )
        self.assertEqual(self.calls, 1)

    def test_predecessor_mutation_inside_generation_is_caught_before_bundle(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        predecessor = tx.artifact_paths(self.raw, "first")["complete"]

        def mutate_during_generation(rows, sampling):
            generated = self.generate(rows, sampling)
            value = tx.read_canonical(predecessor)
            value["schema_version"] = True
            self.rewrite(predecessor, value)
            return generated

        with self.assertRaisesRegex(RuntimeError, "changed after exact authentication"):
            tx.run_transaction(
                raw_dir=self.raw,
                invocation="second",
                invocation_order=("first", "second"),
                generate=mutate_during_generation,
                **common,
            )
        self.assertEqual(self.calls, 2)
        self.assertEqual(tx.inventory_state(self.raw, "second"), "started_only")

    def test_predecessor_mutation_during_promotion_is_caught_before_return(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        predecessor = tx.artifact_paths(self.raw, "first")["complete"]
        successor_complete = tx.artifact_paths(self.raw, "second")["complete"]
        original_write = tx.write_exclusive_durable

        def mutate_after_complete(path, value):
            original_write(path, value)
            if path == successor_complete:
                predecessor_value = tx.read_canonical(predecessor)
                predecessor_value["schema_version"] = True
                self.rewrite(predecessor, predecessor_value)

        with mock.patch.object(
            tx, "write_exclusive_durable", side_effect=mutate_after_complete
        ):
            with self.assertRaisesRegex(
                RuntimeError, "changed after exact authentication"
            ):
                tx.run_transaction(
                    raw_dir=self.raw,
                    invocation="second",
                    invocation_order=("first", "second"),
                    generate=self.generate,
                    **common,
                )
        self.assertEqual(self.calls, 2)
        self.assertEqual(tx.inventory_state(self.raw, "second"), "complete")

    def test_predecessor_bundle_mutation_inside_generation_is_caught(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        predecessor_bundle = tx.artifact_paths(self.raw, "first")["bundle"]

        def mutate_bundle(rows, sampling):
            generated = self.generate(rows, sampling)
            value = tx.read_canonical(predecessor_bundle)
            value["schema_version"] = True
            self.rewrite(predecessor_bundle, value)
            return generated

        with self.assertRaisesRegex(RuntimeError, "changed after exact authentication"):
            tx.run_transaction(
                raw_dir=self.raw,
                invocation="second",
                invocation_order=("first", "second"),
                generate=mutate_bundle,
                **common,
            )
        self.assertEqual(self.calls, 2)
        self.assertEqual(tx.inventory_state(self.raw, "second"), "started_only")

    def test_predecessor_generated_mutation_during_publication_is_caught(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        predecessor_generated = tx.artifact_paths(self.raw, "first")["generated"]
        successor_complete = tx.artifact_paths(self.raw, "second")["complete"]
        original_write = tx.write_exclusive_durable

        def mutate_after_complete(path, value):
            original_write(path, value)
            if path == successor_complete:
                predecessor_value = tx.read_canonical(predecessor_generated)
                predecessor_value["schema_version"] = True
                self.rewrite(predecessor_generated, predecessor_value)

        with mock.patch.object(
            tx, "write_exclusive_durable", side_effect=mutate_after_complete
        ):
            with self.assertRaisesRegex(
                RuntimeError, "changed after exact authentication"
            ):
                tx.run_transaction(
                    raw_dir=self.raw,
                    invocation="second",
                    invocation_order=("first", "second"),
                    generate=self.generate,
                    **common,
                )
        self.assertEqual(self.calls, 2)
        self.assertEqual(tx.inventory_state(self.raw, "second"), "complete")

    def test_predecessor_started_mutation_during_recovery_is_caught(self) -> None:
        common = self.registration()
        tx.run_transaction(
            raw_dir=self.raw,
            invocation="first",
            invocation_order=("first", "second"),
            generate=self.generate,
            **common,
        )
        with self.assertRaisesRegex(RuntimeError, "after bundle"):
            tx.run_transaction(
                raw_dir=self.raw,
                invocation="second",
                invocation_order=("first", "second"),
                generate=self.generate,
                crash_after="bundle",
                **common,
            )
        predecessor_started = tx.artifact_paths(self.raw, "first")["started"]
        successor_bundle = tx.artifact_paths(self.raw, "second")["bundle"]
        original_redurable = tx.redurable

        def mutate_during_redurable(path):
            original_redurable(path)
            if path == successor_bundle:
                predecessor_value = tx.read_canonical(predecessor_started)
                predecessor_value["schema_version"] = True
                self.rewrite(predecessor_started, predecessor_value)

        with mock.patch.object(tx, "redurable", side_effect=mutate_during_redurable):
            with self.assertRaisesRegex(
                RuntimeError, "changed after exact authentication"
            ):
                tx.run_transaction(
                    raw_dir=self.raw,
                    invocation="second",
                    invocation_order=("first", "second"),
                    generate=self.fail_generate,
                    **common,
                )
        self.assertEqual(self.calls, 2)


if __name__ == "__main__":
    unittest.main()
