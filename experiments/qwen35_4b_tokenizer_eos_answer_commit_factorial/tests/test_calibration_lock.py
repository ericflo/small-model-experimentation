from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import calibration_lock as lock  # noqa: E402
from calibration_stage import load_calibration_inputs  # noqa: E402


class CalibrationLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()

    @staticmethod
    def ci(commit: str) -> dict:
        return {
            workflow: {
                "database_id": index + 1,
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
                "url": f"https://github.com/example/{index}",
            }
            for index, workflow in enumerate(lock.REQUIRED_WORKFLOWS)
        }

    def value(self) -> dict:
        commit = "a" * 40
        return lock.build_lock_value(
            implementation_commit=commit,
            critical_files={name: "b" * 64 for name in lock.CRITICAL_FILES},
            frozen_mechanics_blobs={
                name: "c" * 40 for name in lock.FROZEN_MECHANICS_FILES
            },
            inputs=self.inputs,
            ci_evidence=self.ci(commit),
        )

    def test_lock_binds_pair_counts_runtime_and_mechanics_without_authorizing_them(self) -> None:
        value = lock.validate_lock_value(self.value(), inputs=self.inputs)
        self.assertEqual(value["authorization"], "interface_calibration_only")
        self.assertEqual(value["expected_source_rows"], 48)
        self.assertEqual(value["expected_answer_pairs"], 192)
        self.assertEqual(value["expected_answer_requests"], 384)
        self.assertEqual(value["implementation_review_verdict"], "PASS_IMPLEMENTATION")
        self.assertEqual(set(value["frozen_mechanics_blobs"]), set(lock.FROZEN_MECHANICS_FILES))
        self.assertFalse(
            any(
                "mechanics" in path
                for path in value["calibration_runtime_files"]
            )
        )

    def test_lock_rejects_boundary_and_inventory_mutations(self) -> None:
        mutations = []
        value = self.value()
        value["expected_answer_pairs"] = 191
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["implementation_review_verdict"] = "HOLD"
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["calibration_runtime_files"].append(lock.PREFIX + "data/procedural/mechanics_public.jsonl")
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["critical_files"].pop(next(iter(value["critical_files"])))
        mutations.append((value, "critical file inventory"))
        value = self.value()
        value["frozen_mechanics_blobs"].pop(next(iter(value["frozen_mechanics_blobs"])))
        mutations.append((value, "frozen mechanics inventory"))
        for value, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                lock.validate_lock_value(value, inputs=self.inputs)

    def test_placeholder_review_cannot_mint_a_lock(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "has not passed"):
            lock._review_passes()

    def test_prompt_preflight_matches_every_registered_prompt_id(self) -> None:
        registry = self.inputs.tokenizer_receipt["calibration_prompt_token_ids"]

        class FakeRunner:
            @staticmethod
            def prepare(records, thinking, allow_custom):
                if allow_custom:
                    raise AssertionError("custom prompt bypass is forbidden")
                policy = "think512" if thinking == "budget" else "no_think"
                return [
                    SimpleNamespace(
                        record_id=row["id"],
                        prompt_token_ids=registry[row["id"]][policy]["token_ids"],
                    )
                    for row in records
                ]

        receipt = lock._prompt_receipt(FakeRunner(), self.inputs)
        self.assertEqual(receipt["think512"]["rows"], 48)
        self.assertEqual(receipt["no_think"]["rows"], 48)
        self.assertEqual(receipt["think512"]["ids"], receipt["no_think"]["ids"])


if __name__ == "__main__":
    unittest.main()
