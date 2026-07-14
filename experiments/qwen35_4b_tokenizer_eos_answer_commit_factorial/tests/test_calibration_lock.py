from __future__ import annotations

import copy
import sys
import unittest
from unittest import mock
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
        review = {
            "schema_version": 1,
            "verdict": "PASS_IMPLEMENTATION",
            "reviewed_commit": commit,
            "reviewer": "adversarial-reviewer",
            "review_report_sha256": "d" * 64,
            "reviewed_ci": self.ci(commit),
            "adversarial_review_rounds": 3,
            "experimental_model_requests_reviewed": 0,
            "sampled_model_outputs_reviewed": 0,
            "hidden_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }
        return lock.build_lock_value(
            implementation_commit=commit,
            critical_files={name: "b" * 64 for name in lock.CRITICAL_FILES},
            frozen_mechanics_blobs={
                name: "c" * 40 for name in lock.FROZEN_MECHANICS_FILES
            },
            inputs=self.inputs,
            review_receipt=review,
            review_receipt_sha256="e" * 64,
            review_receipt_commit=commit,
            release_commit=commit,
            release_ci=self.ci(commit),
        )

    def test_lock_binds_pair_counts_runtime_and_mechanics_without_authorizing_them(self) -> None:
        value = lock.validate_lock_value(self.value(), inputs=self.inputs)
        self.assertEqual(value["authorization"], "interface_calibration_only")
        self.assertEqual(value["expected_source_rows"], 48)
        self.assertEqual(value["expected_answer_pairs"], 192)
        self.assertEqual(value["expected_answer_requests"], 384)
        self.assertEqual(
            value["implementation_review"]["verdict"], "PASS_IMPLEMENTATION"
        )
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
        value["schema_version"] = True
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["experimental_model_requests_before_lock"] = False
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["engine"]["seed"] = False
        mutations.append((value, "boundary changed"))
        value = self.value()
        value["implementation_review"]["verdict"] = "HOLD"
        mutations.append((value, "review binding changed"))
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

    def test_only_strict_machine_review_receipt_can_mint_a_lock(self) -> None:
        receipt = {
            "schema_version": 1,
            "verdict": "HOLD_IMPLEMENTATION",
            "reviewed_commit": "a" * 40,
            "reviewer": "adversarial-reviewer",
            "review_report_sha256": "d" * 64,
            "reviewed_ci": self.ci("a" * 40),
            "adversarial_review_rounds": 3,
            "experimental_model_requests_reviewed": 0,
            "sampled_model_outputs_reviewed": 0,
            "hidden_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }
        with self.assertRaisesRegex(RuntimeError, "receipt boundary changed"):
            lock.validate_implementation_review(receipt)
        receipt["verdict"] = "PASS_IMPLEMENTATION"
        self.assertEqual(
            lock.validate_implementation_review(receipt)["reviewed_commit"],
            "a" * 40,
        )
        receipt["adversarial_review_rounds"] = 1
        with self.assertRaisesRegex(RuntimeError, "receipt boundary changed"):
            lock.validate_implementation_review(receipt)
        receipt["adversarial_review_rounds"] = 3
        for field in (
            "schema_version",
            "experimental_model_requests_reviewed",
            "sampled_model_outputs_reviewed",
        ):
            bad = copy.deepcopy(receipt)
            bad[field] = True if field == "schema_version" else False
            with self.subTest(field=field), self.assertRaisesRegex(
                RuntimeError, "receipt boundary changed"
            ):
                lock.validate_implementation_review(bad)
        receipt["unexpected"] = True
        with self.assertRaisesRegex(RuntimeError, "schema changed"):
            lock.validate_implementation_review(receipt)

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

    def test_child_tools_and_environment_are_pinned(self) -> None:
        with mock.patch.dict(
            lock.os.environ,
            {
                "PATH": "/tmp/forged",
                "GIT_DIR": "/tmp/forged-git",
                "GH_REPO": "attacker/repository",
                "PYTHONPATH": "/tmp/forged-python",
            },
        ):
            environment = lock._child_environment()
        self.assertEqual(lock._git_command("status")[0], "/usr/bin/git")
        self.assertEqual(lock.GH_EXECUTABLE, "/usr/bin/gh")
        self.assertEqual(
            lock.CANONICAL_REPOSITORY, "ericflo/small-model-experimentation"
        )
        self.assertEqual(
            environment["PATH"],
            f"{lock.ROOT}/.venv-vllm/bin:/usr/local/cuda/bin:/usr/bin:/bin",
        )
        for forbidden in ("GIT_DIR", "GH_REPO", "PYTHONPATH"):
            self.assertNotIn(forbidden, environment)

    def test_live_preflight_schema_and_runtime_are_exact(self) -> None:
        commit = "a" * 40
        runtime = {key: f"value-{key}" for key in lock.RUNTIME_METADATA_KEYS}
        runtime["git_dirty"] = False
        runtime["git_commit"] = commit
        value = {
            "schema_version": 1,
            "decision": "CALIBRATION_LIVE_ENGINE_PREFLIGHT_PASS",
            "model": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "implementation_lock_sha256": "f" * 64,
            "implementation_commit": commit,
            "live_head": commit,
            "live_head_ci": self.ci(commit),
            "engine": lock._normalized(
                lock.dataclasses.asdict(lock.engine_config(self.inputs))
            ),
            "engine_args_sha256": "b" * 64,
            "resolved_cudagraph": {"has_full_cudagraphs": True},
            "resolved_logprobs_mode": "raw_logprobs",
            "adapter": None,
            "rng_isolation": {
                "engine_seed": 0,
                "caller_global_rng_state_restored": True,
            },
            "prompt_receipt": {},
            "runtime": runtime,
            "invocation_order": list(lock.INVOCATION_ORDER),
            "expected_source_rows": 48,
            "expected_answer_pairs": 192,
            "expected_answer_requests": 384,
            "experimental_generation_requests_before_preflight": 0,
            "sampled_model_outputs_before_preflight": 0,
            "hidden_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }
        validate = lambda candidate: lock.validate_live_preflight_value(
            candidate,
            inputs=self.inputs,
            implementation_commit=commit,
            implementation_lock_sha256="f" * 64,
        )
        self.assertEqual(validate(value)["live_head"], commit)
        for mutation in (
            "extra_key",
            "runtime_key",
            "runtime_commit",
            "runtime_dirty",
            "rng_types",
            "counter_bool",
            "schema_bool",
            "engine_bool",
            "engine",
        ):
            bad = copy.deepcopy(value)
            if mutation == "extra_key":
                bad["unexpected"] = True
            elif mutation == "runtime_key":
                bad["runtime"].pop("gpu")
            elif mutation == "runtime_commit":
                bad["runtime"]["git_commit"] = "c" * 40
            elif mutation == "runtime_dirty":
                bad["runtime"]["git_dirty"] = True
            elif mutation == "rng_types":
                bad["rng_isolation"] = {
                    "engine_seed": False,
                    "caller_global_rng_state_restored": 1,
                }
            elif mutation == "counter_bool":
                bad["experimental_generation_requests_before_preflight"] = False
            elif mutation == "schema_bool":
                bad["schema_version"] = True
            elif mutation == "engine_bool":
                bad["engine"]["seed"] = False
            else:
                bad["engine"]["max_model_len"] += 1
            with self.subTest(mutation=mutation), self.assertRaisesRegex(
                RuntimeError, "preflight boundary changed"
            ):
                validate(bad)

    def test_live_preflight_must_descend_from_lock_and_precede_current_head(self) -> None:
        commits = {name: character * 40 for name, character in (
            ("lock", "a"), ("live", "b"), ("head", "c")
        )}
        with mock.patch.object(lock, "_ancestor", side_effect=(True, True)):
            lock.authenticate_live_preflight_ancestry(
                lock_commit=commits["lock"],
                live_head=commits["live"],
                current_head=commits["head"],
            )
        for ancestry in ((False, True), (True, False)):
            with self.subTest(ancestry=ancestry), mock.patch.object(
                lock, "_ancestor", side_effect=ancestry
            ), self.assertRaisesRegex(RuntimeError, "Git ancestry changed"):
                lock.authenticate_live_preflight_ancestry(
                    lock_commit=commits["lock"],
                    live_head=commits["live"],
                    current_head=commits["head"],
                )


if __name__ == "__main__":
    unittest.main()
