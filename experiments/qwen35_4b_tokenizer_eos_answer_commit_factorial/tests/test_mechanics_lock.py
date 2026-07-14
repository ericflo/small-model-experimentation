from __future__ import annotations

import copy
import dataclasses
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import calibration_lock  # noqa: E402
import mechanics_lock  # noqa: E402
from calibration_stage import load_calibration_inputs  # noqa: E402
from transactions import json_bytes, read_canonical  # noqa: E402


class MechanicsLockTests(unittest.TestCase):
    def test_calibration_verifier_adapter_uses_real_signature_and_restores_git(self) -> None:
        original_git = mechanics_lock.calibration_authority._git

        def fake_verify(*, verify_network=True):
            self.assertEqual(
                mechanics_lock.calibration_authority._git(
                    "status", "--porcelain=v1", "--untracked-files=all"
                ),
                "",
            )
            return {"verified": verify_network}

        with mock.patch.object(mechanics_lock, "_git", return_value=""), mock.patch.object(
            mechanics_lock.calibration_authority,
            "verify_calibration_lock",
            side_effect=fake_verify,
        ):
            self.assertEqual(
                mechanics_lock._verify_calibration_lock_for_mechanics(),
                {"verified": True},
            )
        self.assertIs(mechanics_lock.calibration_authority._git, original_git)

    def test_calibration_verifier_adapter_rejects_other_dirt(self) -> None:
        with mock.patch.object(mechanics_lock, "_git", return_value=" M README.md"):
            with self.assertRaises(RuntimeError):
                mechanics_lock._verify_calibration_lock_for_mechanics()

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = read_canonical(EXP / "runs/calibration/decision.json")

    @staticmethod
    def ci(commit: str) -> dict[str, dict[str, object]]:
        return {
            workflow: {
                "database_id": index + 1,
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
                "url": f"https://github.com/example/actions/{index + 1}",
            }
            for index, workflow in enumerate(calibration_lock.REQUIRED_WORKFLOWS)
        }

    def review(self, commit: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "verdict": "PASS_IMPLEMENTATION",
            "reviewed_commit": commit,
            "reviewer": "fresh-test-reviewer",
            "review_report_sha256": "a" * 64,
            "reviewed_ci": self.ci(commit),
            "adversarial_review_rounds": 1,
            "allowed_tests_passed": 99,
            "allowed_tests_total": 99,
            "experimental_model_requests_reviewed": 0,
            "sampled_model_outputs_reviewed": 0,
            "hidden_files_read": [],
            "qualification_files_read": [],
            "confirmation_files_read": [],
            "benchmark_files_read": [],
        }

    def test_lock_binds_reviewed_runtime_winner_and_frozen_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calibration_path = root / "calibration.json"
            decision_path = root / "decision.json"
            calibration_path.write_text("{}\n")
            decision_path.write_bytes(json_bytes(self.decision))
            frozen = {
                name: "b" * 40 for name in calibration_lock.FROZEN_MECHANICS_FILES
            }
            calibration = {
                "implementation_commit": "1" * 40,
                "frozen_mechanics_blobs": frozen,
            }
            implementation = "2" * 40
            release = "3" * 40
            critical = {
                name: "c" * 64 for name in mechanics_lock.MECHANICS_CRITICAL_FILES
            }
            with mock.patch.object(
                mechanics_lock, "CALIBRATION_LOCK", calibration_path
            ), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", decision_path
            ):
                value = mechanics_lock.build_mechanics_lock_value(
                    calibration_lock=calibration,
                    calibration_decision=self.decision,
                    calibration_decision_sha256=mechanics_lock.sha256_file(
                        decision_path
                    ),
                    implementation_review=self.review(implementation),
                    critical_files=critical,
                    release_commit=release,
                    release_ci=self.ci(release),
                    inputs=self.inputs,
                )
                self.assertEqual(
                    mechanics_lock.validate_mechanics_lock_value(
                        value,
                        calibration_lock=calibration,
                        calibration_decision=self.decision,
                        inputs=self.inputs,
                    ),
                    value,
                )
                self.assertEqual(
                    value["selected_interface"],
                    "tokenizer_eos_no_think_program_slot",
                )
                self.assertEqual(value["implementation_commit"], implementation)
                self.assertEqual(value["release_commit"], release)
                self.assertNotEqual(
                    value["sampling"]["direct"]["run_seed"],
                    value["sampling"]["suffix_materialized"]["run_seed"],
                )
                mutated = copy.deepcopy(value)
                mutated["experimental_generation_requests_before_lock"] = False
                with self.assertRaisesRegex(RuntimeError, "boundary"):
                    mechanics_lock.validate_mechanics_lock_value(
                        mutated,
                        calibration_lock=calibration,
                        calibration_decision=self.decision,
                        inputs=self.inputs,
                    )

    def test_hidden_authorization_exact_compares_visible_reanalysis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visible.json"
            calibration_decision = root / "calibration_decision.json"
            visible = {
                "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                "selector_uses_hidden": False,
                "hidden_files_read": [],
                "benchmark_files_read": [],
                "tasks": {},
            }
            path.write_bytes(json_bytes(visible))
            calibration_decision.write_bytes(json_bytes(self.decision))
            forged_reanalysis = {**visible, "tasks": {"forged": {}}}
            with mock.patch.object(
                mechanics_lock, "verify_mechanics_lock"
            ), mock.patch.object(
                mechanics_lock, "load_calibration_inputs", return_value=self.inputs
            ), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", calibration_decision
            ), mock.patch.object(
                mechanics_lock, "load_analysis_tokenizer", return_value=object()
            ), mock.patch.object(
                mechanics_lock, "analyze_visible", return_value=forged_reanalysis
            ):
                with self.assertRaisesRegex(RuntimeError, "exact visible analysis"):
                    mechanics_lock.authorize_hidden_read(path)

    def test_hidden_authorization_rejects_boolean_schema_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "visible.json"
            calibration_decision = root / "calibration_decision.json"
            expected_visible = {
                "schema_version": 1,
                "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                "selector_uses_hidden": False,
                "hidden_files_read": [],
                "benchmark_files_read": [],
                "tasks": {},
            }
            observed = copy.deepcopy(expected_visible)
            observed["schema_version"] = True
            path.write_bytes(json_bytes(observed))
            calibration_decision.write_bytes(json_bytes(self.decision))
            with mock.patch.object(
                mechanics_lock, "verify_mechanics_lock"
            ), mock.patch.object(
                mechanics_lock, "load_calibration_inputs", return_value=self.inputs
            ), mock.patch.object(
                mechanics_lock, "CALIBRATION_DECISION", calibration_decision
            ), mock.patch.object(
                mechanics_lock, "load_analysis_tokenizer", return_value=object()
            ), mock.patch.object(
                mechanics_lock, "analyze_visible", return_value=expected_visible
            ):
                with self.assertRaisesRegex(RuntimeError, "exact visible analysis"):
                    mechanics_lock.authorize_hidden_read(path)

    def test_preflight_replay_rejects_full_hostile_mutation_set(self) -> None:
        @dataclasses.dataclass(frozen=True)
        class FakeConfig:
            tensor_parallel_size: int = 1

        commit = "4" * 40
        runner = types.SimpleNamespace(
            config=FakeConfig(),
            engine_args={"seed": 0, "model": "Qwen/Qwen3.5-4B"},
            resolved_cudagraph={"has_full_cudagraphs": True},
            resolved_logprobs_mode="raw_logprobs",
            adapter_info=None,
        )
        runtime = {
            "python": "3.12.3",
            "python_executable": "/tmp/.venv-vllm/bin/python",
            "platform": "linux-test",
            "packages": {"vllm": "0.24.0+cu129"},
            "environment_lock": {"sha256": "a" * 64},
            "uv": "uv 0.test",
            "cuda_toolkit": "cuda test",
            "gpu": "gpu test",
            "vllm_enable_v1_multiprocessing": "0",
            "git_commit": commit,
            "git_dirty": False,
        }
        lock = {
            "selected_interface": "tokenizer_eos_no_think_program_slot",
            "sampling": {"direct": {"thinking": "off", "max_tokens": 24}},
        }
        clean = {
            "schema_version": 1,
            "decision": "MECHANICS_LIVE_ENGINE_PREFLIGHT_PASS",
            "model": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "mechanics_lock_sha256": "",
            "live_head": commit,
            "live_head_ci": self.ci(commit),
            "selected_interface": lock["selected_interface"],
            "sampling": lock["sampling"],
            "runner_sha256": mechanics_lock.sha256_file(
                EXP / "src/vllm_runner.py"
            ),
            "engine": dataclasses.asdict(runner.config),
            "engine_args_sha256": mechanics_lock.canonical_sha256(
                runner.engine_args
            ),
            "resolved_cudagraph": runner.resolved_cudagraph,
            "resolved_logprobs_mode": runner.resolved_logprobs_mode,
            "adapter": None,
            "rng_isolation": {
                "engine_seed": 0,
                "caller_global_rng_state_restored": True,
            },
            "runtime": runtime,
            "experimental_generation_requests_before_preflight": 0,
            "sampled_model_outputs_before_preflight": 0,
            "hidden_files_read": [],
            "benchmark_files_read": [],
        }
        mutations = {
            "schema_bool": lambda value: value.__setitem__("schema_version", True),
            "request_counter_bool": lambda value: value.__setitem__(
                "experimental_generation_requests_before_preflight", False
            ),
            "output_counter_bool": lambda value: value.__setitem__(
                "sampled_model_outputs_before_preflight", False
            ),
            "git_commit": lambda value: value["runtime"].__setitem__(
                "git_commit", "5" * 40
            ),
            "platform": lambda value: value["runtime"].__setitem__(
                "platform", "hostile"
            ),
            "environment_lock": lambda value: value["runtime"].__setitem__(
                "environment_lock", {"sha256": "b" * 64}
            ),
            "uv": lambda value: value["runtime"].__setitem__("uv", "hostile"),
            "cuda": lambda value: value["runtime"].__setitem__(
                "cuda_toolkit", "hostile"
            ),
            "multiprocessing": lambda value: value["runtime"].__setitem__(
                "vllm_enable_v1_multiprocessing", "1"
            ),
            "extra_runtime_key": lambda value: value["runtime"].__setitem__(
                "hostile", True
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "lock.json"
            lock_path.write_bytes(json_bytes({"lock": True}))
            clean["mechanics_lock_sha256"] = mechanics_lock.sha256_file(lock_path)
            dirty_runtime = copy.deepcopy(runtime)
            dirty_runtime["git_dirty"] = True
            clean_path = root / "clean.json"
            clean_path.write_bytes(json_bytes(clean))
            with mock.patch.object(
                mechanics_lock, "MECHANICS_LOCK", lock_path
            ), mock.patch.object(
                mechanics_lock, "verify_mechanics_lock", return_value=lock
            ), mock.patch.object(
                mechanics_lock,
                "_validate_loaded_runner",
                return_value={"runtime": dirty_runtime},
            ), mock.patch.object(
                mechanics_lock, "verify_recorded_ci"
            ):
                self.assertEqual(
                    mechanics_lock.publish_or_verify_mechanics_preflight(
                        runner=runner,
                        inputs=self.inputs,
                        path=clean_path,
                    ),
                    clean,
                )
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    path = root / f"{name}.json"
                    hostile = copy.deepcopy(clean)
                    mutate(hostile)
                    path.write_bytes(json_bytes(hostile))
                    with mock.patch.object(
                        mechanics_lock, "MECHANICS_LOCK", lock_path
                    ), mock.patch.object(
                        mechanics_lock, "verify_mechanics_lock", return_value=lock
                    ), mock.patch.object(
                        mechanics_lock,
                        "_validate_loaded_runner",
                        return_value={"runtime": dirty_runtime},
                    ), mock.patch.object(
                        mechanics_lock, "verify_recorded_ci"
                    ):
                        with self.assertRaisesRegex(
                            RuntimeError, "recorded mechanics preflight changed|CI evidence"
                        ):
                            mechanics_lock.publish_or_verify_mechanics_preflight(
                                runner=runner,
                                inputs=self.inputs,
                                path=path,
                            )


if __name__ == "__main__":
    unittest.main()
