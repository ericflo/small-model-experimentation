from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import calibration_lock as lock  # noqa: E402
from calibration_stage import engine_config, load_calibration_inputs  # noqa: E402


class CalibrationLockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()

    @staticmethod
    def ci(commit: str) -> dict[str, dict[str, object]]:
        return {
            workflow: {
                "database_id": index + 10,
                "head_sha": commit,
                "status": "completed",
                "conclusion": "success",
                "url": f"https://github.com/example/actions/{index + 10}",
            }
            for index, workflow in enumerate(lock.REQUIRED_WORKFLOWS)
        }

    def value(self) -> dict[str, object]:
        commit = "a" * 40
        return lock.build_lock_value(
            implementation_commit=commit,
            critical_files={name: "b" * 64 for name in lock.CRITICAL_FILES},
            inputs=self.inputs,
            ci_evidence=self.ci(commit),
        )

    def test_lock_schema_survives_canonical_json_round_trip(self) -> None:
        value = json.loads(json.dumps(self.value(), sort_keys=True))
        self.assertEqual(
            lock.validate_lock_value(value, inputs=self.inputs), value
        )
        self.assertEqual(value["invocation_order"], [
            "calibration_thoughts",
            "think512_freeform",
            "think512_program_slot",
            "no_think_freeform",
            "no_think_program_slot",
        ])
        self.assertEqual(value["authorization"], "interface_calibration_only")

    def test_lock_rejects_model_plan_critical_and_ci_mutations(self) -> None:
        mutations = []
        model = copy.deepcopy(self.value())
        model["model"] = "other"
        mutations.append((model, "boundary"))
        plan = copy.deepcopy(self.value())
        plan["sampling"]["think512_freeform"]["thinking_budget"] = 511
        mutations.append((plan, "boundary"))
        critical = copy.deepcopy(self.value())
        critical["critical_files"].pop(lock.CRITICAL_FILES[-1])
        mutations.append((critical, "inventory"))
        ci = copy.deepcopy(self.value())
        ci["implementation_ci"][lock.REQUIRED_WORKFLOWS[0]]["conclusion"] = "failure"
        mutations.append((ci, "CI evidence"))
        for value, message in mutations:
            with self.subTest(message=message), self.assertRaisesRegex(
                RuntimeError, message
            ):
                lock.validate_lock_value(value, inputs=self.inputs)

    def test_ci_query_requires_both_exact_green_workflows(self) -> None:
        commit = "c" * 40
        rows = [
            {
                "databaseId": index + 20,
                "headSha": commit,
                "status": "completed",
                "conclusion": "success",
                "workflowName": workflow,
                "url": f"https://github.com/example/actions/{index + 20}",
            }
            for index, workflow in enumerate(lock.REQUIRED_WORKFLOWS)
        ]
        with mock.patch.object(
            lock.subprocess, "check_output", return_value=json.dumps(rows)
        ):
            evidence = lock.query_green_ci(commit)
        self.assertEqual(tuple(evidence), lock.REQUIRED_WORKFLOWS)

        rerun = {
            **rows[0],
            "databaseId": 999,
            "url": "https://github.com/example/actions/999",
        }
        with mock.patch.object(
            lock.subprocess,
            "check_output",
            return_value=json.dumps([rerun, *rows]),
        ):
            lock.verify_recorded_ci(commit, evidence)

        rows[0]["conclusion"] = "failure"
        with mock.patch.object(
            lock.subprocess, "check_output", return_value=json.dumps(rows)
        ), self.assertRaisesRegex(RuntimeError, "not green"):
            lock.query_green_ci(commit)

    def fake_runner(self):
        config = engine_config(self.inputs)

        class Tokenizer:
            @staticmethod
            def encode(text, *, add_special_tokens):
                if text != "PROGRAM:" or add_special_tokens:
                    raise AssertionError("unexpected tokenizer probe")
                return [78041, 25]

        class Runner:
            def prepare(inner_self, records, thinking, allow_custom_prompts):
                self.assertFalse(allow_custom_prompts)
                return [
                    SimpleNamespace(record_id=row["id"], prompt_token_ids=[index + 1])
                    for index, row in enumerate(records)
                ]

            @staticmethod
            def runtime_metadata():
                return {
                    "python": "3.12.0",
                    "python_executable": "/tmp/.venv-vllm/bin/python",
                    "packages": {
                        "vllm": "0.24.0+cu129",
                        "torch": "2.11.0+cu129",
                        "transformers": "5.13.0",
                    },
                    "gpu": "NVIDIA RTX 6000 Ada Generation",
                }

        runner = Runner()
        runner.config = config
        runner.engine_args = {
            "model": "Qwen/Qwen3.5-4B",
            "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "tokenizer_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "dtype": "bfloat16",
            "async_scheduling": False,
        }
        runner.resolved_logprobs_mode = "raw_logprobs"
        runner.close_ids = [248069, 271]
        runner.tokenizer = Tokenizer()
        runner.resolved_cudagraph = {
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            "max_cudagraph_capture_size": 64,
            "has_full_cudagraphs": True,
        }
        runner.llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    scheduler_config=SimpleNamespace(
                        max_num_seqs=64,
                        max_num_batched_tokens=16384,
                        async_scheduling=False,
                    ),
                    model_config=SimpleNamespace(
                        max_model_len=4096, dtype="bfloat16"
                    ),
                    parallel_config=SimpleNamespace(
                        world_size=1,
                        tensor_parallel_size=1,
                        data_parallel_size=1,
                    ),
                    cache_config=SimpleNamespace(
                        enable_prefix_caching=False,
                        mamba_cache_mode="none",
                        num_gpu_blocks=64,
                    ),
                )
            )
        )
        return runner

    def test_loaded_engine_preflight_checks_exact_runtime_and_geometry(self) -> None:
        runner = self.fake_runner()
        receipt = lock._validate_loaded_runner(runner, self.inputs)
        self.assertEqual(receipt["prompts"]["thinking"]["rows"], 48)
        runner.llm.llm_engine.vllm_config.scheduler_config.async_scheduling = True
        with self.assertRaisesRegex(RuntimeError, "vLLM geometry"):
            lock._validate_loaded_runner(runner, self.inputs)


if __name__ == "__main__":
    unittest.main()
