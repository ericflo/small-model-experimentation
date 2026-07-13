from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from calibration_stage import (  # noqa: E402
    CALIBRATION_RELATIVE_READS,
    INTERFACE_ARMS,
    INVOCATION_ORDER,
    analyze_calibration,
    canonical_sha256,
    engine_config,
    load_calibration_inputs,
    run_calibration_transactions,
    sampling_configs,
)
from transactions import MODEL_ID, MODEL_REVISION, sha256_file  # noqa: E402


class _FakeCalibrationRunner:
    def __init__(self, runner_path: Path, *, fail_on_call: bool = False):
        self.runner_path = runner_path
        self.fail_on_call = fail_on_call
        self.calls: list[str] = []

    def _metadata(
        self, rows: list[dict[str, Any]], sampling: Any, mode: str
    ) -> dict[str, Any]:
        outputs = [row["outputs"][0] for row in rows]
        sampled = sum(output["n_sampled_tokens"] for output in outputs)
        physical = (
            sum(len(output["stage2_token_ids"]) for output in outputs)
            if mode == "shared_thought_continuation"
            else sampled
        )
        return {
            "schema_version": 5,
            "generation_mode": mode,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": sha256_file(self.runner_path),
            "sampling": dataclasses.asdict(sampling),
            "counts": {
                "requests": len(rows),
                "completions": len(rows),
                "sampled_tokens": sampled,
                "physical_sampled_tokens": physical,
                "reused_sampled_tokens": sampled - physical,
            },
        }

    def _called(self, name: str) -> None:
        if self.fail_on_call:
            raise AssertionError("completed transaction attempted a model call")
        self.calls.append(name)

    def generate_thought_prefixes(self, records, sampling):
        self._called("calibration_thoughts")
        rows = []
        for index, record in enumerate(records):
            token = 1000 + index
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    "outputs": [
                        {
                            "stage1_token_ids": [token],
                            "retained_thinking_token_ids": [token],
                            "seed_stage1": 2000 + index,
                            "stage2_token_ids": [],
                            "n_sampled_tokens": 1,
                        }
                    ],
                }
            )
        return rows, self._metadata(rows, sampling, "shared_thought_prefixes")

    def generate_from_thought_prefixes(
        self, records, thought_rows, thought_metadata, sampling
    ):
        name = "think512_program_slot" if sampling.answer_prefix else "think512_freeform"
        self._called(name)
        prefix = [78041, 25] if sampling.answer_prefix else []
        rows = []
        for index, (record, source_row) in enumerate(zip(records, thought_rows)):
            source = source_row["outputs"][0]
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    "outputs": [
                        {
                            "text": "reason</think>\n\n" + record["meta"]["expected"],
                            "stage1_token_ids": source["stage1_token_ids"],
                            "retained_thinking_token_ids": source[
                                "retained_thinking_token_ids"
                            ],
                            "seed_stage1": source["seed_stage1"],
                            "seed_stage2": 3000 + index,
                            "seed_domain_stage1": "thought",
                            "answer_prefix_token_ids": prefix,
                            "n_answer_tokens": 5,
                            "n_thinking_tokens": 1,
                            "stage2_token_ids": [5000 + index],
                            "n_sampled_tokens": 2,
                            "finish_reason": "stop",
                            "stage1_finish_reason": "stop",
                        }
                    ],
                }
            )
        metadata = self._metadata(rows, sampling, "shared_thought_continuation")
        metadata["thought_source_sha256"] = canonical_sha256(
            {"rows": thought_rows, "runner_metadata": thought_metadata}
        )
        return rows, metadata

    def generate(self, records, sampling):
        name = "no_think_program_slot" if sampling.answer_prefix else "no_think_freeform"
        self._called(name)
        prefix = [78041, 25] if sampling.answer_prefix else []
        rows = [
            {
                "id": record["id"],
                "meta": record["meta"],
                "outputs": [
                    {
                        "text": record["meta"]["expected"],
                        "stage1_token_ids": [4000 + index],
                        "seed_stage1": 3000 + index,
                        "seed_stage2": None,
                        "seed_domain_stage1": "answer",
                        "answer_prefix_token_ids": prefix,
                        "n_answer_tokens": 5,
                        "n_thinking_tokens": 0,
                        "stage2_token_ids": [],
                        "n_sampled_tokens": 1,
                        "finish_reason": "stop",
                        "stage1_finish_reason": "stop",
                    }
                ],
            }
            for index, record in enumerate(records)
        ]
        return rows, self._metadata(rows, sampling, "full_generation")


class CalibrationStageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        for relative in CALIBRATION_RELATIVE_READS:
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(EXP / relative, destination)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_calibration_loader_is_identical_with_every_mechanics_file_absent(self) -> None:
        absent = (
            self.root / "data/procedural/mechanics_public.jsonl",
            self.root / "data/procedural/mechanics_audit.jsonl",
            self.root / "data/procedural/mechanics_gold.jsonl",
            self.root / "runs/prepared/transport_requests.jsonl",
            self.root / "runs/prepared/direct_requests.jsonl",
            self.root / "runs/prepared/suffix_materialized_requests.jsonl",
        )
        self.assertTrue(all(not path.exists() for path in absent))
        isolated = load_calibration_inputs(self.root)
        repository = load_calibration_inputs(EXP)
        self.assertEqual(isolated, repository)
        self.assertEqual(tuple(isolated.read_receipt), CALIBRATION_RELATIVE_READS)
        self.assertEqual(len(isolated.records), 48)

    def test_sampling_and_batch_plan_is_complete_and_fixed(self) -> None:
        inputs = load_calibration_inputs(self.root)
        plan = sampling_configs(inputs)
        self.assertEqual(tuple(plan), INVOCATION_ORDER)
        self.assertEqual(tuple(plan)[1:], INTERFACE_ARMS)
        self.assertEqual(engine_config(inputs).max_num_seqs, 64)
        self.assertEqual(plan["calibration_thoughts"].thinking_budget, 512)
        self.assertEqual(plan["think512_freeform"].answer_prefix, "")
        self.assertEqual(
            plan["think512_program_slot"].answer_prefix, "PROGRAM:"
        )
        self.assertEqual(plan["no_think_freeform"].temperature, 0.6)
        self.assertEqual(plan["no_think_program_slot"].max_tokens, 24)
        self.assertTrue(all(value.n == 1 for value in plan.values()))

    def test_fake_end_to_end_transactions_share_once_and_restart_without_calls(self) -> None:
        inputs = load_calibration_inputs(self.root)
        raw = self.root / "runs/calibration/raw"
        lock = self.root / "runs/calibration/implementation_lock.json"
        preflight = self.root / "runs/calibration/live_preflight.json"
        runner_path = self.root / "src/vllm_runner.py"
        lock.parent.mkdir(parents=True, exist_ok=True)
        runner_path.parent.mkdir(parents=True, exist_ok=True)
        lock.write_text(json.dumps({"locked": True}) + "\n")
        preflight.write_text(json.dumps({"preflight": True}) + "\n")
        runner_path.write_text("# fake exact runner\n")
        fake = _FakeCalibrationRunner(runner_path)
        chain = run_calibration_transactions(
            inputs=inputs,
            runner=fake,
            raw_dir=raw,
            implementation_lock_path=lock,
            live_preflight_path=preflight,
            runner_path=runner_path,
            prepared_path=self.root
            / "runs/prepared/calibration_requests.jsonl",
        )
        self.assertEqual(tuple(fake.calls), INVOCATION_ORDER)
        self.assertEqual(chain["sampled_outputs"], 48 * 5)
        decision = analyze_calibration(
            inputs=inputs,
            raw_dir=raw,
            prepared_path=self.root
            / "runs/prepared/calibration_requests.jsonl",
            implementation_lock_path=lock,
            live_preflight_path=preflight,
            runner_path=runner_path,
        )
        self.assertEqual(decision["decision"], "CALIBRATION_INTERFACE_SELECTED")
        self.assertEqual(decision["winner"], "think512_freeform")
        self.assertTrue(decision["pairing"]["exact_stage1_token_pairing"])
        self.assertEqual(
            decision["metrics"]["think512_program_slot"]["exact_echo_successes"],
            48,
        )

        restarted = _FakeCalibrationRunner(runner_path, fail_on_call=True)
        run_calibration_transactions(
            inputs=inputs,
            runner=restarted,
            raw_dir=raw,
            implementation_lock_path=lock,
            live_preflight_path=preflight,
            runner_path=runner_path,
            prepared_path=self.root
            / "runs/prepared/calibration_requests.jsonl",
        )
        self.assertEqual(restarted.calls, [])

    def test_tampered_calibration_input_fails_before_any_stage_plan(self) -> None:
        request = self.root / "runs/prepared/calibration_requests.jsonl"
        request.write_text(request.read_text() + "\n")
        with self.assertRaisesRegex(RuntimeError, "preoutcome boundary"):
            load_calibration_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
