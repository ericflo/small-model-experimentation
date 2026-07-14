from __future__ import annotations

import dataclasses
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import mechanics_stage as stage  # noqa: E402
from calibration_stage import load_calibration_inputs  # noqa: E402
from transactions import MODEL_ID, MODEL_REVISION, sha256_file  # noqa: E402


def _complete_calibration_decision(inputs):
    arms = list(inputs.config["interface"]["fixed_winner_priority"])
    metrics = {}
    for index, arm in enumerate(arms):
        successes = 48 if index == 0 else 0
        metrics[arm] = {
            "rows": 48,
            "exact_echo_successes": successes,
            "parse_successes": successes,
            "answer_cap_contacts": 0,
            "thinking_cap_contacts": 0,
            "arity_counts": {"2": 24, "3": 24},
            "by_arity": {
                str(arity): {
                    "rows": 24,
                    "exact_echo_successes": successes // 2,
                    "parse_successes": successes // 2,
                    "answer_cap_contacts": 0,
                    "thinking_cap_contacts": 0,
                }
                for arity in (2, 3)
            },
        }
    return {
        "schema_version": 1,
        "stage": "interface_calibration",
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "decision": "CALIBRATION_INTERFACE_SELECTED",
        "winner": arms[0],
        "fixed_priority": arms,
        "qualification": {arm: index == 0 for index, arm in enumerate(arms)},
        "selection_uses_metric_ranking": False,
        "metrics": metrics,
        "scored_rows_sha256": {arm: "a" * 64 for arm in arms},
        "pairing": {},
        "transaction_chain": {},
        "calibration_read_receipt": {},
        "hidden_files_read": [],
        "qualification_files_read": [],
        "confirmation_files_read": [],
        "benchmark_files_read": [],
        "implementation_lock_sha256": "b" * 64,
        "live_preflight_sha256": "c" * 64,
    }


class FakeRunner:
    def __init__(self, runner_path: Path):
        self.runner_path = runner_path
        self.calls = 0

    def generate(self, records, sampling):
        self.calls += 1
        rows = []
        for record in records:
            expected = record["meta"]["expected"]
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    "outputs": [
                        {
                            "text": "reason</think>\n\n" + expected,
                            "seed_domain_stage1": "thought",
                            "n_answer_tokens": 5,
                            "n_thinking_tokens": 2,
                            "n_sampled_tokens": 7,
                            "n_stage1_prompt_tokens": 10,
                            "n_stage2_prompt_tokens": 14,
                            "finish_reason": "stop",
                            "stage1_finish_reason": "stop",
                        }
                    ],
                }
            )
        return rows, {
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": sha256_file(self.runner_path),
            "sampling": dataclasses.asdict(sampling),
            "counts": {"requests": len(rows), "completions": len(rows)},
        }


class MechanicsStageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs = load_calibration_inputs()
        cls.decision = _complete_calibration_decision(cls.inputs)

    def test_selected_sampling_uses_separate_frozen_direct_seed(self) -> None:
        plan = stage.mechanics_sampling_plan(self.decision, self.inputs)
        self.assertEqual(tuple(plan), stage.MECHANICS_INVOCATION_ORDER)
        self.assertEqual(plan["transport"]["run_seed"], 2026072802)
        self.assertEqual(plan["suffix_materialized"]["run_seed"], 2026072802)
        self.assertEqual(plan["direct"]["run_seed"], 2026072803)
        self.assertEqual(plan["transport"]["thinking_budget"], 512)

    def test_transport_transaction_and_gate_are_exact_and_restart_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "transport.jsonl"
            rows = []
            for index in range(24):
                arity = 2 if index % 2 == 0 else 3
                expected = "PROGRAM: A | B" + (" | C" if arity == 3 else "")
                rows.append(
                    {
                        "id": f"transport-{index:02d}",
                        "messages": [{"role": "user", "content": "fake"}],
                        "meta": {
                            "task_id": f"task-{index:02d}",
                            "family": "transport",
                            "arity": arity,
                            "expected": expected,
                        },
                    }
                )
            prepared.write_text("".join(json.dumps(row) + "\n" for row in rows))
            lock = root / "lock.json"
            preflight = root / "preflight.json"
            runner_path = root / "runner.py"
            lock.write_text("{}\n")
            preflight.write_text("{}\n")
            runner_path.write_text("# exact fake runner\n")
            raw = root / "raw"
            fake = FakeRunner(runner_path)
            with mock.patch.dict(stage.PREPARED_PATHS, {"transport": prepared}):
                receipt = stage.run_transport_transaction(
                    decision=self.decision,
                    inputs=self.inputs,
                    runner=fake,
                    raw_dir=raw,
                    mechanics_lock_path=lock,
                    live_preflight_path=preflight,
                    runner_path=runner_path,
                )
                decision = stage.analyze_transport(
                    decision=self.decision,
                    inputs=self.inputs,
                    raw_dir=raw,
                    mechanics_lock_path=lock,
                    live_preflight_path=preflight,
                    runner_path=runner_path,
                )
                stage.run_transport_transaction(
                    decision=self.decision,
                    inputs=self.inputs,
                    runner=fake,
                    raw_dir=raw,
                    mechanics_lock_path=lock,
                    live_preflight_path=preflight,
                    runner_path=runner_path,
                )
            self.assertEqual(receipt["registered_invocations"], ["transport"])
            self.assertEqual(decision["decision"], "SELECTED_INTERFACE_TRANSPORT_PASS")
            self.assertEqual(decision["metrics"]["exact_echo_successes"], 24)
            self.assertEqual(fake.calls, 1)

    def test_hidden_scoring_keeps_selector_primary_and_coverage_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gold_path = Path(directory) / "gold.jsonl"
            program = (("reverse", None), ("reverse", None), ("reverse", None))
            visible_tasks: dict[str, Any] = {}
            gold = []
            for index in range(24):
                task_id = f"task-{index:02d}"
                selection = {
                    "selected_candidate_id": "candidate",
                    "scored": [
                        {
                            "candidate_id": "candidate",
                            "full_program": program,
                        }
                    ],
                }
                visible_tasks[task_id] = {
                    "selections": {
                        arm: selection
                        for arm in (
                            "materialized",
                            "name_only",
                            "shuffled",
                            "direct_sampled",
                            "direct_logical",
                        )
                    }
                }
                gold.append(
                    {
                        "task_id": task_id,
                        "hidden": [{"input": [1, 2, 3], "output": [3, 2, 1]}],
                    }
                )
            gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold))
            result = stage.score_hidden(
                visible={
                    "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                    "selector_uses_hidden": False,
                    "tasks": visible_tasks,
                },
                gold_path=gold_path,
                config=self.inputs.config,
            )
            self.assertEqual(result["primary_selected_accuracy"]["materialized"], 1.0)
            self.assertEqual(
                result["oracle_proposal_coverage_diagnostic"]["materialized"], 1.0
            )
            self.assertEqual(
                result["decision"], "MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL"
            )


if __name__ == "__main__":
    unittest.main()
