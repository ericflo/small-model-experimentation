from __future__ import annotations

import dataclasses
import copy
import hashlib
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
import transactions as tx  # noqa: E402
from calibration_stage import (  # noqa: E402
    authenticate_full_generation_bundle,
    load_calibration_inputs,
)
from transactions import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    json_bytes,
    read_canonical,
    sha256_file,
)
from vllm_runner import SamplingConfig, _stable_seed  # noqa: E402


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


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        if tokenize or not add_generation_prompt:
            raise AssertionError("unexpected fake chat-template call")
        return ("THINK:" if enable_thinking else "OFF:") + messages[0]["content"]

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("fake tokenizer forbids special-token injection")
        if text == "</think>\n\n":
            return [248069, 271]
        if text == "PROGRAM:":
            return [78041, 25]
        return [1000 + ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("fake tokenizer preserves all tokens")
        pieces = []
        for token_id in token_ids:
            if token_id == 248069:
                pieces.append("</think>")
            elif token_id == 271:
                pieces.append("\n\n")
            elif token_id == 78041:
                pieces.append("PROGRAM")
            elif token_id == 25:
                pieces.append(":")
            elif token_id == 248044:
                pieces.append("<|endoftext|>")
            else:
                pieces.append(chr(token_id - 1000))
        return "".join(pieces)


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(
        b"".join(token_id.to_bytes(4, "big") for token_id in token_ids)
    ).hexdigest()


class FakeRunner:
    def __init__(self, runner_path: Path, tokenizer: FakeTokenizer):
        self.runner_path = runner_path
        self.tokenizer = tokenizer
        self.calls = 0

    def generate(self, records, sampling):
        self.calls += 1
        rows = []
        for record in records:
            expected = record["meta"]["expected"]
            rendered = self.tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            original = self.tokenizer.encode(rendered, add_special_tokens=False)
            retained = self.tokenizer.encode("reason", add_special_tokens=False)
            answer_ids = self.tokenizer.encode(expected, add_special_tokens=False)
            stage1 = retained + [248044]
            stage2 = answer_ids + [248044]
            close = [248069, 271]
            final_ids = retained + close + answer_ids
            thought_seed = _stable_seed(
                sampling.run_seed, record["id"], -1, "thought"
            )
            answer_seed = _stable_seed(
                sampling.run_seed, record["id"], 0, "answer"
            )
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                    "effective_prompt_sha256": _token_ids_sha256(original),
                    "n_prompt_tokens": len(original),
                    "n_original_prompt_tokens": len(original),
                    "prompt_channel": "thinking",
                    "answer_prefix_token_ids": [],
                    "prompt_logprobs": None,
                    "outputs": [
                        {
                            "sample_index": 0,
                            "stage1_parent_seed": thought_seed,
                            "seed_stage1": thought_seed,
                            "seed_stage2": answer_seed,
                            "seed_domain_stage1": "thought",
                            "seed_domain_stage2": "answer",
                            "text": self.tokenizer.decode(
                                final_ids, skip_special_tokens=False
                            ),
                            "token_ids": final_ids,
                            "stage1_token_ids": stage1,
                            "retained_thinking_token_ids": retained,
                            "answer_prefix_token_ids": [],
                            "injected_token_ids": close,
                            "stage2_token_ids": stage2,
                            "n_answer_tokens": len(answer_ids),
                            "n_thinking_tokens": len(retained),
                            "n_sampled_tokens": len(stage1) + len(stage2),
                            "n_injected_tokens": len(close),
                            "n_completion_tokens": len(final_ids),
                            "n_terminal_tokens_trimmed": 2,
                            "n_stage1_prompt_tokens": len(original),
                            "n_stage2_prompt_tokens": len(original)
                            + len(retained)
                            + len(close),
                            "thinking_closed": True,
                            "forced_close": True,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": "stop",
                            "stage1_stop_reason": 248044,
                            "truncated": False,
                        }
                    ],
                }
            )
        outputs = [row["outputs"][0] for row in rows]
        stage1_prompt = sum(row["n_stage1_prompt_tokens"] for row in outputs)
        stage2_prompt = sum(row["n_stage2_prompt_tokens"] for row in outputs)
        logical_prompt = stage1_prompt + stage2_prompt
        sampled = sum(row["n_sampled_tokens"] for row in outputs)
        return rows, {
            "generation_mode": "full_generation",
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": sha256_file(self.runner_path),
            "sampling": dataclasses.asdict(sampling),
            "counts": {
                "requests": len(rows),
                "completions": len(rows),
                "unique_input_prompt_tokens": sum(
                    row["n_prompt_tokens"] for row in rows
                ),
                "stage1_logical_prompt_tokens": stage1_prompt,
                "stage2_logical_prompt_tokens": stage2_prompt,
                "logical_model_input_tokens": logical_prompt,
                "logical_prompt_tokens": logical_prompt,
                "physical_prompt_tokens": logical_prompt,
                "reused_prompt_tokens": 0,
                "sampled_tokens": sampled,
                "physical_sampled_tokens": sampled,
                "reused_sampled_tokens": 0,
                "logical_model_tokens": logical_prompt + sampled,
                "physical_model_tokens": logical_prompt + sampled,
                "reused_model_tokens": 0,
                "injected_tokens": sum(
                    row["n_injected_tokens"] for row in outputs
                ),
            },
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
            tokenizer = FakeTokenizer()
            fake = FakeRunner(runner_path, tokenizer)
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
                    tokenizer=tokenizer,
                )
                original_bundle = read_canonical(
                    artifact_paths(raw, "transport")["bundle"]
                )
                sampling = SamplingConfig(
                    **stage.mechanics_sampling_plan(
                        self.decision, self.inputs
                    )["transport"]
                )
                for label, mutate in (
                    (
                        "prompt",
                        lambda bundle: bundle["rows"][0].__setitem__(
                            "prompt_channel", "off"
                        ),
                    ),
                    (
                        "seed",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "seed_stage2",
                            bundle["rows"][0]["outputs"][0]["seed_stage2"] + 1,
                        ),
                    ),
                    (
                        "text",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "text", "PROGRAM: A | B"
                        ),
                    ),
                    (
                        "cost",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "n_sampled_tokens",
                            bundle["rows"][0]["outputs"][0]["n_sampled_tokens"]
                            + 1,
                        ),
                    ),
                    (
                        "finish",
                        lambda bundle: (
                            bundle["rows"][0]["outputs"][0].__setitem__(
                                "finish_reason", "length"
                            ),
                            bundle["rows"][0]["outputs"][0].__setitem__(
                                "truncated", True
                            ),
                        ),
                    ),
                    (
                        "thought_internal_stop",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "stage1_token_ids", [248044, 1100, 248044]
                        ),
                    ),
                    (
                        "answer_internal_stop",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "stage2_token_ids", [248044, 1100, 248044]
                        ),
                    ),
                    (
                        "thought_cap",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "stage1_token_ids", [1100] * 513
                        ),
                    ),
                    (
                        "answer_cap",
                        lambda bundle: bundle["rows"][0]["outputs"][0].__setitem__(
                            "stage2_token_ids", [1100] * 25
                        ),
                    ),
                ):
                    with self.subTest(forgery=label):
                        forged = copy.deepcopy(original_bundle)
                        mutate(forged)
                        with self.assertRaises(RuntimeError):
                            authenticate_full_generation_bundle(
                                records=rows,
                                bundle=forged,
                                sampling=sampling,
                                tokenizer=tokenizer,
                                tokenizer_receipt=self.inputs.tokenizer_receipt,
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
                paths = artifact_paths(raw, "transport")
                forged = copy.deepcopy(original_bundle)
                forged["rows"][0]["prompt_channel"] = "off"
                paths["bundle"].write_bytes(json_bytes(forged))
                generated = tx._generated_receipt_value(
                    invocation="transport",
                    started_path=paths["started"],
                    bundle_path=paths["bundle"],
                    bundle=forged,
                )
                paths["generated"].write_bytes(json_bytes(generated))
                complete = tx._complete_value(
                    invocation="transport",
                    started_path=paths["started"],
                    bundle_path=paths["bundle"],
                    generated_path=paths["generated"],
                    predecessor_complete_sha256=None,
                )
                paths["complete"].write_bytes(json_bytes(complete))
                with self.assertRaisesRegex(RuntimeError, "semantic fields changed"):
                    stage.analyze_transport(
                        decision=self.decision,
                        inputs=self.inputs,
                        raw_dir=raw,
                        mechanics_lock_path=lock,
                        live_preflight_path=preflight,
                        runner_path=runner_path,
                        tokenizer=tokenizer,
                    )
            self.assertEqual(receipt["registered_invocations"], ["transport"])
            self.assertEqual(decision["decision"], "SELECTED_INTERFACE_TRANSPORT_PASS")
            self.assertEqual(decision["metrics"]["exact_echo_successes"], 24)
            self.assertEqual(fake.calls, 1)

    def test_hidden_scoring_keeps_selector_primary_and_coverage_diagnostic(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            gold_path = Path(directory) / "gold.jsonl"
            public_path = Path(directory) / "public.jsonl"
            program = (("reverse", None), ("reverse", None), ("reverse", None))
            visible_tasks: dict[str, Any] = {}
            gold = []
            public = []
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
                public.append(
                    {
                        "task_id": task_id,
                        "depth": 3,
                        "visible": [
                            {"input": [1, 2, 3], "output": [3, 2, 1]}
                        ],
                        "unlabeled_probe_inputs": [[4, 5, 6]],
                    }
                )
            gold_path.write_text("".join(json.dumps(row) + "\n" for row in gold))
            public_path.write_text(
                "".join(json.dumps(row) + "\n" for row in public)
            )
            result = stage.score_hidden(
                visible={
                    "schema_version": 1,
                    "decision": "MECHANICS_VISIBLE_SELECTION_FROZEN",
                    "winner": "think512_freeform",
                    "generation_abi_pass": True,
                    "generation_metrics": {},
                    "generation_authentication": {},
                    "selector_uses_hidden": False,
                    "tasks": visible_tasks,
                    "transaction_chain": {},
                    "public_sha256": hashlib.sha256(
                        public_path.read_bytes()
                    ).hexdigest(),
                    "hidden_files_read": [],
                    "benchmark_files_read": [],
                },
                gold_path=gold_path,
                public_path=public_path,
                config=self.inputs.config,
                program_inventory=(program,),
            )
            self.assertEqual(result["primary_selected_accuracy"]["materialized"], 1.0)
            self.assertEqual(
                result["oracle_proposal_coverage_diagnostic"]["materialized"], 1.0
            )
            self.assertEqual(
                result["decision"], "MATERIALIZED_RESIDUAL_LARGE_EFFECT_PILOT_FAIL"
            )
            self.assertEqual(
                result["report_only_exhaustive_cpu_ceiling"]["coverage"], 1.0
            )
            self.assertFalse(
                result["report_only_paired_inference"]["affects_gate"]
            )


if __name__ == "__main__":
    unittest.main()
