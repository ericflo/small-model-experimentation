from __future__ import annotations

import dataclasses
import hashlib
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
    _authenticate_factorial_pairing,
    analyze_calibration,
    canonical_sha256,
    engine_config,
    load_calibration_inputs,
    run_calibration_transactions,
    sampling_configs,
)
from transactions import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    artifact_paths,
    sha256_file,
)
from vllm_runner import _stable_seed  # noqa: E402


class _FakeTokenizer:
    eos_token_id = 248046

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
        if text == "</think>":
            return [248069]
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


class _FakeCalibrationRunner:
    def __init__(
        self,
        runner_path: Path,
        tokenizer: _FakeTokenizer,
        *,
        fail_on_call: bool = False,
    ):
        self.runner_path = runner_path
        self.tokenizer = tokenizer
        self.fail_on_call = fail_on_call
        self.calls: list[str] = []

    def _prompt_fields(
        self, record: dict[str, Any], *, thinking: bool, prefix: list[int]
    ) -> tuple[dict[str, Any], list[int]]:
        rendered = self.tokenizer.apply_chat_template(
            record["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
        original = self.tokenizer.encode(rendered, add_special_tokens=False)
        effective = original + ([] if thinking else prefix)
        return {
            "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "effective_prompt_sha256": _token_ids_sha256(effective),
            "n_prompt_tokens": len(effective),
            "n_original_prompt_tokens": len(original),
            "prompt_channel": "thinking" if thinking else "off",
            "answer_prefix_token_ids": prefix,
            "prompt_logprobs": None,
        }, original

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
        logical_prompt = sum(
            output["n_stage1_prompt_tokens"] + output["n_stage2_prompt_tokens"]
            for output in outputs
        )
        physical_prompt = (
            sum(output["n_stage2_prompt_tokens"] for output in outputs)
            if mode == "shared_thought_continuation"
            else logical_prompt
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
                "logical_prompt_tokens": logical_prompt,
                "physical_prompt_tokens": physical_prompt,
                "reused_prompt_tokens": logical_prompt - physical_prompt,
                "logical_model_tokens": logical_prompt + sampled,
                "physical_model_tokens": physical_prompt + physical,
                "reused_model_tokens": logical_prompt
                + sampled
                - physical_prompt
                - physical,
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
            retained = self.tokenizer.encode("reason", add_special_tokens=False)
            stage1 = retained + [248044]
            prompt, original = self._prompt_fields(record, thinking=True, prefix=[])
            seed = _stable_seed(sampling.run_seed, record["id"], -1, "thought")
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    **prompt,
                    "outputs": [
                        {
                            "sample_index": 0,
                            "stage1_parent_seed": seed,
                            "seed_stage1": seed,
                            "seed_stage2": None,
                            "seed_domain_stage1": "thought",
                            "seed_domain_stage2": None,
                            "text": "reason",
                            "token_ids": retained,
                            "stage1_token_ids": stage1,
                            "retained_thinking_token_ids": retained,
                            "answer_prefix_token_ids": [],
                            "injected_token_ids": [],
                            "stage2_token_ids": [],
                            "n_thinking_tokens": len(retained),
                            "n_answer_tokens": 0,
                            "n_sampled_tokens": len(stage1),
                            "n_injected_tokens": 0,
                            "n_completion_tokens": len(retained),
                            "n_terminal_tokens_trimmed": 1,
                            "n_tokens_discarded_after_close": 0,
                            "n_stage1_prompt_tokens": len(original),
                            "n_stage2_prompt_tokens": 0,
                            "thinking_closed": False,
                            "forced_close": False,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": "stop",
                            "stage1_stop_reason": 248044,
                            "truncated": False,
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
            prompt, original = self._prompt_fields(
                record, thinking=True, prefix=prefix
            )
            expected = record["meta"]["expected"]
            continuation = expected[len("PROGRAM:") :] if prefix else expected
            answer_ids = self.tokenizer.encode(
                continuation, add_special_tokens=False
            )
            stage2 = answer_ids + [248044]
            close = [248069, 271]
            final_ids = (
                source["retained_thinking_token_ids"]
                + close
                + prefix
                + answer_ids
            )
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    **prompt,
                    "outputs": [
                        {
                            "sample_index": 0,
                            "stage1_parent_seed": source["stage1_parent_seed"],
                            "text": self.tokenizer.decode(
                                final_ids, skip_special_tokens=False
                            ),
                            "token_ids": final_ids,
                            "stage1_token_ids": source["stage1_token_ids"],
                            "retained_thinking_token_ids": source[
                                "retained_thinking_token_ids"
                            ],
                            "seed_stage1": source["seed_stage1"],
                            "seed_stage2": _stable_seed(
                                sampling.run_seed, record["id"], 0, "answer"
                            ),
                            "seed_domain_stage1": "thought",
                            "seed_domain_stage2": "answer",
                            "answer_prefix_token_ids": prefix,
                            "injected_token_ids": close + prefix,
                            "n_answer_tokens": len(answer_ids),
                            "n_thinking_tokens": len(
                                source["retained_thinking_token_ids"]
                            ),
                            "stage2_token_ids": stage2,
                            "n_stage1_prompt_tokens": len(original),
                            "n_stage2_prompt_tokens": len(original)
                            + len(source["retained_thinking_token_ids"])
                            + len(close)
                            + len(prefix),
                            "n_sampled_tokens": len(source["stage1_token_ids"])
                            + len(stage2),
                            "n_injected_tokens": len(close) + len(prefix),
                            "n_completion_tokens": len(final_ids),
                            "n_terminal_tokens_trimmed": 2,
                            "n_tokens_discarded_after_close": 0,
                            "thinking_closed": True,
                            "forced_close": True,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": source[
                                "stage1_finish_reason"
                            ],
                            "stage1_stop_reason": 248044,
                            "truncated": False,
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
        rows = []
        for index, record in enumerate(records):
            prompt, original = self._prompt_fields(
                record, thinking=False, prefix=prefix
            )
            expected = record["meta"]["expected"]
            answer = expected[len("PROGRAM:") :] if prefix else expected
            answer_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            stage1 = answer_ids + [248044]
            final_ids = prefix + answer_ids
            seed = _stable_seed(sampling.run_seed, record["id"], 0, "answer")
            rows.append(
                {
                    "id": record["id"],
                    "meta": record["meta"],
                    **prompt,
                    "outputs": [
                        {
                            "sample_index": 0,
                            "stage1_parent_seed": seed,
                            "seed_stage1": seed,
                            "seed_stage2": None,
                            "seed_domain_stage1": "answer",
                            "seed_domain_stage2": None,
                            "text": self.tokenizer.decode(
                                final_ids, skip_special_tokens=False
                            ),
                            "token_ids": final_ids,
                            "stage1_token_ids": stage1,
                            "answer_prefix_token_ids": prefix,
                            "injected_token_ids": prefix,
                            "stage2_token_ids": [],
                            "n_answer_tokens": len(answer_ids),
                            "n_thinking_tokens": 0,
                            "n_stage1_prompt_tokens": len(original) + len(prefix),
                            "n_stage2_prompt_tokens": 0,
                            "n_sampled_tokens": len(stage1),
                            "n_injected_tokens": len(prefix),
                            "n_completion_tokens": len(final_ids),
                            "n_terminal_tokens_trimmed": 1,
                            "thinking_closed": False,
                            "forced_close": False,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": "stop",
                            "stage1_stop_reason": 248044,
                            "truncated": False,
                        }
                    ],
                }
            )
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
        tokenizer = _FakeTokenizer()
        fake = _FakeCalibrationRunner(runner_path, tokenizer)
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
            tokenizer=tokenizer,
        )
        self.assertEqual(decision["decision"], "CALIBRATION_INTERFACE_SELECTED")
        self.assertEqual(decision["winner"], "think512_freeform")
        self.assertTrue(decision["pairing"]["exact_stage1_token_pairing"])
        self.assertEqual(
            decision["metrics"]["think512_program_slot"]["exact_echo_successes"],
            48,
        )

        bundles = {
            invocation: json.loads(
                artifact_paths(raw, invocation)["bundle"].read_text()
            )
            for invocation in INVOCATION_ORDER
        }
        forged = json.loads(json.dumps(bundles))
        output = forged["no_think_freeform"]["rows"][0]["outputs"][0]
        output["seed_domain_stage1"] = "thought"
        output["text"] = "junk</think>\n\n" + inputs.records[0]["meta"]["expected"]
        with self.assertRaisesRegex(RuntimeError, "semantic fields changed"):
            _authenticate_factorial_pairing(forged, inputs, tokenizer)

        forged_seed = json.loads(json.dumps(bundles))
        source = forged_seed["calibration_thoughts"]["rows"][0]["outputs"][0]
        source["stage1_parent_seed"] += 1
        source["seed_stage1"] += 1
        for arm in ("think512_freeform", "think512_program_slot"):
            output = forged_seed[arm]["rows"][0]["outputs"][0]
            output["stage1_parent_seed"] += 1
            output["seed_stage1"] += 1
            output["seed_stage2"] += 1
        for arm in ("no_think_freeform", "no_think_program_slot"):
            output = forged_seed[arm]["rows"][0]["outputs"][0]
            output["stage1_parent_seed"] += 1
            output["seed_stage1"] += 1
        with self.assertRaisesRegex(RuntimeError, "stable seed changed"):
            _authenticate_factorial_pairing(forged_seed, inputs, tokenizer)

        forged_finish = json.loads(json.dumps(bundles))
        output = forged_finish["no_think_freeform"]["rows"][0]["outputs"][0]
        output["finish_reason"] = "length"
        output["stage1_finish_reason"] = "length"
        output["truncated"] = True
        with self.assertRaisesRegex(RuntimeError, "registered cap"):
            _authenticate_factorial_pairing(forged_finish, inputs, tokenizer)

        for label, arm, field, size in (
            ("thought", "calibration_thoughts", "stage1_token_ids", 513),
            ("thinking answer", "think512_freeform", "stage2_token_ids", 25),
            ("no-think answer", "no_think_freeform", "stage1_token_ids", 25),
        ):
            with self.subTest(over_cap=label):
                forged_cap = json.loads(json.dumps(bundles))
                output = forged_cap[arm]["rows"][0]["outputs"][0]
                output[field] = [1100] * size
                with self.assertRaisesRegex(RuntimeError, "registered cap"):
                    _authenticate_factorial_pairing(forged_cap, inputs, tokenizer)

        restarted = _FakeCalibrationRunner(
            runner_path, tokenizer, fail_on_call=True
        )
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
