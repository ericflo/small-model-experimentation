from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import sys
import types
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from mechanics_runtime import (  # noqa: E402
    authenticate_selected_interface_bundle,
    generate_selected_interface,
)
import vllm_runner  # noqa: E402
from vllm_runner import SamplingConfig  # noqa: E402


class FakeTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ):
        if tokenize or not add_generation_prompt or enable_thinking:
            raise AssertionError("unexpected chat-template mode")
        return "OFF:" + messages[0]["content"]

    def encode(self, text, *, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special-token injection is forbidden")
        if text == "PROGRAM:":
            return [78041, 25]
        return [1000 + ord(character) for character in text]

    def decode(self, token_ids, *, skip_special_tokens=False):
        if skip_special_tokens:
            raise AssertionError("special tokens must remain visible")
        pieces = []
        for token_id in token_ids:
            if token_id == 78041:
                pieces.append("PROGRAM")
            elif token_id == 25:
                pieces.append(":")
            elif token_id == 248046:
                pieces.append("<|im_end|>")
            else:
                pieces.append(chr(token_id - 1000))
        return "".join(pieces)


class FakeLLM:
    def __init__(self, tokenizer, expected):
        self.tokenizer = tokenizer
        self.expected = expected

    def generate(self, prompts, params, *, use_tqdm, lora_request):
        if use_tqdm or lora_request is not None or len(prompts) != len(self.expected):
            raise AssertionError("unexpected fake generation geometry")
        result = []
        for expected in self.expected:
            suffix = expected.removeprefix("PROGRAM:")
            raw = self.tokenizer.encode(suffix, add_special_tokens=False) + [248046]
            completion = types.SimpleNamespace(
                index=0,
                token_ids=raw,
                finish_reason="stop",
                stop_reason=248046,
                cumulative_logprob=-1.0,
                logprobs=None,
            )
            result.append(types.SimpleNamespace(outputs=[completion]))
        return result


class FakeRunner:
    def __init__(self, records):
        self.tokenizer = FakeTokenizer()
        self.llm = FakeLLM(
            self.tokenizer, [record["meta"]["expected"] for record in records]
        )
        self.lora_request = None
        self.engine = {"fake_engine": "bound"}
        self.engine_args = {"seed": 0, "fake": True}
        self.resolved_cudagraph = {"mode": "fake"}
        self.resolved_logprobs_mode = "raw_logprobs"
        self.runtime = {
            "python": "3.12.0",
            "python_executable": "/pinned/python",
            "platform": "fake",
            "packages": {"vllm": "0.24.0+cu129"},
            "environment_lock": {"sha256": "a" * 64},
            "uv": "uv 1",
            "cuda_toolkit": "cuda 12.9",
            "gpu": "fake gpu",
            "vllm_enable_v1_multiprocessing": "0",
            "git_commit": "b" * 40,
            "git_dirty": True,
        }

    def prepare(self, records, thinking, allow_custom_prompts):
        if thinking != "off" or allow_custom_prompts:
            raise AssertionError("winner escaped no-think")
        prepared = []
        for record in records:
            text = self.tokenizer.apply_chat_template(
                record["messages"],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            prepared.append(
                types.SimpleNamespace(
                    record_id=record["id"],
                    meta=record["meta"],
                    prompt_text=text,
                    prompt_token_ids=self.tokenizer.encode(text),
                    prompt_channel="off",
                )
            )
        return prepared

    def _check_context(self, prepared, sampling, prefix_ids):
        if not prepared or not prefix_ids or sampling.max_tokens != 24:
            raise AssertionError("context check changed")

    def _params(self, sampling, *, max_tokens, seed, n, stop_token_id):
        if max_tokens != 24 or n != 1 or stop_token_id != 248046:
            raise AssertionError("registered stop changed")
        return types.SimpleNamespace(seed=seed)

    def _generation_summary(
        self, rows, sampling, elapsed, *, generation_mode, extra
    ):
        outputs = [row["outputs"][0] for row in rows]
        prompt = sum(output["n_stage1_prompt_tokens"] for output in outputs)
        sampled = sum(output["n_sampled_tokens"] for output in outputs)
        injected = sum(output["n_injected_tokens"] for output in outputs)
        return {
            "schema_version": 6,
            "generation_mode": generation_mode,
            "model": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "runner_sha256": hashlib.sha256(
                Path(vllm_runner.__file__).resolve().read_bytes()
            ).hexdigest(),
            "engine": self.engine,
            "engine_args": self.engine_args,
            "resolved_cudagraph": self.resolved_cudagraph,
            "resolved_logprobs_mode": self.resolved_logprobs_mode,
            "sampling": dataclasses.asdict(sampling),
            "resolved_sampling": sampling.resolved_sampling(),
            "adapter": None,
            "think_token_ids": {
                "open": 248068,
                "close": 248069,
                "forced_close_sequence": [248069, 271],
                "thinking_prompt_suffix": [248045, 74455, 198, 248068, 198],
                "no_thinking_prompt_suffix": [
                    248045,
                    74455,
                    198,
                    248068,
                    271,
                    248069,
                    271,
                ],
            },
            "termination": {
                "hf_model_eos_token_id": 248044,
                "vllm_tokenizer_eos_ignored": 248046,
            },
            "rng_isolation": {
                "engine_seed": 0,
                "caller_global_rng_state_restored": True,
            },
            "counts": {
                "requests": len(rows),
                "completions": len(rows),
                "unique_input_prompt_tokens": sum(
                    row["n_prompt_tokens"] for row in rows
                ),
                "stage1_logical_prompt_tokens": prompt,
                "stage2_logical_prompt_tokens": 0,
                "logical_model_input_tokens": prompt,
                "logical_prompt_tokens": prompt,
                "physical_prompt_tokens": prompt,
                "reused_prompt_tokens": 0,
                "sampled_tokens": sampled,
                "physical_sampled_tokens": sampled,
                "reused_sampled_tokens": 0,
                "logical_model_tokens": prompt + sampled,
                "physical_model_tokens": prompt + sampled,
                "reused_model_tokens": 0,
                "injected_tokens": injected,
            },
            "timing": {
                "model_load_seconds": 1.0,
                "generation_seconds": elapsed,
                "sampled_tokens_per_second": sampled / elapsed,
            },
            "runtime": self.runtime,
            **extra,
        }


class MechanicsRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "id": "r0",
                "messages": [{"role": "user", "content": "echo"}],
                "meta": {"expected": "PROGRAM: A | B", "arity": 2},
            },
            {
                "id": "r1",
                "messages": [{"role": "user", "content": "echo"}],
                "meta": {"expected": "PROGRAM: A | B | C", "arity": 3},
            },
        ]
        self.sampling = SamplingConfig(
            thinking="off",
            n=1,
            max_tokens=24,
            answer_max_tokens=24,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            run_seed=123,
            answer_prefix="PROGRAM:",
            paired_answer_seed=True,
        )
        self.receipt = {
            "program_slot_prefix_token_ids": [78041, 25],
            "termination": {
                "tokenizer_eos_token_id": 248046,
                "hf_model_eos_token_id": 248044,
            },
            "think_token_ids": {
                "open": [248068],
                "close": [248069],
                "forced_close_sequence": [248069, 271],
                "thinking_prompt_suffix": [248045, 74455, 198, 248068, 198],
                "no_thinking_prompt_suffix": [
                    248045,
                    74455,
                    198,
                    248068,
                    271,
                    248069,
                    271,
                ],
            },
        }

    @staticmethod
    def engine_receipt(runner):
        runtime = copy.deepcopy(runner.runtime)
        runtime["git_dirty"] = False
        return {
            "runner_sha256": hashlib.sha256(
                Path(vllm_runner.__file__).resolve().read_bytes()
            ).hexdigest(),
            "engine": runner.engine,
            "engine_args_sha256": hashlib.sha256(
                json.dumps(
                    runner.engine_args, sort_keys=True, separators=(",", ":")
                ).encode()
            ).hexdigest(),
            "resolved_cudagraph": runner.resolved_cudagraph,
            "resolved_logprobs_mode": runner.resolved_logprobs_mode,
            "adapter": None,
            "rng_isolation": {
                "engine_seed": 0,
                "caller_global_rng_state_restored": True,
            },
            "runtime": runtime,
        }

    def test_generate_and_authenticate_exact_tokenizer_eos_boundary(self) -> None:
        runner = FakeRunner(self.records)
        rows, metadata = generate_selected_interface(
            runner, self.records, self.sampling
        )
        bundle = {"rows": rows, "runner_metadata": metadata}
        receipt = authenticate_selected_interface_bundle(
            records=self.records,
            bundle=bundle,
            sampling=self.sampling,
            tokenizer=runner.tokenizer,
            tokenizer_receipt=self.receipt,
            engine_receipt=self.engine_receipt(runner),
        )
        self.assertEqual(receipt["rows"], 2)
        self.assertTrue(receipt["terminal_geometry_authenticated"])
        self.assertTrue(rows[0]["outputs"][0]["text"].endswith("<|im_end|>"))

    def test_authentication_rejects_internal_or_spoofed_stop(self) -> None:
        runner = FakeRunner(self.records)
        rows, metadata = generate_selected_interface(
            runner, self.records, self.sampling
        )
        for label, mutate in (
            (
                "internal",
                lambda value: value["rows"][0]["outputs"][0][
                    "raw_answer_token_ids"
                ].insert(0, 248046),
            ),
            (
                "boolean_cap",
                lambda value: value["rows"][0]["outputs"][0].__setitem__(
                    "answer_cap_contact", 0
                ),
            ),
            (
                "integer_thinking_closed",
                lambda value: value["rows"][0]["outputs"][0].__setitem__(
                    "thinking_closed", 0
                ),
            ),
            (
                "integer_sampling_boolean",
                lambda value: value["runner_metadata"]["sampling"].__setitem__(
                    "paired_answer_seed", 1
                ),
            ),
            (
                "boolean_count",
                lambda value: value["runner_metadata"]["counts"].__setitem__(
                    "reused_prompt_tokens", False
                ),
            ),
            (
                "forged_logprobs",
                lambda value: value["rows"][0]["outputs"][0].__setitem__(
                    "stage1_logprobs", [{"forged": True}]
                ),
            ),
            (
                "unregistered_output_key",
                lambda value: value["rows"][0]["outputs"][0].__setitem__(
                    "unregistered", 0
                ),
            ),
        ):
            with self.subTest(label=label):
                forged = copy.deepcopy({"rows": rows, "runner_metadata": metadata})
                mutate(forged)
                with self.assertRaises(RuntimeError):
                    authenticate_selected_interface_bundle(
                        records=self.records,
                        bundle=forged,
                        sampling=self.sampling,
                        tokenizer=runner.tokenizer,
                        tokenizer_receipt=self.receipt,
                        engine_receipt=self.engine_receipt(runner),
                    )

    def test_authentication_survives_exact_durable_json_round_trip(self) -> None:
        runner = FakeRunner(self.records)
        rows, metadata = generate_selected_interface(
            runner, self.records, self.sampling
        )
        bundle = json.loads(json.dumps({"rows": rows, "runner_metadata": metadata}))
        receipt = authenticate_selected_interface_bundle(
            records=self.records,
            bundle=bundle,
            sampling=self.sampling,
            tokenizer=runner.tokenizer,
            tokenizer_receipt=self.receipt,
            engine_receipt=self.engine_receipt(runner),
        )
        self.assertEqual(receipt["rows"], 2)


if __name__ == "__main__":
    unittest.main()
