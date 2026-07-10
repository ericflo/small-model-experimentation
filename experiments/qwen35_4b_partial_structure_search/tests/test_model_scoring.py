from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "model_scoring.py"
SPEC = importlib.util.spec_from_file_location("partial_structure_model_scoring", MODULE_PATH)
assert SPEC and SPEC.loader
scoring = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = scoring
SPEC.loader.exec_module(scoring)


class FakeLogprob:
    def __init__(
        self,
        logprob: float,
        *,
        rank: int | None = None,
        decoded_token: str | None = None,
    ) -> None:
        self.logprob = logprob
        self.rank = rank
        self.decoded_token = decoded_token


def fake_completion(
    token_ids: list[int],
    *,
    logprobs: list[dict[int, object]] | None = None,
    finish_reason: str = "length",
    stop_reason: object = None,
    cumulative_logprob: float | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        token_ids=token_ids,
        logprobs=logprobs,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        cumulative_logprob=cumulative_logprob,
    )


def fake_request(completion: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(outputs=[completion], prompt_logprobs=None)


class FakeTokenizer:
    def __init__(self) -> None:
        self.label_ids = {
            label: 100 + index
            for index, label in enumerate(scoring.NEXT_OPERATION_LABELS)
        }

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise AssertionError("scoring must disable tokenizer-added special tokens")
        if text in self.label_ids:
            return [self.label_ids[text]]
        if text == scoring.ANSWER_PREFIX:
            return [510, 511]
        if text == scoring.THINKING_ANSWER_SUFFIX:
            return [248069, 512, 513]
        if text.startswith("<THINKING_CHAT>"):
            return [8001] + [1000 + ord(character) for character in text[15:]]
        if text.startswith("<OFF_CHAT>"):
            return [8002] + [1000 + ord(character) for character in text[10:]]
        return [9000 + ord(character) for character in text]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        if skip_special_tokens:
            raise AssertionError("diagnostic decoding must preserve special tokens")
        names = {
            700: "reason-one",
            701: "reason-two",
            702: "natural-reason",
            703: "discarded-answer",
            248069: "</think>",
            999: "<eos>",
        }
        return "|".join(names.get(token_id, f"tok-{token_id}") for token_id in token_ids)


class FakeLLM:
    def __init__(self, output_batches: list[list[SimpleNamespace]]) -> None:
        self.output_batches = list(output_batches)
        self.calls: list[dict[str, object]] = []

    def generate(
        self,
        prompts: list[dict[str, list[int]]],
        params: list[object],
        *,
        use_tqdm: bool,
        lora_request: object,
    ) -> list[SimpleNamespace]:
        self.calls.append(
            {
                "prompts": prompts,
                "params": params,
                "use_tqdm": use_tqdm,
                "lora_request": lora_request,
            }
        )
        if not self.output_batches:
            raise AssertionError("fake vLLM has no queued output batch")
        output = self.output_batches.pop(0)
        if len(output) != len(prompts):
            raise AssertionError("fake output/prompt batch length mismatch")
        return output


class FakeRunner:
    def __init__(self, output_batches: list[list[SimpleNamespace]]) -> None:
        self.tokenizer = FakeTokenizer()
        self.think_close_id = 248069
        self.hf_eos_id = 999
        self.config = SimpleNamespace(max_model_len=100_000)
        self.lora_request = SimpleNamespace(name="fake-adapter")
        self.llm = FakeLLM(output_batches)

    def _render_messages(
        self, messages: list[dict[str, str]], enable_thinking: bool
    ) -> str:
        tag = "<THINKING_CHAT>" if enable_thinking else "<OFF_CHAT>"
        body = "\n".join(
            f"{message['role']}::{message['content']}" for message in messages
        )
        return tag + body

    @staticmethod
    def _prompt_channel(token_ids: list[int]) -> str:
        if token_ids[0] == 8001:
            return "thinking"
        if token_ids[0] == 8002:
            return "off"
        return "custom"

    @staticmethod
    def _trim_hf_eos(token_ids: list[int]) -> list[int]:
        return token_ids[: token_ids.index(999)] if 999 in token_ids else list(token_ids)

    @staticmethod
    def _decode(token_ids: list[int]) -> str:
        return FakeTokenizer().decode(token_ids, skip_special_tokens=False)

    @staticmethod
    def _params(
        sampling: object, *, max_tokens: int, seed: int, n: int
    ) -> SimpleNamespace:
        return SimpleNamespace(
            sampling=sampling,
            max_tokens=max_tokens,
            seed=seed,
            n=n,
        )


def viability_record(record_id: str = "r1") -> dict[str, object]:
    return {
        "id": record_id,
        "task_text": "Map each visible input to its output.",
        "visible_examples": [
            {"input": [1, 2], "output": [2, 4]},
            {"input": [3], "output": [6]},
        ],
        "candidate_prefix": ["double"],
        "remaining_steps": 2,
        "depth": 3,
    }


class PromptConstructionTests(unittest.TestCase):
    def test_binary_prompt_is_stable_and_uses_only_caller_visible_examples(self) -> None:
        messages = scoring.build_binary_viability_messages(
            " task line one\r\ntask line two ",
            [" first op ", "second op"],
            3,
            visible_examples=[{"output": [2], "input": [1]}],
        )

        self.assertEqual(messages[0]["content"], scoring.VIABILITY_SYSTEM_PROMPT)
        user = messages[1]["content"]
        self.assertIn("task line one\ntask line two", user)
        self.assertNotIn("\r", user)
        self.assertIn('{"input":[1],"output":[2]}', user)
        self.assertIn("1. first op\n2. second op", user)
        self.assertIn("OPEN SLOTS\n3", user)
        self.assertIn("A = viable", user)
        self.assertIn("B = not viable", user)

    def test_next_operation_prompt_has_exactly_the_frozen_A_to_P_menu(self) -> None:
        choices = [f"operation-{index:02d}" for index in range(16)]
        messages = scoring.build_next_operation_messages(
            "visible task", [], 4, choices, visible_examples="x -> y"
        )
        user = messages[1]["content"]

        self.assertIn("A = operation-00", user)
        self.assertIn("P = operation-15", user)
        self.assertEqual(sum(f"{label} =" in user for label in scoring.NEXT_OPERATION_LABELS), 16)
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            scoring.build_next_operation_messages("task", [], 1, choices[:-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            scoring.build_next_operation_messages("task", [], 1, ["same"] * 16)
        with self.assertRaisesRegex(ValueError, "at least one"):
            scoring.build_next_operation_messages("task", [], 0, choices)


class LabelAndLogprobTests(unittest.TestCase):
    def test_label_ids_are_dynamic_single_tokens_and_distinct(self) -> None:
        tokenizer = FakeTokenizer()
        resolved = scoring.single_token_label_ids(tokenizer, ("A", "B", "P"))
        self.assertEqual(resolved, {"A": 100, "B": 101, "P": 115})

        class SplitTokenizer(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                if text == "B":
                    return [1, 2]
                return super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaisesRegex(RuntimeError, "exactly one token"):
            scoring.single_token_label_ids(SplitTokenizer(), ("A", "B"))

        class CollisionTokenizer(FakeTokenizer):
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                if text in {"A", "B"}:
                    return [42]
                return super().encode(text, add_special_tokens=add_special_tokens)

        with self.assertRaisesRegex(RuntimeError, "duplicate token IDs"):
            scoring.single_token_label_ids(CollisionTokenizer(), ("A", "B"))

    def test_extract_accepts_raw_and_json_logprobs_then_normalizes_stably(self) -> None:
        token_ids = {"A": 100, "B": 101}
        raw = SimpleNamespace(
            logprobs=[
                {
                    100: FakeLogprob(-1000.0, rank=10, decoded_token="A"),
                    101: FakeLogprob(-1001.0, rank=20, decoded_token="B"),
                }
            ]
        )
        extracted = scoring.extract_target_logprobs(raw, token_ids)
        probabilities = scoring.normalize_label_logprobs(extracted)
        self.assertAlmostEqual(probabilities["A"], 1 / (1 + math.exp(-1)))
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

        normalized_runner_output = {
            "stage1_logprobs": [
                {
                    "100": {"logprob": -2.0, "rank": 3},
                    "101": {"logprob": -3.0, "rank": 7},
                }
            ]
        }
        self.assertEqual(
            scoring.extract_target_logprobs(normalized_runner_output, token_ids),
            {"A": -2.0, "B": -3.0},
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            scoring.extract_target_logprobs([{"100": {"logprob": -1.0}}], token_ids)
        with self.assertRaisesRegex(ValueError, "negative-infinite"):
            scoring.normalize_label_logprobs({"A": -math.inf, "B": -math.inf})


class NoThinkScoringTests(unittest.TestCase):
    def test_no_think_ptrue_uses_targeted_vllm_logits_and_full_accounting(self) -> None:
        runner = FakeRunner(
            [
                [
                    fake_request(
                        fake_completion(
                            [777],
                            logprobs=[
                                {
                                    100: FakeLogprob(-0.2, rank=2, decoded_token="A"),
                                    101: FakeLogprob(-1.2, rank=5, decoded_token="B"),
                                    777: FakeLogprob(-0.1, rank=1, decoded_token="X"),
                                }
                            ],
                        )
                    ),
                    fake_request(
                        fake_completion(
                            [101],
                            logprobs=[
                                {
                                    100: FakeLogprob(-2.0, decoded_token="A"),
                                    101: FakeLogprob(-0.1, decoded_token="B"),
                                }
                            ],
                        )
                    ),
                ]
            ]
        )
        scorer = scoring.ModelScorer(runner)
        records = [viability_record("r1"), viability_record("r2")]

        rows, summary = scorer.score_no_think_viability(records, run_seed=71)

        self.assertEqual([row["id"] for row in rows], ["r1", "r2"])
        self.assertAlmostEqual(rows[0]["score"], 1 / (1 + math.exp(-1)))
        self.assertLess(rows[1]["score"], 0.2)
        self.assertFalse(rows[0]["forced_close"])
        self.assertFalse(rows[0]["raw"]["sampled_label_is_audited"])
        self.assertEqual(rows[0]["meta"]["depth"], 3)
        self.assertEqual(rows[0]["request_count"], 1)
        self.assertEqual(rows[0]["sampled_tokens"], 1)
        self.assertEqual(
            rows[0]["accounting"]["injected_prompt_tokens"],
            len(scorer.answer_prefix_ids),
        )
        self.assertEqual(summary["vllm_batch_calls"], 1)
        self.assertEqual(summary["accounting"]["requests"], 2)
        self.assertEqual(summary["accounting"]["completions"], 2)
        self.assertEqual(summary["accounting"]["sampled_tokens"], 2)
        self.assertEqual(
            summary["accounting"]["prefill_tokens"],
            sum(len(prompt["prompt_token_ids"]) for prompt in runner.llm.calls[0]["prompts"]),
        )
        params = runner.llm.calls[0]["params"]
        self.assertEqual(params[0].sampling.logprobs, 2)
        self.assertEqual(params[0].sampling.logprob_token_ids, (100, 101))
        self.assertNotEqual(params[0].seed, params[1].seed)


class ThinkingScoringTests(unittest.TestCase):
    def test_two_pass_thinking_retains_exact_prefix_and_counts_both_requests(self) -> None:
        runner = FakeRunner(
            [
                [
                    fake_request(
                        fake_completion(
                            [700, 701, 999],
                            finish_reason="length",
                            cumulative_logprob=-2.5,
                        )
                    ),
                    fake_request(
                        fake_completion(
                            [702, 248069, 703],
                            finish_reason="stop",
                            stop_reason=248069,
                            cumulative_logprob=-1.5,
                        )
                    ),
                ],
                [
                    fake_request(
                        fake_completion(
                            [100],
                            logprobs=[
                                {
                                    100: FakeLogprob(-0.1, decoded_token="A"),
                                    101: FakeLogprob(-2.1, decoded_token="B"),
                                }
                            ],
                        )
                    ),
                    fake_request(
                        fake_completion(
                            [101],
                            logprobs=[
                                {
                                    100: FakeLogprob(-1.1, decoded_token="A"),
                                    101: FakeLogprob(-0.1, decoded_token="B"),
                                }
                            ],
                        )
                    ),
                ],
            ]
        )
        scorer = scoring.ModelScorer(runner)
        rows, summary = scorer.score_thinking_viability(
            [viability_record("forced"), viability_record("natural")],
            thinking_budget=32,
            run_seed=19,
        )

        self.assertEqual(len(runner.llm.calls), 2)
        phase1_prompts = runner.llm.calls[0]["prompts"]
        score_prompts = runner.llm.calls[1]["prompts"]
        self.assertEqual(
            score_prompts[0]["prompt_token_ids"],
            phase1_prompts[0]["prompt_token_ids"]
            + [700, 701]
            + scorer.thinking_answer_suffix_ids,
        )
        self.assertEqual(
            score_prompts[1]["prompt_token_ids"],
            phase1_prompts[1]["prompt_token_ids"]
            + [702]
            + scorer.thinking_answer_suffix_ids,
        )
        self.assertTrue(rows[0]["forced_close"])
        self.assertFalse(rows[1]["forced_close"])
        self.assertEqual(rows[0]["raw"]["retained_thinking_token_ids"], [700, 701])
        self.assertEqual(rows[1]["raw"]["retained_thinking_token_ids"], [702])
        self.assertEqual(rows[0]["request_count"], 2)
        self.assertEqual(rows[0]["sampled_tokens"], 4)
        self.assertEqual(rows[0]["accounting"]["retained_thinking_tokens"], 2)
        self.assertEqual(rows[0]["accounting"]["discarded_sampled_tokens"], 1)
        self.assertEqual(rows[1]["accounting"]["discarded_sampled_tokens"], 2)
        self.assertEqual(
            rows[0]["prefill_tokens"],
            len(phase1_prompts[0]["prompt_token_ids"])
            + len(score_prompts[0]["prompt_token_ids"]),
        )
        self.assertEqual(
            rows[0]["accounting"]["injected_prompt_tokens"],
            len(scorer.thinking_answer_suffix_ids),
        )
        self.assertGreater(rows[0]["score"], 0.8)
        self.assertLess(rows[1]["score"], 0.5)
        self.assertEqual(summary["vllm_batch_calls"], 2)
        self.assertEqual(summary["accounting"]["requests"], 4)
        self.assertEqual(summary["accounting"]["sampled_tokens"], 8)
        self.assertEqual(summary["accounting"]["retained_thinking_tokens"], 3)


class NextOperationScoringTests(unittest.TestCase):
    def test_sixteen_way_targeted_readout_returns_choice_distribution(self) -> None:
        choices = [f"op-{index}" for index in range(16)]
        logprobs = {
            100 + index: FakeLogprob(float(index - 16), decoded_token=label)
            for index, label in enumerate(scoring.NEXT_OPERATION_LABELS)
        }
        runner = FakeRunner(
            [[fake_request(fake_completion([115], logprobs=[logprobs]))]]
        )
        scorer = scoring.ModelScorer(runner)
        record = viability_record("next")
        record["operations"] = choices

        rows, summary = scorer.score_next_operation_likelihood([record], run_seed=5)

        row = rows[0]
        self.assertEqual(row["predicted_label"], "P")
        self.assertEqual(row["predicted_choice"], "op-15")
        self.assertEqual(row["ranked_choices"][0], "op-15")
        self.assertEqual(len(row["choice_probabilities"]), 16)
        self.assertAlmostEqual(sum(row["choice_probabilities"].values()), 1.0)
        self.assertEqual(row["score"], row["choice_probabilities"]["op-15"])
        params = runner.llm.calls[0]["params"][0]
        self.assertEqual(params.sampling.logprobs, 16)
        self.assertEqual(len(params.sampling.logprob_token_ids), 16)
        self.assertEqual(summary["accounting"]["requests"], 1)
        self.assertEqual(summary["accounting"]["sampled_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
