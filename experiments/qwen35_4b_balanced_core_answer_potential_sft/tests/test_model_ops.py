from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model_ops import (  # noqa: E402
    AnswerPotentialModel,
    _extract_observed_logprob,
    _logsumexp,
    _loop_diagnostic,
    answer_mention,
    make_foreign_controls,
    make_token_shuffled_controls,
)


class ModelOperationHelperTests(unittest.TestCase):
    def test_canonical_joint_aggregation_scores_boundary_and_answer(self) -> None:
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                self.assert_text = text
                return [9, 10]

        model = object.__new__(AnswerPotentialModel)
        model.runner = SimpleNamespace(tokenizer=Tokenizer())
        model.answer_boundary_ids = [4, 5]
        model.thinking_prompt = MethodType(
            lambda self, item: (item["prompt"], [1, 2, 3]), model
        )

        def fake_score(self, requests, *, chunk_size):
            self.seen_chunk_size = chunk_size
            output = []
            for request in requests:
                target = request["full_token_ids"][request["answer_start"] :]
                values = (
                    [-1.0, -1.0, -2.0, -2.0]
                    if request["condition"] == "empty"
                    else [-0.5, -0.5, -1.0, -1.0]
                )
                output.append(
                    {
                        **{
                            key: value
                            for key, value in request.items()
                            if key != "full_token_ids"
                        },
                        "answer_token_ids": target,
                        "answer_token_logprobs": values,
                    }
                )
            return output

        model._score_sequence_requests = MethodType(fake_score, model)
        model.metadata = MethodType(
            lambda self, **kwargs: {"operation": kwargs["operation"]}, model
        )
        model.capacity_receipt = MethodType(lambda self, **kwargs: kwargs, model)
        item = {
            "id": "task",
            "prompt": "p",
            "canonical_answer": "7",
            "potential_scorable": True,
        }
        trace = {
            "trace_id": "trace",
            "task_id": "task",
            "family": "f",
            "level": 2,
            "token_ids": [6, 7],
            "n_tokens": 2,
            "natural_close": True,
            "loop_flag": False,
        }
        rows, receipt = model.score_canonical_joint([item], [trace], chunk_size=17)
        self.assertEqual(len(rows), 1)
        self.assertEqual(model.seen_chunk_size, 17)
        self.assertEqual(rows[0]["boundary_token_ids"], [4, 5])
        self.assertEqual(rows[0]["answer_token_ids"], [9, 10])
        self.assertEqual(rows[0]["answer_gain_sum"], 2.0)
        self.assertEqual(rows[0]["joint_gain_sum"], 3.0)
        self.assertEqual(rows[0]["answer_gain_per_answer_token"], 1.0)
        self.assertEqual(rows[0]["joint_gain_per_answer_token"], 1.5)
        self.assertEqual(rows[0]["full_sequence_tokens"], 9)
        self.assertEqual(receipt["operation"], "canonical_joint_answer_potential")

    def test_observed_prompt_token_is_extracted_even_if_not_top_ranked(self) -> None:
        entry = {
            11: SimpleNamespace(logprob=-0.1),
            42: SimpleNamespace(logprob=-3.5),
        }
        self.assertEqual(_extract_observed_logprob(entry, 42), -3.5)
        with self.assertRaisesRegex(RuntimeError, "missing"):
            _extract_observed_logprob(entry, 7)

    def test_logsumexp(self) -> None:
        got = _logsumexp([-2.0, -3.0])
        expected = math.log(math.exp(-2.0) + math.exp(-3.0))
        self.assertTrue(math.isclose(got, expected))

    def test_controls_preserve_task_and_length_contracts(self) -> None:
        traces = []
        for task, length in (("a", 3), ("b", 5), ("c", 4)):
            traces.append(
                {
                    "trace_id": task,
                    "task_id": task,
                    "family": "f",
                    "level": 1,
                    "token_ids": list(range(length)),
                    "n_tokens": length,
                    "text": task,
                }
            )
        shuffled = make_token_shuffled_controls(traces, seed=19)
        foreign = make_foreign_controls(traces)
        self.assertEqual(len(shuffled), len(traces))
        self.assertEqual(len(foreign), len(traces))
        self.assertEqual(
            [len(row["token_ids"]) for row in shuffled],
            [row["n_tokens"] for row in traces],
        )
        self.assertTrue(
            all(row["foreign_source_task_id"] != row["task_id"] for row in foreign)
        )

    def test_answer_mentions_use_boundaries(self) -> None:
        self.assertEqual(answer_mention("we get 42 exactly", "42"), 7)
        self.assertIsNone(answer_mention("we get 142 exactly", "42"))

    def test_loop_diagnostic_requires_an_exact_periodic_suffix(self) -> None:
        coherent = []
        for index in range(100):
            coherent.extend([1, 2, 3, 1000 + index])
        self.assertGreaterEqual(_loop_diagnostic(coherent)["max_trigram_count"], 8)
        self.assertFalse(_loop_diagnostic(coherent)["loop_flag"])

        periodic = list(range(80)) + ([7, 8, 9, 10] * 20)
        result = _loop_diagnostic(periodic)
        self.assertTrue(result["loop_flag"])
        self.assertEqual(result["periodic_suffix_period"], 4)
        self.assertGreaterEqual(result["periodic_suffix_tokens"], 64)


if __name__ == "__main__":
    unittest.main()
