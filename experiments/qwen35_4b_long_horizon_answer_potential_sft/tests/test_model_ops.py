from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from model_ops import (  # noqa: E402
    _extract_observed_logprob,
    _logsumexp,
    answer_mention,
    make_foreign_controls,
    make_token_shuffled_controls,
)


class ModelOperationHelperTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
