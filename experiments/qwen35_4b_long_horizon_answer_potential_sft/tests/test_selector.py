from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from selector import (  # noqa: E402
    deranged_sources,
    select_quality_diverse,
    sft_record,
    structural_distance,
)


class SelectorTests(unittest.TestCase):
    def test_quality_then_diversity_not_brevity(self) -> None:
        rows = [
            {"trace_id": "a", "answer_gain": 2.0, "text": "same plan 1 then finish", "n_tokens": 500},
            {"trace_id": "b", "answer_gain": 1.9, "text": "same plan 2 then finish", "n_tokens": 10},
            {"trace_id": "c", "answer_gain": 1.8, "text": "graph invariant backwards proof", "n_tokens": 900},
        ]
        got = select_quality_diverse(rows, metric="answer_gain", top_candidates=3, near_best=0.25, max_selected=2)
        self.assertEqual([row["trace_id"] for row in got], ["a", "c"])

    def test_normalization_removes_numeric_identity(self) -> None:
        self.assertLess(structural_distance("take 41 then add 2", "take 99 then add 7"), 0.1)

    def test_derangement_never_uses_same_task(self) -> None:
        rows = []
        for task in ("a", "b", "c"):
            for index in range(2):
                rows.append({"task_id": task, "trace_id": f"{task}{index}", "family": "f", "level": 1, "n_tokens": 10 + index, "token_ids": [index], "text": task})
        got = deranged_sources(rows)
        self.assertEqual(len(got), len(rows))
        self.assertTrue(all(row["task_id"] != row["shuffle_source_task_id"] for row in got))
        self.assertEqual(len({row["shuffle_source_trace_id"] for row in got}), len(rows))

    def test_shuffle_sft_audit_uses_actual_source(self) -> None:
        class Tokenizer:
            eos_token_id = 99

            def apply_chat_template(self, *args, **kwargs):
                return "prompt<think>\n"

            def encode(self, text, add_special_tokens=False):
                del add_special_tokens
                values = {
                    "prompt<think>\n": [1, 2, 3],
                    "<think>\n": [2, 3],
                    "</think>\n\nANSWER: ": [4, 5],
                    "7": [7],
                }
                return values[text]

        trace = {
            "trace_id": "target-trace",
            "task_id": "target-task",
            "token_ids": [8],
            "source_kind": "potential_shuffle",
            "shuffle_source_trace_id": "actual-source-trace",
            "shuffle_source_task_id": "actual-source-task",
        }
        item = {"id": "target-task", "family": "f", "level": 1, "prompt": "p", "canonical_answer": "7"}
        row = sft_record(arm="potential_shuffle", item=item, trace=trace, tokenizer=Tokenizer(), ordinal=1, max_length=20)
        self.assertEqual(row["source_trace_id"], "actual-source-trace")
        self.assertEqual(row["source_task_id"], "actual-source-task")


if __name__ == "__main__":
    unittest.main()
