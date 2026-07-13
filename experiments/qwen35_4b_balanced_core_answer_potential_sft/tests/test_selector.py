from __future__ import annotations

import sys
import unittest
from itertools import permutations
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from selector import (  # noqa: E402
    deranged_sources,
    select_task,
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

    def test_quality_diverse_falls_back_to_second_ranked_for_balance(self) -> None:
        rows = [
            {"trace_id": "a", "answer_gain": 2.0, "text": "best", "n_tokens": 500},
            {"trace_id": "b", "answer_gain": 1.0, "text": "second", "n_tokens": 600},
            {"trace_id": "c", "answer_gain": 0.5, "text": "distant", "n_tokens": 700},
        ]
        got = select_quality_diverse(
            rows,
            metric="answer_gain",
            top_candidates=3,
            near_best=0.25,
            max_selected=2,
        )
        self.assertEqual([row["trace_id"] for row in got], ["a", "b"])
        self.assertEqual(
            [row["selection_mode"] for row in got],
            ["best", "fallback_second_ranked"],
        )
        self.assertEqual([row["selection_gap_from_best"] for row in got], [0.0, 1.0])

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

    def test_derangement_has_global_minimum_length_cost(self) -> None:
        lengths = {"a0": 1, "a1": 30, "b0": 2, "b1": 31, "c0": 8, "c1": 40}
        rows = [
            {
                "task_id": key[0],
                "trace_id": key,
                "family": "f",
                "level": 1,
                "n_tokens": length,
                "token_ids": [length],
                "text": key,
                "answer_gain_per_answer_token": float(length),
                "selection_mode": f"mode-{key}",
                "selection_gap_from_best": float(length) / 10.0,
            }
            for key, length in lengths.items()
        ]
        got = deranged_sources(rows)
        actual = sum(
            abs(int(row["shuffle_target_trace_tokens"]) - int(row["n_tokens"]))
            for row in got
        )
        ordered = sorted(rows, key=lambda row: (row["n_tokens"], row["task_id"], row["trace_id"]))
        possible = [
            sum(abs(left["n_tokens"] - right["n_tokens"]) for left, right in zip(ordered, perm))
            for perm in permutations(ordered)
            if all(left["task_id"] != right["task_id"] for left, right in zip(ordered, perm))
        ]
        self.assertEqual(actual, min(possible))
        by_source = {row["shuffle_source_trace_id"]: row for row in got}
        for source_id, row in by_source.items():
            self.assertEqual(row["selection_mode"], f"mode-{source_id}")
            self.assertEqual(
                row["selection_mode"], row["shuffle_source_selection_mode"]
            )
            self.assertNotEqual(
                row["shuffle_target_task_id"], row["shuffle_source_task_id"]
            )
            self.assertEqual(
                row["shuffle_target_selection_mode"],
                f"mode-{row['shuffle_target_trace_id']}",
            )

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
            "selection_mode": "source-mode",
            "selection_gap_from_best": 0.75,
            "shuffle_target_trace_id": "target-trace",
            "shuffle_target_task_id": "target-task",
            "shuffle_target_selection_mode": "target-mode",
            "shuffle_target_selection_gap_from_best": 0.0,
            "shuffle_source_trace_id": "actual-source-trace",
            "shuffle_source_task_id": "actual-source-task",
            "shuffle_source_selection_mode": "source-mode",
            "shuffle_source_selection_gap_from_best": 0.75,
        }
        item = {"id": "target-task", "family": "f", "level": 1, "prompt": "p", "canonical_answer": "7"}
        row = sft_record(arm="potential_shuffle", item=item, trace=trace, tokenizer=Tokenizer(), ordinal=1, max_length=20)
        self.assertEqual(row["source_trace_id"], "actual-source-trace")
        self.assertEqual(row["source_task_id"], "actual-source-task")
        self.assertEqual(row["selection_mode"], "source-mode")
        self.assertEqual(row["shuffle_target_selection_mode"], "target-mode")
        self.assertEqual(row["shuffle_source_selection_mode"], "source-mode")

    def test_task_selection_adds_shortest_and_keeps_random_distinct(self) -> None:
        traces = [
            {
                "trace_id": f"t{index}",
                "task_id": "task",
                "family": "f",
                "level": 1,
                "natural_close": True,
                "loop_flag": False,
                "n_tokens": length,
                "token_ids": [index],
                "text": f"plan {index}",
                "source_kind": "independent",
            }
            for index, length in enumerate((100, 20, 40, 60, 80))
        ]
        scores = [
            {
                "trace_id": row["trace_id"],
                "full_sequence_tokens": row["n_tokens"] + 10,
                "answer_gain_per_answer_token": 10.0 - index,
                "joint_gain_per_answer_token": 5.0 - index,
            }
            for index, row in enumerate(traces)
        ]
        rollouts = [
            {"trace_id": row["trace_id"], "any_success": index % 2 == 0}
            for index, row in enumerate(traces)
        ]
        got = select_task(
            traces,
            scores,
            rollouts,
            selector_config={
                "max_train_length": 1000,
                "top_candidates": 5,
                "near_best_nats_per_answer_token": 10.0,
                "max_per_task": 2,
            },
            seed=7,
        )
        self.assertEqual(
            [row["trace_id"] for row in got["shortest_natural"]], ["t1", "t2"]
        )
        answer_ids = {row["trace_id"] for row in got["answer_potential"]}
        random_ids = {row["trace_id"] for row in got["random_natural"]}
        self.assertFalse(answer_ids & random_ids)


if __name__ == "__main__":
    unittest.main()
