from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sibling_policy.py"
SPEC = importlib.util.spec_from_file_location("successful_sibling_policy", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
POLICY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(POLICY)


def source(skill: str = "execute", task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "selection_skill": skill,
        "kind": f"u_{skill}",
        "messages": [{"role": "user", "content": "solve"}],
        "answer": "ANSWER: RIGHT",
        "surface": "fixture",
        "level": 1,
    }


def output(
    index: int,
    *,
    answer: str = "RIGHT",
    sampled: int = 60,
    thinking: int = 48,
    closed: bool = True,
    canonical: bool = True,
) -> dict:
    tail = f"ANSWER: {answer}<|im_end|>" if canonical else f"extra\nANSWER: {answer}<|im_end|>"
    return {
        "sample_index": index,
        "text": f"reasoning {index}\n</think>\n\n{tail}\n",
        "n_sampled_tokens": sampled,
        "n_thinking_tokens": thinking,
        "thinking_closed": closed,
        "finish_reason": "stop" if closed else "length",
        "truncated": not closed,
    }


class SiblingPolicyTests(unittest.TestCase):
    def test_residual_skills_are_prospective_and_exclude_saturated_or_thin_skills(self) -> None:
        self.assertEqual(
            POLICY.EXPECTED_SKILLS,
            ("induct", "execute", "trace", "verify", "repair", "optimize", "abstain", "state", "order", "probe"),
        )
        self.assertEqual(POLICY.SELECTION_SEED, 55116)

    def test_only_hard_greedy_failures_are_eligible(self) -> None:
        row = source()
        wrong = POLICY.grade_greedy(row, {"outputs": [output(0, answer="WRONG")]})
        right = POLICY.grade_greedy(row, {"outputs": [output(0)]})
        self.assertTrue(wrong["hard_failure"])
        self.assertEqual(wrong["reasons"], ["wrong_answer"])
        self.assertFalse(right["hard_failure"])

    def test_sibling_requires_correct_canonical_short_natural_stop(self) -> None:
        row = source()
        self.assertTrue(POLICY.parse_successful_sibling(row, output(0))["qualified"])
        self.assertFalse(POLICY.parse_successful_sibling(row, output(0, answer="WRONG"))["qualified"])
        self.assertFalse(POLICY.parse_successful_sibling(row, output(0, closed=False))["qualified"])
        self.assertFalse(POLICY.parse_successful_sibling(row, output(0, thinking=769))["qualified"])
        self.assertFalse(POLICY.parse_successful_sibling(row, output(0, canonical=False))["qualified"])

    def test_shortest_qualified_sample_wins(self) -> None:
        samples = [output(index, answer="WRONG") for index in range(16)]
        samples[4] = output(4, sampled=90, thinking=70)
        samples[9] = output(9, sampled=50, thinking=40)
        best, grades = POLICY.choose_best_sibling(source(), {"outputs": samples})
        self.assertIsNotNone(best)
        self.assertEqual(best["sample_index"], 9)
        self.assertEqual(sum(item["qualified"] for item in grades), 2)

    def test_selection_requires_four_supported_tasks_per_skill(self) -> None:
        template = POLICY.parse_successful_sibling(source(), output(0))
        candidates = []
        for skill in POLICY.EXPECTED_SKILLS:
            for index in range(4):
                candidates.append({
                    "task_id": f"{skill}-{index}",
                    "skill": skill,
                    "kind": f"u_{skill}",
                    "best": {**template, "n_sampled_tokens": 40 + index},
                })
        selected, availability = POLICY.select_balanced(candidates)
        self.assertEqual(len(selected), 40)
        self.assertEqual(Counter(item["skill"] for item in selected), Counter({skill: 4 for skill in POLICY.EXPECTED_SKILLS}))
        self.assertEqual(set(availability.values()), {4})
        selected, _ = POLICY.select_balanced(candidates[:-1])
        self.assertEqual(selected, [])

    def test_training_row_uses_sampled_trace_not_oracle_think(self) -> None:
        row = source()
        best = POLICY.parse_successful_sibling(row, output(3))
        selected = {"task_id": row["task_id"], "skill": "execute", "best": best}
        training = POLICY.training_row(row, selected, "a" * 64)
        self.assertEqual(training["think"], "reasoning 3")
        self.assertEqual(training["answer"], row["answer"])
        self.assertFalse(training["teacher"]["oracle_trace_used"])
        self.assertNotIn("_audit", training)


if __name__ == "__main__":
    unittest.main()
