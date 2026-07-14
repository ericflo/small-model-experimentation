from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_restarts.py"
SPEC = importlib.util.spec_from_file_location("counterfactual_restart_miner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def source(skill: str = "execute", task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "selection_skill": skill,
        "kind": f"u_{skill}",
        "messages": [{"role": "user", "content": "solve"}],
        "think": "Compute once, verify once, and commit.",
        "answer": "ANSWER: RIGHT",
        "surface": "fixture",
        "level": 1,
        "n_think_tokens": 9,
        "_audit": {"truth_valid": True},
    }


def rollout(*, answer: str | None = "WRONG", think_tokens: int = 40, truncated: bool = False) -> dict:
    answer_text = "" if answer is None else f"\n</think>\n\nANSWER: {answer}<|im_end|>\n"
    return {
        "id": "task-1",
        "outputs": [{
            "text": "parent reasoning" + answer_text,
            "n_thinking_tokens": think_tokens,
            "n_sampled_tokens": think_tokens + (0 if answer is None else 4),
            "thinking_closed": answer is not None and not truncated,
            "finish_reason": "length" if truncated else "stop",
            "truncated": truncated,
        }],
    }


class RestartMinerTests(unittest.TestCase):
    def test_wrong_answer_is_a_hard_failure(self) -> None:
        item = MODULE.classify(source(), rollout())
        self.assertTrue(item["eligible"])
        self.assertTrue(item["hard_failure"])
        self.assertEqual(item["reasons"], ["wrong_answer"])

    def test_correct_bounded_answer_is_not_selected(self) -> None:
        item = MODULE.classify(source(), rollout(answer="RIGHT"))
        self.assertFalse(item["eligible"])
        self.assertEqual(item["reasons"], [])

    def test_correct_overbudget_answer_is_a_policy_failure(self) -> None:
        item = MODULE.classify(source(), rollout(answer="RIGHT", think_tokens=129))
        self.assertTrue(item["eligible"])
        self.assertFalse(item["hard_failure"])
        self.assertEqual(item["reasons"], ["over_think_budget"])

    def test_cap_contact_is_hard_even_without_answer(self) -> None:
        item = MODULE.classify(source(), rollout(answer=None, think_tokens=1024, truncated=True))
        self.assertTrue(item["hard_failure"])
        self.assertIn("cap_contact", item["reasons"])
        self.assertIn("missing_answer", item["reasons"])

    def test_selection_requires_four_per_every_skill(self) -> None:
        items = []
        for skill in MODULE.EXPECTED_SKILLS:
            for index in range(4):
                item = MODULE.classify(source(skill, f"{skill}-{index}"), rollout())
                items.append(item)
        selected, availability = MODULE.select_inventory(items)
        self.assertEqual(len(selected), 52)
        self.assertEqual(Counter(item["skill"] for item in selected), Counter({skill: 4 for skill in MODULE.EXPECTED_SKILLS}))
        self.assertEqual(set(availability.values()), {4})
        selected, _ = MODULE.select_inventory(items[:-1])
        self.assertEqual(selected, [])

    def test_restart_discards_parent_prefix_and_preserves_oracle(self) -> None:
        row_source = source()
        item = MODULE.classify(row_source, rollout())
        row = MODULE.restart_row(row_source, item, "a" * 64)
        self.assertEqual(row["messages"], row_source["messages"])
        self.assertEqual(row["think"], row_source["think"])
        self.assertEqual(row["answer"], row_source["answer"])
        self.assertNotIn("assistant_prefix_token_ids", row)
        self.assertFalse(row["failure_selection"]["parent_prefix_in_training_context"])


if __name__ == "__main__":
    unittest.main()
