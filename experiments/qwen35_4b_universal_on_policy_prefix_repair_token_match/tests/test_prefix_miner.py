from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "mine_prefix_repairs.py"
SPEC = importlib.util.spec_from_file_location("on_policy_prefix_miner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def task(failure_class: str, task_id: str = "task-1") -> dict:
    return {
        "task_id": task_id,
        "failure_class": failure_class,
        "messages": [{"role": "user", "content": "solve"}],
        "oracle_think": "Verified corrective reasoning.",
        "answer": "ANSWER: RIGHT",
    }


def rollout(
    failure_class: str,
    *,
    thought: str = "parent reasoning",
    answer: str = "WRONG",
    thinking_tokens: int | None = None,
    truncated: bool = False,
) -> dict:
    count = thinking_tokens if thinking_tokens is not None else max(1, len(thought))
    prefix_ids = list(range(1000, 1000 + count))
    return {
        "id": "task-1",
        "meta": {"failure_class": failure_class},
        "outputs": [
            {
                "text": f"{thought}</think>\n\nANSWER: {answer}",
                "token_ids": prefix_ids + [MODULE.THINK_CLOSE_ID, 2000],
                "n_thinking_tokens": count,
                "n_answer_tokens": 1,
                "truncated": truncated,
                "finish_reason": "length" if truncated else "stop",
                "seed_stage1": 17,
            }
        ],
    }


class PrefixMinerTests(unittest.TestCase):
    def test_wrong_answer_selects_exact_parent_prefix_and_masks_it(self) -> None:
        source = task("state_transition")
        event = rollout("state_transition")
        with mock.patch.object(MODULE, "QUOTA_PER_CLASS", 1):
            rows, inventory = MODULE.analyze([source], [event])
        self.assertTrue(inventory["quota_satisfied"])
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertTrue(row["prefix_loss_masked"])
        self.assertEqual(row["assistant_prefix_token_ids"], event["outputs"][0]["token_ids"][:-2])
        self.assertEqual(row["assistant_prefix_text"], "parent reasoning")
        self.assertIn("wrong_answer", row["provenance"]["failure_reasons"])
        self.assertEqual(row["provenance"]["first_observable_boundary"], "answer_boundary")

    def test_exact_correct_answer_is_not_mislabeled_as_failure(self) -> None:
        source = task("probe_scoring")
        event = rollout("probe_scoring", answer="RIGHT")
        with mock.patch.object(MODULE, "QUOTA_PER_CLASS", 1):
            rows, inventory = MODULE.analyze([source], [event])
        self.assertEqual(rows, [])
        self.assertFalse(inventory["quota_satisfied"])
        self.assertEqual(inventory["available_reachable_failures"], {"probe_scoring": 0})

    def test_delayed_commit_cuts_at_first_token_beyond_frozen_budget(self) -> None:
        source = task("commit_serialization")
        event = rollout(
            "commit_serialization", answer="RIGHT", thinking_tokens=40
        )
        with mock.patch.object(MODULE, "QUOTA_PER_CLASS", 1):
            rows, inventory = MODULE.analyze([source], [event])
        self.assertTrue(inventory["quota_satisfied"])
        self.assertEqual(len(rows[0]["assistant_prefix_token_ids"]), 33)
        self.assertNotIn("assistant_prefix_text", rows[0])
        self.assertEqual(
            rows[0]["provenance"]["first_observable_boundary"],
            "first_token_beyond_commit_budget",
        )

    def test_reference_cycle_execution_is_an_observable_policy_failure(self) -> None:
        source = task("declaration_operation")
        event = rollout(
            "declaration_operation",
            thought="I will apply the cycle before the listed steps.",
            answer="RIGHT",
        )
        reasons = MODULE.failure_reasons(source, event["outputs"][0])
        self.assertEqual(reasons, ["declaration_executed_as_operation"])

    def test_generation_cap_is_a_failure_even_without_an_answer(self) -> None:
        source = task("bounded_induction")
        event = rollout(
            "bounded_induction", thought="search repeats", answer="WRONG", truncated=True
        )
        reasons = MODULE.failure_reasons(source, event["outputs"][0])
        self.assertIn("generation_cap", reasons)
        self.assertIn("wrong_answer", reasons)


if __name__ == "__main__":
    unittest.main()
