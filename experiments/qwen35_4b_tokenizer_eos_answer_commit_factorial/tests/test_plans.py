from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from plans import (  # noqa: E402
    completion_cost,
    conservative_first_over,
    freeze_taskwise_matches,
    pool_cost,
)


def output(sampled: int, stage1_prompt: int, stage2_prompt: int) -> dict[str, int]:
    return {
        "n_sampled_tokens": sampled,
        "n_stage1_prompt_tokens": stage1_prompt,
        "n_stage2_prompt_tokens": stage2_prompt,
    }


class PlanTests(unittest.TestCase):
    def test_completion_and_pool_costs_are_exact(self) -> None:
        first = output(7, 11, 13)
        second = output(5, 3, 0)
        self.assertEqual(
            completion_cost(first),
            {"sampled_tokens": 7, "logical_model_tokens": 31},
        )
        self.assertEqual(
            pool_cost([first, second]),
            {"sampled_tokens": 12, "logical_model_tokens": 39},
        )
        for field in first:
            invalid = copy.deepcopy(first)
            invalid[field] = True
            with self.subTest(field=field), self.assertRaisesRegex(
                ValueError, "invalid completion cost"
            ):
                completion_cost(invalid)

    def test_first_over_boundary_and_overshoot_are_conservative(self) -> None:
        pool = [output(3, 10, 0), output(4, 10, 0), output(8, 10, 0)]
        exact = conservative_first_over(
            pool, target=7, metric="sampled_tokens"
        )
        self.assertEqual(
            exact,
            {
                "metric": "sampled_tokens",
                "target": 7,
                "first_over_k": 2,
                "first_over_cost": 7,
                "under_k": 1,
                "under_cost": 3,
                "pool_exhausted": False,
            },
        )
        overshoot = conservative_first_over(
            pool, target=8, metric="sampled_tokens"
        )
        self.assertEqual(overshoot["first_over_k"], 3)
        self.assertEqual(overshoot["first_over_cost"], 15)
        self.assertEqual(overshoot["under_k"], 2)
        self.assertEqual(overshoot["under_cost"], 7)

    def test_pool_exhaustion_is_explicit(self) -> None:
        exhausted = conservative_first_over(
            [output(2, 1, 0), output(3, 1, 0)],
            target=6,
            metric="sampled_tokens",
        )
        self.assertEqual(exhausted["first_over_k"], None)
        self.assertEqual(exhausted["first_over_cost"], 5)
        self.assertEqual(exhausted["under_k"], 2)
        self.assertEqual(exhausted["under_cost"], 5)
        self.assertTrue(exhausted["pool_exhausted"])

    def test_taskwise_plan_uses_materialized_treatment_and_direct_order(self) -> None:
        treatment = [output(5, 7, 11), output(4, 6, 10)]
        direct = [output(3, 2, 0), output(3, 2, 0), output(3, 30, 0)]
        plan = freeze_taskwise_matches(
            task_id="task-1",
            treatment_outputs=treatment,
            direct_outputs=direct,
        )
        self.assertEqual(
            plan["treatment"],
            {"sampled_tokens": 9, "logical_model_tokens": 43},
        )
        self.assertEqual(plan["sampled"]["first_over_k"], 3)
        self.assertEqual(plan["logical"]["first_over_k"], 3)
        self.assertEqual(plan["direct_pool_rows"], 3)
        self.assertEqual(len(plan["resource_plan_sha256"]), 64)
        self.assertEqual(
            plan,
            freeze_taskwise_matches(
                task_id="task-1",
                treatment_outputs=treatment,
                direct_outputs=direct,
            ),
        )
        reversed_plan = freeze_taskwise_matches(
            task_id="task-1",
            treatment_outputs=treatment,
            direct_outputs=list(reversed(direct)),
        )
        self.assertNotEqual(plan, reversed_plan)


if __name__ == "__main__":
    unittest.main()
