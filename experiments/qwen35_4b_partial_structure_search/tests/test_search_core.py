from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as F  # noqa: E402
import experiment_common as C  # noqa: E402
import search_core as S  # noqa: E402


class SearchCoreTests(unittest.TestCase):
    def test_expand_and_select_are_deterministic(self) -> None:
        candidates = S.expand([()])
        self.assertEqual(len(candidates), 16)
        scores = {prefix: float(index) for index, prefix in enumerate(candidates)}
        selected = S.select_beam(candidates, scores, 2)
        self.assertEqual(selected, [candidates[-1], candidates[-2]])

    def test_visible_fill_and_hidden_grade_are_separate(self) -> None:
        pipeline = (("add_k", 1),)
        task = F.build_task_from_pipeline(
            task_id="fill-task",
            seed=1,
            pipeline=pipeline,
            visible_inputs=[[1, 2]],
            label_probe_inputs=[[3]],
            hidden_inputs=[[4, 5]],
        )
        out = S.evaluate_leaves(task, [("add_k",)], fill_cap=16)
        self.assertTrue(out["pool_hidden_coverage"])
        self.assertTrue(out["selected_hidden_success"])
        self.assertGreater(out["fill_cap_used"], 0)
        self.assertGreaterEqual(out["evaluation_wall_seconds"], 0.0)

    def test_hash_seeded_leaf_order_is_deterministic_complete_and_task_specific(self) -> None:
        first = list(S.iter_hash_seeded_skeletons("task-a", 2))
        again = list(S.iter_hash_seeded_skeletons("task-a", 2))
        other = list(S.iter_hash_seeded_skeletons("task-b", 2))
        self.assertEqual(first, again)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 16**2)
        self.assertEqual(len(set(first)), 16**2)
        self.assertEqual(set(first), set(F.enumerate_skeletons(2)))

    def test_budget_truncated_brute_streams_complete_leaves_to_exact_fill_cap(self) -> None:
        task = F.build_task_from_pipeline(
            task_id="budget-brute",
            seed=2,
            pipeline=(("reverse", None), ("sort_asc", None)),
            visible_inputs=[[3, 1, 2]],
            label_probe_inputs=[[4, 2, 1, 3]],
            hidden_inputs=[[8, 6, 7]],
            require_exact_depth=False,
        )
        out = S.evaluate_budget_truncated_brute(task, fill_cap=11)
        self.assertEqual(out["planned_complete_skeletons"], 16**2)
        self.assertEqual(out["planned_concrete_parameterized_leaves"], 32**2)
        self.assertEqual(out["fill_cap_used"], 11)
        self.assertEqual(
            out["interpreter_accounting"]["parameter_fills_attempted"], 11
        )
        self.assertEqual(
            out["attempted_complete_skeletons"], len(out["leaves"])
        )
        self.assertLessEqual(out["attempted_complete_skeletons"], 11)
        self.assertFalse(out["enumeration_exhausted"])
        self.assertGreaterEqual(out["evaluation_wall_seconds"], 0.0)

        expected_leaves = [
            list(prefix)
            for prefix in list(S.iter_hash_seeded_skeletons("budget-brute", 2))[
                : out["attempted_complete_skeletons"]
            ]
        ]
        self.assertEqual(out["leaves"], expected_leaves)

    def test_budget_truncated_arm_records_task_wall_and_work_summary(self) -> None:
        task = F.build_task_from_pipeline(
            task_id="budget-arm",
            seed=3,
            pipeline=(("add_k", 1),),
            visible_inputs=[[1, 2]],
            label_probe_inputs=[[3, 4]],
            hidden_inputs=[[5, 6]],
        )
        result = S.run_budget_truncated_brute([task], fill_cap=7)
        row = result["rows"][0]
        self.assertIsNone(result["beam_width"])
        self.assertEqual(
            row["expanded_prefix_nodes"], row["attempted_complete_skeletons"]
        )
        self.assertEqual(result["summary"]["parameter_fills_attempted"], 7)
        self.assertEqual(
            result["summary"]["attempted_complete_skeletons"],
            row["attempted_complete_skeletons"],
        )
        self.assertGreaterEqual(row["task_wall_seconds"], row["evaluation_wall_seconds"])
        self.assertGreaterEqual(result["wall_seconds"], row["task_wall_seconds"])

    def test_surface_prior_uses_depth_pooled_backoffs_at_unseen_depth(self) -> None:
        rows = [
            {
                "candidate_prefix": ["reverse", "sort_asc"],
                "live": True,
            },
            {
                "candidate_prefix": ["reverse", "sort_desc"],
                "live": False,
            },
            {
                "candidate_prefix": ["negate", "sort_asc"],
                "live": True,
            },
            {
                "candidate_prefix": ["negate", "sort_desc"],
                "live": False,
            },
        ]
        score = S.fit_surface_prior(rows)
        # No length-5 observations exist. The pooled child prior must still
        # distinguish the terminal operations instead of returning 0.5 for all.
        asc = score(("add_k", "mul_k", "mod_k", "square", "sort_asc"))
        desc = score(("add_k", "mul_k", "mod_k", "square", "sort_desc"))
        self.assertGreater(asc, desc)
        self.assertNotEqual(asc, 0.5)

        # When the preceding operation was observed, the pooled parent-child
        # relation is also available across unseen prefix lengths.
        parent_asc = score(("add_k", "mul_k", "mod_k", "reverse", "sort_asc"))
        parent_desc = score(("add_k", "mul_k", "mod_k", "reverse", "sort_desc"))
        self.assertGreater(parent_asc, parent_desc)


if __name__ == "__main__":
    unittest.main()
