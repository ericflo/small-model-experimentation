from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "stats.py"
SPEC = importlib.util.spec_from_file_location("partial_search_stats", MODULE)
stats = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(stats)


class StatisticsTests(unittest.TestCase):
    def test_auroc_ties_and_perfect_order(self) -> None:
        self.assertEqual(stats.auroc([0.0, 1.0], [0, 1]), 1.0)
        self.assertEqual(stats.auroc([1.0, 0.0], [0, 1]), 0.0)
        self.assertEqual(stats.auroc([0.5, 0.5], [0, 1]), 0.5)
        self.assertIsNone(stats.auroc([0.0, 1.0], [1, 1]))

    def test_macro_group_auroc_ignores_single_class_groups(self) -> None:
        rows = [
            {"task_id": "a", "prefix_len": 1, "score": 0.0, "live": 0},
            {"task_id": "a", "prefix_len": 1, "score": 1.0, "live": 1},
            {"task_id": "b", "prefix_len": 1, "score": 0.2, "live": 0},
            {"task_id": "b", "prefix_len": 1, "score": 0.3, "live": 0},
        ]
        value, groups = stats.macro_group_auroc(rows, "score")
        self.assertEqual(value, 1.0)
        self.assertEqual(groups, 1)

    def test_cluster_bootstrap_and_mcnemar(self) -> None:
        out = stats.cluster_bootstrap(
            {"a": 1.0, "b": 0.0}, lambda xs: sum(xs) / len(xs), reps=100, seed=3
        )
        self.assertEqual(out["estimate"], 0.5)
        self.assertEqual(out["n_tasks"], 2)
        exact = stats.mcnemar_exact([True, True, False], [False, True, False])
        self.assertEqual(exact["a_only"], 1)
        self.assertEqual(exact["b_only"], 0)
        self.assertEqual(exact["p_two_sided"], 1.0)


if __name__ == "__main__":
    unittest.main()
