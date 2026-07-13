from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import plans  # noqa: E402
import stats  # noqa: E402


def _completion(sampled: int, stage1: int, stage2: int) -> dict[str, int]:
    return {
        "n_sampled_tokens": sampled,
        "n_stage1_prompt_tokens": stage1,
        "n_stage2_prompt_tokens": stage2,
    }


class PlanTests(unittest.TestCase):
    def test_logical_cost_does_not_double_count_injected_close(self) -> None:
        self.assertEqual(
            plans.completion_cost(_completion(5, 10, 20)),
            {"sampled_tokens": 5, "logical_model_tokens": 35},
        )

    def test_conservative_first_over_is_taskwise_and_inclusive(self) -> None:
        outputs = [_completion(4, 5, 0), _completion(7, 5, 0), _completion(10, 5, 0)]
        match = plans.conservative_first_over(outputs, target=8, metric="sampled_tokens")
        self.assertEqual(match["first_over_k"], 2)
        self.assertEqual(match["first_over_cost"], 11)
        self.assertEqual(match["under_k"], 1)


class StatsTests(unittest.TestCase):
    def test_exact_one_sided_mcnemar_orientation(self) -> None:
        result = stats.one_sided_mcnemar([1, 1, 1, 0], [0, 0, 1, 0])
        self.assertEqual((result["b"], result["c"]), (2, 0))
        self.assertEqual(result["p_value"], 0.25)

    def test_holm_stops_at_first_failure(self) -> None:
        result = stats.holm({"b": 0.03, "a": 0.01, "c": 0.04}, alpha=0.05)
        self.assertTrue(result["decisions"]["a"])
        self.assertFalse(result["decisions"]["b"])
        self.assertFalse(result["decisions"]["c"])

    def test_stratified_bootstrap_is_deterministic(self) -> None:
        treatment = [1] * 48
        control = [0] * 48
        blocks = [0] * 24 + [1] * 24
        first = stats.stratified_bootstrap_lower(
            treatment, control, blocks, seed=7, resamples=100
        )
        second = stats.stratified_bootstrap_lower(
            treatment, control, blocks, seed=7, resamples=100
        )
        self.assertEqual(first, 1.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
