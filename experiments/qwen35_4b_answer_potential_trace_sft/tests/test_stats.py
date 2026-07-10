from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from stats import kendall_tau_b, paired_bootstrap, quantile, roc_auc  # noqa: E402


class StatisticsTests(unittest.TestCase):
    def test_roc_auc_handles_perfect_reverse_ties_and_single_class(self) -> None:
        labels = [False, False, True, True]
        self.assertEqual(roc_auc(labels, [0, 1, 2, 3]), 1.0)
        self.assertEqual(roc_auc(labels, [3, 2, 1, 0]), 0.0)
        self.assertEqual(roc_auc(labels, [1, 1, 1, 1]), 0.5)
        self.assertIsNone(roc_auc([True, True], [0, 1]))

    def test_kendall_tau_b(self) -> None:
        self.assertEqual(kendall_tau_b([1, 2, 3], [1, 2, 3]), 1.0)
        self.assertEqual(kendall_tau_b([1, 2, 3], [3, 2, 1]), -1.0)
        tied = kendall_tau_b([1, 1, 2], [1, 2, 3])
        self.assertIsNotNone(tied)
        self.assertGreater(tied, 0.0)

    def test_paired_bootstrap_is_deterministic(self) -> None:
        pairs = {f"t{i}": (1.0, 0.0) for i in range(8)}
        first = paired_bootstrap(pairs, resamples=100, seed=7)
        second = paired_bootstrap(pairs, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["mean_delta"], 1.0)
        self.assertEqual(first["ci95_low"], 1.0)

    def test_quantile_interpolates(self) -> None:
        self.assertTrue(math.isclose(quantile([0, 10], 0.25), 2.5))


if __name__ == "__main__":
    unittest.main()
