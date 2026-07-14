from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from stats import (  # noqa: E402
    one_sided_mcnemar,
    paired_bootstrap_interval,
    paired_report,
)


class StatsTests(unittest.TestCase):
    def test_exact_one_sided_mcnemar_known_values(self) -> None:
        self.assertEqual(one_sided_mcnemar(0, 0), 1.0)
        self.assertEqual(one_sided_mcnemar(3, 0), 0.125)
        self.assertEqual(one_sided_mcnemar(0, 3), 1.0)
        self.assertEqual(one_sided_mcnemar(2, 2), 11 / 16)

    def test_paired_bootstrap_is_deterministic_and_paired(self) -> None:
        treatment = [True, True, False, True, False, True]
        control = [False, True, False, False, True, True]
        first = paired_bootstrap_interval(
            treatment, control, seed=43, resamples=1_000
        )
        second = paired_bootstrap_interval(
            treatment, control, seed=43, resamples=1_000
        )
        self.assertEqual(first, second)
        report = paired_report(treatment, control, seed=43, resamples=1_000)
        self.assertEqual(report["effect"], 1 / 6)
        self.assertEqual(report["treatment_only"], 2)
        self.assertEqual(report["control_only"], 1)
        self.assertTrue(report["report_only"])


if __name__ == "__main__":
    unittest.main()
