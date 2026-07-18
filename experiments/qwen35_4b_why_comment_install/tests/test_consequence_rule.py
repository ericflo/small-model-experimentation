"""The frozen, TIGHTENED two-directional transfer consequence, over its truth table.

INSTALLED_CODING iff the why_comment composite beats base by >= 3 PROBLEMS on
>= 1 target dataset AND does not regress the other past the 0.02 retention
tolerance. RETENTION_FAIL iff it regresses EITHER dataset past 0.02 (priority).
NULL iff no >= 3-problem gain and no regression past tolerance. No third state.

The >= 3-problem bar is the fix for bet #1's letter-of-the-law false positive: a
+1-problem HumanEval bump near ceiling must read NULL, not INSTALLED. Identical
rule to the self-repair cell (bet #2); this cell reuses it verbatim.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import measure_transfer as mt  # noqa: E402

# Base coding baselines (shared fitness harness, from bet #1's measurement).
BASE_HE_PASS, BASE_HE_TOTAL = 125, 164   # 0.7622
BASE_MBPP_PASS, BASE_MBPP_TOTAL = 113, 200  # 0.5650


def reading(treat_he_pass, treat_mbpp_pass):
    return mt.consequence_reading(
        BASE_HE_PASS, BASE_HE_TOTAL, BASE_MBPP_PASS, BASE_MBPP_TOTAL,
        treat_he_pass, treat_mbpp_pass,
    )


class TestTightenedConsequenceRule(unittest.TestCase):
    def test_three_problem_gain_on_humaneval_is_installed(self):
        r = reading(BASE_HE_PASS + 3, BASE_MBPP_PASS)  # +3 HE, MBPP flat
        self.assertEqual(r["verdict"], "INSTALLED_CODING")
        self.assertEqual(r["frozen_claim"], mt.INSTALLED_CLAIM)
        self.assertTrue(r["humaneval_meaningful_gain"])
        self.assertTrue(r["mbpp_retention_holds"])

    def test_two_problem_gain_is_null_not_installed(self):
        # Below the tightened bar: a +2-problem bump does NOT count as installed.
        r = reading(BASE_HE_PASS + 2, BASE_MBPP_PASS)
        self.assertEqual(r["verdict"], "NULL")
        self.assertFalse(r["any_meaningful_gain"])

    def test_one_problem_gain_is_null_the_bet1_technicality(self):
        # Exactly bet #1's letter-of-the-law false positive: now NULL.
        r = reading(BASE_HE_PASS + 1, BASE_MBPP_PASS)
        self.assertEqual(r["verdict"], "NULL")
        self.assertEqual(r["frozen_claim"], mt.NULL_CLAIM)

    def test_exactly_three_problems_is_the_boundary(self):
        self.assertEqual(reading(BASE_HE_PASS + 3, BASE_MBPP_PASS)["verdict"], "INSTALLED_CODING")
        self.assertEqual(reading(BASE_HE_PASS + 2, BASE_MBPP_PASS)["verdict"], "NULL")

    def test_meaningful_gain_on_mbpp_with_humaneval_flat_is_installed(self):
        r = reading(BASE_HE_PASS, BASE_MBPP_PASS + 5)  # +5 MBPP
        self.assertEqual(r["verdict"], "INSTALLED_CODING")
        self.assertTrue(r["mbpp_meaningful_gain"])

    def test_regression_past_tol_on_mbpp_is_retention_fail(self):
        r = reading(BASE_HE_PASS, BASE_MBPP_PASS - 5)  # -5/200 = -0.025 > 0.02
        self.assertEqual(r["verdict"], "RETENTION_FAIL")
        self.assertEqual(r["frozen_claim"], mt.RETENTION_FAIL_CLAIM)
        self.assertTrue(r["mbpp_regressed_past_tol"])

    def test_mbpp_drop_exactly_two_percent_holds_retention(self):
        # -4/200 = -0.02 exactly -> retention holds; no gain -> NULL.
        r = reading(BASE_HE_PASS, BASE_MBPP_PASS - 4)
        self.assertEqual(r["verdict"], "NULL")
        self.assertFalse(r["mbpp_regressed_past_tol"])

    def test_gain_on_one_but_big_regression_on_other_is_retention_fail_priority(self):
        r = reading(BASE_HE_PASS + 5, BASE_MBPP_PASS - 13)  # HE +5 but MBPP collapses
        self.assertEqual(r["verdict"], "RETENTION_FAIL")
        self.assertTrue(r["humaneval_meaningful_gain"])
        self.assertTrue(r["any_regressed_past_tol"])

    def test_no_change_is_null(self):
        r = reading(BASE_HE_PASS, BASE_MBPP_PASS)
        self.assertEqual(r["verdict"], "NULL")
        self.assertEqual(r["frozen_claim"], mt.NULL_CLAIM)

    def test_helpers_are_exact(self):
        self.assertTrue(mt.meaningful_gain(3))
        self.assertFalse(mt.meaningful_gain(2))
        self.assertFalse(mt.regressed(-0.02))
        self.assertTrue(mt.regressed(-0.02 - 1e-6))
        self.assertFalse(mt.regressed(0.0))
        self.assertFalse(mt.regressed(0.5))

    def test_no_third_state_over_a_grid(self):
        for dhe in (-8, -4, -3, -1, 0, 1, 2, 3, 6):
            for dmb in (-8, -4, -3, -1, 0, 1, 2, 3, 6):
                r = reading(BASE_HE_PASS + dhe, BASE_MBPP_PASS + dmb)
                self.assertIn(r["verdict"], ("INSTALLED_CODING", "RETENTION_FAIL", "NULL"))
                self.assertTrue(r["no_third_state"])
                if r["any_regressed_past_tol"]:
                    self.assertEqual(r["verdict"], "RETENTION_FAIL")
                elif r["any_meaningful_gain"]:
                    self.assertEqual(r["verdict"], "INSTALLED_CODING")
                else:
                    self.assertEqual(r["verdict"], "NULL")

    def test_records_all_four_numbers(self):
        r = reading(130, 120)
        self.assertEqual(r["base"]["humaneval"]["passed"], 125)
        self.assertEqual(r["base"]["mbpp"]["passed"], 113)
        self.assertEqual(r["treatment"]["humaneval"]["passed"], 130)
        self.assertEqual(r["treatment"]["mbpp"]["passed"], 120)
        self.assertEqual(r["problem_delta"], {"humaneval": 5, "mbpp": 7})


class TestPairedDeltas(unittest.TestCase):
    def test_mcnemar_counts(self):
        base = [
            {"task_id": "a", "passed": True},
            {"task_id": "b", "passed": True},
            {"task_id": "c", "passed": False},
            {"task_id": "d", "passed": False},
        ]
        treat = [
            {"task_id": "a", "passed": True},   # both
            {"task_id": "b", "passed": False},  # only base (McNemar b, a regression)
            {"task_id": "c", "passed": True},   # only treatment (McNemar c, a gain)
            {"task_id": "d", "passed": False},  # neither
        ]
        out = mt.paired_deltas(base, treat)
        self.assertEqual(out["both_pass"], 1)
        self.assertEqual(out["only_base_passes"], 1)
        self.assertEqual(out["only_treatment_passes"], 1)
        self.assertEqual(out["neither_passes"], 1)
        self.assertEqual(out["net_treatment_minus_base"], 0)
        self.assertEqual(out["n_paired"], 4)


if __name__ == "__main__":
    unittest.main()
