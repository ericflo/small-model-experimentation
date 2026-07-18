"""The frozen two-directional transfer consequence, over its truth table.

INSTALLED_CODING iff the exec_trace composite strictly beats base on >=1 target
dataset AND does not regress the other past the 0.02 retention tolerance.
RETENTION_FAIL iff it regresses EITHER dataset past 0.02 (priority — the
forgetting risk realized). NULL iff no strict improvement and no regression.
No third state.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import measure_transfer as mt  # noqa: E402

BASE_HE = 0.7622
BASE_MBPP = 0.5650


def reading(treat_he, treat_mbpp, base_he=BASE_HE, base_mbpp=BASE_MBPP):
    return mt.consequence_reading(base_he, base_mbpp, treat_he, treat_mbpp)


class TestConsequenceRule(unittest.TestCase):
    def test_strict_gain_on_humaneval_with_retention_is_installed(self):
        r = reading(BASE_HE + 0.03, BASE_MBPP)  # HE up, MBPP flat
        self.assertEqual(r["verdict"], "INSTALLED_CODING")
        self.assertEqual(r["frozen_claim"], mt.INSTALLED_CLAIM)
        self.assertTrue(r["humaneval_improved"])
        self.assertTrue(r["mbpp_retention_holds"])

    def test_strict_gain_on_mbpp_with_small_humaneval_dip_within_tol_is_installed(self):
        r = reading(BASE_HE - 0.02, BASE_MBPP + 0.04)  # HE dips exactly 0.02 (allowed)
        self.assertEqual(r["verdict"], "INSTALLED_CODING")
        self.assertFalse(r["humaneval_regressed_past_tol"])
        self.assertTrue(r["mbpp_improved"])

    def test_regression_past_tol_on_mbpp_is_retention_fail(self):
        r = reading(BASE_HE, BASE_MBPP - 0.05)
        self.assertEqual(r["verdict"], "RETENTION_FAIL")
        self.assertEqual(r["frozen_claim"], mt.RETENTION_FAIL_CLAIM)
        self.assertTrue(r["mbpp_regressed_past_tol"])

    def test_gain_on_one_but_big_regression_on_other_is_retention_fail_priority(self):
        # HE strictly improves, but MBPP collapses past tolerance -> RETENTION_FAIL wins.
        r = reading(BASE_HE + 0.05, BASE_MBPP - 0.10)
        self.assertEqual(r["verdict"], "RETENTION_FAIL")
        self.assertTrue(r["humaneval_improved"])
        self.assertTrue(r["any_regressed_past_tol"])

    def test_no_change_is_null(self):
        r = reading(BASE_HE, BASE_MBPP)
        self.assertEqual(r["verdict"], "NULL")
        self.assertEqual(r["frozen_claim"], mt.NULL_CLAIM)
        self.assertFalse(r["any_improved"])
        self.assertFalse(r["any_regressed_past_tol"])

    def test_both_dip_within_tol_no_gain_is_null(self):
        r = reading(BASE_HE - 0.01, BASE_MBPP - 0.02)  # within tol, no gain
        self.assertEqual(r["verdict"], "NULL")

    def test_retention_boundary_is_exact(self):
        # A drop of exactly 0.02 holds retention; a hair past it fails.
        self.assertFalse(mt.regressed(-0.02))
        self.assertTrue(mt.regressed(-0.02 - 1e-6))
        self.assertFalse(mt.regressed(-0.019))
        self.assertFalse(mt.regressed(0.0))
        self.assertFalse(mt.regressed(0.5))

    def test_improve_is_strict(self):
        self.assertFalse(mt.improved(0.0))
        self.assertTrue(mt.improved(1e-6))
        self.assertFalse(mt.improved(-1e-6))

    def test_no_third_state_over_a_grid(self):
        deltas = (-0.20, -0.05, -0.02, -0.01, 0.0, 0.01, 0.05)
        for dh in deltas:
            for dm in deltas:
                r = reading(BASE_HE + dh, BASE_MBPP + dm)
                self.assertIn(r["verdict"], ("INSTALLED_CODING", "RETENTION_FAIL", "NULL"))
                self.assertTrue(r["no_third_state"])
                # totality is consistent with the priority rule
                if r["any_regressed_past_tol"]:
                    self.assertEqual(r["verdict"], "RETENTION_FAIL")
                elif r["any_improved"]:
                    self.assertEqual(r["verdict"], "INSTALLED_CODING")
                else:
                    self.assertEqual(r["verdict"], "NULL")

    def test_records_all_four_numbers(self):
        r = reading(0.80, 0.60)
        self.assertEqual(r["base"], {"humaneval": BASE_HE, "mbpp": BASE_MBPP})
        self.assertEqual(r["treatment"], {"humaneval": 0.80, "mbpp": 0.60})


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
            {"task_id": "b", "passed": False},  # only base (regression, McNemar b)
            {"task_id": "c", "passed": True},   # only treatment (gain, McNemar c)
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
