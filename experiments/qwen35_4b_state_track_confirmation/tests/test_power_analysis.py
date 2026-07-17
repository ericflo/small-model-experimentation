import importlib.util
import sys
import unittest
from fractions import Fraction
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


POWER = load_module("stc_power_analysis", "power_analysis.py")


class PowerArithmeticTests(unittest.TestCase):
    def test_preregistered_numbers_recompute(self) -> None:
        self.assertTrue(POWER._matches(POWER.computed()))

    def test_null_marginals_are_exact_closed_forms(self) -> None:
        self.assertEqual(POWER.P_WINS_GE4_NULL, Fraction(22, 64))
        self.assertEqual(float(POWER.P_WINS_GE4_NULL), 0.34375)
        self.assertEqual(POWER.P_MEAN_GT0_NULL, Fraction(1, 2))
        self.assertEqual(POWER.PREREGISTERED["p_wins_ge4_null"], 0.3438)
        self.assertEqual(POWER.PREREGISTERED["p_mean_gt0_null"], 0.5)

    def test_null_false_positive_lies_between_the_two_bounds(self) -> None:
        alpha = POWER.PREREGISTERED["p_false_confirmed_null"]
        self.assertGreater(alpha, POWER.JOINT_INDEP_BOUND)  # 0.17188
        self.assertLess(alpha, POWER.JOINT_MARGINAL_BOUND)  # 0.34375
        self.assertAlmostEqual(alpha, 0.311, places=3)

    def test_null_joint_is_scale_free(self) -> None:
        # Under mu=0 the joint does not depend on sigma_d (scale invariance).
        a = POWER.joint_confirmed(0.0, 0.02)
        b = POWER.joint_confirmed(0.0, 0.05)
        self.assertAlmostEqual(a, b, places=4)

    def test_power_is_monotone_in_sigma_and_high(self) -> None:
        power = POWER.PREREGISTERED["power_confirmed"]
        self.assertGreater(power["0.02"], power["0.025"])
        self.assertGreater(power["0.025"], power["0.03"])
        # Even at the widest sigma the confirmation retains > 0.75 power.
        self.assertGreater(power["0.03"], 0.75)

    def test_confirmed_is_bounded_by_its_two_gates(self) -> None:
        for key in ("0.02", "0.025", "0.03"):
            self.assertLessEqual(
                POWER.PREREGISTERED["power_confirmed"][key],
                POWER.PREREGISTERED["power_wins_ge4"][key] + POWER.CHECK_TOL,
            )
            self.assertLessEqual(
                POWER.PREREGISTERED["power_confirmed"][key],
                POWER.PREREGISTERED["power_mean_gt0"][key] + POWER.CHECK_TOL,
            )

    def test_three_outcomes_partition_the_alternative(self) -> None:
        for key in ("0.02", "0.025", "0.03"):
            total = (
                POWER.PREREGISTERED["power_confirmed"][key]
                + POWER.PREREGISTERED["p_ambiguous_effect"][key]
                + POWER.PREREGISTERED["p_not_confirmed_effect"][key]
            )
            self.assertAlmostEqual(total, 1.0, places=2)

    def test_sigma_d_arithmetic_is_recorded(self) -> None:
        self.assertEqual(POWER.SIGMA_ARM, 0.03)
        self.assertEqual(POWER.SIGMA_D_HEADLINE, 0.025)
        self.assertEqual(POWER.SIGMA_D_RANGE, (0.020, 0.025, 0.030))
        # sigma_d = sigma_arm * sqrt(2 (1 - rho)) => rho in (0.5, 0.78).
        self.assertAlmostEqual(POWER.RHO_BY_SIGMA_D[0.030], 0.5, places=3)
        self.assertGreater(POWER.RHO_BY_SIGMA_D[0.020], POWER.RHO_BY_SIGMA_D[0.030])

    def test_mu_effect_matches_the_prior_lift(self) -> None:
        self.assertEqual(POWER.MU_EFFECT, 0.0256)
        self.assertEqual(POWER.EVENTS, 6)
        self.assertEqual(POWER.WINS_THRESHOLD, 4)


if __name__ == "__main__":
    unittest.main()
