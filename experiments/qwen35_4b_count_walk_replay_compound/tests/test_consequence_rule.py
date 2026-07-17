"""The frozen two-directional consequence rule, driven over its truth table.

COMPOUNDED iff candidate aggregate strictly > parent aggregate AND no family
strictly below the parent by more than 0.1 (frozen comparison
``candidate_family >= parent_family - 0.1 - 1e-9``) AND candidate aggregate
strictly > base aggregate. BOUNDED otherwise; no third state. The per-family
boundary is exercised integer-exactly on both score lattices the families
live on (k/10 and k/60). Strictly-above on the aggregates carries the
frozen 1e-12 tie guard: distinct per-family multisets with exactly equal
RATIONAL aggregates can float-render one ulp apart, and a true rational tie
must resolve BOUNDED, never COMPOUNDED.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import run_benchmark as rb  # noqa: E402


FAMILIES = sorted(rb.PUBLIC_FAMILIES)


def make_event(aggregate: float, per_family: dict[str, float] | None = None) -> dict:
    families = {family: 0.2 for family in FAMILIES}
    if per_family:
        families.update(per_family)
    return {"aggregate": aggregate, "per_family": families}


def make_events(
    base_aggregate: float = 0.08,
    parent_aggregate: float = 0.33,
    candidate_aggregate: float = 0.36,
    parent_families: dict[str, float] | None = None,
    candidate_families: dict[str, float] | None = None,
) -> dict[str, dict]:
    return {
        "base": make_event(base_aggregate, {family: 0.0 for family in FAMILIES}),
        "count_walk": make_event(parent_aggregate, parent_families),
        "replay_compound": make_event(candidate_aggregate, candidate_families),
    }


class TestConsequenceRule(unittest.TestCase):
    def reading(self, events: dict[str, dict]) -> dict:
        return rb.consequence_reading(events, "replay_compound")

    def test_clean_win_is_compounded_with_the_frozen_claim(self):
        reading = self.reading(make_events())
        self.assertEqual(reading["verdict"], "COMPOUNDED")
        self.assertEqual(reading["frozen_claim"], rb.COMPOUNDED_CLAIM)
        self.assertTrue(reading["no_third_state"])

    def test_aggregate_tie_with_parent_is_bounded(self):
        reading = self.reading(
            make_events(parent_aggregate=0.33, candidate_aggregate=0.33)
        )
        self.assertEqual(reading["verdict"], "BOUNDED")
        self.assertEqual(reading["frozen_claim"], rb.BOUNDED_CLAIM)
        self.assertFalse(reading["aggregate_strictly_beats_parent"])

    def test_aggregate_below_parent_is_bounded(self):
        reading = self.reading(
            make_events(parent_aggregate=0.33, candidate_aggregate=0.32)
        )
        self.assertEqual(reading["verdict"], "BOUNDED")

    def test_aggregate_tie_with_base_is_bounded(self):
        reading = self.reading(
            make_events(base_aggregate=0.36, candidate_aggregate=0.36)
        )
        self.assertEqual(reading["verdict"], "BOUNDED")
        self.assertFalse(reading["aggregate_strictly_beats_base"])

    def test_family_below_parent_by_exactly_point_one_is_still_compounded(self):
        reading = self.reading(
            make_events(
                parent_families={"menders": 0.3},
                candidate_families={"menders": 0.2},
            )
        )
        self.assertEqual(reading["verdict"], "COMPOUNDED")
        self.assertTrue(reading["family_table"]["menders"]["within_slack"])
        self.assertEqual(reading["families_below_slack"], [])

    def test_family_below_parent_by_a_hair_more_than_point_one_is_bounded(self):
        reading = self.reading(
            make_events(
                parent_families={"menders": 0.30000001},
                candidate_families={"menders": 0.2},
            )
        )
        self.assertEqual(reading["verdict"], "BOUNDED")
        self.assertFalse(reading["family_table"]["menders"]["within_slack"])
        self.assertEqual(reading["families_below_slack"], ["menders"])

    def test_two_families_each_within_slack_still_compound(self):
        reading = self.reading(
            make_events(
                parent_families={"menders": 0.3, "rites": 0.1},
                candidate_families={"menders": 0.2, "rites": 0.0},
            )
        )
        self.assertEqual(reading["verdict"], "COMPOUNDED")

    def test_k10_lattice_boundary_is_exact_everywhere(self):
        # One full episode below the parent passes at every k/10 lattice
        # point; two below fails at every point.
        for tenths in range(1, 11):
            parent_value = tenths / 10
            with self.subTest(parent=parent_value, delta="one_episode"):
                self.assertTrue(
                    rb.family_within_slack((tenths - 1) / 10, parent_value)
                )
            if tenths >= 2:
                with self.subTest(parent=parent_value, delta="two_episodes"):
                    self.assertFalse(
                        rb.family_within_slack((tenths - 2) / 10, parent_value)
                    )

    def test_k60_lattice_boundary_is_exact_everywhere(self):
        # 0.1 is six steps on the k/60 lattice: exactly six below passes,
        # seven below fails, at every lattice point.
        for k in range(6, 61):
            parent_value = k / 60
            with self.subTest(parent=parent_value, delta="six_steps"):
                self.assertTrue(rb.family_within_slack((k - 6) / 60, parent_value))
            if k >= 7:
                with self.subTest(parent=parent_value, delta="seven_steps"):
                    self.assertFalse(
                        rb.family_within_slack((k - 7) / 60, parent_value)
                    )

    def test_family_above_parent_is_always_within_slack(self):
        self.assertTrue(rb.family_within_slack(0.9, 0.1))
        self.assertTrue(rb.family_within_slack(0.1, 0.1))

    def test_goal_gate_reading_is_descriptive_and_counts_strict_wins(self):
        events = make_events(
            candidate_families={family: 0.2 for family in FAMILIES}
        )
        events["replay_compound"]["per_family"]["menders"] = 0.0
        reading = rb.goal_gate_reading(events)
        self.assertFalse(reading["included_in_consequence"])
        candidate = reading["per_arm"]["replay_compound"]
        self.assertEqual(candidate["strict_wins"], 9)
        self.assertFalse(candidate["goal_gate_pass"])
        self.assertIn("menders", candidate["ties"])
        parent = reading["per_arm"]["count_walk"]
        self.assertEqual(parent["strict_wins"], 10)
        self.assertTrue(parent["goal_gate_pass"])

    def test_frozen_claim_texts_are_the_preregistered_sentences(self):
        self.assertIn("replay compounding holds at stage 8", rb.COMPOUNDED_CLAIM)
        self.assertIn("raised-floor confirmation", rb.COMPOUNDED_CLAIM)
        self.assertIn("diminishing returns at stage 8", rb.BOUNDED_CLAIM)
        self.assertIn("different move class", rb.BOUNDED_CLAIM)

    def test_slack_constants_are_frozen(self):
        self.assertEqual(rb.PER_FAMILY_SLACK, 0.1)
        self.assertEqual(rb.SLACK_EPSILON, 1e-9)
        self.assertEqual(rb.AGG_TIE_EPSILON, 1e-12)


class TestAggregateTieGuard(unittest.TestCase):
    """The 1e-12 aggregate tie guard on the gateway-reported floats.

    Demonstrated failure mode (pre-guard): two DISTINCT per-family multisets
    with exactly equal rational aggregates (both sum to 4.6, mean 0.46)
    float-render one ulp apart (0.45999999999999996 vs 0.46000000000000008),
    flipping BOUNDED to COMPOUNDED under a bare ``>``. Python 3.12's sum()
    is Neumaier-compensated, so summation order is not the mechanism and
    math.fsum does not fix it; the explicit tie epsilon does.
    """

    # The demonstrated 1-ulp flip pair: exact rational aggregate 46/100 both.
    PARENT_FAMILIES = [1.0, 0.1, 0.6, 0.8, 0.1, 0.0, 0.1, 1.0, 0.8, 0.1]
    CANDIDATE_FAMILIES = [0.9, 0.2, 0.6, 0.8, 0.1, 0.0, 0.1, 1.0, 0.8, 0.1]

    def paired_events(self, parent_values, candidate_values):
        parent_families = dict(zip(FAMILIES, parent_values))
        candidate_families = dict(zip(FAMILIES, candidate_values))
        return make_events(
            base_aggregate=0.08,
            parent_aggregate=sum(parent_values) / 10,
            candidate_aggregate=sum(candidate_values) / 10,
            parent_families=parent_families,
            candidate_families=candidate_families,
        )

    def test_the_flip_pair_renders_one_ulp_apart(self):
        parent_aggregate = sum(self.PARENT_FAMILIES) / 10
        candidate_aggregate = sum(self.CANDIDATE_FAMILIES) / 10
        # The rational aggregates are exactly equal; the floats are not.
        self.assertNotEqual(parent_aggregate, candidate_aggregate)
        self.assertGreater(candidate_aggregate, parent_aggregate)
        self.assertLess(abs(candidate_aggregate - parent_aggregate), 1e-15)

    def test_true_rational_tie_rendered_one_ulp_above_is_bounded(self):
        # Pre-guard this read COMPOUNDED (candidate renders 1 ulp above the
        # parent); the tie guard must read the true rational tie as BOUNDED.
        reading = rb.consequence_reading(
            self.paired_events(self.PARENT_FAMILIES, self.CANDIDATE_FAMILIES),
            "replay_compound",
        )
        self.assertFalse(reading["aggregate_strictly_beats_parent"])
        self.assertEqual(reading["verdict"], "BOUNDED")

    def test_true_rational_tie_rendered_one_ulp_below_is_bounded(self):
        # The mirrored pair: the candidate renders 1 ulp BELOW the parent;
        # still a tie, still BOUNDED.
        reading = rb.consequence_reading(
            self.paired_events(self.CANDIDATE_FAMILIES, self.PARENT_FAMILIES),
            "replay_compound",
        )
        self.assertFalse(reading["aggregate_strictly_beats_parent"])
        self.assertEqual(reading["verdict"], "BOUNDED")

    def test_true_rational_tie_with_base_is_bounded(self):
        events = self.paired_events(self.PARENT_FAMILIES, self.CANDIDATE_FAMILIES)
        # Make the parent easily beaten and move the rendered-ulp tie to the
        # candidate-vs-base clause instead.
        events["base"]["aggregate"] = sum(self.PARENT_FAMILIES) / 10
        events["count_walk"]["aggregate"] = 0.1
        reading = rb.consequence_reading(events, "replay_compound")
        self.assertTrue(reading["aggregate_strictly_beats_parent"])
        self.assertFalse(reading["aggregate_strictly_beats_base"])
        self.assertEqual(reading["verdict"], "BOUNDED")

    def test_genuine_small_aggregate_win_remains_compounded(self):
        # A genuine +0.002 aggregate win (well above the 1e-12 guard, well
        # below one lattice step) must remain COMPOUNDED.
        reading = self.reading_for_delta(0.002)
        self.assertTrue(reading["aggregate_strictly_beats_parent"])
        self.assertEqual(reading["verdict"], "COMPOUNDED")

    def test_smallest_real_lattice_difference_clears_the_guard(self):
        # The smallest real aggregate difference (~1.7e-3, one k/60 step on
        # one family over ten families) is nine orders above the epsilon.
        reading = self.reading_for_delta(1 / 60 / 10)
        self.assertTrue(reading["aggregate_strictly_beats_parent"])
        self.assertEqual(reading["verdict"], "COMPOUNDED")

    def reading_for_delta(self, delta: float) -> dict:
        return rb.consequence_reading(
            make_events(
                parent_aggregate=0.33,
                candidate_aggregate=0.33 + delta,
            ),
            "replay_compound",
        )

    def test_aggregate_strictly_above_truth_table(self):
        self.assertFalse(rb.aggregate_strictly_above(0.46, 0.46))
        self.assertFalse(
            rb.aggregate_strictly_above(0.46000000000000008, 0.45999999999999996)
        )
        self.assertFalse(rb.aggregate_strictly_above(0.33, 0.33 + 1e-13))
        self.assertFalse(rb.aggregate_strictly_above(0.33 + 1e-13, 0.33))
        self.assertFalse(rb.aggregate_strictly_above(0.32, 0.33))
        self.assertTrue(rb.aggregate_strictly_above(0.33 + 1.7e-3, 0.33))

    def test_reading_records_the_tie_epsilon(self):
        reading = rb.consequence_reading(make_events(), "replay_compound")
        self.assertEqual(reading["aggregate_tie_epsilon"], 1e-12)


if __name__ == "__main__":
    unittest.main()
