"""Consequence-partition boundaries and the frozen readings.

The ordered total partition: ZERO_ROOT_COMPARABLE iff the zero-root
aggregate strictly beats base AND zero-root strict wins >= original
strict wins on this seed minus one; ZERO_ROOT_DEGRADED otherwise. Exact
ties are never strict wins. Every input must land in exactly one cell.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))

import check_benchmark as cb  # noqa: E402


def family_table(value: float, **overrides: float) -> dict[str, float]:
    table = {family: value for family in cb.FAMILIES}
    table.update(overrides)
    return table


def scores(base_agg, base_fam, orig_agg, orig_fam, zero_agg, zero_fam) -> dict:
    return {
        "base": {"aggregate": base_agg, "per_family": base_fam},
        cb.ORIGINAL_ARM: {"aggregate": orig_agg, "per_family": orig_fam},
        cb.ZERO_ROOT_ARM: {"aggregate": zero_agg, "per_family": zero_fam},
    }


def wins_table(base: dict[str, float], wins: int) -> dict[str, float]:
    """A treated table that strictly beats base on exactly ``wins`` families."""
    table = dict(base)
    for family in cb.FAMILIES[:wins]:
        table[family] = base[family] + 0.1
    return table


class TestGoalGateRow(unittest.TestCase):
    def test_ties_are_never_strict_wins(self):
        base = family_table(0.5)
        treated = family_table(0.5, menders=0.6)
        row = cb.goal_gate_row(base, treated)
        self.assertEqual(row["strict_wins"], 1)
        self.assertEqual(row["wins"], ["menders"])
        self.assertEqual(len(row["ties"]), 9)
        self.assertFalse(row["goal_gate_pass"])

    def test_pass_requires_all_ten_strict_wins(self):
        base = family_table(0.1)
        row = cb.goal_gate_row(base, wins_table(base, 10))
        self.assertTrue(row["goal_gate_pass"])
        self.assertEqual(row["strict_wins"], 10)
        row9 = cb.goal_gate_row(base, wins_table(base, 9))
        self.assertFalse(row9["goal_gate_pass"])


class TestConsequencePartition(unittest.TestCase):
    def _consequence(self, orig_wins: int, zero_wins: int, zero_agg: float,
                     base_agg: float = 0.10) -> dict:
        base = family_table(0.1)
        return cb.consequence_reading(
            scores(
                base_agg, base,
                0.35, wins_table(base, orig_wins),
                zero_agg, wins_table(base, zero_wins),
            )
        )

    def test_comparable_when_equal_wins_and_aggregate_beats_base(self):
        reading = self._consequence(orig_wins=10, zero_wins=10, zero_agg=0.33)
        self.assertEqual(reading["consequence"], "ZERO_ROOT_COMPARABLE")
        self.assertEqual(reading["required_strict_wins"], 9)
        self.assertEqual(
            reading["statement"],
            cb.CONSEQUENCE_STATEMENTS["ZERO_ROOT_COMPARABLE"],
        )

    def test_comparable_at_exactly_original_minus_one(self):
        reading = self._consequence(orig_wins=10, zero_wins=9, zero_agg=0.33)
        self.assertEqual(reading["consequence"], "ZERO_ROOT_COMPARABLE")
        self.assertTrue(reading["strict_wins_bar_met"])

    def test_degraded_at_original_minus_two(self):
        reading = self._consequence(orig_wins=10, zero_wins=8, zero_agg=0.33)
        self.assertEqual(reading["consequence"], "ZERO_ROOT_DEGRADED")
        self.assertFalse(reading["strict_wins_bar_met"])
        self.assertEqual(
            reading["statement"],
            cb.CONSEQUENCE_STATEMENTS["ZERO_ROOT_DEGRADED"],
        )

    def test_aggregate_tie_is_degraded_even_with_ten_wins(self):
        reading = self._consequence(
            orig_wins=10, zero_wins=10, zero_agg=0.10, base_agg=0.10
        )
        self.assertFalse(reading["zero_root_aggregate_beats_base"])
        self.assertEqual(reading["consequence"], "ZERO_ROOT_DEGRADED")

    def test_aggregate_loss_is_degraded(self):
        reading = self._consequence(
            orig_wins=10, zero_wins=10, zero_agg=0.05, base_agg=0.10
        )
        self.assertEqual(reading["consequence"], "ZERO_ROOT_DEGRADED")

    def test_zero_original_wins_lowers_the_bar_to_minus_one(self):
        # Degenerate edge: original wins 0 -> bar is -1, trivially met; the
        # consequence then rides entirely on the aggregate condition.
        reading = self._consequence(orig_wins=0, zero_wins=0, zero_agg=0.33)
        self.assertEqual(reading["required_strict_wins"], -1)
        self.assertEqual(reading["consequence"], "ZERO_ROOT_COMPARABLE")

    def test_partition_is_total_and_ordered(self):
        for orig in (0, 5, 10):
            for zero in (0, 4, 9, 10):
                for agg in (0.05, 0.10, 0.33):
                    reading = self._consequence(orig, zero, agg)
                    self.assertIn(reading["consequence"], cb.CONSEQUENCES)
                    expected = (
                        "ZERO_ROOT_COMPARABLE"
                        if agg > 0.10 and zero >= orig - 1
                        else "ZERO_ROOT_DEGRADED"
                    )
                    self.assertEqual(reading["consequence"], expected)


class TestReadings(unittest.TestCase):
    def setUp(self):
        base = family_table(0.1, menders=0.2, rites=0.3, warren=0.1)
        original = family_table(0.4, menders=0.25, rites=0.35, warren=0.2)
        zero = family_table(0.3, menders=0.22, rites=0.30, warren=0.25)
        self.scores = scores(0.10, base, 0.35, original, 0.30, zero)

    def test_prefix_contribution_is_zero_root_minus_original(self):
        reading = cb.prefix_contribution_reading(self.scores)
        self.assertEqual(reading["framing"], cb.PREFIX_CONTRIBUTION_FRAMING)
        self.assertAlmostEqual(reading["aggregate"], -0.05)
        self.assertAlmostEqual(reading["per_family"]["menders"], 0.22 - 0.25)
        self.assertAlmostEqual(reading["per_family"]["warren"], 0.25 - 0.2)
        self.assertEqual(set(reading["per_family"]), set(cb.FAMILIES))

    def test_margins_reading_covers_menders_rites_warren_for_both_arms(self):
        reading = cb.margins_reading(self.scores)
        self.assertEqual(reading["families"], ["menders", "rites", "warren"])
        self.assertIn("no statechain stage", reading["statechain_note"])
        self.assertAlmostEqual(
            reading["per_arm"][cb.ORIGINAL_ARM]["menders"], 0.05
        )
        self.assertAlmostEqual(
            reading["per_arm"][cb.ZERO_ROOT_ARM]["rites"], 0.0
        )
        self.assertAlmostEqual(
            reading["per_arm"][cb.ZERO_ROOT_ARM]["warren"], 0.15
        )

    def test_per_family_reading_has_all_three_arms(self):
        reading = cb.per_family_reading(self.scores)
        self.assertEqual(set(reading["aggregates"]), set(cb.MODEL_ORDER))
        self.assertEqual(set(reading["per_family"]), set(cb.MODEL_ORDER))

    def test_goal_gate_reading_covers_both_composites(self):
        reading = cb.goal_gate_reading(self.scores)
        self.assertEqual(set(reading["per_arm"]), set(cb.COMPOSITE_ARMS))


class TestBudgetIntegrity(unittest.TestCase):
    def _budget(self, over: str | None) -> dict:
        return {
            label: {
                "within_budget": label != over,
                "wall_seconds": 100.0,
            }
            for label in cb.MODEL_ORDER
        }

    def test_all_within_budget_is_valid(self):
        reading = cb.budget_integrity_reading(self._budget(None))
        self.assertTrue(reading["paired_comparison_valid"])
        self.assertIsNone(reading["reason"])

    def test_any_over_budget_arm_invalidates_the_paired_comparison(self):
        reading = cb.budget_integrity_reading(self._budget(cb.ZERO_ROOT_ARM))
        self.assertFalse(reading["paired_comparison_valid"])
        self.assertIn(cb.ZERO_ROOT_ARM, reading["reason"])


class TestBuildReadout(unittest.TestCase):
    def _inputs(self):
        base = family_table(0.1)
        s = scores(
            0.10, base, 0.35, wins_table(base, 10), 0.33, wins_table(base, 10)
        )
        budget = {
            label: {"within_budget": True, "wall_seconds": 10.0}
            for label in cb.MODEL_ORDER
        }
        receipts = {
            label: {"path": f"receipt/{label}.json", "sha256": "0" * 64}
            for label in cb.MODEL_ORDER
        }
        anchor = {
            "sha256": "1" * 64,
            "tree_sha256": "2" * 64,
            "weights_sha256": "3" * 64,
        }
        return s, budget, receipts, anchor

    def test_readout_assembles_with_the_reference_implementation(self):
        s, budget, receipts, anchor = self._inputs()
        readout = cb.build_readout(
            s, budget, dict(cb.REFERENCE_IMPLEMENTATION), receipts, anchor,
            "4" * 64, "5" * 64,
        )
        self.assertEqual(readout["consequence"], "ZERO_ROOT_COMPARABLE")
        self.assertEqual(readout["outcome"], "ZERO_ROOT_READ_COMPLETE")
        self.assertIsNone(readout["promoted"])
        self.assertFalse(readout["benchmark_data_read"])
        self.assertEqual(
            set(readout["readings"]),
            {
                "per_family", "goal_gate", "prefix_contribution",
                "budget_integrity", "margins", "consequence",
            },
        )
        self.assertEqual(readout["seed"], 78159)

    def test_readout_refuses_a_drifted_implementation_signature(self):
        s, budget, receipts, anchor = self._inputs()
        drifted = dict(cb.REFERENCE_IMPLEMENTATION, source_file_count=57)
        with self.assertRaises(ValueError):
            cb.build_readout(s, budget, drifted, receipts, anchor, "4" * 64, "5" * 64)


if __name__ == "__main__":
    unittest.main()
