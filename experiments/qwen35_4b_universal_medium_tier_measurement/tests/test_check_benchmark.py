"""Unit tests for the preregistered reading computations (synthetic receipts).

Everything here runs on synthetic score tables: no model, no gateway, no
files. Covers the forensics-style strict-win/goal-gate counting, the base
sanity envelope (inclusive bounds), the ordering comparison against the
frozen quick reference (hygiene_explore null at quick), the blocking-family
derivation, and the readout schema.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_benchmark as cb  # noqa: E402


def per_family(value: float, **overrides: float) -> dict[str, float]:
    values = {family: value for family in cb.FAMILIES}
    for family, override in overrides.items():
        if family not in values:
            raise AssertionError(f"unknown family: {family}")
        values[family] = override
    return values


def synthetic_scores() -> dict[str, dict]:
    return {
        "base": {"aggregate": 0.10, "per_family": per_family(0.1)},
        # Strictly above base on every family: the recorded goal gate passes.
        "designed_fresh": {"aggregate": 0.40, "per_family": per_family(0.3)},
        # One exact tie: nine strict wins, no pass.
        "replay_repeat": {
            "aggregate": 0.50,
            "per_family": per_family(0.4, menders=0.1),
        },
        # One strict loss and one tie: eight strict wins, no pass.
        "hygiene_explore": {
            "aggregate": 0.30,
            "per_family": per_family(0.2, sirens=0.05, warren=0.1),
        },
    }


def synthetic_profile() -> dict:
    return {
        "events": 95,
        "families": {family: {"min": 0.0, "max": 0.7} for family in cb.FAMILIES},
    }


class TestGoalGateCounting(unittest.TestCase):
    def setUp(self) -> None:
        scores = synthetic_scores()
        self.table = cb.goal_gate_table(
            {label: row["per_family"] for label, row in scores.items()}
        )

    def test_all_ten_strict_wins_pass(self) -> None:
        row = self.table["designed_fresh"]
        self.assertEqual(row["strict_wins"], 10)
        self.assertEqual(row["wins"], list(cb.FAMILIES))
        self.assertEqual(row["losses"], [])
        self.assertEqual(row["ties"], [])
        self.assertTrue(row["goal_gate_pass"])

    def test_a_tie_is_not_a_strict_win(self) -> None:
        row = self.table["replay_repeat"]
        self.assertEqual(row["strict_wins"], 9)
        self.assertEqual(row["ties"], ["menders"])
        self.assertEqual(row["losses"], [])
        self.assertFalse(row["goal_gate_pass"])

    def test_losses_and_ties_recorded_separately(self) -> None:
        row = self.table["hygiene_explore"]
        self.assertEqual(row["strict_wins"], 8)
        self.assertEqual(row["losses"], ["sirens"])
        self.assertEqual(row["ties"], ["warren"])
        self.assertFalse(row["goal_gate_pass"])

    def test_blocking_families_are_the_not_strictly_won(self) -> None:
        blocking = cb.blocking_families(self.table)
        self.assertEqual(blocking["designed_fresh"], [])
        self.assertEqual(blocking["replay_repeat"], ["menders"])
        self.assertEqual(blocking["hygiene_explore"], ["sirens", "warren"])


class TestBaseEnvelope(unittest.TestCase):
    def test_inclusive_bounds(self) -> None:
        profile = synthetic_profile()
        profile["families"]["chronicle"] = {"min": 0.1, "max": 0.7}
        reading = cb.base_envelope(per_family(0.1), profile)
        self.assertTrue(reading["families"]["chronicle"]["inside"])
        self.assertTrue(reading["all_inside"])
        self.assertEqual(reading["historical_base_events"], 95)

    def test_below_min_is_outside(self) -> None:
        profile = synthetic_profile()
        profile["families"]["sirens"] = {"min": 0.2, "max": 0.6}
        reading = cb.base_envelope(per_family(0.1), profile)
        self.assertFalse(reading["families"]["sirens"]["inside"])
        self.assertFalse(reading["all_inside"])

    def test_above_max_is_outside(self) -> None:
        profile = synthetic_profile()
        profile["families"]["menders"] = {"min": 0.0, "max": 0.3}
        reading = cb.base_envelope(per_family(0.1, menders=0.4), profile)
        self.assertFalse(reading["families"]["menders"]["inside"])
        self.assertFalse(reading["all_inside"])


class TestOrderingReading(unittest.TestCase):
    def test_matching_quick_ordering(self) -> None:
        reading = cb.ordering_reading(
            {
                "base": 0.10,
                "designed_fresh": 0.40,
                "replay_repeat": 0.50,
                "hygiene_explore": 0.30,
            }
        )
        self.assertEqual(
            reading["medium_ranking"],
            ["replay_repeat", "designed_fresh", "hygiene_explore", "base"],
        )
        self.assertEqual(
            reading["quick_ranking_measured_arms"],
            ["replay_repeat", "designed_fresh", "base"],
        )
        self.assertTrue(reading["ordering_matches_quick_on_measured_arms"])
        self.assertTrue(reading["medium_strictly_ordered"])
        self.assertIsNone(reading["quick_aggregates"]["hygiene_explore"])

    def test_inverted_ordering_detected(self) -> None:
        reading = cb.ordering_reading(
            {
                "base": 0.10,
                "designed_fresh": 0.50,
                "replay_repeat": 0.40,
                "hygiene_explore": 0.60,
            }
        )
        self.assertEqual(
            reading["medium_ranking_quick_measured_arms"],
            ["designed_fresh", "replay_repeat", "base"],
        )
        self.assertFalse(reading["ordering_matches_quick_on_measured_arms"])

    def test_hygiene_position_does_not_affect_the_match(self) -> None:
        reading = cb.ordering_reading(
            {
                "base": 0.10,
                "designed_fresh": 0.40,
                "replay_repeat": 0.50,
                "hygiene_explore": 0.99,
            }
        )
        self.assertTrue(reading["ordering_matches_quick_on_measured_arms"])

    def test_exact_ties_are_flagged_not_strictly_ordered(self) -> None:
        reading = cb.ordering_reading(
            {
                "base": 0.10,
                "designed_fresh": 0.40,
                "replay_repeat": 0.40,
                "hygiene_explore": 0.30,
            }
        )
        self.assertFalse(reading["medium_strictly_ordered"])


class TestReadoutSchema(unittest.TestCase):
    def setUp(self) -> None:
        self.readout = cb.build_readout(
            synthetic_scores(),
            synthetic_profile(),
            {
                label: {"path": f"runs/benchmark/x/{label}.json", "sha256": "0" * 64}
                for label in cb.MODEL_ORDER
            },
            "f" * 64,
        )

    def test_top_level_schema(self) -> None:
        self.assertEqual(
            set(self.readout),
            {
                "schema_version", "experiment_id", "stage", "name", "tier",
                "think_budget", "seed", "benchmark_data_read", "promoted",
                "outcome", "design_receipt_sha256", "forensics_analysis_sha256",
                "quick_reference", "receipts", "scores", "readings",
            },
        )
        self.assertEqual(self.readout["schema_version"], 1)
        self.assertEqual(self.readout["stage"], "medium_tier_measurement_readout")
        self.assertEqual(self.readout["tier"], "medium")
        self.assertEqual(self.readout["think_budget"], 1024)
        self.assertEqual(self.readout["seed"], 78150)

    def test_measurement_intake_never_promotes(self) -> None:
        self.assertIsNone(self.readout["promoted"])
        self.assertIs(self.readout["benchmark_data_read"], False)
        self.assertEqual(self.readout["outcome"], "MEASUREMENT_READ_COMPLETE")

    def test_readings_are_exactly_the_four_preregistered(self) -> None:
        self.assertEqual(
            set(self.readout["readings"]),
            {
                "aggregate_ordering", "goal_gate", "base_sanity_envelope",
                "blocking_families",
            },
        )
        gate = self.readout["readings"]["goal_gate"]
        self.assertEqual(set(gate), set(cb.TREATED_ARMS))
        self.assertTrue(gate["designed_fresh"]["goal_gate_pass"])
        ordering = self.readout["readings"]["aggregate_ordering"]
        self.assertTrue(ordering["ordering_matches_quick_on_measured_arms"])
        envelope = self.readout["readings"]["base_sanity_envelope"]
        self.assertTrue(envelope["all_inside"])

    def test_scores_carried_verbatim(self) -> None:
        self.assertEqual(self.readout["scores"], synthetic_scores())


if __name__ == "__main__":
    unittest.main()
