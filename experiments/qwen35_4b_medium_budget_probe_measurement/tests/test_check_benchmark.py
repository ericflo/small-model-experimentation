"""Unit tests for the preregistered budget-probe readings (synthetic receipts).

Everything here runs on synthetic score tables: no model, no gateway, no
files. Covers the SCOPED budget-movement booleans (moved fires only for
pairs that were exactly zero at tb1024 and are above zero at tb8192; a
status-quo repeat of designed_fresh's already-nonzero rites can never
fire them), the seed-confounded budget contrast against a synthetic
tb1024 table including the fail-closed benchmark-implementation equality
condition, the forensics-style strict-win/goal-gate counting, the
budget-integrity scoping (paired_comparison_valid flips on any
over-budget arm while scores stay recorded), and the readout schema.
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
        # menders and rites both at zero: no movement for base.
        "base": {
            "aggregate": 0.08,
            "per_family": per_family(0.1, menders=0.0, rites=0.0),
        },
        # Strictly above base on every family, menders AND rites both
        # positive: the recorded goal gate passes and both families moved.
        "designed_fresh": {
            "aggregate": 0.40,
            "per_family": per_family(0.3, menders=0.2, rites=0.15),
        },
        # menders and rites stay at zero (exact ties with base): eight
        # strict wins, no pass, no movement.
        "replay_repeat": {
            "aggregate": 0.50,
            "per_family": per_family(0.4, menders=0.0, rites=0.0),
        },
        # menders barely positive, rites positive, one strict loss
        # (sirens): nine strict wins, no pass, movement on both probes.
        "hygiene_explore": {
            "aggregate": 0.30,
            "per_family": per_family(0.2, menders=0.05, sirens=0.05),
        },
    }


def synthetic_reference_scores() -> dict[str, dict]:
    # Mirrors the real premise shape of the pinned tb1024 seed-78150 event:
    # menders 0.0 for all four arms; rites 0.0 everywhere EXCEPT
    # designed_fresh (0.1).
    return {
        "base": {
            "aggregate": 0.05,
            "per_family": per_family(0.05, menders=0.0, rites=0.0),
        },
        "designed_fresh": {
            "aggregate": 0.32,
            "per_family": per_family(0.25, menders=0.0, rites=0.1),
        },
        "replay_repeat": {
            "aggregate": 0.30,
            "per_family": per_family(0.35, menders=0.0, rites=0.0),
        },
        "hygiene_explore": {
            "aggregate": 0.34,
            "per_family": per_family(0.15, menders=0.0, rites=0.0, sirens=0.2),
        },
    }


def reference_per_family() -> dict[str, dict[str, float]]:
    return {
        label: row["per_family"]
        for label, row in synthetic_reference_scores().items()
    }


def synthetic_implementation() -> dict:
    return {
        "runner_sha256": "a" * 64,
        "source_inventory_sha256": "b" * 64,
        "source_file_count": 56,
    }


def synthetic_budget(**overrides: dict) -> dict[str, dict]:
    budget = {
        label: {"within_budget": True, "wall_seconds": 100.0 + index}
        for index, label in enumerate(cb.MODEL_ORDER)
    }
    for label, override in overrides.items():
        if label not in budget:
            raise AssertionError(f"unknown arm: {label}")
        budget[label].update(override)
    return budget


class TestBudgetMovement(unittest.TestCase):
    def setUp(self) -> None:
        scores = synthetic_scores()
        self.per_family = {
            label: row["per_family"] for label, row in scores.items()
        }
        self.reference = reference_per_family()
        self.reading = cb.budget_movement(self.per_family, self.reference)

    def test_families_probed_are_menders_and_rites(self) -> None:
        self.assertEqual(self.reading["families_probed"], ["menders", "rites"])
        self.assertEqual(cb.BUDGET_FAMILIES, ("menders", "rites"))
        self.assertIn("movement_rule", self.reading)
        self.assertIn("0.1 for designed_fresh", self.reading["premise"])

    def test_all_four_arms_are_read_including_base(self) -> None:
        self.assertEqual(set(self.reading["per_arm"]), set(cb.MODEL_ORDER))

    def test_zero_is_not_movement(self) -> None:
        for label in ("base", "replay_repeat"):
            row = self.reading["per_arm"][label]
            self.assertFalse(row["menders_moved"], label)
            self.assertFalse(row["rites_moved"], label)
            self.assertFalse(row["either_moved"], label)

    def test_genuine_zero_to_positive_flip_fires(self) -> None:
        row = self.reading["per_arm"]["designed_fresh"]
        self.assertTrue(row["menders_moved"])  # 0.0 -> 0.2
        tiny = self.reading["per_arm"]["hygiene_explore"]
        self.assertEqual(tiny["menders"], 0.05)
        self.assertTrue(tiny["menders_moved"])  # 0.0 -> 0.05
        self.assertTrue(tiny["rites_moved"])  # 0.0 -> 0.2

    def test_already_nonzero_pair_never_fires_even_when_it_rises(self) -> None:
        # designed_fresh rites: tb1024 0.1 -> tb8192 0.15. A rise, but the
        # pair was already nonzero at tb1024, so it is excluded from the
        # booleans (only menders keeps either_moved true here).
        row = self.reading["per_arm"]["designed_fresh"]
        self.assertEqual(row["rites"], 0.15)
        self.assertEqual(row["rites_tb1024"], 0.1)
        self.assertFalse(row["rites_moved"])
        self.assertTrue(row["either_moved"])

    def test_status_quo_repeat_fires_nothing(self) -> None:
        # MAJOR-1 regression: if every arm merely repeats its tb1024
        # per-family values (designed_fresh rites stays 0.1 > 0), NO
        # movement boolean may fire and any_arm_moved must be false.
        repeat = {
            label: dict(values) for label, values in self.reference.items()
        }
        reading = cb.budget_movement(repeat, self.reference)
        for label, row in reading["per_arm"].items():
            self.assertFalse(row["menders_moved"], label)
            self.assertFalse(row["rites_moved"], label)
            self.assertFalse(row["either_moved"], label)
        self.assertFalse(reading["any_arm_moved"])
        self.assertEqual(
            reading["already_nonzero_at_tb1024"],
            [
                {
                    "arm": "designed_fresh",
                    "family": "rites",
                    "tb1024": 0.1,
                    "tb8192": 0.1,
                }
            ],
        )

    def test_already_nonzero_descriptive_field(self) -> None:
        self.assertEqual(
            self.reading["already_nonzero_at_tb1024"],
            [
                {
                    "arm": "designed_fresh",
                    "family": "rites",
                    "tb1024": 0.1,
                    "tb8192": 0.15,
                }
            ],
        )

    def test_tb1024_values_recorded_per_arm(self) -> None:
        for label in cb.MODEL_ORDER:
            row = self.reading["per_arm"][label]
            self.assertEqual(row["menders_tb1024"], self.reference[label]["menders"])
            self.assertEqual(row["rites_tb1024"], self.reference[label]["rites"])

    def test_any_arm_moved_aggregates_over_arms(self) -> None:
        self.assertTrue(self.reading["any_arm_moved"])
        flat = {
            label: per_family(0.3, menders=0.0, rites=0.0)
            for label in cb.MODEL_ORDER
        }
        self.assertFalse(
            cb.budget_movement(flat, self.reference)["any_arm_moved"]
        )

    def test_full_per_family_table_carried_verbatim(self) -> None:
        self.assertEqual(self.reading["per_family"], self.per_family)


class TestBudgetContrast(unittest.TestCase):
    def setUp(self) -> None:
        self.reading = cb.budget_contrast(
            synthetic_scores(),
            synthetic_reference_scores(),
            synthetic_implementation(),
            synthetic_implementation(),
        )

    def test_block_is_labeled_cross_seed_confound(self) -> None:
        self.assertIs(self.reading["cross_seed_confound"], True)
        self.assertIn("not a causal isolation", self.reading["note"])
        self.assertIn("implementation equality", self.reading["note"])

    def test_matching_signature_is_surfaced_on_both_sides(self) -> None:
        block = self.reading["benchmark_implementation"]
        self.assertEqual(block["tb8192"], synthetic_implementation())
        self.assertEqual(block["tb1024"], synthetic_implementation())
        self.assertIs(block["identical"], True)

    def test_implementation_mismatch_aborts_fail_closed(self) -> None:
        for drift in (
            {"runner_sha256": "c" * 64},
            {"source_inventory_sha256": "d" * 64},
            {"source_file_count": 57},
        ):
            with self.assertRaises(ValueError, msg=str(drift)):
                cb.budget_contrast(
                    synthetic_scores(),
                    synthetic_reference_scores(),
                    {**synthetic_implementation(), **drift},
                    synthetic_implementation(),
                )

    def test_equality_guard_is_exact(self) -> None:
        cb.require_implementation_equality(
            synthetic_implementation(), synthetic_implementation()
        )
        with self.assertRaises(ValueError):
            cb.require_implementation_equality(
                synthetic_implementation(),
                {**synthetic_implementation(), "source_file_count": 55},
            )

    def test_reference_pin_carried(self) -> None:
        self.assertEqual(self.reading["reference"], cb.TB1024_REFERENCE)
        self.assertEqual(self.reading["reference"]["seed"], 78150)
        self.assertEqual(self.reading["reference"]["think_budget"], 1024)
        self.assertIs(self.reading["reference"]["cross_seed_confound"], True)

    def test_per_family_deltas(self) -> None:
        row = self.reading["per_arm"]["designed_fresh"]
        self.assertAlmostEqual(row["per_family_delta"]["menders"], 0.2)
        self.assertAlmostEqual(row["per_family_delta"]["rites"], 0.05)
        self.assertAlmostEqual(row["per_family_delta"]["chronicle"], 0.05)
        self.assertEqual(set(row["per_family_delta"]), set(cb.FAMILIES))

    def test_negative_deltas_preserved(self) -> None:
        row = self.reading["per_arm"]["hygiene_explore"]
        self.assertAlmostEqual(row["per_family_delta"]["sirens"], -0.15)
        self.assertAlmostEqual(row["aggregate_delta"], -0.04)

    def test_aggregates_recorded_on_both_sides(self) -> None:
        row = self.reading["per_arm"]["base"]
        self.assertEqual(row["aggregate_tb8192"], 0.08)
        self.assertEqual(row["aggregate_tb1024"], 0.05)
        self.assertAlmostEqual(row["aggregate_delta"], 0.03)
        self.assertEqual(set(self.reading["per_arm"]), set(cb.MODEL_ORDER))


class TestGoalGateCounting(unittest.TestCase):
    def setUp(self) -> None:
        scores = synthetic_scores()
        self.table = cb.goal_gate_table(
            {label: row["per_family"] for label, row in scores.items()}
        )

    def test_only_treated_arms_are_gated(self) -> None:
        self.assertEqual(set(self.table), set(cb.TREATED_ARMS))

    def test_all_ten_strict_wins_pass(self) -> None:
        row = self.table["designed_fresh"]
        self.assertEqual(row["strict_wins"], 10)
        self.assertEqual(row["wins"], list(cb.FAMILIES))
        self.assertEqual(row["losses"], [])
        self.assertEqual(row["ties"], [])
        self.assertTrue(row["goal_gate_pass"])

    def test_a_tie_is_not_a_strict_win(self) -> None:
        row = self.table["replay_repeat"]
        self.assertEqual(row["strict_wins"], 8)
        self.assertEqual(row["ties"], ["menders", "rites"])
        self.assertEqual(row["losses"], [])
        self.assertFalse(row["goal_gate_pass"])

    def test_losses_and_ties_recorded_separately(self) -> None:
        row = self.table["hygiene_explore"]
        self.assertEqual(row["strict_wins"], 9)
        self.assertEqual(row["losses"], ["sirens"])
        self.assertEqual(row["ties"], [])
        self.assertFalse(row["goal_gate_pass"])


class TestBudgetIntegrity(unittest.TestCase):
    def test_all_within_budget_is_valid(self) -> None:
        reading = cb.budget_integrity(synthetic_budget())
        self.assertTrue(reading["all_within_budget"])
        self.assertTrue(reading["paired_comparison_valid"])
        self.assertIsNone(reading["reason"])
        self.assertEqual(set(reading["per_arm"]), set(cb.MODEL_ORDER))
        self.assertEqual(reading["per_arm"]["base"]["wall_seconds"], 100.0)

    def test_any_over_budget_arm_invalidates_the_paired_comparison(self) -> None:
        reading = cb.budget_integrity(
            synthetic_budget(replay_repeat={"within_budget": False})
        )
        self.assertFalse(reading["all_within_budget"])
        self.assertFalse(reading["paired_comparison_valid"])
        self.assertIn("replay_repeat", reading["reason"])
        self.assertIn("scores recorded", reading["reason"])
        # The flag never drops the arm's record.
        self.assertIs(reading["per_arm"]["replay_repeat"]["within_budget"], False)

    def test_every_over_budget_arm_is_named(self) -> None:
        reading = cb.budget_integrity(
            synthetic_budget(
                base={"within_budget": False},
                hygiene_explore={"within_budget": False},
            )
        )
        self.assertFalse(reading["paired_comparison_valid"])
        self.assertIn("base", reading["reason"])
        self.assertIn("hygiene_explore", reading["reason"])

    def test_wall_seconds_carried_verbatim(self) -> None:
        budget = synthetic_budget(designed_fresh={"wall_seconds": 5400.5})
        reading = cb.budget_integrity(budget)
        self.assertEqual(
            reading["per_arm"]["designed_fresh"]["wall_seconds"], 5400.5
        )


class TestReadoutSchema(unittest.TestCase):
    def build(
        self, budget: dict[str, dict], implementation: dict | None = None
    ) -> dict:
        return cb.build_readout(
            synthetic_scores(),
            budget,
            implementation or synthetic_implementation(),
            synthetic_reference_scores(),
            synthetic_implementation(),
            {
                label: {"path": f"runs/benchmark/x/{label}.json", "sha256": "0" * 64}
                for label in cb.MODEL_ORDER
            },
            "f" * 64,
        )

    def setUp(self) -> None:
        self.readout = self.build(synthetic_budget())

    def test_top_level_schema(self) -> None:
        self.assertEqual(
            set(self.readout),
            {
                "schema_version", "experiment_id", "stage", "name", "tier",
                "think_budget", "seed", "benchmark_data_read", "promoted",
                "outcome", "paired_comparison_valid", "design_receipt_sha256",
                "tb1024_reference", "receipts", "scores", "budget", "readings",
            },
        )
        self.assertEqual(self.readout["schema_version"], 1)
        self.assertEqual(self.readout["stage"], "medium_budget_probe_readout")
        self.assertEqual(self.readout["tier"], "medium")
        self.assertEqual(self.readout["think_budget"], 8192)
        self.assertEqual(self.readout["seed"], 78152)

    def test_measurement_intake_never_promotes(self) -> None:
        self.assertIsNone(self.readout["promoted"])
        self.assertIs(self.readout["benchmark_data_read"], False)
        self.assertEqual(self.readout["outcome"], "MEASUREMENT_READ_COMPLETE")

    def test_readings_are_exactly_the_four_preregistered(self) -> None:
        self.assertEqual(
            set(self.readout["readings"]),
            {
                "budget_movement", "budget_contrast", "goal_gate",
                "budget_integrity",
            },
        )
        gate = self.readout["readings"]["goal_gate"]
        self.assertEqual(set(gate), set(cb.TREATED_ARMS))
        self.assertTrue(gate["designed_fresh"]["goal_gate_pass"])
        movement = self.readout["readings"]["budget_movement"]
        self.assertTrue(movement["any_arm_moved"])
        contrast = self.readout["readings"]["budget_contrast"]
        self.assertIs(contrast["cross_seed_confound"], True)

    def test_within_budget_event_is_a_valid_paired_comparison(self) -> None:
        self.assertIs(self.readout["paired_comparison_valid"], True)
        self.assertIsNone(self.readout["readings"]["budget_integrity"]["reason"])

    def test_over_budget_arm_scopes_but_keeps_scores(self) -> None:
        readout = self.build(
            synthetic_budget(hygiene_explore={"within_budget": False})
        )
        self.assertIs(readout["paired_comparison_valid"], False)
        integrity = readout["readings"]["budget_integrity"]
        self.assertIs(integrity["paired_comparison_valid"], False)
        self.assertIn("hygiene_explore", integrity["reason"])
        # Scores and the other readings are still fully recorded.
        self.assertEqual(readout["scores"], synthetic_scores())
        self.assertEqual(
            set(readout["readings"]["budget_movement"]["per_arm"]),
            set(cb.MODEL_ORDER),
        )

    def test_scores_and_budget_carried_verbatim(self) -> None:
        self.assertEqual(self.readout["scores"], synthetic_scores())
        self.assertEqual(self.readout["budget"], synthetic_budget())
        self.assertEqual(self.readout["tb1024_reference"], cb.TB1024_REFERENCE)

    def test_implementation_mismatch_aborts_the_whole_readout(self) -> None:
        with self.assertRaises(ValueError):
            self.build(
                synthetic_budget(),
                implementation={
                    **synthetic_implementation(),
                    "runner_sha256": "e" * 64,
                },
            )


if __name__ == "__main__":
    unittest.main()
