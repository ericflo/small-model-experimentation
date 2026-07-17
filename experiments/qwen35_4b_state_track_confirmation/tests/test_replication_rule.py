import importlib.util
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module("stc_run_benchmark_rule", "run_benchmark.py")

SEEDS = BENCH.SEED_ORDER
PARENT = BENCH.FROZEN_PARENT
CANDIDATE = BENCH.CANDIDATE


def aggregates(deltas, parent=0.30):
    """Six events with a fixed parent aggregate and per-event candidate deltas."""
    assert len(deltas) == len(SEEDS)
    return {
        seed: {PARENT: parent, CANDIDATE: parent + deltas[index]}
        for index, seed in enumerate(SEEDS)
    }


class PairedRuleTruthTableTests(unittest.TestCase):
    def test_five_wins_positive_mean_is_confirmed(self) -> None:
        reading = BENCH.paired_reading(aggregates([0.02, 0.02, 0.02, 0.02, 0.02, -0.01]))
        self.assertEqual(reading["wins"], 5)
        self.assertGreater(reading["mean_delta"], 0)
        self.assertEqual(reading["verdict"], "CONFIRMED")
        self.assertIn("replicates across sealed seeds", reading["frozen_claim"])
        self.assertIn("program reference composite", reading["frozen_claim"])

    def test_exactly_four_wins_positive_mean_is_confirmed(self) -> None:
        # The threshold ceil(2*6/3) = 4 is inclusive.
        reading = BENCH.paired_reading(aggregates([0.03, 0.03, 0.03, 0.03, -0.01, -0.01]))
        self.assertEqual(reading["wins"], 4)
        self.assertGreater(reading["mean_delta"], 0)
        self.assertEqual(reading["verdict"], "CONFIRMED")

    def test_all_negative_is_not_confirmed(self) -> None:
        reading = BENCH.paired_reading(aggregates([-0.01] * 6))
        self.assertEqual(reading["wins"], 0)
        self.assertLess(reading["mean_delta"], 0)
        self.assertEqual(reading["verdict"], "NOT_CONFIRMED")
        self.assertIn("does not replicate", reading["frozen_claim"])
        self.assertIn("retired as seed noise", reading["frozen_claim"])

    def test_four_wins_but_negative_mean_is_not_confirmed(self) -> None:
        # The wins>=4-but-mean<=0 edge: mean<=0 dominates -> NOT_CONFIRMED.
        reading = BENCH.paired_reading(aggregates([0.01, 0.01, 0.01, 0.01, -0.05, -0.05]))
        self.assertEqual(reading["wins"], 4)
        self.assertLessEqual(reading["mean_delta"], 0)
        self.assertFalse(reading["mean_delta_strictly_positive"])
        self.assertEqual(reading["verdict"], "NOT_CONFIRMED")

    def test_five_wins_but_negative_mean_is_not_confirmed(self) -> None:
        reading = BENCH.paired_reading(aggregates([0.01, 0.01, 0.01, 0.01, 0.01, -0.20]))
        self.assertEqual(reading["wins"], 5)
        self.assertLess(reading["mean_delta"], 0)
        self.assertEqual(reading["verdict"], "NOT_CONFIRMED")

    def test_positive_mean_but_three_wins_is_ambiguous(self) -> None:
        reading = BENCH.paired_reading(aggregates([0.05, 0.05, 0.05, -0.01, -0.01, -0.01]))
        self.assertEqual(reading["wins"], 3)
        self.assertGreater(reading["mean_delta"], 0)
        self.assertEqual(reading["verdict"], "AMBIGUOUS")
        self.assertIn("mechanism-differentiated or larger-N", reading["frozen_claim"])

    def test_the_frozen_observed_effect_replicated_uniformly_is_confirmed(self) -> None:
        # Every event reproduces the 78169 lift exactly -> 6 wins, CONFIRMED.
        reading = BENCH.paired_reading(aggregates([0.02557] * 6))
        self.assertEqual(reading["wins"], 6)
        self.assertEqual(reading["verdict"], "CONFIRMED")


class TieGuardTests(unittest.TestCase):
    def test_within_guard_deltas_are_not_wins_and_close_not_confirmed(self) -> None:
        # A true rational tie rendered one ulp apart is a tie: not a win,
        # not a strictly-positive mean -> NOT_CONFIRMED.
        reading = BENCH.paired_reading(aggregates([5e-13] * 6))
        self.assertEqual(reading["wins"], 0)
        self.assertFalse(reading["mean_delta_strictly_positive"])
        self.assertEqual(reading["verdict"], "NOT_CONFIRMED")

    def test_delta_above_guard_counts_as_a_win(self) -> None:
        reading = BENCH.paired_reading(aggregates([2e-12] * 6))
        self.assertEqual(reading["wins"], 6)
        self.assertTrue(reading["mean_delta_strictly_positive"])
        self.assertEqual(reading["verdict"], "CONFIRMED")

    def test_guard_value_is_frozen_at_1e_12(self) -> None:
        self.assertEqual(BENCH.AGG_TIE_EPSILON, 1e-12)
        self.assertEqual(BENCH.WINS_THRESHOLD, 4)
        self.assertEqual(BENCH.EVENTS, 6)


class VerdictPartitionTests(unittest.TestCase):
    def test_partition_is_total_with_no_fourth_state(self) -> None:
        grid = (-0.05, -0.005, 0.0, 0.005, 0.05)
        # Sweep the last three per-event deltas; keep the first three fixed at
        # a small positive so the branch coverage is broad.
        for d4 in grid:
            for d5 in grid:
                for d6 in grid:
                    reading = BENCH.paired_reading(
                        aggregates([0.01, 0.01, 0.01, d4, d5, d6])
                    )
                    verdict = reading["verdict"]
                    self.assertIn(
                        verdict, {"CONFIRMED", "NOT_CONFIRMED", "AMBIGUOUS"}
                    )
                    mean_pos = reading["mean_delta_strictly_positive"]
                    wins = reading["wins"]
                    if mean_pos and wins >= BENCH.WINS_THRESHOLD:
                        self.assertEqual(verdict, "CONFIRMED")
                    elif not mean_pos:
                        self.assertEqual(verdict, "NOT_CONFIRMED")
                    else:
                        self.assertEqual(verdict, "AMBIGUOUS")

    def test_per_event_records_both_aggregates_and_delta(self) -> None:
        reading = BENCH.paired_reading(aggregates([0.02, -0.03, 0.02, 0.02, 0.02, 0.02]))
        first = str(SEEDS[0])
        second = str(SEEDS[1])
        self.assertAlmostEqual(reading["per_event"][first]["paired_delta"], 0.02)
        self.assertTrue(reading["per_event"][first]["candidate_wins"])
        self.assertAlmostEqual(reading["per_event"][second]["paired_delta"], -0.03)
        self.assertFalse(reading["per_event"][second]["candidate_wins"])
        self.assertEqual(reading["per_event"][first]["count_walk_aggregate"], 0.30)


class FinitenessAndShapeTests(unittest.TestCase):
    def test_invalid_aggregate_fails_closed(self) -> None:
        for bad in (float("nan"), float("inf"), -0.1, 1.1, True, None):
            table = aggregates([0.02] * 6)
            table[SEEDS[0]][CANDIDATE] = bad
            with self.assertRaises(ValueError):
                BENCH.paired_reading(table)

    def test_wrong_seed_set_fails_closed(self) -> None:
        table = aggregates([0.02] * 6)
        table[99999] = table.pop(SEEDS[0])
        with self.assertRaisesRegex(ValueError, "six frozen new events"):
            BENCH.paired_reading(table)

    def test_missing_arm_fails_closed(self) -> None:
        table = aggregates([0.02] * 6)
        del table[SEEDS[0]][PARENT]
        with self.assertRaisesRegex(ValueError, "both arms"):
            BENCH.paired_reading(table)

    def test_aggregate_strictly_above_uses_the_guard(self) -> None:
        self.assertFalse(BENCH.aggregate_strictly_above(0.30 + 5e-13, 0.30))
        self.assertTrue(BENCH.aggregate_strictly_above(0.30 + 2e-12, 0.30))
        self.assertFalse(BENCH.aggregate_strictly_above(0.30, 0.30))
        self.assertFalse(BENCH.aggregate_strictly_above(0.30, 0.31))


class RuleTextTests(unittest.TestCase):
    def test_rule_text_is_the_frozen_contract(self) -> None:
        self.assertIn("CONFIRMED iff mean_d strictly positive", BENCH.REPLICATION_RULE)
        self.assertIn("wins >= 4", BENCH.REPLICATION_RULE)
        self.assertIn("state_track_aggregate - count_walk_aggregate", BENCH.REPLICATION_RULE)
        self.assertIn("1e-12 tie guard", BENCH.REPLICATION_RULE)
        self.assertIn("dominates even", BENCH.REPLICATION_RULE)
        self.assertIn("never pooled", BENCH.REPLICATION_RULE)
        self.assertIn("No fourth state", BENCH.REPLICATION_RULE)
        self.assertEqual(
            set(BENCH.FROZEN_CLAIMS),
            {"CONFIRMED", "NOT_CONFIRMED", "AMBIGUOUS"},
        )

    def test_prior_event_is_never_pooled(self) -> None:
        reading = BENCH.paired_reading(aggregates([0.02] * 6))
        self.assertEqual(reading["events_counted"], list(SEEDS))
        self.assertNotIn(BENCH.PRIOR_EVENT["seed"], reading["events_counted"])
        self.assertFalse(reading["prior_event_pooled"])


if __name__ == "__main__":
    unittest.main()
