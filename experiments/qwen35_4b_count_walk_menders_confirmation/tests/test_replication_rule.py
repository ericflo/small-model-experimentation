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


BENCH = load_module("cwmc_run_benchmark_rule", "run_benchmark.py")

SEEDS = BENCH.SEED_ORDER
ARMS = BENCH.MODEL_ORDER
CANDIDATE = BENCH.CANDIDATE
CONTROLS = BENCH.CONTROL_ARMS


def menders(candidate=(0.0, 0.0, 0.0, 0.0), base=(0.0, 0.0, 0.0, 0.0),
            parent=(0.0, 0.0, 0.0, 0.0), replay=(0.0, 0.0, 0.0, 0.0)):
    values = {
        "base": base,
        "zero_root_parent": parent,
        "replay_ctl7": replay,
        "count_walk": candidate,
    }
    return {
        seed: {arm: values[arm][index] for arm in ARMS}
        for index, seed in enumerate(SEEDS)
    }


class EpisodeConversionTests(unittest.TestCase):
    def test_full_episode_draws_are_integer_exact(self) -> None:
        self.assertEqual(BENCH.menders_episodes(0.0), 0)
        self.assertEqual(BENCH.menders_episodes(0.1), 1)
        self.assertEqual(BENCH.menders_episodes(0.2), 2)
        self.assertEqual(BENCH.menders_episodes(1.0), 10)

    def test_partial_credit_draw_rounds_to_zero_episodes(self) -> None:
        # The program's recorded partial draw (1/60 = 0.0167) is a raw hit
        # but contributes zero episodes under the frozen round(10*score).
        self.assertEqual(BENCH.menders_episodes(0.016666666666666666), 0)

    def test_invalid_scores_fail_closed(self) -> None:
        for bad in (float("nan"), float("inf"), -0.1, 1.1, True, None, "0.1"):
            with self.assertRaises(ValueError):
                BENCH.menders_episodes(bad)


class ReplicationRuleTruthTableTests(unittest.TestCase):
    def test_all_zero_is_not_replicated_as_seed_noise(self) -> None:
        reading = BENCH.replication_reading(menders())
        self.assertEqual(reading["hits_c"], 0)
        self.assertEqual(reading["verdict"], "NOT_REPLICATED")
        self.assertIn("seed noise", reading["frozen_claim"])
        self.assertIn("expression-cost law stands", reading["frozen_claim"])

    def test_candidate_zero_with_control_draw_is_still_not_replicated(self) -> None:
        # hits_c == 0 closes the branch regardless of control behavior.
        reading = BENCH.replication_reading(menders(replay=(0.1, 0.0, 0.1, 0.0)))
        self.assertEqual(reading["hits_c"], 0)
        self.assertEqual(reading["verdict"], "NOT_REPLICATED")

    def test_single_hit_is_ambiguous_even_with_dominance(self) -> None:
        reading = BENCH.replication_reading(menders(candidate=(0.1, 0.0, 0.0, 0.0)))
        self.assertEqual(reading["hits_c"], 1)
        self.assertTrue(reading["candidate_dominates_every_control"])
        self.assertEqual(reading["verdict"], "AMBIGUOUS")
        self.assertIn("mechanism-differentiated NEW design", reading["frozen_claim"])

    def test_two_hits_with_all_controls_zero_is_replicated(self) -> None:
        reading = BENCH.replication_reading(menders(candidate=(0.1, 0.0, 0.1, 0.0)))
        self.assertEqual(reading["hits_c"], 2)
        self.assertEqual(reading["episode_totals"][CANDIDATE], 2)
        self.assertTrue(reading["candidate_dominates_every_control"])
        self.assertEqual(reading["verdict"], "REPLICATED")
        self.assertIn("no control matches", reading["frozen_claim"])
        self.assertIn("first confirmed menders capability movement",
                      reading["frozen_claim"])

    def test_episode_tie_with_any_control_is_not_dominance(self) -> None:
        # E_c == E_j for one control: hits_c >= 2 but the strict dominance
        # clause fails -> AMBIGUOUS, never REPLICATED.
        reading = BENCH.replication_reading(
            menders(candidate=(0.1, 0.1, 0.0, 0.0), replay=(0.1, 0.0, 0.1, 0.0))
        )
        self.assertEqual(reading["hits_c"], 2)
        self.assertEqual(reading["episode_totals"][CANDIDATE], 2)
        self.assertEqual(reading["episode_totals"]["replay_ctl7"], 2)
        self.assertFalse(reading["candidate_dominates_control"]["replay_ctl7"])
        self.assertTrue(reading["candidate_dominates_control"]["base"])
        self.assertEqual(reading["verdict"], "AMBIGUOUS")

    def test_control_exceeding_candidate_is_ambiguous(self) -> None:
        reading = BENCH.replication_reading(
            menders(candidate=(0.1, 0.1, 0.0, 0.0),
                    parent=(0.1, 0.1, 0.1, 0.0))
        )
        self.assertEqual(reading["hits_c"], 2)
        self.assertFalse(reading["candidate_dominates_control"]["zero_root_parent"])
        self.assertEqual(reading["verdict"], "AMBIGUOUS")

    def test_two_partial_hits_carry_zero_episodes_and_stay_ambiguous(self) -> None:
        partial = 0.016666666666666666
        reading = BENCH.replication_reading(
            menders(candidate=(partial, partial, 0.0, 0.0))
        )
        self.assertEqual(reading["hits_c"], 2)
        self.assertEqual(reading["episode_totals"][CANDIDATE], 0)
        self.assertFalse(reading["candidate_dominates_every_control"])
        self.assertEqual(reading["verdict"], "AMBIGUOUS")

    def test_four_hits_with_dominance_is_replicated(self) -> None:
        reading = BENCH.replication_reading(
            menders(candidate=(0.1, 0.2, 0.1, 0.1), replay=(0.1, 0.0, 0.0, 0.0))
        )
        self.assertEqual(reading["hits_c"], 4)
        self.assertEqual(reading["episode_totals"][CANDIDATE], 5)
        self.assertEqual(reading["verdict"], "REPLICATED")

    def test_verdict_partition_is_total_with_no_fourth_state(self) -> None:
        grid = (0.0, 0.016666666666666666, 0.1, 0.2)
        for c1 in grid:
            for c2 in grid:
                for r1 in grid:
                    reading = BENCH.replication_reading(
                        menders(candidate=(c1, c2, 0.1, 0.0),
                                replay=(r1, 0.0, 0.0, 0.0))
                    )
                    self.assertIn(
                        reading["verdict"],
                        {"REPLICATED", "NOT_REPLICATED", "AMBIGUOUS"},
                    )
                    hits = reading["hits_c"]
                    dominant = reading["candidate_dominates_every_control"]
                    if hits >= 2 and dominant:
                        self.assertEqual(reading["verdict"], "REPLICATED")
                    elif hits == 0:
                        self.assertEqual(reading["verdict"], "NOT_REPLICATED")
                    else:
                        self.assertEqual(reading["verdict"], "AMBIGUOUS")

    def test_prior_event_is_never_pooled(self) -> None:
        reading = BENCH.replication_reading(menders(candidate=(0.1, 0.1, 0.0, 0.0)))
        self.assertEqual(reading["events_counted"], list(SEEDS))
        self.assertNotIn(BENCH.PRIOR_EVENT["seed"], reading["events_counted"])
        self.assertFalse(reading["prior_event_pooled"])

    def test_wrong_seed_set_fails_closed(self) -> None:
        table = menders()
        table[99999] = table.pop(SEEDS[0])
        with self.assertRaisesRegex(ValueError, "four frozen new events"):
            BENCH.replication_reading(table)

    def test_missing_arm_fails_closed(self) -> None:
        table = menders()
        del table[SEEDS[0]]["base"]
        with self.assertRaisesRegex(ValueError, "all four arms"):
            BENCH.replication_reading(table)

    def test_rule_text_is_the_frozen_contract(self) -> None:
        self.assertIn("round(10*score)", BENCH.REPLICATION_RULE)
        self.assertIn("hits_c >= 2", BENCH.REPLICATION_RULE)
        self.assertIn("EVERY control", BENCH.REPLICATION_RULE)
        self.assertIn("never pooled", BENCH.REPLICATION_RULE)
        self.assertIn("No fourth state", BENCH.REPLICATION_RULE)
        self.assertEqual(
            set(BENCH.FROZEN_CLAIMS),
            {"REPLICATED", "NOT_REPLICATED", "AMBIGUOUS"},
        )


if __name__ == "__main__":
    unittest.main()
