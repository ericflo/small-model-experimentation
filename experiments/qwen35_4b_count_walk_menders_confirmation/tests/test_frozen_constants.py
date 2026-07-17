import importlib.util
import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module("cwmc_run_benchmark_constants", "run_benchmark.py")
POWER = load_module("cwmc_power_analysis", "power_analysis.py")

FAMILIES = sorted(BENCH.PUBLIC_FAMILIES)


class FrozenEventIdentityTests(unittest.TestCase):
    def test_event_identity(self) -> None:
        self.assertEqual(BENCH.FROZEN_NAME, "confirmation")
        self.assertEqual(BENCH.FROZEN_TIER, "medium")
        self.assertEqual(BENCH.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(BENCH.SEED_ORDER, (78164, 78165, 78166, 78167))
        self.assertEqual(
            BENCH.MODEL_ORDER,
            ("base", "zero_root_parent", "replay_ctl7", "count_walk"),
        )
        self.assertEqual(BENCH.CANDIDATE, "count_walk")
        self.assertEqual(
            BENCH.CONTROL_ARMS, ("base", "zero_root_parent", "replay_ctl7")
        )
        self.assertEqual(BENCH.MENDERS_FAMILY, "menders")

    def test_prior_event_is_78163_and_never_counted(self) -> None:
        self.assertEqual(BENCH.PRIOR_EVENT["seed"], 78163)
        self.assertFalse(BENCH.PRIOR_EVENT["counted_in_verdict"])
        self.assertNotIn(BENCH.PRIOR_EVENT["seed"], BENCH.SEED_ORDER)
        self.assertEqual(
            BENCH.PRIOR_EVENT["summary_sha256"],
            "a8c394758aeea8255389b1d7c2b6d7c3f37d6072d9ea226f1b4786a8eee191af",
        )


class FrozenPinTests(unittest.TestCase):
    def test_gateway_pin_matches_the_reference_cell(self) -> None:
        self.assertEqual(
            BENCH.GATEWAY_SHA256,
            "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17",
        )

    def test_tree_and_weights_pins_are_the_design_time_constants(self) -> None:
        self.assertEqual(
            BENCH.FROZEN_TREE_SHA256,
            {
                "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
                "zero_root_parent": "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d",
                "replay_ctl7": "044a4599ac5264e00256f66f65215ea497d3631d8aebd3467b698253648e484a",
                "count_walk": "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1",
            },
        )
        self.assertEqual(
            BENCH.FROZEN_WEIGHTS_SHA256,
            {
                "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
                "zero_root_parent": "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932",
                "replay_ctl7": "c5035b4db47e4da582a805ca009747a5618ef5badc35d960ca216e586dd3ab9d",
                "count_walk": "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3",
            },
        )
        self.assertEqual(BENCH.WEIGHTS_SIZE_BYTES, 9_078_620_536)

    def test_merge_receipt_pins(self) -> None:
        self.assertEqual(
            BENCH.COMMITTED_MERGE_RECEIPTS["replay_ctl7"][1],
            "3f65b4c6f4a8b0574a574a89d417c174c3762de6f93508bed8a5a987b91e224c",
        )
        self.assertEqual(
            BENCH.COMMITTED_MERGE_RECEIPTS["count_walk"][1],
            "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36",
        )
        self.assertEqual(
            BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256,
            "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b",
        )

    def test_no_todo_pin_placeholder_exists(self) -> None:
        # Every pin is a filled 64-hex constant at design time.
        for table in (BENCH.FROZEN_TREE_SHA256, BENCH.FROZEN_WEIGHTS_SHA256):
            for value in table.values():
                self.assertRegex(value, r"^[0-9a-f]{64}$")
        source = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
        self.assertNotIn("TODO-PIN", source.replace("no TODO-PIN slot", ""))

    def test_provenance_copies_exist_with_pinned_shas(self) -> None:
        for copy_relative, (source_relative, expected) in sorted(
            BENCH.PROVENANCE_COPIES.items()
        ):
            copy_path = EXP / copy_relative
            source_path = ROOT / source_relative
            self.assertTrue(copy_path.is_file(), copy_relative)
            self.assertEqual(BENCH.sha256_file(copy_path), expected, copy_relative)
            self.assertEqual(
                copy_path.read_bytes(), source_path.read_bytes(), copy_relative
            )


class PriorReferenceTests(unittest.TestCase):
    def test_prior_summary_authenticates_as_the_mechanism_answer(self) -> None:
        prior = BENCH.load_prior_reference()
        menders = {
            label: prior["scores"][label]["per_family"]["menders"]
            for label in BENCH.MODEL_ORDER
        }
        self.assertEqual(
            menders,
            {
                "base": 0.0,
                "zero_root_parent": 0.0,
                "replay_ctl7": 0.0,
                "count_walk": 0.1,
            },
        )
        self.assertEqual(
            prior["benchmark_implementation"], BENCH.PRIOR_IMPLEMENTATION
        )

    def test_prior_report_converts_episodes(self) -> None:
        prior = BENCH.load_prior_reference()
        report = BENCH.prior_event_report(prior["scores"])
        self.assertEqual(
            report["menders_episodes_per_arm"],
            {"base": 0, "zero_root_parent": 0, "replay_ctl7": 0, "count_walk": 1},
        )
        self.assertFalse(report["counted_in_verdict"])
        self.assertTrue(report["mechanism_answer"])


def synthetic_scores(menders_candidate: dict[int, float]) -> dict:
    scores = {}
    for seed in BENCH.SEED_ORDER:
        per_seed = {}
        for label in BENCH.MODEL_ORDER:
            per_family = {family: 0.2 for family in FAMILIES}
            per_family["menders"] = (
                menders_candidate.get(seed, 0.0) if label == "count_walk" else 0.0
            )
            per_seed[label] = {"aggregate": 0.25, "per_family": per_family}
        scores[seed] = per_seed
    return scores


def synthetic_budget(within: bool = True) -> dict:
    return {
        seed: {
            label: {"within_budget": within, "wall_seconds": 100.0}
            for label in BENCH.MODEL_ORDER
        }
        for seed in BENCH.SEED_ORDER
    }


def synthetic_receipts() -> dict:
    return {
        seed: {
            label: {"path": f"receipt/{seed}/{label}.json", "sha256": "0" * 64}
            for label in BENCH.MODEL_ORDER
        }
        for seed in BENCH.SEED_ORDER
    }


class ReadoutSchemaTests(unittest.TestCase):
    def _build(self, menders_candidate: dict[int, float], within: bool = True) -> dict:
        prior = BENCH.load_prior_reference()
        return BENCH.build_readout(
            synthetic_scores(menders_candidate),
            synthetic_budget(within),
            dict(BENCH.PRIOR_IMPLEMENTATION),
            prior,
            synthetic_receipts(),
        )

    def test_readout_schema_and_verdict_wiring(self) -> None:
        readout = self._build({78164: 0.1, 78166: 0.1})
        self.assertEqual(
            set(readout),
            {
                "schema_version", "experiment_id", "stage", "name", "tier",
                "think_budget", "seeds", "benchmark_data_read", "promoted",
                "outcome", "verdict", "frozen_claim", "paired_comparison_valid",
                "provenance", "prior_event", "benchmark_implementation",
                "receipts", "scores", "budget", "readings",
            },
        )
        self.assertEqual(readout["schema_version"], 1)
        self.assertEqual(readout["experiment_id"], EXP.name)
        self.assertEqual(readout["seeds"], list(BENCH.SEED_ORDER))
        self.assertFalse(readout["benchmark_data_read"])
        self.assertIsNone(readout["promoted"])
        self.assertEqual(readout["outcome"], "CONFIRMATION_READ_COMPLETE")
        self.assertEqual(readout["verdict"], "REPLICATED")
        self.assertEqual(
            readout["frozen_claim"], BENCH.FROZEN_CLAIMS["REPLICATED"]
        )
        self.assertEqual(
            set(readout["readings"]),
            {"replication", "per_seed", "budget_integrity"},
        )
        self.assertEqual(readout["prior_event"]["seed"], 78163)
        self.assertFalse(readout["prior_event"]["counted_in_verdict"])
        per_seed = readout["readings"]["per_seed"]
        self.assertEqual(set(per_seed), {str(seed) for seed in BENCH.SEED_ORDER})
        for block in per_seed.values():
            self.assertTrue(block["descriptive_only"])
            self.assertEqual(set(block["aggregates"]), set(BENCH.MODEL_ORDER))
            self.assertEqual(
                set(block["goal_gates_vs_base"]), set(BENCH.TREATED_ARMS)
            )
            self.assertEqual(
                set(block["candidate_vs_controls"]),
                {f"count_walk_minus_{label}" for label in BENCH.CONTROL_ARMS},
            )

    def test_zero_candidate_readout_is_not_replicated(self) -> None:
        readout = self._build({})
        self.assertEqual(readout["verdict"], "NOT_REPLICATED")
        self.assertIn("seed noise", readout["frozen_claim"])

    def test_single_hit_readout_is_ambiguous(self) -> None:
        readout = self._build({78165: 0.1})
        self.assertEqual(readout["verdict"], "AMBIGUOUS")

    def test_over_budget_arm_scopes_but_never_gates(self) -> None:
        readout = self._build({78164: 0.1, 78166: 0.1}, within=False)
        self.assertEqual(readout["verdict"], "REPLICATED")
        self.assertFalse(readout["paired_comparison_valid"])
        self.assertIsNotNone(readout["readings"]["budget_integrity"]["reason"])

    def test_implementation_drift_refuses_the_readout(self) -> None:
        prior = BENCH.load_prior_reference()
        drifted = dict(BENCH.PRIOR_IMPLEMENTATION)
        drifted["source_file_count"] = 57
        with self.assertRaisesRegex(ValueError, "not comparable"):
            BENCH.build_readout(
                synthetic_scores({}),
                synthetic_budget(),
                drifted,
                prior,
                synthetic_receipts(),
            )


class VerdictAndReviewMarkerTests(unittest.TestCase):
    def test_verdict_string_is_the_frozen_marker(self) -> None:
        self.assertEqual(BENCH.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`.")

    def test_review_gate_refuses_without_the_marker(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "benchmark_design_review.md"
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text("# Review\n\nLooks fine.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text(f"# Review\n\n{BENCH.BENCH_VERDICT}\n", encoding="utf-8")
            BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")


class ScoreFinitenessTests(unittest.TestCase):
    def test_valid_scores(self) -> None:
        for value in (0.0, 0.5, 1.0, 1):
            self.assertTrue(BENCH._valid_score(value))

    def test_invalid_scores(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01, True, None, "0.5"):
            self.assertFalse(BENCH._valid_score(value))

    def test_wall_seconds_guard(self) -> None:
        for value in (0.0, 1, 3600.5):
            self.assertTrue(BENCH._valid_wall_seconds(value))
        for value in (float("nan"), float("inf"), -1.0, True, None, "1"):
            self.assertFalse(BENCH._valid_wall_seconds(value))


class PowerArithmeticTests(unittest.TestCase):
    def test_preregistered_numbers_recompute_exactly(self) -> None:
        self.assertEqual(POWER.computed(), POWER.PREREGISTERED)

    def test_headline_false_positive_and_power_values(self) -> None:
        self.assertEqual(POWER.PREREGISTERED["p_hits_ge2_null"], 0.0523)
        self.assertEqual(POWER.PREREGISTERED["p_false_replicated_null"], 0.0450)
        self.assertEqual(
            POWER.PREREGISTERED["p_false_replicated_sensitivity"], 0.0947
        )
        self.assertEqual(
            POWER.PREREGISTERED["power_hits_ge2"],
            {"0.4": 0.5248, "0.5": 0.6875, "0.65": 0.8735},
        )
        self.assertEqual(
            POWER.PREREGISTERED["power_replicated"],
            {"0.4": 0.4717, "0.5": 0.6289, "0.65": 0.8230},
        )

    def test_noise_model_matches_the_frozen_audit(self) -> None:
        self.assertEqual(POWER.EVENTS, 4)
        self.assertEqual(POWER.CONTROL_ARMS, 3)
        self.assertEqual(POWER.OBSERVED_ARM_EVENTS, 29)
        self.assertEqual(POWER.OBSERVED_FULL_EPISODE_DRAWS, 3)
        self.assertEqual(POWER.OBSERVED_RAW_POSITIVE_DRAWS, 5)
        self.assertEqual(float(POWER.NULL_P), 0.1)


class GoalGateRowTests(unittest.TestCase):
    def test_strict_win_partition(self) -> None:
        base = {family: 0.1 for family in FAMILIES}
        treated = dict(base)
        treated["menders"] = 0.2
        treated["warren"] = 0.0
        row = BENCH.goal_gate_row(base, treated)
        self.assertEqual(row["strict_wins"], 1)
        self.assertEqual(row["wins"], ["menders"])
        self.assertEqual(row["losses"], ["warren"])
        self.assertEqual(len(row["ties"]), 8)
        self.assertFalse(row["goal_gate_pass"])

    def test_ten_strict_wins_pass(self) -> None:
        base = {family: 0.1 for family in FAMILIES}
        treated = {family: 0.2 for family in FAMILIES}
        self.assertTrue(BENCH.goal_gate_row(base, treated)["goal_gate_pass"])


if __name__ == "__main__":
    unittest.main()
