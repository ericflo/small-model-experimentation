import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCH = load_module("count_walk_run_benchmark", "run_benchmark.py")

FAMILIES = sorted(BENCH.PUBLIC_FAMILIES)


def per_family(base: float, overrides: dict[str, float] | None = None) -> dict:
    values = {family: base for family in FAMILIES}
    values.update(overrides or {})
    return values


def synthetic_events(
    candidate_aggregate: float = 0.6,
    candidate_overrides: dict[str, float] | None = None,
) -> dict:
    return {
        "base": {"aggregate": 0.2, "per_family": per_family(0.1, {"menders": 0.0, "rites": 0.0})},
        "zero_root_parent": {
            "aggregate": 0.5,
            "per_family": per_family(0.4, {"menders": 0.0, "rites": 0.0}),
        },
        "replay_ctl7": {
            "aggregate": 0.45,
            "per_family": per_family(0.35, {"menders": 0.0, "rites": 0.0}),
        },
        "count_walk": {
            "aggregate": candidate_aggregate,
            "per_family": per_family(0.45, candidate_overrides or {"menders": 0.0}),
        },
    }


class LedgerContractTests(unittest.TestCase):
    OPENED = {
        "name": "pilot",
        "phase": "opened",
        "seed": 78163,
        "think_budget": 1024,
        "tier": "medium",
    }

    def _write(self, directory: Path, rows: list[dict]) -> Path:
        ledger = directory / "benchmark_events.jsonl"
        ledger.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
            encoding="utf-8",
        )
        return ledger

    def test_missing_ledger_is_unconsumed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            BENCH.require_unconsumed_ledger(
                Path(directory) / "missing.jsonl", self.OPENED, False
            )

    def test_closed_record_refuses_forever(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = self._write(
                Path(directory), [self.OPENED, {**self.OPENED, "phase": "closed"}]
            )
            for resume in (False, True):
                with self.assertRaisesRegex(ValueError, "one-event budget"):
                    BENCH.require_unconsumed_ledger(ledger, self.OPENED, resume)

    def test_malformed_record_counts_as_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = self._write(Path(directory), [{"seed": 78163}])
            with self.assertRaisesRegex(ValueError, "one-event budget"):
                BENCH.require_unconsumed_ledger(ledger, self.OPENED, True)

    def test_opened_record_requires_resume(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = self._write(Path(directory), [self.OPENED])
            with self.assertRaisesRegex(ValueError, "--resume"):
                BENCH.require_unconsumed_ledger(ledger, self.OPENED, False)
            BENCH.require_unconsumed_ledger(ledger, self.OPENED, True)

    def test_mismatched_opened_record_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            ledger = self._write(Path(directory), [{**self.OPENED, "seed": 78161}])
            with self.assertRaisesRegex(ValueError, "does not match"):
                BENCH.require_unconsumed_ledger(ledger, self.OPENED, True)


class ReviewGateContractTests(unittest.TestCase):
    def test_missing_or_unverdicted_review_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            review = Path(directory) / "benchmark_design_review.md"
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text("# Review\n\nLooks fine.\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not been authorized"):
                BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")
            review.write_text(
                f"# Review\n\n{BENCH.BENCH_VERDICT}\n", encoding="utf-8"
            )
            BENCH.require_verdict(review, BENCH.BENCH_VERDICT, "benchmark review")

    def test_verdict_string_is_the_frozen_marker(self) -> None:
        self.assertEqual(BENCH.BENCH_VERDICT, "**Verdict:** `PASS_BENCHMARK_EVENT`.")


class ScoreFinitenessTests(unittest.TestCase):
    def test_valid_scores(self) -> None:
        for value in (0.0, 0.5, 1.0, 1):
            self.assertTrue(BENCH._valid_score(value))

    def test_invalid_scores(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01, True, None, "0.5"):
            self.assertFalse(BENCH._valid_score(value))


class PilotGateTests(unittest.TestCase):
    def test_strict_wins_pass(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.6), "count_walk")
        self.assertTrue(gate["passes_pilot_gate"])

    def test_tie_with_parent_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.5), "count_walk")
        self.assertTrue(gate["strictly_beats_base_aggregate"])
        self.assertTrue(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["strictly_beats_immediate_parent_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])

    def test_below_replay_control_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.4), "count_walk")
        self.assertFalse(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])


class GoalGateTests(unittest.TestCase):
    def test_rites_only_flip_is_nine_wins_and_still_fails(self) -> None:
        # menders 0 for every arm: a rites flip alone gives nine strict wins
        # and the goal gate stays failed — recorded, never the pilot pass.
        events = synthetic_events(candidate_overrides={"rites": 0.1, "menders": 0.0})
        candidate = BENCH.goal_gate_reading(events)["per_arm"]["count_walk"]
        self.assertEqual(candidate["strict_wins"], 9)
        self.assertIn("rites", candidate["wins"])
        self.assertEqual(candidate["ties"], ["menders"])
        self.assertEqual(candidate["losses"], [])
        self.assertFalse(candidate["goal_gate_pass"])

    def test_menders_and_rites_flip_passes_the_goal_gate(self) -> None:
        events = synthetic_events(
            candidate_overrides={"menders": 0.1, "rites": 0.1}
        )
        reading = BENCH.goal_gate_reading(events)
        candidate = reading["per_arm"]["count_walk"]
        self.assertEqual(candidate["strict_wins"], 10)
        self.assertTrue(candidate["goal_gate_pass"])
        self.assertEqual(candidate["losses"], [])
        parent = reading["per_arm"]["zero_root_parent"]
        self.assertEqual(parent["strict_wins"], 8)
        self.assertEqual(parent["ties"], ["menders", "rites"])
        self.assertFalse(parent["goal_gate_pass"])

    def test_new_loss_is_recorded(self) -> None:
        events = synthetic_events(
            candidate_overrides={"menders": 0.1, "rites": 0.1, "warren": 0.05}
        )
        candidate = BENCH.goal_gate_reading(events)["per_arm"]["count_walk"]
        self.assertIn("warren", candidate["losses"])
        self.assertFalse(candidate["goal_gate_pass"])

    def test_reading_carries_the_frozen_power_statement(self) -> None:
        reading = BENCH.goal_gate_reading(synthetic_events())
        self.assertFalse(reading["included_in_pilot_gate"])
        self.assertTrue(reading["recorded_either_way"])
        self.assertIn("menders", reading["power_statement"])
        self.assertIn("0-margin", reading["power_statement"])
        self.assertIn("mechanism answer", reading["power_statement"])
        self.assertIn("10/10", reading["power_statement"])
        self.assertIn("confirmation cell", reading["power_statement"])


def synthetic_promotion(
    candidate_next: int = 30,
    parent_next: int = 5,
    replay_next: int = 4,
    promoted: str | None = "count_walk",
    rows: int = 40,
) -> dict:
    return {
        "promoted": promoted,
        "mechanism_reading": {
            "enumeration_fidelity_per_arm": {
                "count_walk": {
                    "rows_with_readout": rows,
                    "canonical_next": candidate_next,
                },
                "zero_root_parent": {
                    "rows_with_readout": rows,
                    "canonical_next": parent_next,
                },
                "replay_ctl7": {
                    "rows_with_readout": rows,
                    "canonical_next": replay_next,
                },
            }
        },
    }


class FidelityPreconditionTests(unittest.TestCase):
    def test_holds_at_high_fidelity_over_both_controls(self) -> None:
        result = BENCH.fidelity_precondition(synthetic_promotion(30, 5, 4))
        self.assertTrue(result["holds"])
        self.assertTrue(result["meets_min_rate"])
        self.assertTrue(result["strictly_beats_parent_rate"])
        self.assertTrue(result["strictly_beats_replay_rate"])
        self.assertAlmostEqual(result["candidate_rate"], 0.75)

    def test_exact_half_meets_the_min_rate(self) -> None:
        # F >= 0.50 is inclusive: 20/40 meets the bar (integer-exact).
        result = BENCH.fidelity_precondition(synthetic_promotion(20, 5, 4))
        self.assertTrue(result["meets_min_rate"])
        self.assertTrue(result["holds"])
        below = BENCH.fidelity_precondition(synthetic_promotion(19, 5, 4))
        self.assertFalse(below["meets_min_rate"])
        self.assertFalse(below["holds"])

    def test_tie_with_a_control_fails_the_strict_clause(self) -> None:
        result = BENCH.fidelity_precondition(synthetic_promotion(25, 25, 4))
        self.assertTrue(result["meets_min_rate"])
        self.assertFalse(result["strictly_beats_parent_rate"])
        self.assertFalse(result["holds"])

    def test_unpromoted_candidate_never_holds(self) -> None:
        result = BENCH.fidelity_precondition(
            synthetic_promotion(30, 5, 4, promoted=None)
        )
        self.assertFalse(result["candidate_promoted"])
        self.assertFalse(result["holds"])

    def test_malformed_readout_fails_closed(self) -> None:
        promotion = synthetic_promotion()
        del promotion["mechanism_reading"]["enumeration_fidelity_per_arm"][
            "replay_ctl7"
        ]["canonical_next"]
        with self.assertRaisesRegex(ValueError, "enumeration-fidelity"):
            BENCH.fidelity_precondition(promotion)


class MendersReadingTests(unittest.TestCase):
    def test_candidate_nonzero_where_controls_zero_is_the_mechanism_answer(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.1})
        reading = BENCH.menders_reading(events, "count_walk", synthetic_promotion())
        self.assertEqual(reading["family"], "menders")
        self.assertTrue(reading["candidate_nonzero"])
        self.assertTrue(reading["controls_all_zero"])
        self.assertTrue(reading["mechanism_answer"])
        self.assertEqual(reading["frozen_interpretation"], "MECHANISM_ANSWER")
        self.assertAlmostEqual(reading["candidate_minus_base"], 0.1)
        self.assertAlmostEqual(reading["candidate_minus_parent"], 0.1)
        self.assertAlmostEqual(reading["candidate_minus_replay"], 0.1)
        self.assertFalse(reading["included_in_pilot_gate"])
        self.assertTrue(reading["recorded_either_way"])
        self.assertIn("rerun feedback", reading["frozen_question"])

    def test_positive_takes_precedence_even_when_the_precondition_holds(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.1})
        reading = BENCH.menders_reading(
            events, "count_walk", synthetic_promotion(30, 5, 4)
        )
        self.assertTrue(reading["fidelity_precondition"]["holds"])
        self.assertEqual(reading["frozen_interpretation"], "MECHANISM_ANSWER")

    def test_zero_draw_with_precondition_is_turn_budget_scoped(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.0})
        reading = BENCH.menders_reading(
            events, "count_walk", synthetic_promotion(30, 5, 4)
        )
        self.assertFalse(reading["mechanism_answer"])
        self.assertEqual(reading["frozen_interpretation"], "TURN_BUDGET_SCOPED")
        self.assertIn("NOT refuted", reading["interpretation"])
        self.assertIn("pure-enumeration route", reading["interpretation"])

    def test_zero_draw_without_precondition_fails_on_its_own_terms(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.0})
        reading = BENCH.menders_reading(
            events, "count_walk", synthetic_promotion(10, 5, 4)
        )
        self.assertEqual(
            reading["frozen_interpretation"], "FAILED_ON_ITS_OWN_TERMS"
        )
        self.assertIn("own terms", reading["interpretation"])

    def test_zero_draw_has_no_third_state(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.0})
        for promotion in (
            synthetic_promotion(30, 5, 4),
            synthetic_promotion(19, 5, 4),
            synthetic_promotion(25, 25, 4),
        ):
            reading = BENCH.menders_reading(events, "count_walk", promotion)
            self.assertIn(
                reading["frozen_interpretation"],
                {"TURN_BUDGET_SCOPED", "FAILED_ON_ITS_OWN_TERMS"},
            )
        self.assertEqual(
            BENCH.menders_reading(
                events, "count_walk", synthetic_promotion()
            )["frozen_consequence_order"],
            ["MECHANISM_ANSWER", "TURN_BUDGET_SCOPED", "FAILED_ON_ITS_OWN_TERMS"],
        )

    def test_nonzero_without_clean_contrast_is_outside_the_frozen_pair(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.2})
        events["zero_root_parent"]["per_family"]["menders"] = 0.1
        reading = BENCH.menders_reading(events, "count_walk", synthetic_promotion())
        self.assertTrue(reading["candidate_nonzero"])
        self.assertFalse(reading["controls_all_zero"])
        self.assertFalse(reading["mechanism_answer"])
        self.assertIsNone(reading["frozen_interpretation"])
        self.assertAlmostEqual(reading["candidate_minus_parent"], 0.1)

    def test_reading_quotes_the_episode_success_simulation(self) -> None:
        reading = BENCH.menders_reading(
            synthetic_events(), "count_walk", synthetic_promotion()
        )
        sim = reading["episode_success_simulation"]
        self.assertEqual(sim["holdout_from_scratch"]["mean"], 27.1)
        self.assertEqual(sim["holdout_from_scratch"]["median"], 20.5)
        self.assertEqual(sim["holdout_from_scratch"]["max"], 78)
        self.assertEqual(
            sim["holdout_from_scratch"]["share_needing_more_than_10_turns"], 0.8
        )
        self.assertEqual(sim["treatment_from_scratch"]["mean"], 32.64375)
        self.assertEqual(
            sim["treatment_from_scratch"]["share_needing_more_than_10_turns"],
            0.86875,
        )

    def test_simulation_constants_match_the_frozen_design_receipts(self) -> None:
        receipt = json.loads(
            (EXP / "data" / "local_design_receipt.json").read_text(encoding="utf-8")
        )
        sim = receipt["episode_success_simulation"]
        holdout = BENCH.EPISODE_SUCCESS_SIMULATION["holdout_from_scratch"]
        self.assertEqual(sim["from_scratch_distribution"]["mean"], holdout["mean"])
        self.assertEqual(sim["from_scratch_distribution"]["max"], holdout["max"])
        self.assertEqual(sim["from_scratch_distribution"]["min"], holdout["min"])
        turns = sorted(sim["per_row_from_scratch"])
        n = len(turns)
        median = (turns[n // 2 - 1] + turns[n // 2]) / 2 if n % 2 == 0 else turns[n // 2]
        self.assertEqual(median, holdout["median"])
        share = sum(t > 10 for t in turns) / n
        self.assertEqual(share, holdout["share_needing_more_than_10_turns"])
        manifest = json.loads(
            (EXP / "data" / "corpus_manifest.json").read_text(encoding="utf-8")
        )
        counts = {
            int(k): v
            for k, v in manifest["balance"]["episode_success_turns_counts"].items()
        }
        total = sum(counts.values())
        treatment = BENCH.EPISODE_SUCCESS_SIMULATION["treatment_from_scratch"]
        self.assertEqual(total, 160)
        mean = sum(k * v for k, v in counts.items()) / total
        self.assertEqual(mean, treatment["mean"])
        self.assertEqual(max(counts), treatment["max"])
        self.assertEqual(min(counts), treatment["min"])
        share = sum(v for k, v in counts.items() if k > 10) / total
        self.assertEqual(share, treatment["share_needing_more_than_10_turns"])
        expanded = sorted(k for k, v in counts.items() for _ in range(v))
        median = (expanded[79] + expanded[80]) / 2
        self.assertEqual(median, treatment["median"])

    def test_per_arm_values_are_carried_for_all_four_arms(self) -> None:
        reading = BENCH.menders_reading(
            synthetic_events(), "count_walk", synthetic_promotion()
        )
        self.assertEqual(
            set(reading["per_arm"]),
            {"base", "zero_root_parent", "replay_ctl7", "count_walk"},
        )

    def test_reading_carries_the_two_direction_power_statement(self) -> None:
        reading = BENCH.menders_reading(
            synthetic_events(), "count_walk", synthetic_promotion()
        )
        statement = reading["power_statement"]
        self.assertIn("mechanism answer", statement)
        self.assertIn("menders", statement)
        self.assertIn("TURN_BUDGET_SCOPED", statement)
        self.assertIn(">= 0.50", statement)
        self.assertIn("NOT refuted", statement)
        self.assertIn("No third state", statement)
        self.assertIn("27.1", statement)
        self.assertIn("80.0%", statement)
        self.assertIn("bounded", statement)


class FrozenBenchmarkConstantsTests(unittest.TestCase):
    def test_event_identity(self) -> None:
        self.assertEqual(BENCH.FROZEN_NAME, "pilot")
        self.assertEqual(BENCH.FROZEN_TIER, "medium")
        self.assertEqual(BENCH.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(BENCH.FROZEN_SEED, 78163)
        self.assertEqual(
            BENCH.MODEL_ORDER,
            ("base", "zero_root_parent", "replay_ctl7", "count_walk"),
        )
        self.assertEqual(BENCH.FROZEN_CANDIDATES, ("count_walk",))
        self.assertEqual(BENCH.MENDERS_FAMILY, "menders")

    def test_inherited_pins_are_filled_and_trained_pins_fail_closed(self) -> None:
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["base"])
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["zero_root_parent"])
        self.assertIsNotNone(BENCH.FROZEN_WEIGHTS_SHA256["base"])
        with self.assertRaisesRegex(ValueError, "TODO-PIN"):
            BENCH.require_pin(None, "FROZEN_TREE_SHA256['replay_ctl7']")
        for label in ("replay_ctl7", "count_walk"):
            for table in (
                BENCH.FROZEN_TREE_SHA256,
                BENCH.FROZEN_WEIGHTS_SHA256,
            ):
                pin = table[label]
                self.assertTrue(pin is None or isinstance(pin, str))

    def test_unfilled_slot_pins_refuse_the_event(self) -> None:
        # All six trained-arm slots (tree/weights/receipt x two arms) must
        # refuse fail-closed while any is None or malformed.
        filled = all(
            isinstance(value, str)
            for value in (
                BENCH.FROZEN_TREE_SHA256["replay_ctl7"],
                BENCH.FROZEN_TREE_SHA256["count_walk"],
                BENCH.FROZEN_WEIGHTS_SHA256["replay_ctl7"],
                BENCH.FROZEN_WEIGHTS_SHA256["count_walk"],
                BENCH.REPLAY_CTL7_MERGE_RECEIPT_SHA256,
                BENCH.COUNT_WALK_MERGE_RECEIPT_SHA256,
            )
        )
        if filled:
            BENCH.require_todo_pins_filled()
        else:
            with self.assertRaisesRegex(ValueError, "TODO-PIN"):
                BENCH.require_todo_pins_filled()

    def test_parent_provenance_pins_are_the_lifecycle22_receipt(self) -> None:
        self.assertEqual(
            BENCH.ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256,
            "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b",
        )
        self.assertEqual(
            BENCH.FROZEN_TREE_SHA256["zero_root_parent"],
            "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d",
        )
        self.assertEqual(
            BENCH.FROZEN_WEIGHTS_SHA256["zero_root_parent"],
            "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932",
        )

    def test_gateway_receipt_authentication_rejects_nan_scores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model"
            model.mkdir()
            (model / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
            receipt_sha = BENCH.sha256_file(model / "merge_receipt.json")
            event = {
                "schema_version": 1,
                "stage": "menagerie_aggregate_gateway",
                "tier": "medium",
                "think_budget": 1024,
                "seed": 78163,
                "backend": "qwen_vllm",
                "model": str(model),
                "model_merge_receipt_sha256": receipt_sha,
                "benchmark_runner_sha256": "x",
                "benchmark_source_inventory_sha256": "y",
                "benchmark_source_file_count": 3,
                "aggregate": 0.5,
                "per_family": {family: 0.5 for family in FAMILIES},
                "within_budget": True,
                "wall_seconds": 1.0,
            }
            path = Path(directory) / "event.json"
            path.write_text(json.dumps(event), encoding="utf-8")
            self.assertEqual(BENCH.load_event(path, model)["aggregate"], 0.5)
            event["per_family"]["menders"] = math.nan
            path.write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                BENCH.load_event(path, model)
            event["per_family"]["menders"] = 0.5
            event["extra_key"] = 1
            path.write_text(json.dumps(event), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "failed authentication"):
                BENCH.load_event(path, model)


if __name__ == "__main__":
    unittest.main()
