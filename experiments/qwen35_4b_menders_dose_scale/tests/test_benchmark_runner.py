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


BENCH = load_module("menders_run_benchmark", "run_benchmark.py")

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
        "base": {"aggregate": 0.2, "per_family": per_family(0.1, {"menders": 0.0})},
        "hygiene_explore_parent": {
            "aggregate": 0.5,
            "per_family": per_family(0.4, {"menders": 0.0}),
        },
        "replay_ctl3": {
            "aggregate": 0.45,
            "per_family": per_family(0.35, {"menders": 0.0}),
        },
        "feedloop_scale": {
            "aggregate": candidate_aggregate,
            "per_family": per_family(0.45, candidate_overrides or {"menders": 0.0}),
        },
    }


def valid_closed_record(receipts: dict | None = None) -> dict:
    return {
        "name": "pilot",
        "phase": "closed",
        "tier": "medium",
        "think_budget": 1024,
        "seed": 78158,
        "summary": str(BENCH.EVENT_DIR / "summary.json"),
        "summary_sha256": "a" * 64,
        "receipts": receipts
        or {label: "b" * 64 for label in BENCH.MODEL_ORDER},
    }


class LedgerContractTests(unittest.TestCase):
    OPENED = {
        "name": "pilot",
        "phase": "opened",
        "seed": 78158,
        "think_budget": 1024,
        "tier": "medium",
    }

    def test_opened_record_matches_the_frozen_shape(self) -> None:
        self.assertEqual(BENCH.opened_record(), self.OPENED)

    def test_missing_ledger_is_fresh(self) -> None:
        self.assertEqual(BENCH.ledger_plan([], False), "fresh")

    def test_closed_record_refuses_forever(self) -> None:
        rows = [self.OPENED, valid_closed_record()]
        for resume in (False, True):
            with self.assertRaisesRegex(ValueError, "one-event budget"):
                BENCH.ledger_plan(rows, resume)

    def test_malformed_closed_record_fails_closed(self) -> None:
        # A closed record missing the receipts pins is NOT a valid history.
        broken = valid_closed_record()
        del broken["receipts"]
        with self.assertRaisesRegex(ValueError, "not a valid one-event record"):
            BENCH.ledger_plan([self.OPENED, broken], True)

    def test_opened_record_requires_resume(self) -> None:
        with self.assertRaisesRegex(ValueError, "--resume"):
            BENCH.ledger_plan([self.OPENED], False)
        self.assertEqual(BENCH.ledger_plan([self.OPENED], True), "crashed")

    def test_mismatched_opened_record_refuses(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match"):
            BENCH.ledger_plan([{**self.OPENED, "seed": 78154}], True)

    def test_trailing_rows_fail_closed(self) -> None:
        rows = [self.OPENED, valid_closed_record(), {"extra": True}]
        with self.assertRaisesRegex(ValueError, "not a valid one-event record"):
            BENCH.ledger_plan(rows, True)


class ClosedRecordShapeTests(unittest.TestCase):
    def test_valid_record_passes(self) -> None:
        self.assertTrue(BENCH.is_closed_record(valid_closed_record()))

    def test_wrong_arm_set_fails(self) -> None:
        record = valid_closed_record({"base": "b" * 64})
        self.assertFalse(BENCH.is_closed_record(record))

    def test_non_hex_receipt_fails(self) -> None:
        receipts = {label: "b" * 64 for label in BENCH.MODEL_ORDER}
        receipts["base"] = "not-a-sha"
        self.assertFalse(BENCH.is_closed_record(valid_closed_record(receipts)))

    def test_extra_key_fails(self) -> None:
        record = valid_closed_record()
        record["extra"] = 1
        self.assertFalse(BENCH.is_closed_record(record))

    def test_wrong_seed_fails(self) -> None:
        record = valid_closed_record()
        record["seed"] = 78154
        self.assertFalse(BENCH.is_closed_record(record))


class StaleFilesAndRecoveryTests(unittest.TestCase):
    def test_unopened_event_requires_a_clean_slate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            event_dir = Path(directory)
            self.assertEqual(BENCH.stale_event_files(event_dir), [])
            (event_dir / "base.json").write_text("{}", encoding="utf-8")
            (event_dir / "summary.json").write_text("{}", encoding="utf-8")
            self.assertEqual(
                BENCH.stale_event_files(event_dir), ["base.json", "summary.json"]
            )

    def test_missing_directory_is_clean(self) -> None:
        self.assertEqual(BENCH.stale_event_files(Path("/nonexistent/event")), [])

    def test_crashed_summary_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            summary = Path(directory) / "summary.json"
            summary.write_bytes(b'{"a": 1}\n')
            BENCH.reconcile_crashed_summary(summary, b'{"a": 1}\n')
            with self.assertRaisesRegex(ValueError, "deterministic regeneration"):
                BENCH.reconcile_crashed_summary(summary, b'{"a": 2}\n')


class ScoreFinitenessTests(unittest.TestCase):
    def test_valid_scores(self) -> None:
        for value in (0.0, 0.5, 1.0, 1):
            self.assertTrue(BENCH._valid_score(value))

    def test_invalid_scores(self) -> None:
        for value in (float("nan"), float("inf"), -0.01, 1.01, True, None, "0.5"):
            self.assertFalse(BENCH._valid_score(value))


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


class PilotGateTests(unittest.TestCase):
    def test_strict_wins_pass(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.6), "feedloop_scale")
        self.assertTrue(gate["passes_pilot_gate"])

    def test_tie_with_parent_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.5), "feedloop_scale")
        self.assertTrue(gate["strictly_beats_base_aggregate"])
        self.assertTrue(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["strictly_beats_immediate_parent_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])

    def test_below_replay_control_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.4), "feedloop_scale")
        self.assertFalse(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])


class GoalGateTests(unittest.TestCase):
    def test_menders_tie_caps_the_reading_at_nine(self) -> None:
        # menders is the last blocking family: with the candidate tied at 0
        # the goal gate stays failed at nine strict wins.
        reading = BENCH.goal_gate_reading(synthetic_events())
        candidate = reading["per_arm"]["feedloop_scale"]
        self.assertEqual(candidate["strict_wins"], 9)
        self.assertEqual(candidate["ties"], ["menders"])
        self.assertEqual(candidate["losses"], [])
        self.assertFalse(candidate["goal_gate_pass"])
        self.assertEqual(candidate["menders_margin_vs_base"], 0.0)

    def test_menders_flip_passes_the_goal_gate(self) -> None:
        events = synthetic_events(candidate_overrides={"menders": 0.1})
        reading = BENCH.goal_gate_reading(events)
        candidate = reading["per_arm"]["feedloop_scale"]
        self.assertEqual(candidate["strict_wins"], 10)
        self.assertTrue(candidate["goal_gate_pass"])
        self.assertIn("menders", candidate["wins"])
        self.assertGreater(candidate["menders_margin_vs_base"], 0)
        parent = reading["per_arm"]["hygiene_explore_parent"]
        self.assertEqual(parent["ties"], ["menders"])
        self.assertFalse(parent["goal_gate_pass"])

    def test_new_loss_is_recorded(self) -> None:
        events = synthetic_events(
            candidate_overrides={"menders": 0.1, "warren": 0.05}
        )
        candidate = BENCH.goal_gate_reading(events)["per_arm"]["feedloop_scale"]
        self.assertIn("warren", candidate["losses"])
        self.assertFalse(candidate["goal_gate_pass"])

    def test_reading_carries_the_frozen_power_statement(self) -> None:
        reading = BENCH.goal_gate_reading(synthetic_events())
        self.assertFalse(reading["included_in_pilot_gate"])
        self.assertTrue(reading["recorded_either_way"])
        self.assertIn("menders > 0", reading["power_statement"])
        self.assertIn("10/10", reading["power_statement"])
        self.assertIn("confirmation cell", reading["power_statement"])
        self.assertIn("dose scale", reading["power_statement"])


class FrozenBenchmarkConstantsTests(unittest.TestCase):
    def test_event_identity(self) -> None:
        self.assertEqual(BENCH.FROZEN_NAME, "pilot")
        self.assertEqual(BENCH.FROZEN_TIER, "medium")
        self.assertEqual(BENCH.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(BENCH.FROZEN_SEED, 78158)
        self.assertEqual(
            BENCH.MODEL_ORDER,
            ("base", "hygiene_explore_parent", "replay_ctl3", "feedloop_scale"),
        )
        self.assertEqual(BENCH.FROZEN_CANDIDATES, ("feedloop_scale",))
        self.assertEqual(BENCH.LOCAL_SEED, 88037)

    def test_inherited_pins_are_filled_and_trained_pins_fail_closed(self) -> None:
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["base"])
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["hygiene_explore_parent"])
        self.assertIsNotNone(BENCH.FROZEN_WEIGHTS_SHA256["base"])
        with self.assertRaisesRegex(ValueError, "TODO-PIN"):
            BENCH.require_pin(None, "FROZEN_TREE_SHA256['replay_ctl3']")
        for label in ("replay_ctl3", "feedloop_scale"):
            for table in (
                BENCH.FROZEN_TREE_SHA256,
                BENCH.FROZEN_WEIGHTS_SHA256,
            ):
                pin = table[label]
                self.assertTrue(pin is None or isinstance(pin, str))
            relative, receipt_pin = BENCH.COMMITTED_MERGE_RECEIPTS[label]
            self.assertIn(f"runs/merges/{label}.json", relative)
            self.assertTrue(receipt_pin is None or isinstance(receipt_pin, str))

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
                "seed": 78158,
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
