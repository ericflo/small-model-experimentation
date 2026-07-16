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


BENCH = load_module("statechain_run_benchmark", "run_benchmark.py")

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
        "replay_ctl4": {
            "aggregate": 0.45,
            "per_family": per_family(0.35, {"menders": 0.0, "rites": 0.0}),
        },
        "statechain_clean": {
            "aggregate": candidate_aggregate,
            "per_family": per_family(0.45, candidate_overrides or {}),
        },
    }


class LedgerContractTests(unittest.TestCase):
    OPENED = {
        "name": "pilot",
        "phase": "opened",
        "seed": 78160,
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
            ledger = self._write(Path(directory), [{"seed": 78160}])
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
            ledger = self._write(Path(directory), [{**self.OPENED, "seed": 78151}])
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
        gate = BENCH.pilot_gate(synthetic_events(0.6), "statechain_clean")
        self.assertTrue(gate["passes_pilot_gate"])

    def test_tie_with_parent_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.5), "statechain_clean")
        self.assertTrue(gate["strictly_beats_base_aggregate"])
        self.assertTrue(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["strictly_beats_immediate_parent_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])

    def test_below_replay_control_fails(self) -> None:
        gate = BENCH.pilot_gate(synthetic_events(0.4), "statechain_clean")
        self.assertFalse(gate["strictly_beats_replay_control_aggregate"])
        self.assertFalse(gate["passes_pilot_gate"])


class GoalGateTests(unittest.TestCase):
    def test_rites_only_flip_is_the_reachable_maximum_and_still_fails(self) -> None:
        # menders is 0 for every believable arm, so 9/10 is the ceiling: a
        # rites flip alone gives nine strict wins and the goal gate stays
        # failed — recorded, never part of the pilot pass.
        events = synthetic_events(candidate_overrides={"rites": 0.1, "menders": 0.0})
        candidate = BENCH.goal_gate_reading(events)["per_arm"]["statechain_clean"]
        self.assertEqual(candidate["strict_wins"], 9)
        self.assertIn("rites", candidate["wins"])
        self.assertEqual(candidate["ties"], ["menders"])
        self.assertEqual(candidate["losses"], [])
        self.assertFalse(candidate["goal_gate_pass"])

    def test_double_tie_flip_passes_the_goal_gate(self) -> None:
        events = synthetic_events(
            candidate_overrides={"menders": 0.1, "rites": 0.1}
        )
        reading = BENCH.goal_gate_reading(events)
        candidate = reading["per_arm"]["statechain_clean"]
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
        candidate = BENCH.goal_gate_reading(events)["per_arm"]["statechain_clean"]
        self.assertIn("warren", candidate["losses"])
        self.assertFalse(candidate["goal_gate_pass"])

    def test_reading_carries_the_frozen_power_statement(self) -> None:
        reading = BENCH.goal_gate_reading(synthetic_events())
        self.assertFalse(reading["included_in_pilot_gate"])
        self.assertTrue(reading["recorded_either_way"])
        self.assertIn("9/10", reading["power_statement"])
        self.assertIn("menders", reading["power_statement"])
        self.assertIn("rites", reading["power_statement"])
        self.assertIn("CLEAN LINEAGE", reading["power_statement"])
        self.assertIn("confirmation cell", reading["power_statement"])


class RitesConversionTests(unittest.TestCase):
    def test_conversion_reading_on_clean_lineage(self) -> None:
        events = synthetic_events(candidate_overrides={"rites": 0.3})
        events["zero_root_parent"]["per_family"]["rites"] = 0.1
        events["replay_ctl4"]["per_family"]["rites"] = 0.1
        reading = BENCH.rites_conversion_reading(events, "statechain_clean")
        self.assertEqual(reading["family"], "rites")
        self.assertTrue(reading["converts_on_clean_lineage"])
        self.assertAlmostEqual(reading["candidate_minus_parent"], 0.2)
        self.assertAlmostEqual(reading["candidate_minus_replay"], 0.2)
        self.assertFalse(reading["included_in_pilot_gate"])

    def test_conversion_requires_strict_wins_over_both_controls(self) -> None:
        events = synthetic_events(candidate_overrides={"rites": 0.1})
        events["zero_root_parent"]["per_family"]["rites"] = 0.1
        events["replay_ctl4"]["per_family"]["rites"] = 0.0
        reading = BENCH.rites_conversion_reading(events, "statechain_clean")
        self.assertFalse(reading["converts_on_clean_lineage"])


class FrozenBenchmarkConstantsTests(unittest.TestCase):
    def test_event_identity(self) -> None:
        self.assertEqual(BENCH.FROZEN_NAME, "pilot")
        self.assertEqual(BENCH.FROZEN_TIER, "medium")
        self.assertEqual(BENCH.FROZEN_THINK_BUDGET, 1024)
        self.assertEqual(BENCH.FROZEN_SEED, 78160)
        self.assertEqual(
            BENCH.MODEL_ORDER,
            ("base", "zero_root_parent", "replay_ctl4", "statechain_clean"),
        )
        self.assertEqual(BENCH.FROZEN_CANDIDATES, ("statechain_clean",))

    def test_inherited_pins_are_filled_and_trained_pins_fail_closed(self) -> None:
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["base"])
        self.assertIsNotNone(BENCH.FROZEN_TREE_SHA256["zero_root_parent"])
        self.assertIsNotNone(BENCH.FROZEN_WEIGHTS_SHA256["base"])
        with self.assertRaisesRegex(ValueError, "TODO-PIN"):
            BENCH.require_pin(None, "FROZEN_TREE_SHA256['replay_ctl4']")
        for label in ("replay_ctl4", "statechain_clean"):
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
                BENCH.FROZEN_TREE_SHA256["replay_ctl4"],
                BENCH.FROZEN_TREE_SHA256["statechain_clean"],
                BENCH.FROZEN_WEIGHTS_SHA256["replay_ctl4"],
                BENCH.FROZEN_WEIGHTS_SHA256["statechain_clean"],
                BENCH.REPLAY_CTL4_MERGE_RECEIPT_SHA256,
                BENCH.STATECHAIN_CLEAN_MERGE_RECEIPT_SHA256,
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
                "seed": 78160,
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
