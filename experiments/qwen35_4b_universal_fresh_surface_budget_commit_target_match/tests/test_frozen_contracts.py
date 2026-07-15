import importlib.util
import json
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


class FrozenContractTests(unittest.TestCase):
    def test_stream_receipt_matches_frozen_exposure(self) -> None:
        receipt = json.loads(
            (EXP / "data" / "stream_token_receipt.json").read_text(encoding="utf-8")
        )
        for pair, deltas in receipt["deltas"].items():
            for axis in ("forward", "nonzero_target", "absolute_loss_mass_x5"):
                self.assertEqual(deltas[axis], 0, f"{pair} differs on {axis}")
        for name, entry in receipt["files"].items():
            self.assertEqual(entry["rows"], 1520, name)
            self.assertEqual(entry["skipped_rows"], 0, name)
            self.assertLessEqual(entry["max_sequence_tokens"], 4096, name)

    def test_stream_manifest_training_geometry(self) -> None:
        manifest = json.loads(
            (EXP / "data" / "stream_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["outcome"], "PASS_EXPOSURE_MATCH")
        training = manifest["training"]
        self.assertEqual(training["rows_per_arm"], 1520)
        self.assertEqual(training["optimizer_steps"], 190)
        self.assertEqual(training["seed"], 51)
        self.assertEqual(training["learning_rate"], 1e-5)
        self.assertFalse(training["authorized"])
        selection = manifest["selection"]
        self.assertEqual(selection["shared_replay_rows"], 1280)
        self.assertEqual(selection["shared_position_aligned_rows"], 1280)
        overlap = (
            set(selection["replay_core_source_indices"])
            & (
                set(selection["designed_replay_filler_source_indices"])
                | set(selection["budget_replay_filler_source_indices"])
                | set(selection["replay_control_source_indices"])
            )
        )
        self.assertFalse(overlap)

    def test_local_gate_seed_and_bars_are_frozen(self) -> None:
        check = load_module("contracts_check_local", "check_local.py")
        self.assertEqual(check.SEED, 88013)
        self.assertEqual(check.ROWS, 104)
        self.assertEqual(check.PER_KIND, 8)
        self.assertEqual(
            check.ARMS,
            ("replay_after_close_parent", "replay_repeat", "designed_fresh", "budget_commit"),
        )
        self.assertIn("BUDGET", check.ABSTENTION_ANSWERS)

    def test_benchmark_event_seed_is_frozen(self) -> None:
        bench = load_module("contracts_run_benchmark", "run_benchmark.py")
        self.assertEqual(bench.FROZEN_SEED, 78143)

    def test_gate_input_has_no_oracle_fields(self) -> None:
        for line in (EXP / "data" / "local_input_seed88013.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.assertEqual(set(row), {"id", "messages", "meta"})
            raw = json.dumps(row)
            for banned in ('"answer"', '"think"', '"_audit"', '"truth_valid"'):
                self.assertNotIn(banned, raw)


if __name__ == "__main__":
    unittest.main()
