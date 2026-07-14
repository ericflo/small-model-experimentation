from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
MODULE_PATH = EXP / "scripts" / "materialize_doses.py"
SPEC = importlib.util.spec_from_file_location("mid_density_materialize", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class DoseConstructionTests(unittest.TestCase):
    def test_checked_in_bytes_are_deterministic(self) -> None:
        outputs_a, manifest_a = MODULE.build_outputs()
        outputs_b, manifest_b = MODULE.build_outputs()
        self.assertEqual(outputs_a, outputs_b)
        self.assertEqual(manifest_a, manifest_b)
        for name, value in outputs_a.items():
            self.assertEqual((EXP / "data" / name).read_bytes(), value)

    def test_nested_slot_replacement_is_exact(self) -> None:
        outputs, manifest = MODULE.build_outputs()
        rows = {
            name: value.decode("utf-8").splitlines()
            for name, value in outputs.items()
        }
        self.assertTrue(manifest["selection"]["nested_slot_replacement"])
        self.assertEqual(len(rows["replay_repeat.jsonl"]), 1520)
        self.assertEqual(
            sum(a != b for a, b in zip(rows["replay_repeat.jsonl"], rows["designed160.jsonl"])),
            160,
        )
        self.assertEqual(
            sum(a != b for a, b in zip(rows["designed160.jsonl"], rows["designed240.jsonl"])),
            80,
        )
        self.assertEqual(
            sum(a != b for a, b in zip(rows["replay_repeat.jsonl"], rows["designed240.jsonl"])),
            240,
        )

    def test_every_designed_skill_is_present_at_both_doses(self) -> None:
        outputs, _ = MODULE.build_outputs()
        expected = {
            "u_abstain", "u_count", "u_execute", "u_induct", "u_optimize",
            "u_order", "u_probe", "u_repair", "u_route", "u_select",
            "u_state", "u_trace", "u_verify",
        }
        for name in ("designed160.jsonl", "designed240.jsonl"):
            rows = [json.loads(line) for line in outputs[name].decode("utf-8").splitlines()]
            present = {row["kind"] for row in rows if row.get("family") == "universal"}
            self.assertEqual(present, expected)

    def test_forward_token_exposure_is_exactly_equal(self) -> None:
        _, manifest = MODULE.build_outputs()
        tokens = manifest["selection"]["estimated_arm_forward_tokens"]
        self.assertEqual(set(tokens.values()), {1405510})
        receipt = json.loads((EXP / "data" / "dose_token_receipt.json").read_text())
        observed = {row["total_forward_tokens_per_epoch"] for row in receipt["files"]}
        self.assertEqual(observed, {1405510})
        self.assertEqual(receipt["skipped_rows"], 0)


if __name__ == "__main__":
    unittest.main()
