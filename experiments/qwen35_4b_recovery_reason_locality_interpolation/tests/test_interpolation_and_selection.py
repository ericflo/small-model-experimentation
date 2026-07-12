from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path

import torch
import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"local_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


interpolate = load_script("interpolate_adapters")
locality = load_script("audit_locality_ladder")
selection = load_script("select_interpolation")
analysis = load_script("analyze_primary")


class InterpolationAndSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_interpolation_endpoints_and_midpoint_are_exact(self) -> None:
        action = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        reason = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(torch.equal(interpolate.interpolate_delta(action, reason, 0.0), action))
        self.assertTrue(torch.equal(interpolate.interpolate_delta(action, reason, 1.0), reason))
        self.assertTrue(
            torch.equal(
                interpolate.interpolate_delta(action, reason, 0.5),
                torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
            )
        )
        with self.assertRaises(ValueError):
            interpolate.interpolate_delta(action, reason, 1.1)

    def test_ladder_and_model_are_frozen(self) -> None:
        self.assertEqual(self.cfg["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(
            self.cfg["interpolation"]["candidate_lambdas"], [0.10, 0.18, 0.24, 0.30]
        )
        self.assertEqual(
            [selection.candidate_lambda(f"reason_mix_{round(x * 100):03d}") for x in [0.10, 0.18, 0.24, 0.30]],
            [0.10, 0.18, 0.24, 0.30],
        )

    def test_locality_blocks_are_frozen_disjoint_and_nonbenchmark(self) -> None:
        blocks = []
        for key in ("screen_contexts", "confirm_contexts"):
            path = ROOT / self.cfg["locality"][key]
            payload = json.loads(path.read_text())
            self.assertEqual(payload["count"], 48)
            self.assertEqual(len(payload["contexts"]), 48)
            blocks.append({row["content_sha256"] for row in payload["contexts"]})
        self.assertFalse(blocks[0] & blocks[1])
        run_text = (EXP / "scripts" / "run.py").read_text()
        self.assertNotIn("benchmarks/", run_text)
        self.assertNotIn("benchmarks.", run_text)

    def test_entropy_and_varentropy_are_finite(self) -> None:
        entropy, varentropy = locality.uncertainty(torch.tensor([0.2, -0.1, 1.3]))
        self.assertTrue(torch.isfinite(torch.tensor(entropy)))
        self.assertTrue(torch.isfinite(torch.tensor(varentropy)))
        self.assertGreater(entropy, 0.0)
        self.assertGreaterEqual(varentropy, 0.0)

    def test_selection_score_prefers_validity_after_success_and_worst_case(self) -> None:
        def payload(invalid: float) -> dict:
            return {
                "aggregate": {
                    "success": 0.8,
                    "invalid_action_rate_per_turn": invalid,
                    "per_scenario": {
                        "failed_test": {
                            "success": 0.8,
                            "immediate_transition_rate": 0.8,
                            "changed_patch_within_two": 0.8,
                        },
                        "rejected_patch": {
                            "success": 0.8,
                            "immediate_transition_rate": 0.8,
                            "changed_patch_within_two": None,
                        },
                    },
                }
            }

        self.assertGreater(
            selection.score(payload(0.02), "reason_mix_018"),
            selection.score(payload(0.10), "reason_mix_010"),
        )

    def test_paired_bootstrap_is_casewise(self) -> None:
        left = {
            "a": {"success": True},
            "b": {"success": False},
            "c": {"success": True},
        }
        right = {
            "a": {"success": False},
            "b": {"success": False},
            "c": {"success": True},
        }
        result = analysis.paired_delta(left, right, seed=7)
        self.assertEqual(result["left_only"], 1)
        self.assertEqual(result["right_only"], 0)
        self.assertAlmostEqual(result["delta"], 1 / 3)


if __name__ == "__main__":
    unittest.main()
