from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from train_mopd_round import (  # noqa: E402
    _arm_samples,
    _matched_loss_scale,
    _target_assignments,
    _training_units,
)


def sample(index: int, role: str, teacher: str = "deep", *, kind="atom", level=1) -> dict:
    targets = {"soup": {}}
    if role != "anchor":
        targets.update({"quick": {}, "deep": {}})
    return {
        "id": f"{role}-{index}",
        "meta": {
            "role": role,
            "primary_teacher": teacher if role == "capability" else "soup",
            "kind": kind,
            "level": level,
            "prompt_tokens_truncated": 0,
        },
        "targets": targets,
        "positions": torch.arange(3),
    }


class MopdRoundTests(unittest.TestCase):
    def test_control_loss_scale_matches_primary_initial_pressure(self):
        self.assertAlmostEqual(_matched_loss_scale(0.2, 0.05), 0.25)
        self.assertEqual(_matched_loss_scale(0.2, None), 1.0)
        for invalid in (0.0, -0.1, float("inf")):
            with self.assertRaises(ValueError):
                _matched_loss_scale(invalid, 0.1)
            with self.assertRaises(ValueError):
                _matched_loss_scale(0.1, invalid)

    def test_primary_and_anchor_assignments_are_exact(self):
        samples = [sample(i, "capability") for i in range(6)]
        samples += [sample(i + 6, "anchor") for i in range(2)]
        assignments = _target_assignments(samples, "primary", seed=1)
        self.assertEqual(sum(value == "quick" for value in assignments.values()), 0)
        self.assertEqual(sum(value == "deep" for value in assignments.values()), 6)
        self.assertEqual(sum(value == "soup" for value in assignments.values()), 2)
        units = _training_units(samples, assignments, seed=2)
        self.assertEqual(len(units), len(samples))
        self.assertEqual(len({unit["sample"]["id"] for unit in units}), len(samples))

    def test_non_advantage_route_uses_only_matched_controls_and_shared_anchors(self):
        samples = [sample(i, "capability") for i in range(3)]
        samples += [sample(i + 3, "route_control") for i in range(3)]
        samples += [sample(i + 6, "anchor") for i in range(2)]
        selected = _arm_samples(samples, "non_advantage_route")
        self.assertEqual(
            {row["meta"]["role"] for row in selected}, {"route_control", "anchor"}
        )
        assignments = _target_assignments(selected, "non_advantage_route", seed=64)
        self.assertEqual(sum(value == "deep" for value in assignments.values()), 3)
        self.assertEqual(sum(value == "soup" for value in assignments.values()), 2)

    def test_wrong_teacher_targets_quick_on_exact_primary_states(self):
        samples = [sample(i, "capability") for i in range(3)]
        samples += [sample(i + 3, "anchor") for i in range(2)]
        selected = _arm_samples(samples, "wrong_teacher")
        assignments = _target_assignments(selected, "wrong_teacher", seed=65)
        self.assertEqual(sum(value == "quick" for value in assignments.values()), 3)
        self.assertEqual(sum(value == "soup" for value in assignments.values()), 2)

    def test_selected_control_can_be_identified_as_prefix_truncated(self):
        from training_units import prompt_truncation_violations

        samples = [sample(0, "route_control"), sample(1, "anchor")]
        samples[0]["meta"]["prompt_tokens_truncated"] = 7
        selected = _arm_samples(samples, "non_advantage_route")
        self.assertEqual(
            prompt_truncation_violations(
                selected, required_roles={"route_control", "anchor"}
            )[0]["id"],
            "route_control-0",
        )

    def test_locality_subset_preserves_registered_mixture(self):
        samples = [sample(i, "capability") for i in range(60)]
        samples += [sample(i + 60, "anchor") for i in range(20)]
        assignments = _target_assignments(samples, "primary", seed=1)
        units = _training_units(samples, assignments, seed=42, required=20)
        counts = {
            target: sum(unit["target"] == target for unit in units)
            for target in ("quick", "deep", "soup")
        }
        self.assertEqual(counts, {"quick": 0, "deep": 15, "soup": 5})
        self.assertEqual(len({unit["sample"]["id"] for unit in units}), 20)

if __name__ == "__main__":
    unittest.main()
