from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from train_mopd_round import (  # noqa: E402
    _matched_loss_scale,
    _target_assignments,
    _training_units,
)


def sample(index: int, role: str, teacher: str = "quick", *, kind="atom", level=1) -> dict:
    targets = {"soup": {}}
    if role == "capability":
        targets.update({"quick": {}, "deep": {}})
    return {
        "id": f"{role}-{index}",
        "meta": {
            "role": role,
            "primary_teacher": teacher if role == "capability" else "soup",
            "kind": kind,
            "level": level,
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
        samples = [sample(i, "capability", "quick") for i in range(3)]
        samples += [sample(i + 3, "capability", "deep") for i in range(3)]
        samples += [sample(i + 6, "anchor") for i in range(2)]
        assignments = _target_assignments(samples, "primary", seed=1)
        self.assertEqual(sum(value == "quick" for value in assignments.values()), 3)
        self.assertEqual(sum(value == "deep" for value in assignments.values()), 3)
        self.assertEqual(sum(value == "soup" for value in assignments.values()), 2)
        units = _training_units(samples, assignments, seed=2)
        self.assertEqual(len(units), len(samples))
        self.assertEqual(len({unit["sample"]["id"] for unit in units}), len(samples))

    def test_shuffled_preserves_teacher_counts_but_breaks_mapping(self):
        samples = [
            sample(i, "capability", "quick", kind="atom" if i < 6 else "episode")
            for i in range(10)
        ]
        samples += [
            sample(i + 10, "capability", "deep", kind="atom" if i < 4 else "episode")
            for i in range(10)
        ]
        assignments = _target_assignments(samples, "shuffled", seed=54)
        self.assertEqual(sum(value == "quick" for value in assignments.values()), 10)
        self.assertEqual(sum(value == "deep" for value in assignments.values()), 10)
        original = {value["id"]: value["meta"]["primary_teacher"] for value in samples}
        self.assertNotEqual(assignments, original)
        for kind in ("atom", "episode"):
            rows = [value for value in samples if value["meta"]["kind"] == kind]
            self.assertEqual(
                sum(assignments[value["id"]] == "quick" for value in rows),
                sum(value["meta"]["primary_teacher"] == "quick" for value in rows),
            )

    def test_shuffled_fails_if_kind_quota_makes_breaking_impossible(self):
        samples = [sample(i, "capability", "quick", kind="atom") for i in range(3)]
        samples += [sample(i + 3, "capability", "deep", kind="episode") for i in range(3)]
        with self.assertRaisesRegex(ValueError, "cannot break routing"):
            _target_assignments(samples, "shuffled", seed=54)

    def test_locality_subset_preserves_registered_mixture(self):
        samples = [sample(i, "capability", "quick") for i in range(30)]
        samples += [sample(i + 30, "capability", "deep") for i in range(30)]
        samples += [sample(i + 60, "anchor") for i in range(20)]
        assignments = _target_assignments(samples, "primary", seed=1)
        units = _training_units(samples, assignments, seed=42, required=20)
        counts = {
            target: sum(unit["target"] == target for unit in units)
            for target in ("quick", "deep", "soup")
        }
        self.assertEqual(counts, {"quick": 8, "deep": 7, "soup": 5})
        self.assertEqual(len({unit["sample"]["id"] for unit in units}), 20)

    def test_coarse_and_fixed_deep_controls(self):
        samples = [
            sample(0, "capability", "deep", kind="atom", level=1),
            sample(1, "capability", "quick", kind="episode", level=2),
            sample(2, "anchor"),
        ]
        coarse = _target_assignments(samples, "coarse", seed=1)
        self.assertEqual(coarse, {"capability-0": "quick", "capability-1": "deep", "anchor-2": "soup"})
        fixed = _target_assignments(samples, "fixed_deep", seed=1)
        self.assertEqual(fixed, {"capability-0": "deep", "capability-1": "deep", "anchor-2": "soup"})


if __name__ == "__main__":
    unittest.main()
