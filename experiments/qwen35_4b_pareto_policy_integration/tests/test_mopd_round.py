from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from train_mopd_round import _training_units  # noqa: E402


def sample(index: int, stratum: str) -> dict:
    return {
        "id": f"{stratum}-{index}",
        "meta": {"stratum": stratum},
        "policy_mask": torch.ones(600, dtype=torch.bool),
    }


class MopdRoundTests(unittest.TestCase):
    def test_units_honor_retention_fraction_without_reusing_target_chunk(self):
        samples = [sample(index, "quick") for index in range(96)] + [
            sample(index, "deep") for index in range(96)
        ]
        units = _training_units(samples, quick_fraction=0.25, micro_steps=160, seed=42)
        self.assertEqual(len(units), 160)
        self.assertEqual(
            sum(unit["sample"]["meta"]["stratum"] == "quick" for unit in units), 40
        )
        self.assertEqual(
            sum(unit["sample"]["meta"]["stratum"] == "deep" for unit in units), 120
        )
        identities = {(unit["sample"]["id"], unit["chunk"]) for unit in units}
        self.assertEqual(len(identities), len(units))


if __name__ == "__main__":
    unittest.main()
