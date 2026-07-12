from __future__ import annotations

import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from task_data import SPLIT_SPECS, audit_splits, generate_split, load_family  # noqa: E402


class ProceduralSplitTests(unittest.TestCase):
    def test_frozen_counts_and_cross_split_firewall(self) -> None:
        splits = {name: generate_split(name) for name in SPLIT_SPECS}
        audit = audit_splits(splits)
        self.assertTrue(audit["passed"])
        self.assertEqual(
            audit["split_counts"],
            {
                "termination_pilot": 27,
                "calibration": 135,
                "train": 1080,
                "iid_eval": 540,
                "hard_eval": 180,
                "held_family_eval": 180,
                "rendering_eval": 80,
            },
        )

    def test_train_generation_does_not_import_heldout_registry(self) -> None:
        for name in list(sys.modules):
            if name.startswith("gym.heldout_families"):
                del sys.modules[name]
        generate_split("calibration")
        self.assertNotIn("gym.heldout_families", sys.modules)

    def test_all_registered_variants_pass_the_family_verifier(self) -> None:
        rows = generate_split("calibration")
        for row in rows:
            module = load_family(row["family"])
            self.assertEqual(
                module.score_atom(row, f"ANSWER: {row['canonical_answer']}"),
                1.0,
                row["id"],
            )
            for variant in row["answer_variants"]:
                self.assertEqual(
                    module.score_atom(row, f"ANSWER: {variant}"), 1.0, row["id"]
                )

    def test_stallwright_is_excluded_from_confirmatory_potential(self) -> None:
        rows = generate_split("rendering_eval")
        stallwright = [row for row in rows if row["family"] == "stallwright"]
        self.assertTrue(stallwright)
        self.assertTrue(all(not row["potential_scorable"] for row in stallwright))
        self.assertTrue(
            all("combinatorial" in row["potential_exclusion_reason"] for row in stallwright)
        )


if __name__ == "__main__":
    unittest.main()
