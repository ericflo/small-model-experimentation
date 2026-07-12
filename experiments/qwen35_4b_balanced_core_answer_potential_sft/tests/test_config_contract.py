from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]


class ConfigContractTests(unittest.TestCase):
    def test_balanced_candidate_count_and_no_512_cap(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        sampling = config["sampling"]
        self.assertEqual(
            config["splits"]["full_train_tasks"]
            * sampling["train_independent_n"],
            23040,
        )
        self.assertGreaterEqual(sampling["natural_close_allowance"], 12288)
        self.assertEqual(config["sft"]["max_length"], 16000)
        self.assertNotIn("train_branch_n", sampling)

    def test_complete_seed42_matrix_is_registered(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        self.assertEqual(
            config["sft"]["arms"],
            [
                "random_natural",
                "success_rft",
                "shortest_natural",
                "answer_potential",
                "joint_potential",
                "potential_shuffle",
            ],
        )

    def test_stage_a_counts_and_controls(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        self.assertEqual(len(config["splits"]["core_families"]), 3)
        self.assertEqual(config["splits"]["core_iid_tasks"], 180)
        self.assertEqual(config["splits"]["core_hard_tasks"], 60)
        self.assertEqual(config["splits"]["held_stage_a_tasks"], 60)
        self.assertTrue(config["scoring"]["canonical_only"])
        self.assertEqual(
            config["scoring"]["backend"],
            "transformers_bf16_sdpa_single_context",
        )

    def test_frozen_files_materialize_balanced_scopes(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        families = set(config["splits"]["core_families"])

        def rows(name: str):
            return [
                json.loads(line)
                for line in (EXP / "data" / "procedural" / f"{name}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

        train = [row for row in rows("train") if row["family"] in families]
        iid = [row for row in rows("iid_eval") if row["family"] in families]
        hard = [row for row in rows("hard_eval") if row["family"] in families]
        self.assertEqual(len(train), 360)
        self.assertEqual(len(iid), 180)
        self.assertEqual(len(hard), 60)

        held_counts = {}
        held_stage_a = []
        for row in rows("held_family_eval"):
            key = (row["family"], row["level"])
            ordinal = held_counts.get(key, 0)
            held_counts[key] = ordinal + 1
            if ordinal < config["splits"]["held_stage_a_per_family_level"]:
                held_stage_a.append(row)
        self.assertEqual(len(held_stage_a), 60)


if __name__ == "__main__":
    unittest.main()
