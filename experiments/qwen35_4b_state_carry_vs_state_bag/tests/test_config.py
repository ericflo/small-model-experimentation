from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MODEL_ID, MODEL_REVISION, load_config, validate_config


class ConfigContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def test_only_permitted_model_revision_and_backend(self) -> None:
        self.assertEqual(self.config["model"]["id"], MODEL_ID)
        self.assertEqual(self.config["model"]["revision"], MODEL_REVISION)
        self.assertEqual(self.config["model"]["backend"], "transformers")
        for key, value in (
            ("id", "another/model"),
            ("revision", "main"),
            ("backend", "vllm"),
        ):
            changed = copy.deepcopy(self.config)
            changed["model"][key] = value
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_loop_is_two_complete_qwen_motifs(self) -> None:
        architecture = self.config["architecture"]
        self.assertEqual((architecture["loop_start"], architecture["loop_end"]), (12, 20))
        self.assertEqual(architecture["loop_end"] - architecture["loop_start"], 8)
        self.assertEqual(len(architecture["expected_layer_pattern"]), 4)

    def test_train_and_extrapolation_depths_do_not_overlap(self) -> None:
        substrate = self.config["substrate"]
        self.assertLess(max(substrate["train_depths"]), min(substrate["extrapolation_depths"]))

    def test_smoke_inheritance_preserves_scientific_contract(self) -> None:
        smoke = load_config(ROOT / "configs" / "smoke.yaml")
        self.assertEqual(smoke["model"], self.config["model"])
        self.assertEqual(smoke["architecture"], self.config["architecture"])
        self.assertLess(smoke["substrate"]["train_examples"], self.config["substrate"]["train_examples"])


if __name__ == "__main__":
    unittest.main()
