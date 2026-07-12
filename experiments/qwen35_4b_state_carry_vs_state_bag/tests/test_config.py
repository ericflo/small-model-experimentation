from __future__ import annotations

import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_CONTRACT_FILES,
    SOURCE_CONTRACT_VERSION,
    is_confirmatory_config,
    load_config,
    require_confirmatory_config,
    resolved_config_receipt,
    source_contract_sha256,
    validate_config,
)


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
        self.assertTrue(is_confirmatory_config(self.config))
        self.assertFalse(is_confirmatory_config(smoke))
        require_confirmatory_config(self.config)
        with self.assertRaises(RuntimeError):
            require_confirmatory_config(smoke)

    def test_pilot_and_full_training_seeds_are_frozen_and_disjoint(self) -> None:
        self.assertEqual(self.config["training"]["pilot_seed"], 7401)
        self.assertEqual(self.config["training"]["train_seeds"], [7411, 7412, 7413])
        self.assertNotIn(
            self.config["training"]["pilot_seed"],
            self.config["training"]["train_seeds"],
        )
        for pilot_seed, train_seeds in (
            (7411, [7411, 7412, 7413]),
            (7401, [7411, 7412, 9999]),
        ):
            changed = copy.deepcopy(self.config)
            changed["training"]["pilot_seed"] = pilot_seed
            changed["training"]["train_seeds"] = train_seeds
            with self.assertRaises(ValueError):
                validate_config(changed)

    def test_only_continuous_semantic_echo_is_registered(self) -> None:
        changed = copy.deepcopy(self.config)
        changed["architecture"]["semantic_echo"]["mode"] = "mixed"
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_data_split_seeds_are_explicit_and_pairwise_distinct(self) -> None:
        seeds = self.config["substrate"]["seeds"]
        self.assertIn("joint", seeds)
        self.assertIn("pilot_depth", seeds)
        self.assertIn("pilot_joint", seeds)
        self.assertIn("pilot_counterfactual", seeds)
        self.assertIn("pilot_validation", seeds)
        self.assertEqual(len(seeds), len(set(seeds.values())))
        changed = copy.deepcopy(self.config)
        changed["substrate"]["seeds"]["pilot_joint"] = seeds["pilot_depth"]
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_revised_gates_and_sample_more_allocation_are_frozen(self) -> None:
        self.assertNotIn("min_state_node_accuracy", self.config["gates"])
        self.assertEqual(self.config["training"]["save_every_steps"], 1500)
        expected_gates = {
            "min_state_joint_accuracy": 0.40,
            "min_edge_cut_gain": 0.0,
            "min_joint_holdout_carry_minus_bag": 0.0,
            "min_carry_answer_mode_rate": 0.95,
            "min_sample_more_parse_rate": 0.95,
            "max_sample_more_cap_contact_rate": 0.05,
        }
        for key, value in expected_gates.items():
            self.assertEqual(self.config["gates"][key], value)
        changed = copy.deepcopy(self.config)
        changed["gates"]["min_edge_cut_gain"] = -0.01
        with self.assertRaises(ValueError):
            validate_config(changed)

        sample_more = self.config["evaluation"]["sample_more"]
        self.assertEqual(sample_more["desired_tokens_per_transition"], 24)
        self.assertEqual(sample_more["fixed_overhead_tokens"], 32)
        self.assertEqual(sample_more["max_new_tokens"], 512)
        self.assertEqual(sample_more["max_samples"], 8)
        changed = copy.deepcopy(self.config)
        changed["evaluation"]["sample_more"]["temperature"] = 0.7
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_source_contract_is_versioned_and_sensitive_to_runtime_source(self) -> None:
        digest = source_contract_sha256()
        self.assertEqual(len(digest), 64)
        receipt = resolved_config_receipt(self.config)
        self.assertEqual(receipt["source_contract_version"], SOURCE_CONTRACT_VERSION)
        self.assertEqual(receipt["source_contract_sha256"], digest)
        with tempfile.TemporaryDirectory() as directory:
            copy_root = Path(directory)
            for relative_path in SOURCE_CONTRACT_FILES:
                source = ROOT / relative_path
                destination = copy_root / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
            self.assertEqual(source_contract_sha256(copy_root), digest)
            changed = copy_root / "src" / "substrate.py"
            changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertNotEqual(source_contract_sha256(copy_root), digest)


if __name__ == "__main__":
    unittest.main()
