from __future__ import annotations

import ast
import copy
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    BACKEND,
    CONFIRMATORY_CONFIG_SHA256,
    EXPERIMENT_ID,
    MODEL_ID,
    MODEL_REVISION,
    SOURCE_CONTRACT_FILES,
    SOURCE_CONTRACT_VERSION,
    config_sha256,
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
        cls.default = load_config(ROOT / "configs" / "default.yaml")
        cls.smoke = load_config(ROOT / "configs" / "smoke.yaml")

    def test_exact_confirmatory_identity_model_and_digest(self) -> None:
        self.assertEqual(EXPERIMENT_ID, "qwen35_4b_state_formation_capacity_adjudication")
        self.assertEqual(MODEL_ID, "Qwen/Qwen3.5-4B")
        self.assertEqual(MODEL_REVISION, "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
        self.assertEqual(BACKEND, "transformers")
        self.assertEqual(SOURCE_CONTRACT_VERSION, 5)
        self.assertEqual(
            CONFIRMATORY_CONFIG_SHA256,
            "eeb4e828526f750dce1258bcc91d03114c80688d300112e03d18c9d911489393",
        )
        self.assertEqual(self.default["experiment_id"], EXPERIMENT_ID)
        self.assertEqual(self.default["model"]["id"], MODEL_ID)
        self.assertEqual(self.default["model"]["revision"], MODEL_REVISION)
        self.assertEqual(self.default["model"]["backend"], BACKEND)
        self.assertEqual(config_sha256(self.default), CONFIRMATORY_CONFIG_SHA256)
        self.assertEqual(self.default["paths"]["design_receipt"], "reports/design_receipt.json")
        self.assertTrue(is_confirmatory_config(self.default))
        require_confirmatory_config(self.default)

        for key, value in (("id", "forbidden/model"), ("revision", "main"), ("backend", "vllm")):
            changed = copy.deepcopy(self.default)
            changed["model"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_config(changed)

    def test_smoke_only_reduces_counts_and_steps(self) -> None:
        self.assertFalse(is_confirmatory_config(self.smoke))
        with self.assertRaises(RuntimeError):
            require_confirmatory_config(self.smoke)
        self.assertEqual(self.smoke["model"], self.default["model"])
        self.assertEqual(self.smoke["architecture"], self.default["architecture"])
        self.assertEqual(self.smoke["training"]["objectives"], self.default["training"]["objectives"])
        self.assertLess(self.smoke["training"]["train_steps"], self.default["training"]["train_steps"])
        self.assertLess(self.smoke["substrate"]["train_examples"], self.default["substrate"]["train_examples"])

    def test_matched_capacity_contract_is_exact(self) -> None:
        adaptation = self.default["architecture"]["adaptation"]
        self.assertEqual(set(adaptation), {"lora", "fullrank"})
        self.assertEqual(adaptation["lora"]["rank"], 32)
        self.assertEqual(adaptation["lora"]["expected_parameters"], 16_232_448)
        self.assertEqual(adaptation["fullrank"]["expected_parameters"], 892_272_640)
        self.assertEqual(adaptation["lora"]["expected_targets"], 62)
        self.assertEqual(self.default["architecture"]["max_recurrence"], 12)
        self.assertEqual(3 * adaptation["lora"]["expected_targets"], 186)
        self.assertEqual(11 * adaptation["lora"]["expected_targets"], 682)
        self.assertEqual(adaptation["lora"]["dropout"], adaptation["fullrank"]["dropout"])
        self.assertEqual(adaptation["lora"]["scale"], adaptation["fullrank"]["scale"])
        self.assertEqual(adaptation["lora"]["expected_targets"], adaptation["fullrank"]["expected_targets"])
        self.assertTrue(adaptation["lora"]["active_on_extra_calls_only"])
        self.assertTrue(adaptation["fullrank"]["active_on_extra_calls_only"])

        changed = copy.deepcopy(self.default)
        changed["architecture"]["adaptation"]["lora"]["dropout"] = 0.0
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_seven_fresh_splits_and_seeds_are_frozen(self) -> None:
        seeds = self.default["substrate"]["seeds"]
        self.assertEqual(
            seeds,
            {
                "train": 73301,
                "validation": 73302,
                "depth": 73303,
                "joint": 73304,
                "contrast_depth": 73305,
                "contrast_joint": 73306,
                "contrast_validation": 73307,
            },
        )
        self.assertEqual(len(seeds), len(set(seeds.values())))
        self.assertEqual(
            self.default["evaluation"]["trigger_splits"],
            ["validation", "depth_extrapolation", "joint_holdout"],
        )
        self.assertEqual(
            self.default["evaluation"]["sealed_contrast_splits"],
            ["contrast_validation", "contrast_depth", "contrast_joint"],
        )
        self.assertEqual(self.default["substrate"]["contrast_validation_examples"], 768)
        self.assertEqual(self.smoke["substrate"]["contrast_validation_examples"], 24)
        self.assertLess(
            max(self.default["substrate"]["train_depths"]),
            min(self.default["substrate"]["extrapolation_depths"]),
        )

        changed = copy.deepcopy(self.default)
        changed["substrate"]["seeds"]["contrast_joint"] = seeds["joint"]
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_joint_and_state_only_objectives_and_seeds_are_frozen(self) -> None:
        self.assertEqual(self.default["training"]["train_seeds"], [7411, 7412, 7413])
        self.assertEqual(
            self.default["training"]["objectives"],
            {
                "joint": {
                    "answer_loss_weight": 1.0,
                    "state_loss_weight": 0.5,
                    "fixed_point_loss_weight": 0.05,
                },
                "state_only": {
                    "answer_loss_weight": 0.0,
                    "state_loss_weight": 0.5,
                    "fixed_point_loss_weight": 0.05,
                },
            },
        )
        self.assertEqual(self.default["training"]["adaptation_gradient_clip"], 1.0)
        self.assertEqual(self.default["training"]["common_gradient_clip"], 1.0)
        self.assertEqual(
            self.default["training"]["positive_control"],
            {
                "rows": 48,
                "updates": 256,
                "seed": 73991,
                "depths": [2, 3, 4],
                "examples_per_cell": 2,
                "min_overfit_final_joint_accuracy": 0.95,
                "min_oracle_readout_accuracy": 0.99,
            },
        )

        changed = copy.deepcopy(self.default)
        changed["training"]["objectives"]["state_only"]["answer_loss_weight"] = 0.01
        with self.assertRaises(ValueError):
            validate_config(changed)

    def test_source_contract_is_complete_versioned_and_content_sensitive(self) -> None:
        self.assertEqual(
            SOURCE_CONTRACT_FILES,
            (
                "scripts/archive_failed_attempt.py",
                "scripts/run.py",
                "reports/implementation_review.md",
                "src/__init__.py",
                "src/adaptation.py",
                "src/analysis.py",
                "src/config.py",
                "src/data_pipeline.py",
                "src/design_boundary.py",
                "src/gpu_runner.py",
                "src/initialization.py",
                "src/mechanics.py",
                "src/optimizer_receipts.py",
                "src/state_loop_model.py",
                "src/substrate.py",
                "tests/__init__.py",
                "tests/test_archive_failed_attempt.py",
                "tests/test_analysis.py",
                "tests/test_config.py",
                "tests/test_data_parity.py",
                "tests/test_design_boundary.py",
                "tests/test_fullrank_delta.py",
                "tests/test_initialization.py",
                "tests/test_mechanics.py",
                "tests/test_objectives.py",
                "tests/test_optimizer_receipts.py",
                "tests/test_positive_control.py",
                "tests/test_receipt_contracts.py",
                "tests/test_static_contracts.py",
                "tests/test_substrate.py",
            ),
        )
        digest = source_contract_sha256()
        self.assertEqual(len(digest), 64)
        receipt = resolved_config_receipt(self.default)
        self.assertEqual(receipt["source_contract_version"], SOURCE_CONTRACT_VERSION)
        self.assertEqual(receipt["source_contract_sha256"], digest)
        self.assertEqual(receipt["capacities"], ["lora", "fullrank"])
        self.assertEqual(receipt["objectives"], ["joint", "state_only"])

        with tempfile.TemporaryDirectory() as directory:
            copy_root = Path(directory)
            for relative in SOURCE_CONTRACT_FILES:
                destination = copy_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            self.assertEqual(source_contract_sha256(copy_root), digest)
            changed = copy_root / "src" / "adaptation.py"
            changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
            self.assertNotEqual(source_contract_sha256(copy_root), digest)

    def test_every_runtime_local_import_is_source_bound(self) -> None:
        contracted = set(SOURCE_CONTRACT_FILES)
        for relative in SOURCE_CONTRACT_FILES:
            path = ROOT / relative
            if path.suffix != ".py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                candidates: set[str] = set()
                if isinstance(node, ast.ImportFrom):
                    if node.level and relative.startswith("src/") and node.module:
                        candidates.add(f"src/{node.module.replace('.', '/')}.py")
                    elif node.module == "src":
                        candidates.add("src/__init__.py")
                        for alias in node.names:
                            if (ROOT / "src" / f"{alias.name}.py").is_file():
                                candidates.add(f"src/{alias.name}.py")
                    elif node.module and node.module.startswith("src."):
                        candidates.add("src/__init__.py")
                        candidates.add(f"{node.module.replace('.', '/')}.py")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "src":
                            candidates.add("src/__init__.py")
                        elif alias.name.startswith("src."):
                            candidates.add("src/__init__.py")
                            candidates.add(f"{alias.name.replace('.', '/')}.py")
                for candidate in candidates:
                    if (ROOT / candidate).is_file():
                        self.assertIn(candidate, contracted)


if __name__ == "__main__":
    unittest.main()
