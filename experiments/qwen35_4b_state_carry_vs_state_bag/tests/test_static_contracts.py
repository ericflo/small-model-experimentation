from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StaticContractTests(unittest.TestCase):
    def test_source_never_imports_benchmarks(self) -> None:
        for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")]:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                self.assertFalse(any(name == "benchmarks" or name.startswith("benchmarks.") for name in names))

    def test_no_result_bearing_vllm_runner_remains(self) -> None:
        self.assertFalse((ROOT / "src" / "vllm_runner.py").exists())
        self.assertFalse((ROOT / "tests" / "test_vllm_runner.py").exists())

    def test_python_sources_compile_without_gpu_dependencies(self) -> None:
        for path in [*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")]:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_sample_more_enforces_compute_and_natural_close(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("sample_budget > recurrent_budget", source)
        self.assertIn('if "</think>" not in text', source)
        self.assertIn('config["evaluation"]["sample_more"]', source)
        for field in (
            '"continuation_token_ids"',
            '"decoded_samples"',
            '"natural_close"',
            '"cap_contact"',
        ):
            self.assertIn(field, source)

    def test_gpu_loader_fails_on_transformers_drift(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("Transformers drift", source)
        self.assertIn("loaded model commit", source)

    def test_model_stages_are_phase_and_source_gated(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        for contract in (
            "source_contract_sha256",
            "requirements_training_lock_sha256",
            "checkpoint_identity_sha256",
            "training_order_sha256",
            "MODEL_SMOKE_PASS",
            "PILOT_PROMOTION_READY",
            "MECHANISTIC_DEPTH_POSITIVE",
            "expected_seed",
            "expected_step",
            "expected_phase",
        ):
            self.assertIn(contract, source)

    def test_g0_checks_both_edges_step_gradients_and_worst_k(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn('for arm in ("carry", "bag")', source)
        self.assertIn('("lora", "state", "sufficiency", "step")', source)
        self.assertIn('"k1_carry_bag_max_logit_abs_error"', source)
        self.assertIn('"worst_k_finite"', source)

    def test_pilot_and_causal_outputs_are_firewalled(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        for split in ("pilot_validation", "pilot_depth", "pilot_joint", "pilot_counterfactual"):
            self.assertIn(split, source)
        self.assertIn('"edge_cut_primary_only"', source)
        self.assertIn('"a_to_b"', source)
        self.assertIn('"b_to_a"', source)
        self.assertIn('"geometry_equal"', source)
        self.assertIn('"counterfactual_swap_row_file_sha256"', source)

    def test_cli_passes_explicit_seed_and_gate_receipts(self) -> None:
        source = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")
        self.assertIn("expected_seed=args.seed", source)
        self.assertIn("--model-smoke-receipt", source)
        self.assertIn("--promotion-receipt", source)
        self.assertIn("--mechanism-receipt", source)
        self.assertIn("--seed is required", source)


if __name__ == "__main__":
    unittest.main()
