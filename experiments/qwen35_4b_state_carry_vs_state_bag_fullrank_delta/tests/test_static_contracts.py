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

    def test_full_rank_delta_replaces_peft_and_g4_is_deferred(self) -> None:
        runner = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        model = (ROOT / "src" / "state_loop_model.py").read_text(encoding="utf-8")
        analysis = (ROOT / "src" / "analysis.py").read_text(encoding="utf-8")
        cli = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")
        combined = runner + model + cli
        self.assertNotIn("from peft", combined)
        self.assertNotIn("import peft", combined)
        self.assertNotIn('"sample-more"', cli)
        self.assertNotIn("def _sample_more", analysis)
        self.assertNotIn("def _deployment_comparison", analysis)
        self.assertIn("class FullRankDeltaBank", model)
        self.assertIn("dtype=torch.float32", model)
        self.assertIn("nn.init.zeros_", model)
        self.assertIn("for step in range(2, k + 1)", model)
        self.assertIn("deltas_enabled=True", model)

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
            "expected_seed",
            "expected_step",
            "expected_phase",
        ):
            self.assertIn(contract, source)

    def test_g0_checks_both_edges_step_gradients_and_worst_k(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn('for arm in ("carry", "bag")', source)
        self.assertIn('("delta", "state", "sufficiency", "step")', source)
        self.assertIn("k1_batch = _encode_row", source)
        self.assertIn("_forward(wrapper, k1_batch, k=1", source)
        self.assertIn("bag_k1_delta_calls", source)
        self.assertIn("worst_k4_batch = _encode_row", source)
        self.assertIn("_forward(wrapper, worst_k4_batch, k=4, mode=arm)", source)
        self.assertIn('"worst_k4_prompt_tokens"', source)
        self.assertIn('"k1_carry_bag_max_logit_abs_error"', source)
        self.assertIn('"worst_k_finite"', source)
        self.assertIn("optimizer.step()", source)
        self.assertIn('"peak_reserved_headroom_gib"', source)
        self.assertIn('"recurrent_logit_max_abs_error"', source)
        self.assertIn("wrapper.load_extra_state_dict", source)
        self.assertIn('config["paths"]["large_artifacts_dir"]', source)

    def test_counterfactual_tuple_and_checkpoint_validation_regressions(self) -> None:
        source = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        b_to_a = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Tuple)
            and node.elts
            and isinstance(node.elts[0], ast.Constant)
            and node.elts[0].value == "b_to_a"
        ]
        self.assertEqual(len(b_to_a), 1)
        self.assertEqual(
            [
                element.value if isinstance(element, ast.Constant) else element.id
                for element in b_to_a[0].elts
            ],
            [
                "b_to_a",
                "second",
                "first",
                "second_batch",
                "first_batch",
                "second_output",
                "first_output",
            ],
        )
        self.assertEqual(
            source.count('raise RuntimeError("checkpoint full-rank parameter-count receipt mismatch")'),
            1,
        )

    def test_parent_data_parity_is_generated_and_training_gated(self) -> None:
        pipeline = (ROOT / "src" / "data_pipeline.py").read_text(encoding="utf-8")
        runner = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("canonical_rows_receipt", pipeline)
        self.assertIn("PARENT_DATA_PARITY_PASS", pipeline)
        self.assertIn("regenerated rows differ from available parent artifacts", pipeline)
        self.assertIn("validate_parent_data_parity(config, path, manifest)", runner)

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
        self.assertIn("--trigger-receipt", source)
        self.assertNotIn("--mechanism-receipt", source)
        self.assertIn("--seed is required", source)


if __name__ == "__main__":
    unittest.main()
