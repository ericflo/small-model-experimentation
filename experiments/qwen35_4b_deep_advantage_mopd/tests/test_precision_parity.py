from __future__ import annotations

import json
import sys
import tempfile
import unittest
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from precision_parity import (  # noqa: E402
    canonical_tensor_sha256,
    endpoint_logit_metrics,
    mean_numeric_metrics,
    position_objective_metrics,
    probe_identity_sha256,
    replay_comparison,
    select_registered_probe_units,
    summarize_objective_rows,
    update_logit_metrics,
)


def _sample(sample_id: str, role: str, target: str, *, truncated: int = 0) -> dict:
    return {
        "id": sample_id,
        "meta": {"role": role, "prompt_tokens_truncated": truncated},
        "targets": {target: {"indices": torch.tensor([[0]]), "log_probs": torch.tensor([[0.0]])}},
    }


def _diagnostic_module():
    path = EXP / "scripts" / "diagnose_nf4_bf16_parity.py"
    spec = spec_from_file_location("diagnose_nf4_bf16_parity", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RegisteredProbeTests(unittest.TestCase):
    def test_selection_reconstructs_lexicographic_six_two_from_consumed_ledger(self) -> None:
        samples = [
            *[_sample(f"deep-{index:02d}", "capability", "deep") for index in range(8)],
            *[_sample(f"soup-{index:02d}", "anchor", "soup") for index in range(3)],
            _sample("control-00", "route_control", "deep"),
        ]
        ledger = [
            {
                "sample_id": sample["id"],
                "target": "deep" if sample["meta"]["role"] == "capability" else "soup",
                "role": sample["meta"]["role"],
            }
            for sample in reversed(samples[:-1])
        ]
        units = select_registered_probe_units(samples, ledger)
        self.assertEqual(
            [unit["sample"]["id"] for unit in units],
            [
                "deep-00", "deep-01", "deep-02", "deep-03", "deep-04", "deep-05",
                "soup-00", "soup-01",
            ],
        )
        self.assertEqual(len(probe_identity_sha256(units)), 64)

    def test_selection_rejects_shortened_probe_prefix(self) -> None:
        samples = [
            *[
                _sample(
                    f"deep-{index:02d}",
                    "capability",
                    "deep",
                    truncated=1 if index == 0 else 0,
                )
                for index in range(6)
            ],
            *[_sample(f"soup-{index:02d}", "anchor", "soup") for index in range(2)],
        ]
        ledger = [
            {
                "sample_id": sample["id"],
                "target": "deep" if sample["meta"]["role"] == "capability" else "soup",
                "role": sample["meta"]["role"],
            }
            for sample in samples
        ]
        with self.assertRaisesRegex(ValueError, "full prefix"):
            select_registered_probe_units(samples, ledger)


class LogitMetricTests(unittest.TestCase):
    def test_endpoint_metrics_are_softmax_offset_invariant(self) -> None:
        nf4 = torch.tensor([1.0, 0.2, -0.4, 2.0, -1.0])
        bf16 = nf4 + 17.0
        metrics = endpoint_logit_metrics(
            nf4, bf16, torch.tensor([3, 0]), top_k=2
        )
        self.assertLess(metrics["maximum_abs_centered_logit_error"], 2e-6)
        self.assertLess(metrics["total_variation"], 2e-6)
        self.assertLess(abs(metrics["jensen_shannon_divergence_nats"]), 2e-6)
        self.assertTrue(metrics["top1_agreement"])
        self.assertEqual(metrics["topk_overlap_fraction"], 1.0)

    def test_update_metrics_identical_movements_have_unit_cosine(self) -> None:
        before = torch.tensor([0.0, 1.0, 2.0, -2.0])
        after = torch.tensor([0.5, 0.5, 3.0, -2.5])
        metrics = update_logit_metrics(before, after, before + 8.0, after - 3.0)
        self.assertAlmostEqual(metrics["update_cosine_similarity"], 1.0, places=6)
        self.assertLess(metrics["maximum_abs_centered_update_error"], 2e-6)
        self.assertAlmostEqual(metrics["bf16_to_nf4_update_norm_ratio"], 1.0, places=6)
        self.assertTrue(metrics["update_cosine_defined"])
        self.assertTrue(metrics["update_norm_ratio_defined"])

    def test_update_metrics_mark_zero_norms_undefined(self) -> None:
        before = torch.tensor([0.0, 1.0, 2.0, -2.0])
        metrics = update_logit_metrics(before, before, before + 8.0, before + 8.0)
        self.assertTrue(metrics["nf4_update_degenerate"])
        self.assertTrue(metrics["bf16_update_degenerate"])
        self.assertFalse(metrics["update_cosine_defined"])
        self.assertFalse(metrics["update_norm_ratio_defined"])
        self.assertIsNone(metrics["update_cosine_similarity"])
        self.assertIsNone(metrics["bf16_to_nf4_update_norm_ratio"])

    def test_nullable_update_aggregation_preserves_defined_fraction(self) -> None:
        before = torch.tensor([0.0, 1.0, 2.0, -2.0])
        after = torch.tensor([0.5, 0.5, 3.0, -2.5])
        rows = [
            update_logit_metrics(before, before, before, before),
            update_logit_metrics(before, after, before + 8.0, after - 3.0),
        ]
        result = mean_numeric_metrics(
            rows,
            boolean_keys={
                "nf4_update_degenerate",
                "bf16_update_degenerate",
                "update_cosine_defined",
                "update_norm_ratio_defined",
            },
            nullable_keys={
                "update_cosine_similarity",
                "bf16_to_nf4_update_norm_ratio",
            },
        )
        self.assertEqual(result["update_cosine_similarity_defined_fraction"], 0.5)
        self.assertAlmostEqual(
            result["mean_update_cosine_similarity"], 1.0, places=6
        )

    def test_canonical_hash_ignores_source_dtype_and_layout(self) -> None:
        value = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        transposed_roundtrip = value.t().contiguous().t()
        self.assertEqual(
            canonical_tensor_sha256(value),
            canonical_tensor_sha256(transposed_roundtrip.float()),
        )


class ObjectiveSummaryTests(unittest.TestCase):
    def test_replay_tolerance_is_engineering_only_and_explicit(self) -> None:
        result = replay_comparison([1.000001, 2.0], [1.0, 2.0])
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["rows"]), 2)
        failed = replay_comparison([1.1], [1.0])
        self.assertFalse(failed["passed"])

    def test_objective_summary_preserves_equal_unit_weighting(self) -> None:
        rows = [
            {
                "objective": {
                    "nf4_before": 0.4,
                    "nf4_after": 0.2,
                    "bf16_before": 0.5,
                    "bf16_after": 0.25,
                }
            },
            {
                "objective": {
                    "nf4_before": 0.2,
                    "nf4_after": 0.1,
                    "bf16_before": 0.3,
                    "bf16_after": 0.15,
                }
            },
        ]
        result = summarize_objective_rows(rows)
        self.assertAlmostEqual(result["mean_nf4_objective_gain"], 0.15)
        self.assertAlmostEqual(result["mean_bf16_objective_gain"], 0.2)
        self.assertEqual(result["gain_sign_agreement_fraction"], 1.0)
        self.assertAlmostEqual(
            result["equal_unit_macro_mean"]["nf4_before"], 0.3
        )

    def test_objective_summary_reports_mean_absolute_gain_error_without_cancellation(self) -> None:
        rows = [
            {
                "objective": {
                    "nf4_before": 0.4,
                    "nf4_after": 0.2,
                    "bf16_before": 0.5,
                    "bf16_after": 0.2,
                }
            },
            {
                "objective": {
                    "nf4_before": 0.4,
                    "nf4_after": 0.2,
                    "bf16_before": 0.3,
                    "bf16_after": 0.2,
                }
            },
        ]
        result = summarize_objective_rows(rows)
        self.assertAlmostEqual(result["mean_bf16_minus_nf4_gain"], 0.0)
        self.assertAlmostEqual(result["mean_abs_bf16_nf4_gain_error"], 0.1)
        self.assertAlmostEqual(result["maximum_abs_bf16_nf4_gain_error"], 0.1)

    def test_position_metrics_do_not_allow_mean_cancellation(self) -> None:
        metrics = position_objective_metrics(
            torch.tensor([0.4, 0.2]),
            torch.tensor([0.2, 0.1]),
            torch.tensor([0.2, 0.4]),
            torch.tensor([0.1, 0.2]),
        )
        self.assertAlmostEqual(
            metrics["mean_abs_bf16_nf4_objective_error_before"], 0.2
        )
        self.assertAlmostEqual(metrics["gain_sign_agreement_fraction"], 1.0)

    def test_zero_gain_is_not_reported_as_positive_sign_agreement(self) -> None:
        rows = [
            {
                "objective": {
                    "nf4_before": 0.2,
                    "nf4_after": 0.2,
                    "bf16_before": 0.2,
                    "bf16_after": 0.1,
                }
            },
            {
                "objective": {
                    "nf4_before": 0.3,
                    "nf4_after": 0.2,
                    "bf16_before": 0.3,
                    "bf16_after": 0.2,
                }
            },
        ]
        result = summarize_objective_rows(rows)
        self.assertEqual(result["nf4_zero_gain_count"], 1)
        self.assertEqual(result["gain_sign_agreement_fraction"], 0.5)


class DiagnosticScriptContractTests(unittest.TestCase):
    def test_exclusive_receipt_writer_never_overwrites(self) -> None:
        module = _diagnostic_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "receipt.json"
            module._write_json_exclusive(path, {"first": True})
            with self.assertRaisesRegex(SystemExit, "refusing to overwrite"):
                module._write_json_exclusive(path, {"first": False})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_post_score_snapshot_detects_artifact_mutation(self) -> None:
        module = _diagnostic_module()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.bin"
            path.write_bytes(b"before")
            row = {
                "path": str(path),
                "sha256": module._sha256_bytes(b"before"),
                "bytes": len(b"before"),
            }
            snapshot = {
                "file_count": 1,
                "inventory_sha256": module._canonical_json_sha256([row]),
                "files": [row],
            }
            verified = module._verify_artifact_snapshot(snapshot)
            self.assertTrue(verified["verified_unchanged_after_scoring"])
            path.write_bytes(b"after")
            with self.assertRaisesRegex(SystemExit, "changed during parity scoring"):
                module._verify_artifact_snapshot(snapshot)

    def test_adapter_attachment_requires_exact_saved_loaded_tensors(self) -> None:
        module = _diagnostic_module()
        saved = {
            "base_model.model.layer.lora_A.weight": torch.tensor([[1.0, 2.0]]),
            "base_model.model.layer.lora_B.weight": torch.tensor([[3.0], [4.0]]),
        }
        expected = module._adapter_state_structure(saved)
        loaded = {key: value.clone() for key, value in saved.items()}
        result = module._validate_adapter_attachment(
            expected,
            saved,
            loaded,
            SimpleNamespace(missing_keys=["base.weight"], unexpected_keys=[]),
        )
        self.assertTrue(result["all_tensors_exact"])
        self.assertEqual(result["tensor_count"], 2)
        loaded["base_model.model.layer.lora_B.weight"][0, 0] += 1.0
        with self.assertRaisesRegex(SystemExit, "differs from saved"):
            module._validate_adapter_attachment(
                expected,
                saved,
                loaded,
                SimpleNamespace(missing_keys=[], unexpected_keys=[]),
            )

    def test_shard_index_is_hashed_and_must_cover_exact_weight_inventory(self) -> None:
        module = _diagnostic_module()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / "model.safetensors.index.json"
            index.write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "model.a": "model-00001-of-00002.safetensors",
                            "model.b": "model-00002-of-00002.safetensors",
                        }
                    }
                ),
                encoding="utf-8",
            )
            receipt = module._validated_shard_index(
                root,
                {
                    "model-00001-of-00002.safetensors",
                    "model-00002-of-00002.safetensors",
                },
            )
            self.assertIsNotNone(receipt)
            self.assertEqual(len(receipt["sha256"]), 64)
            with self.assertRaisesRegex(SystemExit, "inventory mismatch"):
                module._validated_shard_index(
                    root, {"model-00001-of-00002.safetensors"}
                )

    def test_cpu_fake_model_scores_every_registered_position(self) -> None:
        module = _diagnostic_module()

        class FakeModel:
            device = torch.device("cpu")

            def eval(self):
                return self

            def __call__(self, *, input_ids, logits_to_keep, **_kwargs):
                vocab = 5
                values = torch.arange(
                    logits_to_keep * vocab, dtype=torch.float32
                ).reshape(1, logits_to_keep, vocab)
                return SimpleNamespace(logits=values)

        sample = {
            "id": "deep-00",
            "meta": {"role": "capability", "prompt_tokens_truncated": 0},
            "prompt_ids": torch.tensor([9, 8]),
            "completion_ids": torch.tensor([7, 6, 5, 4]),
            "positions": torch.tensor([1, 2]),
            "targets": {
                "deep": {
                    "indices": torch.tensor([[4, 3], [4, 3]], dtype=torch.int32),
                    "log_probs": torch.tensor([[-0.1, -1.1], [-0.1, -1.1]]),
                }
            },
        }
        scored = module._score_model(
            FakeModel(), [{"sample": sample, "target": "deep"}],
            top_k=2, view_name="cpu_fake",
        )["deep-00"]
        self.assertEqual(scored["target_positions"], 2)
        self.assertEqual(scored["midpoint_position"], 2)
        self.assertEqual(tuple(scored["objective"].shape), (2,))
        self.assertTrue(torch.isfinite(scored["objective"]).all())

    def test_script_declares_no_authorization_or_scientific_gate(self) -> None:
        script = (
            EXP / "scripts" / "diagnose_nf4_bf16_parity.py"
        ).read_text(encoding="utf-8")
        protocol = (
            EXP / "reports" / "nf4_bf16_parity_protocol.md"
        ).read_text(encoding="utf-8")
        self.assertIn('"status": "interpretation_only"', script)
        self.assertIn('"downstream_authorization": None', script)
        self.assertIn('"scientific_measurements_have_gate": False', script)
        self.assertNotIn("_require_gate", script)
        self.assertIn("cannot stop, rescue, select, or", protocol)
        self.assertIn("not wired into `scripts/run.py`", protocol)


if __name__ == "__main__":
    unittest.main()
