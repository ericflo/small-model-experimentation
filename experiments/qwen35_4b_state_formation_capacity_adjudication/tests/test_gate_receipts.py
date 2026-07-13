from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import gate_receipts as gate_receipts  # noqa: E402
from src.gate_receipts import (  # noqa: E402
    LORA_MISS_BRANCH,
    POSTCONTRAST_FULLRANK_MISS_BRANCH,
    STAGE_B_CONTRAST_BRANCH,
    STAGE_B_FULLRANK_MISS_BRANCH,
    canonical_repo_path,
    canonical_sha256,
    lineage_entry,
    receipt_with_identity,
    reopen_lineage,
    stable_setup_receipt,
    validate_branch_authorization,
    validate_g0_pass,
    validate_positive_control_pass,
    validate_receipt_identity,
)
from src.config import load_config  # noqa: E402
from src.oracle_control import (  # noqa: E402
    generate_control_rows,
    produce_oracle_analysis_receipt,
)


EXPERIMENT = "experiments/qwen35_4b_state_formation_capacity_adjudication"


class GateReceiptContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(ROOT / "configs" / "default.yaml")
        cls.registered_control_rows, _ = generate_control_rows(
            config,
            {
                "files": {
                    split: {"rows": 0, "structural_fingerprints": []}
                    for split in (
                        "train",
                        "validation",
                        "depth_extrapolation",
                        "joint_holdout",
                        "contrast_validation",
                        "contrast_depth",
                        "contrast_joint",
                    )
                }
            },
        )

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        self.identity = {
            "experiment_id": "qwen35_4b_state_formation_capacity_adjudication",
            "model_id": "Qwen/Qwen3.5-4B",
            "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
            "backend": "transformers",
            "config_sha256": "1" * 64,
            "source_contract_sha256": "2" * 64,
            "requirements_training_lock_sha256": "3" * 64,
            "design_receipt_sha256": "4" * 64,
            "design_receipt_identity_sha256": "5" * 64,
        }
        self.setup = self._setup()
        self.data_sha256 = "6" * 64
        self.g0_relative = f"{EXPERIMENT}/runs/setup/g0_lora_seed7411.json"
        self.control_relative = (
            f"{EXPERIMENT}/runs/setup/positive_control_lora_seed7411.json"
        )
        self.control_rows = copy.deepcopy(self.registered_control_rows)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _setup(self) -> dict:
        device = {
            "name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "uuid": "GPU-synthetic",
            "total_memory_gib": 95.0,
            "free_memory_gib_before_load": 91.0,
        }
        targets = [f"model.layers.12.synthetic_{index:03d}" for index in range(62)]
        manifest = [
            {
                "key": f"d{index:03d}",
                "target": target,
                "shapes": [[32, 3], [4, 32]],
                "dtype": "torch.float32",
                "parameters": 224,
            }
            for index, target in enumerate(targets)
        ]
        files = [
            {
                "filename": filename,
                "resolved_revision": self.identity["model_revision"],
                "bytes": index + 1,
                "sha256": f"{index + 1:x}" * 64,
            }
            for index, filename in enumerate(
                sorted(
                    (
                        "chat_template.jinja",
                        "config.json",
                        "merges.txt",
                        "model.safetensors-00001-of-00001.safetensors",
                        "model.safetensors.index.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                    )
                )
            )
        ]
        adaptation_parameters = 224 * 62
        trainable = {
            "total": adaptation_parameters + 100,
            "adaptation": adaptation_parameters,
            "common": 100,
            "tensor_count": 127,
            "names_sha256": "e" * 64,
            "values_sha256": "f" * 64,
        }
        return {
            "capacity": "lora",
            "model_seed": 7411,
            "tokenizer": {
                "state_token_id": 248063,
                "answer_token_ids": [357, 417, 351, 414],
                "runtime_model_config_commit_hash": None,
                "pinned_snapshot": {
                    "model_id": self.identity["model_id"],
                    "requested_revision": self.identity["model_revision"],
                    "resolved_revision": self.identity["model_revision"],
                    "snapshot_layout": f"snapshots/{self.identity['model_revision']}",
                    "files": files,
                    "files_sha256": canonical_sha256({"files": files}),
                },
            },
            "adaptation_targets": targets,
            "adaptation_targets_sha256": hashlib.sha256(
                "\n".join(targets).encode("utf-8")
            ).hexdigest(),
            "adaptation_target_manifest": manifest,
            "adaptation_target_manifest_sha256": canonical_sha256(
                {"targets": manifest}
            ),
            "adaptation_parameters": adaptation_parameters,
            "adaptation_zero_function": {
                "nonzero_output_weights": 0,
                "max_abs_output_weight": 0.0,
            },
            "shared_initialization": {
                "receipt_identity_sha256": "9" * 64,
                "metadata": {"tensor_values_sha256": "9" * 64},
            },
            "trainable_parameters": trainable,
            "dropout_control": {
                "active_nn_dropout_modules": [],
                "model_config_dropout_values": {"attention_dropout": 0.0},
                "matched_adaptation_dropout": 0.05,
            },
            "environment": {
                "python": "3.11.0",
                "device": copy.deepcopy(device),
                "nested_audit": {"free_memory_gib_before_load": "must-remain"},
            },
            "installed_environment_lock": {"sha256": "a" * 64},
            "preflight_device": copy.deepcopy(device),
        }

    def _fullrank_setup(self) -> dict:
        setup = copy.deepcopy(self.setup)
        setup["capacity"] = "fullrank"
        manifest = []
        for index, target in enumerate(setup["adaptation_targets"]):
            manifest.append(
                {
                    "key": f"d{index:03d}",
                    "target": target,
                    "shapes": [[4, 3]],
                    "dtype": "torch.float32",
                    "parameters": 12,
                }
            )
        parameters = 12 * len(manifest)
        setup["adaptation_target_manifest"] = manifest
        setup["adaptation_target_manifest_sha256"] = canonical_sha256(
            {"targets": manifest}
        )
        setup["adaptation_parameters"] = parameters
        setup["trainable_parameters"].update(
            {
                "adaptation": parameters,
                "total": parameters + 100,
                "tensor_count": len(manifest) + 3,
                "values_sha256": "a" * 64,
            }
        )
        return setup

    @staticmethod
    def _dropout(cycles: int, targets: int = 62) -> dict:
        return {
            "calls": cycles * targets,
            "cycles": cycles,
            "cycle_order_identical": True,
            "each_cycle_exact_target_set": True,
            "call_manifest_sha256": "b" * 64,
            "cycle_manifest_sha256s": ["c" * 64] * cycles,
            "mask_sha256": "d" * 64,
        }

    @staticmethod
    def _gradient_summary(*, aggregate: bool) -> dict:
        def group(label: str, tensors: int = 1, *, present: bool = True) -> dict:
            count = tensors if present else 0
            return {
                "tensors": tensors,
                "with_gradient": count,
                "finite": count,
                "nonzero": count,
                "items": [
                    {
                        "name": f"{label}.{index}",
                        "has_gradient": present,
                        "finite": present,
                        "norm": 1.0 if present else None,
                    }
                    for index in range(tensors)
                ],
            }

        return {
            "adaptation": group("adaptation", 2),
            "initializer": group("initializer"),
            "step": group("step"),
            "sufficiency": group("sufficiency"),
            "damping": group("damping"),
            "aggregate_exempt": group("aggregate", present=aggregate),
            "all_required_tensors_finite_nonzero": True,
            "base_gradient_tensors": 0,
        }

    def _peft_reference(self) -> dict:
        def regime(
            dtype: str, dropout: float, autocast: bool, atol: float, rtol: float
        ) -> dict:
            return {
                "passes": True,
                "dtype": dtype,
                "dropout": dropout,
                "autocast": autocast,
                "output_shape_dtype_equal": True,
                "max_output_abs_error": 0.0,
                "max_a_gradient_abs_error": 0.0,
                "max_b_gradient_abs_error": 0.0,
                "atol": atol,
                "rtol": rtol,
                "custom_dropout_receipt": self._dropout(1, targets=1),
            }

        return {
            "peft_version": "0.19.1",
            "scale": 2.0,
            "device": "cuda",
            "actual_adaptation_bank_hook": True,
            "exact_fp32_dropout_disabled": regime(
                "torch.float32", 0.0, False, 1e-6, 1e-5
            ),
            "live_bfloat16_dropout_0_05": regime(
                "torch.bfloat16", 0.05, True, 2e-3, 1e-2
            ),
        }

    def _optimizer(self, *, positive: bool, setup: dict | None = None) -> dict:
        setup = self.setup if setup is None else setup
        adaptation_parameters = int(setup["adaptation_parameters"])
        adaptation_tensors = sum(
            len(item["shapes"]) for item in setup["adaptation_target_manifest"]
        )
        common_parameters = int(setup["trainable_parameters"]["common"])
        common_tensors = int(setup["trainable_parameters"]["tensor_count"]) - adaptation_tensors
        exemptions = 1 if positive else 0
        active_tensors = adaptation_tensors + common_tensors - exemptions
        adaptation_bytes = adaptation_parameters * 8
        common_bytes = (common_parameters - exemptions) * 8
        total_bytes = adaptation_bytes + common_bytes + active_tensors * 4
        return {
            "tensors": active_tensors * 3,
            "bytes_by_dtype": {"torch.float32": total_bytes},
            "total_bytes": total_bytes,
            "total_gib": total_bytes / (1024**3),
            "delta_parameters_audited": adaptation_tensors,
            "delta_moment_tensors": adaptation_tensors * 2,
            "delta_moment_bytes": adaptation_bytes,
            "delta_states_complete": True,
            "delta_state_manifest_sha256": "1" * 64,
            "groups": [
                {
                    "group_name": "adaptation",
                    "parameters": adaptation_tensors,
                    "moment_tensors": adaptation_tensors * 2,
                    "moment_bytes": adaptation_bytes,
                    "registered_missing_state_exemptions": 0,
                    "state_manifest_sha256": "2" * 64,
                    "required_states_complete_and_finite": True,
                },
                {
                    "group_name": "common_state",
                    "parameters": common_tensors,
                    "moment_tensors": (common_tensors - exemptions) * 2,
                    "moment_bytes": common_bytes,
                    "registered_missing_state_exemptions": exemptions,
                    "state_manifest_sha256": "3" * 64,
                    "required_states_complete_and_finite": True,
                },
            ],
            "all_required_group_states_complete_and_finite": True,
            "registered_missing_state_exemptions": exemptions,
        }

    def _identity_payload(self, status: str, phase: str) -> dict:
        return {
            "schema_version": 1,
            "status": status,
            **self.identity,
            "phase": phase,
        }

    def _write(self, relative: str, payload: dict) -> tuple[Path, dict]:
        path = self.repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        receipt = receipt_with_identity(payload)
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path, receipt

    def _rewrite(self, relative: str, receipt: dict) -> dict:
        payload = copy.deepcopy(receipt)
        payload.pop("receipt_identity_sha256", None)
        rehashed = receipt_with_identity(payload)
        (self.repo / relative).write_text(
            json.dumps(rehashed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return rehashed

    def _g0_payload(self) -> dict:
        state_gradients = self._gradient_summary(aggregate=False)
        joint_gradients = self._gradient_summary(aggregate=True)
        return {
            **self._identity_payload("MODEL_SMOKE_PASS", "lora_g0"),
            "capacity": "lora",
            "model_seed": 7411,
            "data_manifest_sha256": self.data_sha256,
            "setup": copy.deepcopy(self.setup),
            "peft_formula_reference": self._peft_reference(),
            "branch_authorization": None,
            "k1_max_logit_abs_error_before_optimizer": 0.0,
            "k1_adaptation_calls": 0,
            "k1_max_logit_abs_error_after_optimizer": 0.0,
            "k1_adaptation_calls_after_optimizer": 0,
            "zero_function_enabled_minus_disabled_error": 0.0,
            "two_step_gradient_probe": [
                {
                    "step": step,
                    "loss": 0.5,
                    "dropout_probe": self._dropout(3),
                    "gradients": copy.deepcopy(state_gradients),
                    "preclip_adaptation_gradient_norm": 0.2,
                    "preclip_common_gradient_norm": 0.3,
                }
                for step in (1, 2)
            ],
            "live_joint_backward_probe": {
                "objective": "joint",
                "loss": 1.2,
                "answer_loss": 0.7,
                "elapsed_seconds": 0.1,
                "peak_allocated_gib": 12.0,
                "dropout_probe": self._dropout(3),
                "gradients": joint_gradients,
                "all_joint_trainable_groups_finite_nonzero": True,
                "preclip_adaptation_gradient_norm": 0.2,
                "preclip_common_gradient_norm": 0.3,
                "adaptation_applied_clip_scale": 1.0,
                "common_state_applied_clip_scale": 1.0,
            },
            "optimizer_state": self._optimizer(positive=False),
            "timed_ten_step_probe": {
                "steps": 10,
                "losses": [0.5] * 10,
                "elapsed_seconds": 2.0,
                "seconds_per_step": 0.2,
            },
            "worst_depth": 12,
            "worst_setup_row": {
                "id": "setup_g0_worst_depth-73992-synthetic",
                "seed": 73992,
                "structural_fingerprint": "1" * 64,
                "cross_result_structural_overlap": 0,
            },
            "worst_call_receipt": self._dropout(11),
            "worst_forward_seconds": 0.4,
            "peak_allocated_gib": 12.0,
            "peak_reserved_gib": 13.0,
            "elapsed_seconds": 3.0,
            "checkpoint_roundtrip": {
                "destructive_adaptation_digest_changed": True,
                "destructive_common_digest_changed": True,
                "restored_adaptation_digest_equal": True,
                "restored_common_digest_equal": True,
                "max_logit_abs_error": 0.0,
            },
            "common_initialization_rng_isolation": {
                "tensor_manifest_equal": True,
                "tensor_values_sha256": "9" * 64,
                "expected_tensor_values_sha256": "9" * 64,
            },
            "free_memory_gib_after_g0": 8.0,
            "authorizes_positive_control": True,
            "authorizes_training": False,
            "authorizes_result_training": False,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": ["train"],
            "sealed_contrast_payloads_opened": [],
            "training_or_evaluation_started": False,
            "scientific_evidence": False,
        }

    def _write_g0(self) -> tuple[Path, dict, dict]:
        path, receipt = self._write(self.g0_relative, self._g0_payload())
        return path, receipt, lineage_entry(self.repo, path, receipt)

    def _g0_validation(self, path: Path, *, expected_setup: dict | None = None) -> dict:
        return validate_g0_pass(
            self.repo,
            path,
            canonical_relative_path=self.g0_relative,
            expected_identity=self.identity,
            capacity="lora",
            model_seed=7411,
            data_manifest_sha256=self.data_sha256,
            expected_setup=self.setup if expected_setup is None else expected_setup,
            expected_branch_authorization=None,
            k1_max_logit_abs_error=0.00001,
            train_k=4,
            max_recurrence=12,
            expected_adaptation_targets=62,
            expected_adaptation_parameters=224 * 62,
            expected_adaptation_dropout=0.05,
            expected_adaptation_scale=2.0,
            expected_lora_rank=32,
            expected_peft_version="0.19.1",
            adaptation_gradient_clip=1.0,
            common_gradient_clip=1.0,
            worst_depth_seed=73992,
        )

    @staticmethod
    def _evaluation(step: int, mode: str, *, rows: int = 48, correct: int = 0) -> dict:
        return {
            "step": step,
            "adaptation_mode": mode,
            "overall": {
                "rows": rows,
                "terminal_correct_counts": {"joint": correct},
                "joint_final_accuracy": correct / rows,
            },
        }

    def _positive_payload(self, g0_lineage: dict) -> dict:
        evaluations = []
        for step in (0, 1, 16, 64, 128, 256):
            evaluations.append(
                self._evaluation(step, "intact", correct=48 if step == 256 else 0)
            )
            evaluations.append(self._evaluation(step, "disabled"))
        setup = copy.deepcopy(self.setup)
        setup["environment"]["device"]["free_memory_gib_before_load"] = 88.0
        setup["preflight_device"]["free_memory_gib_before_load"] = 87.0
        grid = {
            f"{family}|{template}|depth={depth}|query={query}": 2
            for family in ("phase_branch", "checksum_branch")
            for template in ("ledger", "prose")
            for depth in (2, 3, 4)
            for query in ("node", "checksum")
        }
        optimizer_probes = [
            {
                "step": step,
                "microbatch_start": (step - 1) * 16 + 1,
                "microbatch_end": step * 16,
                "microbatches": 16,
                "adaptation_gradient_finite": True,
                "common_state_gradient_finite": True,
                "base_trainable_parameters": 0,
                "adaptation_learning_rate": 0.0002,
                "common_state_learning_rate": 0.0002,
                "adaptation_preclip_gradient_norm": 0.2,
                "common_state_preclip_gradient_norm": 0.3,
                "adaptation_applied_clip_scale": 1.0,
                "common_state_applied_clip_scale": 1.0,
            }
            for step in (1, 16, 64, 128, 256)
        ]
        final_trainable = copy.deepcopy(setup["trainable_parameters"])
        final_trainable["values_sha256"] = "0" * 64
        canonical_rows = hashlib.sha256()
        for row in self.control_rows:
            canonical_rows.update(
                json.dumps(row, sort_keys=True, separators=(",", ":")).encode()
                + b"\n"
            )
        return {
            **self._identity_payload(
                "POSITIVE_CONTROL_PASS", "lora_positive_control"
            ),
            "capacity": "lora",
            "model_seed": 7411,
            "data_manifest_sha256": self.data_sha256,
            "g0_lineage": g0_lineage,
            "branch_authorization": None,
            "setup": setup,
            "shared_initialization": setup["shared_initialization"],
            "control_rows": {
                "seed": 73991,
                "rows": 48,
                "grid": grid,
                "canonical_rows_sha256": canonical_rows.hexdigest(),
                "cross_result_structural_overlap": 0,
            },
            "oracle_analysis": produce_oracle_analysis_receipt(self.control_rows),
            "oracle_readout_accuracy": 1.0,
            "overfit_rows": 48,
            "overfit_updates": 256,
            "overfit_gradient_accumulation": 16,
            "overfit_microbatches": 4096,
            "overfit_final_joint_accuracy": 1.0,
            "overfit_final_joint_correct": 48,
            "training_diagnostics": {
                "fixed_probe_steps": [0, 1, 16, 64, 128, 256],
                "geometry": {
                    "rows": 48,
                    "optimizer_updates": 256,
                    "gradient_accumulation": 16,
                    "singleton_microbatches": 4096,
                    "loss_divisor": 16,
                    "optimizer_zero_grad_calls": 257,
                    "adaptation_clip_calls": 256,
                    "common_state_clip_calls": 256,
                    "optimizer_step_calls": 256,
                    "early_stopping": False,
                    "checkpoint_selection": False,
                },
                "completed_updates": 256,
                "completed_microbatches": 4096,
                "evaluations": evaluations,
                "optimizer_step_probes": optimizer_probes,
                "minimum_applied_clip_scales": {
                    "adaptation": 1.0,
                    "common_state": 1.0,
                },
                "parameter_values_changed": True,
                "optimizer_state": self._optimizer(positive=True),
                "initial_trainable_parameters": copy.deepcopy(
                    setup["trainable_parameters"]
                ),
                "final_trainable_parameters": final_trainable,
                "final_parameter_delta_norms": {
                    "adaptation_output": {"l2_delta_norm": 1.0},
                    "common_state": {"l2_delta_norm": 1.0},
                },
            },
            "authorizes_training": True,
            "authorizes_result_training": True,
            "authorizes_result_evaluation": False,
            "benchmark_files_read": 0,
            "result_payloads_opened": [],
            "sealed_contrast_payloads_opened": [],
            "scientific_evidence": False,
        }

    def _positive_validation(
        self, path: Path, g0_lineage: dict, *, expected_setup: dict | None = None
    ) -> dict:
        return validate_positive_control_pass(
            self.repo,
            path,
            canonical_relative_path=self.control_relative,
            expected_identity=self.identity,
            capacity="lora",
            model_seed=7411,
            data_manifest_sha256=self.data_sha256,
            expected_setup=self.setup if expected_setup is None else expected_setup,
            expected_branch_authorization=None,
            expected_g0_lineage=g0_lineage,
            expected_control_rows=self.control_rows,
            control_seed=73991,
            control_rows=48,
            control_updates=256,
            gradient_accumulation=16,
            min_oracle_readout_accuracy=0.99,
            min_overfit_final_joint_accuracy=0.95,
            expected_adaptation_targets=62,
            expected_adaptation_parameters=224 * 62,
            expected_adaptation_dropout=0.05,
            expected_lora_rank=32,
            control_families=("phase_branch", "checksum_branch"),
            control_templates=("ledger", "prose"),
            control_depths=(2, 3, 4),
            control_query_kinds=("node", "checksum"),
            control_examples_per_cell=2,
            learning_rate=0.0002,
            adaptation_gradient_clip=1.0,
            common_gradient_clip=1.0,
        )

    def _analysis_relative(self, filename: str) -> str:
        return f"{EXPERIMENT}/analysis/{filename}"

    def _write_lora_miss(self) -> tuple[Path, dict, dict]:
        relative = self._analysis_relative("lora_joint_trigger.json")
        path, receipt = self._write(
            relative,
            {
                **self._identity_payload(
                    "LORA_JOINT_MISS_CONTROLS_REQUIRED", "lora_joint_analysis"
                ),
                "verdict": "LORA_JOINT_MISS_CONTROLS_REQUIRED",
                "analysis_phase": "lora_joint",
                "next_stage": "run_lora_state_only_and_fullrank_joint",
                "authorization": None,
                "formation": {"status": "TRAINED_DEPTH_MISS", "passes": False},
            },
        )
        return path, receipt, lineage_entry(self.repo, path, receipt)

    def _write_lora_control(self, root: dict) -> tuple[Path, dict, dict]:
        relative = self._analysis_relative("lora_control.json")
        path, receipt = self._write(
            relative,
            {
                **self._identity_payload(
                    "LORA_STATE_ONLY_CONTROL_COMPLETE", "lora_control_analysis"
                ),
                "verdict": "LORA_CAN_FORM_STATE_STATE_ONLY",
                "analysis_phase": "lora_control",
                "next_stage": "continue_mandatory_stage_b_seal",
                "authorization": root,
                "formation": {"status": "STATE_FORMATION_PASS", "passes": True},
                "lora_state_only_annotation": "LORA_CAN_FORM_STATE_STATE_ONLY",
            },
        )
        return path, receipt, lineage_entry(self.repo, path, receipt)

    def _write_stage_b(
        self, root: dict, control: dict, *, fullrank_passes: bool
    ) -> tuple[Path, dict, dict, str]:
        status = (
            "STAGE_B_CONTRAST_AUTHORIZED"
            if fullrank_passes
            else "FULLRANK_STATE_ONLY_REQUIRED"
        )
        next_stage = (
            "evaluate_exact_six_joint_contrast_cells"
            if fullrank_passes
            else "run_fullrank_state_only_control"
        )
        relative = self._analysis_relative("stage_b_seal.json")
        path, receipt = self._write(
            relative,
            {
                **self._identity_payload(status, "stage_b_seal_analysis"),
                "verdict": status,
                "analysis_phase": "stage_b_seal",
                "next_stage": next_stage,
                "authorization": root,
                "lora_control_analysis": control,
                "lora_joint_formation": {
                    "status": "TRAINED_DEPTH_MISS",
                    "passes": False,
                },
                "lora_state_only_formation": {
                    "status": "STATE_FORMATION_PASS",
                    "passes": True,
                },
                "lora_state_only_annotation": "LORA_CAN_FORM_STATE_STATE_ONLY",
                "fullrank_trigger_formation": {
                    "status": (
                        "STATE_FORMATION_PASS"
                        if fullrank_passes
                        else "TRAINED_DEPTH_MISS"
                    ),
                    "passes": fullrank_passes,
                },
                "matching": {"status": "STAGE_B_MATCHING_VALID"},
                "contrast_firewall": {"status": "CONTRAST_FIREWALL_UNOPENED"},
            },
        )
        return path, receipt, lineage_entry(self.repo, path, receipt), status

    def test_full_identity_and_exact_lineage_reopen_reject_rehashed_tamper(self) -> None:
        path, receipt, lineage = self._write_g0()
        validate_receipt_identity(
            receipt,
            self.identity,
            expected_status="MODEL_SMOKE_PASS",
            expected_phase="lora_g0",
        )
        self.assertEqual(reopen_lineage(self.repo, lineage), receipt)

        changed = copy.deepcopy(receipt)
        changed["model_id"] = "forbidden/model"
        changed = self._rewrite(self.g0_relative, changed)
        changed_lineage = lineage_entry(self.repo, path, changed)
        with self.assertRaisesRegex(RuntimeError, "model_id"):
            reopen_lineage(
                self.repo,
                changed_lineage,
                expected_identity=self.identity,
                expected_status="MODEL_SMOKE_PASS",
                expected_phase="lora_g0",
            )

    def test_canonical_paths_reject_dotdot_redundant_and_symlink_aliases(self) -> None:
        path, receipt, lineage = self._write_g0()
        for alias in (
            self.g0_relative.replace("/runs/", "//runs/"),
            self.g0_relative.replace("/runs/", "/analysis/../runs/"),
            f"/{self.g0_relative}",
        ):
            with self.subTest(alias=alias), self.assertRaises(RuntimeError):
                canonical_repo_path(self.repo, alias)

        link_relative = f"{EXPERIMENT}/runs/setup/g0_alias.json"
        link = self.repo / link_relative
        os.symlink(path.name, link)
        alias_lineage = dict(lineage, path=link_relative)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            reopen_lineage(self.repo, alias_lineage)

    def test_stable_setup_is_exact_and_excludes_only_two_free_memory_leaves(self) -> None:
        changed = copy.deepcopy(self.setup)
        changed["environment"]["device"]["free_memory_gib_before_load"] = 1.0
        changed["preflight_device"]["free_memory_gib_before_load"] = 2.0
        self.assertEqual(stable_setup_receipt(changed), stable_setup_receipt(self.setup))

        changed["environment"]["nested_audit"]["free_memory_gib_before_load"] = "changed"
        self.assertNotEqual(stable_setup_receipt(changed), stable_setup_receipt(self.setup))
        for mutation in ("extra", "missing_leaf"):
            malformed = copy.deepcopy(self.setup)
            if mutation == "extra":
                malformed["decoy"] = True
            else:
                malformed["preflight_device"].pop("free_memory_gib_before_load")
            with self.subTest(mutation=mutation), self.assertRaises(RuntimeError):
                stable_setup_receipt(malformed)

    def test_g0_pass_validates_access_binding_and_core_geometry(self) -> None:
        path, baseline, _ = self._write_g0()
        self.assertEqual(self._g0_validation(path)["status"], "MODEL_SMOKE_PASS")
        mutations = {
            "scientific_evidence": lambda item: item.__setitem__("scientific_evidence", True),
            "result_access": lambda item: item.__setitem__("result_payloads_opened", []),
            "k1_threshold": lambda item: item.__setitem__(
                "k1_max_logit_abs_error_after_optimizer", 0.00002
            ),
            "k4_calls": lambda item: item["two_step_gradient_probe"][1][
                "dropout_probe"
            ].__setitem__("calls", 185),
            "joint_gradient": lambda item: item["live_joint_backward_probe"][
                "gradients"
            ]["aggregate_exempt"].__setitem__("nonzero", 0),
            "worst_depth": lambda item: item.__setitem__("worst_depth", 11),
            "worst_overlap": lambda item: item["worst_setup_row"].__setitem__(
                "cross_result_structural_overlap", 1
            ),
            "roundtrip": lambda item: item["checkpoint_roundtrip"].__setitem__(
                "max_logit_abs_error", 0.1
            ),
            "setup_decoy": lambda item: item["setup"].__setitem__("decoy", True),
            "branch_decoy": lambda item: item.__setitem__(
                "branch_authorization", {"decoy": {"status": "VALID"}}
            ),
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            self._rewrite(self.g0_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._g0_validation(path)
        self._rewrite(self.g0_relative, baseline)

    def test_rehashed_setup_semantic_forgeries_are_rejected(self) -> None:
        path, baseline, _ = self._write_g0()

        def reverse_order(setup: dict) -> None:
            setup["adaptation_targets"].reverse()
            setup["adaptation_target_manifest"].reverse()
            for index, item in enumerate(setup["adaptation_target_manifest"]):
                item["key"] = f"d{index:03d}"
            setup["adaptation_targets_sha256"] = hashlib.sha256(
                "\n".join(setup["adaptation_targets"]).encode("utf-8")
            ).hexdigest()
            setup["adaptation_target_manifest_sha256"] = canonical_sha256(
                {"targets": setup["adaptation_target_manifest"]}
            )

        def change_parameter_sum(setup: dict) -> None:
            setup["adaptation_target_manifest"][0]["parameters"] += 1
            setup["adaptation_target_manifest_sha256"] = canonical_sha256(
                {"targets": setup["adaptation_target_manifest"]}
            )
            setup["adaptation_parameters"] += 1
            setup["trainable_parameters"]["adaptation"] += 1
            setup["trainable_parameters"]["total"] += 1

        def remove_snapshot_file(setup: dict) -> None:
            snapshot = setup["tokenizer"]["pinned_snapshot"]
            snapshot["files"] = [
                item for item in snapshot["files"] if item["filename"] != "tokenizer.json"
            ]
            snapshot["files_sha256"] = canonical_sha256({"files": snapshot["files"]})

        mutations = {
            "ordered_targets": reverse_order,
            "manifest_parameter_sum": change_parameter_sum,
            "zero_function": lambda setup: setup.__setitem__(
                "adaptation_zero_function",
                {"nonzero_output_weights": 1, "max_abs_output_weight": 0.0},
            ),
            "dropout": lambda setup: setup["dropout_control"].__setitem__(
                "matched_adaptation_dropout", 0.0
            ),
            "trainable_geometry": lambda setup: setup["trainable_parameters"].__setitem__(
                "tensor_count", 124
            ),
            "pinned_tokenizer_snapshot": remove_snapshot_file,
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate["setup"])
            self._rewrite(self.g0_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._g0_validation(path, expected_setup=candidate["setup"])

    def test_rehashed_peft_regime_and_fullrank_reference_forgeries_are_rejected(self) -> None:
        path, baseline, _ = self._write_g0()
        mutations = {
            "exact_pass": lambda item: item["peft_formula_reference"][
                "exact_fp32_dropout_disabled"
            ].__setitem__("passes", False),
            "live_tolerance": lambda item: item["peft_formula_reference"][
                "live_bfloat16_dropout_0_05"
            ].__setitem__("atol", 0.5),
            "live_geometry": lambda item: item["peft_formula_reference"][
                "live_bfloat16_dropout_0_05"
            ]["custom_dropout_receipt"].__setitem__("calls", 2),
            "live_error": lambda item: item["peft_formula_reference"][
                "live_bfloat16_dropout_0_05"
            ].__setitem__("max_output_abs_error", 0.003),
            "lora_none": lambda item: item.__setitem__("peft_formula_reference", None),
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            self._rewrite(self.g0_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._g0_validation(path)

        fullrank_setup = self._fullrank_setup()
        fullrank = copy.deepcopy(baseline)
        fullrank.update(
            {
                "phase": "fullrank_g0",
                "capacity": "fullrank",
                "setup": fullrank_setup,
                "peft_formula_reference": None,
                "optimizer_state": self._optimizer(
                    positive=False, setup=fullrank_setup
                ),
            }
        )
        fullrank_relative = f"{EXPERIMENT}/runs/setup/g0_fullrank_seed7411.json"
        fullrank_path, _ = self._write(fullrank_relative, {
            key: value for key, value in fullrank.items() if key != "receipt_identity_sha256"
        })

        def validate_fullrank() -> dict:
            return validate_g0_pass(
                self.repo,
                fullrank_path,
                canonical_relative_path=fullrank_relative,
                expected_identity=self.identity,
                capacity="fullrank",
                model_seed=7411,
                data_manifest_sha256=self.data_sha256,
                expected_setup=fullrank_setup,
                expected_branch_authorization=None,
                k1_max_logit_abs_error=0.00001,
                train_k=4,
                max_recurrence=12,
                expected_adaptation_targets=62,
                expected_adaptation_parameters=12 * 62,
                expected_adaptation_dropout=0.05,
                expected_adaptation_scale=2.0,
                expected_lora_rank=32,
                expected_peft_version="0.19.1",
                adaptation_gradient_clip=1.0,
                common_gradient_clip=1.0,
                worst_depth_seed=73992,
            )

        self.assertEqual(validate_fullrank()["capacity"], "fullrank")
        forged = copy.deepcopy(fullrank)
        forged["peft_formula_reference"] = self._peft_reference()
        self._rewrite(fullrank_relative, forged)
        with self.assertRaisesRegex(RuntimeError, "full-rank PEFT"):
            validate_fullrank()

    def test_rehashed_g0_resource_clip_optimizer_and_rng_forgeries_are_rejected(self) -> None:
        path, baseline, _ = self._write_g0()
        mutations = {
            "worst_seed": lambda item: item["worst_setup_row"].__setitem__("seed", 73993),
            "elapsed": lambda item: item.__setitem__("elapsed_seconds", 1.0),
            "memory": lambda item: item.__setitem__("peak_reserved_gib", 11.0),
            "clip_scale": lambda item: item["live_joint_backward_probe"].__setitem__(
                "common_state_applied_clip_scale", 0.5
            ),
            "optimizer_moments": lambda item: item["optimizer_state"].__setitem__(
                "delta_moment_tensors", 246
            ),
            "optimizer_bytes": lambda item: item["optimizer_state"]["groups"][0].__setitem__(
                "moment_bytes", 1
            ),
            "rng_init_binding": lambda item: item[
                "common_initialization_rng_isolation"
            ].update(
                {
                    "tensor_values_sha256": "8" * 64,
                    "expected_tensor_values_sha256": "8" * 64,
                }
            ),
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            self._rewrite(self.g0_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._g0_validation(path)

    def test_positive_control_pass_validates_g0_access_geometry_and_thresholds(self) -> None:
        _, _, g0_lineage = self._write_g0()
        path, baseline = self._write(
            self.control_relative, self._positive_payload(g0_lineage)
        )
        self.assertEqual(
            self._positive_validation(path, g0_lineage)["status"],
            "POSITIVE_CONTROL_PASS",
        )
        mutations = {
            "evaluation_authorization": lambda item: item.__setitem__(
                "authorizes_result_evaluation", True
            ),
            "scientific_evidence": lambda item: item.__setitem__("scientific_evidence", True),
            "g0_decoy": lambda item: item.__setitem__(
                "g0_lineage", {"decoy": g0_lineage}
            ),
            "early_stop": lambda item: item["training_diagnostics"]["geometry"].__setitem__(
                "early_stopping", True
            ),
            "microbatches": lambda item: item.__setitem__("overfit_microbatches", 4095),
            "parameter_motion": lambda item: item["training_diagnostics"][
                "final_parameter_delta_norms"
            ]["adaptation_output"].__setitem__("l2_delta_norm", 0.0),
            "stable_nested_decoy": lambda item: item["setup"]["environment"][
                "nested_audit"
            ].__setitem__("free_memory_gib_before_load", "forged"),
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            self._rewrite(self.control_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._positive_validation(path, g0_lineage)

        forged_oracle = copy.deepcopy(baseline)
        oracle = forged_oracle["oracle_analysis"]
        oracle["terminal_joint_correct"] = 47
        oracle["terminal_joint_accuracy"] = 47 / 48
        oracle.pop("receipt_identity_sha256")
        oracle["receipt_identity_sha256"] = canonical_sha256(oracle)
        self._rewrite(self.control_relative, forged_oracle)
        with self.assertRaisesRegex(RuntimeError, "exact recomputation"):
            self._positive_validation(path, g0_lineage)

        below = copy.deepcopy(baseline)
        below["overfit_final_joint_correct"] = 45
        below["overfit_final_joint_accuracy"] = 45 / 48
        final = next(
            row
            for row in below["training_diagnostics"]["evaluations"]
            if row["step"] == 256 and row["adaptation_mode"] == "intact"
        )
        final["overall"]["terminal_correct_counts"]["joint"] = 45
        final["overall"]["joint_final_accuracy"] = 45 / 48
        self._rewrite(self.control_relative, below)
        with self.assertRaisesRegex(RuntimeError, "threshold"):
            self._positive_validation(path, g0_lineage)

    def test_rehashed_positive_factorial_clip_and_optimizer_forgeries_are_rejected(self) -> None:
        _, _, g0_lineage = self._write_g0()
        path, baseline = self._write(
            self.control_relative, self._positive_payload(g0_lineage)
        )
        mutations = {
            "factorial_label": lambda item: item["control_rows"]["grid"].update(
                {"decoy|ledger|depth=2|query=node": 2}
            ),
            "factorial_count": lambda item: item["control_rows"]["grid"].__setitem__(
                "phase_branch|ledger|depth=2|query=node", 1
            ),
            "probe_clip": lambda item: item["training_diagnostics"][
                "optimizer_step_probes"
            ][0].__setitem__("common_state_applied_clip_scale", 0.5),
            "probe_learning_rate": lambda item: item["training_diagnostics"][
                "optimizer_step_probes"
            ][0].__setitem__("adaptation_learning_rate", 0.1),
            "minimum_clip": lambda item: item["training_diagnostics"][
                "minimum_applied_clip_scales"
            ].__setitem__("adaptation", 1.1),
            "optimizer_exemption": lambda item: item["training_diagnostics"][
                "optimizer_state"
            ].__setitem__("registered_missing_state_exemptions", 0),
            "initial_trainables": lambda item: item["training_diagnostics"][
                "initial_trainable_parameters"
            ].__setitem__("values_sha256", "7" * 64),
        }
        for name, mutate in mutations.items():
            candidate = copy.deepcopy(baseline)
            mutate(candidate)
            self._rewrite(self.control_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                self._positive_validation(path, g0_lineage)

    @mock.patch.object(gate_receipts, "_validate_branch_evidence")
    def test_status_specific_named_branch_ancestry_and_return_contract(
        self, _evidence: mock.Mock
    ) -> None:
        lora_path, _, root = self._write_lora_miss()
        lora = validate_branch_authorization(
            self.repo,
            lora_path,
            canonical_relative_path=self._analysis_relative("lora_joint_trigger.json"),
            branch=LORA_MISS_BRANCH,
            expected_identity=self.identity,
        )
        self.assertEqual(lora["root_lora_miss_lineage"], root)
        self.assertIsNone(lora["stage_b_lineage"])
        _, _, control = self._write_lora_control(root)
        stage_path, _, stage, _ = self._write_stage_b(root, control, fullrank_passes=True)
        stage_result = validate_branch_authorization(
            self.repo,
            stage_path,
            canonical_relative_path=self._analysis_relative("stage_b_seal.json"),
            branch=STAGE_B_CONTRAST_BRANCH,
            expected_identity=self.identity,
        )
        self.assertEqual(stage_result["root_lora_miss_lineage"], root)
        self.assertEqual(stage_result["stage_b_lineage"], stage)

        post_relative = self._analysis_relative("fullrank_joint.json")
        post_path, _ = self._write(
            post_relative,
            {
                **self._identity_payload(
                    "FULLRANK_STATE_ONLY_REQUIRED", "fullrank_joint_analysis"
                ),
                "verdict": "FULLRANK_STATE_ONLY_REQUIRED",
                "analysis_phase": "fullrank_joint",
                "next_stage": "run_fullrank_state_only_control",
                "authorization": stage,
                "lora_sealed_contrast_formation": {
                    "status": "TRAINED_DEPTH_MISS",
                    "passes": False,
                },
                "trigger_formation": {
                    "status": "STATE_FORMATION_PASS",
                    "passes": True,
                },
                "sealed_contrast_formation": {
                    "status": "TRAINED_DEPTH_MISS",
                    "passes": False,
                },
            },
        )
        post = validate_branch_authorization(
            self.repo,
            post_path,
            canonical_relative_path=post_relative,
            branch=POSTCONTRAST_FULLRANK_MISS_BRANCH,
            expected_identity=self.identity,
        )
        self.assertEqual(post["root_lora_miss_lineage"], root)
        self.assertEqual(post["stage_b_lineage"], stage)

        stage_path, _, _, _ = self._write_stage_b(
            root, control, fullrank_passes=False
        )
        miss = validate_branch_authorization(
            self.repo,
            stage_path,
            canonical_relative_path=self._analysis_relative("stage_b_seal.json"),
            branch=STAGE_B_FULLRANK_MISS_BRANCH,
            expected_identity=self.identity,
        )
        self.assertEqual(miss["root_lora_miss_lineage"], root)

    @mock.patch.object(gate_receipts, "_validate_branch_evidence")
    def test_rehashed_registered_formation_and_direct_control_forgeries_are_rejected(
        self, _evidence: mock.Mock
    ) -> None:
        _, _, root = self._write_lora_miss()
        control_path, control_receipt, control = self._write_lora_control(root)
        stage_path, stage_receipt, stage, _ = self._write_stage_b(
            root, control, fullrank_passes=True
        )
        stage_relative = self._analysis_relative("stage_b_seal.json")

        def validate_stage() -> dict:
            return validate_branch_authorization(
                self.repo,
                stage_path,
                canonical_relative_path=stage_relative,
                branch=STAGE_B_CONTRAST_BRANCH,
                expected_identity=self.identity,
            )

        stage_mutations = {
            "root_formation_detached": lambda item: item["lora_joint_formation"].__setitem__(
                "decoy", True
            ),
            "control_formation_detached": lambda item: item[
                "lora_state_only_formation"
            ].__setitem__("decoy", True),
            "control_annotation": lambda item: item.__setitem__(
                "lora_state_only_annotation", None
            ),
        }
        for name, mutate in stage_mutations.items():
            candidate = copy.deepcopy(stage_receipt)
            mutate(candidate)
            self._rewrite(stage_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                validate_stage()

        # Preserve exact lineage while forging the control's own semantics.
        forged_control = copy.deepcopy(control_receipt)
        forged_control["verdict"] = "STATE_FORMATION_PASS"
        forged_control = self._rewrite(
            self._analysis_relative("lora_control.json"), forged_control
        )
        forged_control_lineage = lineage_entry(self.repo, control_path, forged_control)
        forged_stage = copy.deepcopy(stage_receipt)
        forged_stage["lora_control_analysis"] = forged_control_lineage
        self._rewrite(stage_relative, forged_stage)
        with self.assertRaisesRegex(RuntimeError, "LoRA-control verdict"):
            validate_stage()

        # Restore the canonical control and Stage-B seal for the postcontrast tests.
        control_receipt = self._rewrite(
            self._analysis_relative("lora_control.json"), control_receipt
        )
        control = lineage_entry(self.repo, control_path, control_receipt)
        _, stage_receipt, stage, _ = self._write_stage_b(
            root, control, fullrank_passes=True
        )
        post_relative = self._analysis_relative("fullrank_joint.json")
        post_path, post_receipt = self._write(
            post_relative,
            {
                **self._identity_payload(
                    "FULLRANK_STATE_ONLY_REQUIRED", "fullrank_joint_analysis"
                ),
                "verdict": "FULLRANK_STATE_ONLY_REQUIRED",
                "analysis_phase": "fullrank_joint",
                "next_stage": "run_fullrank_state_only_control",
                "authorization": stage,
                "lora_sealed_contrast_formation": {
                    "status": "TRAINED_DEPTH_MISS",
                    "passes": False,
                },
                "trigger_formation": {
                    "status": "STATE_FORMATION_PASS",
                    "passes": True,
                },
                "sealed_contrast_formation": {
                    "status": "TRAINED_DEPTH_MISS",
                    "passes": False,
                },
            },
        )

        def validate_post() -> dict:
            return validate_branch_authorization(
                self.repo,
                post_path,
                canonical_relative_path=post_relative,
                branch=POSTCONTRAST_FULLRANK_MISS_BRANCH,
                expected_identity=self.identity,
            )

        post_mutations = {
            "trigger_miss": lambda item: item.__setitem__(
                "trigger_formation",
                {"status": "TRAINED_DEPTH_MISS", "passes": False},
            ),
            "sealed_pass": lambda item: item.__setitem__(
                "sealed_contrast_formation",
                {"status": "STATE_FORMATION_PASS", "passes": True},
            ),
        }
        for name, mutate in post_mutations.items():
            candidate = copy.deepcopy(post_receipt)
            mutate(candidate)
            self._rewrite(post_relative, candidate)
            with self.subTest(name=name), self.assertRaises(RuntimeError):
                validate_post()

    def test_rehashed_nested_decoy_cannot_substitute_for_named_ancestry(self) -> None:
        _, _, root = self._write_lora_miss()
        _, _, control = self._write_lora_control(root)
        stage_path, stage_receipt, _, _ = self._write_stage_b(
            root, control, fullrank_passes=True
        )
        decoy = copy.deepcopy(stage_receipt)
        decoy["authorization"] = None
        decoy["arbitrary_nested_decoy"] = {"valid_root": root}
        self._rewrite(self._analysis_relative("stage_b_seal.json"), decoy)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            validate_branch_authorization(
                self.repo,
                stage_path,
                canonical_relative_path=self._analysis_relative("stage_b_seal.json"),
                branch=STAGE_B_CONTRAST_BRANCH,
                expected_identity=self.identity,
            )


if __name__ == "__main__":
    unittest.main()
