from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import matched_compute as M  # noqa: E402


def training(seed: int, *, tokens: int = 300, wall: float = 30.0) -> dict:
    return {
        "arm": "reflection_correct",
        "seed": seed,
        "compute": {
            "schema_version": 1,
            "amortization_horizon": "full_training_charged_to_each_confirmation_split",
            "forward_tokens": tokens // 3,
            "forward_backward_multiplier": 3,
            "token_forward_equivalents": tokens,
            "model_load_seconds": wall / 3,
            "training_seconds": wall * 2 / 3,
            "gpu_phase_wall_seconds": wall,
        },
    }


def metadata(
    seed: int | None,
    *,
    logical: int = 100,
    sampled: int = 50,
    load: float = 10.0,
    generation: float = 20.0,
) -> dict:
    return {
        "model_override": None if seed is None else {"source_seed": seed},
        "counts": {
            "logical_model_input_tokens": logical,
            "sampled_tokens": sampled,
        },
        "timing": {
            "model_load_seconds": load,
            "generation_seconds": generation,
        },
    }


class MatchedComputeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_target_charges_full_training_to_each_seed_then_takes_max(self) -> None:
        target = M.target_compute_budget(
            [
                (training(47, tokens=300, wall=30), metadata(47)),
                (training(53, tokens=600, wall=60), metadata(53)),
            ],
            {47, 53},
        )
        self.assertEqual(target["per_seed"]["47"]["total_token_forward_equivalents"], 450)
        self.assertEqual(target["required_token_forward_equivalents"], 750)
        self.assertEqual(target["required_gpu_phase_wall_seconds"], 90.0)

    def test_both_token_and_wall_units_are_required_for_compute_stop(self) -> None:
        target = {
            "required_token_forward_equivalents": 400,
            "required_gpu_phase_wall_seconds": 50.0,
        }
        cumulative = M.cumulative_reservoir_compute(
            [
                metadata(None, logical=300, sampled=100, load=10, generation=10),
                metadata(None, logical=10, sampled=10, load=10, generation=10),
                metadata(None, logical=10, sampled=10, load=10, generation=20),
            ]
        )
        self.assertIsNone(M.first_budget_prefix(cumulative[:2], target))
        self.assertEqual(M.first_budget_prefix(cumulative, target), 2)

    def test_persistent_reservoir_charges_one_load_and_every_generation(self) -> None:
        cumulative = M.cumulative_reservoir_compute(
            [metadata(None), metadata(None), metadata(None)]
        )
        self.assertEqual(cumulative[-1]["gpu_phase_wall_seconds"], 70.0)
        self.assertEqual(cumulative[-1]["token_forward_equivalents"], 450)
        changed_load = [metadata(None), metadata(None, load=11)]
        with self.assertRaisesRegex(ValueError, "persistent model load"):
            M.cumulative_reservoir_compute(changed_load)

    def test_training_compute_and_seed_pairing_fail_closed(self) -> None:
        forged = training(47)
        forged["compute"]["token_forward_equivalents"] += 1
        with self.assertRaisesRegex(ValueError, "invalid matched-compute"):
            M.validate_training_compute(forged)
        with self.assertRaisesRegex(ValueError, "both unique seeds"):
            M.target_compute_budget(
                [(training(47), metadata(47)), (training(47), metadata(47))],
                {47, 53},
            )

    def test_reservoir_contract_is_fixed_outcome_blind_blocks(self) -> None:
        contract = self.config["evaluation"]["frozen_sample_more"]
        self.assertEqual(contract["mode"], "outcome_blind_compute_stopped_reservoir")
        self.assertEqual(contract["same_backend"], "vllm")
        self.assertEqual(contract["block_candidate_count"], 16)
        self.assertEqual(contract["maximum_blocks"], 16)
        self.assertEqual(len(contract["block_run_seeds"]), 16)
        self.assertEqual(len(set(contract["block_run_seeds"])), 16)
        self.assertEqual(
            contract["stop_inputs"],
            "compute_receipts_only_no_labels_scores_or_correctness",
        )

    def test_final_comparison_requires_strict_paired_and_family_gain(self) -> None:
        correct = []
        coverage = {}
        family = {}
        families = ("list", "string", "register")
        for index in range(60):
            task_id = f"t{index:03d}"
            row_family = families[index % 3]
            correct.append(
                {
                    "task_id": task_id,
                    "family": row_family,
                    "coverage_at_16": float(index < 54),
                }
            )
            coverage[task_id] = float(index < 30)
            family[task_id] = row_family
        result = M._comparison(
            correct,
            {
                "coverage": coverage,
                "family": family,
                "budget_pass": True,
                "candidates_per_task": 64,
            },
            {"paired_task_resamples": 500, "seed": 991},
            self.config["decision_gates"]["final_matched_compute"],
            0,
        )
        self.assertTrue(result["pass"])
        tied = [{**row, "coverage_at_16": coverage[row["task_id"]]} for row in correct]
        tied_result = M._comparison(
            tied,
            {
                "coverage": coverage,
                "family": family,
                "budget_pass": True,
                "candidates_per_task": 64,
            },
            {"paired_task_resamples": 500, "seed": 991},
            self.config["decision_gates"]["final_matched_compute"],
            0,
        )
        self.assertFalse(tied_result["pass"])

    def test_manifest_replay_accepts_compute_stop_and_rejects_early_stop(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write(name: str, value: object) -> Path:
                path = root / name
                path.write_text(json.dumps(value, sort_keys=True) + "\n")
                return path

            input_path = write("input.jsonl", {"id": "t0"})
            labels_path = write("labels.jsonl", {"id": "t0"})
            receipt_path = write("receipt.json", {"rows": 1})
            stage_path = write("stage.json", {})
            target_paths = {}
            target_refs = []
            targets = []
            for seed in (47, 53):
                training_value = training(seed, tokens=3, wall=1.0)
                metadata_value = metadata(
                    seed, logical=1, sampled=1, load=1.0, generation=1.0
                )
                training_path = write(f"training-{seed}.json", training_value)
                metadata_path = write(f"correct-{seed}.json", metadata_value)
                target_paths[seed] = (training_path, metadata_path)
                target_refs.append(
                    {
                        "seed": seed,
                        "training_receipt": M.artifact_ref(training_path),
                        "correct_confirmation_metadata": M.artifact_ref(metadata_path),
                    }
                )
                targets.append((training_value, metadata_value))
            contract = self.config["evaluation"]["frozen_sample_more"]
            blocks = []
            block_metadata = []
            for index, seed in enumerate(contract["block_run_seeds"][:2]):
                generated = write(
                    f"block-{index}.jsonl",
                    {
                        "id": "t0",
                        "outputs": [
                            {"sample_index": sample_index}
                            for sample_index in range(16)
                        ],
                    },
                )
                metadata_value = metadata(
                    None, logical=2, sampled=1, load=1.0, generation=1.0
                )
                metadata_value.update(
                    runtime={
                        "git_root": "/synthetic",
                        "git_commit": "a" * 40,
                        "git_head_mode": "detached",
                        "cwd": "/synthetic",
                    },
                    generation_stage={"stage_receipt_path": str(stage_path.resolve())},
                    input={"sha256": M.sha256_file(input_path)},
                )
                metadata_path = write(f"block-{index}.meta.json", metadata_value)
                blocks.append(
                    {
                        "index": index,
                        "run_seed": seed,
                        "generated": M.artifact_ref(generated),
                        "metadata": M.artifact_ref(metadata_path),
                    }
                )
                block_metadata.append(metadata_value)
            target_budget = M.target_compute_budget(targets, {47, 53})
            cumulative = M.cumulative_reservoir_compute(block_metadata)
            self.assertEqual(M.first_budget_prefix(cumulative, target_budget), 1)
            manifest = {
                "schema_version": 1,
                "experiment_id": self.config["experiment_id"],
                "config_sha256": M.sha256_file(EXP / "configs" / "default.yaml"),
                "producer": {
                    "script_sha256": M.sha256_file(
                        EXP / "scripts" / "run_frozen_reservoir.py"
                    ),
                    "module_sha256": M.sha256_file(EXP / "src" / "matched_compute.py"),
                    "runner_sha256": M.sha256_file(EXP / "src" / "vllm_runner.py"),
                    "git_commit": "a" * 40,
                },
                "worktree": {
                    "repo_root": "/synthetic",
                    "git_commit": "a" * 40,
                    "head_mode": "detached",
                    "cwd": "/synthetic",
                },
                "stage_receipt": M.artifact_ref(stage_path),
                "input": M.artifact_ref(input_path),
                "input_receipt": M.artifact_ref(receipt_path),
                "targets": target_refs,
                "target_budget": target_budget,
                "block_contract": contract,
                "blocks": blocks,
                "cumulative_compute": cumulative,
                "stop": {
                    "decision": "FIRST_COMPLETE_BLOCK_REACHES_BOTH_BUDGETS",
                    "first_budget_prefix_index": 1,
                    "completed_blocks": 2,
                },
                "outcome_blindness": {
                    "accepted_label_or_score_paths": False,
                    "read_correctness_fields": False,
                    "stop_fields": [
                        "training_receipt.compute",
                        "correct_confirmation_metadata.counts",
                        "correct_confirmation_metadata.timing",
                        "frozen_block_metadata.counts",
                        "frozen_block_metadata.timing",
                    ],
                },
            }
            manifest_path = write("manifest.json", manifest)
            score_row = {
                "task_id": "t0",
                "family": "list",
                "coverage_at_16": 0.0,
            }
            patches = (
                mock.patch.object(
                    M,
                    "validate_action_inputs",
                    return_value=(
                        "confirmation",
                        {"t0": ("list", 3)},
                        {"prompt_sha256": M.sha256_file(input_path)},
                    ),
                ),
                mock.patch.object(M, "validate_sampling"),
                mock.patch.object(
                    M, "validate_generation_protocol", return_value="same-runtime"
                ),
                mock.patch.object(M, "score_generation_rows", return_value=[score_row]),
            )
            with patches[0], patches[1], patches[2], patches[3]:
                result = M.validate_reservoir_manifest(
                    manifest_path,
                    config=self.config,
                    config_path=EXP / "configs" / "default.yaml",
                    experiment_root=EXP,
                    labels_path=labels_path,
                    expected_frozen_generated=M.path_from_ref(
                        blocks[0]["generated"], "generated"
                    ),
                    expected_frozen_metadata=M.path_from_ref(
                        blocks[0]["metadata"], "metadata"
                    ),
                    expected_targets=target_paths,
                )
            self.assertTrue(result["budget_pass"])

            manifest["blocks"] = blocks[:1]
            manifest["cumulative_compute"] = cumulative[:1]
            manifest["stop"] = {
                "decision": "MAXIMUM_BLOCKS_EXHAUSTED_WITHOUT_BOTH_BUDGETS",
                "first_budget_prefix_index": None,
                "completed_blocks": 1,
            }
            manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n")
            patches = (
                mock.patch.object(
                    M,
                    "validate_action_inputs",
                    return_value=(
                        "confirmation",
                        {"t0": ("list", 3)},
                        {"prompt_sha256": M.sha256_file(input_path)},
                    ),
                ),
                mock.patch.object(M, "validate_sampling"),
                mock.patch.object(
                    M, "validate_generation_protocol", return_value="same-runtime"
                ),
            )
            with patches[0], patches[1], patches[2], self.assertRaisesRegex(
                ValueError, "stopped before"
            ):
                M.validate_reservoir_manifest(
                    manifest_path,
                    config=self.config,
                    config_path=EXP / "configs" / "default.yaml",
                    experiment_root=EXP,
                    labels_path=labels_path,
                    expected_frozen_generated=M.path_from_ref(
                        blocks[0]["generated"], "generated"
                    ),
                    expected_frozen_metadata=M.path_from_ref(
                        blocks[0]["metadata"], "metadata"
                    ),
                    expected_targets=target_paths,
                )


if __name__ == "__main__":
    unittest.main()
