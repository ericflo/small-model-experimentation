from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
RUN_PATH = EXP / "scripts" / "run.py"
SPEC = importlib.util.spec_from_file_location(
    "balanced_core_answer_potential_run_for_tests", RUN_PATH
)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import guard
    raise RuntimeError(f"could not load {RUN_PATH}")
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


class OperationContractValidatorTests(unittest.TestCase):
    def test_amendment_freezes_selection_and_validator_dependencies(self) -> None:
        frozen = set(run.AMENDMENT_FROZEN_FILES)
        prefix = "experiments/qwen35_4b_balanced_core_answer_potential_sft/"
        for relative in (
            "configs/default.yaml",
            "data/procedural/train.jsonl",
            "scripts/run.py",
            "src/io_utils.py",
            "src/selector.py",
            "src/shards.py",
            "src/task_data.py",
        ):
            self.assertIn(prefix + relative, frozen)
        self.assertFalse(any("/." in relative for relative in frozen))

    def test_amendment_commit_is_stable_and_recovers_after_rebase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = root / "receipt.json"
            write_json(receipt, {"amendment_commit": "commit-a"})

            def stable_run(command: list[str], *, check: bool = True):
                del check
                if command[1:3] == ["rev-parse", "HEAD"]:
                    output, returncode = "commit-c\n", 0
                elif command[1:3] == ["merge-base", "--is-ancestor"]:
                    output, returncode = "", 0
                else:  # pragma: no cover - unexpected command is a test failure
                    self.fail(f"unexpected stable command: {command}")
                return subprocess.CompletedProcess(command, returncode, output, "")

            with (
                mock.patch.object(run, "ROOT", root),
                mock.patch.object(run, "AMENDMENT_RECEIPT_PATH", receipt),
                mock.patch.object(run, "_run", side_effect=stable_run),
            ):
                self.assertEqual(
                    run.resolve_preselection_amendment_commit(), "commit-a"
                )

            def rebased_run(command: list[str], *, check: bool = True):
                del check
                if command[1:3] == ["rev-parse", "HEAD"]:
                    output, returncode = "commit-b-prime\n", 0
                elif command[1:3] == ["merge-base", "--is-ancestor"]:
                    output, returncode = "", 1
                elif command[1:3] == ["cat-file", "-e"]:
                    output, returncode = "", 0
                elif command[1] == "log":
                    output, returncode = "commit-b-prime\n", 0
                elif command[1:3] == ["rev-parse", "commit-b-prime^"]:
                    output, returncode = "commit-a-prime\n", 0
                else:  # pragma: no cover - unexpected command is a test failure
                    self.fail(f"unexpected rebased command: {command}")
                return subprocess.CompletedProcess(command, returncode, output, "")

            with (
                mock.patch.object(run, "ROOT", root),
                mock.patch.object(run, "AMENDMENT_RECEIPT_PATH", receipt),
                mock.patch.object(run, "_run", side_effect=rebased_run),
            ):
                self.assertEqual(
                    run.resolve_preselection_amendment_commit(),
                    "commit-a-prime",
                )

    def test_selection_rejects_stale_upstream_operation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = {
                "train_independent": "f" * 64,
                "train_independent_scores": "b" * 64,
                "train_rollouts_r1": "c" * 64,
            }
            for stage, contract in stages.items():
                write_json(
                    root / "pools" / stage / "index.json",
                    {
                        "schema_version": 1,
                        "operation_contract_sha256": contract,
                        "shards": {},
                    },
                )
            config = {
                "sampling": {
                    "train_independent_n": 64,
                    "train_rollouts_per_trace": 1,
                },
                "scoring": {"backend": "unused-after-contract-failure"},
            }
            with (
                mock.patch.object(run, "external_root", return_value=root),
                mock.patch.object(
                    run, "raw_pool_operation_contract", return_value="a" * 64
                ),
                mock.patch.object(
                    run, "score_operation_contract", return_value="b" * 64
                ),
                mock.patch.object(
                    run, "rollout_operation_contract", return_value="c" * 64
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "independent operation contract mismatch"
                ):
                    run.selection_evidence_bundle(
                        config, allow_missing_contracts=False
                    )

    def test_selection_validation_does_not_backfill_missing_contracts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for stage in (
                "train_independent",
                "train_independent_scores",
                "train_rollouts_r1",
            ):
                path = root / "pools" / stage / "index.json"
                write_json(path, {"schema_version": 1, "shards": {}})
                paths.append(path)
            before = {path: path.read_bytes() for path in paths}
            config = {
                "sampling": {
                    "train_independent_n": 64,
                    "train_rollouts_per_trace": 1,
                }
            }
            with (
                mock.patch.object(run, "external_root", return_value=root),
                mock.patch.object(
                    run,
                    "selection_evidence_contracts",
                    return_value={
                        "independent": "a" * 64,
                        "independent_scores": "b" * 64,
                        "rollouts": "c" * 64,
                    },
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "independent operation contract is not sealed"
                ):
                    run.selection_evidence_bundle(
                        config, allow_missing_contracts=False
                    )
            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_incomplete_preseal_validation_is_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for stage in (
                "train_independent",
                "train_independent_scores",
                "train_rollouts_r1",
            ):
                path = root / "pools" / stage / "index.json"
                value = {
                    "schema_version": 1,
                    "stage": stage,
                    "split": "train",
                    "model": run.MODEL_ID,
                    "revision": run.MODEL_REVISION,
                    "shards": {},
                }
                if stage == "train_independent_scores":
                    value["backend"] = "test-backend"
                write_json(path, value)
                paths.append(path)
            before = {path: path.read_bytes() for path in paths}
            config = {
                "sampling": {"train_independent_n": 64},
                "scoring": {"backend": "test-backend"},
                "splits": {"full_train_tasks": 1},
            }
            with (
                mock.patch.object(run, "external_root", return_value=root),
                mock.patch.object(
                    run,
                    "selection_evidence_contracts",
                    return_value={
                        "independent": "a" * 64,
                        "independent_scores": "b" * 64,
                        "rollouts": "c" * 64,
                    },
                ),
                mock.patch.object(
                    run,
                    "load_core_train",
                    return_value=[{"id": "task-1"}],
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "exact frozen task scope"
                ):
                    run.selection_evidence_bundle(
                        config, allow_missing_contracts=True
                    )
            self.assertEqual(before, {path: path.read_bytes() for path in paths})

    def test_partial_operation_attestation_restart_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stages = {
                "independent": "train_independent",
                "independent_scores": "train_independent_scores",
                "rollouts": "train_rollouts_r1",
            }
            contracts = {
                "independent": "a" * 64,
                "independent_scores": "b" * 64,
                "rollouts": "c" * 64,
            }
            indices = {}
            for key, stage in stages.items():
                value = {"schema_version": 1, "shards": {}}
                if key == "independent":
                    value["operation_contract_sha256"] = contracts[key]
                path = root / "pools" / stage / "index.json"
                write_json(path, value)
                indices[key] = value
            preserved_path = root / "pools" / "train_independent" / "index.json"
            preserved = preserved_path.read_bytes()
            with mock.patch.object(run, "external_root", return_value=root):
                run.apply_selection_operation_contracts(
                    {}, indices, contracts
                )
                first = {
                    stage: (root / "pools" / stage / "index.json").read_bytes()
                    for stage in stages.values()
                }
                run.apply_selection_operation_contracts(
                    {}, indices, contracts
                )
                second = {
                    stage: (root / "pools" / stage / "index.json").read_bytes()
                    for stage in stages.values()
                }
            self.assertEqual(first, second)
            self.assertEqual(preserved, preserved_path.read_bytes())

    def test_selection_rejects_a_missing_committed_amendment_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(
                run,
                "AMENDMENT_RECEIPT_PATH",
                Path(directory) / "missing.json",
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "missing committed pre-selection amendment receipt"
                ):
                    run.validate_preselection_amendment_receipt({})


class EvaluationCacheContractTests(unittest.TestCase):
    def test_stale_contract_and_fingerprint_force_cache_regeneration(self) -> None:
        class FakeRunner:
            generations = 0

            def __init__(self, _engine: object) -> None:
                pass

            def __enter__(self) -> "FakeRunner":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def generate(
                self, _records: list[dict[str, object]], _sampling: object
            ) -> tuple[list[dict[str, object]], dict[str, object]]:
                type(self).generations += 1
                return [{"id": "task-1", "outputs": []}], {"fake": True}

        item = {
            "id": "task-1",
            "prompt": "toy prompt",
            "canonical_answer": "7",
            "family": "toy",
            "level": 1,
        }
        config = {
            "sft": {"arms": []},
            "engine": {"test_engine_contract": 1},
            "evaluation": {
                "stage_a_splits": ["core_iid"],
                "stage_b_full_splits": ["core_iid"],
                "sampled_k": 8,
                "natural_max_tokens": 64,
                "sampling_temperature": 0.6,
                "sampling_top_p": 0.95,
                "sampling_top_k": 20,
                "greedy_seed": 11,
                "sample_seed": 12,
            },
        }

        def score_rows(
            _items: dict[str, dict[str, object]], rows: list[dict[str, object]]
        ) -> tuple[list[dict[str, object]], dict[str, float]]:
            return rows, {"sample_accuracy": 0.0, "pass_at_n": 0.0}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runs = root / "tracked_runs"
            with (
                mock.patch.object(run, "RUNS_DIR", runs),
                mock.patch.object(run, "validate_preselection_amendment_receipt"),
                mock.patch.object(run, "design_boundary_receipt"),
                mock.patch.object(run, "external_root", return_value=root),
                mock.patch.object(
                    run, "load_evaluation_scope", return_value=[item]
                ),
                mock.patch.object(run, "engine_config", return_value=object()),
                mock.patch.object(run, "VLLMRunner", FakeRunner),
                mock.patch.object(run, "_score_evaluation_rows", side_effect=score_rows),
            ):
                kwargs = {
                    "mode": "greedy",
                    "phase": "stage_a",
                    "arms_override": ["base"],
                }
                run.evaluate_matrix(config, **kwargs)
                self.assertEqual(FakeRunner.generations, 1)

                # An unchanged, valid cache is reused.
                run.evaluate_matrix(config, **kwargs)
                self.assertEqual(FakeRunner.generations, 1)

                index_path = (
                    root
                    / "evaluation"
                    / "seed42"
                    / "stage_a"
                    / "greedy"
                    / "index.json"
                )
                index = json.loads(index_path.read_text(encoding="utf-8"))
                entry = index["arms"]["base"]["core_iid"]
                entry["contract_sha256"] = "0" * 64
                write_json(index_path, index)
                run.evaluate_matrix(config, **kwargs)
                self.assertEqual(FakeRunner.generations, 2)

                index = json.loads(index_path.read_text(encoding="utf-8"))
                entry = index["arms"]["base"]["core_iid"]
                entry["model_fingerprint"] = "1" * 64
                write_json(index_path, index)
                run.evaluate_matrix(config, **kwargs)
                self.assertEqual(FakeRunner.generations, 3)


class SelectedDatasetManifestTests(unittest.TestCase):
    def test_manifest_binds_exact_dataset_and_amendment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root / "experiments" / "synthetic"
            runs = exp / "runs"
            config_path = exp / "configs" / "default.yaml"
            selector_path = exp / "src" / "selector.py"
            dataset = root / "external" / "answer_potential.jsonl.gz"
            for path, payload in (
                (config_path, b"config\n"),
                (selector_path, b"# selector\n"),
                (dataset, b"selected-dataset"),
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
            amendment_path = runs / "preselection_amendment_receipt.json"
            evidence_indexes = {"synthetic": {"index_sha256": "a" * 64}}
            write_json(
                amendment_path,
                {
                    "amendment_commit": "commit-a",
                    "evidence_indexes": evidence_indexes,
                },
            )
            contracts = {"synthetic": "b" * 64}
            artifact = {
                "path": str(dataset),
                "sha256": run.sha256_file(dataset),
                "bytes": dataset.stat().st_size,
                "rows": 1,
            }
            manifest = {
                "preselection_amendment": {
                    "receipt": str(amendment_path),
                    "receipt_sha256": run.sha256_file(amendment_path),
                    "amendment_commit": "commit-a",
                    "evidence_indexes": evidence_indexes,
                    "operation_contracts": contracts,
                    "selector_sha256": run.sha256_file(selector_path),
                    "run_sha256": run.sha256_file(RUN_PATH),
                    "config_sha256": run.sha256_file(config_path),
                },
                "arms": {
                    "answer_potential": {
                        "rows": 1,
                        "artifact": artifact,
                    }
                },
            }
            manifest_path = exp / "data" / "sft_manifest.json"
            summary_path = runs / "selection_summary.json"
            write_json(manifest_path, manifest)
            write_json(summary_path, manifest)

            def committed_sha(_commit: str, relative: str) -> str:
                return run.sha256_file(root / relative)

            with (
                mock.patch.object(run, "ROOT", root),
                mock.patch.object(run, "EXP", exp),
                mock.patch.object(run, "RUNS_DIR", runs),
                mock.patch.object(run, "CONFIG_PATH", config_path),
                mock.patch.object(run, "AMENDMENT_RECEIPT_PATH", amendment_path),
                mock.patch.object(
                    run,
                    "selection_evidence_contracts",
                    return_value=contracts,
                ),
                mock.patch.object(
                    run, "git_file_sha256", side_effect=committed_sha
                ),
            ):
                got = run.validate_selected_dataset_manifest(
                    {}, arm="answer_potential", dataset=dataset
                )
                self.assertEqual(got["artifact"]["sha256"], artifact["sha256"])
                dataset.write_bytes(b"changed-selected-dataset")
                with self.assertRaisesRegex(
                    RuntimeError, "selected dataset artifact mismatch"
                ):
                    run.validate_selected_dataset_manifest(
                        {}, arm="answer_potential", dataset=dataset
                    )


class TrainingReceiptValidatorTests(unittest.TestCase):
    def make_fixture(
        self, root: Path
    ) -> tuple[dict[str, object], Path, Path, dict[str, object]]:
        exp = root / "experiment"
        repo = root / "repo"
        dataset = root / "answer_potential.jsonl.gz"
        output = root / "adapter"
        output.mkdir(parents=True)
        dataset.write_bytes(b"fixed-dataset")
        weights = output / "adapter_model.safetensors"
        weights.write_bytes(b"fixed-adapter")
        train_script = exp / "scripts" / "train_think.py"
        train_script.parent.mkdir(parents=True)
        train_script.write_text("# fixed trainer\n", encoding="utf-8")
        training_lock = repo / "requirements-training.lock.txt"
        training_lock.parent.mkdir(parents=True)
        training_lock.write_text("fixed==1\n", encoding="utf-8")
        write_json(
            exp / "data" / "sft_manifest.json",
            {
                "arms": {
                    "answer_potential": {
                        "rows": 16,
                        "forward_tokens": 100,
                        "supervised_weighted_tokens": 50.25,
                    }
                }
            },
        )
        amendment_receipt = exp / "runs" / "preselection_amendment_receipt.json"
        write_json(amendment_receipt, {"synthetic": True})
        config: dict[str, object] = {
            "sft": {
                "epochs": 2.0,
                "batch_size": 1,
                "gradient_accumulation": 16,
                "weight_prompt": 0.0,
                "weight_think": 0.5,
                "weight_close_answer": 1.0,
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "learning_rate": 2e-4,
                "max_length": 16_000,
            }
        }
        receipt: dict[str, object] = {
            "arm": "answer_potential",
            "seed": 42,
            "smoke": False,
            "model": run.MODEL_ID,
            "revision": run.MODEL_REVISION,
            "dataset_sha256": run.sha256_file(dataset),
            "selection_manifest_sha256": run.sha256_file(
                exp / "data" / "sft_manifest.json"
            ),
            "preselection_amendment_receipt_sha256": run.sha256_file(
                amendment_receipt
            ),
            "rows": 16,
            "epochs": 2.0,
            "completed_epochs": 2.0,
            "optimizer_steps": 2,
            "skipped_rows": 0,
            "loss_weights": {
                "prompt": 0.0,
                "think": 0.5,
                "close_answer": 1.0,
            },
            "training_contract": {
                "script_sha256": run.sha256_file(train_script),
                "rank": 32,
                "alpha": 64,
                "dropout": 0.05,
                "learning_rate": 2e-4,
                "batch_size": 1,
                "gradient_accumulation": 16,
                "max_length": 16_000,
                "optimizer": "paged_adamw_8bit",
                "scheduler": "cosine",
                "warmup_ratio": 0.03,
            },
            "seed_contract": {
                "global_seed_before_model": True,
                "lora_seed_reset_before_adapter_init": True,
                "trainer_seed": 42,
                "data_seed": 42,
            },
            "adapter_initial_state": {"sha256": "2" * 64, "tensors": 256},
            "adapter_tensor_manifest": {"sha256": "3" * 64, "tensors": 256},
            "training_lock": {"sha256": run.sha256_file(training_lock)},
            "dataset_forward_tokens_one_pass": 100,
            "actual_examples_seen": 32,
            "actual_forward_tokens_seen": 200,
            "dataset_supervised_weighted_tokens_one_pass": 50.25,
            "actual_supervised_weighted_tokens_seen": 100.5,
            "artifacts": {
                "adapter_model.safetensors": {"sha256": run.sha256_file(weights)}
            },
        }
        return config, dataset, output, receipt

    def validate(
        self,
        config: dict[str, object],
        dataset: Path,
        output: Path,
        receipt: dict[str, object],
    ) -> None:
        with mock.patch.object(
            run,
            "validate_selected_dataset_manifest",
            return_value={
                "rows": 16,
                "forward_tokens": 100,
                "supervised_weighted_tokens": 50.25,
            },
        ):
            run.validate_training_receipt(
                config,
                arm="answer_potential",
                seed=42,
                dataset=dataset,
                output=output,
                receipt=receipt,
            )

    def test_valid_receipt_passes_and_stale_fields_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, dataset, output, receipt = self.make_fixture(root)
            with (
                mock.patch.object(run, "EXP", root / "experiment"),
                mock.patch.object(run, "ROOT", root / "repo"),
                mock.patch.object(
                    run,
                    "AMENDMENT_RECEIPT_PATH",
                    root
                    / "experiment"
                    / "runs"
                    / "preselection_amendment_receipt.json",
                ),
            ):
                self.validate(config, dataset, output, receipt)

                stale_cases = {
                    "dataset_sha256": ("dataset_sha256", "4" * 64),
                    "optimizer_steps": ("optimizer_steps", 1),
                    "training_contract": (
                        "training_contract",
                        {**receipt["training_contract"], "learning_rate": 1e-4},
                    ),
                    "actual_supervised_tokens_seen": (
                        "actual_supervised_weighted_tokens_seen",
                        100.0,
                    ),
                }
                for failed_check, (field, stale_value) in stale_cases.items():
                    with self.subTest(failed_check=failed_check):
                        stale = copy.deepcopy(receipt)
                        stale[field] = stale_value
                        with self.assertRaisesRegex(RuntimeError, failed_check):
                            self.validate(config, dataset, output, stale)

    def test_stale_adapter_artifact_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, dataset, output, receipt = self.make_fixture(root)
            (output / "adapter_model.safetensors").write_bytes(b"changed-adapter")
            with (
                mock.patch.object(run, "EXP", root / "experiment"),
                mock.patch.object(run, "ROOT", root / "repo"),
                mock.patch.object(
                    run,
                    "AMENDMENT_RECEIPT_PATH",
                    root
                    / "experiment"
                    / "runs"
                    / "preselection_amendment_receipt.json",
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "invalid cached training artifact"
                ):
                    self.validate(config, dataset, output, receipt)


class MergedCheckpointChainTests(unittest.TestCase):
    def test_valid_chain_passes_and_stale_application_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arm = "answer_potential"
            adapter = root / "adapters" / "seed42" / arm
            merged = root / "merged" / "seed42" / arm
            adapter.mkdir(parents=True)
            merged.mkdir(parents=True)
            adapter_sha256 = "a" * 64
            tensor_manifest = {"sha256": "b" * 64, "tensors": 256}
            write_json(
                adapter / "training_receipt.json",
                {
                    "artifacts": {
                        "adapter_model.safetensors": {
                            "sha256": adapter_sha256
                        }
                    },
                    "adapter_tensor_manifest": tensor_manifest,
                },
            )
            application_path = merged / "merge_application_receipt.json"
            write_json(
                application_path,
                {
                    "adapter_sha256": adapter_sha256,
                    "applied_lora_pairs": 128,
                },
            )
            fingerprint = {"sha256": "c" * 64, "files": []}
            write_json(
                merged / "merge_receipt.json",
                {
                    "arm": arm,
                    "seed": 42,
                    "adapter_sha256": adapter_sha256,
                    "adapter_tensor_manifest": tensor_manifest,
                    "applied_lora_pairs": 128,
                    "merge_application_receipt_sha256": run.sha256_file(
                        application_path
                    ),
                    "merged_checkpoint_fingerprint": fingerprint,
                },
            )
            with (
                mock.patch.object(run, "external_root", return_value=root),
                mock.patch.object(run, "validate_training_receipt"),
                mock.patch.object(
                    run, "checkpoint_fingerprint", return_value=fingerprint
                ),
            ):
                self.assertEqual(
                    run.validated_merged_fingerprint({}, arm), "c" * 64
                )
                write_json(
                    application_path,
                    {
                        "adapter_sha256": "d" * 64,
                        "applied_lora_pairs": 128,
                    },
                )
                with self.assertRaisesRegex(
                    RuntimeError, "merged checkpoint fingerprint mismatch"
                ):
                    run.validated_merged_fingerprint({}, arm)


if __name__ == "__main__":
    unittest.main()
