from __future__ import annotations

import json
import hashlib
import os
import copy
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import scientific_artifacts as artifacts  # noqa: E402


ENGINE = {
    "max_model_len": 65536,
    "gpu_memory_utilization": 0.9,
    "max_num_seqs": 64,
    "max_num_batched_tokens": 32768,
    "enable_prefix_caching": False,
    "enforce_eager": False,
    "adapter": None,
}


class ScientificArtifactTests(unittest.TestCase):
    def _protocol_binding(self) -> dict:
        core = {
            "schema_version": 1,
            "experiment_id": artifacts.EXPERIMENT_ID,
            "files": [],
            "smoke_libraries": {
                "base": {"library_id": "base-fixture", "content_sha256": "b" * 64},
                "designed_ceiling": {
                    "library_id": "designed-fixture",
                    "content_sha256": "d" * 64,
                },
            },
            "library_scope": "fixture",
        }
        payload = json.dumps(
            core, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        return {**core, "binding_sha256": hashlib.sha256(payload).hexdigest()}

    def _sampling(self, *, budget: int, k: int) -> dict:
        return {
            "thinking": "budget",
            "thinking_budget": budget,
            "n": k,
            "max_tokens": 512,
            "answer_max_tokens": 512,
            "greedy": False,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "run_seed": 2701,
            "shuffle_thinking": False,
            "logprobs": None,
            "prompt_logprobs": None,
            "logprob_token_ids": [],
            "allow_custom_prompts": False,
        }

    def _preflight(
        self,
        arm: str,
        *,
        count: int = artifacts.SCIENTIFIC_N_RECORDS,
        budget: int = 32768,
    ) -> dict:
        reserve = budget + artifacts.FORCED_CLOSE_TOKENS + 512
        records = []
        for index in range(count):
            records.append(
                {
                    "id": f"task-{index}::{arm}",
                    "input_record_sha256": f"{index + 1:064x}",
                    "rendered_prompt_sha256": f"{index + 101:064x}",
                    "prompt_tokens": 100 + index,
                    "prompt_plus_reserve_tokens": reserve + 100 + index,
                }
            )
        return {
            "schema_version": 1,
            "pass": True,
            "max_model_len": 65536,
            "generation_reserve_tokens": reserve,
            "n_records": count,
            "min_prompt_tokens": 100,
            "max_prompt_tokens": 100 + count - 1,
            "max_prompt_plus_reserve_tokens": reserve + 99 + count,
            "records": records,
        }

    def _metadata(self, *, rows: list[dict], budget: int, k: int) -> dict:
        unique_prompt_tokens = sum(int(row["n_prompt_tokens"]) for row in rows)
        stage1_logical = sum(
            int(output["n_stage1_prompt_tokens"])
            for row in rows
            for output in row["outputs"]
        )
        stage2_logical = sum(
            int(output["n_stage2_prompt_tokens"])
            for row in rows
            for output in row["outputs"]
        )
        sampled_tokens = sum(
            int(output["n_sampled_tokens"])
            for row in rows
            for output in row["outputs"]
        )
        injected_tokens = sum(
            int(output["n_injected_tokens"])
            for row in rows
            for output in row["outputs"]
        )
        return {
            "schema_version": artifacts.RUNNER_SCHEMA_VERSION,
            "model": artifacts.MODEL_ID,
            "model_revision": artifacts.MODEL_REVISION,
            "runner_sha256": artifacts.RUNNER_SHA256,
            "adapter": None,
            "sampling": self._sampling(budget=budget, k=k),
            "resolved_sampling": dict(artifacts._EXPECTED_RESOLVED_SAMPLING),
            "engine": dict(ENGINE),
            "engine_args": dict(artifacts._EXPECTED_ENGINE_ARGS),
            "think_token_ids": json.loads(json.dumps(artifacts._EXPECTED_THINK_TOKEN_IDS)),
            "termination": dict(artifacts._EXPECTED_TERMINATION),
            "rng_isolation": dict(artifacts._EXPECTED_RNG_ISOLATION),
            "counts": {
                "requests": len(rows),
                "completions": len(rows) * k,
                "unique_input_prompt_tokens": unique_prompt_tokens,
                "stage1_logical_prompt_tokens": stage1_logical,
                "stage2_logical_prompt_tokens": stage2_logical,
                "logical_model_input_tokens": stage1_logical + stage2_logical,
                "sampled_tokens": sampled_tokens,
                "injected_tokens": injected_tokens,
            },
        }

    @staticmethod
    def _selection_artifacts(paths: artifacts.BundlePaths) -> dict[str, str]:
        return {
            "rows": hashlib.sha256(paths.rows.read_bytes()).hexdigest(),
            "meta": hashlib.sha256(paths.metadata.read_bytes()).hexdigest(),
            "preflight": hashlib.sha256(paths.preflight.read_bytes()).hexdigest(),
            "receipt": hashlib.sha256(paths.receipt.read_bytes()).hexdigest(),
        }

    def _passing_selection(
        self, *, budget: int, completed: dict[str, dict]
    ) -> dict:
        return {
            "schema_version": 1,
            "run": "smoke",
            "pass": True,
            "selected_thinking_budget": budget,
            "selection_rule": "first adequate tier",
            "selection_uses_output_content": False,
            "lower_tiers_excluded_from_scoring": True,
            "scientific_probe_k": artifacts.SCIENTIFIC_PROBE_K,
            "probe_policy": "frozen fixture",
            "probes_excluded_from_promotion_scoring_and_prefix_pooling": True,
            "tiers": [
                {
                    "budget": budget,
                    "status": "selectable",
                    "tier_mode": "complete_k12_matrix",
                    "complete": True,
                    "adequate": True,
                    "rejecting_arm": None,
                    "scientific_probe": None,
                    "arms": {
                        arm: {
                            "status": "complete",
                            "termination": {"adequate": True},
                            "artifacts": self._selection_artifacts(completed[arm]["paths"]),
                        }
                        for arm in artifacts.SCIENTIFIC_MATRIX_ARMS
                    },
                }
            ],
        }

    def _write_complete_inputs(
        self,
        root: Path,
        prefix: str,
        *,
        arm: str,
        k: int | None = None,
        budget: int = 32768,
        role: str = "complete_matrix_arm",
        tier_mode: str = "complete_k12_matrix",
        commit: bool = True,
    ) -> dict:
        if k is None:
            k = (
                artifacts.SCIENTIFIC_PROBE_K
                if role == "termination_probe"
                else artifacts.SCIENTIFIC_MATRIX_K
            )
        preflight = self._preflight(arm, budget=budget)
        artifacts.write_preflight_only(root, prefix, preflight)
        paths = artifacts.bundle_paths(root, prefix)
        rows = []
        for record in preflight["records"]:
            task_id = str(record["id"]).removesuffix(f"::{arm}")
            parent_seed = artifacts._stable_seed(
                artifacts.SCIENTIFIC_RUN_SEED, str(record["id"]), -1, "stage1"
            )
            rows.append(
                {
                    "id": record["id"],
                    "meta": {
                        "task_id": task_id,
                        "split": "smoke_reuse" if len(rows) >= 6 else "smoke_no_reuse",
                        "arm": arm,
                        "library_id": f"fixture-{arm}",
                        "max_surface_calls": 5,
                        "max_expanded_primitive_depth": 5,
                        "macros_callable": arm != "base",
                        "prompt_kind": "solve_program",
                    },
                    "prompt_sha256": record["rendered_prompt_sha256"],
                    "n_prompt_tokens": record["prompt_tokens"],
                    "prompt_channel": "thinking",
                    "prompt_logprobs": None,
                    "outputs": [
                        {
                            "sample_index": sample_index,
                            "stage1_parent_seed": parent_seed,
                            "seed_stage1": parent_seed + sample_index,
                            "seed_stage2": None,
                            "text": "opaque and deliberately uninspected",
                            "token_ids": [248069, sample_index + 1],
                            "stage1_token_ids": [248069, sample_index + 1, 248044],
                            "injected_token_ids": [],
                            "stage2_token_ids": [],
                            "n_thinking_tokens": 0,
                            "n_answer_tokens": 1,
                            "n_sampled_tokens": 3,
                            "n_injected_tokens": 0,
                            "n_completion_tokens": 2,
                            "n_terminal_tokens_trimmed": 1,
                            "n_stage1_prompt_tokens": record["prompt_tokens"],
                            "n_stage2_prompt_tokens": 0,
                            "thinking_closed": True,
                            "forced_close": False,
                            "finish_reason": "stop",
                            "stop_reason": 248044,
                            "stage1_finish_reason": "stop",
                            "stage1_stop_reason": 248044,
                            "truncated": False,
                        }
                        for sample_index in range(k)
                    ],
                }
            )
        paths.rows.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        sampling = self._sampling(budget=budget, k=k)
        metadata = self._metadata(rows=rows, budget=budget, k=k)
        paths.metadata.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        expected_identity = {
            "model": artifacts.MODEL_ID,
            "model_revision": artifacts.MODEL_REVISION,
            "runner_sha256": artifacts.RUNNER_SHA256,
            "sampling": sampling,
            "engine": ENGINE,
        }
        receipt = None
        if commit:
            receipt = artifacts.commit_receipt(
                root,
                prefix,
                role=role,
                tier_mode=tier_mode,
                thinking_budget=budget,
                arm=arm,
                k=k,
                expected_identity=expected_identity,
            )
        return {
            "paths": paths,
            "receipt": receipt,
            "identity": expected_identity,
            "rows": rows,
            "metadata": metadata,
        }

    def test_root_precedence_and_absolute_requirement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            explicit = Path(directory) / "explicit"
            environmental = Path(directory) / "environmental"
            env = {artifacts.ARTIFACT_ROOT_ENV: str(environmental)}
            self.assertEqual(artifacts.resolve_artifact_root(explicit, environ=env), explicit)
            self.assertEqual(artifacts.resolve_artifact_root(environ=env), environmental)
            self.assertEqual(
                artifacts.resolve_artifact_root(environ={}), artifacts.DEFAULT_ARTIFACT_ROOT
            )
        with self.assertRaisesRegex(artifacts.ScientificArtifactError, "absolute"):
            artifacts.resolve_artifact_root("relative/artifacts", environ={})

    def test_containment_namespace_and_symlinks_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "traversal"):
                artifacts.safe_path(root, "../escape.json")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "must be relative"):
                artifacts.safe_path(root, "/escape.json")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "bundle prefix"):
                artifacts.bundle_paths(root, "smoke/think_32768/base")

            target = Path(directory) / "real-root"
            target.mkdir()
            linked_root = Path(directory) / "linked-root"
            linked_root.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                artifacts.resolve_artifact_root(linked_root)

            internal_target = Path(directory) / "internal-target"
            internal_target.mkdir()
            (root / "smoke_tiers").symlink_to(internal_target, target_is_directory=True)
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                artifacts.bundle_paths(root, "smoke_tiers/think_32768/base")

    def test_preflight_only_is_the_one_valid_incomplete_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            self.assertEqual(artifacts.bundle_state(root, prefix)["status"], "absent")
            preflight = self._preflight("base")
            first = artifacts.write_preflight_only(root, prefix, preflight)
            second = artifacts.write_preflight_only(root, prefix, preflight)
            self.assertEqual(first, second)
            state = artifacts.bundle_state(root, prefix)
            self.assertEqual(state["status"], "preflight_only")
            self.assertEqual(state["n_records"], artifacts.SCIENTIFIC_N_RECORDS)
            self.assertEqual(state["k"], artifacts.SCIENTIFIC_MATRIX_K)

            changed = self._preflight("base")
            changed["records"][0]["prompt_tokens"] += 1
            with self.assertRaisesRegex(
                artifacts.ScientificArtifactError, "mismatch|differs"
            ):
                artifacts.write_preflight_only(root, prefix, changed)

            paths = artifacts.bundle_paths(root, prefix)
            paths.rows.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "partial"):
                artifacts.bundle_state(root, prefix)

    def test_fixed_geometry_rejects_budget_arm_record_and_k_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "unregistered"):
                artifacts.bundle_paths(root, "smoke_tiers/think_8192/base")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "base-only"):
                artifacts.bundle_paths(
                    root, "smoke_budget_probes/think_49152/designed_ceiling"
                )
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "exactly 12"):
                artifacts.write_preflight_only(
                    root,
                    "smoke_tiers/think_32768/base",
                    self._preflight("base", count=11),
                )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "K must be 12"):
                self._write_complete_inputs(
                    root,
                    "smoke_tiers/think_32768/base",
                    arm="base",
                    k=artifacts.SCIENTIFIC_PROBE_K,
                )

    def test_exact_runner_protocol_and_row_identity_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            result = self._write_complete_inputs(
                root,
                "smoke_tiers/think_32768/base",
                arm="base",
                commit=False,
            )
            valid = result["metadata"]
            self.assertEqual(
                artifacts._identity_from_metadata(
                    valid,
                    n_records=artifacts.SCIENTIFIC_N_RECORDS,
                    k=artifacts.SCIENTIFIC_MATRIX_K,
                    thinking_budget=32768,
                )["runner_schema_version"],
                artifacts.RUNNER_SCHEMA_VERSION,
            )
            mutations = {
                "runner schema": lambda value: value.__setitem__("schema_version", 2),
                "adapter": lambda value: value.__setitem__("adapter", {"path": "forbidden"}),
                "sampling": lambda value: value["sampling"].__setitem__("temperature", 0.7),
                "resolved sampling": lambda value: value["resolved_sampling"].__setitem__("top_p", 0.9),
                "engine": lambda value: value["engine"].__setitem__("max_num_seqs", 32),
                "engine args": lambda value: value["engine_args"].__setitem__("async_scheduling", True),
                "think ids": lambda value: value["think_token_ids"].__setitem__("close", 1),
                "termination": lambda value: value["termination"].__setitem__("hf_model_eos_token_id", 1),
                "rng": lambda value: value["rng_isolation"].__setitem__("engine_seed", 1),
            }
            for label, mutate in mutations.items():
                with self.subTest(label=label):
                    changed = copy.deepcopy(valid)
                    mutate(changed)
                    with self.assertRaises(artifacts.ScientificArtifactError):
                        artifacts._identity_from_metadata(
                            changed,
                            n_records=artifacts.SCIENTIFIC_N_RECORDS,
                            k=artifacts.SCIENTIFIC_MATRIX_K,
                            thinking_budget=32768,
                        )

            ordered = artifacts._ordered_records(
                json.loads(result["paths"].preflight.read_text(encoding="utf-8"))
            )
            row_mutations = {
                "task": lambda rows: rows[0]["meta"].pop("task_id"),
                "arm": lambda rows: rows[0]["meta"].__setitem__("arm", "designed_ceiling"),
                "prompt": lambda rows: rows[0].__setitem__("prompt_channel", "final"),
                "sample": lambda rows: rows[0]["outputs"][0].__setitem__("sample_index", 1),
                "seed": lambda rows: rows[0]["outputs"][0].__setitem__("stage1_parent_seed", 1),
            }
            for label, mutate in row_mutations.items():
                with self.subTest(label=f"row-{label}"):
                    changed_rows = copy.deepcopy(result["rows"])
                    mutate(changed_rows)
                    mutated_path = root / f"mutated-{label}.jsonl"
                    mutated_path.write_text(
                        "".join(json.dumps(row) + "\n" for row in changed_rows),
                        encoding="utf-8",
                    )
                    with self.assertRaises(artifacts.ScientificArtifactError):
                        artifacts._validate_rows_identity(
                            mutated_path,
                            ordered_records=ordered,
                            k=artifacts.SCIENTIFIC_MATRIX_K,
                            arm="base",
                        )

    def test_amendment9_answer_restart_row_remains_receiptable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            result = self._write_complete_inputs(
                root,
                "smoke_tiers/think_32768/base",
                arm="base",
                commit=False,
            )
            row = result["rows"][0]
            output = row["outputs"][0]
            output.update(
                {
                    "seed_stage2": artifacts._stable_seed(
                        artifacts.SCIENTIFIC_RUN_SEED, row["id"], 0, "stage2"
                    ),
                    "token_ids": [17, 248069, 271, 42],
                    "stage1_token_ids": [17, 248069, 99, 248044],
                    "retained_thinking_token_ids": [17],
                    "injected_token_ids": [248069, 271],
                    "stage2_token_ids": [42, 248044],
                    "n_thinking_tokens": 1,
                    "n_answer_tokens": 1,
                    "n_sampled_tokens": 6,
                    "n_injected_tokens": 2,
                    "n_completion_tokens": 4,
                    "n_terminal_tokens_trimmed": 2,
                    "n_stage2_prompt_tokens": row["n_prompt_tokens"] + 3,
                    "forced_close": False,
                    "stage1_finish_reason": "length",
                    "stage1_stop_reason": None,
                }
            )
            result["paths"].rows.write_text(
                "".join(json.dumps(item) + "\n" for item in result["rows"]),
                encoding="utf-8",
            )
            metadata = self._metadata(
                rows=result["rows"],
                budget=32768,
                k=artifacts.SCIENTIFIC_MATRIX_K,
            )
            result["paths"].metadata.write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            receipt = artifacts.commit_receipt(
                root,
                "smoke_tiers/think_32768/base",
                role="complete_matrix_arm",
                tier_mode="complete_k12_matrix",
                thinking_budget=32768,
                arm="base",
                k=artifacts.SCIENTIFIC_MATRIX_K,
            )
            self.assertEqual(receipt["n_completions"], 144)
            artifacts.fsync_tree_and_parent(root)

    def test_receipt_is_last_written_and_binds_every_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            paths = result["paths"]
            receipt = result["receipt"]
            self.assertGreaterEqual(
                paths.receipt.stat().st_mtime_ns,
                max(paths.preflight.stat().st_mtime_ns, paths.rows.stat().st_mtime_ns,
                    paths.metadata.stat().st_mtime_ns),
            )
            self.assertEqual(receipt["commit_state"], "complete")
            self.assertEqual(receipt["n_records"], artifacts.SCIENTIFIC_N_RECORDS)
            self.assertEqual(
                receipt["n_completions"],
                artifacts.SCIENTIFIC_N_RECORDS * artifacts.SCIENTIFIC_MATRIX_K,
            )
            self.assertEqual(receipt["ordered_records"][0]["id"], "task-0::base")
            self.assertEqual(
                receipt["identity"]["sampling"],
                self._sampling(budget=32768, k=artifacts.SCIENTIFIC_MATRIX_K),
            )
            self.assertEqual(receipt["identity"]["engine"], ENGINE)
            self.assertEqual(receipt["identity"]["runner_sha256"], artifacts.RUNNER_SHA256)
            self.assertEqual(receipt["identity"]["model"], artifacts.MODEL_ID)
            verified = artifacts.verify_receipt(
                root, prefix, expected=result["identity"]
            )
            self.assertEqual(verified, receipt)
            self.assertEqual(artifacts.bundle_state(root, prefix)["status"], "complete")
            # Committing an exact completed bundle is idempotent, not a rewrite.
            again = artifacts.commit_receipt(
                root,
                prefix,
                role="complete_matrix_arm",
                tier_mode="complete_k12_matrix",
                thinking_budget=32768,
                arm="base",
                k=artifacts.SCIENTIFIC_MATRIX_K,
                expected_identity=result["identity"],
            )
            self.assertEqual(again, receipt)

    def test_receipt_corruption_missing_files_and_identity_drift_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "expectation"):
                artifacts.verify_receipt(
                    root, prefix, expected={"runner_sha256": "b" * 64}
                )
            result["paths"].rows.write_text(
                result["paths"].rows.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "size/hash/path"):
                artifacts.verify_receipt(root, prefix)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_tiers/think_32768/base"
            result = self._write_complete_inputs(root, prefix, arm="base")
            result["paths"].metadata.unlink()
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "missing"):
                artifacts.verify_receipt(root, prefix)

    def test_probe_receipt_has_distinct_nonselectable_role(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            prefix = "smoke_budget_probes/think_49152/base"
            result = self._write_complete_inputs(
                root,
                prefix,
                arm="base",
                budget=49152,
                role="termination_probe",
                tier_mode="termination_probe_only",
            )
            self.assertEqual(result["receipt"]["role"], "termination_probe")
            self.assertEqual(result["receipt"]["tier_mode"], "termination_probe_only")

    def test_catalog_is_deterministic_and_selection_is_logical_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "external"
            completed = {
                "base": self._write_complete_inputs(
                    root, "smoke_tiers/think_16384/base", arm="base", budget=16384
                )
            }
            completed["designed_ceiling"] = self._write_complete_inputs(
                root,
                "smoke_tiers/think_16384/designed_ceiling",
                arm="designed_ceiling",
                budget=16384,
            )
            artifacts.write_preflight_only(
                root,
                "smoke_budget_probes/think_49152/base",
                self._preflight("base", budget=49152),
            )
            selection = workspace / "smoke_budget_selection.json"
            selection.write_text(
                json.dumps(
                    self._passing_selection(budget=16384, completed=completed)
                )
                + "\n",
                encoding="utf-8",
            )
            selected_entries = {
                "designed_ceiling": "matrix/think_16384/designed_ceiling",
                "base": "matrix/think_16384/base",
            }
            first = artifacts.build_catalog(
                root,
                protocol_binding=self._protocol_binding(),
                selection_file=selection,
                selected_budget=16384,
                selected_entries=selected_entries,
            )
            second = artifacts.build_catalog(
                root,
                protocol_binding=self._protocol_binding(),
                selection_file=selection,
                selected_budget=16384,
                selected_entries=dict(reversed(list(selected_entries.items()))),
            )
            self.assertEqual(first, second)
            self.assertEqual(
                [entry["id"] for entry in first["entries"]],
                sorted(entry["id"] for entry in first["entries"]),
            )
            probe = next(
                entry for entry in first["entries"] if entry["id"] == "probe/think_49152/base"
            )
            self.assertEqual(probe["status"], "preflight_only")
            self.assertEqual(
                first["selected"]["selection_path"], artifacts.SELECTION_LOGICAL_PATH
            )
            self.assertEqual(
                first["selected"]["arms"]["base"], "matrix/think_16384/base"
            )
            self.assertFalse((root / "smoke").exists())

            catalog_path = workspace / "scientific_smoke_artifact_catalog.json"
            artifacts.write_catalog(catalog_path, first)
            self.assertEqual(
                artifacts.verify_catalog(
                    catalog_path,
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                ),
                first,
            )
            valid_selection = self._passing_selection(
                budget=16384, completed=completed
            )
            semantic_mutations = {
                "adequacy": lambda value: value["tiers"][0].__setitem__(
                    "adequate", False
                ),
                "artifact": lambda value: value["tiers"][0]["arms"]["base"][
                    "artifacts"
                ].__setitem__("rows", "0" * 64),
            }
            for label, mutate in semantic_mutations.items():
                with self.subTest(label=f"selection-{label}"):
                    changed = copy.deepcopy(valid_selection)
                    mutate(changed)
                    selection.write_text(
                        json.dumps(changed) + "\n", encoding="utf-8"
                    )
                    changed_catalog = copy.deepcopy(first)
                    changed_catalog["selected"]["selection_bytes"] = selection.stat().st_size
                    changed_catalog["selected"]["selection_sha256"] = hashlib.sha256(
                        selection.read_bytes()
                    ).hexdigest()
                    with self.assertRaises(artifacts.ScientificArtifactError):
                        artifacts.validate_selection(
                            selection,
                            changed_catalog,
                            budget_ladder=artifacts.SCIENTIFIC_BUDGETS,
                            arms=artifacts.SCIENTIFIC_MATRIX_ARMS,
                        )
            selection.write_text(
                json.dumps({"pass": False, "selected_thinking_budget": None}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(artifacts.ScientificArtifactError):
                artifacts.verify_catalog(
                    catalog_path,
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                )

    def test_catalog_rejects_probe_selection_missing_external_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            root = workspace / "external"
            probe = self._write_complete_inputs(
                root,
                "smoke_budget_probes/think_49152/base",
                arm="base",
                budget=49152,
                role="termination_probe",
                tier_mode="termination_probe_only",
            )
            selection = workspace / "selection.json"
            selection.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(artifacts.ScientificArtifactError, "exact base/designed"):
                artifacts.build_catalog(
                    root,
                    protocol_binding=self._protocol_binding(),
                    selection_file=selection,
                    selected_budget=49152,
                    selected_entries={"base": "probe/think_49152/base"},
                )

            catalog = artifacts.build_catalog(
                root, protocol_binding=self._protocol_binding()
            )
            catalog_path = workspace / "catalog.json"
            artifacts.write_catalog(catalog_path, catalog)
            probe["paths"].rows.unlink()
            with self.assertRaises(artifacts.ScientificArtifactError):
                artifacts.verify_catalog(
                    catalog_path, root, protocol_binding=self._protocol_binding()
                )

        if hasattr(os, "symlink"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "external"
                root.mkdir()
                target = Path(directory) / "target.json"
                target.write_text("{}\n", encoding="utf-8")
                linked = root / "smoke_tiers"
                linked.symlink_to(target.parent, target_is_directory=True)
                with self.assertRaisesRegex(artifacts.ScientificArtifactError, "symlink"):
                    artifacts.build_catalog(
                        root, protocol_binding=self._protocol_binding()
                    )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            (root / "smoke_tiers" / "think_32768" / "unexpected").mkdir(
                parents=True
            )
            with self.assertRaisesRegex(
                artifacts.ScientificArtifactError, "unexpected directory"
            ):
                artifacts.build_catalog(root, protocol_binding=self._protocol_binding())


if __name__ == "__main__":
    unittest.main()
