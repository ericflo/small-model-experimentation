from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
SCRIPTS = EXP / "scripts"
sys.path[:0] = [str(SRC), str(SCRIPTS)]

import analyze  # noqa: E402
import scientific_artifacts as store  # noqa: E402

SPEC = importlib.util.spec_from_file_location("capacity_run", SCRIPTS / "run.py")
assert SPEC and SPEC.loader
run = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run)


class CapacityProtocolTests(unittest.TestCase):
    def protocol_binding(self) -> dict:
        core = {
            "schema_version": 1,
            "experiment_id": store.EXPERIMENT_ID,
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
        return {**core, "binding_sha256": hashlib.sha256(run._canonical_bytes(core)).hexdigest()}

    def preflight(self, arm: str, *, budget: int, k: int, capacity: int = 995328) -> dict:
        reserve = budget + 514
        records = [
            {
                "id": f"task-{index:02d}::{arm}",
                "input_record_sha256": f"{index + 1:064x}",
                "rendered_prompt_sha256": f"{index + 101:064x}",
                "prompt_tokens": 100 + index,
                "prompt_plus_reserve_tokens": reserve + 100 + index,
            }
            for index in range(12)
        ]
        max_total = reserve + 111
        block_size = 16
        rounded = ((max_total + 15) // 16) * 16
        active = min(12 * k, store.expected_max_num_seqs(budget))
        return {
            "schema_version": 1,
            "pass": True,
            "protocol_binding": self.protocol_binding(),
            "max_model_len": 65536,
            "generation_reserve_tokens": reserve,
            "n_records": 12,
            "min_prompt_tokens": 100,
            "max_prompt_tokens": 111,
            "max_prompt_plus_reserve_tokens": max_total,
            "records": records,
            "capacity_fit": {
                "source": "vllm_config.cache_config.kv_cache_size_tokens",
                "kv_cache_size_tokens": capacity,
                "block_size": block_size,
                "live_max_model_len": 65536,
                "max_num_seqs": store.expected_max_num_seqs(budget),
                "logical_sequences": 12 * k,
                "active_sequences": active,
                "max_prompt_plus_reserve_tokens": max_total,
                "rounded_tokens_per_sequence": rounded,
                "required_cache_tokens": active * rounded,
                "pass": active * rounded <= capacity,
            },
        }

    @staticmethod
    def sampling(budget: int, k: int) -> dict:
        return store._expected_sampling(budget=budget, k=k)

    def rows(self, preflight: dict, arm: str, k: int) -> list[dict]:
        rows = []
        for record in preflight["records"]:
            parent_seed = store._stable_seed(
                store.SCIENTIFIC_RUN_SEED, record["id"], -1, "stage1"
            )
            rows.append(
                {
                    "id": record["id"],
                    "meta": {
                        "task_id": record["id"].removesuffix(f"::{arm}"),
                        "split": "smoke_reuse",
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
                            "sample_index": index,
                            "stage1_parent_seed": parent_seed,
                            "seed_stage1": parent_seed + index,
                            "seed_stage2": None,
                            "text": "opaque",
                            "token_ids": [248069, index + 1],
                            "stage1_token_ids": [248069, index + 1, 248044],
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
                        for index in range(k)
                    ],
                }
            )
        return rows

    def metadata(self, rows: list[dict], budget: int, k: int, runtime_tag: str = "A") -> dict:
        stage1 = sum(
            output["n_stage1_prompt_tokens"] for row in rows for output in row["outputs"]
        )
        stage2 = sum(
            output["n_stage2_prompt_tokens"] for row in rows for output in row["outputs"]
        )
        max_num_seqs = store.expected_max_num_seqs(budget)
        return {
            "schema_version": store.RUNNER_SCHEMA_VERSION,
            "model": store.MODEL_ID,
            "model_revision": store.MODEL_REVISION,
            "runner_sha256": store.RUNNER_SHA256,
            "adapter": None,
            "sampling": self.sampling(budget, k),
            "resolved_sampling": dict(store._EXPECTED_RESOLVED_SAMPLING),
            "engine": store._expected_engine(budget),
            "engine_args": store._expected_engine_args(budget),
            "think_token_ids": json.loads(json.dumps(store._EXPECTED_THINK_TOKEN_IDS)),
            "termination": dict(store._EXPECTED_TERMINATION),
            "rng_isolation": dict(store._EXPECTED_RNG_ISOLATION),
            "runtime": {
                "python": "3.12",
                "packages": {"vllm": runtime_tag},
                "gpu": "Ada",
                "git_commit": "ignored",
                "git_dirty": True,
            },
            "counts": {
                "requests": len(rows),
                "completions": len(rows) * k,
                "unique_input_prompt_tokens": sum(row["n_prompt_tokens"] for row in rows),
                "stage1_logical_prompt_tokens": stage1,
                "stage2_logical_prompt_tokens": stage2,
                "logical_model_input_tokens": stage1 + stage2,
                "sampled_tokens": len(rows) * k * 3,
                "injected_tokens": 0,
            },
        }

    def complete(
        self,
        root: Path,
        prefix: str,
        *,
        arm: str,
        budget: int,
        k: int,
        runtime_tag: str = "A",
        adequate: bool = True,
    ) -> dict:
        preflight = self.preflight(arm, budget=budget, k=k)
        store.write_preflight_only(root, prefix, preflight)
        paths = store.bundle_paths(root, prefix)
        rows = self.rows(preflight, arm, k)
        if not adequate:
            # Three unresolved boundary contacts are enough to fail the strict
            # <5% gate for either K4 (48 samples) or K12 (144 samples only if
            # expanded below).  Use four so both fixture geometries fail.
            for output in [item for row in rows for item in row["outputs"]][:4]:
                output["forced_close"] = True
        paths.rows.write_text(
            "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
        paths.metadata.write_text(
            json.dumps(self.metadata(rows, budget, k, runtime_tag), sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return store.commit_receipt(
            root,
            prefix,
            role="termination_probe" if k == 4 else "complete_matrix_arm",
            tier_mode="termination_probe_only" if k == 4 else "complete_k12_matrix",
            thinking_budget=budget,
            arm=arm,
            k=k,
        )

    def test_frozen_mapping_copies_and_records(self) -> None:
        audit = run.validate_frozen_protocol(run.load_config())
        self.assertEqual(audit["max_num_seqs_by_budget"], {49152: 19, 61440: 15})
        self.assertEqual(audit["record_hashes"], run.EXPECTED_RECORD_LIST_HASHES)

    def test_predecessor_root_overlap_is_rejected(self) -> None:
        predecessor = store.PREDECESSOR_ARTIFACT_ROOT
        for path in (predecessor, predecessor / "nested", predecessor.parent):
            with self.subTest(path=path), self.assertRaisesRegex(
                store.ScientificArtifactError, "predecessor root"
            ):
                store.resolve_artifact_root(path)

    def test_capacity_fit_and_prompt_mutation_fail_before_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            prefix = "smoke_budget_probes/think_49152/base"
            bad = self.preflight("base", budget=49152, k=4, capacity=100)
            with self.assertRaisesRegex(store.ScientificArtifactError, "insufficient"):
                store.write_preflight_only(root, prefix, bad)
            wrong_concurrency = self.preflight("base", budget=49152, k=4)
            wrong_concurrency["capacity_fit"]["max_num_seqs"] = 64
            with self.assertRaisesRegex(store.ScientificArtifactError, "max_num_seqs"):
                store.write_preflight_only(root, prefix, wrong_concurrency)

    def test_old_maxseq64_metadata_cannot_commit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            prefix = "smoke_budget_probes/think_49152/base"
            preflight = self.preflight("base", budget=49152, k=4)
            store.write_preflight_only(root, prefix, preflight)
            paths = store.bundle_paths(root, prefix)
            rows = self.rows(preflight, "base", 4)
            paths.rows.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            metadata = self.metadata(rows, 49152, 4)
            metadata["engine"]["max_num_seqs"] = 64
            metadata["engine_args"]["max_num_seqs"] = 64
            metadata["engine_args"]["max_cudagraph_capture_size"] = 64
            paths.metadata.write_text(json.dumps(metadata), encoding="utf-8")
            with self.assertRaisesRegex(store.ScientificArtifactError, "engine protocol"):
                store.commit_receipt(
                    root,
                    prefix,
                    role="termination_probe",
                    tier_mode="termination_probe_only",
                    thinking_budget=49152,
                    arm="base",
                    k=4,
                )

    def test_probe_and_matrix_runtime_identity_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            probe = self.complete(
                root,
                "smoke_budget_probes/think_49152/base",
                arm="base",
                budget=49152,
                k=4,
                runtime_tag="A",
            )
            base = self.complete(
                root,
                "smoke_tiers/think_49152/base",
                arm="base",
                budget=49152,
                k=12,
                runtime_tag="B",
            )
            self.assertNotEqual(
                store.comparable_protocol_identity(probe),
                store.comparable_protocol_identity(base),
            )
            catalog = store.build_catalog(root, protocol_binding=self.protocol_binding())
            entries = {entry["id"]: entry for entry in catalog["entries"]}
            with (
                mock.patch.object(
                    store, "build_protocol_binding", return_value=self.protocol_binding()
                ),
                self.assertRaisesRegex(ValueError, "runtime/engine"),
            ):
                run._selection_tier(root, 49152, entries)

    def test_unknown_or_partial_bundle_fails_inventory_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            root.mkdir()
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with mock.patch.object(run, "ANALYSIS", Path(directory) / "analysis"):
                with self.assertRaises(store.ScientificArtifactError):
                    run._reconcile_inventory(root)

    def test_nonblocking_fixed_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory) / "fixed.lock"
            root = Path(directory) / "root"
            with mock.patch.object(store, "LOCK_PATH", lock):
                with run._run_lock(root):
                    with self.assertRaisesRegex(RuntimeError, "another capacity-fit"):
                        with run._run_lock(root):
                            pass

    def test_analyzer_never_decodes_when_second_receipt_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_exp = Path(directory)
            (fake_exp / "analysis").mkdir()
            (fake_exp / "analysis" / "smoke_budget_selection.json").write_text(
                json.dumps({"tiers": []}), encoding="utf-8"
            )
            fake_preflight = fake_exp / "preflight.json"
            fake_preflight.write_text(
                json.dumps({"protocol_binding": {"binding_sha256": "x"}}),
                encoding="utf-8",
            )
            with (
                mock.patch.object(analyze, "EXP", fake_exp),
                mock.patch.object(store, "build_protocol_binding", return_value={"binding_sha256": "x"}),
                mock.patch.object(store, "verify_catalog", return_value={"selected": {}}),
                mock.patch.object(store, "validate_selection"),
                mock.patch.object(
                    store,
                    "selected_bundle_prefixes",
                    return_value=(49152, {"base": "base", "designed_ceiling": "designed"}),
                ),
                mock.patch.object(
                    store,
                    "verify_receipt",
                    side_effect=[{"identity": {"sampling": {"n": 12}}}, ValueError("missing")],
                ),
                mock.patch.object(
                    store,
                    "bundle_paths",
                    return_value=SimpleNamespace(preflight=fake_preflight),
                ),
                mock.patch.object(analyze, "_arm_semantics") as semantics,
            ):
                with self.assertRaisesRegex(ValueError, "missing"):
                    analyze.analyze_matrix(Path(directory) / "root")
                semantics.assert_not_called()

    def test_termination_thresholds_and_decoded_content_invariance(self) -> None:
        outputs = []
        for index in range(48):
            output = {
                "sample_index": index % 4,
                "n_thinking_tokens": 10,
                "n_answer_tokens": 10,
                "forced_close": False,
                "stage1_finish_reason": "stop",
                "finish_reason": "stop",
                "truncated": False,
                "stage1_token_ids": [1, 2, 3],
                "text": "decoded content A",
            }
            if index < 12:
                output["forced_close"] = True
                output["retained_thinking_token_ids"] = [7, 8] * 4096
            elif index < 14:
                output["forced_close"] = True
            if 14 <= index < 16:
                output["n_answer_tokens"] = 512
            outputs.append(output)
        rows = [
            {"id": f"task-{row}", "meta": {"correct": False}, "outputs": outputs[row * 4 : row * 4 + 4]}
            for row in range(12)
        ]
        baseline = analyze.termination_metrics(rows, budget=49152)
        self.assertTrue(baseline["adequate"])
        self.assertEqual(baseline["periodic_loop_contacts"], 12)
        self.assertEqual(baseline["unresolved_cap_contacts"], 2)
        self.assertEqual(baseline["answer_limit_contacts"], 2)
        mutated = json.loads(json.dumps(rows))
        for row in mutated:
            row["meta"]["correct"] = True
            for output in row["outputs"]:
                output["text"] = "completely different decoded answer"
        self.assertEqual(
            analyze.termination_metrics(mutated, budget=49152), baseline
        )
        mutated[3]["outputs"][2]["forced_close"] = True
        self.assertFalse(analyze.termination_metrics(mutated, budget=49152)["adequate"])

    def test_phase_transition_table_and_fresh_prefixes(self) -> None:
        root = Path("/tmp/model-free-capacity-fit-state")
        cases = [
            ("probe", 49152, "needs_probe", None),
            ("base", 49152, "needs_base", None),
            ("designed", 49152, "needs_designed", None),
            ("probe", 61440, "needs_probe", "probe_rejected"),
            ("probe", 61440, "needs_probe", "base_rejected"),
            ("probe", 61440, "needs_probe", "designed_rejected"),
        ]
        for phase, budget, current, lower in cases:
            target = run._prefix(phase, budget)[0]
            with (
                self.subTest(phase=phase, budget=budget, lower=lower),
                mock.patch.object(
                    run,
                    "_rung_outcome",
                    side_effect=(
                        (lambda _root, value: lower if value == 49152 else current)
                        if budget == 61440
                        else (lambda _root, _value: current)
                    ),
                ),
                mock.patch.object(
                    store,
                    "bundle_state",
                    return_value={"status": "absent", "relative_prefix": target},
                ),
            ):
                run.authorize_phase(root, phase, budget)
        with (
            mock.patch.object(run, "_rung_outcome", return_value="matrix_adequate"),
            mock.patch.object(store, "bundle_state", return_value={"status": "absent"}),
            self.assertRaisesRegex(ValueError, "completed rejected 49k"),
        ):
            run.authorize_phase(root, "probe", 61440)
        probe_prefix, _, probe_k, _ = run._prefix("probe", 49152)
        base_prefix, _, base_k, _ = run._prefix("base", 49152)
        self.assertNotEqual(probe_prefix, base_prefix)
        self.assertEqual((probe_k, base_k), (4, 12))

    def test_live_context_field_and_capacity_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            prefix = "smoke_budget_probes/think_61440/base"
            preflight = self.preflight("base", budget=61440, k=4)
            preflight["capacity_fit"]["live_max_model_len"] = 131072
            with self.assertRaisesRegex(store.ScientificArtifactError, "live vLLM"):
                store.write_preflight_only(root, prefix, preflight)

    def test_receipt_before_catalog_reconciles_but_completed_deletion_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            analysis_dir = Path(directory) / "analysis"
            analysis_dir.mkdir()
            binding = self.protocol_binding()
            empty = store.build_catalog(root, protocol_binding=binding)
            store.write_catalog(analysis_dir / "scientific_smoke_artifact_catalog.json", empty)
            prefix = "smoke_budget_probes/think_49152/base"
            preflight = self.preflight("base", budget=49152, k=4)
            store.write_preflight_only(root, prefix, preflight)
            with (
                mock.patch.object(run, "ANALYSIS", analysis_dir),
                mock.patch.object(store, "build_protocol_binding", return_value=binding),
            ):
                run._checkpoint_preflight(root, prefix)
                self.complete(root, prefix, arm="base", budget=49152, k=4)
                reconciled = run._reconcile_inventory(root)
                self.assertEqual(reconciled["entries"][0]["status"], "complete")
                shutil.rmtree(root)
                root.mkdir()
                with self.assertRaisesRegex(ValueError, "disappeared"):
                    run._reconcile_inventory(root)

    def test_k4_probe_cannot_fill_or_resolve_a_selected_k12_arm(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            binding = self.protocol_binding()
            self.complete(
                root,
                "smoke_budget_probes/think_49152/base",
                arm="base",
                budget=49152,
                k=4,
            )
            self.complete(
                root,
                "smoke_tiers/think_49152/designed_ceiling",
                arm="designed_ceiling",
                budget=49152,
                k=12,
            )
            catalog = store.build_catalog(root, protocol_binding=binding)
            catalog["selected"] = {
                "thinking_budget": 49152,
                "selection_path": store.SELECTION_LOGICAL_PATH,
                "selection_bytes": 1,
                "selection_sha256": "a" * 64,
                "arms": {
                    "base": "probe/think_49152/base",
                    "designed_ceiling": "matrix/think_49152/designed_ceiling",
                },
            }
            with self.assertRaisesRegex(
                store.ScientificArtifactError, "selected base is a probe"
            ):
                store.selected_bundle_prefixes(catalog, store.SCIENTIFIC_MATRIX_ARMS)

    def test_success_finalizer_is_first_adequate_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            analysis_dir = Path(directory) / "analysis"
            analysis_dir.mkdir()
            binding = self.protocol_binding()
            self.complete(
                root,
                "smoke_budget_probes/think_49152/base",
                arm="base",
                budget=49152,
                k=4,
            )
            for arm in store.SCIENTIFIC_MATRIX_ARMS:
                self.complete(
                    root,
                    f"smoke_tiers/think_49152/{arm}",
                    arm=arm,
                    budget=49152,
                    k=12,
                )
            store.write_catalog(
                analysis_dir / "scientific_smoke_artifact_catalog.json",
                store.build_catalog(root, protocol_binding=binding),
            )
            with (
                mock.patch.object(run, "ANALYSIS", analysis_dir),
                mock.patch.object(store, "build_protocol_binding", return_value=binding),
            ):
                run._refresh_catalog(root)
                selection_path = analysis_dir / "smoke_budget_selection.json"
                first_selection = selection_path.read_bytes()
                first_catalog = (
                    analysis_dir / "scientific_smoke_artifact_catalog.json"
                ).read_bytes()
                selection = json.loads(first_selection)
                self.assertTrue(selection["pass"])
                self.assertEqual(selection["selected_thinking_budget"], 49152)
                self.assertEqual([tier["budget"] for tier in selection["tiers"]], [49152])
                self.assertEqual(selection["tiers"][0]["scientific_probe"]["k"], 4)
                self.assertTrue(
                    all(
                        state["status"] == "complete"
                        for state in selection["tiers"][0]["arms"].values()
                    )
                )
                run._refresh_catalog(root)
                self.assertEqual(selection_path.read_bytes(), first_selection)
                self.assertEqual(
                    (analysis_dir / "scientific_smoke_artifact_catalog.json").read_bytes(),
                    first_catalog,
                )

    def test_terminal_61k_failure_finalizer_is_unselected_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "root"
            analysis_dir = Path(directory) / "analysis"
            analysis_dir.mkdir()
            binding = self.protocol_binding()
            for budget in store.SCIENTIFIC_BUDGETS:
                self.complete(
                    root,
                    f"smoke_budget_probes/think_{budget}/base",
                    arm="base",
                    budget=budget,
                    k=4,
                    adequate=False,
                )
            store.write_catalog(
                analysis_dir / "scientific_smoke_artifact_catalog.json",
                store.build_catalog(root, protocol_binding=binding),
            )
            with (
                mock.patch.object(run, "ANALYSIS", analysis_dir),
                mock.patch.object(store, "build_protocol_binding", return_value=binding),
            ):
                run._refresh_catalog(root)
                selection_path = analysis_dir / "smoke_budget_selection.json"
                first_selection = selection_path.read_bytes()
                first_catalog = (
                    analysis_dir / "scientific_smoke_artifact_catalog.json"
                ).read_bytes()
                selection = json.loads(first_selection)
                self.assertFalse(selection["pass"])
                self.assertIsNone(selection["selected_thinking_budget"])
                self.assertEqual(
                    [tier["budget"] for tier in selection["tiers"]],
                    list(store.SCIENTIFIC_BUDGETS),
                )
                self.assertTrue(
                    all(tier["status"] == "probe_only_rejected" for tier in selection["tiers"])
                )
                self.assertIsNone(json.loads(first_catalog)["selected"])
                run._refresh_catalog(root)
                self.assertEqual(selection_path.read_bytes(), first_selection)
                self.assertEqual(
                    (analysis_dir / "scientific_smoke_artifact_catalog.json").read_bytes(),
                    first_catalog,
                )

    def test_analyzer_recomputes_lower_history_before_semantic_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_exp = Path(directory)
            analysis_dir = fake_exp / "analysis"
            analysis_dir.mkdir()
            selection = {
                "tiers": [
                    {
                        "budget": 49152,
                        "scientific_probe": {"termination": {"adequate": False}},
                        "arms": {
                            "base": {"status": "skipped"},
                            "designed_ceiling": {"status": "skipped"},
                        },
                    },
                    {
                        "budget": 61440,
                        "scientific_probe": {"termination": {"adequate": True}},
                        "arms": {
                            "base": {"status": "complete", "termination": {"adequate": True}},
                            "designed_ceiling": {
                                "status": "complete",
                                "termination": {"adequate": True},
                            },
                        },
                    },
                ]
            }
            (analysis_dir / "smoke_budget_selection.json").write_text(
                json.dumps(selection), encoding="utf-8"
            )
            with (
                mock.patch.object(analyze, "EXP", fake_exp),
                mock.patch.object(store, "build_protocol_binding", return_value={"binding_sha256": "x"}),
                mock.patch.object(store, "verify_catalog", return_value={"selected": {}}),
                mock.patch.object(store, "validate_selection"),
                mock.patch.object(store, "verify_receipt", return_value={"identity": {}}),
                mock.patch.object(store, "comparable_protocol_identity", return_value={"same": True}),
                mock.patch.object(
                    store,
                    "bundle_paths",
                    return_value=SimpleNamespace(rows=fake_exp / "opaque.jsonl"),
                ),
                mock.patch.object(analyze, "read_rows", return_value=[{"outputs": [{}]}]),
                mock.patch.object(
                    analyze,
                    "termination_metrics",
                    return_value={"adequate": True},
                ),
                mock.patch.object(analyze, "_arm_semantics") as semantics,
                self.assertRaisesRegex(ValueError, "selection termination audit drift"),
            ):
                analyze.analyze_matrix(fake_exp / "external")
            semantics.assert_not_called()

    def test_symlink_alias_root_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            real = Path(directory) / "real"
            real.mkdir()
            alias = Path(directory) / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(store.ScientificArtifactError, "symlink"):
                store.resolve_artifact_root(alias)

    def test_analyzer_does_not_load_hidden_inputs_when_designed_is_inadequate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake_exp = Path(directory)
            (fake_exp / "analysis").mkdir()
            (fake_exp / "analysis" / "smoke_budget_selection.json").write_text(
                json.dumps({"tiers": []}), encoding="utf-8"
            )
            preflight = fake_exp / "preflight.json"
            preflight.write_text(
                json.dumps({"protocol_binding": {"binding_sha256": "x"}}), encoding="utf-8"
            )
            rows_base = fake_exp / "base.jsonl"
            rows_designed = fake_exp / "designed.jsonl"
            rows_base.write_text("{}\n", encoding="utf-8")
            rows_designed.write_text("{}\n", encoding="utf-8")
            receipt = {"identity": {"sampling": {"n": 12}}}
            with (
                mock.patch.object(analyze, "EXP", fake_exp),
                mock.patch.object(store, "build_protocol_binding", return_value={"binding_sha256": "x"}),
                mock.patch.object(store, "verify_catalog", return_value={"selected": {}}),
                mock.patch.object(store, "validate_selection"),
                mock.patch.object(
                    store,
                    "selected_bundle_prefixes",
                    return_value=(49152, {"base": "base", "designed_ceiling": "designed"}),
                ),
                mock.patch.object(store, "verify_receipt", return_value=receipt),
                mock.patch.object(
                    store,
                    "comparable_protocol_identity",
                    return_value={"runtime": "same"},
                ),
                mock.patch.object(
                    store,
                    "bundle_paths",
                    side_effect=[
                        SimpleNamespace(preflight=preflight, rows=rows_base),
                        SimpleNamespace(preflight=preflight, rows=rows_designed),
                    ],
                ),
                mock.patch.object(
                    analyze,
                    "read_rows",
                    side_effect=[[{"outputs": [{}]}], [{"outputs": [{}]}]],
                ),
                mock.patch.object(
                    analyze,
                    "termination_metrics",
                    side_effect=[{"adequate": True}, {"adequate": False}],
                ),
                mock.patch.object(analyze, "_arm_semantics") as semantics,
            ):
                with self.assertRaisesRegex(ValueError, "designed_ceiling matrix termination"):
                    analyze.analyze_matrix(Path(directory) / "root")
                semantics.assert_not_called()


if __name__ == "__main__":
    unittest.main()
