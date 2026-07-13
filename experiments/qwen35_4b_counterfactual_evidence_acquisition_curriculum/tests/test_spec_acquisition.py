from __future__ import annotations

import sys
import unittest
import importlib.util
import json
import tempfile
from unittest import mock
from collections import Counter
from collections import defaultdict
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import harness  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402

TRAIN_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_train", EXP / "scripts" / "train.py"
)
assert TRAIN_SPEC and TRAIN_SPEC.loader
train = importlib.util.module_from_spec(TRAIN_SPEC)
TRAIN_SPEC.loader.exec_module(train)

EVAL_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_eval", EXP / "scripts" / "eval_repo_agent.py"
)
assert EVAL_SPEC and EVAL_SPEC.loader
eval_repo_agent = importlib.util.module_from_spec(EVAL_SPEC)
EVAL_SPEC.loader.exec_module(eval_repo_agent)

LOCALITY_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_locality", EXP / "scripts" / "audit_locality.py"
)
assert LOCALITY_SPEC and LOCALITY_SPEC.loader
audit_locality = importlib.util.module_from_spec(LOCALITY_SPEC)
LOCALITY_SPEC.loader.exec_module(audit_locality)

UNCERTAINTY_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_uncertainty",
    EXP / "scripts" / "audit_transition_uncertainty.py",
)
assert UNCERTAINTY_SPEC and UNCERTAINTY_SPEC.loader
audit_transition_uncertainty = importlib.util.module_from_spec(UNCERTAINTY_SPEC)
UNCERTAINTY_SPEC.loader.exec_module(audit_transition_uncertainty)

RUN_SPEC = importlib.util.spec_from_file_location(
    "counterfactual_run", EXP / "scripts" / "run.py"
)
assert RUN_SPEC and RUN_SPEC.loader
run_pipeline = importlib.util.module_from_spec(RUN_SPEC)
RUN_SPEC.loader.exec_module(run_pipeline)


def output(action: dict, *, run_on: str = "") -> dict:
    return {
        "text": f"brief plan\n</think>\n\n{repo_agent.action_text(action)}{run_on}",
        "n_sampled_tokens": 20,
        "n_thinking_tokens": 5,
        "n_answer_tokens": 15,
        "thinking_closed": True,
        "forced_close": False,
    }


class CounterfactualTaskTests(unittest.TestCase):
    def test_inferred_pairs_differ_only_at_discriminator_and_cross_fail(self) -> None:
        tasks = repo_tasks.make_pairs(
            repo_tasks.ALL_FAMILIES, 3, 94100, "pair_smoke"
        )
        by_pair = defaultdict(list)
        for task in tasks:
            by_pair[task.pair_id].append(task)
        self.assertEqual(len(by_pair), len(repo_tasks.ALL_FAMILIES) * 3)
        self.assertEqual(
            {task.evidence_channel for task in tasks}, set(repo_tasks.EVIDENCE_CHANNELS)
        )
        for members in by_pair.values():
            self.assertEqual(len(members), 2)
            a, b = sorted(members, key=lambda task: task.branch)
            self.assertEqual(a.issue, b.issue)
            self.assertEqual(repo_tasks.pair_static_digest(a), repo_tasks.pair_static_digest(b))
            differing = {path for path in a.files if a.files[path] != b.files[path]}
            self.assertEqual(differing, {a.evidence_path})
            for source, counterpart in ((a, b), (b, a)):
                env = repo_tasks.RepoEnv(counterpart)
                try:
                    patch = source.oracle_patches[0]
                    self.assertTrue(
                        env.patch(patch.path, patch.old, patch.new).startswith("PATCH_OK")
                    )
                    self.assertFalse(all(env.score_workspace()))
                finally:
                    env.close()

    def test_initial_partial_oracle_executable_invariants(self) -> None:
        tasks = repo_tasks.make_pairs(
            repo_tasks.ALL_FAMILIES, 1, 94200, "exec_smoke"
        )
        for task in tasks:
            with self.subTest(task=task.task_id):
                env = repo_tasks.RepoEnv(task)
                try:
                    self.assertEqual(env.score_workspace(), (False, False))
                finally:
                    env.close()
                env = repo_tasks.RepoEnv(task)
                try:
                    env.apply_partial()
                    self.assertEqual(env.score_workspace(), (False, False))
                finally:
                    env.close()
                env = repo_tasks.RepoEnv(task)
                try:
                    env.apply_oracle()
                    self.assertEqual(env.score_workspace(), (True, True))
                finally:
                    env.close()

    def test_bank_and_eval_use_disjoint_path_skins(self) -> None:
        bank_tasks = repo_tasks.make_pairs(
            repo_tasks.TRAIN_FAMILIES, 3, 93310, "acquisition_bank"
        )
        qualification_tasks = repo_tasks.make_pairs(
            repo_tasks.TRAIN_FAMILIES, 3, 93310, "qualification_probe"
        )
        transfer_tasks = repo_tasks.make_pairs(
            repo_tasks.TRANSFER_FAMILIES, 3, 93310, "transfer_probe"
        )
        by_regime = {
            "bank": {task.evidence_path for task in bank_tasks},
            "qualification": {task.evidence_path for task in qualification_tasks},
            "transfer": {task.evidence_path for task in transfer_tasks},
        }
        self.assertFalse(by_regime["bank"] & by_regime["qualification"])
        self.assertFalse(by_regime["bank"] & by_regime["transfer"])
        self.assertFalse(by_regime["qualification"] & by_regime["transfer"])
        self.assertEqual(
            {task.acquisition_query_skin for task in transfer_tasks}, {"signature"}
        )

    def test_explicit_control_states_name_policy_in_issue(self) -> None:
        tasks = repo_tasks.make_pairs(
            repo_tasks.TRAIN_FAMILIES,
            1,
            94300,
            "explicit_smoke",
            explicit_contract=True,
        )
        for a, b in zip(tasks[::2], tasks[1::2]):
            self.assertNotEqual(a.issue, b.issue)
            self.assertIn("Explicit edge policy", a.issue)
            self.assertIn("Explicit edge policy", b.issue)


class ControlledStateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = repo_tasks.make_pairs(
            ("spec_lookup",), 1, 94400, "agent_smoke"
        )[0]

    def test_controlled_prefixes_expose_source_then_optional_evidence(self) -> None:
        ambiguous = repo_agent.Episode(self.task, 0, scenario="ambiguous_source")
        try:
            self.assertEqual(len(ambiguous.prefix_steps), 1)
            self.assertEqual(
                ambiguous.prefix_steps[0]["action"]["path"],
                self.task.oracle_patches[0].path,
            )
        finally:
            ambiguous.env.close()
        injected = repo_agent.Episode(self.task, 0, scenario="evidence_injected")
        try:
            self.assertEqual(len(injected.prefix_steps), 2)
            self.assertEqual(
                injected.prefix_steps[-1]["action"]["query"],
                self.task.acquisition_query,
            )
            self.assertIn(self.task.evidence_marker, injected.prefix_steps[-1]["observation"])
        finally:
            injected.env.close()

    def test_control_search_matches_operator_without_exposing_discriminator(self) -> None:
        control = repo_agent.Episode(
            self.task, 0, scenario="nondiscriminating_search_injected"
        )
        try:
            self.assertEqual(len(control.prefix_steps), 2)
            step = control.prefix_steps[-1]
            self.assertEqual(step["action"]["tool"], "search")
            self.assertNotIn(self.task.evidence_path, step["observation"])
            self.assertNotIn(self.task.evidence_marker, step["observation"])
        finally:
            control.env.close()

    def test_history_canonicalizes_valid_tool_call_and_keeps_raw_receipt(self) -> None:
        episode = repo_agent.Episode(self.task, 0, scenario="ambiguous_source")
        try:
            action = {"tool": "read", "path": self.task.evidence_path}
            episode.consume(output(action, run_on="\nThis must not enter history."))
            self.assertIn("This must not enter history", episode.steps[-1]["raw"])
            assistant_messages = [
                row["content"] for row in episode.messages if row["role"] == "assistant"
            ]
            self.assertNotIn("This must not enter history", assistant_messages[-1])
            self.assertTrue(episode.steps[-1]["history_canonicalized"])
        finally:
            episode.env.close()

    def test_logical_compute_counts_injected_close_tokens_only_in_stage2_prompt(self) -> None:
        episode = repo_agent.Episode(self.task, 0, scenario="ambiguous_source")
        try:
            row = output({"tool": "read", "path": self.task.evidence_path})
            row.update({
                "n_stage1_prompt_tokens": 100,
                "n_stage2_prompt_tokens": 50,
                "n_sampled_tokens": 20,
                "n_injected_tokens": 2,
            })
            episode.consume(row)
            self.assertEqual(episode.steps[-1]["logical_model_tokens"], 170)
            self.assertEqual(episode.steps[-1]["n_injected_tokens"], 2)
            result = episode.finish()
            self.assertEqual(result["logical_model_tokens"], 170)
            self.assertEqual(result["injected_tokens"], 2)
        finally:
            episode.env.close()

    def test_visible_verification_without_hidden_success_still_enters_commit_denominator(self) -> None:
        rows = []
        for branch in (0, 1):
            rows.append({
                "case_id": f"case-{branch}",
                "task_id": f"task-{branch}",
                "family": "spec_lookup",
                "pair_id": "pair-0",
                "branch": branch,
                "evidence_channel": "tests",
                "task": {
                    "evidence_path_regime": "transfer",
                    "acquisition_query_skin": "signature",
                },
                "explicit_contract": False,
                "scenario": "normal",
                "workspace_success": False,
                "evidence_acquired_before_first_patch": False,
                "first_changed_patch_full_correct": False,
                "first_changed_patch_cross_fails_counterpart": False,
                "first_changed_patch_before_generated_test": False,
                "unnecessary_evidence_before_first_patch": False,
                "any_search_before_first_patch": False,
                "non_source_inspects_before_first_patch": 0,
                "verified_after_final_patch": True,
                "commit_after_pass": False,
                "submitted": False,
                "rejected_patch_changed_immediately": False,
                "rejected_patch_changed_within_two": False,
                "rejected_patch_valid_changed_within_two": False,
                "failed_test_diagnose_or_revise_immediately": False,
                "failed_test_changed_patch_within_two": False,
                "sampled_tokens": 1,
                "logical_model_input_tokens": 1,
                "injected_tokens": 0,
                "logical_model_tokens": 2,
                "turns": 1,
                "invalid_actions": 0,
                "trajectory": 0,
                "submitted_success": False,
                "steps": [{
                    "operator": "TEST",
                    "n_answer_tokens": 1,
                    "finish_reason": "stop",
                    "forced_close": False,
                    "stage2_finish_reason": "stop",
                }],
            })
        aggregate = eval_repo_agent.aggregate(rows, "deep", 1024)
        self.assertEqual(aggregate["verified_given_success"], 0.0)
        self.assertEqual(aggregate["commit_given_verified"], 0.0)
        self.assertTrue(all(row["verified_after_final_patch"] for row in aggregate["cases"]))

    def test_evidence_then_oracle_patch_scores_preverifier_success(self) -> None:
        episode = repo_agent.Episode(self.task, 0, scenario="ambiguous_source")
        patch = self.task.oracle_patches[0]
        episode.consume(output({"tool": "read", "path": self.task.evidence_path}))
        episode.consume(output({
            "tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new,
        }))
        episode.consume(output({"tool": "test"}))
        episode.consume(output({"tool": "submit"}))
        result = episode.finish()
        self.assertTrue(result["evidence_acquired_before_first_patch"])
        self.assertTrue(result["first_changed_patch_full_correct"])
        self.assertTrue(result["workspace_success"])

    def test_length_finish_below_nominal_answer_cap_is_a_contact(self) -> None:
        self.assertTrue(eval_repo_agent.answer_limit_contact(
            {
                "n_answer_tokens": 17,
                "finish_reason": "length",
                "forced_close": False,
            },
            1024,
        ))
        self.assertFalse(eval_repo_agent.answer_limit_contact(
            {
                "n_answer_tokens": 17,
                "finish_reason": "stop",
                "forced_close": False,
            },
            1024,
        ))

    def test_correct_patch_after_generated_test_is_not_preverifier(self) -> None:
        episode = repo_agent.Episode(self.task, 0, scenario="ambiguous_source")
        try:
            patch = self.task.oracle_patches[0]
            episode.consume(output({"tool": "read", "path": self.task.evidence_path}))
            episode.consume(output({"tool": "test"}))
            episode.consume(output({
                "tool": "patch", "path": patch.path,
                "old": patch.old, "new": patch.new,
            }))
            result = episode.finish()
            self.assertTrue(result["first_changed_patch_full_correct"])
            self.assertFalse(result["first_changed_patch_before_generated_test"])
        finally:
            episode.env.close()


class BankTests(unittest.TestCase):
    def test_evidence_rows_replay_and_cover_three_registered_strata(self) -> None:
        task = repo_tasks.make_pairs(
            ("spec_tags",), 1, 94500, "bank_smoke"
        )[0]
        rows, receipt = bank.evidence_transition_rows(task)
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            {row["transition"] for row in rows},
            {
                "start_to_inspect_source",
                "ambiguous_source_to_inspect_evidence",
                "evidence_to_policy_patch",
            },
        )
        self.assertTrue(receipt["final_visible_pass"])
        self.assertTrue(receipt["final_hidden_pass"])
        self.assertTrue(all(row["think_weight"] == 0.0 for row in rows))


class TokenizerProvenanceTests(unittest.TestCase):
    @staticmethod
    def _write_tokenizer(root: Path, *, local: bool = True, semantic: int = 1) -> None:
        (root / "chat_template.jinja").write_text("{{ messages }}\n")
        (root / "tokenizer.json").write_text('{"version":"1.0"}\n')
        (root / "tokenizer_config.json").write_text(json.dumps({
            "is_local": local,
            "local_files_only": local,
            "semantic_setting": semantic,
        }) + "\n")

    def test_raw_identity_and_narrow_compatibility_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_tokenizer(root, local=True)
            first = harness.tokenizer_provenance(root)
            self.assertEqual(
                tuple(first["tokenizer_files"]), harness.TOKENIZER_FILE_NAMES
            )
            self.assertEqual(
                first["tokenizer_manifest_sha256"],
                harness.tokenizer_manifest_sha256(first["tokenizer_files"]),
            )
            harness.validate_tokenizer_provenance(root, first)

            self._write_tokenizer(root, local=False)
            relocated = harness.tokenizer_provenance(root)
            self.assertNotEqual(
                first["tokenizer_manifest_sha256"],
                relocated["tokenizer_manifest_sha256"],
            )
            self.assertEqual(
                first["tokenizer_compatibility_sha256"],
                relocated["tokenizer_compatibility_sha256"],
            )

            self._write_tokenizer(root, local=False, semantic=2)
            changed = harness.tokenizer_provenance(root)
            self.assertNotEqual(
                relocated["tokenizer_compatibility_sha256"],
                changed["tokenizer_compatibility_sha256"],
            )
            with self.assertRaisesRegex(ValueError, "provenance drift"):
                harness.validate_tokenizer_provenance(root, relocated)

    def test_frozen_start_and_anchor_identities_match_config(self) -> None:
        cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        root = EXP.parents[1]
        start = harness.tokenizer_provenance(root / cfg["model"]["start_checkpoint"])
        anchor = harness.tokenizer_provenance(root / cfg["model"]["locality_anchor"])
        self.assertEqual(
            start["tokenizer_manifest_sha256"],
            cfg["model"]["start_tokenizer_manifest_sha256"],
        )
        self.assertEqual(
            anchor["tokenizer_manifest_sha256"],
            cfg["model"]["anchor_tokenizer_manifest_sha256"],
        )
        self.assertEqual(
            {
                start["tokenizer_compatibility_sha256"],
                anchor["tokenizer_compatibility_sha256"],
            },
            {cfg["model"]["tokenizer_compatibility_sha256"]},
        )

    def test_missing_or_extra_tokenizer_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_tokenizer(root)
            (root / "special_tokens_map.json").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "unexpected"):
                harness.tokenizer_file_manifest(root)
            (root / "special_tokens_map.json").unlink()
            (root / "chat_template.jinja").unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                harness.tokenizer_file_manifest(root)

    def test_locality_rejects_same_length_different_token_ids(self) -> None:
        provenance = {"tokenizer_compatibility_sha256": "a" * 64}
        with self.assertRaisesRegex(SystemExit, "token IDs differ"):
            audit_locality.assert_compatible_tokenizations(
                provenance,
                provenance,
                ["same prompt"],
                ["same prompt"],
                [[11, 12, 13]],
                [[11, 99, 13]],
            )


class CheckpointRegistrationTests(unittest.TestCase):
    def test_parent_control_file_drift_fails_before_model_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repository = Path(directory) / "repository"
            exp = repository / "experiments" / EXP.name
            model = repository / "models" / "start"
            model.mkdir(parents=True)
            exp.mkdir(parents=True)
            (model / "model.safetensors").write_bytes(b"synthetic-weight")
            (model / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "vocab_size": 248320,
                    "hidden_size": 2560,
                    "num_hidden_layers": 32,
                    "eos_token_id": 248044,
                },
            }) + "\n")
            (model / "generation_config.json").write_text(
                '{"eos_token_id":248044}\n'
            )
            TokenizerProvenanceTests._write_tokenizer(model)
            merge = {
                "model_lineage": "Qwen/Qwen3.5-4B",
                "model_revision": "test-revision",
                "weight_files": [{
                    "name": "model.safetensors",
                    "sha256": train.sha256_file(model / "model.safetensors"),
                }],
            }
            (model / "merge_receipt.json").write_text(
                json.dumps(merge, sort_keys=True) + "\n"
            )
            tokenizer = harness.tokenizer_provenance(model)
            cfg = {
                "model": {
                    "id": "Qwen/Qwen3.5-4B",
                    "revision": "test-revision",
                    "start_checkpoint": "models/start",
                    "start_weight_sha256": train.sha256_file(
                        model / "model.safetensors"
                    ),
                    "start_config_sha256": train.sha256_file(
                        model / "config.json"
                    ),
                    "start_generation_config_sha256": train.sha256_file(
                        model / "generation_config.json"
                    ),
                    "start_merge_receipt_sha256": train.sha256_file(
                        model / "merge_receipt.json"
                    ),
                    "start_tokenizer_manifest_sha256": tokenizer[
                        "tokenizer_manifest_sha256"
                    ],
                    "tokenizer_compatibility_sha256": tokenizer[
                        "tokenizer_compatibility_sha256"
                    ],
                },
                "artifacts": {"root": "artifacts/synthetic"},
            }
            lock = exp / "runs" / "preregistration_receipt.json"
            harness.validate_registered_checkpoint(
                exp, model, cfg, lock, "start"
            )
            controls = {
                "config.json": "config hash drifted",
                "generation_config.json": "generation-config hash drifted",
                "merge_receipt.json": "merge-receipt hash drifted",
            }
            for name, message in controls.items():
                with self.subTest(control=name):
                    path = model / name
                    original = path.read_bytes()
                    path.write_bytes(original + b" \n")
                    with self.assertRaisesRegex(ValueError, message):
                        harness.validate_registered_checkpoint(
                            exp, model, cfg, lock, "start"
                        )
                    path.write_bytes(original)


class UncertaintyGeometryTests(unittest.TestCase):
    def test_sample_count_and_strata_are_both_config_bound(self) -> None:
        cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        rows = int(cfg["uncertainty"]["rows_per_transition"])
        strata = int(cfg["uncertainty"]["strata"])
        audit_transition_uncertainty.validate_registered_sample_geometry(
            rows, strata, cfg
        )
        with self.assertRaisesRegex(SystemExit, "rows-per-transition"):
            audit_transition_uncertainty.validate_registered_sample_geometry(
                rows + 1, strata, cfg
            )
        with self.assertRaisesRegex(SystemExit, "strata"):
            audit_transition_uncertainty.validate_registered_sample_geometry(
                rows, strata + 1, cfg
            )


class TrainerGeometryTests(unittest.TestCase):
    @staticmethod
    def _synthetic_rows() -> list[dict]:
        rows = []
        for transition in train.TRANSITIONS:
            operator = (
                "PATCH" if "patch" in transition
                else "VERIFY" if transition == "patch_ok_to_verify"
                else "COMMIT" if transition == "passed_test_to_commit"
                else "INSPECT"
            )
            for copy_index in range(2):
                for pair_index in range(4):
                    for branch in (0, 1):
                        pair_id = f"{transition}-pair-{pair_index}"
                        rows.append({
                            "transition": transition,
                            "operator": operator,
                            "task_id": f"{pair_id}-b{branch}",
                            "row_id": f"{pair_id}-b{branch}-copy{copy_index}",
                            "pair_id": pair_id,
                            "branch": branch,
                            "weighted_action_mass": 1000.0 / 16.0,
                        })
        return rows

    def test_copy_round_scheduler_preserves_dyad_and_task_diversity(self) -> None:
        encoded = self._synthetic_rows()
        ordered, receipt = train.make_batches(encoded, 4, 9, 53)
        self.assertTrue(receipt["every_optimizer_step_contains_all_transitions"])
        for index in range(0, len(ordered), 4):
            chunk = ordered[index:index + 4]
            self.assertEqual(len({row["task_id"] for row in chunk}), 4)
            self.assertEqual(
                sorted(Counter(row["pair_id"] for row in chunk).values()), [2, 2]
            )
        for index in range(0, len(ordered), 4 * 9):
            cycle = ordered[index:index + 4 * 9]
            self.assertEqual(
                Counter(row["transition"] for row in cycle),
                Counter({transition: 4 for transition in train.TRANSITIONS}),
            )

    def test_fixed_denominator_preserves_equal_transition_mass(self) -> None:
        encoded = self._synthetic_rows()
        receipt = train.loss_mass_receipt(encoded, microbatches_per_epoch=36)
        transition_values = receipt["effective_normalized_mass_by_transition"]
        self.assertEqual(len({round(value, 9) for value in transition_values.values()}), 1)
        self.assertGreater(
            receipt["effective_normalized_mass_by_operator"]["PATCH"],
            receipt["effective_normalized_mass_by_operator"]["COMMIT"],
        )

    def test_physical_forward_lengths_strip_all_logical_right_padding(self) -> None:
        mask = train.torch.tensor([
            [1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
        ])
        self.assertEqual(train.unpadded_row_lengths(mask), [3, 5, 2, 4])
        with self.assertRaisesRegex(ValueError, "contiguous"):
            train.unpadded_row_lengths(train.torch.tensor([[1, 0, 1]]))


class TrainingAuthorizationTests(unittest.TestCase):
    def test_skeletal_pass_booleans_cannot_self_authorize_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "qwen35_4b_counterfactual_evidence_acquisition_curriculum"
            (exp / "analysis").mkdir(parents=True)
            (exp / "configs").mkdir()
            (exp / "scripts").mkdir()
            (exp / "configs" / "default.yaml").write_text(
                "evaluation:\n"
                "  interface_answer_rungs: [1024, 2048, 4096]\n"
                "model:\n"
                f"  anchor_weight_sha256: {'a' * 64}\n"
                f"  start_weight_sha256: {'b' * 64}\n"
            )
            auditor = exp / "scripts" / "audit_locality.py"
            auditor.write_text("# frozen locality auditor\n")
            lock = exp / "preregistration_receipt.json"
            lock.write_text('{"status":"locked"}\n')
            qualification_path = exp / "analysis" / "qualification_gate.json"
            qualification_path.write_text(json.dumps({
                "gate": {"passed": True, "verdict": "ACQUISITION_QUALIFIED"},
                "training_authorized": True,
                "selected_answer_max_tokens": 2048,
            }) + "\n")
            locality_path = exp / "analysis" / "locality_start_vs_anchor.json"
            locality_path.write_text(json.dumps({
                "gate": {"passed": True},
                "before_model_weight_sha256": "a" * 64,
                "after_model_weight_sha256": "b" * 64,
                "auditor_sha256": train.sha256_file(auditor),
            }) + "\n")
            authorization_path = exp / "analysis" / "training_authorization.json"
            authorization_path.write_text(json.dumps({
                "schema_version": 1,
                "stage": "training_authorization",
                "experiment_id": exp.name,
                "design_lock_sha256": train.sha256_file(lock),
                "ancestor_receipts": {
                    "qualification_gate": {
                        "path": str(qualification_path.resolve()),
                        "sha256": train.sha256_file(qualification_path),
                    },
                    "lineage_locality_gate": {
                        "path": str(locality_path.resolve()),
                        "sha256": train.sha256_file(locality_path),
                    },
                },
                "selected_answer_max_tokens": 2048,
                "checks": {
                    "acquisition_qualified": True,
                    "lineage_locality_feasible": True,
                },
                "gate": {"passed": True, "verdict": "TRAINING_AUTHORIZED"},
                "training_authorized": True,
                "menagerie_authorized": False,
            }) + "\n")
            original_exp = train.EXP
            try:
                train.EXP = exp
                with self.assertRaisesRegex(SystemExit, "frozen ancestor gates"):
                    train.validate_training_authorization(authorization_path, lock)
            finally:
                train.EXP = original_exp


class OrchestrationGuardTests(unittest.TestCase):
    def test_fresh_gate_unlinks_stale_pass_before_allowed_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "gate.json"
            output_path.write_text(json.dumps({"gate": {"passed": True}}))
            with mock.patch.object(run_pipeline, "command", return_value=4):
                with self.assertRaisesRegex(SystemExit, "did not write"):
                    run_pipeline.run_gate_fresh(output_path, ["analyzer"])
            self.assertFalse(output_path.exists())

    def test_unlogged_reservation_is_detected_but_logged_pair_is_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            reservation = (
                exp / "runs" / "menagerie_reservations" / "quick_seed71311.json"
            )
            reservation.parent.mkdir(parents=True)
            reservation.write_text('{"seed":71311}\n')
            cfg = {"menagerie": {"paired_seeds": {"quick": 71311}}}
            original_exp = run_pipeline.EXP
            try:
                run_pipeline.EXP = exp
                self.assertEqual(
                    run_pipeline.unlogged_menagerie_reservations(cfg, []),
                    [reservation],
                )
                self.assertEqual(
                    run_pipeline.unlogged_menagerie_reservations(
                        cfg, [{"tier": "quick", "seed": 71311}]
                    ),
                    [],
                )
            finally:
                run_pipeline.EXP = original_exp

    def test_external_seed_collision_uses_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exp = root / "experiments" / "current"
            other = (
                root / "experiments" / "other" / "runs"
                / "menagerie_reservations" / "quick_seed71311.json"
            )
            other.parent.mkdir(parents=True)
            other.write_text(json.dumps({"tier": "quick", "seed": 71311}))
            cfg = {"menagerie": {"paired_seeds": {"quick": 71311}}}
            original_root, original_exp = run_pipeline.ROOT, run_pipeline.EXP
            try:
                run_pipeline.ROOT, run_pipeline.EXP = root, exp
                collisions = run_pipeline.external_menagerie_seed_collisions(
                    cfg, []
                )
                self.assertEqual(collisions, {71311: [other]})
            finally:
                run_pipeline.ROOT, run_pipeline.EXP = original_root, original_exp

    def test_seed_unavailable_records_prior_local_exposure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            lock = exp / "runs" / "preregistration_receipt.json"
            lock.parent.mkdir(parents=True)
            lock.write_text('{"status":"locked"}\n')
            authorization = exp / "analysis" / "whitebox_authorization.json"
            authorization.parent.mkdir(parents=True)
            authorization.write_text('{"menagerie_authorized":true}\n')
            external = Path(directory) / "external_reservation.json"
            external.write_text('{"seed":71312}\n')
            local_log = exp / "runs" / "menagerie_log.jsonl"
            local_log.write_text('{"tier":"quick","seed":71311}\n')
            original_exp = run_pipeline.EXP
            try:
                run_pipeline.EXP = exp
                code = run_pipeline.terminalize_unavailable_menagerie_seed(
                    {71312: [external]}, authorization, [local_log]
                )
                self.assertEqual(code, 4)
                terminal = json.loads(
                    (exp / "runs" / "terminal_disposition.json").read_text()
                )
                availability = json.loads(
                    (exp / "analysis" / "menagerie_seed_availability.json").read_text()
                )
                self.assertTrue(terminal["menagerie_exposed"])
                self.assertTrue(availability["local_exposure_evidence_present"])
            finally:
                run_pipeline.EXP = original_exp


class CloseoutGuardTests(unittest.TestCase):
    def _write_surfaces(self, root: Path, exp: Path) -> dict:
        experiment_id = exp.name
        (exp / "reports").mkdir(parents=True)
        (exp / "README.md").write_text("**Status:** finished\n")
        (exp / "reports" / "report.md").write_text("FINAL_VERDICT\n")
        (exp / "experiment_log.md").write_text("FINAL_VERDICT\n")
        (exp / "reports" / "artifact_manifest.yaml").write_text(
            f"experiment_id: {experiment_id}\nverdict: FINAL_VERDICT\n"
        )
        (exp / "metadata.yaml").write_text(
            f"id: {experiment_id}\nfile_counts:\n  reports: 2\ntotal_files: 5\n"
        )
        program = root / "research_programs" / "agentic_breadth_installation"
        program.mkdir(parents=True)
        (program / "evidence.md").write_text(f"{experiment_id} FINAL_VERDICT")
        (program / "backlog.md").write_text(f"{experiment_id} FINAL_VERDICT")
        knowledge = root / "knowledge"
        (knowledge / "claims").mkdir(parents=True)
        (knowledge / "program_scorecards.md").write_text(
            f"{experiment_id} FINAL_VERDICT"
        )
        (knowledge / "synthesis.md").write_text("no shared claim warranted\n")
        (knowledge / "experiment_brief.json").write_text(json.dumps({
            "experiments": {
                experiment_id: {
                    "verdict_tag": "Finished result",
                    "plain_answer": "The final gate stopped as registered.",
                }
            }
        }))
        (knowledge / "experiment_viz.json").write_text(json.dumps({
            "experiments": {
                experiment_id: {
                    "charts": [{
                        "headline": True,
                        "title": "Final result",
                        "source": f"experiments/{experiment_id}/reports/result.json",
                    }]
                }
            }
        }))
        (knowledge / "claims" / "claim_ledger.json").write_text("{}\n")
        (knowledge / "experiment_status.json").write_text(
            json.dumps({"experiments": {}})
        )
        (knowledge / "experiment_dates.json").write_text(json.dumps({
            "experiments": {experiment_id: {"start": "2026-07-13"}}
        }))
        for name in ("experiment_catalog.csv", "experiment_readiness.csv"):
            (knowledge / name).write_text(f"id,title\n{experiment_id},Result\n")
        return {
            "terminal": True,
            "experiment_id": experiment_id,
            "verdict": "FINAL_VERDICT",
        }

    def test_closeout_rejects_lingering_in_progress_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            exp = root / "experiments" / EXP.name
            payload = self._write_surfaces(root, exp)
            status = root / "knowledge" / "experiment_status.json"
            status.write_text(json.dumps({
                "experiments": {exp.name: {"status": "in-progress"}}
            }))
            original_root, original_exp = run_pipeline.ROOT, run_pipeline.EXP
            try:
                run_pipeline.ROOT, run_pipeline.EXP = root, exp
                with self.assertRaisesRegex(SystemExit, "remains in"):
                    run_pipeline._validate_closeout_semantics(payload)
            finally:
                run_pipeline.ROOT, run_pipeline.EXP = original_root, original_exp

    def test_closeout_rejects_design_only_practitioner_brief(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            exp = root / "experiments" / EXP.name
            payload = self._write_surfaces(root, exp)
            brief = root / "knowledge" / "experiment_brief.json"
            brief.write_text(json.dumps({
                "experiments": {
                    exp.name: {
                        "plain_answer": "Designed and preregistered; no model has run."
                    }
                }
            }))
            original_root, original_exp = run_pipeline.ROOT, run_pipeline.EXP
            try:
                run_pipeline.ROOT, run_pipeline.EXP = root, exp
                with self.assertRaisesRegex(SystemExit, "active design"):
                    run_pipeline._validate_closeout_semantics(payload)
            finally:
                run_pipeline.ROOT, run_pipeline.EXP = original_root, original_exp

    def test_terminal_disposition_blocks_same_experiment_rescue(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / EXP.name
            terminal = exp / "runs" / "terminal_disposition.json"
            terminal.parent.mkdir(parents=True)
            terminal.write_text(json.dumps({
                "terminal": True,
                "verdict": "CALIBRATION_FAIL",
            }))
            original_exp = run_pipeline.EXP
            try:
                run_pipeline.EXP = exp
                with self.assertRaisesRegex(SystemExit, "rescue or rerun"):
                    run_pipeline.assert_scientific_run_open()
            finally:
                run_pipeline.EXP = original_exp

    def test_positive_terminal_verdict_matches_preregistration(self) -> None:
        self.assertEqual(
            run_pipeline.final_menagerie_verdict(True),
            "COUNTERFACTUAL_EVIDENCE_ACQUISITION_POSITIVE",
        )
        self.assertEqual(
            run_pipeline.final_menagerie_verdict(False),
            "MENAGERIE_FAIL",
        )

    def test_closeout_rejects_stale_program_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            exp = root / "experiments" / EXP.name
            payload = self._write_surfaces(root, exp)
            scorecard = root / "knowledge" / "program_scorecards.md"
            scorecard.write_text(f"{exp.name} is active; no result yet.\n")
            original_root, original_exp = run_pipeline.ROOT, run_pipeline.EXP
            try:
                run_pipeline.ROOT, run_pipeline.EXP = root, exp
                with self.assertRaisesRegex(SystemExit, "terminal verdict"):
                    run_pipeline._validate_closeout_semantics(payload)
            finally:
                run_pipeline.ROOT, run_pipeline.EXP = original_root, original_exp

    def test_closed_seal_rejects_empty_closeout_file_map(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            exp = root / "experiments" / EXP.name
            self._write_surfaces(root, exp)
            payload = {
                "schema_version": 1,
                "terminal": True,
                "experiment_id": exp.name,
                "lifecycle_closed": True,
                "closeout_receipt_requires_push": True,
                "closeout_documentation_commit": "a" * 40,
                "open_terminal_disposition_sha256": "b" * 64,
                "closeout_files": {},
                "closeout_validation": {
                    "command": "make check",
                    "passed": True,
                    "validated_commit": "a" * 40,
                },
            }
            original_root, original_exp = run_pipeline.ROOT, run_pipeline.EXP
            try:
                run_pipeline.ROOT, run_pipeline.EXP = root, exp
                with self.assertRaisesRegex(SystemExit, "forged closeout seal"):
                    run_pipeline._validate_closeout_seal_contract(payload)
            finally:
                run_pipeline.ROOT, run_pipeline.EXP = original_root, original_exp


class DesignLockGuardTests(unittest.TestCase):
    def test_partial_or_moving_lock_is_rejected_before_model_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = (
                Path(directory)
                / "repository"
                / "experiments"
                / "qwen35_4b_counterfactual_evidence_acquisition_curriculum"
            )
            caller = exp / "scripts" / "eval_repo_agent.py"
            caller.parent.mkdir(parents=True)
            caller.write_text("# frozen evaluator\n")
            lock = exp / "runs" / "preregistration_receipt.json"
            lock.parent.mkdir()
            base = {
                "schema_version": 1,
                "status": "locked",
                "experiment_id": exp.name,
                "model_output_precedes_lock": False,
                "frozen_file_order": ["scripts/eval_repo_agent.py"],
                "design_commit": "a" * 40,
            }
            lock.write_text(json.dumps({**base, "frozen_files": {}}))
            with self.assertRaisesRegex(ValueError, "not covered"):
                harness.validate_model_execution_lock(
                    exp, lock, "scripts/eval_repo_agent.py"
                )
            lock.write_text(json.dumps({
                **base,
                "design_commit": "HEAD",
                "frozen_files": {
                    "scripts/eval_repo_agent.py": train.sha256_file(caller)
                },
            }))
            with self.assertRaisesRegex(ValueError, "not covered"):
                harness.validate_model_execution_lock(
                    exp, lock, "scripts/eval_repo_agent.py"
                )
            lock.write_text(json.dumps({
                **base,
                "design_commit": "a" * 40,
                "frozen_files": {
                    "scripts/eval_repo_agent.py": train.sha256_file(caller)
                },
            }))
            with self.assertRaisesRegex(ValueError, "not covered"):
                harness.validate_model_execution_lock(
                    exp, lock, "scripts/eval_repo_agent.py"
                )

    def test_training_rejects_noncanonical_lock_path_before_git_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            exp = Path(directory) / "experiment"
            alternate = exp / "alternate_lock.json"
            alternate.parent.mkdir(parents=True)
            alternate.write_text("{}\n")
            original_exp = train.EXP
            try:
                train.EXP = exp
                with self.assertRaisesRegex(SystemExit, "design lock is missing"):
                    train.validate_design_lock(
                        alternate,
                        bank_receipt_path=exp / "bank.json",
                        arm="evidence_binding",
                        train_sha256="a" * 64,
                    )
            finally:
                train.EXP = original_exp


if __name__ == "__main__":
    unittest.main()
