from __future__ import annotations

import copy
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from confirmation_artifacts import (  # noqa: E402
    MARKER_FILENAMES,
    RAW_FILENAMES,
    _sampled_token_evidence,
    _validate_call_journal,
    _validate_journal_call_geometry,
    _validate_preregistered_confirmation_tasks,
    begin_confirmation_transaction,
    commit_confirmation_score,
    confirmation_admission_binding,
    confirmation_task_hashes,
    confirmation_transaction_state,
    configured_confirmation_raw_root,
    controls_authorization_binding,
    confirmation_raw_dir,
    finalize_confirmation_score,
    journal_confirmation_bundle,
    prepare_confirmation_output,
    quarantine_confirmation_transaction,
    validate_confirmation_geometry,
    validate_confirmation_campaign_tree,
    validate_confirmation_score_artifacts,
)
from io_utils import canonical_hash, sha256_file  # noqa: E402
from confirmation_protocol import capacity_receipt  # noqa: E402
from model_provenance import (  # noqa: E402
    CONFIRMATION_ARM_NAMES,
    SOURCE_RECEIPT,
    SOURCE_RECEIPT_COMMIT,
    confirmation_arm_map_sha256,
)


class ConfirmationArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.score_root = root / "visible"
        self.raw_root = root / "large"
        self.score_path = self.score_root / "block_0" / "arm" / "scores.json"
        self.tag = "block_0_arm"

    def tearDown(self):
        self.temporary.cleanup()

    def _commit(self) -> dict:
        return commit_confirmation_score(
            self.score_path,
            {
                "stage": "unit_test",
                "tag": self.tag,
                "runner_summary": [
                    {"counts": {"sampled_tokens": 3, "requests": 3, "completions": 3}}
                ],
                "items": [
                    {
                        "key": "a",
                        "family": "family",
                        "kind": "atom",
                        "level": 1,
                        "samples": 1,
                        "score": 0.25,
                    },
                    {
                        "key": "b",
                        "family": "family",
                        "kind": "atom",
                        "level": 2,
                        "samples": 1,
                        "score": 0.5,
                    },
                    {
                        "key": "family/episode/L3/s7",
                        "family": "family",
                        "kind": "episode",
                        "level": 3,
                        "samples": 1,
                        "score": 0.75,
                    },
                ],
            },
            atom_rows=[
                {
                    "id": "a",
                    "family": "family",
                    "level": 1,
                    "prompt": "prompt-a",
                    "gold": 1,
                    "answer_domain": [0, 1],
                    "outputs": [{
                        "sample_index": 0,
                        "score": 0.25,
                        "n_sampled_tokens": 1,
                        "stage1_sampled_token_ids": [10],
                        "stage2_sampled_token_ids": [],
                    }],
                },
                {
                    "id": "b",
                    "family": "family",
                    "level": 2,
                    "prompt": "prompt-b",
                    "gold": 2,
                    "answer_domain": None,
                    "outputs": [{
                        "sample_index": 0,
                        "score": 0.5,
                        "n_sampled_tokens": 1,
                        "stage1_sampled_token_ids": [20],
                        "stage2_sampled_token_ids": [],
                    }],
                },
            ],
            episode_rows=[
                {
                    "family": "family",
                    "level": 3,
                    "ep_seed": 7,
                    "rollout": 0,
                    "score": 0.75,
                    "rid": "family-L3-e7-r0",
                    "spec": {"target": 7},
                    "system_prompt": "system",
                    "initial_observation": "initial",
                    "turns": [{
                        "turn": 0,
                        "n_sampled_tokens": 1,
                        "stage1_sampled_token_ids": [30],
                        "stage2_sampled_token_ids": [],
                    }],
                }
            ],
            score_root=self.score_root,
            raw_root=self.raw_root,
        )

    def _validate(self, tag: str | None = None) -> dict:
        return validate_confirmation_score_artifacts(
            self.score_path,
            expected_tag=tag or self.tag,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )

    def test_configured_raw_root_is_frozen_to_this_experiment(self):
        expected = EXP.parents[1] / "large_artifacts" / EXP.name / "confirmation"
        self.assertEqual(
            configured_confirmation_raw_root(
                {"model": {"artifacts_root": f"large_artifacts/{EXP.name}"}}
            ),
            expected.resolve(),
        )
        with self.assertRaisesRegex(ValueError, "frozen experiment root"):
            configured_confirmation_raw_root(
                {"model": {"artifacts_root": self.raw_root}}
            )

    def test_configured_raw_root_rejects_leaf_and_ancestor_symlinks(self):
        root = Path(self.temporary.name) / "symlink-repo"
        experiment = root / "experiments" / "exp"
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        root.mkdir()
        with mock.patch.multiple(
            "confirmation_artifacts", REPO=root, EXP=experiment
        ):
            large = root / "large_artifacts"
            large.symlink_to(outside, target_is_directory=True)
            configured = root / "large_artifacts" / experiment.name
            with self.assertRaisesRegex(ValueError, "symlink"):
                configured_confirmation_raw_root(
                    {"model": {"artifacts_root": str(configured)}}
                )
            large.unlink()
            configured.mkdir(parents=True)
            (configured / "confirmation").symlink_to(
                outside, target_is_directory=True
            )
            with self.assertRaisesRegex(ValueError, "symlink"):
                configured_confirmation_raw_root(
                    {"model": {"artifacts_root": str(configured)}}
                )

    def test_full_geometry_rejects_smoke_or_family_subset(self):
        config = {
            "strata": {
                "trained_families": ["trained"],
                "transfer_families": ["transfer"],
                "quick_atom_levels": [1],
                "deep_atom_levels": [2],
                "deep_episode_levels": [3],
            },
            "confirmation": {
                "atoms_per_family_level": 2,
                "episodes_per_family_level": 1,
            },
            "controls": {"sample_more_k": 8},
        }
        items = []
        for family in ("trained", "transfer"):
            for level, stratum in ((1, "quick"), (2, "deep")):
                for index in range(2):
                    items.append(
                        {
                            "key": f"{family}-L{level}-{index}",
                            "family": family,
                            "kind": "atom",
                            "level": level,
                            "stratum": stratum,
                            "score": 0.5,
                            "samples": 1,
                        }
                    )
            items.append(
                {
                    "key": f"{family}/episode/L3/s7",
                    "family": family,
                    "kind": "episode",
                    "level": 3,
                    "stratum": "deep",
                    "score": 0.5,
                    "samples": 1,
                }
            )
        payload = {
            "scope": "confirmatory",
            "families": ["trained", "transfer"],
            "atoms_per_level": 2,
            "episodes_per_level": 1,
            "decode": "greedy",
            "k": 1,
            "items": items,
        }
        family = SimpleNamespace(LEVELS={1, 2, 3}, HAS_EPISODES=True)
        with mock.patch("confirmation_artifacts.load_family", return_value=family):
            validate_confirmation_geometry(payload, config)
            smoke = {**payload, "atoms_per_level": 1}
            with self.assertRaisesRegex(ValueError, "frozen full geometry"):
                validate_confirmation_geometry(smoke, config)
            subset = {**payload, "families": ["trained"]}
            with self.assertRaisesRegex(ValueError, "frozen full geometry"):
                validate_confirmation_geometry(subset, config)
            missing_cell = {**payload, "items": items[:-1]}
            with self.assertRaisesRegex(ValueError, "cells"):
                validate_confirmation_geometry(missing_cell, config)

    def test_score_is_last_commit_marker_and_descriptors_are_exact(self):
        payload = self._commit()
        raw_dir = confirmation_raw_dir(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(
            set(path.name for path in raw_dir.iterdir()),
            set(RAW_FILENAMES.values())
            | {
                MARKER_FILENAMES["started"],
                MARKER_FILENAMES["generated"],
                MARKER_FILENAMES["complete"],
            },
        )
        self.assertEqual(payload, self._validate())
        self.assertEqual(payload["raw_artifacts"]["atom_rows"]["rows"], 2)
        self.assertEqual(payload["raw_artifacts"]["episode_rows"]["rows"], 1)
        for key, filename in RAW_FILENAMES.items():
            descriptor = payload["raw_artifacts"][key]
            path = raw_dir / filename
            self.assertEqual(descriptor["path"], str(path))
            self.assertEqual(descriptor["bytes"], path.stat().st_size)
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                self.assertEqual(len(handle.readlines()), descriptor["rows"])

    def test_failed_raw_write_is_quarantined_and_never_resampled(self):
        with self.assertRaises(ValueError):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        },
                        {
                            "key": "family/episode/L3/s7",
                            "family": "family",
                            "kind": "episode",
                            "level": 3,
                            "samples": 2,
                            "score": 0.75,
                        },
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.25}],
                    }
                ],
                episode_rows=[
                    {
                        "family": "family",
                        "level": 3,
                        "ep_seed": 7,
                        "rollout": 0,
                        "score": 0.75,
                    },
                    "not-an-object",
                ],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())
        raw_dir = confirmation_raw_dir(
            self.score_path, score_root=self.score_root, raw_root=self.raw_root
        )
        before = {
            path.name: path.read_bytes() for path in raw_dir.iterdir() if path.is_file()
        }
        with self.assertRaisesRegex(ValueError, "QUARANTINED"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertEqual(
            before,
            {path.name: path.read_bytes() for path in raw_dir.iterdir() if path.is_file()},
        )

    def test_raw_row_count_must_match_scored_item_geometry(self):
        with self.assertRaisesRegex(ValueError, "item geometry"):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        }
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.25}],
                    },
                    {
                        "id": "b",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 0, "score": 0.5}],
                    },
                ],
                episode_rows=[],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())

    def test_score_item_semantics_are_recomputed_from_raw_rows(self):
        self._commit()
        payload = json.loads(self.score_path.read_text(encoding="utf-8"))
        payload["items"][0]["score"] = 999.0
        self.score_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "COMPLETE marker|raw semantics"):
            self._validate()

    def test_raw_sample_indexes_must_be_contiguous(self):
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            commit_confirmation_score(
                self.score_path,
                {
                    "stage": "policy_eval",
                    "tag": self.tag,
                    "items": [
                        {
                            "key": "a",
                            "family": "family",
                            "kind": "atom",
                            "level": 1,
                            "samples": 1,
                            "score": 0.25,
                        }
                    ],
                },
                atom_rows=[
                    {
                        "id": "a",
                        "family": "family",
                        "level": 1,
                        "outputs": [{"sample_index": 1, "score": 0.25}],
                    }
                ],
                episode_rows=[],
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertFalse(self.score_path.exists())

    def test_missing_or_tampered_raw_fails_closed(self):
        payload = self._commit()
        raw = Path(payload["raw_artifacts"]["atom_rows"]["path"])
        raw.write_bytes(raw.read_bytes() + b"tamper")
        with self.assertRaisesRegex(ValueError, "descriptor is stale"):
            self._validate()
        raw.unlink()
        with self.assertRaisesRegex(ValueError, "unknown or missing"):
            self._validate()

    def test_tag_and_descriptor_path_are_validated(self):
        self._commit()
        with self.assertRaisesRegex(ValueError, "tag"):
            self._validate("block_0_other")
        payload = json.loads(self.score_path.read_text(encoding="utf-8"))
        payload["raw_artifacts"]["atom_rows"]["path"] = str(
            Path(self.temporary.name) / "escaped.jsonl.gz"
        )
        self.score_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "transaction provenance|raw path"):
            self._validate()

    def test_unknown_partials_are_never_recovered(self):
        self.score_path.parent.mkdir(parents=True)
        (self.score_path.parent / "mystery.bin").write_bytes(b"x")
        with self.assertRaisesRegex(ValueError, "unknown partial"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_formerly_known_orphans_are_never_deleted(self):
        self.score_path.parent.mkdir(parents=True)
        raw_dir = confirmation_raw_dir(
            self.score_path,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        raw_dir.mkdir(parents=True)
        (self.score_path.parent / "atom_rows.jsonl.gz").write_bytes(b"old")
        (self.score_path.parent / ".scores.json.crash.tmp").write_bytes(b"partial")
        (raw_dir / "episode_rows.jsonl.gz").write_bytes(b"old")
        (raw_dir / ".atom_rows.jsonl.gz.crash.tmp").write_bytes(b"partial")
        before = {
            path: path.read_bytes()
            for path in [*self.score_path.parent.iterdir(), *raw_dir.iterdir()]
        }
        with self.assertRaises(ValueError):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_committed_score_rejects_extras_and_cannot_be_auto_recovered(self):
        self._commit()
        (self.score_path.parent / ".scores.json.crash.tmp").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "unknown partials"):
            self._validate()
        with self.assertRaisesRegex(ValueError, "unknown partial|commit marker already exists"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_score_path_escape_is_rejected(self):
        outside = Path(self.temporary.name) / "outside" / "scores.json"
        with self.assertRaisesRegex(ValueError, "escaped"):
            prepare_confirmation_output(
                outside,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_started_reservation_is_no_clobber_and_terminal(self):
        first = begin_confirmation_transaction(
            self.score_path,
            {"tag": self.tag, "task_manifest_sha256": "a" * 64},
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        marker = confirmation_raw_dir(
            self.score_path, score_root=self.score_root, raw_root=self.raw_root
        ) / MARKER_FILENAMES["started"]
        before = marker.read_bytes()
        with self.assertRaises(ValueError):
            begin_confirmation_transaction(
                self.score_path,
                {"tag": self.tag, "task_manifest_sha256": "b" * 64},
                score_root=self.score_root,
                raw_root=self.raw_root,
            )
        self.assertEqual(marker.read_bytes(), before)
        self.assertEqual(
            confirmation_transaction_state(
                self.score_path, score_root=self.score_root, raw_root=self.raw_root
            ),
            "STARTED",
        )

    def test_production_begin_rejects_missing_policy_admission_before_marker(self):
        with mock.patch(
            "confirmation_artifacts.CONFIRMATION_SCORE_ROOT", self.score_root
        ):
            with self.assertRaisesRegex(ValueError, "context is incomplete"):
                begin_confirmation_transaction(
                    self.score_path,
                    {"stage": "policy_eval", "tag": self.tag},
                    score_root=self.score_root,
                    raw_root=self.raw_root,
                )
        self.assertFalse(self.score_path.parent.exists())
        self.assertFalse(self.raw_root.exists())

    def test_started_and_generated_markers_have_exact_schema_and_path_binding(self):
        self._commit()
        raw_dir = confirmation_raw_dir(
            self.score_path, score_root=self.score_root, raw_root=self.raw_root
        )
        marker_mutations = (
            ("started", lambda row: row.update(extra=True)),
            ("started", lambda row: row.update(schema_version=2)),
            ("started", lambda row: row.update(score_path=str(self.score_path) + ".other")),
            ("generated", lambda row: row.update(extra=True)),
            ("generated", lambda row: row.update(schema_version=2)),
        )
        for marker_name, mutate in marker_mutations:
            with self.subTest(marker=marker_name, mutation=mutate):
                marker = raw_dir / MARKER_FILENAMES[marker_name]
                before = marker.read_bytes()
                payload = json.loads(before)
                mutate(payload)
                marker.write_text(json.dumps(payload) + "\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    self._validate()
                marker.write_bytes(before)
        self.assertEqual(self._validate()["tag"], self.tag)

    def test_complete_and_generated_states_resume_without_new_bytes(self):
        committed = self._commit()
        raw_dir = confirmation_raw_dir(
            self.score_path, score_root=self.score_root, raw_root=self.raw_root
        )
        self.score_path.unlink()
        self.assertEqual(
            confirmation_transaction_state(
                self.score_path, score_root=self.score_root, raw_root=self.raw_root
            ),
            "COMPLETE",
        )
        resumed = finalize_confirmation_score(
            self.score_path,
            expected_tag=self.tag,
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(resumed, committed)
        self.score_path.unlink()
        (raw_dir / MARKER_FILENAMES["complete"]).unlink()
        self.assertEqual(
            confirmation_transaction_state(
                self.score_path, score_root=self.score_root, raw_root=self.raw_root
            ),
            "GENERATED",
        )
        self.assertEqual(
            finalize_confirmation_score(
                self.score_path,
                expected_tag=self.tag,
                score_root=self.score_root,
                raw_root=self.raw_root,
            ),
            committed,
        )

    def test_quarantine_hashes_and_retains_each_returned_call_bundle(self):
        begin_confirmation_transaction(
            self.score_path,
            {"tag": self.tag},
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        descriptor = journal_confirmation_bundle(
            self.score_path,
            {"rows": [{"id": "sentinel"}], "summary": {"counts": {}}},
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        bundle = Path(descriptor["path"])
        before = bundle.read_bytes()
        quarantine_confirmation_transaction(
            self.score_path,
            reason="second generation call failed",
            score_root=self.score_root,
            raw_root=self.raw_root,
        )
        self.assertEqual(bundle.read_bytes(), before)
        self.assertEqual(
            confirmation_transaction_state(
                self.score_path, score_root=self.score_root, raw_root=self.raw_root
            ),
            "QUARANTINED",
        )

    def test_sampled_token_count_is_derived_from_stage_ids(self):
        valid = {
            "stage1_sampled_token_ids": [10, 248044],
            "stage2_sampled_token_ids": [20],
            "n_sampled_tokens": 3,
        }
        self.assertEqual(_sampled_token_evidence(valid, label="test"), 3)
        for mutation in (
            {**valid, "n_sampled_tokens": 2},
            {**valid, "n_sampled_tokens": True},
            {**valid, "stage1_sampled_token_ids": [10, False]},
            {key: value for key, value in valid.items() if key != "stage2_sampled_token_ids"},
        ):
            with self.assertRaises(ValueError):
                _sampled_token_evidence(mutation, label="test")

    def test_task_hash_binds_prompt_spec_and_order(self):
        atoms = [
            {"id": "a", "family": "f", "level": 1, "prompt": "p", "gold": 1, "answer_domain": None},
            {"id": "b", "family": "f", "level": 1, "prompt": "q", "gold": 2, "answer_domain": None},
        ]
        episodes = [{
            "family": "f", "level": 2, "ep_seed": 7, "spec": {"x": 1},
            "system_prompt": "s", "initial_observation": "o",
        }]
        baseline = confirmation_task_hashes(atoms, episodes)
        prompt_drift = copy.deepcopy(atoms)
        prompt_drift[0]["prompt"] = "changed"
        self.assertNotEqual(
            baseline["task_manifest_sha256"],
            confirmation_task_hashes(prompt_drift, episodes)["task_manifest_sha256"],
        )
        self.assertNotEqual(
            baseline["ordered_plan_sha256"],
            confirmation_task_hashes(list(reversed(atoms)), episodes)["ordered_plan_sha256"],
        )

    def test_call_journal_replays_text_scores_and_episode_transitions(self):
        def generated_output(text: str, token: int) -> dict:
            return {
                "sample_index": 0,
                "stage1_parent_seed": 1,
                "seed_stage1": 1,
                "seed_stage2": None,
                "text": text,
                "token_ids": [token],
                "stage1_token_ids": [token],
                "retained_thinking_token_ids": [],
                "injected_token_ids": [],
                "stage2_token_ids": [],
                "n_thinking_tokens": 0,
                "n_answer_tokens": 1,
                "n_sampled_tokens": 1,
                "n_injected_tokens": 0,
                "n_completion_tokens": 1,
                "n_terminal_tokens_trimmed": 0,
                "n_stage1_prompt_tokens": 4,
                "n_stage2_prompt_tokens": 0,
                "thinking_closed": True,
                "forced_close": False,
                "finish_reason": "stop",
                "stop_reason": None,
                "stage1_finish_reason": "stop",
                "stage1_stop_reason": None,
                "truncated": False,
                "stage1_cumulative_logprob": -1.0,
                "stage2_cumulative_logprob": None,
                "sampled_cumulative_logprob": -1.0,
                "stage1_logprobs": None,
                "stage2_logprobs": None,
            }

        atom_output = generated_output("atom answer", 10)
        atom_output_b = generated_output("atom answer b", 11)
        episode_output = generated_output("episode action", 20)
        atom_request = {
            "id": "a",
            "meta": None,
            "prompt_sha256": "a" * 64,
            "n_prompt_tokens": 4,
            "prompt_channel": "messages",
            "prompt_logprobs": None,
            "outputs": [atom_output],
        }
        atom_request_b = {
            "id": "b",
            "meta": None,
            "prompt_sha256": "c" * 64,
            "n_prompt_tokens": 5,
            "prompt_channel": "messages",
            "prompt_logprobs": None,
            "outputs": [atom_output_b],
        }
        episode_request = {
            "id": "fake-L1-e7-r0-t0",
            "meta": None,
            "prompt_sha256": "b" * 64,
            "n_prompt_tokens": 4,
            "prompt_channel": "messages",
            "prompt_logprobs": None,
            "outputs": [episode_output],
        }
        input_records = {
            "a": {
                "id": "a",
                "messages": [{"role": "user", "content": "prompt"}],
            },
            "b": {
                "id": "b",
                "messages": [{"role": "user", "content": "prompt-b"}],
            },
            "fake-L1-e7-r0-t0": {
                "id": "fake-L1-e7-r0-t0",
                "messages": [
                    {"role": "system", "content": "system"},
                    {"role": "user", "content": "initial"},
                ],
            },
        }

        def request_evidence(request: dict, prompt_ids: list[int]) -> dict:
            return {
                "schema_version": 1,
                "id": request["id"],
                "record_sha256": canonical_hash(input_records[request["id"]]),
                "prompt_token_ids_sha256": canonical_hash(prompt_ids),
                "prompt_sha256": request["prompt_sha256"],
                "n_prompt_tokens": request["n_prompt_tokens"],
                "prompt_channel": request["prompt_channel"],
            }

        atom_evidence = request_evidence(atom_request, [1, 2, 3, 4])
        atom_evidence_b = request_evidence(atom_request_b, [1, 2, 3, 4, 5])
        episode_evidence = request_evidence(episode_request, [6, 7, 8, 9])
        sampling = {
            "thinking": "budget",
            "thinking_budget": 10,
            "answer_max_tokens": 10,
            "max_tokens": 20,
            "n": 1,
        }
        static_capacity = {
            "formula": {
                "attention_block_tokens": 528,
                "mamba_block_tokens": 16384,
                "full_attention_layers": 8,
                "linear_attention_layers": 24,
                "mamba_groups": 3,
                "enable_prefix_caching": False,
                "mamba_cache_mode": "none",
            },
            "max_model_len": 100,
            "max_num_seqs": 8,
            "kv_cache_size_tokens": 100_000,
            "num_gpu_blocks": 1_000,
            "forced_close_tokens": 2,
        }

        def summary(call: str, rows: list[dict], prompts: list[int]) -> dict:
            outputs = [output for row in rows for output in row["outputs"]]
            stage1 = sum(output["n_stage1_prompt_tokens"] for output in outputs)
            stage2 = sum(output["n_stage2_prompt_tokens"] for output in outputs)
            return {
                "call": call,
                "sampling": dict(sampling),
                "confirmation_capacity": capacity_receipt(
                    static_capacity,
                    prompt_token_lengths=prompts,
                    sampling=SimpleNamespace(**sampling),
                ),
                "counts": {
                    "requests": len(rows),
                    "completions": len(outputs),
                    "unique_input_prompt_tokens": sum(prompts),
                    "stage1_logical_prompt_tokens": stage1,
                    "stage2_logical_prompt_tokens": stage2,
                    "logical_model_input_tokens": stage1 + stage2,
                    "sampled_tokens": sum(
                        output["n_sampled_tokens"] for output in outputs
                    ),
                    "injected_tokens": sum(
                        output["n_injected_tokens"] for output in outputs
                    ),
                },
            }

        summaries = [
            summary("atom", [atom_request, atom_request_b], [4, 5]),
            summary("episode", [episode_request], [4]),
        ]
        bundle_payloads = [
            {
                "rows": [atom_request, atom_request_b],
                "summary": summaries[0],
                "request_evidence": [atom_evidence, atom_evidence_b],
            },
            {
                "rows": [episode_request],
                "summary": summaries[1],
                "request_evidence": [episode_evidence],
            },
        ]
        descriptors = []
        for index, bundle in enumerate(bundle_payloads):
            path = Path(self.temporary.name) / f"bundle_{index:04d}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(bundle) + "\n")
            descriptors.append({"path": str(path)})

        def raw_output(output: dict, request: dict, pre_call: dict) -> dict:
            fields = (
                "sample_index",
                "text",
                "n_thinking_tokens",
                "n_answer_tokens",
                "n_sampled_tokens",
                "thinking_closed",
                "forced_close",
                "finish_reason",
                "truncated",
                "token_ids",
                "retained_thinking_token_ids",
                "injected_token_ids",
            )
            return {
                **{field: copy.deepcopy(output[field]) for field in fields},
                "stage1_sampled_token_ids": list(output["stage1_token_ids"]),
                "stage2_sampled_token_ids": list(output["stage2_token_ids"]),
                "generation_request_sha256": canonical_hash(
                    {key: value for key, value in request.items() if key != "outputs"}
                ),
                "generation_output_sha256": canonical_hash(output),
                "generation_record_sha256": pre_call["record_sha256"],
                "generation_prompt_token_ids_sha256": pre_call[
                    "prompt_token_ids_sha256"
                ],
            }

        atom_raw = raw_output(atom_output, atom_request, atom_evidence)
        atom_raw.update(score=1.0, answer_value="parsed-answer")
        atom_raw_b = raw_output(atom_output_b, atom_request_b, atom_evidence_b)
        atom_raw_b.update(score=1.0, answer_value="parsed-answer")
        episode_turn = raw_output(
            episode_output, episode_request, episode_evidence
        )
        episode_turn.update(
            turn=0,
            action="parsed-action",
            action_ok=True,
            observation="terminal observation",
            context_messages=2,
        )
        atoms = [
            {
                "id": "a",
                "family": "fake",
                "level": 1,
                "prompt": "prompt",
                "gold": "gold",
                "answer_domain": None,
                "outputs": [atom_raw],
            },
            {
                "id": "b",
                "family": "fake",
                "level": 1,
                "prompt": "prompt-b",
                "gold": "gold-b",
                "answer_domain": None,
                "outputs": [atom_raw_b],
            },
        ]
        episodes = [
            {
                "rid": "fake-L1-e7-r0",
                "family": "fake",
                "level": 1,
                "ep_seed": 7,
                "rollout": 0,
                "spec": {"seed": 7},
                "system_prompt": "system",
                "initial_observation": "initial",
                "turns": [episode_turn],
                "done": True,
                "score": 1.0,
                "n_turns": 1,
                "max_turns": 1,
            }
        ]

        class FakeEpisode:
            max_turns = 1

            def __init__(self, seed, level):
                self.spec = {"seed": seed}
                self.last_action_ok = True

            def system_prompt(self):
                return "system"

            def initial_observation(self):
                return "initial"

            def step(self, action):
                self.last_action_ok = action == "parsed-action"
                return "terminal observation", True

            def score(self):
                return 1.0

        family = SimpleNamespace(
            Episode=FakeEpisode,
            score_atom=lambda _task, text: (
                1.0 if text in {"atom answer", "atom answer b"} else 0.0
            ),
        )
        generated = {"call_bundles": descriptors}
        payload = {"stage": "policy_eval", "runner_summary": summaries}
        patches = (
            mock.patch("confirmation_artifacts.load_family", return_value=family),
            mock.patch(
                "confirmation_artifacts.base.extract_answer",
                return_value="parsed-answer",
            ),
            mock.patch(
                "confirmation_artifacts.base.extract_action",
                return_value="parsed-action",
            ),
        )
        with patches[0], patches[1], patches[2]:
            _validate_call_journal(
                payload, generated, atom_rows=atoms, episode_rows=episodes
            )
            understated = copy.deepcopy(summaries[0])
            understated["confirmation_capacity"] = capacity_receipt(
                static_capacity,
                prompt_token_lengths=[1, 1],
                sampling=SimpleNamespace(**sampling),
            )
            with self.assertRaisesRegex(ValueError, "actual request geometry"):
                _validate_journal_call_geometry(
                    understated,
                    request_evidence=[atom_evidence, atom_evidence_b],
                    rows=[atom_request, atom_request_b],
                )
            mutations = []
            changed = copy.deepcopy(atoms)
            changed[0]["outputs"][0]["text"] = "changed"
            mutations.append((changed, episodes))
            changed = copy.deepcopy(atoms)
            changed[0]["outputs"][0]["score"] = 0.0
            mutations.append((changed, episodes))
            changed = copy.deepcopy(atoms)
            changed[0]["outputs"][0]["generation_output_sha256"] = "0" * 64
            mutations.append((changed, episodes))
            changed = copy.deepcopy(episodes)
            changed[0]["turns"][0]["action"] = "wrong"
            mutations.append((atoms, changed))
            changed = copy.deepcopy(episodes)
            changed[0]["turns"][0]["observation"] = "wrong"
            mutations.append((atoms, changed))
            changed = copy.deepcopy(episodes)
            changed[0]["score"] = 0.0
            mutations.append((atoms, changed))
            for changed_atoms, changed_episodes in mutations:
                with self.assertRaises(ValueError):
                    _validate_call_journal(
                        payload,
                        generated,
                        atom_rows=changed_atoms,
                        episode_rows=changed_episodes,
                    )

            journal_mutations = []
            changed = copy.deepcopy(bundle_payloads)
            changed[0]["request_evidence"][0]["record_sha256"] = canonical_hash(
                {
                    "id": "a",
                    "messages": [{"role": "user", "content": "different"}],
                }
            )
            journal_mutations.append(changed)
            changed = copy.deepcopy(bundle_payloads)
            changed[0]["request_evidence"][0]["prompt_sha256"] = "0" * 64
            journal_mutations.append(changed)
            changed = copy.deepcopy(bundle_payloads)
            changed[0]["request_evidence"][0]["prompt_token_ids_sha256"] = (
                "0" * 64
            )
            journal_mutations.append(changed)
            changed = copy.deepcopy(bundle_payloads)
            changed[0]["rows"].reverse()
            changed[0]["request_evidence"].reverse()
            journal_mutations.append(changed)
            for mutation_index, changed_bundles in enumerate(journal_mutations):
                changed_descriptors = []
                for call_index, bundle in enumerate(changed_bundles):
                    path = (
                        Path(self.temporary.name)
                        / f"mutated_{mutation_index}_{call_index}.jsonl.gz"
                    )
                    with gzip.open(path, "wt", encoding="utf-8") as handle:
                        handle.write(json.dumps(bundle) + "\n")
                    changed_descriptors.append({"path": str(path)})
                with self.assertRaises(ValueError):
                    _validate_call_journal(
                        payload,
                        {"call_bundles": changed_descriptors},
                        atom_rows=atoms,
                        episode_rows=episodes,
                    )

    def test_control_authorization_and_global_admission_are_exactly_bound(self):
        root = Path(self.temporary.name) / "repo"
        experiment = root / "experiments" / "exp"
        score_root = experiment / "runs" / "confirmation"
        analysis = experiment / "analysis"
        analysis.mkdir(parents=True)
        files = []
        for relative, contents in (("scripts/a.py", "a\n"), ("src/b.py", "b\n")):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(contents, encoding="utf-8")
            files.append({"path": relative, "sha256": sha256_file(path)})
        inventory = {
            "files": files,
            "file_count": len(files),
            "sha256": canonical_hash(files),
        }
        arms = {
            name: {
                "model": f"/models/{'soup' if name == 'soup_best8' else name}",
                "model_merge_receipt_sha256": "a" * 64,
                "model_config_sha256": "b" * 64,
                "model_inference_inventory_sha256": "c" * 64,
                "decode": "sample8" if name == "soup_best8" else "greedy",
            }
            for name in CONFIRMATION_ARM_NAMES
        }
        arms["soup_best8"] = {**arms["soup"], "decode": "sample8"}
        authorization_path = analysis / "controls_authorization.json"
        authorization_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stage": "semantic_controls_confirmation_authorization",
                    "config_sha256": "c" * 64,
                    "control_code_inventory": inventory,
                    "control_code_inventory_sha256": canonical_hash(inventory),
                    "control_code_inventory_before_sha256": inventory["sha256"],
                    "control_code_inventory_after_sha256": inventory["sha256"],
                    "source_checkpoint_receipt": {
                        "path": str(SOURCE_RECEIPT),
                        "sha256": sha256_file(SOURCE_RECEIPT),
                        "commit": SOURCE_RECEIPT_COMMIT,
                    },
                    "confirmation_arms": arms,
                    "confirmation_arms_sha256": confirmation_arm_map_sha256(arms),
                    "gate": {"passed": True},
                    "downstream_authorization": "sealed_confirmation_evaluation",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        evaluator = experiment / "scripts" / "eval_policy.py"
        evaluator.parent.mkdir(parents=True)
        evaluator.write_text("# evaluator\n", encoding="utf-8")
        evaluator_source = {"sha256": "e" * 64, "file_count": 1}
        with mock.patch.multiple(
            "confirmation_artifacts",
            REPO=root.resolve(),
            EXP=experiment.resolve(),
            CONFIRMATION_SCORE_ROOT=score_root.resolve(),
            confirmation_evaluator_source_inventory=mock.Mock(
                return_value=evaluator_source
            ),
            control_code_inventory=mock.Mock(return_value=inventory),
        ):
            authorization = controls_authorization_binding(
                authorization_path, expected_config_sha256="c" * 64
            )
            admission_path = score_root / "ADMISSION.json"
            admission_path.parent.mkdir(parents=True)
            model = arms["quick"]
            admission_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage": "sealed_confirmation_admission",
                        "config_sha256": "c" * 64,
                        "controls_authorization": authorization,
                        "blocks": [98700],
                        "arms": arms,
                        "evaluator_sha256": sha256_file(evaluator),
                        "evaluator_source_inventory": evaluator_source,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            admission = confirmation_admission_binding(
                admission_path,
                expected_config_sha256="c" * 64,
                expected_controls_authorization=authorization,
                expected_tag="block_0_quick",
                expected_block_seed=98700,
                expected_model=model,
            )
            self.assertEqual(admission["controls_authorization_sha256"], authorization["sha256"])
            admission_payload = json.loads(admission_path.read_text())
            admission_payload["arms"]["deep"]["model"] = "/models/mutated"
            admission_path.write_text(
                json.dumps(admission_payload) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "global admission is stale"):
                confirmation_admission_binding(
                    admission_path,
                    expected_config_sha256="c" * 64,
                    expected_controls_authorization=authorization,
                )
            admission_payload["arms"] = arms
            admission_path.write_text(
                json.dumps(admission_payload) + "\n", encoding="utf-8"
            )
            authorization_payload = json.loads(authorization_path.read_text())
            authorization_payload["confirmation_arms"]["deep"]["model"] = (
                "/models/mutated"
            )
            authorization_path.write_text(
                json.dumps(authorization_payload) + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "arm|stale"):
                controls_authorization_binding(
                    authorization_path, expected_config_sha256="c" * 64
                )
            authorization_payload["confirmation_arms"] = arms
            authorization_path.write_text(
                json.dumps(authorization_payload) + "\n", encoding="utf-8"
            )
            (root / "scripts" / "a.py").write_text("changed\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inventory is stale"):
                controls_authorization_binding(
                    authorization_path, expected_config_sha256="c" * 64
                )

    def test_control_and_admission_seals_reject_leaf_or_ancestor_symlinks(self):
        root = Path(self.temporary.name) / "sealed-paths"
        experiment = root / "experiment"
        outside = root / "outside"
        outside.mkdir(parents=True)
        cases = (
            ("authorization leaf", "authorization", True),
            ("authorization ancestor", "authorization", False),
            ("admission leaf", "admission", True),
            ("admission ancestor", "admission", False),
        )
        for label, kind, leaf in cases:
            with self.subTest(label=label):
                case = root / label.replace(" ", "_")
                case.mkdir()
                exp = case / "experiment"
                score_root = exp / "runs" / "confirmation"
                if kind == "authorization":
                    canonical = exp / "analysis" / "controls_authorization.json"
                else:
                    canonical = score_root / "ADMISSION.json"
                target_dir = case / "target"
                target_dir.mkdir()
                if leaf:
                    canonical.parent.mkdir(parents=True)
                    target = target_dir / canonical.name
                    target.write_text("{}\n", encoding="utf-8")
                    canonical.symlink_to(target)
                else:
                    ancestor = canonical.parent
                    ancestor.parent.mkdir(parents=True)
                    mirrored = target_dir / canonical.name
                    mirrored.write_text("{}\n", encoding="utf-8")
                    ancestor.symlink_to(target_dir, target_is_directory=True)
                with mock.patch.multiple(
                    "confirmation_artifacts",
                    EXP=exp,
                    CONFIRMATION_SCORE_ROOT=score_root,
                ):
                    with self.assertRaisesRegex(ValueError, "symlink"):
                        if kind == "authorization":
                            controls_authorization_binding(
                                canonical, expected_config_sha256="c" * 64
                            )
                        else:
                            confirmation_admission_binding(
                                canonical,
                                expected_config_sha256="c" * 64,
                                expected_controls_authorization={},
                            )

    def test_campaign_tree_is_exhaustive_and_allows_only_registered_partials(self):
        admission_path = self.score_root / "ADMISSION.json"
        admission_path.parent.mkdir(parents=True)
        admission_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "stage": "sealed_confirmation_admission",
                    "blocks": [17],
                    "arms": {"arm": {"model": "m"}},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=False,
            ),
            {},
        )
        prepare_confirmation_output(
            self.score_path, score_root=self.score_root, raw_root=self.raw_root
        )
        self.assertEqual(
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=False,
            ),
            {"block_0/arm": "EMPTY"},
        )
        self._commit()
        (self.score_root / "manifest.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=True,
                require_manifest=True,
            ),
            {"block_0/arm": "COMMITTED"},
        )
        extra = self.score_root / "unregistered.txt"
        extra.write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unregistered entries"):
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=True,
                require_manifest=True,
            )
        extra.unlink()
        extra_block = self.raw_root / "block_9"
        extra_block.mkdir()
        with self.assertRaisesRegex(ValueError, "visible admitted blocks"):
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=True,
                require_manifest=True,
            )
        extra_block.rmdir()
        extra_arm = self.score_root / "block_0" / "unregistered"
        extra_arm.mkdir()
        with self.assertRaisesRegex(ValueError, "unregistered arms"):
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=True,
                require_manifest=True,
            )
        extra_arm.rmdir()
        rogue_bundle = self.raw_root / "block_0" / "arm" / "bundle_9999.jsonl.gz"
        rogue_bundle.write_bytes(b"rogue")
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            validate_confirmation_campaign_tree(
                admission_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
                terminal=True,
                require_manifest=True,
            )

    def test_score_path_wrong_depth_is_rejected(self):
        shallow = self.score_root / "arm" / "scores.json"
        with self.assertRaisesRegex(ValueError, "block/arm"):
            prepare_confirmation_output(
                shallow,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )

    def test_policy_tasks_must_exactly_regenerate_preregistered_block(self):
        experiment = Path(self.temporary.name) / "task-regeneration-experiment"
        config_path = experiment / "configs" / "default.yaml"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("frozen: true\n", encoding="utf-8")
        config = {
            "seeds": {"confirmatory_blocks": [700]},
            "confirmation": {
                "atoms_per_family_level": 2,
                "episodes_per_family_level": 2,
            },
            "controls": {"sample_more_k": 8},
            "strata": {
                "quick_atom_levels": [1],
                "deep_atom_levels": [2],
                "deep_episode_levels": [3],
            },
        }

        class FakeEpisode:
            def __init__(self, seed: int, level: int):
                self.spec = {"seed": seed, "level": level, "bytes": "exact"}
                self.seed = seed
                self.level = level

            def system_prompt(self) -> str:
                return f"system:{self.seed}:{self.level}"

            def initial_observation(self) -> str:
                return f"initial:{self.seed}:{self.level}"

        def gen_atoms(seed: int, level: int, n: int) -> list[dict]:
            return [
                {
                    "id": f"fake-L{level}-s{seed}-i{index}",
                    "family": "fake",
                    "level": level,
                    "prompt": f"prompt:{seed}:{level}:{index}",
                    "gold": {"answer": index},
                    "answer_domain": [index, index + 1],
                }
                for index in range(n)
            ]

        family = SimpleNamespace(
            LEVELS={1, 2, 3},
            HAS_EPISODES=True,
            Episode=FakeEpisode,
            gen_atoms=gen_atoms,
        )
        payload = {
            "stage": "policy_eval",
            "tag": "block_0_arm",
            "scope": "confirmatory",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "block_seed": 700,
            "families": ["fake"],
            "atoms_per_level": 2,
            "episodes_per_level": 2,
            "decode": "greedy",
            "k": 1,
        }
        atoms = [
            item
            for level in (1, 2)
            for item in gen_atoms(700 + level * 1_000, level, 2)
        ]
        episodes = []
        for index in range(2):
            seed = 700 + 50_000_000 + 3_000 + index
            episode = FakeEpisode(seed, 3)
            episodes.append(
                {
                    "rid": f"fake-L3-e{seed}-r0",
                    "family": "fake",
                    "level": 3,
                    "ep_seed": seed,
                    "rollout": 0,
                    "spec": episode.spec,
                    "system_prompt": episode.system_prompt(),
                    "initial_observation": episode.initial_observation(),
                }
            )

        patches = (
            mock.patch("confirmation_artifacts.EXP", experiment),
            mock.patch(
                "confirmation_artifacts.load_config",
                return_value=(config, config_path),
            ),
            mock.patch("confirmation_artifacts.all_families", return_value=["fake"]),
            mock.patch("confirmation_artifacts.load_family", return_value=family),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            _validate_preregistered_confirmation_tasks(
                payload, atom_rows=atoms, episode_rows=episodes
            )
            mutations = []
            stale = copy.deepcopy(payload)
            stale["config_sha256"] = "0" * 64
            mutations.append(("stale config", stale, atoms, episodes))
            wrong_seed = copy.deepcopy(payload)
            wrong_seed["block_seed"] = 701
            mutations.append(("wrong seed", wrong_seed, atoms, episodes))
            substituted_atom = copy.deepcopy(atoms)
            substituted_atom[0]["prompt"] += ":substituted"
            mutations.append(("substituted atom", payload, substituted_atom, episodes))
            atom_bool_for_int = copy.deepcopy(atoms)
            atom_bool_for_int[0]["gold"]["answer"] = False
            mutations.append(
                ("atom bool for int", payload, atom_bool_for_int, episodes)
            )
            reordered_atoms = list(reversed(copy.deepcopy(atoms)))
            mutations.append(("atom order", payload, reordered_atoms, episodes))
            substituted_episode = copy.deepcopy(episodes)
            substituted_episode[0]["spec"]["bytes"] = "substituted"
            mutations.append(
                ("substituted episode", payload, atoms, substituted_episode)
            )
            episode_float_for_int = copy.deepcopy(episodes)
            episode_float_for_int[0]["spec"]["level"] = 3.0
            mutations.append(
                ("episode float for int", payload, atoms, episode_float_for_int)
            )
            reordered_episodes = list(reversed(copy.deepcopy(episodes)))
            mutations.append(("episode order", payload, atoms, reordered_episodes))
            wrong_episode_seed = copy.deepcopy(episodes)
            wrong_episode_seed[0]["ep_seed"] += 1
            mutations.append(
                ("episode seed", payload, atoms, wrong_episode_seed)
            )
            for label, candidate, atom_rows, episode_rows in mutations:
                with self.subTest(label=label), self.assertRaises(ValueError):
                    _validate_preregistered_confirmation_tasks(
                        candidate,
                        atom_rows=atom_rows,
                        episode_rows=episode_rows,
                    )

            duplicate_rollouts = [
                copy.deepcopy(episodes[0]),
                copy.deepcopy(episodes[0]),
            ]
            duplicate_rollouts[1]["rollout"] = 1
            duplicate_rollouts[1]["rid"] = duplicate_rollouts[1]["rid"].replace(
                "-r0", "-r1"
            )
            confirmation_task_hashes([], duplicate_rollouts)
            duplicate_rollouts[1]["spec"]["level"] = 3.0
            with self.assertRaisesRegex(ValueError, "exact task bytes"):
                confirmation_task_hashes([], duplicate_rollouts)

    def test_mirrored_path_cannot_be_redirected_by_symlink(self):
        self.raw_root.mkdir()
        redirected = self.raw_root / "redirected"
        redirected.mkdir()
        (self.raw_root / "block_0").symlink_to(redirected, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            prepare_confirmation_output(
                self.score_path,
                score_root=self.score_root,
                raw_root=self.raw_root,
            )


if __name__ == "__main__":
    unittest.main()
