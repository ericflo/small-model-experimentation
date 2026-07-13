from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import run as run_script  # noqa: E402

from control_rematch import (  # noqa: E402
    CACHE_VARIANT,
    REMATCH_STAGE,
    prompt_truncation_summary,
    rematch_full_prefix_controls,
    samples_semantic_sha256,
    validate_control_overlay_cache,
    validate_control_rematch_manifest,
)
from io_utils import canonical_hash, sha256_file  # noqa: E402
from route_control_matching import (  # noqa: E402
    MATCH_TIERS,
    matched_non_advantage_route_units,
)


def source(state_id: str, *, family="loomfix", kind="episode", level=5) -> dict:
    return {
        "state_id": state_id,
        "family": family,
        "kind": kind,
        "level": level,
    }


def candidate(
    state_id: str,
    *,
    family="loomfix",
    kind="episode",
    level=5,
    teacher=None,
) -> dict:
    return {
        "state_id": state_id,
        "family": family,
        "kind": kind,
        "level": level,
        "primary_teacher": teacher,
    }


def sparse_sample(
    state_id: str,
    role: str,
    *,
    prompt_tokens_truncated: int = 0,
    matched_primary_state_id: str | None = None,
    match_tier: str | None = None,
) -> dict:
    targets = {
        "soup": {
            "indices": torch.tensor([[1]], dtype=torch.int32),
            "log_probs": torch.tensor([[-0.1]], dtype=torch.float32),
        }
    }
    if role != "anchor":
        for policy, index in (("quick", 2), ("deep", 3)):
            targets[policy] = {
                "indices": torch.tensor([[index]], dtype=torch.int32),
                "log_probs": torch.tensor([[-0.2]], dtype=torch.float32),
            }
    return {
        "id": state_id,
        "meta": {
            "family": "loomfix",
            "kind": "episode",
            "level": 5,
            "role": role,
            "primary_teacher": "soup" if role == "anchor" else "deep",
            "observed_route": "abstain" if role == "route_control" else None,
            "matched_primary_state_id": matched_primary_state_id,
            "match_tier": match_tier,
            "original_prompt_tokens": 2 + prompt_tokens_truncated,
            "prompt_tokens_truncated": prompt_tokens_truncated,
            "input_tokens": 3,
        },
        "prompt_ids": torch.tensor([10, 11], dtype=torch.int32),
        "completion_ids": torch.tensor([12], dtype=torch.int32),
        "positions": torch.tensor([0], dtype=torch.int32),
        "targets": targets,
    }


def policy_models(root: Path) -> tuple[dict, dict]:
    paths = {
        "quick": root / "quick",
        "deep": root / "deep",
        "soup": root / "soup",
    }
    for path in paths.values():
        path.mkdir()
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
    config = {
        "model": {
            "quick_teacher": str(paths["quick"]),
            "deep_teacher": str(paths["deep"]),
            "student_checkpoint": str(paths["soup"]),
        },
        "mopd": {
            "top_k": 50,
            "updates_per_round": 2,
            "grad_accum": 1,
            "capability_units_per_round": 1,
            "anchor_units_per_round": 1,
            "max_length": 3072,
        },
    }
    models = {
        policy: {
            "path": str(path.resolve()),
            "config_sha256": sha256_file(path / "config.json"),
            "merge_receipt_sha256": sha256_file(path / "merge_receipt.json"),
        }
        for policy, path in paths.items()
    }
    return config, models


def serialized_overlay_fixture(root: Path) -> dict:
    config_path = root / "config.yaml"
    config_path.write_text("fixture\n", encoding="utf-8")
    config, models = policy_models(root)
    source_cache = root / "source.pt"
    source_samples = [
        sparse_sample("capability", "capability"),
        sparse_sample("anchor", "anchor"),
        sparse_sample(
            "old-control",
            "route_control",
            prompt_tokens_truncated=7,
            matched_primary_state_id="capability",
            match_tier="exact_cell",
        ),
    ]
    source_payload = {
        "schema_version": 1,
        "stage": "matched_all_policy_topk_cache",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "round_manifest": str(root / "round.json"),
        "round_manifest_sha256": "a" * 64,
        "round": 1,
        "top_k": 50,
        "models": models,
        "sample_count": len(source_samples),
        "active_positions": 3,
        "prompt_truncation": prompt_truncation_summary(source_samples),
        "ledgers": {},
        "samples": source_samples,
    }
    torch.save(source_payload, source_cache)
    source_receipt = {
        key: value for key, value in source_payload.items() if key != "samples"
    }
    source_receipt.update(
        {
            "cache_sha256": sha256_file(source_cache),
            "cache_bytes": source_cache.stat().st_size,
        }
    )
    source_receipt_path = source_cache.with_suffix(".pt.receipt.json")
    source_receipt_path.write_text(json.dumps(source_receipt), encoding="utf-8")

    derived_samples = copy.deepcopy(source_samples)
    derived_samples[2] = sparse_sample(
        "new-control",
        "route_control",
        matched_primary_state_id="capability",
        match_tier="exact_cell",
    )
    rematch = root / "rematch.json"
    rematch_payload = {
        "replacement_count": 1,
        "source": {
            "round_manifest": {"sha256": "a" * 64},
            "samples_semantic_sha256": samples_semantic_sha256(source_samples),
        },
        "replacements": [
            {
                "matched_primary_state_id": "capability",
                "source_cache_index": 2,
                "old_state_id": "old-control",
                "new_state_id": "new-control",
            }
        ],
        "rematched_control_mapping": [
            {
                "state_id": "new-control",
                "matched_primary_state_id": "capability",
                "match_tier": "exact_cell",
            }
        ],
    }
    rematch.write_text(json.dumps(rematch_payload), encoding="utf-8")
    copied = derived_samples[:2]
    replacements = [derived_samples[2]]
    derived_payload = copy.deepcopy(source_payload)
    derived_payload.update(
        {
            "cache_variant": CACHE_VARIANT,
            "source_target_cache": str(source_cache.resolve()),
            "source_target_cache_sha256": sha256_file(source_cache),
            "source_target_cache_receipt_sha256": sha256_file(source_receipt_path),
            "control_rematch_manifest": str(rematch.resolve()),
            "control_rematch_manifest_sha256": sha256_file(rematch),
            "replacement_count": 1,
            "copied_sample_count": 2,
            "copied_samples_semantic_sha256": samples_semantic_sha256(copied),
            "replacement_samples_semantic_sha256": samples_semantic_sha256(
                replacements
            ),
            "replacement_state_ids_sha256": hashlib.sha256(
                b"new-control"
            ).hexdigest(),
            "sample_count": 3,
            "active_positions": 3,
            "prompt_truncation": prompt_truncation_summary(derived_samples),
            "ledgers": {"source_cache": {}, "copied_sample_count": 2},
            "samples": derived_samples,
        }
    )
    cache = root / "derived.pt"
    torch.save(derived_payload, cache)
    receipt = {
        key: value for key, value in derived_payload.items() if key != "samples"
    }
    receipt.update(
        {"cache_sha256": sha256_file(cache), "cache_bytes": cache.stat().st_size}
    )
    receipt_path = cache.with_suffix(".pt.receipt.json")
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return {
        "config": config,
        "config_path": config_path,
        "source_cache": source_cache,
        "source_receipt": source_receipt_path,
        "rematch": rematch,
        "cache": cache,
        "receipt": receipt_path,
    }


class ControlRematchTests(unittest.TestCase):
    def test_only_non_advantage_arm_uses_overlay(self):
        source_cache = Path("/tmp/source-cache.pt")
        derived_cache = Path("/tmp/derived-cache.pt")
        rematch = Path("/tmp/rematch.json")
        common = {
            "config": {},
            "config_path": Path("/tmp/config.yaml"),
            "data_root": Path("/tmp/data"),
            "manifest": Path("/tmp/round.json"),
            "source_cache": source_cache,
        }
        with patch.object(
            run_script,
            "_prepare_non_advantage_route_cache",
            return_value=(derived_cache, rematch),
        ) as prepare:
            for arm in (
                "primary",
                "wrong_teacher",
                "offpolicy_sft",
                "soup50",
            ):
                self.assertEqual(
                    run_script._effective_control_target_cache(arm, **common),
                    (source_cache, None),
                )
            prepare.assert_not_called()
            self.assertEqual(
                run_script._effective_control_target_cache(
                    "non_advantage_route", **common
                ),
                (derived_cache, rematch),
            )
            prepare.assert_called_once()

    def test_noop_overlay_reuses_source_and_rejects_unexpected_derived_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            overlay_root = data_root / "control_overlays" / "non_advantage_route"
            overlay_root.mkdir(parents=True)
            rematch = overlay_root / "rematch_manifest.json"
            rematch.write_text("{}\n", encoding="utf-8")
            source_cache = root / "source.pt"
            source_cache.write_text("source", encoding="utf-8")
            with patch.object(
                run_script,
                "validate_control_rematch_manifest",
                return_value={"replacement_count": 0},
            ), patch.object(
                run_script,
                "_paths",
                return_value={"quick": root, "deep": root, "soup": root},
            ):
                self.assertEqual(
                    run_script._prepare_non_advantage_route_cache(
                        {},
                        root / "config.yaml",
                        data_root=data_root,
                        manifest=root / "round.json",
                        source_cache=source_cache,
                    ),
                    (source_cache, rematch),
                )
                (overlay_root / "all_policy_targets.pt").write_text(
                    "unexpected", encoding="utf-8"
                )
                with self.assertRaisesRegex(SystemExit, "unexpected derived cache"):
                    run_script._prepare_non_advantage_route_cache(
                        {},
                        root / "config.yaml",
                        data_root=data_root,
                        manifest=root / "round.json",
                        source_cache=source_cache,
                    )

    def test_partial_overlay_directory_fails_before_builder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_root = root / "data"
            overlay_root = data_root / "control_overlays" / "non_advantage_route"
            overlay_root.mkdir(parents=True)
            (overlay_root / "partial.tmp").write_text("partial", encoding="utf-8")
            with patch.object(run_script, "_run") as runner, patch.object(
                run_script,
                "_paths",
                return_value={"quick": root, "deep": root, "soup": root},
            ):
                with self.assertRaisesRegex(SystemExit, "partial non-advantage"):
                    run_script._prepare_non_advantage_route_cache(
                        {},
                        root / "config.yaml",
                        data_root=data_root,
                        manifest=root / "round.json",
                        source_cache=root / "source.pt",
                    )
            runner.assert_not_called()

    def test_length_filter_replaces_only_offender_in_registered_order(self):
        primaries = [source("primary-a"), source("primary-b", family="patchwheel")]
        candidates = [
            candidate("control-a0"),
            candidate("control-a1"),
            candidate("control-b0", family="patchwheel"),
            candidate("control-b1", family="patchwheel"),
        ]
        original = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        rematched, audit = rematch_full_prefix_controls(
            capability_units=primaries,
            original_controls=original,
            candidates=candidates,
            eligible_state_ids={"control-a1", "control-b0", "control-b1"},
            match_order=MATCH_TIERS,
        )
        by_primary = {row["matched_primary_state_id"]: row for row in rematched}
        self.assertEqual(by_primary["primary-a"]["state_id"], "control-a1")
        self.assertEqual(by_primary["primary-b"]["state_id"], "control-b0")
        self.assertEqual(audit["changed_primary_state_ids"], ["primary-a"])
        self.assertEqual(audit["unchanged_control_count"], 1)
        self.assertEqual(
            audit["original_match_tier_counts"], audit["match_tier_counts"]
        )

    def test_rematch_rejects_looser_tier_for_offender(self):
        primaries = [source("primary")]
        candidates = [
            candidate("control-exact"),
            candidate("control-family-kind", level=4),
        ]
        original = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        with self.assertRaisesRegex(ValueError, "changed offender match tiers"):
            rematch_full_prefix_controls(
                capability_units=primaries,
                original_controls=original,
                candidates=candidates,
                eligible_state_ids={"control-family-kind"},
                match_order=MATCH_TIERS,
            )

    def test_noop_rematch_preserves_exact_mapping_and_geometry(self):
        primaries = [source("primary")]
        candidates = [candidate("control")]
        original = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        rematched, audit = rematch_full_prefix_controls(
            capability_units=primaries,
            original_controls=original,
            candidates=candidates,
            eligible_state_ids={"control"},
            match_order=MATCH_TIERS,
        )
        self.assertEqual(rematched, original)
        self.assertEqual(audit["changed_primary_state_ids"], [])
        self.assertEqual(audit["original_match_tier_counts"], audit["match_tier_counts"])

    def test_match_tier_precedes_candidate_lexicographic_order(self):
        primaries = [source("primary")]
        candidates = [
            candidate("a-family-kind", level=4),
            candidate("z-exact"),
        ]
        matched = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        self.assertEqual(matched[0]["state_id"], "z-exact")
        self.assertEqual(matched[0]["match_tier"], "exact_cell")

    def test_filter_cascade_into_nonoffender_fails_closed(self):
        primaries = [source("primary-a"), source("primary-b")]
        candidates = [
            candidate("control-a"),
            candidate("control-b"),
            candidate("control-c"),
        ]
        original = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        with self.assertRaisesRegex(ValueError, "changed nonoffending"):
            rematch_full_prefix_controls(
                capability_units=primaries,
                original_controls=original,
                candidates=candidates,
                eligible_state_ids={"control-b", "control-c"},
                match_order=MATCH_TIERS,
            )

    def test_insufficient_full_prefix_supply_fails(self):
        primaries = [source("primary")]
        candidates = [candidate("control")]
        original = matched_non_advantage_route_units(
            primaries, candidates, MATCH_TIERS
        )
        with self.assertRaisesRegex(ValueError, "no kind-preserving"):
            rematch_full_prefix_controls(
                capability_units=primaries,
                original_controls=original,
                candidates=candidates,
                eligible_state_ids=set(),
                match_order=MATCH_TIERS,
            )

    def test_sample_semantic_hash_detects_tensor_change(self):
        samples = [
            {
                "id": "sample",
                "meta": {"role": "route_control"},
                "prompt_ids": torch.tensor([1, 2], dtype=torch.int32),
                "targets": {
                    "deep": {"log_probs": torch.tensor([-1.0], dtype=torch.float32)}
                },
            }
        ]
        before = samples_semantic_sha256(samples)
        copied = [
            {
                **samples[0],
                "prompt_ids": samples[0]["prompt_ids"].clone(),
                "targets": {
                    "deep": {
                        "log_probs": samples[0]["targets"]["deep"][
                            "log_probs"
                        ].clone()
                    }
                },
            }
        ]
        self.assertEqual(samples_semantic_sha256(copied), before)
        copied[0]["prompt_ids"][1] = 3
        self.assertNotEqual(samples_semantic_sha256(copied), before)

    def test_prompt_truncation_summary_is_role_bound(self):
        summary = prompt_truncation_summary(
            [
                {
                    "id": "control-cut",
                    "meta": {"role": "route_control", "prompt_tokens_truncated": 7},
                },
                {
                    "id": "anchor-full",
                    "meta": {"role": "anchor", "prompt_tokens_truncated": 0},
                },
            ]
        )
        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["by_role"]["route_control"]["total_tokens"], 7)
        self.assertEqual(summary["by_role"]["anchor"]["sample_count"], 0)

    def test_rematch_manifest_revalidates_transitive_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            source_manifest = root / "round.json"
            source_cache = root / "targets.pt"
            source_receipt = root / "targets.pt.receipt.json"
            tokenizer_files = [
                root / "tokenizer.json",
                root / "tokenizer_config.json",
                root / "chat_template.jinja",
            ]
            for path, text in (
                (config_path, "config"),
                (source_receipt, "receipt"),
            ):
                path.write_text(text, encoding="utf-8")
            for path in tokenizer_files:
                path.write_text(path.name, encoding="utf-8")
            artifact_paths = {
                role: root / f"{role.replace('.', '_')}.jsonl"
                for role in (
                    "states", "anchors", "branches.quick", "branches.deep",
                    "branches.student",
                )
            }
            for role, path in artifact_paths.items():
                path.write_text(role, encoding="utf-8")
            source_artifacts = {
                "batch": 0,
                "states": {
                    "path": str(artifact_paths["states"]),
                    "sha256": sha256_file(artifact_paths["states"]),
                },
                "anchors": {
                    "path": str(artifact_paths["anchors"]),
                    "sha256": sha256_file(artifact_paths["anchors"]),
                },
                "branches": {
                    policy: {
                        "path": str(artifact_paths[f"branches.{policy}"]),
                        "sha256": sha256_file(artifact_paths[f"branches.{policy}"]),
                    }
                    for policy in ("quick", "deep", "student")
                },
            }
            config = {
                "model": {"student_checkpoint": str(root)},
                "mopd": {"max_length": 3072, "capability_units_per_round": 2},
            }
            original_mapping = [
                {
                    "state_id": "old",
                    "matched_primary_state_id": "primary",
                    "match_tier": "exact_cell",
                    "observed_route": "abstain",
                    "family": "loomfix",
                    "kind": "episode",
                    "level": 5,
                },
                {
                    "state_id": "same",
                    "matched_primary_state_id": "primary-2",
                    "match_tier": "exact_cell",
                    "observed_route": "abstain",
                    "family": "patchwheel",
                    "kind": "episode",
                    "level": 5,
                },
            ]
            rematched_mapping = [{**original_mapping[0], "state_id": "new"}, original_mapping[1]]
            source_samples = [
                sparse_sample(
                    "old",
                    "route_control",
                    prompt_tokens_truncated=7,
                    matched_primary_state_id="primary",
                    match_tier="exact_cell",
                ),
                sparse_sample(
                    "same",
                    "route_control",
                    matched_primary_state_id="primary-2",
                    match_tier="exact_cell",
                ),
            ]
            torch.save({"samples": source_samples}, source_cache)
            source_manifest.write_text(
                json.dumps(
                    {
                        "stage": "online_advantage_training_round",
                        "config_sha256": sha256_file(config_path),
                        "round": 1,
                        "artifacts": [source_artifacts],
                        "candidate_counts": {"quick_routed": 1, "abstained": 2},
                        "control_units": original_mapping,
                    }
                ),
                encoding="utf-8",
            )
            candidate_records = [
                {
                    "batch": 0,
                    "role": role,
                    "path": str(path.resolve()),
                    "sha256": sha256_file(path),
                }
                for role, path in artifact_paths.items()
            ]
            candidate_records.sort(key=lambda row: (row["batch"], row["role"]))
            payload = {
                "schema_version": 1,
                "stage": REMATCH_STAGE,
                "config_sha256": sha256_file(config_path),
                "round": 1,
                "source": {
                    "round_manifest": {
                        "path": str(source_manifest),
                        "sha256": sha256_file(source_manifest),
                    },
                    "target_cache": {
                        "path": str(source_cache),
                        "sha256": sha256_file(source_cache),
                    },
                    "target_cache_receipt": {
                        "path": str(source_receipt),
                        "sha256": sha256_file(source_receipt),
                    },
                    "samples_semantic_sha256": samples_semantic_sha256(
                        source_samples
                    ),
                },
                "match_order": list(MATCH_TIERS),
                "max_length": 3072,
                "audit": {
                    "offender_primary_state_ids": ["primary"],
                    "changed_primary_state_ids": ["primary"],
                    "original_control_count": 2,
                    "rematched_control_count": 2,
                    "candidate_count": 3,
                    "eligible_candidate_count": 2,
                    "ineligible_candidate_count": 1,
                    "unchanged_control_count": 1,
                    "original_match_tier_counts": {
                        "exact_cell": 2,
                        "family_kind": 0,
                        "kind_level": 0,
                        "kind": 0,
                    },
                    "match_tier_counts": {
                        "exact_cell": 2,
                        "family_kind": 0,
                        "kind_level": 0,
                        "kind": 0,
                    },
                    "original_mapping_sha256": canonical_hash(original_mapping),
                    "rematched_mapping_sha256": canonical_hash(rematched_mapping),
                },
                "replacement_count": 1,
                "original_control_mapping": original_mapping,
                "rematched_control_mapping": rematched_mapping,
                "replacements": [
                    {
                        "matched_primary_state_id": "primary",
                        "old_state_id": "old",
                        "new_state_id": "new",
                        "old_match_tier": "exact_cell",
                        "new_match_tier": "exact_cell",
                        "old_fit": {"prompt_tokens_truncated": 7},
                        "new_fit": {"prompt_tokens_truncated": 0},
                    }
                ],
                "candidate_artifacts": candidate_records,
                "tokenizer": {
                    "path": str(root),
                    "files": [
                        {"path": str(path), "sha256": sha256_file(path)}
                        for path in tokenizer_files
                    ]
                },
                "gate": {"passed": True},
            }
            rematch = root / "rematch.json"
            rematch.write_text(json.dumps(payload), encoding="utf-8")
            validate_control_rematch_manifest(
                rematch,
                config=config,
                config_path=config_path,
                source_manifest=source_manifest,
                source_cache=source_cache,
            )
            corrupted = json.loads(rematch.read_text(encoding="utf-8"))
            corrupted["original_control_mapping"][0]["family"] = "forged"
            corrupted["audit"]["original_mapping_sha256"] = canonical_hash(
                corrupted["original_control_mapping"]
            )
            rematch.write_text(json.dumps(corrupted), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "original_control_mapping.source_manifest"
            ):
                validate_control_rematch_manifest(
                    rematch,
                    config=config,
                    config_path=config_path,
                    source_manifest=source_manifest,
                    source_cache=source_cache,
                )
            rematch.write_text(json.dumps(payload), encoding="utf-8")
            artifact_paths["states"].write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "candidate_artifacts"):
                validate_control_rematch_manifest(
                    rematch,
                    config=config,
                    config_path=config_path,
                    source_manifest=source_manifest,
                    source_cache=source_cache,
                )

    def test_overlay_cache_recomputes_real_serialized_invariants(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = serialized_overlay_fixture(root)
            validate_control_overlay_cache(
                fixture["cache"],
                rematch_manifest=fixture["rematch"],
                source_cache=fixture["source_cache"],
                config=fixture["config"],
                config_path=fixture["config_path"],
            )

            payload = torch.load(
                fixture["cache"], map_location="cpu", weights_only=False
            )
            payload["samples"][0]["prompt_ids"][0] = 999
            torch.save(payload, fixture["cache"])
            receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
            receipt["cache_sha256"] = sha256_file(fixture["cache"])
            receipt["cache_bytes"] = fixture["cache"].stat().st_size
            fixture["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "copied_samples_semantic_sha256"
            ):
                validate_control_overlay_cache(
                    fixture["cache"],
                    rematch_manifest=fixture["rematch"],
                    source_cache=fixture["source_cache"],
                    config=fixture["config"],
                    config_path=fixture["config_path"],
                )

    def test_overlay_cache_revalidates_source_hash_and_partial_pair(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = serialized_overlay_fixture(Path(temporary))
            fixture["source_cache"].write_bytes(
                fixture["source_cache"].read_bytes() + b"changed"
            )
            with self.assertRaisesRegex(ValueError, "source_target_cache_sha256"):
                validate_control_overlay_cache(
                    fixture["cache"],
                    rematch_manifest=fixture["rematch"],
                    source_cache=fixture["source_cache"],
                    config=fixture["config"],
                    config_path=fixture["config_path"],
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = serialized_overlay_fixture(Path(temporary))
            fixture["receipt"].unlink()
            with self.assertRaisesRegex(ValueError, "partial control overlay"):
                validate_control_overlay_cache(
                    fixture["cache"],
                    rematch_manifest=fixture["rematch"],
                    source_cache=fixture["source_cache"],
                    config=fixture["config"],
                    config_path=fixture["config_path"],
                )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = serialized_overlay_fixture(Path(temporary))
            fixture["cache"].write_text("not a torch payload", encoding="utf-8")
            receipt = json.loads(fixture["receipt"].read_text(encoding="utf-8"))
            receipt["cache_sha256"] = sha256_file(fixture["cache"])
            receipt["cache_bytes"] = fixture["cache"].stat().st_size
            fixture["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "serialized_payload"):
                validate_control_overlay_cache(
                    fixture["cache"],
                    rematch_manifest=fixture["rematch"],
                    source_cache=fixture["source_cache"],
                    config=fixture["config"],
                    config_path=fixture["config_path"],
                )


if __name__ == "__main__":
    unittest.main()
