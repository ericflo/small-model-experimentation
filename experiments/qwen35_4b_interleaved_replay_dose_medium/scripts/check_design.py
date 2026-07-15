#!/usr/bin/env python3
"""Recompute the model-free interleaved-replay dose design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_v2 as axis  # noqa: E402
from check_local import ANSWER_NORMALIZATION  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "cbc9ae6d132d09b9bac2eb43010c2f3eb051993493d6ea659950ac07f0a1e903"
TREATMENT_PATH = EXP / "data" / "sft_hygiene_explore.jsonl"
TREATMENT_SHA256 = "8b3e97919c62cbb0893add281dc1d3ae881aa0138d0d1721043fec26b0c22cf1"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77119
TREATMENT_MIX = "hygiene=40,explore=40"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_hygiene_explore_destack_medium"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_clean"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_clean.json"
PARENT_RECEIPT_SHA256 = "24367084da5415ec8c3f922202a2028cc2930c4b99d87ea5e32b93e5c3b90332"
PARENT_EXTERNAL_RECEIPT_SHA256 = "aac945fb0d38f157d9c35518fb4c5de99cc5cac0ace3a280202f9dc038d77186"
PARENT_TREE_SHA256 = "19759e12b1301a15a2f9b2db311ffff7c08e3d8b0d3237c9e7cd718fa8dc7f67"
PARENT_WEIGHTS_SHA256 = "2cef3e5e7ddfbee5d2d2c3a878d64f360dc6994764b9856be7e319ce5187d0b4"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
WARM_START_ADAPTER = (
    ROOT / "large_artifacts" / PARENT_EXP.name / "adapters" / "replay_clean"
)
WARM_START_CONFIG_SHA256 = "015bb13568c411c94d24460a9007e1f0d8fe3eb6c9749ad938958490de84d961"
WARM_START_WEIGHTS_SHA256 = "f6f910ed1c1dcc843f43e09a562556b8e76ee40096aa7123fd70800d94fc6bb8"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_interleaved2", "dose_after_replay")
CANDIDATE_ARMS = ("dose_after_replay",)
PARENT_EVAL_LABEL = "interleaved_parent"
AXIS_KINDS = ("u_explore", "u_hygiene")
EXPECTED_KIND_COUNTS = {
    "u_explore": 40,
    "u_hygiene": 40,
}
PUBLIC_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites",
    "siftstack", "sirens", "stockade", "toolsmith", "warren",
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen_corpora() -> dict:
    for path, expected in (
        (MANIFEST, MANIFEST_SHA256),
        (TREATMENT_PATH, TREATMENT_SHA256),
        (REPLAY_PATH, REPLAY_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen corpus artifact is absent or changed: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    replay_rows = len(REPLAY_PATH.read_text(encoding="utf-8").splitlines())
    donor_treatment = PARENT_EXP / "data" / "sft_hygiene_explore.jsonl"
    donor_replay = PARENT_EXP / "data" / "sft_blend.jsonl"
    donor_manifest = PARENT_EXP / "data" / "corpus_manifest.json"
    if (
        # The manifest is inherited byte-identically from the donor, so its
        # experiment_id names the donor; the inheritance itself is what this
        # check authenticates.
        manifest.get("experiment_id") != PARENT_EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("mix") != TREATMENT_MIX
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 80
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != EXPECTED_KIND_COUNTS
        or manifest.get("replay", {}).get("sha256") != REPLAY_SHA256
        or replay_rows != REPLAY_ROWS
        or not donor_treatment.is_file()
        or sha256_file(donor_treatment) != TREATMENT_SHA256
        or not donor_replay.is_file()
        or sha256_file(donor_replay) != REPLAY_SHA256
        or not donor_manifest.is_file()
        or sha256_file(donor_manifest) != MANIFEST_SHA256
    ):
        raise ValueError("corpus manifest violates the frozen inheritance contract")
    return manifest


def check_parent_identity() -> None:
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    external = PARENT / "merge_receipt.json"
    weight = PARENT / "model.safetensors"
    warm_config = WARM_START_ADAPTER / "adapter_config.json"
    warm_weights = WARM_START_ADAPTER / "adapter_model.safetensors"
    if (
        sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or parent_receipt.get("name") != "replay_clean"
        or parent_receipt.get("merge_receipt_sha256") != PARENT_EXTERNAL_RECEIPT_SHA256
        or parent_receipt.get("output_tree_sha256") != PARENT_TREE_SHA256
        or {row.get("name"): row.get("sha256") for row in parent_receipt.get("weight_files", [])}
        != {"model.safetensors": PARENT_WEIGHTS_SHA256}
        or parent_receipt.get("adapter_config_sha256") != WARM_START_CONFIG_SHA256
        or parent_receipt.get("adapter_weights_sha256") != WARM_START_WEIGHTS_SHA256
        or not external.is_file()
        or sha256_file(external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or not weight.is_file()
        or weight.stat().st_size != PARENT_WEIGHTS_SIZE_BYTES
        or not warm_config.is_file()
        or sha256_file(warm_config) != WARM_START_CONFIG_SHA256
        or not warm_weights.is_file()
        or sha256_file(warm_weights) != WARM_START_WEIGHTS_SHA256
    ):
        raise ValueError("authenticated replay_clean interleaved-parent identity changed")


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    gate = (EXP / "scripts" / "check_local.py").read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 1520",
        '"optimizer_steps": 190,',
        '"seed": 56,',
        'PARENT_WEIGHTS_SHA256 = "f6f910ed1c1dcc843f43e09a562556b8e76ee40096aa7123fd70800d94fc6bb8"',
        'PARENT_CONFIG_SHA256 = "015bb13568c411c94d24460a9007e1f0d8fe3eb6c9749ad938958490de84d961"',
        "PASS_CONTROL_TRAINING",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88019",
        "LOCAL_ROWS = 124",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88019",
        "AGGREGATE_SEED = 78149",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
    )
    required_gate = (
        "SEED = 88019",
        "AGGREGATE_SEED = 78149",
        "DETECTABILITY_CEILING = 9",
        "GATE_UNDETECTABLE",
        "def required_kind_wins",
        "def normalize_answer",
        '"hygiene_win"',
        '"explore_win"',
    )
    if any(value not in trial for value in required_trial):
        raise ValueError("frozen training wrapper contract changed")
    if any(value not in merge for value in required_merge):
        raise ValueError("frozen merge wrapper contract changed")
    if any(value not in harness for value in required_harness):
        raise ValueError("frozen stage-harness contract changed")
    if any(value not in gate for value in required_gate):
        raise ValueError("frozen promotion-gate contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness, gate)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_lifecycle_contracts()
    banned = tuple(axis.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "interleaved_replay_dose_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "replay_clean",
            "deployment": "explicit_merged_composite",
            "tracked_receipt_sha256": PARENT_RECEIPT_SHA256,
            "external_receipt_sha256": PARENT_EXTERNAL_RECEIPT_SHA256,
            "tree_sha256": PARENT_TREE_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "weights_size_bytes": PARENT_WEIGHTS_SIZE_BYTES,
            "runtime_lora_forbidden": True,
            "warm_start_adapter": {
                "path": WARM_START_ADAPTER.relative_to(ROOT).as_posix(),
                "config_sha256": WARM_START_CONFIG_SHA256,
                "weights_sha256": WARM_START_WEIGHTS_SHA256,
            },
        },
        "seeds": {
            "construction": CONSTRUCTION_SEED,
            "stream_slot_matching_namespace": 55122,
            "training": 56,
            "local_gate": 88019,
            "conditional_aggregate": 78149,
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": 80,
                "sha256": TREATMENT_SHA256,
                "kinds": manifest["corpus"]["kinds"],
                "balance": manifest["balance"],
                "inheritance": {
                    "from_experiment": PARENT_EXP.name,
                    "byte_identical": True,
                    "rederived_from_copied_generator": True,
                },
                "construction": {
                    "generator": "scripts/gen_axis_v2.py",
                    "mix": TREATMENT_MIX,
                    "seed": CONSTRUCTION_SEED,
                    "builder": "scripts/build_corpus.py",
                },
            },
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
                "inheritance": {
                    "from_experiment": PARENT_EXP.name,
                    "byte_identical": True,
                },
            },
            "banned_vocabulary": {
                "tokens": len(banned),
                "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
                "checked_by_generator": True,
            },
        },
        "arms": {
            "control": "replay_interleaved2",
            "candidate": "dose_after_replay",
            "parent_eval_label": PARENT_EVAL_LABEL,
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "dose_after_replay": {"treatment_rows": 80, "replay_filler_rows": 160},
                "replay_interleaved2": {"replay_rows": 240},
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "position_aligned_identical_core_slots": True,
            "slot_order_seed": 55122,
            "exact_match_axes": ["forward", "nonzero_target", "absolute_loss_mass_x5"],
            "match_limit": 0,
            "solver": {
                "method": "scipy.optimize.milp",
                "backend": "highs",
                "mip_rel_gap": 0,
                "time_limit_seconds": 600,
                "filler_and_control_solved_jointly": True,
            },
        },
        "training_plan": {
            "training_authorized": False,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 8,
            "optimizer_steps": 190,
            "lr": 1e-5,
            "rank": 32,
            "alpha": 64,
            "w_think": 0.2,
            "w_close": 0.2,
            "max_length": 4096,
            "seed": 56,
            "rows_per_arm": 1520,
            "zero_skipped_rows_required": True,
            "warm_start": "replay_clean",
            "required_before_training": [
                "committed exact three-axis stream token receipt",
                "second adversarial compute review with PASS_CONTROL_TRAINING",
            ],
        },
        "local_gate": {
            "seed": 88019,
            "rows": 124,
            "instruments": {
                "axis_holdout": {
                    "rows": 20,
                    "per_kind": 10,
                    "kinds": list(AXIS_KINDS),
                    "generator": "scripts/gen_axis_v2.py",
                },
                "retention": {
                    "rows": 104,
                    "per_kind": 8,
                    "skills": 13,
                    "generator": "scripts/gen_curriculum.py",
                },
            },
            "arms": [PARENT_EVAL_LABEL, *ARMS],
            "answer_normalization": ANSWER_NORMALIZATION,
            "promotion": {
                "axis_total_strictly_beats": [PARENT_EVAL_LABEL, "replay_interleaved2"],
                "detectable_kind_definition": (
                    "an axis kind where NEITHER control scores >= 9 of 10 "
                    "on the holdout"
                ),
                "detectability_ceiling_correct_of_10": 9,
                "undetectable_kinds_excluded_and_reported_as_not_detectable": True,
                "axis_kind_wins_at_least": "ceil(2/3 * detectable_kinds)",
                "with_two_detectable_kinds_required_wins": 2,
                "both_kinds_must_strictly_win_when_both_detectable": True,
                "axis_kind_win_requires_strictly_above_max_of_both_controls": True,
                "zero_detectable_kinds_fails_closed_as": "GATE_UNDETECTABLE",
                "recovery_flags_recorded_unconditionally": ["hygiene_win", "explore_win"],
                "retention_correct_band": 5,
                "retention_cap_contact_band": 3,
                "retention_parsed_band": 3,
                "route_abstentions_at_most": 4,
                "no_absolute_per_kind_floors": True,
                "single_candidate": "dose_after_replay",
                "no_passing_candidate_keeps_aggregate_seed_sealed": True,
            },
        },
        "benchmark_plan": {
            "seed": 78149,
            "tier": "medium",
            "think_budget": 1024,
            "name": "pilot",
            "gateway": "scripts/run_benchmark_aggregate.py",
            "models": {
                "base": "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
                "parent": f"large_artifacts/{PARENT_EXP.name}/merged/replay_clean",
                "control": f"large_artifacts/{EXP.name}/merged/replay_interleaved2",
                "candidate": f"large_artifacts/{EXP.name}/merged/dose_after_replay",
            },
            "base_weights_sha256": (
                "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db"
            ),
            "public_families": list(PUBLIC_FAMILIES),
            "gates": {
                "candidate_aggregate_strictly_above_base": True,
                "candidate_aggregate_strictly_above_control": True,
                "candidate_aggregate_strictly_above_parent": True,
            },
            "goal_gate": {
                "every_public_family_strictly_above_base": (
                    "recorded and reported from the same event as the goal "
                    "gate; medium-tier family scores have finer granularity "
                    "than quick and the all-families gate is empirically "
                    "reachable there (8 of 92 historical medium events), but "
                    "it is NOT part of the pilot pass"
                ),
            },
        },
        "checkpoint_policy": {
            "next_authorized_stage": "train-control",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "preceding_receipt_committed_at_head": True,
            "full_check_rebase_push_two_workflow_gate_between_expensive_stages": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "code": {
            "curriculum_generator_sha256": sha256_file(EXP / "scripts" / "gen_curriculum.py"),
            "axis_v2_generator_sha256": sha256_file(EXP / "scripts" / "gen_axis_v2.py"),
            "corpus_builder_sha256": sha256_file(EXP / "scripts" / "build_corpus.py"),
            "trainer_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    value = (json.dumps(build_receipt(), indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    if args.check:
        if not args.out.is_file() or args.out.read_bytes() != value:
            parser.error("design receipt is absent or changed")
    else:
        if args.out.exists():
            parser.error("refusing to overwrite design receipt")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
    print(json.dumps({"out": str(args.out), "sha256": sha256_bytes(value)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
