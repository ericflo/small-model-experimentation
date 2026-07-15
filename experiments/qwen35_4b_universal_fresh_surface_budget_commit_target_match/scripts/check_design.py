#!/usr/bin/env python3
"""Recompute the model-free fresh-surface/budget-commit design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_fresh_curriculum as fresh  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "7b580e807972a5cbc186e2a4afff0a3d6c821c0bb7874fcb090b9dc24461381b"
ARM_D_PATH = EXP / "data" / "sft_fresh_designed160.jsonl"
ARM_D_SHA256 = "e599f1563f6e4fe68aa43ce64bbe450c264170fb07dc17dba9ea74e694f284d5"
ARM_B_PATH = EXP / "data" / "sft_fresh_budget160.jsonl"
ARM_B_SHA256 = "ecece8e294f0b1c34a086705b05773d6004c6a9a239283aa26ee1c1bbad39800"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_after_close"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_after_close.json"
PARENT_RECEIPT_SHA256 = "bc78f33218afb99b4ebd5b173f1f24aa628b20fad82d627b00529cabf911d550"
PARENT_EXTERNAL_RECEIPT_SHA256 = "aa763255cb3b05599e765948d3a3db1787d5813b1cfafbdc7e1c21653ae745a3"
PARENT_WEIGHTS_SHA256 = "7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
WARM_START_ADAPTER = (
    ROOT / "large_artifacts" / PARENT_EXP.name / "adapters" / "replay_after_close"
)
WARM_START_CONFIG_SHA256 = "0dfd9bda8a835926a87337782cc09b1e11e841a36f46b99c83fbae9bc89e120f"
WARM_START_WEIGHTS_SHA256 = "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_repeat", "designed_fresh", "budget_commit")
CANDIDATE_ARMS = ("designed_fresh", "budget_commit")
PARENT_EVAL_LABEL = "replay_after_close_parent"
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
        (ARM_D_PATH, ARM_D_SHA256),
        (ARM_B_PATH, ARM_B_SHA256),
        (REPLAY_PATH, REPLAY_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"frozen corpus artifact is absent or changed: {path}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    replay_rows = len(REPLAY_PATH.read_text(encoding="utf-8").splitlines())
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != 77116
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("arm_b_subset_of_arm_d") is not True
        or manifest.get("arm_d", {}).get("rows") != 160
        or manifest.get("arm_d", {}).get("sha256") != ARM_D_SHA256
        or manifest.get("arm_b", {}).get("rows") != 160
        or manifest.get("arm_b", {}).get("sha256") != ARM_B_SHA256
        or replay_rows != REPLAY_ROWS
    ):
        raise ValueError("corpus manifest violates the frozen construction contract")
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
        or parent_receipt.get("name") != "replay_after_close"
        or parent_receipt.get("merge_receipt_sha256") != PARENT_EXTERNAL_RECEIPT_SHA256
        or parent_receipt.get("weight_files", [{}])[0].get("sha256") != PARENT_WEIGHTS_SHA256
        or not external.is_file()
        or sha256_file(external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or not weight.is_file()
        or weight.stat().st_size != PARENT_WEIGHTS_SIZE_BYTES
        or not warm_config.is_file()
        or sha256_file(warm_config) != WARM_START_CONFIG_SHA256
        or not warm_weights.is_file()
        or sha256_file(warm_weights) != WARM_START_WEIGHTS_SHA256
    ):
        raise ValueError("authenticated replay-parent identity changed")


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 1520",
        '"optimizer_steps": 190,',
        '"seed": 51,',
        'PARENT_WEIGHTS_SHA256 = "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d"',
        "PASS_CONTROL_TRAINING",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88013",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-designed"',
        '"train-budget"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88013",
        "AGGREGATE_SEED = 78143",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
    )
    if any(value not in trial for value in required_trial):
        raise ValueError("frozen training wrapper contract changed")
    if any(value not in merge for value in required_merge):
        raise ValueError("frozen merge wrapper contract changed")
    if any(value not in harness for value in required_harness):
        raise ValueError("frozen stage-harness contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_lifecycle_contracts()
    banned = tuple(fresh.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "fresh_surface_budget_commit_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "replay_after_close",
            "deployment": "explicit_merged_composite",
            "tracked_receipt_sha256": PARENT_RECEIPT_SHA256,
            "external_receipt_sha256": PARENT_EXTERNAL_RECEIPT_SHA256,
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
            "construction": 77116,
            "stream_slot_matching_namespace": 55117,
            "training": 51,
            "local_gate": 88013,
            "conditional_aggregate": 78143,
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "arm_d": {
                "path": ARM_D_PATH.relative_to(EXP).as_posix(),
                "rows": 160,
                "sha256": ARM_D_SHA256,
                "kinds": manifest["arm_d"]["kinds"],
            },
            "arm_b": {
                "path": ARM_B_PATH.relative_to(EXP).as_posix(),
                "rows": 160,
                "sha256": ARM_B_SHA256,
                "kinds": manifest["arm_b"]["kinds"],
                "designed_subset_rows": manifest["arm_b"]["designed_subset_rows"],
                "budget_rows": manifest["arm_b"]["budget_rows"],
            },
            "arm_b_subset_of_arm_d": True,
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
            },
            "banned_vocabulary": {
                "tokens": len(banned),
                "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
                "checked_by_generator": True,
            },
        },
        "arms": {
            "control": "replay_repeat",
            "candidate_designed": "designed_fresh",
            "candidate_budget": "budget_commit",
            "parent_eval_label": PARENT_EVAL_LABEL,
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "designed_fresh": {"treatment_rows": 160, "replay_filler_rows": 80},
                "budget_commit": {"treatment_rows": 160, "replay_filler_rows": 80},
                "replay_repeat": {"replay_rows": 240},
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "position_aligned_identical_core_slots": True,
            "slot_order_seed": 55117,
            "exact_match_axes": ["forward", "nonzero_target", "absolute_loss_mass_x5"],
            "match_limit": 0,
            "solver": {
                "method": "scipy.optimize.milp",
                "backend": "highs",
                "mip_rel_gap": 0,
                "time_limit_seconds": 600,
                "fillers_and_control_solved_jointly": True,
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
            "seed": 51,
            "rows_per_arm": 1520,
            "zero_skipped_rows_required": True,
            "warm_start": "replay_after_close",
            "required_before_training": [
                "committed exact three-axis stream token receipt",
                "second adversarial compute review with PASS_CONTROL_TRAINING",
            ],
        },
        "local_gate": {
            "seed": 88013,
            "rows": 104,
            "tasks_per_skill": 8,
            "skills": 13,
            "generator": "scripts/gen_curriculum.py",
            "generator_surfaces": "original_uc2",
            "fresh_surface_generator_excluded": True,
            "arms": [PARENT_EVAL_LABEL, *ARMS],
            "bars_per_candidate": {
                "parsed_min": 96,
                "correct_min": 68,
                "cap_contacts_max": 8,
                "feasible_route_abstentions_max": 4,
                "per_kind_correct_min": {"u_execute": 4, "u_induct": 4, "u_probe": 4},
            },
            "relative_wins_strict": {
                "total_correct_beats": [PARENT_EVAL_LABEL, "replay_repeat"],
                "target_24_row_correct_beats": [PARENT_EVAL_LABEL, "replay_repeat"],
                "target_kinds": ["u_execute", "u_induct", "u_probe"],
                "target_rows": 24,
            },
            "promotion": {
                "single_winner": True,
                "tie_break_order": [
                    "total_correct",
                    "target_24_row_correct",
                    "fewer_cap_contacts",
                    "budget_commit",
                ],
                "no_passing_candidate_keeps_aggregate_seed_sealed": True,
            },
        },
        "benchmark_plan": {
            "seed": 78143,
            "tier": "quick",
            "think_budget": 1024,
            "gateway": "scripts/run_benchmark_aggregate.py",
            "models": {
                "base": "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
                "parent": f"large_artifacts/{PARENT_EXP.name}/merged/replay_after_close",
                "control": f"large_artifacts/{EXP.name}/merged/replay_repeat",
                "candidate": f"large_artifacts/{EXP.name}/merged/<promoted>",
            },
            "public_families": list(PUBLIC_FAMILIES),
            "gates": {
                "candidate_aggregate_strictly_above_base": True,
                "every_public_family_strictly_above_base": True,
                "candidate_aggregate_strictly_above_control": True,
                "candidate_aggregate_strictly_above_parent": True,
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
            "fresh_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_fresh_curriculum.py"
            ),
            "corpus_builder_sha256": sha256_file(EXP / "scripts" / "build_corpora.py"),
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
