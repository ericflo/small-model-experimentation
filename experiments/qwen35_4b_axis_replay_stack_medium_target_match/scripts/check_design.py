#!/usr/bin/env python3
"""Recompute the model-free axis-on-replay stack design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_curriculum as axis  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "dd881023b4437211cf05936d29599e3cd8eee0ded07fd7dba07849c0cd1231e4"
TREATMENT_PATH = EXP / "data" / "sft_axis160.jsonl"
TREATMENT_SHA256 = "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
DONOR_EXP = ROOT / "experiments" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_repeat"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_repeat.json"
PARENT_RECEIPT_SHA256 = "22384463d7825ec2a0b95faeaeb273264d7331f4584f8b7e9e58a60545398af1"
PARENT_EXTERNAL_RECEIPT_SHA256 = "d3b184010f0470078e77e25796c572f41c177451f0157ced35d4e4d818a11b5b"
PARENT_TREE_SHA256 = "4c4f3561efbcafe1b9f777f4bd21bf4949ff89177f77946d0fa0f88cafafacd7"
PARENT_WEIGHTS_SHA256 = "3df45004fcf42519ce28cdcfedcbb39b0907662f8ecfb8a87b13b416087d0072"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
WARM_START_ADAPTER = (
    ROOT / "large_artifacts" / PARENT_EXP.name / "adapters" / "replay_repeat"
)
WARM_START_CONFIG_SHA256 = "bf5ade0b3328489d5ba676aa497e311d9883f70908cf56bda69f73882e232bac"
WARM_START_WEIGHTS_SHA256 = "20be87b5c7a7969d006b2825d3937b10fd0627ea2358af02879451039a07cd36"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_squared", "axis_on_replay")
CANDIDATE_ARMS = ("axis_on_replay",)
PARENT_EVAL_LABEL = "replay_parent"
AXIS_KINDS = ("u_explore", "u_hygiene", "u_protocol", "u_tracefix")
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
    expected_kinds = {kind: 40 for kind in AXIS_KINDS}
    donor_corpus = DONOR_EXP / "data" / "sft_axis160.jsonl"
    donor_manifest = DONOR_EXP / "data" / "corpus_manifest.json"
    if (
        # The corpus and manifest are inherited byte-identically from the
        # goal-gap axis experiment; the manifest keeps the donor's identity.
        manifest.get("experiment_id") != DONOR_EXP.name
        or manifest.get("construction_seed") != 77117
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != expected_kinds
        or replay_rows != REPLAY_ROWS
        or not donor_corpus.is_file()
        or sha256_file(donor_corpus) != TREATMENT_SHA256
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
        or parent_receipt.get("name") != "replay_repeat"
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
        raise ValueError("authenticated replay_repeat-parent identity changed")


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 1520",
        '"optimizer_steps": 190,',
        '"seed": 53,',
        'PARENT_WEIGHTS_SHA256 = "20be87b5c7a7969d006b2825d3937b10fd0627ea2358af02879451039a07cd36"',
        'PARENT_CONFIG_SHA256 = "bf5ade0b3328489d5ba676aa497e311d9883f70908cf56bda69f73882e232bac"',
        "PASS_CONTROL_TRAINING",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88015",
        "LOCAL_ROWS = 144",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88015",
        "AGGREGATE_SEED = 78145",
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
    banned = tuple(axis.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "axis_on_replay_stack_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "replay_repeat",
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
            "construction": 77117,
            "construction_inherited_from": DONOR_EXP.name,
            "stream_slot_matching_namespace": 55119,
            "training": 53,
            "local_gate": 88015,
            "conditional_aggregate": 78145,
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": 160,
                "sha256": TREATMENT_SHA256,
                "kinds": manifest["corpus"]["kinds"],
                "balance": manifest["balance"],
                "inheritance": {
                    "from_experiment": DONOR_EXP.name,
                    "byte_identical": True,
                    "authenticator": "scripts/build_corpus.py",
                },
            },
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
            "control": "replay_squared",
            "candidate": "axis_on_replay",
            "parent_eval_label": PARENT_EVAL_LABEL,
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "axis_on_replay": {"treatment_rows": 160, "replay_filler_rows": 80},
                "replay_squared": {"replay_rows": 240},
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "position_aligned_identical_core_slots": True,
            "slot_order_seed": 55119,
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
            "seed": 53,
            "rows_per_arm": 1520,
            "zero_skipped_rows_required": True,
            "warm_start": "replay_repeat",
            "required_before_training": [
                "committed exact three-axis stream token receipt",
                "second adversarial compute review with PASS_CONTROL_TRAINING",
            ],
        },
        "local_gate": {
            "seed": 88015,
            "rows": 144,
            "instruments": {
                "axis_holdout": {
                    "rows": 40,
                    "per_kind": 10,
                    "kinds": list(AXIS_KINDS),
                    "generator": "scripts/gen_axis_curriculum.py",
                },
                "retention": {
                    "rows": 104,
                    "per_kind": 8,
                    "skills": 13,
                    "generator": "scripts/gen_curriculum.py",
                },
            },
            "arms": [PARENT_EVAL_LABEL, *ARMS],
            "promotion": {
                "axis_total_strictly_beats": [PARENT_EVAL_LABEL, "replay_squared"],
                "axis_kind_wins_at_least": 3,
                "axis_kind_win_requires_strictly_above_max_of_both_controls": True,
                "retention_correct_band": 5,
                "retention_cap_contact_band": 3,
                "retention_parsed_band": 3,
                "route_abstentions_at_most": 4,
                "no_absolute_per_kind_floors": True,
                "single_candidate": "axis_on_replay",
                "no_passing_candidate_keeps_aggregate_seed_sealed": True,
            },
        },
        "benchmark_plan": {
            "seed": 78145,
            "tier": "medium",
            "think_budget": 1024,
            "gateway": "scripts/run_benchmark_aggregate.py",
            "models": {
                "base": "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized",
                "parent": f"large_artifacts/{PARENT_EXP.name}/merged/replay_repeat",
                "control": f"large_artifacts/{EXP.name}/merged/replay_squared",
                "candidate": f"large_artifacts/{EXP.name}/merged/axis_on_replay",
            },
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
                    "reachable there, but it is NOT part of the pilot pass"
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
            "axis_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_axis_curriculum.py"
            ),
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
