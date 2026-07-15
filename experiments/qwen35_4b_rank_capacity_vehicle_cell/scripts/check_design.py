#!/usr/bin/env python3
"""Recompute the model-free rank-capacity vehicle-cell design receipt."""

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
from check_local import ANSWER_NORMALIZATION  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "c7e4427a9b0322776484be567f6ab2e4340fadf808fe59a357ac4325d6283b60"
TREATMENT_PATH = EXP / "data" / "sft_axis160.jsonl"
TREATMENT_SHA256 = "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77117
DONOR_EXP = ROOT / "experiments" / "qwen35_4b_dose_diversity_mechanism_cell"
DONOR_MANIFEST_SHA256 = "b029c71389ec98314bf5ee8c7e92c828ac996899beaeeb0e6cd469c6ed9eec1f"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "designed_fresh"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "designed_fresh.json"
PARENT_RECEIPT_SHA256 = "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2"
PARENT_EXTERNAL_RECEIPT_SHA256 = "10b0de69bbda6edf8e216f45667b15f73557102f0661461fec620786e8a9170f"
PARENT_TREE_SHA256 = "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255"
PARENT_WEIGHTS_SHA256 = "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
# Lineage constants recorded inside the parent's committed merge receipt (the
# adapter that PRODUCED the designed_fresh composite). No warm start exists in
# this cell — the fresh rank-64 adapter trains on the composite itself.
PARENT_LINEAGE_ADAPTER_CONFIG_SHA256 = "5966461bd9dbfe280b226940e79eec49030cc44e4131089147634df549dc4055"
PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256 = "36f41095c2d628e4706694e7d64d16aba815870a1d3660af0e24b14dc0e6b442"
DOSE_EXP_NAME = "qwen35_4b_dose_diversity_mechanism_cell"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("axis160_r64",)
CANDIDATE_ARMS = ("axis160_r64",)
PARENT_EVAL_LABEL = "clean_parent"
GATE_ARMS = ("clean_parent", "axis160_direct", "axis160_r64")
AXIS_KINDS = ("u_explore", "u_hygiene", "u_protocol", "u_tracefix")
PUBLISHED_COMPOSITES = {
    "axis160_direct": {
        "experiment": DOSE_EXP_NAME,
        "composite": f"large_artifacts/{DOSE_EXP_NAME}/merged/axis160_direct",
        "merge_receipt": (
            f"experiments/{DOSE_EXP_NAME}/runs/merges/axis160_direct.json"
        ),
        "merge_receipt_sha256": (
            "7b878bb357e044c58a5ba27f34365906059259237e657ca77c2ad2e8fb77ea39"
        ),
        "tree_sha256": (
            "b773fd439eca753e70d3b16862497ee0ba9783cdb024ddff84b4595d10b7da61"
        ),
        "weights_sha256": (
            "3cd4272ee386dc3b8e878ef3b2ac9ebf94dd8d47ef7d9d480f921e7e45b37279"
        ),
    },
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_frozen_corpora() -> dict:
    if MANIFEST_SHA256 is None:
        raise ValueError("corpus manifest pin is unfilled (TODO-PIN)")
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
    donor_treatment = DONOR_EXP / "data" / "sft_axis160.jsonl"
    donor_replay = DONOR_EXP / "data" / "sft_blend.jsonl"
    donor_manifest = DONOR_EXP / "data" / "corpus_manifest.json"
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != expected_kinds
        or replay_rows != REPLAY_ROWS
        # Byte-identical inheritance from the goal-gap donor: the corpus rows
        # come from the donor's frozen file, regenerated here from the copied
        # generator at the same construction seed.
        or not donor_treatment.is_file()
        or sha256_file(donor_treatment) != TREATMENT_SHA256
        or not donor_replay.is_file()
        or sha256_file(donor_replay) != REPLAY_SHA256
        or not donor_manifest.is_file()
        or sha256_file(donor_manifest) != DONOR_MANIFEST_SHA256
    ):
        raise ValueError("corpus manifest violates the frozen inheritance contract")
    return manifest


def check_parent_identity() -> None:
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    external = PARENT / "merge_receipt.json"
    weight = PARENT / "model.safetensors"
    if (
        sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or parent_receipt.get("name") != "designed_fresh"
        or parent_receipt.get("merge_receipt_sha256") != PARENT_EXTERNAL_RECEIPT_SHA256
        or parent_receipt.get("output_tree_sha256") != PARENT_TREE_SHA256
        or {row.get("name"): row.get("sha256") for row in parent_receipt.get("weight_files", [])}
        != {"model.safetensors": PARENT_WEIGHTS_SHA256}
        or parent_receipt.get("adapter_config_sha256")
        != PARENT_LINEAGE_ADAPTER_CONFIG_SHA256
        or parent_receipt.get("adapter_weights_sha256")
        != PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256
        or not external.is_file()
        or sha256_file(external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or not weight.is_file()
        or weight.stat().st_size != PARENT_WEIGHTS_SIZE_BYTES
    ):
        raise ValueError("authenticated designed_fresh-parent identity changed")


def check_published_composites() -> None:
    for label, pins in PUBLISHED_COMPOSITES.items():
        receipt_path = ROOT / pins["merge_receipt"]
        if (
            not receipt_path.is_file()
            or sha256_file(receipt_path) != pins["merge_receipt_sha256"]
        ):
            raise ValueError(f"published composite receipt changed: {label}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or payload.get("output_tree_sha256") != pins["tree_sha256"]
            or {
                row.get("name"): row.get("sha256")
                for row in payload.get("weight_files", [])
            }
            != {"model.safetensors": pins["weights_sha256"]}
            or Path(payload.get("merged", "")).resolve()
            != (ROOT / pins["composite"]).resolve()
        ):
            raise ValueError(f"published composite receipt violates pins: {label}")


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
        '"seed": 58,',
        "LORA_RANK = 64",
        "LORA_ALPHA = 128",
        'MODEL_PATH_WEIGHTS_SHA256 = "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979"',
        'MODEL_PATH_TREE_SHA256 = "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255"',
        '"fresh_adapter": True,',
        "--model-path",
        "PASS_CONTROL_TRAINING",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "--base-model",
        'BASE_COMPOSITE_RECEIPT_SHA256 = "10b0de69bbda6edf8e216f45667b15f73557102f0661461fec620786e8a9170f"',
        "PASS_LOCAL_EVENT",
        "LOCAL_SEED = 88021",
        "LOCAL_ROWS = 144",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-candidate"',
        '"merge-candidate"',
        '"local"',
        "LOCAL_SEED = 88021",
        "PASS_CONTROL_TRAINING",
        "PASS_LOCAL_EVENT",
    )
    required_gate = (
        "SEED = 88021",
        "RETAINED_AT_LEAST = -5",
        "FORGOT_AT_MOST = -6",
        "def normalize_answer",
        "def evaluate_mechanism",
        "def capacity_mechanism_verdict",
        "install_preserved",
        '"CAPACITY_SUPPORTED"',
        '"CAPACITY_REFUTED"',
        '"SCREEN_INSTABILITY"',
    )
    if any(value not in trial for value in required_trial):
        raise ValueError("frozen training wrapper contract changed")
    # The fresh-adapter contract: no warm-start pathway may exist anywhere in
    # the training or merge wrapper of this cell (prose mentions excluded).
    for token in ("--warm-start", "warm_start", "warm-start"):
        if token in trial or token in merge:
            raise ValueError("a warm-start path leaked into the fresh-adapter cell")
    if any(value not in merge for value in required_merge):
        raise ValueError("frozen merge wrapper contract changed")
    if any(value not in harness for value in required_harness):
        raise ValueError("frozen stage-harness contract changed")
    if any(value not in gate for value in required_gate):
        raise ValueError("frozen mechanism-gate contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness, gate)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")
    if "run_benchmark" in harness or (EXP / "scripts" / "run_benchmark.py").exists():
        raise ValueError("a benchmark stage leaked into the mechanism cell")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_published_composites()
    check_lifecycle_contracts()
    banned = tuple(axis.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "rank_capacity_vehicle_design",
        "mechanism_cell": True,
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "designed_fresh",
            "eval_label": PARENT_EVAL_LABEL,
            "deployment": "explicit_merged_composite",
            "training_base_via_model_path": True,
            "tracked_receipt_sha256": PARENT_RECEIPT_SHA256,
            "external_receipt_sha256": PARENT_EXTERNAL_RECEIPT_SHA256,
            "tree_sha256": PARENT_TREE_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
            "weights_size_bytes": PARENT_WEIGHTS_SIZE_BYTES,
            "runtime_lora_forbidden": True,
            "lineage_adapter_receipt_pins": {
                "config_sha256": PARENT_LINEAGE_ADAPTER_CONFIG_SHA256,
                "weights_sha256": PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256,
            },
        },
        "published_composite_arms": PUBLISHED_COMPOSITES,
        "seeds": {
            "construction": CONSTRUCTION_SEED,
            "stream_slot_matching_namespace": 55124,
            "training": 58,
            "local_gate": 88021,
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
                    "rederived_from_copied_generator": True,
                    "donor_manifest_sha256": DONOR_MANIFEST_SHA256,
                },
                "construction": {
                    "generator": "scripts/gen_axis_curriculum.py",
                    "mix": axis.ARM_MIX,
                    "seed": CONSTRUCTION_SEED,
                    "builder": "scripts/build_corpus.py",
                },
            },
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
                "inheritance": {
                    "from_experiment": DONOR_EXP.name,
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
            "trained_candidate": "axis160_r64",
            "control_arm_trained": False,
            "parent_eval_label": PARENT_EVAL_LABEL,
            "gate_arms": list(GATE_ARMS),
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "axis160_r64": {"treatment_rows": 160, "replay_filler_rows": 80},
                "notional_control": {
                    "replay_rows": 240,
                    "materialized": False,
                    "recorded_in_manifest_for_exactness_bookkeeping": True,
                },
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "slot_order_seed": 55124,
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
            "rank": 64,
            "alpha": 128,
            "w_think": 0.2,
            "w_close": 0.2,
            "max_length": 4096,
            "seed": 58,
            "rows_per_arm": 1520,
            "zero_skipped_rows_required": True,
            "fresh_adapter": True,
            "warm_start": None,
            "model_path_base_composite": {
                "path": PARENT.relative_to(ROOT).as_posix(),
                "tree_sha256": PARENT_TREE_SHA256,
                "weights_sha256": PARENT_WEIGHTS_SHA256,
            },
            "required_before_training": [
                "committed exact three-axis stream token receipt",
                "second adversarial compute review with PASS_CONTROL_TRAINING",
            ],
        },
        "local_gate": {
            "seed": 88021,
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
            "arms": list(GATE_ARMS),
            "answer_normalization": ANSWER_NORMALIZATION,
            "mechanism_readout": {
                "no_promotion": True,
                "retention_delta_r64": (
                    "axis160_r64 retention correct minus clean_parent "
                    "retention correct"
                ),
                "retention_delta_r32": (
                    "axis160_direct retention correct minus clean_parent "
                    "retention correct (seed-88020 measured -9; re-measured "
                    "fresh here)"
                ),
                "capacity_mechanism_verdict": {
                    "SCREEN_INSTABILITY": "retention_delta_r32 >= -5",
                    "CAPACITY_SUPPORTED": (
                        "retention_delta_r32 <= -6 and "
                        "retention_delta_r64 >= -5"
                    ),
                    "CAPACITY_REFUTED": (
                        "retention_delta_r32 <= -6 and "
                        "retention_delta_r64 <= -6"
                    ),
                },
                "install_preserved_flag": (
                    "axis160_r64 axis total correct >= axis160_direct axis "
                    "total correct on this same screen"
                ),
                "complete_event_always_exits_zero": True,
            },
        },
        "mechanism_plan": {
            "no_benchmark_stage": True,
            "no_aggregate_seed_exists": True,
            "stages": ["train-candidate", "merge-candidate", "local"],
        },
        "checkpoint_policy": {
            "next_authorized_stage": "train-candidate",
            "one_stage_per_invocation": True,
            "clean_pushed_main_required": True,
            "preceding_receipt_committed_at_head": True,
            "full_check_rebase_push_two_workflow_gate_between_expensive_stages": True,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "no_aggregate_seed_exists": True,
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
