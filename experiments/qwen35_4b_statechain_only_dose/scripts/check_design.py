#!/usr/bin/env python3
"""Recompute the model-free statechain-only dose design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_statechain_curriculum as statechain  # noqa: E402
from check_local import ANSWER_NORMALIZATION  # noqa: E402


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "97aa9f504c43e7b124c46f17ae367c862c1f67952ff1c72c197b70f019390467"
TREATMENT_PATH = EXP / "data" / "sft_statechain_only.jsonl"
TREATMENT_SHA256 = "ab6c78458eb8a41f42ebee25e79354d42beb558f78bb3d348dd7902ca3a9bad3"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77140
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_hygiene_explore_destack_medium"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "hygiene_explore"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "hygiene_explore.json"
PARENT_RECEIPT_SHA256 = "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
PARENT_EXTERNAL_RECEIPT_SHA256 = "9eafc8b0b01b125c3cb2eaa3cf8b808ecb7b7ed40172cbbb2f92117be75995a9"
PARENT_TREE_SHA256 = "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
PARENT_WEIGHTS_SHA256 = "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
PARENT_WEIGHTS_SIZE_BYTES = 9_078_620_536
# Lineage constants recorded inside the parent's committed merge receipt (the
# adapter that PRODUCED the hygiene_explore composite). No warm start exists
# in this cell — both fresh rank-32 adapters train on the composite itself.
PARENT_LINEAGE_ADAPTER_CONFIG_SHA256 = "2174356710c961e3b4df956c6add4b8cc8bd129a01e872c3a20829f46bdeae84"
PARENT_LINEAGE_ADAPTER_WEIGHTS_SHA256 = "7e28d6d152e7c2dbf7641d8516b9b47a1465b34967476ab01389d941b9563316"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ARMS = ("replay_ctl2", "statechain_only")
CANDIDATE_ARMS = ("statechain_only",)
PARENT_EVAL_LABEL = "hygiene_explore_parent"
GATE_ARMS = ("hygiene_explore_parent", "replay_ctl2", "statechain_only")
TREATMENT_KIND = "u_statechain"
LOCAL_SEED = 88033
SCREEN_SEEDS = (88034, 88035, 88036)
AGGREGATE_SEED = 78154


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
    expected_kinds = {TREATMENT_KIND: 160}
    expected_surfaces = {
        formalism: 40 for formalism in statechain.STATECHAIN_FORMALISMS
    }
    surface_audit = manifest.get("surface_freshness_audit", {})
    overlap_audit = manifest.get("row_overlap_audit", {})
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != expected_kinds
        or manifest.get("corpus", {}).get("surfaces") != expected_surfaces
        or replay_rows != REPLAY_ROWS
        # The surface-freshness grep audit must cover every fresh token
        # against every pinned predecessor corpus with zero hits.
        or not surface_audit.get("sources")
        or any(
            source.get("hits") != 0
            for source in surface_audit.get("sources", {}).values()
        )
        or surface_audit.get("tokens") != list(statechain.FRESH_SURFACE_TOKENS)
        # The row-overlap audit must cover every pinned source with zero
        # canonical-message overlap: the retained brewvat/courierloft rows
        # are FRESH instances, not copies of the reference cell's rows.
        or not overlap_audit.get("sources")
        or any(
            source.get("overlap") != 0
            for source in overlap_audit.get("sources", {}).values()
        )
    ):
        raise ValueError("corpus manifest violates the frozen construction contract")
    return manifest


def check_parent_identity() -> None:
    parent_receipt = json.loads(PARENT_RECEIPT.read_text(encoding="utf-8"))
    external = PARENT / "merge_receipt.json"
    weight = PARENT / "model.safetensors"
    if (
        sha256_file(PARENT_RECEIPT) != PARENT_RECEIPT_SHA256
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or parent_receipt.get("name") != "hygiene_explore"
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
        raise ValueError("authenticated hygiene_explore-parent identity changed")


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    gate = (EXP / "scripts" / "check_local.py").read_text(encoding="utf-8")
    bench = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 1520",
        '"optimizer_steps": 190,',
        '"seed": 67,',
        "LORA_RANK = 32",
        "LORA_ALPHA = 64",
        'MODEL_PATH_WEIGHTS_SHA256 = "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"',
        'MODEL_PATH_TREE_SHA256 = "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"',
        '"fresh_adapter": True,',
        "--model-path",
        "PASS_CONTROL_TRAINING",
        'ARM_PREREQUISITES = {\n    "replay_ctl2": (),\n    "statechain_only": ("replay_ctl2",),\n}',
        "PUBLISHED_ARM_HASHES",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "--base-model",
        'BASE_COMPOSITE_RECEIPT_SHA256 = "9eafc8b0b01b125c3cb2eaa3cf8b808ecb7b7ed40172cbbb2f92117be75995a9"',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88033",
        "LOCAL_ROWS = 352",
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88033",
        "SCREEN_SEEDS = (88034, 88035, 88036)",
        "AGGREGATE_SEED = 78154",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
        "PASS_LOCAL_EVENT",
        "PASS_BENCHMARK_EVENT",
    )
    required_gate = (
        "SEED = 88033",
        "SCREEN_SEEDS = (88034, 88035, 88036)",
        "AGGREGATE_SEED = 78154",
        "RETENTION_CORRECT_BAND = 5",
        "RETENTION_CAP_BAND = 3",
        "RETENTION_PARSED_BAND = 3",
        "def normalize_answer",
        "def pooled_sd",
        "def evaluate_promotion",
        "def pooled_band_checks",
        '"single_kind_dose_no_per_kind_split"',
        '"pooled_k3"',
        "no_absolute_per_kind_floors",
    )
    required_bench = (
        'FROZEN_NAME = "pilot"',
        'FROZEN_TIER = "medium"',
        "FROZEN_THINK_BUDGET = 1024",
        "FROZEN_SEED = 78154",
        "def require_unconsumed_ledger",
        "def _valid_score",
        "def pilot_gate",
        "def goal_gate_reading",
        "PASS_BENCHMARK_EVENT",
        "POWER_STATEMENT",
        'BASE_MERGE_RECEIPT_SHA256 = (\n    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"\n)',
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
        raise ValueError("frozen promotion-gate contract changed")
    if any(value not in bench for value in required_bench):
        raise ValueError("frozen benchmark-runner contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness, gate, bench)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_lifecycle_contracts()
    banned = tuple(statechain.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "statechain_only_dose_design",
        "model": {"id": MODEL_ID, "revision": MODEL_REVISION, "loaded": False, "calls": 0},
        "parent": {
            "experiment": PARENT_EXP.name,
            "arm": "hygiene_explore",
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
        "seeds": {
            "construction": CONSTRUCTION_SEED,
            "stream_slot_matching_namespace": 55131,
            "training": 67,
            "local_gate": LOCAL_SEED,
            "retention_screens": list(SCREEN_SEEDS),
            "seed_freshness": {
                "pinned": [
                    CONSTRUCTION_SEED,
                    55131,
                    67,
                    LOCAL_SEED,
                    *SCREEN_SEEDS,
                    AGGREGATE_SEED,
                ],
                "substitution_required": False,
                "verified": (
                    "every pinned seed verified grep-fresh in seed contexts "
                    "across experiments/, knowledge/, research_programs/, "
                    "scripts/, and configs/ at design time; 88031 is taken by "
                    "a prior experiment's run seed, 88032 was left unused by "
                    "the frozen design, and 78152/78153 were spent by prior "
                    "cells"
                ),
                "rule_if_collision": "next free integer, recorded here",
            },
            "aggregate_sealed": AGGREGATE_SEED,
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": 160,
                "sha256": TREATMENT_SHA256,
                "kinds": manifest["corpus"]["kinds"],
                "surfaces": manifest["corpus"]["surfaces"],
                "balance": manifest["balance"],
                "construction": {
                    "generator": "scripts/gen_statechain_curriculum.py",
                    "mix": statechain.ARM_MIX,
                    "seed": CONSTRUCTION_SEED,
                    "builder": "scripts/build_corpus.py",
                },
                "statechain_contract": manifest["statechain_contract"],
                "surface_freshness_audit": {
                    "tokens": len(manifest["surface_freshness_audit"]["tokens"]),
                    "token_inventory_sha256": manifest["surface_freshness_audit"][
                        "token_inventory_sha256"
                    ],
                    "inherited_tokens": len(
                        manifest["surface_freshness_audit"]["inherited_tokens"][
                            "tokens"
                        ]
                    ),
                    "sources_checked": len(
                        manifest["surface_freshness_audit"]["sources"]
                    ),
                    "total_hits": 0,
                },
                "row_overlap_audit": {
                    "sources_checked": len(manifest["row_overlap_audit"]["sources"]),
                    "total_overlap": 0,
                },
            },
            "replay": {
                "path": REPLAY_PATH.relative_to(EXP).as_posix(),
                "rows": REPLAY_ROWS,
                "sha256": REPLAY_SHA256,
                "inheritance": {
                    "byte_identical_to_every_predecessor_copy": True,
                },
            },
            "banned_vocabulary": {
                "tokens": len(banned),
                "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
                "checked_by_generator": True,
                "extended_with_reference_feedloop_pools": True,
                "retained_statechain_surfaces_not_banned": True,
            },
        },
        "arms": {
            "trained_control": "replay_ctl2",
            "trained_candidate": "statechain_only",
            "control_trains_first": True,
            "parent_eval_label": PARENT_EVAL_LABEL,
            "gate_arms": list(GATE_ARMS),
        },
        "stream_geometry": {
            "rows_per_arm": 1520,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 240,
            "blocks": {
                "replay_ctl2": {"replay_rows": 240},
                "statechain_only": {"treatment_rows": 160, "replay_filler_rows": 80},
            },
            "blocks_and_core_pairwise_disjoint_replay_indices": True,
            "slot_order_seed": 55131,
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
            "seed": 67,
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
            "seed": LOCAL_SEED,
            "screen_seeds": list(SCREEN_SEEDS),
            "rows_per_arm": 352,
            "instruments": {
                "axis_holdout": {
                    "rows": 40,
                    "kind": TREATMENT_KIND,
                    "per_surface": 10,
                    "surfaces": list(statechain.STATECHAIN_FORMALISMS),
                    "generator": "scripts/gen_statechain_curriculum.py",
                },
                "retention": {
                    "rows_per_screen": 104,
                    "per_kind": 8,
                    "skills": 13,
                    "screens": len(SCREEN_SEEDS),
                    "generator": "scripts/gen_curriculum.py",
                    "adjudication": "pooled_k3",
                },
            },
            "arms": list(GATE_ARMS),
            "answer_normalization": ANSWER_NORMALIZATION,
            "promotion": {
                "axis_total_strictly_beats_parent_and_replay": True,
                "single_kind_dose_no_per_kind_split": True,
                "per_surface_reported_not_gated": True,
                "retention_pooled_correct_band": 5,
                "retention_pooled_cap_contact_band": 3,
                "retention_pooled_parsed_band": 3,
                "bands_on_pooled_means_not_per_screen": True,
                "bands_evaluated_on_pooled_sums_times_screens": True,
                "no_absolute_per_kind_floors": True,
            },
        },
        "benchmark_plan": {
            "conditional_on_local_promotion": True,
            "name": "pilot",
            "tier": "medium",
            "think_budget": 1024,
            "aggregate_seed_sealed": AGGREGATE_SEED,
            "model_order": [
                "base",
                PARENT_EVAL_LABEL,
                "replay_ctl2",
                "statechain_only",
            ],
            "pilot_gate": (
                "candidate aggregate strictly > base AND > replay_ctl2 AND > "
                "hygiene_explore_parent"
            ),
            "goal_gate": (
                "all ten public families strictly > base; recorded from the "
                "same event either way, never part of the pilot pass"
            ),
            "power_statement": (
                "menders is 0 for every believable arm (three pedagogies + "
                "the budget lever all closed), so the maximum reachable is "
                "9/10; the reading of interest is whether the LOCAL "
                "statechain install converts to the rites FAMILY (the "
                "parent's rites was 0.0 at 78150; any candidate rites > 0 "
                "with base at 0 is a strict win there)"
            ),
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
            "statechain_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_statechain_curriculum.py"
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
