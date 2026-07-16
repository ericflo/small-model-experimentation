#!/usr/bin/env python3
"""Recompute the model-free feedloop dose-scale design receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_feedloop_curriculum as feedloop  # noqa: E402
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    DOSE_RESPONSE_BASELINE,
    DOSE_RESPONSE_CONSEQUENCES,
)


OUT = EXP / "data" / "design_receipt.json"
MANIFEST = EXP / "data" / "corpus_manifest.json"
MANIFEST_SHA256 = "5617c2c540726223c9b29831aba177a186bd5008de1a2838d7f662ab0249a5b1"
TREATMENT_PATH = EXP / "data" / "sft_feedloop_scale.jsonl"
TREATMENT_SHA256 = "080c3603cd3bfca2b261b797be356f97684aedc4f65becd9b50cbb45706bd2c2"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77150
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
ARMS = ("replay_ctl3", "feedloop_scale")
CANDIDATE_ARMS = ("feedloop_scale",)
PARENT_EVAL_LABEL = "hygiene_explore_parent"
GATE_ARMS = ("hygiene_explore_parent", "replay_ctl3", "feedloop_scale")
TREATMENT_KIND = "u_feedloop"
LOCAL_SEED = 88037
SCREEN_SEEDS = (88038, 88039, 88040)
AGGREGATE_SEED = 78158
# STANDALONE LINEAGE PACKAGE (static core pins; the manifest's candidate
# ``produced`` fields are orchestrator-filled TODO-PINs after training, so
# the manifest bytes are deliberately NOT byte-pinned here — schema and
# copies are enforced by rebuild_lineage.py --verify-inputs, which run.py
# --smoke and the unit tests both run).
LINEAGE_DATASETS = {
    "stage01_replay_refresh.jsonl": (
        "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6",
        1520,
    ),
    "stage02_designed160.jsonl": (
        "5159cf41b6474bdc8640cdb2a4a168587b59232ca8171c7b7057fc6bfe1b40c8",
        1520,
    ),
    "stage03_close_xi__targeted_standard.jsonl": (
        "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00",
        320,
    ),
    "stage04_replay_after_close.jsonl": (
        "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6",
        320,
    ),
    "stage05_designed_fresh.jsonl": (
        "6d4dc303bc159c19a1ffd0c60ca7d08ea64b02909366701b345d888482d67f3f",
        1520,
    ),
    "stage06_hygiene_explore.jsonl": (
        "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f",
        1520,
    ),
}
LINEAGE_CANDIDATE_DATASET = (
    "data/feedloop_scale.jsonl",
    "3aee5f5e8731294e0d8a9b9dd08f014ca34ace9f949c2b46aa09aa1e548adc58",
    2280,
)
LINEAGE_TRAINERS = {
    "lineage_trainers/train_think_stage12.py": (
        "400e4b856a5899db8805510c9f6d8d11da76b35cf30af16856b46f41abdb4e54"
    ),
    "lineage_trainers/train_think_close_stage3.py": (
        "10b4914cdc6c24858a77750f4ab64074fae5cbf36d98f3e4fc26b8958068fb14"
    ),
    "lineage_trainers/train_think_stage456.py": (
        "0cfb126feae6d73238c02362066229fff0b6a846625eed167d7681edde322cc4"
    ),
    "train_think.py": (
        "e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01"
    ),
    "merge_adapter.py": (
        "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"
    ),
}
LINEAGE_TRAINING_SEEDS = (42, 43, 44, 47, 51, 55)
LINEAGE_ROOT = (
    ROOT / "large_artifacts" / EXP.name / "lineage_root" / "blend"
)
LINEAGE_ROOT_FILES = {
    "README.md": (
        "aeb83ceb33aa84802245cc3bc4abe24160d7d08550ca8278baff84d8b24e9ba4",
        5184,
    ),
    "adapter_config.json": (
        "cd764ae869b8a55526e283dd133e1940896b428839c6abf2e55d6ae2a0b32635",
        1094,
    ),
    "adapter_model.safetensors": (
        "ad2ef4fae785debedf5e50932a79bda97869d3efc212f53d48ccb04c59e25d21",
        169903320,
    ),
    "chat_template.jinja": (
        "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        7756,
    ),
    "tokenizer.json": (
        "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        19989325,
    ),
    "tokenizer_config.json": (
        "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
        1125,
    ),
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
    expected_kinds = {TREATMENT_KIND: 800}
    expected_surfaces = {
        formalism: 100 for formalism in feedloop.FEEDLOOP_FORMALISMS
    }
    surface_audit = manifest.get("surface_freshness_audit", {})
    overlap_audit = manifest.get("row_overlap_audit", {})
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 800
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
        or surface_audit.get("tokens") != list(feedloop.FRESH_SURFACE_TOKENS)
        # The row-overlap audit must cover every pinned source with zero
        # canonical-message overlap: the reused feedloop formalism rows are
        # FRESH instances, not copies of the reference cell's rows.
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


def check_lineage_package_static() -> None:
    """Static-core lineage pins (the manifest's schema/copies are enforced by
    rebuild_lineage.py --verify-inputs; the candidate produced fields are
    post-training TODO-PINs, so the manifest bytes are not pinned here)."""
    for name, (digest, _rows) in sorted(LINEAGE_DATASETS.items()):
        path = EXP / "data" / "lineage" / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage dataset copy is absent or changed: {path}")
    relative, digest, _rows = LINEAGE_CANDIDATE_DATASET
    candidate = EXP / relative
    if not candidate.is_file() or sha256_file(candidate) != digest:
        raise ValueError(f"candidate lineage stream is absent or changed: {candidate}")
    for name, digest in sorted(LINEAGE_TRAINERS.items()):
        path = EXP / "scripts" / name
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"lineage trainer/merger copy is absent or changed: {path}")
    if not LINEAGE_ROOT.is_dir() or LINEAGE_ROOT.is_symlink():
        raise ValueError(f"vendored lineage root adapter is absent: {LINEAGE_ROOT}")
    for name, (digest, size) in sorted(LINEAGE_ROOT_FILES.items()):
        path = LINEAGE_ROOT / name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != size
            or sha256_file(path) != digest
        ):
            raise ValueError(f"vendored lineage root file changed: {path}")


def check_lifecycle_contracts() -> None:
    """Substring contracts survive later TODO-PIN fills; hashes would not."""
    trial = (EXP / "scripts" / "train_trial.py").read_text(encoding="utf-8")
    merge = (EXP / "scripts" / "merge_trained_arm.py").read_text(encoding="utf-8")
    harness = (EXP / "scripts" / "run.py").read_text(encoding="utf-8")
    gate = (EXP / "scripts" / "check_local.py").read_text(encoding="utf-8")
    bench = (EXP / "scripts" / "run_benchmark.py").read_text(encoding="utf-8")
    lineage = (EXP / "scripts" / "rebuild_lineage.py").read_text(encoding="utf-8")
    required_trial = (
        "STREAM_TOKEN_RECEIPT_SHA256",
        "EXPECTED_ROWS = 2280",
        '"optimizer_steps": 285,',
        '"seed": 71,',
        "LORA_RANK = 32",
        "LORA_ALPHA = 64",
        'MODEL_PATH_WEIGHTS_SHA256 = "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"',
        'MODEL_PATH_TREE_SHA256 = "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"',
        '"fresh_adapter": True,',
        "--model-path",
        "PASS_CONTROL_TRAINING",
        'ARM_PREREQUISITES = {\n    "replay_ctl3": (),\n    "feedloop_scale": ("replay_ctl3",),\n}',
        "PUBLISHED_ARM_HASHES",
    )
    required_merge = (
        'MERGER_SHA256 = "cb9af8b45ca1e5754cb36f2213b7e25290f6eb16427d1a8b41f0b12b10396672"',
        'payload.get("applied_lora_modules") != 128',
        "--base-model",
        'BASE_COMPOSITE_RECEIPT_SHA256 = "9eafc8b0b01b125c3cb2eaa3cf8b808ecb7b7ed40172cbbb2f92117be75995a9"',
        "PASS_CONTROL_MERGE",
        "LOCAL_SEED = 88037",
        "LOCAL_ROWS = 352",
        'MERGER = EXP / "scripts" / "merge_adapter.py"',
    )
    required_harness = (
        "def require_pushed_checkpoint",
        '"train-control"',
        '"train-candidate"',
        '"merge-arms"',
        '"local"',
        '"benchmark"',
        "LOCAL_SEED = 88037",
        "SCREEN_SEEDS = (88038, 88039, 88040)",
        "AGGREGATE_SEED = 78158",
        "PASS_CONTROL_TRAINING",
        "PASS_CONTROL_MERGE",
        "PASS_LOCAL_EVENT",
        "PASS_BENCHMARK_EVENT",
        '"rebuild_lineage.py"), "--verify-inputs"',
        "def smoke_benchmark_ledger",
    )
    required_gate = (
        "SEED = 88037",
        "SCREEN_SEEDS = (88038, 88039, 88040)",
        "AGGREGATE_SEED = 78158",
        "RETENTION_CORRECT_BAND = 5",
        "RETENTION_CAP_BAND = 3",
        "RETENTION_PARSED_BAND = 3",
        "def normalize_answer",
        "def pooled_sd",
        "def evaluate_promotion",
        "def pooled_band_checks",
        "def dose_response_reading",
        "DOSE_RESPONSE_BASELINE",
        "DOSE_RESPONSE_CONSEQUENCES",
        '"single_kind_dose_no_per_kind_split"',
        '"pooled_k3"',
        "no_absolute_per_kind_floors",
    )
    required_bench = (
        'FROZEN_NAME = "pilot"',
        'FROZEN_TIER = "medium"',
        "FROZEN_THINK_BUDGET = 1024",
        "FROZEN_SEED = 78158",
        "def ledger_plan",
        "def is_closed_record",
        "def reconcile_crashed_summary",
        "def stale_event_files",
        "def _valid_score",
        "def pilot_gate",
        "def goal_gate_reading",
        "PASS_BENCHMARK_EVENT",
        "POWER_STATEMENT",
        '"summary", "summary_sha256", "receipts",',
        'BASE_MERGE_RECEIPT_SHA256 = (\n    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"\n)',
    )
    required_lineage = (
        "def load_manifest",
        "def verify_inputs",
        "def rebuild",
        "CANDIDATE_SEED = 71",
        '"candidate_stage"',
        "pending_fill",
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
    if any(value not in lineage for value in required_lineage):
        raise ValueError("frozen lineage-rebuilder contract changed")
    forbidden = "benchmarks" + "/"
    if any(forbidden in text for text in (trial, merge, harness, gate, bench, lineage)):
        raise ValueError("benchmark content leaked into the lifecycle scripts")


def build_receipt() -> dict:
    manifest = check_frozen_corpora()
    check_parent_identity()
    check_lineage_package_static()
    check_lifecycle_contracts()
    banned = tuple(feedloop.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "menders_dose_scale_design",
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
            "stream_slot_matching_namespace": 55140,
            "training": 71,
            "local_gate": LOCAL_SEED,
            "retention_screens": list(SCREEN_SEEDS),
            "seed_freshness": {
                "pinned": [
                    CONSTRUCTION_SEED,
                    55140,
                    71,
                    LOCAL_SEED,
                    *SCREEN_SEEDS,
                    AGGREGATE_SEED,
                ],
                "substitution_required": False,
                "verified": (
                    "77150/55140/88037/88038/88039/88040/78158 verified "
                    "grep-fresh in seed contexts across experiments/, "
                    "knowledge/, research_programs/, scripts/, configs/, and "
                    "docs/ at design time (zero seed-context hits). Training "
                    "seed 71 verified fresh in the qwen35_4b lineage's "
                    "training-seed contexts (prior training seeds: 42/43/44/"
                    "47/51/55 chain stages, 67 for the reference and "
                    "statechain cells' fresh adapters); its only seed-context "
                    "grep hits are run artifacts of the retired "
                    "sparse_support_memory_executor track and one unrelated "
                    "scorer unit test's run_seed — the same artifact class "
                    "the seed-67 audit accepted — recorded here rather than "
                    "substituted"
                ),
                "rule_if_collision": "next free integer, recorded here",
            },
            "aggregate_sealed": AGGREGATE_SEED,
        },
        "corpora": {
            "manifest_sha256": MANIFEST_SHA256,
            "treatment": {
                "path": TREATMENT_PATH.relative_to(EXP).as_posix(),
                "rows": 800,
                "sha256": TREATMENT_SHA256,
                "kinds": manifest["corpus"]["kinds"],
                "surfaces": manifest["corpus"]["surfaces"],
                "balance": manifest["balance"],
                "construction": {
                    "generator": "scripts/gen_feedloop_curriculum.py",
                    "mix": feedloop.ARM_MIX,
                    "seed": CONSTRUCTION_SEED,
                    "builder": "scripts/build_corpus.py",
                },
                "dose_scale_contract": manifest["dose_scale_contract"],
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
                "extended_with_statechain_surface_pools": True,
                "retained_feedloop_surfaces_not_banned": True,
            },
        },
        "arms": {
            "trained_control": "replay_ctl3",
            "trained_candidate": "feedloop_scale",
            "control_trains_first": True,
            "parent_eval_label": PARENT_EVAL_LABEL,
            "gate_arms": list(GATE_ARMS),
        },
        "stream_geometry": {
            "rows_per_arm": 2280,
            "shared_core_replay_rows": 1280,
            "core_stratified_by": ["family", "kind"],
            "variable_block_rows": 1000,
            "blocks": {
                "replay_ctl3": {"replay_slots": 1000},
                "feedloop_scale": {"treatment_rows": 800, "replay_filler_rows": 200},
            },
            "pool_bind": {
                "replay_pool_rows": 2240,
                "control_arm_max_multiplicity": 2,
                "control_block_draws_from_full_pool": True,
                "repeats_solver_minimized": True,
                "candidate_arm_duplicate_free": True,
                "reason": (
                    "1,280 core + 1,000 distinct control rows exceed the "
                    "2,240-row pool and the treatment's long answer spans "
                    "are unreachable from the 960 non-core rows alone; no "
                    "replay row is seen more than twice in the control "
                    "arm's epoch"
                ),
            },
            "slot_order_seed": 55140,
            "exact_match_axes": ["forward", "nonzero_target", "absolute_loss_mass_x5"],
            "match_limit": 0,
            "solver": {
                "method": "scipy.optimize.milp",
                "backend": "highs",
                "mip_rel_gap": 0,
                "time_limit_seconds": 600,
                "objective": "minimize arm-level repeated control rows",
                "filler_and_control_solved_jointly": True,
            },
        },
        "training_plan": {
            "training_authorized": False,
            "epochs": 1,
            "batch_size": 1,
            "grad_accum": 8,
            "optimizer_steps": 285,
            "lr": 1e-5,
            "rank": 32,
            "alpha": 64,
            "w_think": 0.2,
            "w_close": 0.2,
            "max_length": 4096,
            "seed": 71,
            "rows_per_arm": 2280,
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
                    "per_surface": 5,
                    "surfaces": list(feedloop.FEEDLOOP_FORMALISMS),
                    "generator": "scripts/gen_feedloop_curriculum.py",
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
            "dose_response_reading": {
                "gating": False,
                "baseline": dict(DOSE_RESPONSE_BASELINE),
                "this_cell_dose_rows": 800,
                "dose_multiple": 10,
                "rendered_per_formalism": True,
                "consequence_statements": dict(DOSE_RESPONSE_CONSEQUENCES),
                "recorded_either_way": True,
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
                "replay_ctl3",
                "feedloop_scale",
            ],
            "pilot_gate": (
                "candidate aggregate strictly > base AND > replay_ctl3 AND > "
                "hygiene_explore_parent"
            ),
            "goal_gate": (
                "all ten public families strictly > base; recorded from the "
                "same event either way, never part of the pilot pass"
            ),
            "ledger": {
                "write_ahead_opened_record": True,
                "closed_record_pins_summary_and_all_four_receipt_sha256s": True,
                "closed_refuses_forever": True,
                "unopened_requires_clean_slate": True,
                "crashed_summary_recovery": (
                    "byte-identical deterministic regeneration; divergence "
                    "refuses with both digests"
                ),
            },
            "power_statement": (
                "menders alone gates the all-families goal (nine families "
                "hold vs base on every sealed seed; the ties are 0-margin); "
                "three small-dose pedagogies failed at it and dose scale is "
                "the one permitted mechanism class, so menders > 0 for the "
                "candidate on this seed is the reading of consequence; any "
                "10/10 feeds a fresh confirmation cell (independent seeds + "
                "matched compute) before any claim"
            ),
        },
        "lineage_package": {
            "directive": (
                "standalone-reproducibility gate (AGENTS.md non-negotiables; "
                "docs/quality_gates.md, owner directive 2026-07-15): the "
                "confirmation cell's six-stage package copied byte-identically "
                "and extended with this cell's stage 7 (the candidate's own "
                "training)"
            ),
            "manifest": "data/lineage/lineage_manifest.json",
            "manifest_bytes_not_pinned_reason": (
                "the candidate stage's produced fields are orchestrator-"
                "filled TODO-PINs after train-candidate/merge-arms publish; "
                "schema, chaining, and every copy are enforced fail-closed by "
                "rebuild_lineage.py --verify-inputs at smoke and test time"
            ),
            "datasets": {
                name: {"sha256": digest, "rows": rows}
                for name, (digest, rows) in sorted(LINEAGE_DATASETS.items())
            },
            "candidate_dataset": {
                "path": LINEAGE_CANDIDATE_DATASET[0],
                "sha256": LINEAGE_CANDIDATE_DATASET[1],
                "rows": LINEAGE_CANDIDATE_DATASET[2],
            },
            "trainers": dict(sorted(LINEAGE_TRAINERS.items())),
            "chain_training_seeds": list(LINEAGE_TRAINING_SEEDS),
            "candidate_training_seed": 71,
            "root_adapter": {
                "vendored_path": LINEAGE_ROOT.relative_to(ROOT).as_posix(),
                "files": {
                    name: {"sha256": digest, "size": size}
                    for name, (digest, size) in sorted(LINEAGE_ROOT_FILES.items())
                },
                "provenance_boundary": (
                    "HARD BOUNDARY: no committed creation receipt exists for "
                    "the frozen 'blend' root adapter anywhere in the "
                    "repository; it is vendored by bytes into this cell's own "
                    "artifact storage and is NOT reconstructable from "
                    "committed receipts"
                ),
            },
            "rebuild": "scripts/rebuild_lineage.py",
            "fast_verification": "scripts/rebuild_lineage.py --verify-inputs",
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
            "feedloop_curriculum_generator_sha256": sha256_file(
                EXP / "scripts" / "gen_feedloop_curriculum.py"
            ),
            "corpus_builder_sha256": sha256_file(EXP / "scripts" / "build_corpus.py"),
            "trainer_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
            "rebuild_lineage_sha256": sha256_file(EXP / "scripts" / "rebuild_lineage.py"),
            "external_merger_sha256": sha256_file(EXP / "scripts" / "merge_adapter.py"),
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
