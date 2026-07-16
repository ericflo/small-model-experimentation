#!/usr/bin/env python3
"""Freeze the two-instrument, four-input-file local gate and its model inputs.

The frozen gate comprises FOUR model-facing input files per arm:

- AXIS HOLDOUT (installability + the preregistered non-gating dose-response
  reading): 40 rows, all ``u_feedloop``, 5 per formalism across all eight
  (troughline, trinketcord, crankwheel, sigilslate, barrowyoke, balesled,
  millround, skeinreel), from
  gen_feedloop_curriculum.generate_curriculum(HOLDOUT_MIX, 88037); ids
  ``axis88037_*``.
- THREE RETENTION SCREENS (pooled_k3 protocol): 104 rows each, 8 per each of
  the 13 original skills, from the canonical gen_curriculum.py at seeds
  88038/88039/88040; ids ``ret<seed>_*``. All four input seeds were verified
  grep-fresh in seed contexts at design time; no substitution was required.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to every arm and every input file.
Freshness is enforced fail-closed: the four input files are duplicate-free
internally and pairwise, and carry zero canonical-user-message overlap with
every pinned predecessor corpus and training stream, every prior local gate
(seeds 88013-88036, frozen files, INCLUDING the reference cell's and the
statechain cell's gate input files), regenerated prior local seeds, the
regenerated feedloop treatment corpus, and this cell's own materialized
training streams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as curriculum  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    ARMS,
    CANDIDATES,
    DOSE_RESPONSE_BASELINE,
    DOSE_RESPONSE_CONSEQUENCES,
    RETENTION_CAP_BAND,
    RETENTION_CORRECT_BAND,
    RETENTION_PARSED_BAND,
    SCREEN_SEEDS,
    SEED,
)


AGGREGATE_SEED = 78158
CONSTRUCTION_SEED = 77150
AXIS_MIX = feedloop.HOLDOUT_MIX
RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
AXIS_ROWS = 40
AXIS_PER_SURFACE = 5
RETENTION_ROWS = 104
RETENTION_PER_SKILL = 8
ROWS_PER_ARM = AXIS_ROWS + RETENTION_ROWS * len(SCREEN_SEEDS)
INPUT_SEEDS = (SEED, *SCREEN_SEEDS)
SOURCES = {seed: EXP / "data" / f"local_tasks_seed{seed}.jsonl" for seed in INPUT_SEEDS}
RUNNER_INPUTS = {
    seed: EXP / "data" / f"local_input_seed{seed}.jsonl" for seed in INPUT_SEEDS
}
RECEIPT = EXP / "data" / "local_design_receipt.json"
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
# Frozen corpora that exist at design freeze (fail-closed hash pins).
FROZEN_SOURCES = (
    (EXP / "data" / "sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    (EXP / "data" / "sft_feedloop_scale.jsonl",
     "080c3603cd3bfca2b261b797be356f97684aedc4f65becd9b50cbb45706bd2c2", 800),
)
# Every inherited corpus and every materialized training stream in the
# lineages of the published arms (fail-closed hash pins), copied from the
# statechain cell's audited inventory, PLUS the reference cell's and the
# statechain cell's own treatment corpora and materialized streams.
PREDECESSOR_SOURCES = (
    (
        REFERENCE_EXP / "data" / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        REFERENCE_EXP / "data" / "sft_feedloop_state.jsonl",
        "e6d45ed45632f7d6bea8e300f469a4d4c076eb4d8be420d677bd14c27471083b",
        160,
    ),
    (
        REFERENCE_EXP / "data" / "replay_ctl.jsonl",
        "ef6f321f93804081eefb7b8ae3ca84fe39a46127604c69d81b803628d020e63c",
        1520,
    ),
    (
        REFERENCE_EXP / "data" / "feedloop_state.jsonl",
        "187787a8c6b2d652ba46553cecb8fe451b191e277094268f4df921babf621c2b",
        1520,
    ),
    (
        STATECHAIN_EXP / "data" / "sft_statechain_only.jsonl",
        "ab6c78458eb8a41f42ebee25e79354d42beb558f78bb3d348dd7902ca3a9bad3",
        160,
    ),
    (
        STATECHAIN_EXP / "data" / "replay_ctl2.jsonl",
        "0fecf519a8981ea71a9842f0c5a3615cfe60f157e58420d1ddc5e019f44e9477",
        1520,
    ),
    (
        STATECHAIN_EXP / "data" / "statechain_only.jsonl",
        "2e3a08bb9a50f572df4e904fc7a08c6e0b06317972af74c2119de35e3f524eec",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "sft_fresh_designed160.jsonl",
        "e599f1563f6e4fe68aa43ce64bbe450c264170fb07dc17dba9ea74e694f284d5",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "sft_fresh_budget160.jsonl",
        "ecece8e294f0b1c34a086705b05773d6004c6a9a239283aa26ee1c1bbad39800",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "replay_repeat.jsonl",
        "fa5198c2033dc527fa668f55f8a49865d8015a6d14cc059b319aa058dfad74f0",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "designed_fresh.jsonl",
        "6d4dc303bc159c19a1ffd0c60ca7d08ea64b02909366701b345d888482d67f3f",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "budget_commit.jsonl",
        "7dbc084809c6ae6b8dd794465a7057095e02b9d598c414fdad32dee7d361c8d7",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_axis160.jsonl",
        "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "replay_repeat.jsonl",
        "878f501f3c6665982b7b3e9ffb2de675382d6b366b62a61d0a13e6709841c468",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "axis_curriculum.jsonl",
        "e2561485ae28c9c374b663af4a002ab67e02cc75ec8994d1d89770e6b198e98a",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "sft_axis160.jsonl",
        "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "axis160_direct.jsonl",
        "0cbf3abf04781f3217189dd7b2bdf173d83772b0200b283a7c608dc3d386c1ee",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_rank_capacity_vehicle_cell"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_rank_capacity_vehicle_cell"
        / "data"
        / "sft_axis160.jsonl",
        "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
        160,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_rank_capacity_vehicle_cell"
        / "data"
        / "axis160_r64.jsonl",
        "336a044557cc6802aeffce497634528b844329d872f64d1c54ea4a63311c990c",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
        2240,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "sft_hygiene_explore.jsonl",
        "8b3e97919c62cbb0893add281dc1d3ae881aa0138d0d1721043fec26b0c22cf1",
        80,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "hygiene_explore.jsonl",
        "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f",
        1520,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "replay_clean.jsonl",
        "2189d160c9c3007ccbaf935478b3354b597c71945f6dbf1c32bf4e670ace0017",
        1520,
    ),
)
# ALL predecessor experiments' frozen local gates and screens, seeds
# 88013-88036 (fail-closed hash pins): nine gates, the calibration cell's
# four retention-only screens, the reference cell's four gate input files
# (axis 88026 + screens 88027/88028/88030), and the statechain cell's four
# gate input files (axis 88033 + screens 88034/88035/88036).
PREDECESSOR_GATES = (
    (
        ROOT
        / "experiments"
        / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "data"
        / "local_tasks_seed88013.jsonl",
        "be817bd09f80a1fdf045bfb7638634f5f0e0e34ac6a404a34a98283e49dc5c2b",
        104,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "data"
        / "local_tasks_seed88014.jsonl",
        "4753a277bdcfcab515a8bb5ece541498729107de6002f4be011b57b2eed34873",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_replay_stack_medium_target_match"
        / "data"
        / "local_tasks_seed88015.jsonl",
        "1c019d70751a789a17ef555d51862c5a017f0a3c2bb1ebbce41a1a93496fefd1",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_stack_readjudication_medium_pilot"
        / "data"
        / "local_tasks_seed88016.jsonl",
        "13788cc2e6fc237cfb709d2b22ae24dc6263a91f97c4822b856d098f816020b4",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_axis_corpus_v2_staged_repair"
        / "data"
        / "local_tasks_seed88017.jsonl",
        "746cf1c4efae1cdd6614827f6fe8abb0eb7fb8cf5c2dc0c2c7dac4a6d930db81",
        154,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_hygiene_explore_destack_medium"
        / "data"
        / "local_tasks_seed88018.jsonl",
        "597f10a44674cc12e5f499be8de6804bb040985019b18aacc5527339a26857eb",
        124,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_interleaved_replay_dose_medium"
        / "data"
        / "local_tasks_seed88019.jsonl",
        "6e927f591f9ae9d2edad6e263be3f7c0262b39de4854314a64802e656b98c15b",
        124,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_dose_diversity_mechanism_cell"
        / "data"
        / "local_tasks_seed88020.jsonl",
        "8e4dd8b21febf64048fef5cf15541356d4a5808e45caf5d6f19525e154168822",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_rank_capacity_vehicle_cell"
        / "data"
        / "local_tasks_seed88021.jsonl",
        "64754430f3316d1d61fb81727aebbff03c9f32624f38182fbf5407b81e48d707",
        144,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_retention_screen_calibration"
        / "data"
        / "local_tasks_seed88022.jsonl",
        "e41e47d87b31a1f15dcc3b4978f73730b7acfb690b502388ce497bbed5951114",
        104,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_retention_screen_calibration"
        / "data"
        / "local_tasks_seed88023.jsonl",
        "8b8728960780debe45412b13be5245e1311de71ea3cbdcabdd5e4a2a62469d59",
        104,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_retention_screen_calibration"
        / "data"
        / "local_tasks_seed88024.jsonl",
        "f0cd07500e0be9541bb5c356cf09efdc48c2757e91e21366ef8c34d2e90bb24f",
        104,
    ),
    (
        ROOT
        / "experiments"
        / "qwen35_4b_retention_screen_calibration"
        / "data"
        / "local_tasks_seed88025.jsonl",
        "3ca1704960f701382a5c766bc2bdbebc73e30b94113698f0ac39dc7962591f01",
        104,
    ),
    (
        REFERENCE_EXP / "data" / "local_tasks_seed88026.jsonl",
        "079075e0f51dacac9ea6c87eeea4c2d0e5f9258cb05af885d4cc897479d3866f",
        40,
    ),
    (
        REFERENCE_EXP / "data" / "local_tasks_seed88027.jsonl",
        "f568b7d8cd04ce6e4bf270e46a80b4328978ee53a6c3d34a69b3d1c72a77b79f",
        104,
    ),
    (
        REFERENCE_EXP / "data" / "local_tasks_seed88028.jsonl",
        "09b3d71c8b0fd693a21684f60b21dfa125cf3291b24de5878366026c81d34756",
        104,
    ),
    (
        REFERENCE_EXP / "data" / "local_tasks_seed88030.jsonl",
        "9ff826de30c3b32bae1b63af69c3f2a677f6d876cc1046ea22ad41349dcc3511",
        104,
    ),
    (
        STATECHAIN_EXP / "data" / "local_tasks_seed88033.jsonl",
        "11257caea2557d1fbc1ac86f4f8f061987a1e77cdb448b0bda1fa0a1422c4696",
        40,
    ),
    (
        STATECHAIN_EXP / "data" / "local_tasks_seed88034.jsonl",
        "2264646be93e159184602957d21ef5e3ba650ff1cd6743e4e6b15b8d3726b11e",
        104,
    ),
    (
        STATECHAIN_EXP / "data" / "local_tasks_seed88035.jsonl",
        "a8634bac86e5e0ef5185ee7e93ae079c57e4d2c79a7273207e0ed83b515847b4",
        104,
    ),
    (
        STATECHAIN_EXP / "data" / "local_tasks_seed88036.jsonl",
        "6ff076d7ef8a20093cf2a027f36a0fd8cf26d03fe87e72f9e76d1108b0751b21",
        104,
    ),
)
# Training streams materialize independently of this design; the zero-overlap
# check against them runs lazily whenever the files exist (generation and
# --check both call it), and their content is covered by construction: every
# stream row comes from sft_blend.jsonl or the feedloop treatment corpus.
TRAINING_STREAMS = (
    EXP / "data" / "replay_ctl3.jsonl",
    EXP / "data" / "feedloop_scale.jsonl",
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88037))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_LABEL = "hygiene_explore_parent"
PARENT_MERGE_RECEIPT = (
    ROOT
    / "experiments"
    / "qwen35_4b_hygiene_explore_destack_medium"
    / "runs"
    / "merges"
    / "hygiene_explore.json"
)
PARENT_MERGED = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_hygiene_explore_destack_medium"
    / "merged"
    / "hygiene_explore"
)
EXPECTED_PARENT_MERGE_RECEIPT_SHA256 = (
    "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
)
EXPECTED_PARENT_TREE_SHA256 = (
    "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
)
EXPECTED_PARENT_WEIGHTS_SHA256 = (
    "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
)
EXPECTED_PARENT_FILES = [
    {
        "name": "chat_template.jinja",
        "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "size": 7756,
    },
    {
        "name": "config.json",
        "sha256": "a1c80f0efa6f83f631eaa9c25ffa166e3b1f9db395cc3b14374dfc0962261f60",
        "size": 2829,
    },
    {
        "name": "generation_config.json",
        "sha256": "0c46d8aa4f0ae5e611c961f70b87c83fb696043c1e319337708e96f882180de1",
        "size": 116,
    },
    {
        "name": "merge_receipt.json",
        "sha256": "9eafc8b0b01b125c3cb2eaa3cf8b808ecb7b7ed40172cbbb2f92117be75995a9",
        "size": 876,
    },
    {
        "name": "model.safetensors",
        "sha256": "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f",
        "size": 9078620536,
    },
    {
        "name": "tokenizer.json",
        "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "size": 19989325,
    },
    {
        "name": "tokenizer_config.json",
        "sha256": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
        "size": 1125,
    },
]
MERGED_FILE_NAMES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merge_receipt.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
# Files pinned by hash into the frozen design receipt. eval_local_vllm.py,
# run_benchmark.py, materialize_streams.py, train_trial.py, and
# check_design.py are deliberately NOT pinned here: they carry
# orchestrator-filled TODO-PIN constants that change after this receipt
# freezes; each is authenticated by hash inside its own run receipt instead.
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "feedloop_curriculum": EXP / "scripts" / "gen_feedloop_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "harness": EXP / "scripts" / "run.py",
    "runner": EXP / "src" / "vllm_runner.py",
    # merge_trained_arm.py carries no orchestrator-filled constants, so it is
    # pinned here and demands this exact pin back (code_sha256.merge) before
    # any composite is produced.
    "merge": EXP / "scripts" / "merge_trained_arm.py",
    # Standalone: the external composite merger is copied INTO this cell.
    "external_merger": EXP / "scripts" / "merge_adapter.py",
    "rebuild_lineage": EXP / "scripts" / "rebuild_lineage.py",
}
CODE_PINS_DEFERRED = [
    "scripts/eval_local_vllm.py",
    "scripts/run_benchmark.py",
    "scripts/materialize_streams.py",
    "scripts/train_trial.py",
    "scripts/check_design.py",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl_bytes(rows: list[dict]) -> bytes:
    return "".join(
        json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n" for row in rows
    ).encode()


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def load_frozen(path: Path, expected_sha256: str | None, expected_rows: int) -> list[dict]:
    if expected_sha256 is None:
        raise ValueError(f"frozen source pin is unfilled (TODO-PIN): {path}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen source changed: {path}")
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]
    if len(rows) != expected_rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"unexpected frozen source rows: {path}")
    return rows


def merged_tree_manifest(output: Path) -> list[dict]:
    """Hash the complete, flat merged-composite tree and reject surprises."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"merged composite is not a real directory: {output}")
    children = sorted(output.iterdir(), key=lambda path: path.name)
    if any(path.is_symlink() or not path.is_file() for path in children):
        raise ValueError("merged composite contains a symlink or nested/non-file entry")
    names = {path.name for path in children}
    if names != MERGED_FILE_NAMES:
        raise ValueError(
            "merged composite file set changed: "
            f"missing={sorted(MERGED_FILE_NAMES - names)}, "
            f"unexpected={sorted(names - MERGED_FILE_NAMES)}"
        )
    return [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in children
    ]


def tree_manifest_sha256(manifest: list[dict]) -> str:
    rendered = json.dumps(
        manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    return sha256_bytes(rendered)


def build_axis_rows() -> tuple[list[dict], list[dict]]:
    axis_rows = feedloop.generate_curriculum(AXIS_MIX, SEED)
    for row in axis_rows:
        row["task_id"] = f"axis{SEED}_{row['task_id']}"
    axis_summary = feedloop.validate_generated(axis_rows)
    feedloop.check_banned_vocabulary(axis_rows)
    expected_axis_kinds = {"u_feedloop": AXIS_ROWS}
    expected_axis_surfaces = {
        formalism: AXIS_PER_SURFACE
        for formalism in feedloop.FEEDLOOP_FORMALISMS
    }
    if (
        axis_summary["rows"] != AXIS_ROWS
        or axis_summary["kinds"] != expected_axis_kinds
        or axis_summary["surfaces"] != expected_axis_surfaces
    ):
        raise ValueError("axis holdout no longer has five rows per formalism")
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": SEED,
                "instrument": "axis_holdout",
            },
        }
        for row in axis_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return axis_rows, runner_rows


def build_screen(seed: int) -> tuple[list[dict], list[dict]]:
    retention_rows = curriculum.generate_curriculum(RETENTION_MIX, seed)
    for row in retention_rows:
        row["task_id"] = f"ret{seed}_{row['task_id']}"
    retention_summary = curriculum.validate_generated(retention_rows)
    expected_retention_kinds = {
        f"u_{name}": RETENTION_PER_SKILL for name in curriculum.SKILLS
    }
    if (
        retention_summary["rows"] != RETENTION_ROWS
        or retention_summary["kinds"] != expected_retention_kinds
    ):
        raise ValueError(
            f"retention screen {seed} no longer has eight rows per registered skill"
        )
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": seed,
                "instrument": "retention",
            },
        }
        for row in retention_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("local runner input schema leaked hidden fields")
    return retention_rows, runner_rows


def check_training_streams(local_messages: set[bytes]) -> list[str]:
    """Lazily enforce zero prompt overlap against every existing stream file."""
    present: list[str] = []
    for path in TRAINING_STREAMS:
        if not path.is_file():
            continue
        stream_messages = {
            message_bytes(row) for row in load_jsonl(path) if row.get("messages")
        }
        if local_messages & stream_messages:
            raise ValueError(f"fresh local prompts overlap a training stream: {path}")
        present.append(path.relative_to(EXP).as_posix())
    return present


def overlap_receipt(source_rows_by_seed: dict[int, list[dict]]) -> dict:
    messages_by_seed: dict[int, set[bytes]] = {}
    for seed, rows in source_rows_by_seed.items():
        expected = AXIS_ROWS if seed == SEED else RETENTION_ROWS
        messages = {message_bytes(row) for row in rows}
        if len(messages) != expected:
            raise ValueError(f"input file {seed} collides on canonical user messages")
        messages_by_seed[seed] = messages
    local_messages: set[bytes] = set()
    for seed in INPUT_SEEDS:
        local_messages |= messages_by_seed[seed]
    if len(local_messages) != ROWS_PER_ARM:
        raise ValueError("gate instruments collide on canonical user messages")
    cross_screen: dict[str, int] = {}
    for index, first in enumerate(INPUT_SEEDS):
        for second in INPUT_SEEDS[index + 1 :]:
            overlap = len(messages_by_seed[first] & messages_by_seed[second])
            if overlap:
                raise ValueError(
                    f"input files {first} and {second} share canonical user messages"
                )
            cross_screen[f"{first}x{second}"] = overlap
    frozen_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in FROZEN_SOURCES:
        messages = {
            message_bytes(row)
            for row in load_frozen(path, expected_sha256, expected_rows)
            if row.get("messages")
        }
        overlap = len(local_messages & messages)
        if overlap:
            raise ValueError(f"fresh local prompts overlap frozen corpus: {path}")
        frozen_sources[path.relative_to(EXP).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    predecessor_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in PREDECESSOR_SOURCES:
        messages = {
            message_bytes(row)
            for row in load_frozen(path, expected_sha256, expected_rows)
            if row.get("messages")
        }
        overlap = len(local_messages & messages)
        if overlap:
            raise ValueError(
                f"fresh local prompts overlap an inherited corpus or stream: {path}"
            )
        predecessor_sources[path.relative_to(ROOT).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    # Regenerated feedloop training rows at the frozen construction seed:
    # covers the treatment corpus by construction and pins the generator
    # behavior.
    feedloop_training_messages = {
        message_bytes(row)
        for row in feedloop.generate_curriculum(
            feedloop.ARM_MIX, CONSTRUCTION_SEED
        )
    }
    feedloop_training_overlap = len(local_messages & feedloop_training_messages)
    if feedloop_training_overlap:
        raise ValueError(
            "fresh local prompts overlap regenerated feedloop training rows"
        )
    prior_messages = {
        message_bytes(row)
        for seed in PRIOR_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(curriculum.SMOKE_MIX, seed)
    }
    prior_local_overlap = len(local_messages & prior_messages)
    if prior_local_overlap:
        raise ValueError("fresh local prompts overlap prior local seeds")
    predecessor_gates: dict[str, dict] = {}
    for gate_path, gate_sha256, gate_rows in PREDECESSOR_GATES:
        predecessor_messages = {
            message_bytes(row)
            for row in load_frozen(gate_path, gate_sha256, gate_rows)
            if row.get("messages")
        }
        predecessor_overlap = len(local_messages & predecessor_messages)
        if predecessor_overlap:
            raise ValueError(
                f"fresh local prompts overlap a predecessor's frozen gate: {gate_path}"
            )
        predecessor_gates[gate_path.relative_to(ROOT).as_posix()] = {
            "sha256": gate_sha256,
            "messages_compared": len(predecessor_messages),
            "overlap": predecessor_overlap,
        }
    streams_present = check_training_streams(local_messages)
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "unique_local_messages_per_input": {
            str(seed): len(messages_by_seed[seed]) for seed in INPUT_SEEDS
        },
        "cross_screen_overlap": cross_screen,
        "frozen_sources": frozen_sources,
        "predecessor_sources": predecessor_sources,
        "regenerated_feedloop_training": {
            "mix": feedloop.ARM_MIX,
            "construction_seed": CONSTRUCTION_SEED,
            "messages_compared": len(feedloop_training_messages),
            "overlap": feedloop_training_overlap,
        },
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "predecessor_gates": predecessor_gates,
        "training_streams": {
            "paths": [path.relative_to(EXP).as_posix() for path in TRAINING_STREAMS],
            "present_at_generation_or_check": streams_present,
            "checked_lazily_when_present": True,
            "policy": (
                "zero canonical user-message overlap enforced against every "
                "stream file that exists, at generation time and at --check time"
            ),
            "covered_by_construction": (
                "streams are composed solely of sft_blend.jsonl replay rows and "
                "the feedloop treatment corpus, both checked above"
            ),
        },
    }


def build_outputs(*, authenticate_parent: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    if (
        not PARENT_MERGE_RECEIPT.is_file()
        or sha256_file(PARENT_MERGE_RECEIPT) != EXPECTED_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(f"published prerequisite changed: {PARENT_MERGE_RECEIPT}")
    parent_receipt = json.loads(PARENT_MERGE_RECEIPT.read_text(encoding="utf-8"))
    if (
        parent_receipt.get("name") != "hygiene_explore"
        or parent_receipt.get("model_id") != MODEL_ID
        or parent_receipt.get("model_revision") != MODEL_REVISION
        or Path(parent_receipt.get("merged", "")).resolve() != PARENT_MERGED.resolve()
        or parent_receipt.get("output_tree_sha256") != EXPECTED_PARENT_TREE_SHA256
        or parent_receipt.get("output_files") != EXPECTED_PARENT_FILES
        or parent_receipt.get("weight_files")
        != [{"name": "model.safetensors", "sha256": EXPECTED_PARENT_WEIGHTS_SHA256}]
    ):
        raise ValueError("published hygiene_explore-parent merge receipt violates pins")
    if authenticate_parent:
        parent_files = merged_tree_manifest(PARENT_MERGED)
        if (
            parent_files != EXPECTED_PARENT_FILES
            or tree_manifest_sha256(parent_files) != EXPECTED_PARENT_TREE_SHA256
        ):
            raise ValueError("published hygiene_explore-parent composite tree changed")
    source_rows_by_seed: dict[int, list[dict]] = {}
    runner_rows_by_seed: dict[int, list[dict]] = {}
    source_rows_by_seed[SEED], runner_rows_by_seed[SEED] = build_axis_rows()
    for seed in SCREEN_SEEDS:
        source_rows_by_seed[seed], runner_rows_by_seed[seed] = build_screen(seed)
    sources = {seed: jsonl_bytes(source_rows_by_seed[seed]) for seed in INPUT_SEEDS}
    runner_inputs = {
        seed: jsonl_bytes(runner_rows_by_seed[seed]) for seed in INPUT_SEEDS
    }
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "menders_dose_scale_local_gate_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seed": SEED,
        "screen_seeds": list(SCREEN_SEEDS),
        "seed_freshness": {
            "input_seeds": list(INPUT_SEEDS),
            "substitution_required": False,
            "verified": (
                "88037/88038/88039/88040 verified grep-fresh in seed contexts "
                "across experiments/, knowledge/, research_programs/, scripts/, "
                "configs/, and docs/ at design time (zero seed-context hits); "
                "88033-88036 are the statechain cell's frozen gate seeds and "
                "sit immediately below this cell's block"
            ),
            "rule_if_collision": "next free integer, recorded here",
        },
        "aggregate_seed": AGGREGATE_SEED,
        "rows_per_arm": ROWS_PER_ARM,
        "instruments": {
            "axis_holdout": {
                "generator": "scripts/gen_feedloop_curriculum.py",
                "mix": AXIS_MIX,
                "rows": AXIS_ROWS,
                "kinds": {"u_feedloop": AXIS_ROWS},
                "per_surface": AXIS_PER_SURFACE,
                "surfaces": list(feedloop.FEEDLOOP_FORMALISMS),
                "id_prefix": f"axis{SEED}_",
                "seed": SEED,
            },
            "retention": {
                "generator": "scripts/gen_curriculum.py",
                "mix": RETENTION_MIX,
                "rows_per_screen": RETENTION_ROWS,
                "per_kind": RETENTION_PER_SKILL,
                "screens": len(SCREEN_SEEDS),
                "id_prefixes": [f"ret{seed}_" for seed in SCREEN_SEEDS],
                "adjudication": "pooled_k3",
            },
        },
        "kinds": {
            str(seed): dict(
                sorted(
                    Counter(row["kind"] for row in source_rows_by_seed[seed]).items()
                )
            )
            for seed in INPUT_SEEDS
        },
        "sources": {
            str(seed): {
                "path": SOURCES[seed].relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(sources[seed]),
                "contains_executable_truth": True,
            }
            for seed in INPUT_SEEDS
        },
        "runner_inputs": {
            str(seed): {
                "path": RUNNER_INPUTS[seed].relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(runner_inputs[seed]),
                "schema": ["id", "messages", "meta"],
                "contains_answer": False,
                "contains_oracle": False,
            }
            for seed in INPUT_SEEDS
        },
        "answer_normalization": ANSWER_NORMALIZATION,
        "freshness": overlap_receipt(source_rows_by_seed),
        "backend": {
            "name": "vllm_merged_composite",
            "thinking": "natural",
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": 1024,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "same_runner_and_geometry_for_every_arm": True,
            "runtime_lora_forbidden": True,
        },
        "arms": list(ARMS),
        "candidates": list(CANDIDATES),
        "run_order": {
            "policy": (
                "arm-major: for each arm in the frozen order, the four input "
                "files ascending by seed — 12 authenticated engine events"
            ),
            "sequence": [
                f"{label}_seed{seed}" for label in ARMS for seed in INPUT_SEEDS
            ],
        },
        "gates": {
            "axis_total_strictly_beats_parent_and_replay": True,
            "single_kind_dose_no_per_kind_split": True,
            "per_surface_reported_not_gated": True,
            "retention_adjudication": "pooled_k3_pooled_mean_over_three_screens",
            "retention_correct_band": RETENTION_CORRECT_BAND,
            "retention_cap_contact_band": RETENTION_CAP_BAND,
            "retention_parsed_band": RETENTION_PARSED_BAND,
            "bands_apply_to_pooled_means_not_per_screen": True,
            "bands_evaluated_on_pooled_sums_times_screens": True,
            "no_absolute_per_kind_floors": True,
            "no_passing_candidate_keeps_aggregate_seed_sealed": True,
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
        "prerequisites": {
            "parent_merge_receipt": (
                PARENT_MERGE_RECEIPT.relative_to(ROOT).as_posix()
            ),
            "parent_merge_receipt_sha256": EXPECTED_PARENT_MERGE_RECEIPT_SHA256,
            "parent_merge_files": EXPECTED_PARENT_FILES,
            "parent_merge_tree_sha256": EXPECTED_PARENT_TREE_SHA256,
            "parent_weights_sha256": EXPECTED_PARENT_WEIGHTS_SHA256,
            "parent_eval_label": PARENT_LABEL,
            "arm_training_and_merge_receipts": (
                "deferred: authenticated at merge/eval time via "
                "merge_trained_arm.validate_published_merge"
            ),
        },
        "code_sha256": {
            name: sha256_file(path) for name, path in sorted(CODE_FILES.items())
        },
        "code_pins_deferred": {
            "reason": (
                "these files carry orchestrator-filled TODO-PIN constants; each "
                "is authenticated by hash inside its own run receipt"
            ),
            "files": CODE_PINS_DEFERRED,
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "aggregate_seed_sealed": True,
        },
        "next_authorized_stage": "train-control",
    }
    rendered_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    outputs: dict[Path, bytes] = {}
    for seed in INPUT_SEEDS:
        outputs[SOURCES[seed]] = sources[seed]
        outputs[RUNNER_INPUTS[seed]] = runner_inputs[seed]
    outputs[RECEIPT] = rendered_receipt
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = build_outputs()
    if args.check:
        changed = [
            path
            for path, value in outputs.items()
            if not path.is_file() or path.read_bytes() != value
        ]
        if changed:
            parser.error(
                "local-gate artifacts are absent or changed: "
                + ", ".join(map(str, changed))
            )
    else:
        existing = [path for path in outputs if path.exists()]
        if existing:
            parser.error(
                "refusing to overwrite local-gate artifacts: "
                + ", ".join(map(str, existing))
            )
        for path, value in outputs.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(
        json.dumps(
            {
                "rows_per_arm": ROWS_PER_ARM,
                "sources_sha256": {
                    str(seed): sha256_bytes(outputs[SOURCES[seed]])
                    for seed in INPUT_SEEDS
                },
                "runner_inputs_sha256": {
                    str(seed): sha256_bytes(outputs[RUNNER_INPUTS[seed]])
                    for seed in INPUT_SEEDS
                },
                "receipt_sha256": sha256_bytes(outputs[RECEIPT]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
