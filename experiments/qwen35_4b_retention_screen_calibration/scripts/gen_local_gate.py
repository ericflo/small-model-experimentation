#!/usr/bin/env python3
"""Freeze the four retention-only calibration screens and their model inputs.

This is a pure eval calibration cell: FIVE already-published composites are
re-measured on FOUR fresh retention-only screens, nothing trains, nothing
merges, nothing promotes, and there is no aggregate seed. The frozen design
comprises, for each screen seed 88022/88023/88024/88025:

- RETENTION SCREEN: 104 rows, 8 per each of the 13 original skills, from
  gen_curriculum.generate_curriculum at that seed; ids ``ret<seed>_*``.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to every arm and every screen. Every
composite pin is recomputed here against the committed merge receipts; there
is no deferred pin and no merge script — the pinned code set is the
generator, curriculum, gate, eval, runner, and harness.
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
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    ARMS,
    BAND_MINIMUM,
    HISTORICAL_READINGS,
    PARENT,
    POOLED_K2_MAX_SD,
    SINGLE_SCREEN_MAX_SD,
)


SEEDS = (88022, 88023, 88024, 88025)
RETENTION_MIX = ",".join(f"{name}=8" for name in curriculum.SKILLS)
ROWS_PER_SCREEN = 104
RETENTION_PER_SKILL = 8
TOTAL_ROWS = ROWS_PER_SCREEN * len(SEEDS)
SOURCES = {seed: EXP / "data" / f"local_tasks_seed{seed}.jsonl" for seed in SEEDS}
RUNNER_INPUTS = {
    seed: EXP / "data" / f"local_input_seed{seed}.jsonl" for seed in SEEDS
}
RECEIPT = EXP / "data" / "local_design_receipt.json"
# Every inherited corpus and every materialized training stream in the
# lineages of the five published arms (fail-closed hash pins): the clean
# parent's fresh-surface corpora and streams, the goal-gap axis lineage, the
# dose-diversity rank-32 lineage, the rank-capacity rank-64 lineage, and the
# hygiene-explore de-stack lineage.
PREDECESSOR_SOURCES = (
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
# ALL NINE predecessor experiments' frozen local gates (fail-closed hash
# pins): the fresh-surface 104-task seed-88013 gate, the goal-gap 144-task
# seed-88014 gate, the stack trial's 144-task seed-88015 gate, the
# re-adjudication's 144-task seed-88016 gate, the axis-v2 staged-repair
# 154-task seed-88017 gate, the de-stack trial's 124-task seed-88018 gate,
# the interleaved-dose trial's 124-task seed-88019 gate, the dose-diversity
# mechanism cell's 144-task seed-88020 gate, and the rank-capacity cell's
# 144-task seed-88021 gate.
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
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88022))
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# Every arm is an inherited, externally published composite; every pin is
# filled at design freeze — nothing is deferred and no TODO-PIN exists.
PARENT_EXP_NAME = "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
DESTACK_EXP_NAME = "qwen35_4b_hygiene_explore_destack_medium"
DOSE_EXP_NAME = "qwen35_4b_dose_diversity_mechanism_cell"
RANK_EXP_NAME = "qwen35_4b_rank_capacity_vehicle_cell"
COMPOSITE_RECEIPTS = {
    "axis160_direct": (
        ROOT / "experiments" / DOSE_EXP_NAME / "runs" / "merges"
        / "axis160_direct.json"
    ),
    "axis160_r64": (
        ROOT / "experiments" / RANK_EXP_NAME / "runs" / "merges"
        / "axis160_r64.json"
    ),
    "clean_parent": (
        ROOT / "experiments" / PARENT_EXP_NAME / "runs" / "merges"
        / "designed_fresh.json"
    ),
    "hygiene_explore_direct": (
        ROOT / "experiments" / DESTACK_EXP_NAME / "runs" / "merges"
        / "hygiene_explore.json"
    ),
    "replay_clean": (
        ROOT / "experiments" / DESTACK_EXP_NAME / "runs" / "merges"
        / "replay_clean.json"
    ),
}
MERGED = {
    "axis160_direct": (
        ROOT / "large_artifacts" / DOSE_EXP_NAME / "merged" / "axis160_direct"
    ),
    "axis160_r64": (
        ROOT / "large_artifacts" / RANK_EXP_NAME / "merged" / "axis160_r64"
    ),
    "clean_parent": (
        ROOT / "large_artifacts" / PARENT_EXP_NAME / "merged" / "designed_fresh"
    ),
    "hygiene_explore_direct": (
        ROOT / "large_artifacts" / DESTACK_EXP_NAME / "merged" / "hygiene_explore"
    ),
    "replay_clean": (
        ROOT / "large_artifacts" / DESTACK_EXP_NAME / "merged" / "replay_clean"
    ),
}
EXPECTED_RECEIPT_SHA256 = {
    "axis160_direct": "7b878bb357e044c58a5ba27f34365906059259237e657ca77c2ad2e8fb77ea39",
    "axis160_r64": "bf0032ea7e9d11c819812f9a54025fce8b23c0921032b186098fdc83a77c5e40",
    "clean_parent": "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2",
    "hygiene_explore_direct": "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
    "replay_clean": "24367084da5415ec8c3f922202a2028cc2930c4b99d87ea5e32b93e5c3b90332",
}
EXPECTED_COMPOSITE_NAMES = {
    "axis160_direct": "axis160_direct",
    "axis160_r64": "axis160_r64",
    "clean_parent": "designed_fresh",
    "hygiene_explore_direct": "hygiene_explore",
    "replay_clean": "replay_clean",
}
EXPECTED_TREE_SHA256 = {
    "axis160_direct": "b773fd439eca753e70d3b16862497ee0ba9783cdb024ddff84b4595d10b7da61",
    "axis160_r64": "ebfa63d9a9fcd93149e81c0d443dcfc80ed67690eb8a9593de5f62131a7a3074",
    "clean_parent": "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255",
    "hygiene_explore_direct": "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
    "replay_clean": "19759e12b1301a15a2f9b2db311ffff7c08e3d8b0d3237c9e7cd718fa8dc7f67",
}
EXPECTED_WEIGHTS_SHA256 = {
    "axis160_direct": "3cd4272ee386dc3b8e878ef3b2ac9ebf94dd8d47ef7d9d480f921e7e45b37279",
    "axis160_r64": "707517ff6b7981f8c1a4446484641760f8b07b42e2721852b33208480a97b2a1",
    "clean_parent": "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
    "hygiene_explore_direct": "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f",
    "replay_clean": "2cef3e5e7ddfbee5d2d2c3a878d64f360dc6994764b9856be7e319ce5187d0b4",
}
EXPECTED_FILES = {
    "axis160_direct": [
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
            "sha256": "a32f0d334611321f2ff4c3a6252a35fd1142da0c472c1af895ca9dd96b0f48f1",
            "size": 875,
        },
        {
            "name": "model.safetensors",
            "sha256": "3cd4272ee386dc3b8e878ef3b2ac9ebf94dd8d47ef7d9d480f921e7e45b37279",
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
    ],
    "axis160_r64": [
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
            "sha256": "4a7fc7ef65991844974d5dc7669d1f6e53be9c683a5b3cce9854252258fac22c",
            "size": 970,
        },
        {
            "name": "model.safetensors",
            "sha256": "707517ff6b7981f8c1a4446484641760f8b07b42e2721852b33208480a97b2a1",
            "size": 9078620536,
        },
        {
            "name": "tokenizer.json",
            "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
            "size": 19989325,
        },
        {
            "name": "tokenizer_config.json",
            "sha256": "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87",
            "size": 1123,
        },
    ],
    "clean_parent": [
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
            "sha256": "10b0de69bbda6edf8e216f45667b15f73557102f0661461fec620786e8a9170f",
            "size": 897,
        },
        {
            "name": "model.safetensors",
            "sha256": "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
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
    ],
    "hygiene_explore_direct": [
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
    ],
    "replay_clean": [
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
            "sha256": "aac945fb0d38f157d9c35518fb4c5de99cc5cac0ace3a280202f9dc038d77186",
            "size": 875,
        },
        {
            "name": "model.safetensors",
            "sha256": "2cef3e5e7ddfbee5d2d2c3a878d64f360dc6994764b9856be7e319ce5187d0b4",
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
    ],
}
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
# Files pinned by hash into the frozen design receipt. There is no merge
# script and no deferred pin: eval_local_vllm.py carries no
# orchestrator-filled constant in this cell, so the complete code surface of
# the local event is frozen here.
CODE_FILES = {
    "generator": Path(__file__),
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "eval": EXP / "scripts" / "eval_local_vllm.py",
    "harness": EXP / "scripts" / "run.py",
    "runner": EXP / "src" / "vllm_runner.py",
}


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


def load_frozen(path: Path, expected_sha256: str, expected_rows: int) -> list[dict]:
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


def authenticate_composite_receipt(label: str) -> dict:
    """Authenticate one inherited composite's committed merge receipt."""
    receipt_path = COMPOSITE_RECEIPTS[label]
    if (
        not receipt_path.is_file()
        or sha256_file(receipt_path) != EXPECTED_RECEIPT_SHA256[label]
    ):
        raise ValueError(f"published prerequisite receipt changed: {receipt_path}")
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (
        payload.get("name") != EXPECTED_COMPOSITE_NAMES[label]
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != MERGED[label].resolve()
        or payload.get("output_tree_sha256") != EXPECTED_TREE_SHA256[label]
        or payload.get("output_files") != EXPECTED_FILES[label]
        or payload.get("weight_files")
        != [{"name": "model.safetensors", "sha256": EXPECTED_WEIGHTS_SHA256[label]}]
    ):
        raise ValueError(f"published prerequisite receipt violates pins: {label}")
    return payload


def authenticate_composite_tree(label: str) -> None:
    files = merged_tree_manifest(MERGED[label])
    if (
        files != EXPECTED_FILES[label]
        or tree_manifest_sha256(files) != EXPECTED_TREE_SHA256[label]
    ):
        raise ValueError(f"published composite tree changed: {label}")


def build_screen(seed: int) -> tuple[list[dict], list[dict]]:
    retention_rows = curriculum.generate_curriculum(RETENTION_MIX, seed)
    for row in retention_rows:
        row["task_id"] = f"ret{seed}_{row['task_id']}"
    retention_summary = curriculum.validate_generated(retention_rows)
    expected_retention_kinds = {
        f"u_{name}": RETENTION_PER_SKILL for name in curriculum.SKILLS
    }
    if (
        retention_summary["rows"] != ROWS_PER_SCREEN
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


def overlap_receipt(source_rows_by_seed: dict[int, list[dict]]) -> dict:
    messages_by_seed: dict[int, set[bytes]] = {}
    for seed, rows in source_rows_by_seed.items():
        messages = {message_bytes(row) for row in rows}
        if len(messages) != ROWS_PER_SCREEN:
            raise ValueError(f"screen {seed} collides on canonical user messages")
        messages_by_seed[seed] = messages
    local_messages: set[bytes] = set()
    for seed in SEEDS:
        local_messages |= messages_by_seed[seed]
    if len(local_messages) != TOTAL_ROWS:
        raise ValueError("calibration screens collide on canonical user messages")
    cross_screen: dict[str, int] = {}
    for index, first in enumerate(SEEDS):
        for second in SEEDS[index + 1 :]:
            overlap = len(messages_by_seed[first] & messages_by_seed[second])
            if overlap:
                raise ValueError(
                    f"screens {first} and {second} share canonical user messages"
                )
            cross_screen[f"{first}x{second}"] = overlap
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
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in local_messages),
        "unique_local_messages": len(local_messages),
        "unique_local_messages_per_screen": {
            str(seed): len(messages_by_seed[seed]) for seed in SEEDS
        },
        "cross_screen_overlap": cross_screen,
        "predecessor_sources": predecessor_sources,
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "predecessor_gates": predecessor_gates,
        "training_streams": {
            "paths": [],
            "policy": (
                "this cell materializes no training stream and trains "
                "nothing; every inherited stream in the five arms' lineages "
                "is hash-pinned and checked above"
            ),
        },
    }


def build_outputs(*, authenticate_composites: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    composite_receipts = {
        label: authenticate_composite_receipt(label) for label in ARMS
    }
    if authenticate_composites:
        for label in ARMS:
            authenticate_composite_tree(label)
    source_rows_by_seed: dict[int, list[dict]] = {}
    runner_rows_by_seed: dict[int, list[dict]] = {}
    for seed in SEEDS:
        source_rows_by_seed[seed], runner_rows_by_seed[seed] = build_screen(seed)
    sources = {seed: jsonl_bytes(source_rows_by_seed[seed]) for seed in SEEDS}
    runner_inputs = {seed: jsonl_bytes(runner_rows_by_seed[seed]) for seed in SEEDS}
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "retention_screen_calibration_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seeds": list(SEEDS),
        "rows_per_screen": ROWS_PER_SCREEN,
        "rows_total": TOTAL_ROWS,
        "calibration_cell": True,
        "trains_nothing": True,
        "screens": {
            str(seed): {
                "generator": "scripts/gen_curriculum.py",
                "mix": RETENTION_MIX,
                "rows": ROWS_PER_SCREEN,
                "per_kind": RETENTION_PER_SKILL,
                "id_prefix": f"ret{seed}_",
                "kinds": dict(
                    sorted(
                        Counter(
                            row["kind"] for row in source_rows_by_seed[seed]
                        ).items()
                    )
                ),
                "source": {
                    "path": SOURCES[seed].relative_to(ROOT).as_posix(),
                    "sha256": sha256_bytes(sources[seed]),
                    "contains_executable_truth": True,
                },
                "runner_input": {
                    "path": RUNNER_INPUTS[seed].relative_to(ROOT).as_posix(),
                    "sha256": sha256_bytes(runner_inputs[seed]),
                    "schema": ["id", "messages", "meta"],
                    "contains_answer": False,
                    "contains_oracle": False,
                },
            }
            for seed in SEEDS
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
        "candidates": [],
        "run_order": {
            "policy": (
                "screen-major: for each screen seed ascending, run all five "
                "arms alphabetically — 20 authenticated engine events"
            ),
            "sequence": [
                f"seed{seed}_{label}" for seed in SEEDS for label in ARMS
            ],
        },
        "readings": {
            "no_promotion": True,
            "screen_sd_pooled": (
                "pooled within-arm across-screen sample SD (ddof=1) of "
                "retention correct: sqrt(mean over the five arms of the "
                "per-arm across-screen sample variance)"
            ),
            "recommended_band": (
                f"max({BAND_MINIMUM}, ceil(2 * screen_sd_pooled))"
            ),
            "adjudication_protocol": {
                "single_screen": (
                    f"screen_sd_pooled <= {SINGLE_SCREEN_MAX_SD}"
                ),
                "pooled_k2": (
                    f"{SINGLE_SCREEN_MAX_SD} < screen_sd_pooled <= "
                    f"{POOLED_K2_MAX_SD}"
                ),
                "pooled_k3": f"screen_sd_pooled > {POOLED_K2_MAX_SD}",
            },
            "stability_flags": {
                "rule": (
                    "a historical single-screen reading is stable when it "
                    "lies inside the arm's pooled delta vs clean_parent +/- "
                    "2 x the across-screen sample SD of that delta"
                ),
                "historical_readings": [
                    {"arm": arm, "gate_seed": gate_seed, "delta": delta}
                    for arm, gate_seed, delta in HISTORICAL_READINGS
                ],
            },
            "vehicle_descriptive": (
                "axis160_r64's pooled delta versus axis160_direct's, with "
                "measured-noise intervals — reported, not gated"
            ),
            "delta_reference_arm": PARENT,
            "complete_event_always_exits_zero": True,
            "no_aggregate_seed_exists": True,
        },
        "prerequisites": {
            "inherited_arms": {
                label: {
                    "merge_receipt": (
                        COMPOSITE_RECEIPTS[label].relative_to(ROOT).as_posix()
                    ),
                    "merge_receipt_sha256": EXPECTED_RECEIPT_SHA256[label],
                    "composite": MERGED[label].relative_to(ROOT).as_posix(),
                    "composite_name": EXPECTED_COMPOSITE_NAMES[label],
                    "tree_sha256": EXPECTED_TREE_SHA256[label],
                    "weights_sha256": EXPECTED_WEIGHTS_SHA256[label],
                    "files": EXPECTED_FILES[label],
                    "receipt_authenticated": composite_receipts[label]["name"]
                    == EXPECTED_COMPOSITE_NAMES[label],
                }
                for label in ARMS
            },
        },
        "code_sha256": {
            name: sha256_file(path) for name, path in sorted(CODE_FILES.items())
        },
        "code_pins_deferred": {
            "reason": "none: every code file of the local event is pinned here",
            "files": [],
        },
        "firewall": {
            "benchmark_data_read": False,
            "benchmark_gateway_exposed": False,
            "no_aggregate_seed_exists": True,
        },
        "next_authorized_stage": "local",
    }
    rendered_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    outputs: dict[Path, bytes] = {}
    for seed in SEEDS:
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
                "rows_total": TOTAL_ROWS,
                "sources_sha256": {
                    str(seed): sha256_bytes(outputs[SOURCES[seed]]) for seed in SEEDS
                },
                "runner_inputs_sha256": {
                    str(seed): sha256_bytes(outputs[RUNNER_INPUTS[seed]])
                    for seed in SEEDS
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
