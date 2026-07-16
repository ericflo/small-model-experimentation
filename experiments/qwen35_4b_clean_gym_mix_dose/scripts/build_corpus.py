#!/usr/bin/env python3
"""Freeze the FRESH gym-mix treatment corpus, its manifest, and the audits.

Lifecycle 25: the corpus is the 160-row three-kind gym-mix (60
``u_siren_episode`` + 50 ``u_statechain`` + 50 ``u_mirage_abstain``)
rendered by gen_gym_mix_curriculum.py at the frozen construction seed
77,180. Everything is a pure function of that seed; ``--check``
regenerates and byte-compares. The retired gym-era prefix is never read,
referenced, or used as an input anywhere — only the PUBLIC family
descriptions informed the design. Beyond the generator's case-insensitive
banned-vocabulary scan (the proven statechain inventory EXTENDED with the
sirens/mirage description nouns), this builder runs TWO audits against
the replay pool and EVERY pinned predecessor corpus, training stream, and
statechain-bearing frozen local gate in the arms' lineages and audit
tradition — INCLUDING the lifecycle 18 statechain cell, the lifecycle 23
clean-path cell (whose gates carry statechain instances from the same
byte-copied generator), the menders dose-scale cell, and this cell's own
byte-copied zero-root lineage datasets (fail-closed hash pins):

- fresh-surface grep audit: every claimed fresh token (the siren and
  mirage vocabulary pools plus the two kind surfaces) must have zero
  case-insensitive word-boundary hits in every pinned source;
- row-overlap audit: the generated corpus must share zero canonical user
  messages with every pinned source — this is the receipt that the
  reused statechain formalisms are genuinely FRESH instances, not copies
  of any predecessor's rows.

The retained statechain vocabulary (brewvat/courierloft/peatstove/
muletrack pools) is documented as INHERITED and deliberately excluded
from the fresh-surface claim (it appears in predecessor corpora by
design). The mirage abstain token is ``NOWHERE`` (documented design
decision; ``impossible`` is banned everywhere in this corpus).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_gym_mix_curriculum as gym  # noqa: E402
import gen_statechain_curriculum as statechain  # noqa: E402

CONSTRUCTION_SEED = 77180
CORPUS_PATH = EXP / "data" / "sft_gym_mix.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
EXPECTED_KINDS = {"u_siren_episode": 60, "u_statechain": 50, "u_mirage_abstain": 50}
EXPECTED_SURFACES = {
    "stillroom": 60,
    "counterhouse": 50,
    "brewvat": 13,
    "courierloft": 13,
    "peatstove": 12,
    "muletrack": 12,
}
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
CLEAN_PATH_EXP = ROOT / "experiments" / "qwen35_4b_clean_path_statechain_extension"
MENDERS_EXP = ROOT / "experiments" / "qwen35_4b_menders_dose_scale"
# Every inherited corpus and materialized training stream in the lineages of
# the parent composite and its published siblings (fail-closed hash pins),
# carried forward from the clean-path cell's audited inventory, PLUS the
# lifecycle 18/23 statechain cells' treatments, streams, and gate files (the
# same byte-copied generator mints this cell's fresh statechain instances),
# the menders dose-scale cell's treatment/streams/gates, and this cell's own
# byte-copied zero-root lineage stage datasets. gen_local_gate.py imports
# this single inventory so the corpus and gate audits cannot drift apart.
PREDECESSOR_SOURCES = (
    (REPLAY_PATH, REPLAY_SHA256, REPLAY_ROWS),
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
    # Lifecycle 18 (the SOURCE statechain cell): treatment, streams, gates.
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
    # Lifecycle 23 (the clean-path cell): byte-copied treatment, streams,
    # and its four statechain-bearing gate files (seeds 88041-88045).
    (
        CLEAN_PATH_EXP / "data" / "sft_statechain_only.jsonl",
        "ab6c78458eb8a41f42ebee25e79354d42beb558f78bb3d348dd7902ca3a9bad3",
        160,
    ),
    (
        CLEAN_PATH_EXP / "data" / "replay_ctl4.jsonl",
        "b18e90b25cd9d4ea7341cf5bb27b37b15bd50bdd50dabb927ca2ea19e1b8b39d",
        1520,
    ),
    (
        CLEAN_PATH_EXP / "data" / "statechain_clean.jsonl",
        "bae8af90ebed212a6381b6192ec5a17949630e68d6b7d18edd1d837196efcb55",
        1520,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88041.jsonl",
        "ed16ed460799bc9b3f47767d35d4b0cd28a00f58deccf23c6608704aa07cf0d0",
        40,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88042.jsonl",
        "c9540f614120787ede5afe7454f00ed412fe9edbf8d7f7e5ee078ba9032413d7",
        104,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88044.jsonl",
        "6119e52c18cdce24ad713d88ca77174f43bdf832550a09e90ac463a232f92faf",
        104,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88045.jsonl",
        "0ec31a4d5e71e5931cdf5e9d6b9f977bcc9b8dd908f778cd42c20d1832a251f9",
        104,
    ),
    # The menders dose-scale cell: treatment, streams, gates (its corpus
    # carries its own NEW feedloop formalisms; the fresh-surface audit must
    # clear them too).
    (
        MENDERS_EXP / "data" / "sft_feedloop_scale.jsonl",
        "080c3603cd3bfca2b261b797be356f97684aedc4f65becd9b50cbb45706bd2c2",
        800,
    ),
    (
        MENDERS_EXP / "data" / "feedloop_scale.jsonl",
        "3aee5f5e8731294e0d8a9b9dd08f014ca34ace9f949c2b46aa09aa1e548adc58",
        2280,
    ),
    (
        MENDERS_EXP / "data" / "replay_ctl3.jsonl",
        "02275b9553cc7ff6676ed370a91e1e154dd5d768a4ed36ae41a1ae583b92b56b",
        2280,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88037.jsonl",
        "5ec590bf508c5f9e6aa4758d37819e68eaa4807e400a63ac2f980ad975c45519",
        40,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88038.jsonl",
        "96e03aea5013f871b0ec863590ae5ea62d0620f09b216d0af39eee2bba1d74c9",
        104,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88039.jsonl",
        "9575b27506011ccb1fd32c1c63579610b80ef2c330629dbcfd930ad442942367",
        104,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88040.jsonl",
        "7b0223561fe32da48e6b45a280fae1ade96a8546f5429e76f14f607a6716deed",
        104,
    ),
    # The parent's ENTIRE training lineage: this cell's byte-identical
    # copies of the six zero-root stage datasets (the clean chain).
    (
        EXP / "data" / "lineage" / "stage01_replay_refresh.jsonl",
        "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6",
        1520,
    ),
    (
        EXP / "data" / "lineage" / "stage02_designed160.jsonl",
        "5159cf41b6474bdc8640cdb2a4a168587b59232ca8171c7b7057fc6bfe1b40c8",
        1520,
    ),
    (
        EXP / "data" / "lineage" / "stage03_close_xi__targeted_standard.jsonl",
        "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00",
        320,
    ),
    (
        EXP / "data" / "lineage" / "stage04_replay_after_close.jsonl",
        "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6",
        320,
    ),
    (
        EXP / "data" / "lineage" / "stage05_designed_fresh.jsonl",
        "6d4dc303bc159c19a1ffd0c60ca7d08ea64b02909366701b345d888482d67f3f",
        1520,
    ),
    (
        EXP / "data" / "lineage" / "stage06_hygiene_explore.jsonl",
        "82aa1a78c0a429a48c3db6b94ac84397cea001b041477e7b137b38c21354112f",
        1520,
    ),
)
# ALL predecessor experiments' frozen local gates and screens, seeds
# 88013-88045 (fail-closed hash pins), carried forward from the clean-path
# cell's audited inventory plus the clean-path cell's own four gate files.
# gen_local_gate.py imports this too (its gate-freshness contract).
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
    (
        MENDERS_EXP / "data" / "local_tasks_seed88037.jsonl",
        "5ec590bf508c5f9e6aa4758d37819e68eaa4807e400a63ac2f980ad975c45519",
        40,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88038.jsonl",
        "96e03aea5013f871b0ec863590ae5ea62d0620f09b216d0af39eee2bba1d74c9",
        104,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88039.jsonl",
        "9575b27506011ccb1fd32c1c63579610b80ef2c330629dbcfd930ad442942367",
        104,
    ),
    (
        MENDERS_EXP / "data" / "local_tasks_seed88040.jsonl",
        "7b0223561fe32da48e6b45a280fae1ade96a8546f5429e76f14f607a6716deed",
        104,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88041.jsonl",
        "ed16ed460799bc9b3f47767d35d4b0cd28a00f58deccf23c6608704aa07cf0d0",
        40,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88042.jsonl",
        "c9540f614120787ede5afe7454f00ed412fe9edbf8d7f7e5ee078ba9032413d7",
        104,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88044.jsonl",
        "6119e52c18cdce24ad713d88ca77174f43bdf832550a09e90ac463a232f92faf",
        104,
    ),
    (
        CLEAN_PATH_EXP / "data" / "local_tasks_seed88045.jsonl",
        "0ec31a4d5e71e5931cdf5e9d6b9f977bcc9b8dd908f778cd42c20d1832a251f9",
        104,
    ),
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def check_fresh_surfaces_and_overlap(corpus_rows: list[dict]) -> tuple[dict, dict]:
    """Grep-audit fresh tokens AND row-overlap against every pinned source."""
    tokens = tuple(dict.fromkeys(gym.FRESH_SURFACE_TOKENS))
    if len(tokens) != len(gym.FRESH_SURFACE_TOKENS):
        raise ValueError("fresh surface token inventory contains duplicates")
    inherited = tuple(
        dict.fromkeys(
            statechain.INHERITED_SURFACE_TOKENS + statechain.FRESH_SURFACE_TOKENS
        )
    )
    if set(tokens) & set(inherited):
        raise ValueError("a token is claimed both fresh and inherited")
    if set(tokens) & set(gym.BANNED_PROMPT_TOKENS) or set(inherited) & set(
        gym.BANNED_PROMPT_TOKENS
    ):
        raise ValueError("a claimed surface token is banned")
    patterns = {
        token: re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        for token in tokens
    }
    corpus_messages = {message_bytes(row) for row in corpus_rows}
    if len(corpus_messages) != len(corpus_rows):
        raise ValueError("treatment corpus collides on canonical user messages")
    sources = PREDECESSOR_SOURCES + PREDECESSOR_GATES
    surface_sources: dict[str, dict] = {}
    overlap_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in sources:
        raw = path.read_bytes()
        if sha256_bytes(raw) != expected_sha256:
            raise ValueError(f"pinned predecessor corpus changed: {path}")
        lines = raw.decode("utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line]
        if len(rows) != expected_rows:
            raise ValueError(f"pinned predecessor corpus row count changed: {path}")
        haystack = "\n".join(line for line in lines if line)
        hits = sorted(
            token for token, pattern in patterns.items() if pattern.search(haystack)
        )
        if hits:
            raise ValueError(
                f"fresh surface tokens collide with a predecessor corpus {path}: {hits}"
            )
        messages = {message_bytes(row) for row in rows if row.get("messages")}
        overlap = len(corpus_messages & messages)
        if overlap:
            raise ValueError(
                f"treatment corpus rows overlap a predecessor source: {path}"
            )
        key = path.relative_to(ROOT).as_posix()
        surface_sources[key] = {
            "sha256": expected_sha256,
            "rows": expected_rows,
            "tokens_checked": len(tokens),
            "hits": 0,
        }
        overlap_sources[key] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": 0,
        }
    surface_audit = {
        "tokens": list(tokens),
        "token_inventory_sha256": sha256_bytes("\n".join(tokens).encode("utf-8")),
        "match_rule": "case-insensitive word-boundary regex, zero hits required",
        "inherited_tokens": {
            "tokens": list(inherited),
            "sha256": sha256_bytes("\n".join(inherited).encode("utf-8")),
            "policy": (
                "the statechain formalisms' vocabulary (brewvat/courierloft/"
                "peatstove/muletrack pools) is retained from the PROVEN "
                "byte-copied generator by design; it appears in predecessor "
                "corpora and is excluded from the fresh-surface claim"
            ),
        },
        "sources": surface_sources,
    }
    overlap_audit = {
        "corpus_messages": len(corpus_messages),
        "match_rule": "canonical user-message byte equality, zero overlap required",
        "sources": overlap_sources,
    }
    return surface_audit, overlap_audit


def build() -> tuple[dict[Path, bytes], dict]:
    rows = gym.generate_curriculum(gym.ARM_MIX, CONSTRUCTION_SEED)
    summary = gym.validate_generated(rows)
    gym.check_banned_vocabulary(rows)
    balance = gym.check_corpus_balance(rows)
    if (
        len(rows) != 160
        or summary["kinds"] != EXPECTED_KINDS
        or summary["surfaces"] != EXPECTED_SURFACES
        or balance.get("mirage_class_token_sets_identical") is not True
    ):
        raise ValueError(f"gym-mix corpus mix changed: {summary} / {balance}")
    surface_audit, overlap_audit = check_fresh_surfaces_and_overlap(rows)

    corpus = "".join(
        json.dumps(gym.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    banned = tuple(gym.BANNED_PROMPT_TOKENS)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "generator": "scripts/gen_gym_mix_curriculum.py",
        "prefix_contamination_statement": (
            "the retired gym-era prefix composite is NEVER read, referenced, "
            "or used as an input anywhere in this cell; only the PUBLIC "
            "sirens/rites/mirage family descriptions informed the design, "
            "and every rendered surface is invented vocabulary audited "
            "against every pinned predecessor source"
        ),
        "banned_vocabulary_checked": True,
        "banned_vocabulary": {
            "tokens": len(banned),
            "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
            "base_inventory": "gen_statechain_curriculum.BANNED_PROMPT_TOKENS",
            "extended_with_description_nouns": list(
                gym.DESCRIPTION_NOUN_EXTENSIONS
            ),
            "case_insensitive": True,
        },
        "abstain_token": {
            "token": gym.ABSTAIN_TOKEN,
            "decision": (
                "the mirage-axis abstain answer literal is the invented "
                "token NOWHERE, preferred over the benchmark family's "
                "public IMPOSSIBLE to keep surface distance; 'impossible' "
                "is banned everywhere in this corpus (prompt, think, and "
                "answer, case-insensitively)"
            ),
        },
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": len(rows),
            "mix": gym.ARM_MIX,
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "sha256": sha256_bytes(corpus),
        },
        "gym_mix_contract": {
            "siren_episode": {
                "mechanism": (
                    "the reviewed u_hygiene injection-with-format-matched-"
                    "decoys mechanism (7/7 all-time installs) extended to "
                    "episode form: a multi-step book-retrieval transcript in "
                    "ONE user message; embedded imperatives are book content"
                ),
                "decoy_never_equals_truth": True,
                "imperatives_wellformed_and_embedded": True,
                "injected_rows": balance["siren_injected"],
                "clean_rows": balance["siren_clean"],
                "lookups_per_row": "3-4",
            },
            "statechain": {
                "generator": (
                    "byte-identical copy of the PROVEN lifecycle 18 "
                    "generator (scripts/gen_statechain_curriculum.py); "
                    "machinery imported, never forked; FRESH instances at "
                    "this cell's construction seed"
                ),
                "formalisms": list(statechain.STATECHAIN_FORMALISMS),
                "formalism_counts": balance["statechain_formalisms"],
                "min_hidden_updates": 3,
                "reviewed_invariants_intact": True,
                "fresh_instances_via_row_overlap_audit": True,
            },
            "mirage_abstain": {
                "proof": (
                    "exhaustive enumeration over the FULL domain (1-6) for "
                    "every row: FORCED rows have exactly one value for the "
                    "queried counter across all satisfying fillings; ABSTAIN "
                    "rows provably admit no filling at all or more than one "
                    "value"
                ),
                "forced_rows": balance["mirage_forced"],
                "abstain_rows": balance["mirage_abstain"],
                "abstain_reasons": balance["mirage_abstain_reasons"],
                "class_indistinguishability": {
                    "generated_in_digit_only_pairs": True,
                    "complete_pairs": balance["mirage_complete_pairs"],
                    "class_token_sets_identical": balance[
                        "mirage_class_token_sets_identical"
                    ],
                    "rule": (
                        "each forced/abstain pair shares one surface skeleton "
                        "and differs ONLY in digit characters; across the "
                        "corpus the two classes' alphabetic token sets are "
                        "identical"
                    ),
                },
            },
        },
        "balance": balance,
        "surface_freshness_audit": surface_audit,
        "row_overlap_audit": overlap_audit,
        "replay": {
            "path": REPLAY_PATH.relative_to(EXP).as_posix(),
            "rows": REPLAY_ROWS,
            "sha256": REPLAY_SHA256,
            "byte_identical_to_every_predecessor_copy": True,
        },
    }
    return {CORPUS_PATH: corpus}, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, manifest = build()
    targets = dict(outputs)
    targets[MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"frozen corpus artifact is absent or changed: {path}")
    else:
        conflicts = [
            path
            for path, expected in targets.items()
            if path.exists() and path.read_bytes() != expected
        ]
        if conflicts:
            parser.error(
                f"refusing to overwrite a differing frozen corpus artifact: {conflicts[0]}"
            )
        for path, value in targets.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "corpus_sha256": manifest["corpus"]["sha256"],
        "manifest_sha256": sha256_bytes(targets[MANIFEST_PATH]),
        "kinds": manifest["corpus"]["kinds"],
        "surfaces": manifest["corpus"]["surfaces"],
        "balance": manifest["balance"],
        "surface_tokens_checked": len(manifest["surface_freshness_audit"]["tokens"]),
        "overlap_sources_checked": len(manifest["row_overlap_audit"]["sources"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
