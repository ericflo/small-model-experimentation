#!/usr/bin/env python3
"""Freeze the 200-item repair-verifier 2AFC probe set and its model inputs.

This is a pure eval feasibility probe: ONE inherited, externally published
composite (hygiene_explore, tree 9eb653d7...) is measured on ONE fresh
200-item 2AFC verification instrument under TWO decode configs (think /
nothink). Nothing trains, nothing merges, nothing promotes, and there is
no aggregate seed. The frozen design comprises:

- PROBE SET: 200 fresh two-round eliminative-repair episode instances
  (25 per formalism across all eight: troughline, trinketcord, crankwheel,
  sigilslate, barrowyoke, balesled, millround, skeinreel) selected
  deterministically from a 320-instance pool (40 per formalism) generated
  by the REUSED dose-scale generator machinery
  (gen_feedloop_curriculum.generate_curriculum) at construction seed
  77160, with all of its reviewed invariants (>=2 legal fix candidates
  after round-1 evidence with the round-1-consistent distractor among
  them, exactly 1 survivor after rounds 1+2, extended-grammar legality
  audit). Selection rule (frozen): the first 25 instances per formalism
  in stream order whose WRITTEN sequence also fails trial two — pure
  failure evidence on both trials.
- FULLY SYMMETRIC 2AFC ITEM (no repair history, no attempt narration):
  each selected instance is rendered fresh as (a) the machine spec with
  its legality clauses, (b) the original written sequence, (c) BOTH
  trials' setups, WANTED outcomes, and the OBSERVED outcomes of the
  original (broken) sequence on both trials, (d) two candidate
  single-step changes labeled A and B in identical grammatical form with
  no provenance markers — one is the unique legal both-trials fix, the
  other the generator's legal trial-one-consistent distractor — and
  (e) the ask for the letter of the one change that makes both trials
  come out as wanted. Deciding requires SIMULATING each candidate
  against both trials (execution-based fix verification). A frozen
  MARKER-TOKEN audit rejects any provenance word (tried/attempt/failed/
  earlier/...) anywhere in the rendered prompt, and self-reference
  checks reject any option whose labeled line or quoted change text
  appears elsewhere in the prompt.
- POSITION BALANCE: the correct fix is A on exactly 100 items and B on
  exactly 100, assigned deterministically from the construction seed
  (per-formalism 13/12 split, shuffled by a formalism-keyed stream).
- RE-VERIFICATION: every item's correct/wrong fix properties are
  re-executed against the machine semantics (the written run reproduces
  the recorded trial-one evidence AND fails trial two; the correct fix
  lands on the wanted outcome in BOTH trials; the distractor lands on
  the wanted outcome in trial one but on the recorded failing outcome in
  trial two) in addition to the generator's own audits, and every
  re-implemented state/step renderer is verified against the generator's
  own episode rendering before it is used.
- LISTING-COLLISION AUDIT (frozen artifact reading): a candidate change
  can legitimately coincide with a step already in the written sequence;
  the receipt records the per-option collision counts and the best
  accuracy any collision-keyed guessing heuristic could reach, so the
  artifact ceiling is on file next to the 0.65 signal bar.

Grading applies the frozen answer normalization documented in this receipt
(``answer_normalization``) identically to both arms; the expected answer is
a single letter. Freshness is enforced fail-closed: zero canonical
user-message overlap with the dose-scale corpus and BOTH its materialized
training streams, its 40-row axis holdout and three retention screens,
every pinned predecessor corpus and stream, every predecessor frozen local
gate (seeds 88013-88040), regenerated prior local seeds (88000-88040), and
the regenerated dose-scale treatment/holdout rows. The composite pin is
recomputed here against the committed merge receipt; nothing is deferred
and no TODO-PIN exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
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
    ARM_THINKING,
    CAP_SCOPE_THRESHOLD,
    CHANCE_FLOOR,
    CONFIDENCE,
    CONSEQUENCES,
    CONSTRUCTION_SEED,
    FORMALISMS,
    GATING_ARM,
    MAX_TOKENS,
    PER_FORMALISM,
    POSITIONS,
    PROBE_KIND,
    PROBE_ROWS,
    ROWS_PER_POSITION,
    SIGNAL_MIN_ACCURACY,
)


POOL_PER_FORMALISM = 40
POOL_ROWS = POOL_PER_FORMALISM * len(FORMALISMS)
OVERSAMPLE_MIX = f"feedloop={POOL_ROWS}"
SELECTION_RULE = (
    "first 25 instances per formalism in generator stream order whose "
    "WRITTEN sequence also fails trial two (pure failure evidence on both "
    "trials); instances where the broken sequence coincidentally lands on "
    "trial two's wanted outcome are excluded and counted"
)
ID_PREFIX = f"probe{CONSTRUCTION_SEED}_"
SOURCE = EXP / "data" / f"probe_tasks_seed{CONSTRUCTION_SEED}.jsonl"
RUNNER_INPUT = EXP / "data" / f"probe_input_seed{CONSTRUCTION_SEED}.jsonl"
RECEIPT = EXP / "data" / "local_design_receipt.json"
INSTRUMENT = "verifier_2afc"
ANSWER_PATTERN = re.compile(r"\AANSWER: STEP (\d+): (.+)\Z")
# The frozen generation ask every reused episode prompt ends with. It is
# used ONLY to authenticate the generator's episode rendering (structure
# pin); the 2AFC prompt is rendered fresh and never contains it.
GENERATION_TAIL = (
    "Using the evidence from both trials, name the one change that makes "
    "both trials come out as wanted.\n"
    f"{feedloop.ANSWER_LINE} (format: STEP <k>: <corrected step>)"
)
# The fully symmetric 2AFC rendering: spec + written steps + BOTH trials'
# setups/wanted/observed outcomes of the ORIGINAL (broken) sequence — pure
# failure evidence, no repair history, no attempt narration — then two
# candidates in identical grammatical form.
CHOICE_SECTION_TEMPLATE = (
    "Trial one, starting from {start_a}: the crew wanted {wanted_a}, but "
    "the run finished at {finished_a}.\n"
    "Trial two, starting from {start_b}: the crew wanted {wanted_b}, but "
    "the run finished at {finished_b}.\n"
    "Exactly one written step is off. Two candidate one-step changes are "
    "proposed:\n"
    "A. change step {a_step} to '{a_change}'\n"
    "B. change step {b_step} to '{b_change}'\n"
    "Using the evidence from both trials, answer with the letter of the one "
    "change that makes both trials come out as wanted.\n"
    "End with exactly one line:\n"
    "ANSWER: <A or B>"
)
# Provenance-marker audit: none of these words may appear ANYWHERE in a
# rendered 2AFC prompt (word-boundary, case-insensitive). They are the
# vocabulary that could mark one option as a previously tried/failed
# repair and turn the item into text matching instead of simulation.
MARKER_TOKENS = (
    "attempt", "attempts", "attempted", "attempting",
    "try", "tries", "tried", "trying",
    "earlier", "previous", "previously", "prior", "former", "original",
    "originally", "already", "instead",
    "fail", "fails", "failed", "failing", "failure",
    "wrong", "wrongly", "incorrect", "incorrectly", "mistake", "mistaken",
    "rejected", "ruled", "eliminated", "eliminates", "discarded",
)
_MARKER_PATTERNS = tuple(
    (re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE), token)
    for token in MARKER_TOKENS
)
# Re-execution semantics and rendering for the per-item re-verification:
# the reused module's own apply/describe functions, keyed by formalism,
# plus re-implemented state renderers that are verified per item against
# the generator's own episode rendering before use.
APPLY_BY_FORMALISM = {
    "troughline": feedloop._trough_apply,
    "trinketcord": feedloop._cord_apply,
    "crankwheel": feedloop._wheel_apply,
    "sigilslate": feedloop._slate_apply,
    "barrowyoke": feedloop._yoke_apply,
    "balesled": feedloop._sled_apply,
    "millround": feedloop._round_apply,
    "skeinreel": feedloop._reel_apply,
}
DESCRIBE_BY_FORMALISM = {
    "troughline": feedloop._trough_describe,
    "trinketcord": feedloop._cord_describe,
    "crankwheel": feedloop._wheel_describe,
    "sigilslate": feedloop._slate_describe,
    "barrowyoke": feedloop._yoke_describe,
    "balesled": feedloop._sled_describe,
    "millround": feedloop._round_describe,
    "skeinreel": feedloop._reel_describe,
}

# The dose-scale cell (the corpus this probe must be disjoint from) and its
# artifacts: the 800-row treatment corpus, the replay blend, and BOTH
# materialized 2,280-row training streams (fail-closed hash pins).
MENDERS_EXP = ROOT / "experiments" / "qwen35_4b_menders_dose_scale"
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
DOSE_SCALE_SOURCES = (
    (MENDERS_EXP / "data" / "sft_feedloop_scale.jsonl",
     "080c3603cd3bfca2b261b797be356f97684aedc4f65becd9b50cbb45706bd2c2", 800),
    (MENDERS_EXP / "data" / "sft_blend.jsonl",
     "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2", 2240),
    (MENDERS_EXP / "data" / "feedloop_scale.jsonl",
     "3aee5f5e8731294e0d8a9b9dd08f014ca34ace9f949c2b46aa09aa1e548adc58", 2280),
    (MENDERS_EXP / "data" / "replay_ctl3.jsonl",
     "02275b9553cc7ff6676ed370a91e1e154dd5d768a4ed36ae41a1ae583b92b56b", 2280),
)
# Every inherited corpus and every materialized training stream in the
# lineages of the published composites (fail-closed hash pins), copied
# forward from the dose-scale cell's audited inventory.
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
# 88013-88040 (fail-closed hash pins): the nine 88013-88021 gates, the
# calibration cell's four retention-only screens (88022-88025), the
# reference cell's four gate input files (axis 88026 + screens
# 88027/88028/88030), the statechain cell's four (axis 88033 + screens
# 88034-88036), and the dose-scale cell's four (axis holdout 88037 — the
# dose-scale corpus's holdout — + screens 88038-88040).
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
)
PRIOR_LOCAL_SEEDS = tuple(range(88000, 88041))
DOSE_SCALE_CONSTRUCTION_SEED = 77150
DOSE_SCALE_HOLDOUT_SEED = 88037
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
# The one evaluated model: the hygiene_explore composite, inherited and
# externally published; every pin is filled at design freeze — nothing is
# deferred and no TODO-PIN exists.
DESTACK_EXP_NAME = "qwen35_4b_hygiene_explore_destack_medium"
COMPOSITE_LABEL = "hygiene_explore"
COMPOSITE_RECEIPT = (
    ROOT / "experiments" / DESTACK_EXP_NAME / "runs" / "merges"
    / "hygiene_explore.json"
)
MERGED = ROOT / "large_artifacts" / DESTACK_EXP_NAME / "merged" / "hygiene_explore"
EXPECTED_RECEIPT_SHA256 = (
    "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
)
EXPECTED_TREE_SHA256 = (
    "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
)
EXPECTED_WEIGHTS_SHA256 = (
    "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
)
EXPECTED_FILES = [
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
# Files pinned by hash into the frozen design receipt. There is no merge
# script and no deferred pin: eval_local_vllm.py carries no
# orchestrator-filled constant in this cell, so the complete code surface of
# the local event is frozen here (plus the standalone lineage package's
# reproduction script and merger copy).
CODE_FILES = {
    "generator": Path(__file__),
    "feedloop_curriculum": EXP / "scripts" / "gen_feedloop_curriculum.py",
    "curriculum": EXP / "scripts" / "gen_curriculum.py",
    "gate": EXP / "scripts" / "check_local.py",
    "eval": EXP / "scripts" / "eval_local_vllm.py",
    "harness": EXP / "scripts" / "run.py",
    "runner": EXP / "src" / "vllm_runner.py",
    "rebuild_lineage": EXP / "scripts" / "rebuild_lineage.py",
    "external_merger": EXP / "scripts" / "merge_adapter.py",
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


def authenticate_composite_receipt() -> dict:
    """Authenticate the inherited composite's committed merge receipt."""
    if (
        not COMPOSITE_RECEIPT.is_file()
        or sha256_file(COMPOSITE_RECEIPT) != EXPECTED_RECEIPT_SHA256
    ):
        raise ValueError(f"published prerequisite receipt changed: {COMPOSITE_RECEIPT}")
    payload = json.loads(COMPOSITE_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("name") != COMPOSITE_LABEL
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != MERGED.resolve()
        or payload.get("output_tree_sha256") != EXPECTED_TREE_SHA256
        or payload.get("output_files") != EXPECTED_FILES
        or payload.get("weight_files")
        != [{"name": "model.safetensors", "sha256": EXPECTED_WEIGHTS_SHA256}]
    ):
        raise ValueError("published hygiene_explore merge receipt violates pins")
    return payload


def authenticate_composite_tree() -> None:
    files = merged_tree_manifest(MERGED)
    if (
        files != EXPECTED_FILES
        or tree_manifest_sha256(files) != EXPECTED_TREE_SHA256
    ):
        raise ValueError("published hygiene_explore composite tree changed")


def _run_steps(apply_fn, steps: list[tuple], start):
    current = start
    for op in steps:
        current = apply_fn(op, current)
    return current


def written_trial_two_outcome(row: dict):
    """The observed outcome of the ORIGINAL (broken) sequence on trial two."""
    audit = row["_audit"]
    spec = audit["spec"]
    apply_fn = APPLY_BY_FORMALISM[audit["formalism"]]
    written = [tuple(op) for op in spec["written"]]
    return _run_steps(apply_fn, written, spec["start_b"])


def reverify_fixes(row: dict):
    """Re-execute the machine semantics: both candidates' recorded properties
    must reproduce, on top of the generator's own audits (fail-closed), and
    the WRITTEN sequence must fail BOTH trials (pure failure evidence).
    Returns the written sequence's trial-two outcome."""
    audit = row["_audit"]
    spec = audit["spec"]
    apply_fn = APPLY_BY_FORMALISM[audit["formalism"]]
    written = [tuple(op) for op in spec["written"]]
    true_index, true_op = spec["true_fix"][0], tuple(spec["true_fix"][1])
    wrong_index, wrong_op = spec["wrong_fix"][0], tuple(spec["wrong_fix"][1])
    if (true_index, true_op) == (wrong_index, wrong_op):
        raise ValueError("correct and wrong fixes coincide")
    patched_true = written[:true_index] + [true_op] + written[true_index + 1 :]
    patched_wrong = written[:wrong_index] + [wrong_op] + written[wrong_index + 1 :]
    start_a, start_b = spec["start_a"], spec["start_b"]
    if (
        _run_steps(apply_fn, written, start_a) != spec["finished_a"]
        or spec["finished_a"] == spec["wanted_a"]
        or _run_steps(apply_fn, patched_true, start_a) != spec["wanted_a"]
        or _run_steps(apply_fn, patched_true, start_b) != spec["wanted_b"]
        or _run_steps(apply_fn, patched_wrong, start_a) != spec["wanted_a"]
        or _run_steps(apply_fn, patched_wrong, start_b)
        != spec["finished_b_after_wrong"]
        or spec["finished_b_after_wrong"] == spec["wanted_b"]
    ):
        raise ValueError(
            "re-execution failed to reproduce the recorded fix properties"
        )
    if (
        audit.get("unique_after_round2") is not True
        or audit.get("wrong_in_round1") is not True
        or audit.get("candidates_after_round1", 0) < 2
        or audit.get("bug_step") != true_index + 1
        or audit.get("wrong_attempt_step") != wrong_index + 1
    ):
        raise ValueError("generator audit fields disagree with the spec")
    finished_b_written = _run_steps(apply_fn, written, start_b)
    if finished_b_written == spec["wanted_b"]:
        raise ValueError(
            "the broken sequence succeeds on trial two; the selection rule "
            "must have excluded this instance"
        )
    return finished_b_written


def render_state(formalism: str, vocabulary: list, state) -> str:
    """Re-implemented per-formalism state renderer; verified per item
    against the generator's own episode rendering before use."""
    if formalism in ("troughline", "barrowyoke"):
        return ", ".join(f"{name}={state[name]}" for name in vocabulary)
    if formalism == "trinketcord":
        return " ".join(state) if state else "(no trinkets)"
    if formalism == "balesled":
        return " ".join(state) if state else "(no bales)"
    if formalism == "crankwheel":
        return f"needle on {vocabulary[state[0]]}, hopper {state[1]}"
    if formalism == "millround":
        return f"shoe on {vocabulary[state[0]]}, creel {state[1]}"
    if formalism == "skeinreel":
        return f"{state[0]} coils, lay {state[1]}"
    if formalism == "sigilslate":
        return " ".join(
            f"{index + 1}:{value if value is not None else '~'}"
            for index, value in enumerate(state)
        )
    raise ValueError(f"unknown formalism: {formalism}")


def parse_episode(row: dict) -> dict:
    """Authenticate the generator's episode rendering and extract the reusable
    pieces (scene, rules, step listing, rendered states). Every re-implemented
    renderer output is cross-checked against the episode's own sentences."""
    prompt = row["messages"][0]["content"]
    if prompt.count(GENERATION_TAIL) != 1 or not prompt.endswith(GENERATION_TAIL):
        raise ValueError("episode prompt does not end with the frozen generation ask")
    audit = row["_audit"]
    spec = audit["spec"]
    formalism = audit["formalism"]
    vocabulary = spec["vocabulary"]
    describe = DESCRIBE_BY_FORMALISM[formalism]
    written = [tuple(op) for op in spec["written"]]
    lines = prompt.split("\n")
    scene, rules = lines[0], lines[1]
    if lines[2] != "Steps as written:":
        raise ValueError("episode prompt structure changed (steps header)")
    listing = [
        f"  {index + 1}. {describe(op)}" for index, op in enumerate(written)
    ]
    if lines[3 : 3 + len(written)] != listing:
        raise ValueError("episode step listing disagrees with the described spec")
    renders = {
        key: render_state(formalism, vocabulary, spec[key])
        for key in (
            "start_a", "wanted_a", "finished_a",
            "start_b", "wanted_b", "finished_b_after_wrong",
        )
    }
    trial_one = (
        f"First trial, starting from {renders['start_a']}: the crew wanted "
        f"{renders['wanted_a']}, but the run finished at "
        f"{renders['finished_a']}."
    )
    trial_two_attempt = (
        f"a second trial ran the steps from {renders['start_b']}: the crew "
        f"wanted {renders['wanted_b']}, but it finished at "
        f"{renders['finished_b_after_wrong']}."
    )
    if trial_one not in prompt or trial_two_attempt not in prompt:
        raise ValueError(
            "re-implemented state renderer disagrees with the episode rendering"
        )
    for clause in audit["legality_clauses"]:
        if clause not in rules:
            raise ValueError("legality clause missing from the extracted rules")
    return {
        "scene": scene,
        "rules": rules,
        "listing": listing,
        "renders": renders,
    }


def assign_positions(rows: list[dict]) -> list[str]:
    """Deterministic exact 100/100 position balance from the construction
    seed: per-formalism 13/12 A-splits (even/odd formalism index), each
    shuffled by a formalism-keyed stream."""
    labels: list[str | None] = [None] * len(rows)
    for f_index, formalism in enumerate(FORMALISMS):
        indices = [
            index for index, row in enumerate(rows) if row["surface"] == formalism
        ]
        if len(indices) != PER_FORMALISM:
            raise ValueError(f"formalism balance changed for {formalism}")
        n_a = 13 if f_index % 2 == 0 else 12
        f_labels = ["A"] * n_a + ["B"] * (PER_FORMALISM - n_a)
        random.Random(f"{CONSTRUCTION_SEED}:{formalism}:position").shuffle(f_labels)
        for index, label in zip(indices, f_labels):
            labels[index] = label
    if labels.count("A") != ROWS_PER_POSITION or labels.count("B") != ROWS_PER_POSITION:
        raise ValueError("position assignment is not exactly 100/100")
    return [label for label in labels if label is not None]


def parse_true_fix(row: dict) -> tuple[int, str]:
    match = ANSWER_PATTERN.match(row["answer"])
    if not match:
        raise ValueError(f"episode answer format changed: {row['answer']!r}")
    step = int(match.group(1))
    change = match.group(2)
    if step != row["_audit"]["bug_step"] or "\n" in change:
        raise ValueError("episode answer disagrees with the audit bug step")
    return step, change


def audit_marker_tokens(prompt: str) -> None:
    """Fail closed on any provenance-marker word anywhere in the prompt."""
    for pattern, token in _MARKER_PATTERNS:
        if pattern.search(prompt):
            raise ValueError(f"2AFC prompt carries a provenance marker: {token!r}")


def build_choice_item(row: dict, correct_position: str) -> dict:
    """Convert one audited episode instance into one fully symmetric 2AFC
    verification item (rendered fresh; no repair history, no attempt
    narration, no provenance markers, no option self-reference)."""
    if correct_position not in POSITIONS:
        raise ValueError(f"invalid position label: {correct_position}")
    finished_b_written = reverify_fixes(row)
    parts = parse_episode(row)
    audit = row["_audit"]
    spec = audit["spec"]
    formalism = audit["formalism"]
    describe = DESCRIBE_BY_FORMALISM[formalism]
    true_step, true_change = parse_true_fix(row)
    wrong_step = audit["wrong_attempt_step"]
    wrong_change = audit["wrong_attempt"]
    # The described candidates must equal the module's own descriptions of
    # the audited ops (describer coherence, fail-closed).
    if (
        describe(tuple(spec["true_fix"][1])) != true_change
        or describe(tuple(spec["wrong_fix"][1])) != wrong_change
    ):
        raise ValueError("candidate descriptions disagree with the audited ops")
    if (true_step, true_change) == (wrong_step, wrong_change):
        raise ValueError("correct and wrong options coincide")
    if correct_position == "A":
        a_step, a_change = true_step, true_change
        b_step, b_change = wrong_step, wrong_change
    else:
        a_step, a_change = wrong_step, wrong_change
        b_step, b_change = true_step, true_change
    renders = parts["renders"]
    finished_b_render = render_state(
        formalism, spec["vocabulary"], finished_b_written
    )
    if finished_b_render == renders["wanted_b"]:
        raise ValueError("trial two does not show failure evidence")
    choice_prompt = (
        parts["scene"]
        + "\n"
        + parts["rules"]
        + "\nSteps as written:\n"
        + "\n".join(parts["listing"])
        + "\n"
        + CHOICE_SECTION_TEMPLATE.format(
            start_a=renders["start_a"],
            wanted_a=renders["wanted_a"],
            finished_a=renders["finished_a"],
            start_b=renders["start_b"],
            wanted_b=renders["wanted_b"],
            finished_b=finished_b_render,
            a_step=a_step,
            a_change=a_change,
            b_step=b_step,
            b_change=b_change,
        )
    )
    audit_marker_tokens(choice_prompt)
    # Self-reference checks: each labeled option line and each quoted change
    # text appears ONLY in the option block. The rules line is excluded from
    # the quoted-text scope because the machine documentation quotes every
    # op form symmetrically (a parameterless op's description is identical
    # to its documented form), and the unquoted step listing is the
    # machine's own program, audited separately below.
    option_lines = {
        "A": f"A. change step {a_step} to '{a_change}'",
        "B": f"B. change step {b_step} to '{b_change}'",
    }
    for line in option_lines.values():
        if choice_prompt.count("\n" + line + "\n") != 1:
            raise ValueError("an option line self-references inside the prompt")
    prompt_lines = choice_prompt.split("\n")
    if prompt_lines[1] != parts["rules"]:
        raise ValueError("2AFC prompt structure changed (rules line)")
    outside_rules = "\n".join(prompt_lines[:1] + prompt_lines[2:])
    for change in {a_change, b_change}:
        quoted = f"'{change}'"
        expected = sum(1 for value in (a_change, b_change) if value == change)
        if outside_rules.count(quoted) != expected:
            raise ValueError("a quoted option text appears outside its option line")
    listing_descriptions = {describe(op) for op in map(tuple, spec["written"])}
    return {
        "messages": [{"role": "user", "content": choice_prompt}],
        "answer": f"ANSWER: {correct_position}",
        "kind": PROBE_KIND,
        "surface": row["surface"],
        "level": row["level"],
        "correct_position": correct_position,
        "options": {
            "A": {"step": a_step, "change": a_change},
            "B": {"step": b_step, "change": b_change},
        },
        "true_fix": {"step": true_step, "change": true_change},
        "wrong_fix": {"step": wrong_step, "change": wrong_change},
        "episode_prompt_sha256": sha256_bytes(
            row["messages"][0]["content"].encode()
        ),
        "episode_answer": row["answer"],
        "_audit": {
            **{key: value for key, value in audit.items() if key != "spec"},
            "reexecuted_fix_properties": True,
            "trial_two_written_fails": True,
            "marker_tokens_clean": True,
            "option_text_in_step_listing": {
                "true_fix": true_change in listing_descriptions,
                "wrong_fix": wrong_change in listing_descriptions,
            },
            "finished_b_written": finished_b_written,
            "spec": audit["spec"],
        },
    }


def listing_collision_audit(source_rows: list[dict]) -> dict:
    """The frozen artifact-ceiling reading: a candidate change can coincide
    with a step already in the written sequence; record how often, per
    option, and the best accuracy any collision-keyed guessing heuristic
    could reach (guessing 0.5 wherever the collision pattern is symmetric)."""
    both = neither = true_only = wrong_only = 0
    for row in source_rows:
        flags = row["_audit"]["option_text_in_step_listing"]
        true_hit, wrong_hit = flags["true_fix"], flags["wrong_fix"]
        if true_hit and wrong_hit:
            both += 1
        elif true_hit:
            true_only += 1
        elif wrong_hit:
            wrong_only += 1
        else:
            neither += 1
    total = len(source_rows)
    ceiling = (0.5 * (both + neither) + max(true_only, wrong_only)) / total
    return {
        "both_collide": both,
        "neither_collides": neither,
        "true_fix_only_collides": true_only,
        "wrong_fix_only_collides": wrong_only,
        "collision_heuristic_ceiling": ceiling,
        "note": (
            "a change text coinciding with a listed step is intrinsic to the "
            "task (a fix often restores an op used elsewhere) and carries no "
            "provenance; this reading freezes the best accuracy a "
            "collision-keyed guesser could reach so the artifact ceiling is "
            "on file next to the 0.65 signal bar"
        ),
    }


def check_banned_vocabulary_2afc(rows: list[dict]) -> None:
    """The reused banned-token scan, applied to the 2AFC prompts/answers."""
    patterns = [
        (re.compile(rf"\b{re.escape(token)}\b"), token)
        for token in feedloop.BANNED_PROMPT_TOKENS
    ]
    for index, row in enumerate(rows):
        haystack = "\n".join((row["messages"][0]["content"], row["answer"]))
        for pattern, token in patterns:
            if pattern.search(haystack):
                raise ValueError(f"probe item {index} leaks banned vocabulary: {token!r}")


def select_instances(pool: list[dict]) -> tuple[list[dict], dict]:
    """Apply the frozen deterministic selection rule to the oversampled pool:
    per formalism, in generator stream order, keep the first 25 instances
    whose WRITTEN sequence also fails trial two."""
    selected: list[dict] = []
    stats: dict[str, dict] = {
        formalism: {
            "pool": 0,
            "excluded_trial_two_success_before_quota": 0,
            "selected": 0,
            "unused_after_quota": 0,
        }
        for formalism in FORMALISMS
    }
    for row in pool:
        formalism = row["surface"]
        entry = stats[formalism]
        entry["pool"] += 1
        if entry["selected"] == PER_FORMALISM:
            entry["unused_after_quota"] += 1
            continue
        spec = row["_audit"]["spec"]
        if written_trial_two_outcome(row) == spec["wanted_b"]:
            entry["excluded_trial_two_success_before_quota"] += 1
            continue
        entry["selected"] += 1
        selected.append(row)
    short = {
        formalism: entry["selected"]
        for formalism, entry in stats.items()
        if entry["selected"] != PER_FORMALISM
    }
    if short:
        raise ValueError(f"selection quota not met from the pool: {short}")
    if len(selected) != PROBE_ROWS:
        raise ValueError("selection produced the wrong number of instances")
    return selected, {
        "pool_mix": OVERSAMPLE_MIX,
        "pool_rows": len(pool),
        "rule": SELECTION_RULE,
        "per_formalism": stats,
        "excluded_total": sum(
            entry["excluded_trial_two_success_before_quota"]
            for entry in stats.values()
        ),
    }


def build_probe_rows() -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    """Return (pool, selected instances, 2AFC source rows, runner rows,
    selection stats)."""
    pool = feedloop.generate_curriculum(OVERSAMPLE_MIX, CONSTRUCTION_SEED)
    summary = feedloop.validate_generated(pool)
    feedloop.check_banned_vocabulary(pool)
    expected_surfaces = {
        formalism: POOL_PER_FORMALISM
        for formalism in feedloop.FEEDLOOP_FORMALISMS
    }
    if (
        summary["rows"] != POOL_ROWS
        or summary["kinds"] != {"u_feedloop": POOL_ROWS}
        or summary["surfaces"] != expected_surfaces
    ):
        raise ValueError("instance pool no longer has 40 rows per formalism")
    if tuple(FORMALISMS) != tuple(feedloop.FEEDLOOP_FORMALISMS):
        raise ValueError("frozen formalism order drifted from the generator")
    selected, selection_stats = select_instances(pool)
    positions = assign_positions(selected)
    source_rows: list[dict] = []
    for index, (instance, position) in enumerate(zip(selected, positions)):
        item = build_choice_item(instance, position)
        item["task_id"] = f"{ID_PREFIX}{index:03d}"
        source_rows.append(item)
    check_banned_vocabulary_2afc(source_rows)
    prompts = {row["messages"][0]["content"] for row in source_rows}
    if len(prompts) != PROBE_ROWS:
        raise ValueError("probe items collide on prompts")
    counts = Counter(row["correct_position"] for row in source_rows)
    if counts != Counter({"A": ROWS_PER_POSITION, "B": ROWS_PER_POSITION}):
        raise ValueError(f"probe position balance out of range: {dict(counts)}")
    runner_rows = [
        {
            "id": row["task_id"],
            "messages": row["messages"],
            "meta": {
                "kind": row["kind"],
                "surface": row["surface"],
                "seed": CONSTRUCTION_SEED,
                "instrument": INSTRUMENT,
            },
        }
        for row in source_rows
    ]
    if any(set(row) != {"id", "messages", "meta"} for row in runner_rows):
        raise ValueError("probe runner input schema leaked hidden fields")
    return pool, selected, source_rows, runner_rows, selection_stats


def overlap_receipt(pool: list[dict], source_rows: list[dict]) -> dict:
    probe_messages = {message_bytes(row) for row in source_rows}
    if len(probe_messages) != PROBE_ROWS:
        raise ValueError("probe items collide on canonical user messages")
    episode_messages = {message_bytes(row) for row in pool}
    episode_overlap = len(probe_messages & episode_messages)
    if episode_overlap:
        raise ValueError("2AFC prompts collide with their own episode renderings")
    dose_scale_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in DOSE_SCALE_SOURCES:
        messages = {
            message_bytes(row)
            for row in load_frozen(path, expected_sha256, expected_rows)
            if row.get("messages")
        }
        overlap = len(probe_messages & messages)
        if overlap:
            raise ValueError(
                f"fresh probe prompts overlap a dose-scale artifact: {path}"
            )
        dose_scale_sources[path.relative_to(ROOT).as_posix()] = {
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
        overlap = len(probe_messages & messages)
        if overlap:
            raise ValueError(
                f"fresh probe prompts overlap an inherited corpus or stream: {path}"
            )
        predecessor_sources[path.relative_to(ROOT).as_posix()] = {
            "sha256": expected_sha256,
            "messages_compared": len(messages),
            "overlap": overlap,
        }
    # Regenerated dose-scale rows at the frozen construction/holdout seeds:
    # covers the treatment corpus and its holdout by construction and pins
    # the reused generator's behavior.
    regenerated_treatment = {
        message_bytes(row)
        for row in feedloop.generate_curriculum(
            feedloop.ARM_MIX, DOSE_SCALE_CONSTRUCTION_SEED
        )
    }
    treatment_overlap = len(probe_messages & regenerated_treatment)
    if treatment_overlap:
        raise ValueError(
            "fresh probe prompts overlap regenerated dose-scale treatment rows"
        )
    regenerated_holdout = {
        message_bytes(row)
        for row in feedloop.generate_curriculum(
            feedloop.HOLDOUT_MIX, DOSE_SCALE_HOLDOUT_SEED
        )
    }
    holdout_overlap = len(probe_messages & regenerated_holdout)
    if holdout_overlap:
        raise ValueError(
            "fresh probe prompts overlap regenerated dose-scale holdout rows"
        )
    prior_messages = {
        message_bytes(row)
        for seed in PRIOR_LOCAL_SEEDS
        for row in curriculum.generate_curriculum(curriculum.SMOKE_MIX, seed)
    }
    prior_local_overlap = len(probe_messages & prior_messages)
    if prior_local_overlap:
        raise ValueError("fresh probe prompts overlap prior local seeds")
    predecessor_gates: dict[str, dict] = {}
    for gate_path, gate_sha256, gate_rows in PREDECESSOR_GATES:
        predecessor_messages = {
            message_bytes(row)
            for row in load_frozen(gate_path, gate_sha256, gate_rows)
            if row.get("messages")
        }
        predecessor_overlap = len(probe_messages & predecessor_messages)
        if predecessor_overlap:
            raise ValueError(
                f"fresh probe prompts overlap a predecessor's frozen gate: {gate_path}"
            )
        predecessor_gates[gate_path.relative_to(ROOT).as_posix()] = {
            "sha256": gate_sha256,
            "messages_compared": len(predecessor_messages),
            "overlap": predecessor_overlap,
        }
    return {
        "message_sha256s": sorted(sha256_bytes(value) for value in probe_messages),
        "unique_probe_messages": len(probe_messages),
        "own_episode_renderings": {
            "messages_compared": len(episode_messages),
            "overlap": episode_overlap,
            "note": (
                "the 2AFC prompts are rendered fresh (no attempt narration, "
                "new trial section, option block), so the raw episode "
                "renderings of the full 320-instance pool at this cell's own "
                "seed are disjoint by construction and checked anyway"
            ),
        },
        "dose_scale_sources": dose_scale_sources,
        "predecessor_sources": predecessor_sources,
        "regenerated_dose_scale_treatment": {
            "mix": feedloop.ARM_MIX,
            "construction_seed": DOSE_SCALE_CONSTRUCTION_SEED,
            "messages_compared": len(regenerated_treatment),
            "overlap": treatment_overlap,
        },
        "regenerated_dose_scale_holdout": {
            "mix": feedloop.HOLDOUT_MIX,
            "seed": DOSE_SCALE_HOLDOUT_SEED,
            "messages_compared": len(regenerated_holdout),
            "overlap": holdout_overlap,
        },
        "prior_local_seeds_compared": list(PRIOR_LOCAL_SEEDS),
        "prior_local_mix": curriculum.SMOKE_MIX,
        "prior_local_messages_compared": len(prior_messages),
        "prior_local_overlap": prior_local_overlap,
        "predecessor_gates": predecessor_gates,
        "training_streams": {
            "paths": [],
            "policy": (
                "this cell materializes no training stream and trains "
                "nothing; the dose-scale cell's two 2,280-row streams are "
                "hash-pinned and checked above"
            ),
        },
    }


def build_outputs(*, authenticate_composites: bool = True) -> dict[Path, bytes]:
    for path in CODE_FILES.values():
        if not path.is_file():
            raise ValueError(f"required local-design input is absent: {path}")
    composite_receipt = authenticate_composite_receipt()
    if authenticate_composites:
        authenticate_composite_tree()
    pool, _selected, source_rows, runner_rows, selection_stats = build_probe_rows()
    source = jsonl_bytes(source_rows)
    runner_input = jsonl_bytes(runner_rows)
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "repair_verifier_signal_probe_design",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "loaded": False,
            "calls": 0,
        },
        "seed": CONSTRUCTION_SEED,
        "seed_freshness": {
            "construction_seed": CONSTRUCTION_SEED,
            "runner_seed": CONSTRUCTION_SEED,
            "substitution_required": False,
            "verified": (
                "77160 verified grep-fresh in seed contexts across "
                "experiments/, knowledge/, research_programs/, scripts/, "
                "configs/, and docs/ at design time (zero seed-context "
                "hits; 77150 is the dose-scale cell's construction seed "
                "immediately below this block). The runner seed reuses "
                "77160 — both engine runs are greedy, so the seed is "
                "recorded provenance, not a sampling input; no new seed "
                "is drawn"
            ),
            "rule_if_collision": "next free integer, recorded here",
        },
        "feasibility_probe": True,
        "trains_nothing": True,
        "rows": PROBE_ROWS,
        "judgments_total": PROBE_ROWS * len(ARMS),
        "probe": {
            "generator": "scripts/gen_feedloop_curriculum.py (reused machinery)",
            "instance_pool_mix": OVERSAMPLE_MIX,
            "instance_pool_rows": POOL_ROWS,
            "selection": selection_stats,
            "rows": PROBE_ROWS,
            "per_formalism": PER_FORMALISM,
            "formalisms": list(FORMALISMS),
            "kind": PROBE_KIND,
            "id_prefix": ID_PREFIX,
            "instrument": INSTRUMENT,
            "construction": (
                "fully symmetric 2AFC rendered fresh per instance: machine "
                "spec with legality clauses + the original written sequence "
                "+ BOTH trials' setups, wanted outcomes, and the observed "
                "outcomes of the original (broken) sequence on both trials "
                "(pure failure evidence, no repair history, no attempt "
                "narration) + two candidate single-step changes labeled A "
                "and B in identical grammatical form — the unique legal "
                "both-trials fix and the legal trial-one-consistent "
                "distractor — + the ask for the letter of the one change "
                "that makes both trials come out as wanted; deciding "
                "requires simulating each candidate against both trials "
                "(execution-based fix verification)"
            ),
            "position_balance": {
                "a_correct": ROWS_PER_POSITION,
                "b_correct": ROWS_PER_POSITION,
                "assignment": (
                    "deterministic from the construction seed: per-formalism "
                    "13/12 A-splits (even/odd formalism index) shuffled by "
                    "the stream random.Random(f'{seed}:{formalism}:position')"
                ),
            },
            "marker_token_audit": {
                "tokens": list(MARKER_TOKENS),
                "scope": "every rendered 2AFC prompt, word-boundary, case-insensitive",
                "hits": 0,
                "fail_closed": True,
            },
            "self_reference_audit": {
                "option_line_unique": True,
                "quoted_change_text_only_in_option_block": True,
                "rules_line_excluded_from_quoted_scope": (
                    "the machine documentation quotes every op form "
                    "symmetrically (a parameterless op's description equals "
                    "its documented form), so it cannot mark either option"
                ),
            },
            "listing_collision_audit": listing_collision_audit(source_rows),
            "reverified_per_item": [
                "written run reproduces the recorded trial-one evidence",
                "written run FAILS trial two (pure failure evidence; "
                "selection-rule enforced and re-checked fail-closed)",
                "correct fix lands on the wanted outcome in both trials",
                "distractor lands on the wanted outcome in trial one only "
                "and reproduces the recorded trial-two failure",
                "re-implemented state renderers verified against the "
                "generator's own episode rendering",
                "candidate descriptions equal the module's own op "
                "descriptions",
                "no provenance-marker token anywhere in the prompt",
                "option lines and quoted change texts appear only in the "
                "option block",
                "generator audits (>=2 round-1 candidates, unique round-2 "
                "survivor, extended-grammar legality) re-checked per item",
            ],
            "kinds": dict(
                sorted(Counter(row["kind"] for row in source_rows).items())
            ),
            "surfaces": dict(
                sorted(Counter(row["surface"] for row in source_rows).items())
            ),
            "source": {
                "path": SOURCE.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(source),
                "contains_executable_truth": True,
            },
            "runner_input": {
                "path": RUNNER_INPUT.relative_to(ROOT).as_posix(),
                "sha256": sha256_bytes(runner_input),
                "schema": ["id", "messages", "meta"],
                "contains_answer": False,
                "contains_oracle": False,
            },
        },
        "answer_normalization": ANSWER_NORMALIZATION,
        "freshness": overlap_receipt(pool, source_rows),
        "backend": {
            "name": "vllm_merged_composite",
            "arm_thinking": dict(ARM_THINKING),
            "greedy": True,
            "samples_per_task": 1,
            "max_tokens": MAX_TOKENS,
            "max_model_len": 4096,
            "max_num_seqs": 16,
            "max_num_batched_tokens": 8192,
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
            "run_seed": CONSTRUCTION_SEED,
            "same_runner_and_geometry_for_every_arm": True,
            "runtime_lora_forbidden": True,
        },
        "arms": list(ARMS),
        "gating_arm": GATING_ARM,
        "candidates": [],
        "run_order": {
            "policy": (
                "think first, then nothink — two sequential authenticated "
                "engine events on the one evaluated composite"
            ),
            "sequence": [f"{arm}_seed{CONSTRUCTION_SEED}" for arm in ARMS],
        },
        "readings": {
            "no_promotion": True,
            "2afc_accuracy": (
                "per arm: correct/200 with the exact (Clopper-Pearson) "
                "two-sided 95% binomial CI against the 0.5 chance floor"
            ),
            "per_formalism_accuracy": "25 items per formalism per arm, descriptive",
            "position_bias": (
                "accuracy on A-correct vs B-correct items (100 each), "
                "descriptive"
            ),
            "consequence_partition": {
                "basis": "think arm only; the nothink arm is descriptive",
                "signal_min_accuracy": SIGNAL_MIN_ACCURACY,
                "chance_floor": CHANCE_FLOOR,
                "confidence": CONFIDENCE,
                "statements": dict(CONSEQUENCES),
                "ordered_total_no_third_state": True,
            },
            "cap_contact_diagnostic": {
                "preregistered": True,
                "scope_threshold": CAP_SCOPE_THRESHOLD,
                "rule": (
                    "simulating two candidates over two trials each may "
                    "press the 1,024-token think cap; a SIGNAL_ABSENT "
                    "reading with think-arm cap contacts on more than 20% "
                    "of items is annotated as possibly budget-limited (the "
                    "verdict space stays two-state)"
                ),
            },
            "nothink_role": "C47 substrate-scoping check, reported not gated",
            "complete_event_always_exits_zero": True,
            "no_aggregate_seed_exists": True,
        },
        "prerequisites": {
            "evaluated_composite": {
                "label": COMPOSITE_LABEL,
                "merge_receipt": COMPOSITE_RECEIPT.relative_to(ROOT).as_posix(),
                "merge_receipt_sha256": EXPECTED_RECEIPT_SHA256,
                "composite": MERGED.relative_to(ROOT).as_posix(),
                "tree_sha256": EXPECTED_TREE_SHA256,
                "weights_sha256": EXPECTED_WEIGHTS_SHA256,
                "files": EXPECTED_FILES,
                "receipt_authenticated": composite_receipt["name"]
                == COMPOSITE_LABEL,
                "tree_recomputed_at_boundary": True,
            },
            "standalone_lineage_package": {
                "manifest": "data/lineage/lineage_manifest.json",
                "verify": "scripts/rebuild_lineage.py --verify-inputs",
                "root_adapter_vendored": (
                    "large_artifacts/qwen35_4b_repair_verifier_signal_probe/"
                    "lineage_root/blend"
                ),
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
    return {
        SOURCE: source,
        RUNNER_INPUT: runner_input,
        RECEIPT: rendered_receipt,
    }


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
                "rows": PROBE_ROWS,
                "source_sha256": sha256_bytes(outputs[SOURCE]),
                "runner_input_sha256": sha256_bytes(outputs[RUNNER_INPUT]),
                "receipt_sha256": sha256_bytes(outputs[RECEIPT]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
