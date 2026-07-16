#!/usr/bin/env python3
"""Freeze the 800-row feedloop dose-scale treatment corpus, manifest, audits.

The corpus is the 800-row feedloop-only mix (100 rows per formalism: the
reference cell's ``troughline``/``trinketcord``/``crankwheel``/``sigilslate``
reused as FRESH instances, plus the new ``barrowyoke``/``balesled``/
``millround``/``skeinreel``) rendered by gen_feedloop_curriculum.py at the
frozen construction seed 77150. Everything is a pure function of that seed;
``--check`` regenerates and byte-compares. Beyond the generator's
banned-vocabulary scan (which bans every predecessor surface pool INCLUDING
the statechain formalisms' pools, retaining only the four reused feedloop
formalisms' own nouns), this builder runs TWO audits against the replay pool
and EVERY predecessor corpus, training stream, and frozen local gate in the
arms' lineages — INCLUDING the reference cell's treatment corpus and gate
files AND the statechain cell's treatment corpus, materialized streams, and
four gate input files (fail-closed hash pins):

- fresh-surface grep audit: every claimed fresh token (the four new
  formalisms' vocabulary) must have zero case-insensitive word-boundary hits
  in every pinned source;
- row-overlap audit: the generated corpus must share zero canonical user
  messages with every pinned source — this is the receipt that the reused
  troughline/trinketcord/crankwheel/sigilslate rows are genuinely FRESH
  instances, not copies of the reference cell's rows.

The retained feedloop vocabulary is documented as INHERITED and deliberately
excluded from the fresh-surface claim (it appears in the reference corpus by
design).
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

import gen_feedloop_curriculum as feedloop  # noqa: E402

CONSTRUCTION_SEED = 77150
CORPUS_PATH = EXP / "data" / "sft_feedloop_scale.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
EXPECTED_KINDS = {"u_feedloop": 800}
EXPECTED_SURFACES = {formalism: 100 for formalism in feedloop.FEEDLOOP_FORMALISMS}
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
# Every inherited corpus and materialized training stream in the lineages of
# the parent composite and its published siblings (fail-closed hash pins),
# copied from the statechain cell's audited inventory, plus this cell's own
# byte-identical replay pool, plus the reference cell's OWN treatment corpus,
# materialized streams, and frozen local gate input files, plus the
# statechain cell's OWN treatment corpus, materialized streams, and frozen
# local gate input files.
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
    tokens = tuple(dict.fromkeys(feedloop.FRESH_SURFACE_TOKENS))
    if len(tokens) != len(feedloop.FRESH_SURFACE_TOKENS):
        raise ValueError("fresh surface token inventory contains duplicates")
    inherited = tuple(dict.fromkeys(feedloop.INHERITED_SURFACE_TOKENS))
    if len(inherited) != len(feedloop.INHERITED_SURFACE_TOKENS):
        raise ValueError("inherited surface token inventory contains duplicates")
    if set(tokens) & set(inherited):
        raise ValueError("a token is claimed both fresh and inherited")
    if set(tokens) & set(feedloop.BANNED_PROMPT_TOKENS) or set(inherited) & set(
        feedloop.BANNED_PROMPT_TOKENS
    ):
        raise ValueError("a claimed surface token is banned")
    patterns = {
        token: re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        for token in tokens
    }
    corpus_messages = {message_bytes(row) for row in corpus_rows}
    if len(corpus_messages) != len(corpus_rows):
        raise ValueError("treatment corpus collides on canonical user messages")
    surface_sources: dict[str, dict] = {}
    overlap_sources: dict[str, dict] = {}
    for path, expected_sha256, expected_rows in PREDECESSOR_SOURCES:
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
                "troughline/trinketcord/crankwheel/sigilslate surfaces are "
                "retained from the reference cell by design; they appear in "
                "the reference corpus and are excluded from the fresh-surface "
                "claim"
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
    rows = feedloop.generate_curriculum(feedloop.ARM_MIX, CONSTRUCTION_SEED)
    summary = feedloop.validate_generated(rows)
    feedloop.check_banned_vocabulary(rows)
    balance = feedloop.check_corpus_balance(rows)
    if (
        len(rows) != 800
        or summary["kinds"] != EXPECTED_KINDS
        or summary["surfaces"] != EXPECTED_SURFACES
    ):
        raise ValueError(f"feedloop dose-scale corpus mix changed: {summary}")
    surface_audit, overlap_audit = check_fresh_surfaces_and_overlap(rows)

    corpus = "".join(
        json.dumps(feedloop.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    banned = tuple(feedloop.BANNED_PROMPT_TOKENS)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "generator": "scripts/gen_feedloop_curriculum.py",
        "banned_vocabulary_checked": True,
        "banned_vocabulary": {
            "tokens": len(banned),
            "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
            "extended_with_statechain_surface_pools": True,
            "retained_feedloop_surfaces_not_banned": True,
        },
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": len(rows),
            "mix": feedloop.ARM_MIX,
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "sha256": sha256_bytes(corpus),
        },
        "dose_scale_contract": {
            "single_kind_dose": True,
            "dose_rows": 800,
            "reference_failed_dose_rows": 80,
            "dose_multiple": 10,
            "formalisms": list(feedloop.FEEDLOOP_FORMALISMS),
            "retained_formalisms": list(feedloop.REUSED_FORMALISMS),
            "new_formalisms": list(feedloop.NEW_FORMALISMS),
            "min_round1_candidates": 2,
            "wrong_attempt_in_round1_candidates": True,
            "unique_fix_after_round2": True,
            "parameterized_ops_bounded_in_spec_text": True,
            "extended_probe_amount_bound": feedloop.EXTENDED_AMOUNT_BOUND,
            "extended_probe_scope_by_formalism": dict(
                feedloop.EXTENDED_PROBE_SCOPE
            ),
            "container_names_probed_over_full_pool": [
                "troughline", "barrowyoke",
            ],
            "extended_survivors_out_of_bound_only": True,
            "retained_instances_fresh_via_row_overlap_audit": True,
        },
        "balance": balance,
        "surface_freshness_audit": surface_audit,
        "row_overlap_audit": overlap_audit,
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
            parser.error(f"refusing to overwrite a differing frozen corpus artifact: {conflicts[0]}")
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
