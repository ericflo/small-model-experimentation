#!/usr/bin/env python3
"""Freeze the episode treatment corpus, its manifest, and the surface audit.

The corpus is the 160-row designed episode mix (80 ``u_feedloop`` + 80
``u_statechain``) rendered by gen_episode_curriculum.py at the frozen
construction seed. Everything is a pure function of that seed; ``--check``
regenerates and byte-compares. Beyond the generator's banned-vocabulary scan,
this builder grep-audits every fresh surface token against the replay pool
and EVERY predecessor corpus and training stream in the arms' lineages
(fail-closed hash pins): zero word-boundary hits are allowed, so the episode
surfaces are provably disjoint from everything the parent ever trained on.
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

import gen_episode_curriculum as episode  # noqa: E402

CONSTRUCTION_SEED = 77130
CORPUS_PATH = EXP / "data" / "sft_feedloop_state.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
EXPECTED_KINDS = {"u_feedloop": 80, "u_statechain": 80}
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
# Every inherited corpus and materialized training stream in the lineages of
# the parent composite and its published siblings (fail-closed hash pins),
# copied from the retention-calibration cell's audited inventory, plus this
# cell's own byte-identical replay pool.
PREDECESSOR_SOURCES = (
    (REPLAY_PATH, REPLAY_SHA256, REPLAY_ROWS),
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


def check_fresh_surfaces() -> dict:
    """Grep-audit every fresh surface token against every predecessor corpus."""
    tokens = tuple(dict.fromkeys(episode.FRESH_SURFACE_TOKENS))
    if len(tokens) != len(episode.FRESH_SURFACE_TOKENS):
        raise ValueError("fresh surface token inventory contains duplicates")
    patterns = {
        token: re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE)
        for token in tokens
    }
    sources: dict[str, dict] = {}
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
        sources[path.relative_to(ROOT).as_posix()] = {
            "sha256": expected_sha256,
            "rows": expected_rows,
            "tokens_checked": len(tokens),
            "hits": 0,
        }
    return {
        "tokens": list(tokens),
        "token_inventory_sha256": sha256_bytes("\n".join(tokens).encode("utf-8")),
        "match_rule": "case-insensitive word-boundary regex, zero hits required",
        "sources": sources,
    }


def build() -> tuple[dict[Path, bytes], dict]:
    rows = episode.generate_curriculum(episode.ARM_MIX, CONSTRUCTION_SEED)
    summary = episode.validate_generated(rows)
    episode.check_banned_vocabulary(rows)
    balance = episode.check_corpus_balance(rows)
    if len(rows) != 160 or summary["kinds"] != EXPECTED_KINDS:
        raise ValueError(f"episode corpus mix changed: {summary['kinds']}")
    surface_audit = check_fresh_surfaces()

    corpus = "".join(
        json.dumps(episode.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    banned = tuple(episode.BANNED_PROMPT_TOKENS)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "generator": "scripts/gen_episode_curriculum.py",
        "banned_vocabulary_checked": True,
        "banned_vocabulary": {
            "tokens": len(banned),
            "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
        },
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": len(rows),
            "mix": episode.ARM_MIX,
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "sha256": sha256_bytes(corpus),
        },
        "feedloop_uniqueness_contract": {
            "parameterized_ops_bounded_in_spec_text": True,
            "bounded_enumeration_exhaustive": True,
            "extended_probe_amount_bound": episode.EXTENDED_AMOUNT_BOUND,
            "extended_probe_items": "full module pools",
            "out_of_bound_alternatives_excluded_by_documented_legality_only": True,
            "ambiguity_rule": (
                "a draw whose bounded round-two survivor set is not exactly "
                "the true fix is rejected and redrawn from the same "
                "deterministic rng stream (attempt cap 5000)"
            ),
        },
        "balance": balance,
        "surface_freshness_audit": surface_audit,
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
        "balance": manifest["balance"],
        "surface_tokens_checked": len(manifest["surface_freshness_audit"]["tokens"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
