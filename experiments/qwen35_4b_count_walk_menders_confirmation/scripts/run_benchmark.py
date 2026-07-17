#!/usr/bin/env python3
"""Run the four sealed per-seed CONFIRMATION events through the trusted gateway.

EVAL-ONLY MEASUREMENT, not a promotion: no training, no merging, no
corpus, no candidate gate exists anywhere in this cell. Lifecycle 27's
sealed seed-78163 MECHANISM_ANSWER (count_walk menders 0.1 with base,
zero_root_parent, and replay_ctl7 all at exactly 0.0) is tested for
replication on FOUR independent sealed fresh seeds — 78164, 78165,
78166, 78167 — at tier medium, think budget 1024, FOUR arms per seed in
the frozen order base, zero_root_parent, replay_ctl7, count_walk,
seed-major (a seed's four arms complete and close before the next seed
opens). Sixteen gateway runs total. Discipline copied from the hardened
count-don't-walk pilot runner and the goal-gate confirmation runner:

- the PASS_BENCHMARK_EVENT review verdict, the committed preregistration,
  and the committed provenance copies are enforced HERE, at the
  seed-consuming boundary: a direct invocation cannot consume any seed
  with an unreviewed or drifted contract;
- clean pushed main plus the committed-at-HEAD preregistration, review,
  the two lifecycle-27 merge receipts, lifecycle 22's zero-root lineage
  merge receipt, the sha-pinned prior-event summary, and the four
  byte-identical provenance copies in ``data/provenance/`` are hard
  prerequisites;
- every arm authenticates fail-closed by recomputing its full on-disk
  tree sha256 (which covers the 9GB weights) against pins baked as
  frozen constants at design time — no TODO-PIN slots exist because all
  four composites pre-exist this cell; the two trained arms additionally
  authenticate against lifecycle 27's committed merge receipts (by
  sha256 AND payload), the zero-root parent against lifecycle 22's
  committed lineage merge receipt, and base against its reserialization
  receipt hash;
- K-SEED WRITE-AHEAD ledger: each seed gets its own ``opened`` record
  appended before its first gateway call and its own ``closed`` record
  after its per-seed summary. The closed record sha-pins the summary AND
  ALL FOUR per-arm gateway receipts, so every verdict input is
  provenance-anchored at close time. The only valid ledger history is a
  prefix of the canonical seed-major sequence; a closed record refuses
  its seed forever (completed seeds are never re-run); a crash mid-seed
  leaves a permanent opened record that forces recovery through
  ``--resume`` with the preserved receipts. RECOVERY SEMANTICS: an
  UNOPENED seed requires a clean slate — pre-existing
  receipt/failure/summary files in a never-opened seed's event directory
  refuse unconditionally. If a crash landed in the window between the
  summary write and the closed-record append, the summary is regenerated
  deterministically from the authenticated receipts and must compare
  BYTE-IDENTICAL to the file on disk — a match appends the closed record
  and continues, a mismatch refuses loudly with both digests. Once all
  four seeds close, the confirmation budget is spent and every new
  invocation refuses, resume or not;
- implementation-signature integrity per seed: BEFORE a seed's first
  gateway call the LIVE benchmark implementation signature — computed
  through the trusted gateway's own hash-only inventory functions
  (bytes hashed, never parsed) — must equal the prior event's pinned
  block (pre-consumption; a drifted suite refuses before any GPU run
  or opened record); afterwards the four receipts must share one
  (runner sha256, source inventory sha256, file count) signature AND
  match the same pinned block, before the seed's summary is written —
  all sixteen receipts are thereby anchored to the seed-78163 event,
  fail closed;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a
  NaN can never silently drop a family from any reading;
- ``within_budget`` and ``wall_seconds`` are recorded exactly as
  returned and never gated; the budget_integrity reading scopes the
  paired comparison instead.

After the fourth seed closes, the FROZEN REPLICATION RULE (integer-exact,
two-directional, over the four NEW events only — 78163 is prior
evidence, never pooled) is computed into the terminal readout:
``REPLICATED`` / ``NOT_REPLICATED`` / ``AMBIGUOUS``, no fourth state.
The benchmark suite directory is never read; only
scripts/run_benchmark_aggregate.py runs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPTS = EXP / "scripts"
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
PREREGISTRATION = EXP / "reports" / "preregistration.md"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
PUBLIC_FAMILIES = {
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack",
    "sirens", "stockade", "toolsmith", "warren",
}
GATEWAY_KEYS = {
    "schema_version", "stage", "tier", "think_budget", "seed", "backend", "model",
    "model_merge_receipt_sha256", "benchmark_runner_sha256",
    "benchmark_source_inventory_sha256", "benchmark_source_file_count",
    "aggregate", "per_family", "within_budget", "wall_seconds",
}
FROZEN_NAME = "confirmation"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
# Four fresh sealed seeds, verified grep-fresh in seed contexts across the
# repo at design time (audit recorded in reports/preregistration.md).
SEED_ORDER = (78164, 78165, 78166, 78167)
MENDERS_FAMILY = "menders"
FROZEN_PARENT = "zero_root_parent"
FROZEN_REPLAY_CONTROL = "replay_ctl7"
CANDIDATE = "count_walk"
MODEL_ORDER = ("base", FROZEN_PARENT, FROZEN_REPLAY_CONTROL, CANDIDATE)
CONTROL_ARMS = ("base", FROZEN_PARENT, FROZEN_REPLAY_CONTROL)
TREATED_ARMS = (FROZEN_PARENT, FROZEN_REPLAY_CONTROL, CANDIDATE)
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_zero_root_lineage_rebuild"
        / "merged" / "zero_root_hygiene_explore"
    ),
    FROZEN_REPLAY_CONTROL: (
        ROOT / "large_artifacts" / "qwen35_4b_count_dont_walk_enumeration"
        / "merged" / "replay_ctl7"
    ),
    CANDIDATE: (
        ROOT / "large_artifacts" / "qwen35_4b_count_dont_walk_enumeration"
        / "merged" / "count_walk"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. All four composites pre-exist this cell, so every pin is
# a design-time frozen constant — no TODO-PIN slot exists in this file.
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    FROZEN_PARENT: (
        "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
    ),
    FROZEN_REPLAY_CONTROL: (
        "044a4599ac5264e00256f66f65215ea497d3631d8aebd3467b698253648e484a"
    ),
    CANDIDATE: (
        "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    FROZEN_PARENT: (
        "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
    ),
    FROZEN_REPLAY_CONTROL: (
        "c5035b4db47e4da582a805ca009747a5618ef5badc35d960ca216e586dd3ab9d"
    ),
    CANDIDATE: (
        "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# The two trained arms carry committed merge receipts at lifecycle 27; the
# zero-root parent authenticates against lifecycle 22's committed lineage
# merge receipt; base's reserialization receipt lives inside the composite.
COMMITTED_MERGE_RECEIPTS = {
    FROZEN_REPLAY_CONTROL: (
        "experiments/qwen35_4b_count_dont_walk_enumeration"
        "/runs/merges/replay_ctl7.json",
        "3f65b4c6f4a8b0574a574a89d417c174c3762de6f93508bed8a5a987b91e224c",
    ),
    CANDIDATE: (
        "experiments/qwen35_4b_count_dont_walk_enumeration"
        "/runs/merges/count_walk.json",
        "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36",
    ),
}
EXPECTED_MERGE_RECEIPT_NAMES = {
    FROZEN_REPLAY_CONTROL: "replay_ctl7",
    CANDIDATE: "count_walk",
}
ZERO_ROOT_PARENT_MERGE_RECEIPT = (
    ROOT / "experiments" / "qwen35_4b_zero_root_lineage_rebuild"
    / "runs" / "lineage" / "merge.json"
)
ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256 = (
    "e906caea7c4b86f4a3eacb96affb7cc2fa9b7cc11e11b634b651cabc5dd01d2b"
)
BASE_MERGE_RECEIPT_SHA256 = (
    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
)
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
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
# The committed prior event at seed 78163 whose MECHANISM_ANSWER this cell
# tests for replication. Sha-pinned so a drifted source fails BEFORE any
# seed is consumed; reported alongside the verdict, NEVER pooled into it.
PRIOR_EVENT = {
    "seed": 78163,
    "tier": FROZEN_TIER,
    "think_budget": FROZEN_THINK_BUDGET,
    "summary": (
        "experiments/qwen35_4b_count_dont_walk_enumeration"
        "/runs/benchmark/medium_tb1024_seed78163_pilot/summary.json"
    ),
    "summary_sha256": (
        "a8c394758aeea8255389b1d7c2b6d7c3f37d6072d9ea226f1b4786a8eee191af"
    ),
    "counted_in_verdict": False,
}
# Every receipt of the sixteen must carry exactly this signature (the
# prior-event summary's benchmark_implementation block) or the event fails
# closed at its seed's summary boundary.
PRIOR_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
# Verification-aid copies of the four committed provenance documents
# (data/provenance/<copy> -> committed original). Byte-identity is
# enforced at the seed boundary and in smoke; reproduction of the
# composites themselves is lifecycle 27's / lifecycle 22's own standalone
# rebuild path — this cell produces no model.
PROVENANCE_COPIES = {
    "data/provenance/replay_ctl7_merge.json": (
        COMMITTED_MERGE_RECEIPTS[FROZEN_REPLAY_CONTROL][0],
        COMMITTED_MERGE_RECEIPTS[FROZEN_REPLAY_CONTROL][1],
    ),
    "data/provenance/count_walk_merge.json": (
        COMMITTED_MERGE_RECEIPTS[CANDIDATE][0],
        COMMITTED_MERGE_RECEIPTS[CANDIDATE][1],
    ),
    "data/provenance/zero_root_parent_merge.json": (
        "experiments/qwen35_4b_zero_root_lineage_rebuild/runs/lineage/merge.json",
        ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256,
    ),
    "data/provenance/prior_event_seed78163_summary.json": (
        PRIOR_EVENT["summary"],
        PRIOR_EVENT["summary_sha256"],
    ),
}
EVENT_DIRS = {
    seed: (
        EXP / "runs" / "benchmark"
        / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{seed}_{FROZEN_NAME}"
    )
    for seed in SEED_ORDER
}
READOUT = EXP / "runs" / "benchmark" / "confirmation_readout.json"
LEDGER = EXP / "runs" / "benchmark_events.jsonl"
CLOSED_RECORD_KEYS = frozenset(
    {
        "name", "phase", "tier", "think_budget", "seed",
        "summary", "summary_sha256", "receipts",
    }
)
# THE FROZEN REPLICATION RULE — integer-exact, two-directional, applied
# over the FOUR NEW events only. An event counts as a hit ONLY if it
# contains at least one FULL menders episode: each score contributes
# int(10*score + 1e-9) episodes (floor semantics — a partial-credit draw
# on the k/60 lattice such as 0.0167, 0.05, 0.0667, or 0.15 contributes
# ZERO episodes unless it crosses a full 0.1 step, and a partial-only
# event is NOT a hit; raw >0 draws are recorded descriptively, never
# counted). E_c must STRICTLY exceed E_j for EVERY control j. No fourth
# state. (Review amendment A1+A2, pre-event: hits and episodes now share
# one full-episode semantics, coinciding with the preregistered pricing
# model.)
REPLICATION_RULE = (
    "REPLICATED iff hits_c >= 2 AND E_c > E_j for EVERY control j, where "
    "an event counts as a hit only if it contains at least one FULL "
    "menders episode (score contributes int(10*s + 1e-9) episodes; "
    "partial-credit draws are recorded but never counted), hits_c = "
    "number of new events whose candidate FULL-EPISODE count is > 0, and "
    "E = sum over the four new events of int(10*score + 1e-9) menders "
    "episodes per arm; NOT_REPLICATED iff hits_c == 0; AMBIGUOUS "
    "otherwise (including any E_c tie with a control, which is not "
    "dominance). The 78163 event is prior evidence, never pooled. No "
    "fourth state."
)
FROZEN_CLAIMS = {
    "REPLICATED": (
        "the count_walk composite solves menders episodes at a rate no "
        "control matches; the first confirmed menders capability movement "
        "in the program."
    ),
    "NOT_REPLICATED": (
        "the 78163 reading closes as seed noise; the count-dont-walk dose "
        "did not durably move menders; the expression-cost law stands; the "
        "composite remains a documented artifact (at a true per-event hit "
        "rate of 0.3 this outcome retains probability ≈ 0.24 — the closure "
        "is a preregistered funding decision, not a nonexistence proof)."
    ),
    "AMBIGUOUS": (
        "no claim; further spending on this contrast requires a "
        "mechanism-differentiated NEW design, not more seeds of the same."
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_pushed_main(paths: list[Path]) -> None:
    if git_output(["git", "status", "--short"]):
        raise ValueError("benchmark event requires a clean worktree")
    if git_output(["git", "branch", "--show-current"]) != "main":
        raise ValueError("benchmark event requires branch main")
    if git_output(["git", "rev-parse", "HEAD"]) != git_output(["git", "rev-parse", "origin/main"]):
        raise ValueError("benchmark event requires HEAD == origin/main")
    for path in paths:
        relative = path.resolve().relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"benchmark prerequisite differs from HEAD: {relative}")


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
    return hashlib.sha256(rendered).hexdigest()


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise ValueError(f"{description} has not been authorized: {path}")


def require_provenance_copies() -> None:
    """The four data/provenance/ copies must byte-match pin AND original."""
    for copy_relative, (source_relative, expected) in sorted(
        PROVENANCE_COPIES.items()
    ):
        copy_path = ROOT / "experiments" / EXP.name / copy_relative
        source_path = ROOT / source_relative
        if not copy_path.is_file() or sha256_file(copy_path) != expected:
            raise ValueError(
                f"provenance copy is absent or changed: {copy_relative}"
            )
        if not source_path.is_file() or sha256_file(source_path) != expected:
            raise ValueError(
                f"committed provenance source is absent or changed: {source_relative}"
            )
        if copy_path.read_bytes() != source_path.read_bytes():
            raise ValueError(
                f"provenance copy is not byte-identical to its committed "
                f"source: {copy_relative} != {source_relative}"
            )


def require_zero_root_parent_provenance(model: Path) -> None:
    """The committed lifecycle-22 receipt must pin exactly the parent arm."""
    if (
        not ZERO_ROOT_PARENT_MERGE_RECEIPT.is_file()
        or sha256_file(ZERO_ROOT_PARENT_MERGE_RECEIPT)
        != ZERO_ROOT_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            "committed zero-root parent merge receipt is absent or changed: "
            f"{ZERO_ROOT_PARENT_MERGE_RECEIPT}"
        )
    payload = json.loads(
        ZERO_ROOT_PARENT_MERGE_RECEIPT.read_text(encoding="utf-8")
    )
    if (
        payload.get("experiment_id") != "qwen35_4b_zero_root_lineage_rebuild"
        or payload.get("stage") != "merge"
        or payload.get("name") != "zero_root_hygiene_explore"
        or payload.get("base_model", {}).get("id") != MODEL_ID
        or payload.get("base_model", {}).get("revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != model.resolve()
        or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[FROZEN_PARENT]
        or payload.get("weights_sha256") != FROZEN_WEIGHTS_SHA256[FROZEN_PARENT]
        or payload.get("weights_size_bytes") != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(
            "zero-root parent merge receipt does not describe the frozen parent arm"
        )


def authenticate_model_tree(label: str, model: Path) -> dict:
    """Bind the arm to its published bytes: recompute the full tree hash."""
    manifest = merged_tree_manifest(model)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != FROZEN_TREE_SHA256[label]:
        raise ValueError(f"benchmark arm tree changed for {label}: {observed_tree}")
    files = {row["name"]: row for row in manifest}
    weights = files["model.safetensors"]
    if (
        weights["sha256"] != FROZEN_WEIGHTS_SHA256[label]
        or weights["size"] != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(f"benchmark arm weights changed for {label}")
    if label in COMMITTED_MERGE_RECEIPTS:
        relative, expected = COMMITTED_MERGE_RECEIPTS[label]
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != EXPECTED_MERGE_RECEIPT_NAMES[label]
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[label]
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    elif label == FROZEN_PARENT:
        require_zero_root_parent_provenance(model)
        receipt = json.loads(
            ZERO_ROOT_PARENT_MERGE_RECEIPT.read_text(encoding="utf-8")
        )
        if files["merge_receipt.json"]["sha256"] != receipt.get(
            "inner_merge_receipt_sha256"
        ):
            raise ValueError(
                "zero-root parent composite's inner merge receipt does not "
                "match the committed lineage merge receipt"
            )
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def _valid_score(value: object) -> bool:
    """A gateway score must be a finite float in [0, 1]; NaN never passes."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def _valid_wall_seconds(value: object) -> bool:
    """Wall time must be a finite non-negative number (recorded, never gated)."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def load_event(path: Path, model: Path, seed: int) -> dict:
    """Authenticate one aggregate-gateway receipt against its frozen seed.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: an over-budget arm keeps its scores and is scoped by the
    budget_integrity reading instead of being rejected here. The gateway
    owns budget policy.
    """
    if seed not in SEED_ORDER:
        raise ValueError(f"receipt seed is not one of the frozen four: {seed}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("seed") != seed
        or payload.get("backend") != "qwen_vllm"
        or Path(payload.get("model", "")).resolve() != model.resolve()
        or not isinstance(payload.get("within_budget"), bool)
        or not _valid_wall_seconds(payload.get("wall_seconds"))
        or set(payload.get("per_family", {})) != PUBLIC_FAMILIES
        or not _valid_score(payload.get("aggregate"))
        or any(
            not _valid_score(value)
            for value in payload.get("per_family", {}).values()
        )
        or payload.get("model_merge_receipt_sha256")
        != sha256_file(model / "merge_receipt.json")
    ):
        raise ValueError(f"aggregate gateway event failed authentication: {path}")
    return payload


def opened_record(seed: int) -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": seed,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def is_closed_record(row: object, seed: int) -> bool:
    """A well-formed per-seed closed record; anything else fails closed.

    Besides the summary sha, a closed record must sha-pin ALL FOUR per-arm
    gateway receipts: the verdict inputs are provenance-anchored at close
    time and any later receipt swap fails against these pins.
    """
    return (
        isinstance(row, dict)
        and set(row) == set(CLOSED_RECORD_KEYS)
        and row["name"] == FROZEN_NAME
        and row["phase"] == "closed"
        and row["tier"] == FROZEN_TIER
        and row["think_budget"] == FROZEN_THINK_BUDGET
        and row["seed"] == seed
        and row["summary"] == str(EVENT_DIRS[seed] / "summary.json")
        and isinstance(row["summary_sha256"], str)
        and re.fullmatch(r"[0-9a-f]{64}", row["summary_sha256"]) is not None
        and isinstance(row["receipts"], dict)
        and set(row["receipts"]) == set(MODEL_ORDER)
        and all(
            isinstance(value, str)
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for value in row["receipts"].values()
        )
    )


def ledger_plan(rows: list[object], resume: bool) -> dict[int, dict]:
    """K-seed write-ahead budget: parse the ledger into a per-seed plan.

    The only valid history is a prefix of the canonical seed-major
    sequence opened(78164), closed(78164), ..., opened(78167),
    closed(78167). Anything else — legacy rows, malformed rows,
    out-of-order rows, duplicate opened records, an opened record for a
    later seed before an earlier seed closed — fails closed. A closed
    seed is NEVER re-run; when all four seeds are closed the confirmation
    budget is spent and every new event refuses, resume or not. A
    trailing opened record is a crashed seed: the whole event may only
    continue under an explicit ``--resume``, and the opened record must
    match the frozen per-seed record exactly.
    """
    plan = {seed: {"status": "fresh", "closed": None} for seed in SEED_ORDER}
    if not rows:
        return plan
    index = 0
    for seed in SEED_ORDER:
        if index == len(rows):
            break
        if rows[index] != opened_record(seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} does not match the frozen "
                f"opened record for seed {seed}"
            )
        index += 1
        if index == len(rows):
            plan[seed] = {"status": "crashed", "closed": None}
            break
        if not is_closed_record(rows[index], seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} is not the closed record "
                f"for seed {seed}"
            )
        plan[seed] = {"status": "closed", "closed": rows[index]}
        index += 1
    if index != len(rows):
        raise ValueError(
            "benchmark ledger has rows beyond the frozen four-seed event"
        )
    if all(entry["status"] == "closed" for entry in plan.values()):
        raise ValueError(
            "all four confirmation seeds are closed; the k-seed budget is spent"
        )
    if not resume:
        raise ValueError(
            "benchmark ledger has prior per-seed records; audit the preserved "
            "receipts and use --resume"
        )
    return plan


def authenticate_complete_ledger(rows: list[object]) -> dict[int, dict]:
    """The readout may only be computed through a COMPLETE k-seed ledger.

    Requires EXACTLY the canonical seed-major sequence with all four
    seeds closed and returns the closed records (whose pinned shas anchor
    every summary and receipt). Anything less refuses: the readout is
    never computed from unanchored receipt files.
    """
    if not rows:
        raise ValueError(
            "benchmark ledger is absent or empty; the confirmation readout "
            "requires the complete four-seed write-ahead ledger"
        )
    closed = {}
    index = 0
    for seed in SEED_ORDER:
        if index == len(rows) or rows[index] != opened_record(seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} is not the frozen opened "
                f"record for seed {seed}; the ledger is incomplete or corrupt"
            )
        index += 1
        if index == len(rows):
            raise ValueError(
                f"benchmark ledger ends with a crashed opened record for seed "
                f"{seed}; the confirmation readout requires all four seeds "
                "closed"
            )
        if not is_closed_record(rows[index], seed):
            raise ValueError(
                f"benchmark ledger row {index + 1} is not the closed record "
                f"for seed {seed}"
            )
        closed[seed] = rows[index]
        index += 1
    if index != len(rows):
        raise ValueError(
            "benchmark ledger has rows beyond the frozen four-seed event"
        )
    return closed


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def require_closed_seed_intact(seed: int, closed: dict, models: dict) -> None:
    """A closed seed is never re-run: its preserved artifacts must verify."""
    summary = EVENT_DIRS[seed] / "summary.json"
    if not summary.is_file() or sha256_file(summary) != closed["summary_sha256"]:
        raise ValueError(
            f"closed seed {seed} summary is absent or changed; a closed seed "
            "is never re-run"
        )
    for label in MODEL_ORDER:
        receipt = EVENT_DIRS[seed] / f"{label}.json"
        if (
            not receipt.is_file()
            or sha256_file(receipt) != closed["receipts"][label]
        ):
            raise ValueError(
                f"closed seed {seed} gateway receipt for {label} is absent or "
                "changed; the verdict inputs were sha-pinned at close time"
            )
        load_event(receipt, models[label], seed)


def stale_event_files(output_dir: Path) -> list[str]:
    """Event files that must never predate a seed's opened record.

    An UNOPENED seed requires a clean slate: receipt, failure, or summary
    files already present in its event directory are refused
    unconditionally — only a crashed (opened) seed may reuse preserved
    receipts, whose shas the closed record will pin.
    """
    if not output_dir.is_dir():
        return []
    names = [f"{label}.json" for label in MODEL_ORDER]
    names += [f"{label}.failure.json" for label in MODEL_ORDER]
    names.append("summary.json")
    return sorted(name for name in names if (output_dir / name).exists())


def reconcile_crashed_summary(seed: int, summary_path: Path, rendered: bytes) -> None:
    """Recover the crash window between summary write and closed append.

    The per-seed summary is a pure function of the authenticated receipts
    and the frozen pins, so an honest crash in that window regenerates it
    BYTE-IDENTICALLY; equality lets the closed record be appended and the
    event continue. Any divergence is tampering or drift and refuses
    loudly with both digests.
    """
    existing = summary_path.read_bytes()
    if existing != rendered:
        raise ValueError(
            f"crashed seed {seed} has a summary that does not match its "
            "deterministic regeneration (existing sha256 "
            f"{hashlib.sha256(existing).hexdigest()}, regenerated sha256 "
            f"{hashlib.sha256(rendered).hexdigest()}); audit the event "
            "directory before any recovery"
        )


def load_prior_reference() -> dict:
    """Load the sha-pinned prior-event summary; fail closed on any drift.

    Beyond the byte pin, the loader authenticates that the summary IS the
    recorded lifecycle-27 event: seed 78163 at medium/tb1024, the same
    four-arm order, the same four composites (tree AND weights hashes
    equal to this cell's frozen pins), the pinned implementation
    signature, and the recorded MECHANISM_ANSWER menders pattern
    (candidate 0.1, every control exactly 0.0), re-derived from the
    pinned scores.
    """
    summary = ROOT / PRIOR_EVENT["summary"]
    if not summary.is_file() or sha256_file(summary) != PRIOR_EVENT["summary_sha256"]:
        raise ValueError(
            f"pinned prior-event summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    implementation = payload.get("benchmark_implementation")
    trees = payload.get("model_tree_sha256s", {})
    weights = payload.get("model_weight_sha256s", {})
    if (
        payload.get("seed") != PRIOR_EVENT["seed"]
        or payload.get("tier") != PRIOR_EVENT["tier"]
        or payload.get("think_budget") != PRIOR_EVENT["think_budget"]
        or payload.get("model_order") != list(MODEL_ORDER)
        or implementation != PRIOR_IMPLEMENTATION
        or any(trees.get(label) != FROZEN_TREE_SHA256[label] for label in MODEL_ORDER)
        or any(
            weights.get(label) != FROZEN_WEIGHTS_SHA256[label]
            for label in MODEL_ORDER
        )
    ):
        raise ValueError("prior summary is not the frozen seed-78163 event")
    for label in MODEL_ORDER:
        row = scores.get(label, {})
        if (
            set(row.get("per_family", {})) != PUBLIC_FAMILIES
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"prior summary violates the score shape: {label}")
    menders = {
        label: scores[label]["per_family"][MENDERS_FAMILY] for label in MODEL_ORDER
    }
    if not (
        menders[CANDIDATE] > 0
        and all(menders[label] == 0 for label in CONTROL_ARMS)
        and payload.get("menders_reading", {}).get("frozen_interpretation")
        == "MECHANISM_ANSWER"
    ):
        raise ValueError(
            "prior summary does not carry the recorded MECHANISM_ANSWER "
            "menders pattern"
        )
    return {
        "scores": {
            label: {
                "aggregate": scores[label]["aggregate"],
                "per_family": dict(scores[label]["per_family"]),
            }
            for label in MODEL_ORDER
        },
        "benchmark_implementation": dict(implementation),
    }


def menders_episodes(score: object) -> int:
    """Frozen integer conversion: int(10*score + 1e-9) menders episodes.

    FLOOR semantics (review amendment A1, pre-event): a partial-credit
    draw on the menders k/60 lattice (0.0167, 0.05, 0.0667, 0.15, ...)
    contributes ZERO episodes unless it crosses a full 0.1 step — for
    every lattice point k/60 (k = 0..60) the conversion equals int(k/6),
    unit-tested over the whole lattice via the float k/60 representation.
    The 1e-9 guard absorbs float error at exact multiples of 0.1 without
    ever promoting a genuine partial. Every full-episode draw in program
    history is an exact multiple of 0.1; the two recorded partial-credit
    draws (0.0167) convert to zero episodes and are NOT hits (they are
    recorded descriptively as raw positives).
    """
    if not _valid_score(score):
        raise ValueError(f"menders score is not a finite float in [0, 1]: {score!r}")
    return int(10 * score + 1e-9)


def goal_gate_row(
    base_per_family: dict[str, float], treated_per_family: dict[str, float]
) -> dict:
    """The strict-win partition vs base; descriptive only, never gating."""
    families = sorted(PUBLIC_FAMILIES)
    wins = [f for f in families if treated_per_family[f] > base_per_family[f]]
    losses = [f for f in families if treated_per_family[f] < base_per_family[f]]
    ties = [f for f in families if treated_per_family[f] == base_per_family[f]]
    return {
        "strict_wins": len(wins),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "goal_gate_pass": len(wins) == len(families),
    }


def replication_reading(menders_by_seed: dict[int, dict[str, float]]) -> dict:
    """THE FROZEN REPLICATION RULE over the four new events only.

    Integer-exact and two-directional; the prior 78163 event is never
    pooled. An event counts as a hit ONLY if the arm's FULL-EPISODE
    count int(10*score + 1e-9) is > 0 — hits and episodes share one
    full-episode semantics (review amendment A1+A2), so the rule
    coincides with the preregistered pricing model. Partial-only events
    (raw score > 0 but zero full episodes) are recorded descriptively in
    ``raw_positive`` / ``raw_positive_events_per_arm`` and are neither
    hits nor episodes. E totals convert each event's score to
    int(10*score + 1e-9) episodes per arm; dominance requires E_c
    STRICTLY above every control's total (a tie is not dominance).
    Verdict partition, total, no fourth state: REPLICATED iff
    hits_c >= 2 AND dominance; NOT_REPLICATED iff hits_c == 0;
    AMBIGUOUS otherwise.
    """
    if set(menders_by_seed) != set(SEED_ORDER):
        raise ValueError(
            "replication reading requires exactly the four frozen new events"
        )
    per_event = {}
    for seed in SEED_ORDER:
        arms = menders_by_seed[seed]
        if set(arms) != set(MODEL_ORDER):
            raise ValueError(
                f"replication reading requires all four arms at seed {seed}"
            )
        episodes = {label: menders_episodes(arms[label]) for label in MODEL_ORDER}
        per_event[str(seed)] = {
            "scores": {label: arms[label] for label in MODEL_ORDER},
            "episodes": episodes,
            "candidate_hit": episodes[CANDIDATE] > 0,
            # Descriptive only: raw >0 draws (including partial-only
            # events) are recorded but never counted as hits or episodes.
            "raw_positive": {
                label: arms[label] > 0 for label in MODEL_ORDER
            },
        }
    hits = {
        label: sum(
            1
            for seed in SEED_ORDER
            if menders_episodes(menders_by_seed[seed][label]) > 0
        )
        for label in MODEL_ORDER
    }
    raw_positive_events = {
        label: sum(
            1 for seed in SEED_ORDER if menders_by_seed[seed][label] > 0
        )
        for label in MODEL_ORDER
    }
    totals = {
        label: sum(
            menders_episodes(menders_by_seed[seed][label]) for seed in SEED_ORDER
        )
        for label in MODEL_ORDER
    }
    hits_c = hits[CANDIDATE]
    dominance = {
        label: totals[CANDIDATE] > totals[label] for label in CONTROL_ARMS
    }
    dominant = all(dominance.values())
    if hits_c >= 2 and dominant:
        verdict = "REPLICATED"
    elif hits_c == 0:
        verdict = "NOT_REPLICATED"
    else:
        verdict = "AMBIGUOUS"
    return {
        "rule": REPLICATION_RULE,
        "family": MENDERS_FAMILY,
        "events_counted": list(SEED_ORDER),
        "prior_event_pooled": False,
        "per_event": per_event,
        "hits_per_arm": hits,
        "hits_c": hits_c,
        "raw_positive_events_per_arm": raw_positive_events,
        "raw_positive_descriptive_only": True,
        "episode_totals": totals,
        "candidate_dominates_control": dominance,
        "candidate_dominates_every_control": dominant,
        "verdict": verdict,
        "frozen_claim": FROZEN_CLAIMS[verdict],
        "frozen_claims": dict(FROZEN_CLAIMS),
    }


def per_seed_readings(scores_by_seed: dict[int, dict]) -> dict:
    """Descriptive per-event tables: aggregates, per-family, goal gates,
    and candidate-vs-each-control deltas. Recorded, never gating."""
    readings = {}
    for seed in SEED_ORDER:
        scores = scores_by_seed[seed]
        base = scores["base"]
        candidate = scores[CANDIDATE]
        comparisons = {}
        for label in TREATED_ARMS:
            event = scores[label]
            family_delta = {
                family: event["per_family"][family] - base["per_family"][family]
                for family in sorted(PUBLIC_FAMILIES)
            }
            comparisons[f"{label}_minus_base"] = {
                "aggregate_delta": event["aggregate"] - base["aggregate"],
                "per_family_delta": family_delta,
                "positive_families": sum(v > 0 for v in family_delta.values()),
                "nonnegative_families": sum(v >= 0 for v in family_delta.values()),
                "minimum_family_delta": min(family_delta.values()),
            }
        candidate_vs_controls = {}
        for label in CONTROL_ARMS:
            control = scores[label]
            candidate_vs_controls[f"{CANDIDATE}_minus_{label}"] = {
                "aggregate_delta": candidate["aggregate"] - control["aggregate"],
                "per_family_delta": {
                    family: candidate["per_family"][family]
                    - control["per_family"][family]
                    for family in sorted(PUBLIC_FAMILIES)
                },
                "menders_delta": candidate["per_family"][MENDERS_FAMILY]
                - control["per_family"][MENDERS_FAMILY],
            }
        readings[str(seed)] = {
            "aggregates": {
                label: scores[label]["aggregate"] for label in MODEL_ORDER
            },
            "per_family": {
                label: dict(scores[label]["per_family"]) for label in MODEL_ORDER
            },
            "goal_gates_vs_base": {
                label: goal_gate_row(base["per_family"], scores[label]["per_family"])
                for label in TREATED_ARMS
            },
            "comparisons": comparisons,
            "candidate_vs_controls": candidate_vs_controls,
            "descriptive_only": True,
        }
    return readings


def budget_integrity(budget_by_seed: dict[int, dict]) -> dict:
    """within_budget/wall_seconds per arm per seed; scope only, never gate."""
    per_seed = {}
    over_all = []
    for seed in SEED_ORDER:
        budget = budget_by_seed[seed]
        per_arm = {
            label: {
                "within_budget": budget[label]["within_budget"],
                "wall_seconds": budget[label]["wall_seconds"],
            }
            for label in MODEL_ORDER
        }
        over = [label for label in MODEL_ORDER if not per_arm[label]["within_budget"]]
        per_seed[str(seed)] = {
            "per_arm": per_arm,
            "all_within_budget": not over,
            "paired_comparison_valid": not over,
            "reason": (
                None
                if not over
                else (
                    f"arms exceeded the gateway budget at seed {seed}: {over}; "
                    "the paired comparison at this seed is not budget-matched "
                    "(scores recorded, not compared)"
                )
            ),
        }
        over_all.extend(f"seed {seed}: {label}" for label in over)
    return {
        "per_seed": per_seed,
        "all_within_budget": not over_all,
        "paired_comparison_valid": not over_all,
        "reason": (
            None
            if not over_all
            else (
                f"over-budget arms: {over_all}; the four-seed confirmation "
                "comparison is not budget-matched (scores recorded, not "
                "compared)"
            )
        ),
    }


def prior_event_report(prior_scores: dict[str, dict]) -> dict:
    """The prior 78163 event, reported alongside the verdict, never pooled."""
    menders = {
        label: prior_scores[label]["per_family"][MENDERS_FAMILY]
        for label in MODEL_ORDER
    }
    return {
        "seed": PRIOR_EVENT["seed"],
        "tier": PRIOR_EVENT["tier"],
        "think_budget": PRIOR_EVENT["think_budget"],
        "summary": PRIOR_EVENT["summary"],
        "summary_sha256": PRIOR_EVENT["summary_sha256"],
        "aggregates": {
            label: prior_scores[label]["aggregate"] for label in MODEL_ORDER
        },
        "menders_per_arm": menders,
        "menders_episodes_per_arm": {
            label: menders_episodes(value) for label, value in menders.items()
        },
        "mechanism_answer": True,
        "counted_in_verdict": False,
    }


def require_implementation_equality(
    implementation: dict, prior_implementation: dict
) -> None:
    """Fail-closed comparability anchor to the prior event."""
    if implementation != prior_implementation:
        raise ValueError(
            "benchmark implementation differs from the pinned prior event "
            f"(confirmation={implementation}, prior={prior_implementation}); "
            "the preregistered confirmation is not comparable"
        )


def current_implementation_signature() -> dict:
    """Compute the LIVE benchmark implementation signature pre-consumption.

    Computed THROUGH the sha-authenticated trusted gateway's own
    inventory functions — suite bytes are hashed, never parsed, and no
    benchmark content ever enters this process as data (the hidden-label
    firewall holds; this is exactly what the gateway itself does before
    every event). Used by the per-seed PRE-consumption check so a
    drifted suite refuses BEFORE any GPU run or opened ledger record
    instead of after four spent gateway runs.
    """
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")
    spec = importlib.util.spec_from_file_location(
        "count_walk_confirmation_trusted_gateway", GATEWAY
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    inventory = module.benchmark_source_inventory(module.MENAGERIE.parent)
    return {
        "runner_sha256": module.sha256_file(module.MENAGERIE),
        "source_inventory_sha256": inventory["sha256"],
        "source_file_count": inventory["file_count"],
    }


def build_readout(
    scores_by_seed: dict[int, dict],
    budget_by_seed: dict[int, dict],
    implementation: dict,
    prior: dict,
    receipts: dict[int, dict],
) -> dict:
    """Assemble the readout from pure inputs (unit-testable, no file IO)."""
    require_implementation_equality(
        implementation, prior["benchmark_implementation"]
    )
    menders_by_seed = {
        seed: {
            label: scores_by_seed[seed][label]["per_family"][MENDERS_FAMILY]
            for label in MODEL_ORDER
        }
        for seed in SEED_ORDER
    }
    replication = replication_reading(menders_by_seed)
    integrity = budget_integrity(budget_by_seed)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "count_walk_menders_confirmation_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seeds": list(SEED_ORDER),
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "CONFIRMATION_READ_COMPLETE",
        "verdict": replication["verdict"],
        "frozen_claim": replication["frozen_claim"],
        "paired_comparison_valid": integrity["paired_comparison_valid"],
        "provenance": {
            "ledger": "runs/benchmark_events.jsonl",
            "ledger_complete_sequence_required": True,
            "receipt_sha256s_pinned_in_closed_records": True,
            "summaries_verified_against_ledger_pins": True,
            "receipts_verified_against_ledger_pins": True,
            "receipt_blocks_verified_against_sealed_summaries": True,
        },
        "prior_event": prior_event_report(prior["scores"]),
        "benchmark_implementation": {
            "signature": dict(implementation),
            "prior": dict(prior["benchmark_implementation"]),
            "identical_across_all_sixteen_receipts_and_prior": True,
        },
        "receipts": {str(seed): receipts[seed] for seed in SEED_ORDER},
        "scores": {str(seed): scores_by_seed[seed] for seed in SEED_ORDER},
        "budget": {str(seed): budget_by_seed[seed] for seed in SEED_ORDER},
        "readings": {
            "replication": replication,
            "per_seed": per_seed_readings(scores_by_seed),
            "budget_integrity": integrity,
        },
    }


def require_summary_consistency(
    seed: int, summary: object, events: dict[str, dict]
) -> None:
    """Receipts must equal the sealed summary's recorded blocks, exactly."""
    if (
        not isinstance(summary, dict)
        or summary.get("schema_version") != 1
        or summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != seed
        or summary.get("model_order") != list(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("prior_summary_sha256") != PRIOR_EVENT["summary_sha256"]
    ):
        raise ValueError(f"sealed summary failed authentication for seed {seed}")
    for label in MODEL_ORDER:
        event = events[label]
        if summary.get("scores", {}).get(label) != {
            "aggregate": event["aggregate"],
            "per_family": event["per_family"],
        }:
            raise ValueError(
                f"receipt scores diverge from the sealed summary for seed "
                f"{seed} arm {label}"
            )
        if summary.get("budget", {}).get(label) != {
            "within_budget": event["within_budget"],
            "wall_seconds": event["wall_seconds"],
        }:
            raise ValueError(
                f"receipt budget diverges from the sealed summary for seed "
                f"{seed} arm {label}"
            )
        if summary.get("benchmark_implementation") != {
            "runner_sha256": event["benchmark_runner_sha256"],
            "source_inventory_sha256": event["benchmark_source_inventory_sha256"],
            "source_file_count": event["benchmark_source_file_count"],
        }:
            raise ValueError(
                f"receipt implementation diverges from the sealed summary for "
                f"seed {seed} arm {label}"
            )


def render_readout() -> bytes:
    """Authenticate every input through the ledger, then render the readout.

    Order of anchoring: the complete ledger first (four closed records, in
    order, nothing crashed or trailing), then per seed the sealed summary
    against its pinned sha, then each receipt against the sha its closed
    record pinned at close time, then structural receipt authentication,
    then receipt-versus-summary block equality. Only inputs that survive
    all five layers reach the verdict.
    """
    prior = load_prior_reference()
    require_provenance_copies()
    rows = (
        [
            json.loads(line)
            for line in LEDGER.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if LEDGER.is_file()
        else []
    )
    closed_records = authenticate_complete_ledger(rows)
    scores_by_seed = {}
    budget_by_seed = {}
    receipts = {}
    signatures = set()
    for seed in SEED_ORDER:
        record = closed_records[seed]
        summary_path = EVENT_DIRS[seed] / "summary.json"
        if (
            not summary_path.is_file()
            or sha256_file(summary_path) != record["summary_sha256"]
        ):
            raise ValueError(
                f"sealed summary is absent or does not match its closed "
                f"ledger pin for seed {seed}"
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        events = {}
        receipts[seed] = {}
        for label in MODEL_ORDER:
            path = EVENT_DIRS[seed] / f"{label}.json"
            if not path.is_file():
                raise ValueError(
                    f"gateway receipt is absent for seed {seed} arm {label}: {path}"
                )
            digest = sha256_file(path)
            if digest != record["receipts"][label]:
                raise ValueError(
                    f"gateway receipt does not match its closed ledger pin "
                    f"for seed {seed} arm {label}"
                )
            events[label] = load_event(path, FROZEN_MODEL_PATHS[label], seed)
            receipts[seed][label] = {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": digest,
            }
            signatures.add(
                (
                    events[label]["benchmark_runner_sha256"],
                    events[label]["benchmark_source_inventory_sha256"],
                    events[label]["benchmark_source_file_count"],
                )
            )
        require_summary_consistency(seed, summary, events)
        scores_by_seed[seed] = {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        }
        budget_by_seed[seed] = {
            label: {
                "within_budget": event["within_budget"],
                "wall_seconds": event["wall_seconds"],
            }
            for label, event in events.items()
        }
    if len(signatures) != 1:
        raise ValueError(
            "benchmark implementation changed between the sixteen receipts"
        )
    runner_sha, inventory_sha, file_count = next(iter(signatures))
    implementation = {
        "runner_sha256": runner_sha,
        "source_inventory_sha256": inventory_sha,
        "source_file_count": file_count,
    }
    readout = build_readout(
        scores_by_seed, budget_by_seed, implementation, prior, receipts
    )
    return (
        json.dumps(readout, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def run_gateway_arm(
    args: argparse.Namespace, seed: int, label: str, model: Path, output: Path
) -> None:
    """One gateway call; on failure preserve a sanitized receipt and stop."""
    failure = output.parent / f"{label}.failure.json"
    command = [
        str(PYTHON), str(GATEWAY), "--tier", args.tier, "--seed", str(seed),
        "--model", str(model), "--out", str(output),
        "--think-budget", str(args.think_budget),
    ]
    completed = subprocess.run(
        command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE, text=True, check=False,
    )
    if completed.returncode == 0:
        return
    diagnostic = "unclassified_gateway_failure"
    safe_stderr = completed.stderr.strip()
    aggregate_match = re.fullmatch(
        r"aggregate benchmark gateway failed; category=([a-z_]+); "
        r"private output suppressed",
        safe_stderr,
    )
    runner_match = re.fullmatch(
        r"benchmark runner failed with exit code \d+; private aggregate "
        r"state=([a-z_]+); raw stdout/stderr suppressed",
        safe_stderr,
    )
    if aggregate_match:
        diagnostic = aggregate_match.group(1)
    elif runner_match:
        diagnostic = f"runner_failure_{runner_match.group(1)}"
    failure.write_text(
        json.dumps({
            "schema_version": 1,
            "name": args.name,
            "tier": args.tier,
            "think_budget": args.think_budget,
            "seed": seed,
            "arm": label,
            "model": str(model),
            "model_merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
            "gateway_exit_code": completed.returncode,
            "safe_diagnostic": diagnostic,
            "score_emitted": False,
            "raw_streams_exposed": False,
            "benchmark_output_exposed": False,
        }, indent=1, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    raise SystemExit(
        f"aggregate gateway failed for seed {seed} arm {label} with exit "
        f"{completed.returncode} ({diagnostic}); "
        "private output remained suppressed"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument(
        "--seed", action="append", type=int, required=True,
        help="repeat four times in the frozen order",
    )
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or tuple(args.seed) != SEED_ORDER
        or args.think_budget != FROZEN_THINK_BUDGET
    ):
        parser.error("benchmark event differs from the preregistered confirmation")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe event name")
    models = {}
    for specification in args.model:
        label, separator, raw_path = specification.partition("=")
        path = Path(raw_path).resolve()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
            parser.error(f"invalid model specification: {specification}")
        if label in models or not (path / "merge_receipt.json").is_file():
            parser.error(f"duplicate label or missing merge receipt: {specification}")
        if label not in FROZEN_MODEL_PATHS or path != FROZEN_MODEL_PATHS[label].resolve():
            parser.error(f"model path differs from the frozen arm: {specification}")
        models[label] = path
    if set(models) != set(MODEL_ORDER) or len(models) != len(MODEL_ORDER):
        parser.error(
            "models must be exactly base, zero_root_parent, replay_ctl7, "
            "and count_walk"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    try:
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        prior = load_prior_reference()
        require_provenance_copies()
        require_clean_pushed_main(
            [
                PREREGISTRATION,
                BENCH_REVIEW,
                ROOT / PRIOR_EVENT["summary"],
                ZERO_ROOT_PARENT_MERGE_RECEIPT,
            ]
            + [
                ROOT / relative
                for relative, _ in COMMITTED_MERGE_RECEIPTS.values()
            ]
            + [
                ROOT / "experiments" / EXP.name / copy_relative
                for copy_relative in sorted(PROVENANCE_COPIES)
            ]
        )
        plan = ledger_plan(
            [
                json.loads(line)
                for line in LEDGER.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if LEDGER.is_file()
            else [],
            args.resume,
        )
        model_hashes = {
            label: authenticate_model_tree(label, models[label])
            for label in MODEL_ORDER
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    summaries = {}
    for seed in SEED_ORDER:
        output_dir = EVENT_DIRS[seed]
        summary_path = output_dir / "summary.json"
        if plan[seed]["status"] == "closed":
            # Completed seeds are never re-run; their preserved artifacts
            # must still authenticate before the event may continue.
            try:
                require_closed_seed_intact(seed, plan[seed]["closed"], models)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                parser.error(str(error))
            summaries[seed] = plan[seed]["closed"]["summary_sha256"]
            continue
        if plan[seed]["status"] == "fresh":
            stale = stale_event_files(output_dir)
            if stale:
                parser.error(
                    f"seed {seed} was never opened but its event directory "
                    f"already contains {stale}; unopened seeds require a "
                    "clean slate (only a crashed, opened seed may reuse "
                    "preserved receipts)"
                )
            if output_dir.exists() and not args.resume:
                parser.error(
                    f"partial event directory exists for seed {seed}; use "
                    "--resume after auditing it"
                )
        # PRE-CONSUMPTION implementation anchor (review amendment, minor 2):
        # whenever this seed still has at least one gateway call to make,
        # the LIVE benchmark implementation signature must equal the prior
        # event's pinned block BEFORE the first arm runs — a drifted suite
        # refuses here, before any GPU run or opened ledger record, instead
        # of burning four gateway runs and wedging at the post-check (which
        # is kept below).
        if any(
            not (output_dir / f"{label}.json").exists() for label in MODEL_ORDER
        ):
            try:
                require_implementation_equality(
                    current_implementation_signature(), PRIOR_IMPLEMENTATION
                )
            except ValueError as error:
                parser.error(str(error))
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write-ahead record: the seed is spent the moment its first gateway
        # call can start, so a mid-seed crash leaves a permanent trace. A
        # crashed seed already has its opened record (matched exactly by the
        # ledger plan) and must not open twice.
        if plan[seed]["status"] == "fresh":
            append_ledger(opened_record(seed))
        events = {}
        for label in MODEL_ORDER:
            model = models[label]
            output = output_dir / f"{label}.json"
            failure = output_dir / f"{label}.failure.json"
            if failure.exists():
                parser.error(
                    f"preserved failure exists for seed {seed} arm {label}; "
                    "audit before retrying"
                )
            if not output.exists():
                run_gateway_arm(args, seed, label, model, output)
            events[label] = load_event(output, model, seed)

        signatures = {
            (
                event["benchmark_runner_sha256"],
                event["benchmark_source_inventory_sha256"],
                event["benchmark_source_file_count"],
            )
            for event in events.values()
        }
        if len(signatures) != 1:
            raise SystemExit(
                f"benchmark implementation changed between paired arms at seed {seed}"
            )
        runner_sha, inventory_sha, file_count = next(iter(signatures))
        implementation = {
            "runner_sha256": runner_sha,
            "source_inventory_sha256": inventory_sha,
            "source_file_count": file_count,
        }
        if implementation != PRIOR_IMPLEMENTATION:
            raise SystemExit(
                f"benchmark implementation at seed {seed} differs from the "
                f"pinned prior event ({implementation} != "
                f"{PRIOR_IMPLEMENTATION}); the confirmation is not comparable"
            )

        menders = {
            label: events[label]["per_family"][MENDERS_FAMILY]
            for label in MODEL_ORDER
        }
        payload = {
            "schema_version": 1,
            "name": args.name,
            "tier": args.tier,
            "think_budget": args.think_budget,
            "seed": seed,
            "gateway_sha256": GATEWAY_SHA256,
            "prior_summary_sha256": PRIOR_EVENT["summary_sha256"],
            "model_order": list(MODEL_ORDER),
            "models": {label: str(path) for label, path in models.items()},
            "model_tree_sha256s": {
                label: hashes["tree_sha256"] for label, hashes in model_hashes.items()
            },
            "model_weight_sha256s": {
                label: hashes["weights_sha256"] for label, hashes in model_hashes.items()
            },
            "benchmark_implementation": implementation,
            "scores": {
                label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
                for label, event in events.items()
            },
            "budget": {
                label: {
                    "within_budget": event["within_budget"],
                    "wall_seconds": event["wall_seconds"],
                }
                for label, event in events.items()
            },
            "menders": {
                "per_arm": menders,
                "episodes_per_arm": {
                    label: menders_episodes(value)
                    for label, value in menders.items()
                },
                # A hit requires at least one FULL episode (floor
                # semantics); raw positives are recorded, never counted.
                "candidate_hit": menders_episodes(menders[CANDIDATE]) > 0,
                "candidate_raw_positive": menders[CANDIDATE] > 0,
                "verdict_deferred_to_readout": True,
            },
            "promoted": None,
            "benchmark_data_read": False,
        }
        rendered = (
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if summary_path.exists():
            # Crash window between the summary write and the closed-record
            # append (only a crashed seed can reach here; unopened seeds
            # refused above): the deterministic regeneration must match
            # byte-for-byte, then the close proceeds.
            try:
                reconcile_crashed_summary(seed, summary_path, rendered)
            except ValueError as error:
                parser.error(str(error))
        else:
            summary_path.write_bytes(rendered)
        summary_sha = sha256_file(summary_path)
        # The closed record pins every verdict input: the sealed summary
        # AND all four per-arm gateway receipts, by sha256.
        append_ledger({
            "name": args.name, "phase": "closed", "tier": args.tier,
            "think_budget": args.think_budget, "seed": seed,
            "summary": str(summary_path), "summary_sha256": summary_sha,
            "receipts": {
                label: sha256_file(output_dir / f"{label}.json")
                for label in MODEL_ORDER
            },
        })
        summaries[seed] = summary_sha

    # All four seeds are closed: write (or byte-verify) the terminal readout.
    rendered = render_readout()
    if READOUT.exists():
        if READOUT.read_bytes() != rendered:
            raise SystemExit(
                "published confirmation readout does not match its "
                "deterministic regeneration; audit before any recovery"
            )
    else:
        READOUT.parent.mkdir(parents=True, exist_ok=True)
        READOUT.write_bytes(rendered)
    verdict = json.loads(rendered.decode("utf-8"))["verdict"]
    print(json.dumps({
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seeds": list(SEED_ORDER),
        "summaries_sha256": {str(seed): summaries[seed] for seed in SEED_ORDER},
        "readout": str(READOUT),
        "verdict": verdict,
        "frozen_claim": FROZEN_CLAIMS[verdict],
    }, indent=1, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
