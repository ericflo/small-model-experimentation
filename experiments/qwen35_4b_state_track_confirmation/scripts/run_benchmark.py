#!/usr/bin/env python3
"""Run the six sealed per-seed CONFIRMATION events through the trusted gateway.

EVAL-ONLY MEASUREMENT, not a promotion: no training, no merging, no
corpus, no candidate gate exists anywhere in this cell (lifecycle 31).
Lifecycle 30's single-seed seed-78169 INSTALLED_TRANSFER — the
``state_track`` composite scored aggregate 0.3260 versus the
``count_walk`` parent's 0.3004 (a paired lift of +0.0256) — is tested for
replication on SIX independent sealed fresh seeds (78170, 78171, 78172,
78173, 78174, 78175) at tier medium, think budget 1024, TWO arms per seed
in the frozen order ``count_walk`` then ``state_track``, seed-major (a
seed's two arms complete and close before the next seed opens). Twelve
gateway runs total. There is no ``base`` arm: the two-arm event is the
parent-versus-candidate paired comparison. Discipline copied from the
count-walk menders confirmation runner (lifecycle 28, the k-seed
write-ahead ledger) and the state_track install runner (lifecycle 30, the
two-arm aggregate reading, the in-cell-authoritative provenance, the
1e-12 aggregate tie guard):

- the PASS_BENCHMARK_EVENT review verdict, the committed preregistration,
  and the committed provenance copies are enforced HERE, at the
  seed-consuming boundary: a direct invocation cannot consume any seed
  with an unreviewed or drifted contract;
- clean pushed main plus the committed-at-HEAD preregistration, review,
  the two in-cell provenance copies, and the sha-pinned prior-event
  summary are hard prerequisites;
- every arm authenticates fail-closed by recomputing its full on-disk
  tree sha256 (which covers the 9GB weights) against pins baked as frozen
  constants at design time — both composites pre-exist this cell, so
  every pin is a design-time constant and no TODO-PIN slot exists here.
  Each arm additionally authenticates against this cell's IN-CELL
  sha-pinned provenance copy of its committed merge receipt (payload
  equality AND the composite's inner merge_receipt.json hash); the
  committed sibling original in its own cell is a VERIFICATION AID only —
  byte-identical when present (divergence fails loudly as tamper
  evidence), skipped with a recorded note when absent (owner's standalone
  directive: the in-cell pin is authoritative);
- K-SEED WRITE-AHEAD ledger: each seed gets its own ``opened`` record
  appended before its first gateway call and its own ``closed`` record
  after its per-seed summary. The closed record sha-pins the summary AND
  BOTH per-arm gateway receipts, so every verdict input is
  provenance-anchored at close time. The only valid ledger history is a
  prefix of the canonical seed-major sequence; a closed record refuses
  its seed forever (completed seeds are never re-run); a crash mid-seed
  leaves a permanent opened record that forces recovery through
  ``--resume`` with the preserved receipts. RECOVERY SEMANTICS: an
  UNOPENED seed requires a clean slate — pre-existing
  receipt/failure/summary files in a never-opened seed's event directory
  refuse unconditionally. If a crash landed in the window between the
  summary write and the closed-record append, the summary is regenerated
  deterministically from the authenticated receipts (a pure function of
  the twelve receipts and the committed prerequisites — no wall-clock
  anywhere) and must compare BYTE-IDENTICAL to the file on disk — a match
  appends the closed record and continues, a mismatch refuses loudly with
  both digests. Once all six seeds close, the confirmation budget is
  spent and every new invocation refuses, resume or not;
- implementation-signature integrity per seed: BEFORE a seed's first
  gateway call the LIVE benchmark implementation signature — computed
  through the trusted gateway's own hash-only inventory functions (bytes
  hashed, never parsed) — must equal the prior event's pinned block
  (pre-consumption; a drifted suite refuses before any GPU run or opened
  record); afterwards the two receipts must share one (runner sha256,
  source inventory sha256, file count) signature AND match the same
  pinned block, before the seed's summary is written — all twelve
  receipts are thereby anchored to the seed-78169 event, fail closed;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a
  NaN can never silently drop a family from any reading;
- ``within_budget`` and ``wall_seconds`` are recorded exactly as
  returned and never gated; the budget_integrity reading scopes the
  paired comparison instead.

After the sixth seed closes, the FROZEN PAIRED REPLICATION RULE
(over the six NEW events only — 78169 is prior evidence, never pooled) is
computed into the terminal readout: ``CONFIRMED`` / ``NOT_CONFIRMED`` /
``AMBIGUOUS``, no fourth state. For each event the PAIRED aggregate delta
d_i = state_track_aggregate - count_walk_aggregate is formed on the SAME
seed (common seed-variance cancels — the whole point of pairing); let
``wins`` be the number of events with d_i strictly above the 1e-12 tie
guard and ``mean_d`` the arithmetic mean of the six d_i. CONFIRMED iff
mean_d strictly positive AND wins >= 4 (ceil(2*6/3)); NOT_CONFIRMED iff
mean_d not strictly positive (mean <= 0 dominates, even at wins >= 4);
AMBIGUOUS otherwise (mean_d > 0 but wins < 4). The benchmark suite
directory is never read; only scripts/run_benchmark_aggregate.py runs.
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
# Six fresh sealed seeds, verified grep-fresh in seed contexts across the
# repo at design time (audit recorded in reports/preregistration.md).
SEED_ORDER = (78170, 78171, 78172, 78173, 78174, 78175)
FROZEN_PARENT = "count_walk"
CANDIDATE = "state_track"
# Frozen per-event arm order: the parent (baseline) then the candidate.
MODEL_ORDER = (FROZEN_PARENT, CANDIDATE)
# The frozen aggregate tie guard (lifecycle 30's carried fix): the gateway
# reports float aggregates, and distinct per-family multisets with exactly
# equal RATIONAL aggregates can render one ulp apart (demonstrated:
# 0.45999999999999996 vs 0.46000000000000008). Strictly-above therefore
# means ``(a - b) > AGG_TIE_EPSILON``; ``|delta| <= AGG_TIE_EPSILON`` is a
# true rational tie and a tie is NOT strictly above (never a candidate win,
# never a strictly-positive mean). Real aggregate differences on the
# k/10-per-family lattices are >= ~1.7e-3 per event (and the mean of six
# lands on a 1/3600 lattice, smallest nonzero |mean_d| ~= 2.8e-4), both
# many orders of magnitude above the 1e-12 guard.
AGG_TIE_EPSILON = 1e-12
EVENTS = len(SEED_ORDER)
# The frozen win threshold: a strict two-thirds majority of the six events,
# ceil(2*6/3) = 4.
WINS_THRESHOLD = 4
# The frozen per-family retention tolerance, carried DESCRIPTIVELY from
# lifecycle 30 (never part of this cell's frozen rule): one episode (0.1)
# of slack below the parent per family, exact at the lattice boundary.
PER_FAMILY_SLACK = 0.1
SLACK_EPSILON = 1e-9
FROZEN_MODEL_PATHS = {
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_count_dont_walk_enumeration"
        / "merged" / "count_walk"
    ),
    CANDIDATE: (
        ROOT / "large_artifacts" / "qwen35_4b_state_track_install"
        / "merged" / "state_track"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Both composites pre-exist this cell, so every pin is a
# design-time frozen constant — no TODO-PIN slot exists in this file.
FROZEN_TREE_SHA256 = {
    FROZEN_PARENT: (
        "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
    ),
    CANDIDATE: (
        "45fd2925e417c82e4848b2ca89907934df9e60503b6529af0bddbd8aa359be7e"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    FROZEN_PARENT: (
        "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
    ),
    CANDIDATE: (
        "b4bafbb7d3ff8dedd2fa216bc9c62997d960d43a6cac22a88976245bcc35d1c1"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
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
# Per-arm provenance: the IN-CELL sha-pinned copy under data/provenance/ is
# the hard fail-closed gate; the committed sibling original in its own cell
# is a VERIFICATION AID (byte-identical when present, skipped with a
# recorded note when absent). Each block pins the merge-receipt FILE sha,
# the source-cell identity, and the composite's inner merge_receipt.json
# sha (the per-file normalized pin, lifecycle 30's carried fix).
ARM_PROVENANCE = {
    FROZEN_PARENT: {
        "copy": EXP / "data" / "provenance" / "count_walk_merge.json",
        "copy_relative": "data/provenance/count_walk_merge.json",
        "sibling_original": (
            ROOT / "experiments" / "qwen35_4b_count_dont_walk_enumeration"
            / "runs" / "merges" / "count_walk.json"
        ),
        "sibling_relative": (
            "experiments/qwen35_4b_count_dont_walk_enumeration"
            "/runs/merges/count_walk.json"
        ),
        "merge_receipt_sha256": (
            "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36"
        ),
        "experiment_id": "qwen35_4b_count_dont_walk_enumeration",
        "name": "count_walk",
        "inner_receipt_sha256": (
            "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7"
        ),
    },
    CANDIDATE: {
        "copy": EXP / "data" / "provenance" / "state_track_merge.json",
        "copy_relative": "data/provenance/state_track_merge.json",
        "sibling_original": (
            ROOT / "experiments" / "qwen35_4b_state_track_install"
            / "runs" / "merges" / "state_track.json"
        ),
        "sibling_relative": (
            "experiments/qwen35_4b_state_track_install"
            "/runs/merges/state_track.json"
        ),
        "merge_receipt_sha256": (
            "089f280eab1b6f4afd53e636a49f1b4fd92efd5fa1ee42a1a07e35e49a98c94e"
        ),
        "experiment_id": "qwen35_4b_state_track_install",
        "name": "state_track",
        "inner_receipt_sha256": (
            "d23862f70cdbb71b2b232bee0501e65f45a432cacd3e37189418194e27493a0d"
        ),
    },
}
# The committed prior event at seed 78169 whose INSTALLED_TRANSFER this cell
# tests for replication. Sha-pinned (in-cell verification-aid copy) so a
# drifted source fails BEFORE any seed is consumed; reported alongside the
# verdict, NEVER pooled into it.
PRIOR_EVENT = {
    "seed": 78169,
    "tier": FROZEN_TIER,
    "think_budget": FROZEN_THINK_BUDGET,
    "summary_copy": EXP / "data" / "provenance" / "prior_event_seed78169_summary.json",
    "summary_copy_relative": "data/provenance/prior_event_seed78169_summary.json",
    "summary_sibling": (
        ROOT / "experiments" / "qwen35_4b_state_track_install"
        / "runs" / "benchmark" / "medium_tb1024_seed78169_install" / "summary.json"
    ),
    "summary_sibling_relative": (
        "experiments/qwen35_4b_state_track_install"
        "/runs/benchmark/medium_tb1024_seed78169_install/summary.json"
    ),
    "summary_sha256": (
        "187cc3acfe81016899cb08a8bebf5f6045a6cabba9868edd5379c51708ec1192"
    ),
    "counted_in_verdict": False,
}
# Every receipt of the twelve must carry exactly this signature (the
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
# THE FROZEN PAIRED REPLICATION RULE — two-directional, applied over the
# SIX NEW events only (78169 is prior evidence, never pooled). For each
# event the paired aggregate delta d_i = state_track_aggregate -
# count_walk_aggregate cancels the common per-seed variance. wins = number
# of events with d_i > AGG_TIE_EPSILON (the 1e-12 tie guard); mean_d =
# arithmetic mean of the six d_i, strictly positive iff mean_d >
# AGG_TIE_EPSILON. CONFIRMED iff mean_d strictly positive AND wins >= 4;
# NOT_CONFIRMED iff mean_d not strictly positive (mean <= 0 dominates,
# including the wins >= 4 edge); AMBIGUOUS otherwise. No fourth state.
REPLICATION_RULE = (
    "CONFIRMED iff mean_d strictly positive (mean_d > AGG_TIE_EPSILON) AND "
    "wins >= 4 (ceil(2*6/3)), where d_i = state_track_aggregate - "
    "count_walk_aggregate on the same seed i over the six new events, wins "
    "= number of events with d_i > AGG_TIE_EPSILON (1e-12 tie guard), and "
    "mean_d = arithmetic mean of the six d_i; NOT_CONFIRMED iff mean_d is "
    "not strictly positive (mean_d <= AGG_TIE_EPSILON, which dominates even "
    "when wins >= 4); AMBIGUOUS otherwise (mean_d strictly positive but "
    "wins < 4). The 78169 event is prior evidence, never pooled. No fourth "
    "state."
)
FROZEN_CLAIMS = {
    "CONFIRMED": (
        "the state_track aggregate lift replicates across sealed seeds; the "
        "divergent-skill install is a durable gain and state_track is the "
        "program reference composite; the install-universal-features "
        "doctrine yields real transferable aggregate."
    ),
    "NOT_CONFIRMED": (
        "the 78169 lift does not replicate; it was within the parent's seed "
        "variance; count_walk remains the reference; the single-seed "
        "INSTALLED_TRANSFER is retired as seed noise."
    ),
    "AMBIGUOUS": (
        "directional but not decisive; a mechanism-differentiated or "
        "larger-N design is required, not a re-roll of these seeds."
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


def require_arm_provenance(label: str, model: Path) -> str:
    """The IN-CELL sha-pinned provenance copy must pin exactly this arm.

    The in-cell copy is the hard fail-closed gate; the committed sibling
    original is a verification aid: byte-identical when present (divergence
    fails loudly as tamper evidence), skipped with a recorded note when
    absent. Returns the sibling-original status note.
    """
    block = ARM_PROVENANCE[label]
    copy = block["copy"]
    if not copy.is_file() or sha256_file(copy) != block["merge_receipt_sha256"]:
        raise ValueError(
            f"in-cell {label} provenance copy is absent or changed: {copy}"
        )
    sibling = block["sibling_original"]
    if sibling.is_file():
        if sibling.read_bytes() != copy.read_bytes():
            raise ValueError(
                f"committed {label} sibling merge receipt diverged from the "
                f"in-cell provenance pin: {sibling}"
            )
        sibling_note = "present, byte-identical to the in-cell pin"
    else:
        sibling_note = "absent, in-cell pin authoritative"
        print(
            f"[run_benchmark] {label} sibling original absent; the in-cell "
            "sha-pinned provenance copy is authoritative"
        )
    payload = json.loads(copy.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != block["experiment_id"]
        or payload.get("name") != block["name"]
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != model.resolve()
        or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[label]
        or {
            row.get("name"): row.get("sha256")
            for row in payload.get("weight_files", [])
        }
        != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        or payload.get("merge_receipt_sha256") != block["inner_receipt_sha256"]
    ):
        raise ValueError(
            f"{label} merge receipt does not describe the frozen arm"
        )
    return sibling_note


def require_provenance_copies() -> dict:
    """Both arm provenance copies plus the prior-event summary copy verify.

    The in-cell copies are the hard fail-closed gate (sha-pinned); the
    committed sibling originals are verification aids (byte-identical when
    present, skip-noted when absent). Returns the sibling-presence notes.
    """
    notes = {}
    for label in MODEL_ORDER:
        notes[label] = require_arm_provenance(label, FROZEN_MODEL_PATHS[label])
    copy = PRIOR_EVENT["summary_copy"]
    if not copy.is_file() or sha256_file(copy) != PRIOR_EVENT["summary_sha256"]:
        raise ValueError(
            f"in-cell prior-event summary copy is absent or changed: {copy}"
        )
    sibling = PRIOR_EVENT["summary_sibling"]
    if sibling.is_file():
        if sibling.read_bytes() != copy.read_bytes():
            raise ValueError(
                "committed prior-event summary sibling diverged from the "
                f"in-cell provenance pin: {sibling}"
            )
        notes["prior_event"] = "present, byte-identical to the in-cell pin"
    else:
        notes["prior_event"] = "absent, in-cell pin authoritative"
    return notes


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
    require_arm_provenance(label, model)
    if files["merge_receipt.json"]["sha256"] != ARM_PROVENANCE[label][
        "inner_receipt_sha256"
    ]:
        raise ValueError(
            f"{label} composite's inner merge receipt does not match the "
            "committed merge receipt"
        )
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
    budget_integrity reading instead of being rejected here.
    """
    if seed not in SEED_ORDER:
        raise ValueError(f"receipt seed is not one of the frozen six: {seed}")
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

    Besides the summary sha, a closed record must sha-pin BOTH per-arm
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
    sequence opened(78170), closed(78170), ..., opened(78175),
    closed(78175). Anything else fails closed. A closed seed is NEVER
    re-run; when all six seeds are closed the confirmation budget is spent
    and every new event refuses. A trailing opened record is a crashed
    seed: continuation only under an explicit ``--resume``.
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
            "benchmark ledger has rows beyond the frozen six-seed event"
        )
    if all(entry["status"] == "closed" for entry in plan.values()):
        raise ValueError(
            "all six confirmation seeds are closed; the k-seed budget is spent"
        )
    if not resume:
        raise ValueError(
            "benchmark ledger has prior per-seed records; audit the preserved "
            "receipts and use --resume"
        )
    return plan


def authenticate_complete_ledger(rows: list[object]) -> dict[int, dict]:
    """The readout may only be computed through a COMPLETE k-seed ledger.

    Requires EXACTLY the canonical seed-major sequence with all six seeds
    closed and returns the closed records (whose pinned shas anchor every
    summary and receipt). Anything less refuses.
    """
    if not rows:
        raise ValueError(
            "benchmark ledger is absent or empty; the confirmation readout "
            "requires the complete six-seed write-ahead ledger"
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
                f"{seed}; the confirmation readout requires all six seeds "
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
            "benchmark ledger has rows beyond the frozen six-seed event"
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
    """Event files that must never predate a seed's opened record."""
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
    recorded lifecycle-30 event: seed 78169 at medium/tb1024, the frozen
    parent and candidate composites (tree AND weights hashes equal to this
    cell's frozen pins), the pinned implementation signature, and the
    recorded INSTALLED_TRANSFER consequence with the candidate aggregate
    strictly above the parent (the paired lift being confirmed).
    """
    summary = PRIOR_EVENT["summary_copy"]
    if not summary.is_file() or sha256_file(summary) != PRIOR_EVENT["summary_sha256"]:
        raise ValueError(
            f"pinned prior-event summary is absent or changed: {summary}"
        )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    scores = payload.get("scores", {})
    implementation = payload.get("benchmark_implementation")
    trees = payload.get("model_tree_sha256s", {})
    weights = payload.get("model_weight_sha256s", {})
    consequence = payload.get("consequence", {})
    if (
        payload.get("seed") != PRIOR_EVENT["seed"]
        or payload.get("tier") != PRIOR_EVENT["tier"]
        or payload.get("think_budget") != PRIOR_EVENT["think_budget"]
        or implementation != PRIOR_IMPLEMENTATION
        or any(trees.get(label) != FROZEN_TREE_SHA256[label] for label in MODEL_ORDER)
        or any(
            weights.get(label) != FROZEN_WEIGHTS_SHA256[label]
            for label in MODEL_ORDER
        )
    ):
        raise ValueError("prior summary is not the frozen seed-78169 event")
    for label in MODEL_ORDER:
        row = scores.get(label, {})
        if (
            set(row.get("per_family", {})) != PUBLIC_FAMILIES
            or not _valid_score(row.get("aggregate"))
            or any(not _valid_score(value) for value in row["per_family"].values())
        ):
            raise ValueError(f"prior summary violates the score shape: {label}")
    candidate_agg = scores[CANDIDATE]["aggregate"]
    parent_agg = scores[FROZEN_PARENT]["aggregate"]
    if not (
        consequence.get("verdict") == "INSTALLED_TRANSFER"
        and consequence.get("aggregate_strictly_beats_parent") is True
        and aggregate_strictly_above(candidate_agg, parent_agg)
    ):
        raise ValueError(
            "prior summary does not carry the recorded INSTALLED_TRANSFER "
            "paired lift"
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


def aggregate_strictly_above(a: float, b: float) -> bool:
    """The frozen strictly-above reading on gateway-reported float aggregates.

    ``(a - b) > AGG_TIE_EPSILON``: a TRUE rational tie whose two float
    renderings differ by one ulp is a tie, and a tie is never strictly
    above. Real aggregate differences (>= ~1.7e-3 per event) clear the
    1e-12 guard by nine orders of magnitude.
    """
    return (a - b) > AGG_TIE_EPSILON


def family_within_slack(candidate_value: float, parent_value: float) -> bool:
    """Descriptive per-family retention tolerance (never gates this cell).

    ``candidate >= parent - 0.1 - 1e-9``: a family exactly 0.1 below the
    parent still passes; 0.10000001 below fails. Carried from lifecycle 30
    as a descriptive reading only.
    """
    return candidate_value >= parent_value - PER_FAMILY_SLACK - SLACK_EPSILON


def paired_reading(aggregates_by_seed: dict[int, dict[str, float]]) -> dict:
    """THE FROZEN PAIRED REPLICATION RULE over the six new events only.

    For each event the paired delta d_i = state_track_aggregate -
    count_walk_aggregate cancels common per-seed variance. wins counts
    events with d_i > AGG_TIE_EPSILON (the 1e-12 tie guard); mean_d is the
    arithmetic mean of the six d_i, strictly positive iff mean_d >
    AGG_TIE_EPSILON. Verdict partition, total, no fourth state: CONFIRMED
    iff mean_d strictly positive AND wins >= 4; NOT_CONFIRMED iff mean_d
    not strictly positive (dominates the wins clause); AMBIGUOUS otherwise.
    """
    if set(aggregates_by_seed) != set(SEED_ORDER):
        raise ValueError(
            "paired reading requires exactly the six frozen new events"
        )
    per_event = {}
    deltas = []
    for seed in SEED_ORDER:
        arms = aggregates_by_seed[seed]
        if set(arms) != set(MODEL_ORDER):
            raise ValueError(
                f"paired reading requires both arms at seed {seed}"
            )
        parent = arms[FROZEN_PARENT]
        candidate = arms[CANDIDATE]
        if not _valid_score(parent) or not _valid_score(candidate):
            raise ValueError(
                f"paired reading requires finite aggregates in [0,1] at seed {seed}"
            )
        delta = candidate - parent
        win = aggregate_strictly_above(candidate, parent)
        deltas.append(delta)
        per_event[str(seed)] = {
            "count_walk_aggregate": parent,
            "state_track_aggregate": candidate,
            "paired_delta": delta,
            "candidate_wins": win,
        }
    wins = sum(
        1
        for seed in SEED_ORDER
        if aggregate_strictly_above(
            aggregates_by_seed[seed][CANDIDATE],
            aggregates_by_seed[seed][FROZEN_PARENT],
        )
    )
    mean_d = math.fsum(deltas) / EVENTS
    if not math.isfinite(mean_d):
        raise ValueError("paired reading produced a non-finite mean delta")
    mean_positive = mean_d > AGG_TIE_EPSILON
    if mean_positive and wins >= WINS_THRESHOLD:
        verdict = "CONFIRMED"
    elif not mean_positive:
        verdict = "NOT_CONFIRMED"
    else:
        verdict = "AMBIGUOUS"
    return {
        "rule": REPLICATION_RULE,
        "events_counted": list(SEED_ORDER),
        "prior_event_pooled": False,
        "aggregate_tie_epsilon": AGG_TIE_EPSILON,
        "wins_threshold": WINS_THRESHOLD,
        "per_event": per_event,
        "paired_deltas": deltas,
        "wins": wins,
        "mean_delta": mean_d,
        "mean_delta_strictly_positive": mean_positive,
        "verdict": verdict,
        "frozen_claim": FROZEN_CLAIMS[verdict],
        "frozen_claims": dict(FROZEN_CLAIMS),
    }


def candidate_vs_parent_gate(
    parent_per_family: dict[str, float], candidate_per_family: dict[str, float]
) -> dict:
    """Parent-vs-candidate per-family strict-win partition; descriptive only.

    This cell has NO base arm; the 'goal gate' analog is the candidate
    against the parent, recorded either way and never part of the frozen
    rule.
    """
    families = sorted(PUBLIC_FAMILIES)
    wins = [f for f in families if candidate_per_family[f] > parent_per_family[f]]
    losses = [f for f in families if candidate_per_family[f] < parent_per_family[f]]
    ties = [f for f in families if candidate_per_family[f] == parent_per_family[f]]
    return {
        "strict_wins": len(wins),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "candidate_beats_parent_every_family": len(wins) == len(families),
    }


def per_seed_readings(scores_by_seed: dict[int, dict]) -> dict:
    """Descriptive per-event tables: both aggregates, the paired delta, the
    per-family delta table, the candidate-vs-parent family gate, and the
    per-family within-slack retention flags. Recorded, never gating."""
    readings = {}
    for seed in SEED_ORDER:
        scores = scores_by_seed[seed]
        parent = scores[FROZEN_PARENT]
        candidate = scores[CANDIDATE]
        family_delta = {
            family: candidate["per_family"][family] - parent["per_family"][family]
            for family in sorted(PUBLIC_FAMILIES)
        }
        readings[str(seed)] = {
            "aggregates": {
                label: scores[label]["aggregate"] for label in MODEL_ORDER
            },
            "paired_delta": candidate["aggregate"] - parent["aggregate"],
            "per_family": {
                label: dict(scores[label]["per_family"]) for label in MODEL_ORDER
            },
            "candidate_minus_parent_per_family": family_delta,
            "positive_families": sum(v > 0 for v in family_delta.values()),
            "nonnegative_families": sum(v >= 0 for v in family_delta.values()),
            "minimum_family_delta": min(family_delta.values()),
            "candidate_vs_parent_family_gate": candidate_vs_parent_gate(
                parent["per_family"], candidate["per_family"]
            ),
            "families_below_slack": sorted(
                family
                for family in PUBLIC_FAMILIES
                if not family_within_slack(
                    candidate["per_family"][family], parent["per_family"][family]
                )
            ),
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
                f"over-budget arms: {over_all}; the six-seed confirmation "
                "comparison is not budget-matched (scores recorded, not "
                "compared)"
            )
        ),
    }


def prior_event_report(prior_scores: dict[str, dict]) -> dict:
    """The prior 78169 event, reported alongside the verdict, never pooled."""
    aggregates = {
        label: prior_scores[label]["aggregate"] for label in MODEL_ORDER
    }
    return {
        "seed": PRIOR_EVENT["seed"],
        "tier": PRIOR_EVENT["tier"],
        "think_budget": PRIOR_EVENT["think_budget"],
        "summary": PRIOR_EVENT["summary_sibling_relative"],
        "summary_copy": PRIOR_EVENT["summary_copy_relative"],
        "summary_sha256": PRIOR_EVENT["summary_sha256"],
        "aggregates": aggregates,
        "paired_delta": aggregates[CANDIDATE] - aggregates[FROZEN_PARENT],
        "installed_transfer": True,
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

    Computed THROUGH the sha-authenticated trusted gateway's own inventory
    functions — suite bytes are hashed, never parsed, and no benchmark
    content ever enters this process as data. Used by the per-seed
    PRE-consumption check so a drifted suite refuses BEFORE any GPU run or
    opened ledger record.
    """
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise ValueError("trusted gateway is absent or changed")
    spec = importlib.util.spec_from_file_location(
        "state_track_confirmation_trusted_gateway", GATEWAY
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
    aggregates_by_seed = {
        seed: {label: scores_by_seed[seed][label]["aggregate"] for label in MODEL_ORDER}
        for seed in SEED_ORDER
    }
    paired = paired_reading(aggregates_by_seed)
    integrity = budget_integrity(budget_by_seed)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "state_track_confirmation_readout",
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seeds": list(SEED_ORDER),
        "benchmark_data_read": False,
        "promoted": None,
        "outcome": "CONFIRMATION_READ_COMPLETE",
        "verdict": paired["verdict"],
        "frozen_claim": paired["frozen_claim"],
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
            "identical_across_all_twelve_receipts_and_prior": True,
        },
        "receipts": {str(seed): receipts[seed] for seed in SEED_ORDER},
        "scores": {str(seed): scores_by_seed[seed] for seed in SEED_ORDER},
        "budget": {str(seed): budget_by_seed[seed] for seed in SEED_ORDER},
        "readings": {
            "paired_replication": paired,
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
    """Authenticate every input through the ledger, then render the readout."""
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
            "benchmark implementation changed between the twelve receipts"
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
        help="repeat six times in the frozen order",
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
        parser.error("models must be exactly count_walk and state_track")
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
                PRIOR_EVENT["summary_copy"],
                *[ARM_PROVENANCE[label]["copy"] for label in MODEL_ORDER],
            ]
            # The sibling originals are verification aids: HEAD-checked when
            # present, skipped when absent (the in-cell pin is the gate).
            + [
                ARM_PROVENANCE[label]["sibling_original"]
                for label in MODEL_ORDER
                if ARM_PROVENANCE[label]["sibling_original"].is_file()
            ]
            + (
                [PRIOR_EVENT["summary_sibling"]]
                if PRIOR_EVENT["summary_sibling"].is_file()
                else []
            )
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
        # PRE-CONSUMPTION implementation anchor: whenever this seed still has
        # at least one gateway call to make, the LIVE benchmark
        # implementation signature must equal the prior event's pinned block
        # BEFORE the first arm runs.
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

        parent = events[FROZEN_PARENT]
        candidate = events[CANDIDATE]
        paired_delta = candidate["aggregate"] - parent["aggregate"]
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
            "paired": {
                "count_walk_aggregate": parent["aggregate"],
                "state_track_aggregate": candidate["aggregate"],
                "paired_delta": paired_delta,
                "candidate_wins": aggregate_strictly_above(
                    candidate["aggregate"], parent["aggregate"]
                ),
                "verdict_deferred_to_readout": True,
            },
            "promoted": None,
            "benchmark_data_read": False,
        }
        rendered = (
            json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        if summary_path.exists():
            try:
                reconcile_crashed_summary(seed, summary_path, rendered)
            except ValueError as error:
                parser.error(str(error))
        else:
            summary_path.write_bytes(rendered)
        summary_sha = sha256_file(summary_path)
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

    # All six seeds are closed: write (or byte-verify) the terminal readout.
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
