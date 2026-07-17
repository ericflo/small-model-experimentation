#!/usr/bin/env python3
"""Run the one conditional three-model medium event through the trusted gateway.

CONDITIONAL on local promotion: this event exists only if ``replay_compound``
passed the frozen three-screen retention gate (seeds 88060/88061/88062).
Three composites (base, the COUNT_WALK parent — lifecycle 27's stage-7
candidate, fully documented by this cell's in-cell lineage package — and the
promoted ``replay_compound`` candidate) are evaluated sequentially on the
same sealed fresh seed 78168 at tier medium, think budget 1024, in that
frozen order. Discipline copied from the count-don't-walk cell's hardened
runner, plus the NORMALIZED-HASH code pin: this file is frozen by
check_design.py's normalized hash — exactly the THREE trained-arm pin VALUE
slots (candidate tree, candidate weights, candidate committed merge receipt)
are canonicalized to a fixed placeholder before hashing, so EVERY OTHER BYTE
of this runner (every guard call site included) is byte-frozen pre- and
post-fill, and any drift fails ``check_design --check`` at the
seed-consuming boundary:

- the PASS_BENCHMARK_EVENT review verdict and the frozen design receipts are
  enforced HERE, at the seed-consuming boundary (check_design --check and
  gen_local_gate --check re-run as subprocesses), not only in the harness;
- clean pushed main plus the committed-at-HEAD local design receipt,
  benchmark design review, promotion receipt, local receipt, provenance
  copy, and the candidate's committed merge receipt are hard prerequisites;
- every arm authenticates by recomputing its full on-disk tree sha256 (which
  covers the 9GB weights) against its pin — the candidate's three pins are
  orchestrator-filled TODO-PIN slots that abort while None; the count_walk
  parent additionally authenticates against the IN-CELL sha-pinned
  provenance copy of lifecycle 27's merge receipt (payload equality; the
  committed sibling original is a verification aid — byte-identical when
  present, skipped with a recorded note when absent);
- one-seed WRITE-AHEAD ledger: an ``opened`` record is appended before the
  first gateway call and a ``closed`` record after the summary. Any closed
  record refuses forever; a crashed event forces recovery through
  ``--resume``. BYTE-EQUAL CRASH RECONCILIATION: the summary payload is a
  pure function of the three gateway receipts and the committed
  prerequisites (no wall-clock anywhere), so a crash between the summary
  write and the ledger close reconciles under ``--resume`` by recomputing
  the payload and requiring byte equality with the preserved summary before
  the closed record is appended — a divergent summary refuses forever;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a NaN
  can never silently drop a family from the comparison.

FROZEN TWO-DIRECTIONAL CONSEQUENCE (no third state; integer-exact where the
family scores live on the k/10 and k/60 lattices):

  COMPOUNDED iff the candidate aggregate is STRICTLY above the parent
  aggregate AND no family sits strictly below the parent by more than 0.1
  (every family independently gets at most one episode (0.1) of slack
  below the parent; the rule caps depth per family, not the number of
  families using slack; frozen a priori as the compounding tolerance the
  chain's own stages exhibited; the comparison is frozen as
  ``candidate_family >= parent_family - 0.1 - 1e-9`` so a family exactly
  0.1 below still passes and 0.10000001 below fails) AND the candidate
  aggregate is STRICTLY above the base aggregate. STRICTLY above on the
  aggregates means ``(candidate - other) > 1e-12`` on the
  gateway-reported floats: distinct per-family multisets with exactly
  equal rational aggregates can render one ulp apart (e.g.
  0.45999999999999996 vs 0.46000000000000008), so ``|delta| <= 1e-12``
  is a TRUE RATIONAL TIE and a tie is never strictly above (the BOUNDED
  path); real aggregate differences on these lattices are >= ~1.7e-3.
  Frozen claim: "replay
  compounding holds at stage 8; the composite becomes the program reference
  artifact and feeds the raised-floor confirmation."

  BOUNDED otherwise. Frozen claim: "the replay-compounding law hits
  diminishing returns at stage 8 on this parent; the count_walk composite
  remains the reference; further aggregate pushes need a different move
  class."

The GOAL GATE vs base (all ten families strictly above base, the 10/10
strict-wins reading) is recorded DESCRIPTIVELY for both treated arms either
way and never feeds the consequence. The benchmark suite directory is never
read; only scripts/run_benchmark_aggregate.py runs.
"""

from __future__ import annotations

import argparse
import hashlib
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
LOCAL_DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
SCREEN_SEEDS = (88060, 88061, 88062)
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{SCREEN_SEEDS[0]}.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / f"seed{SCREEN_SEEDS[0]}_promotion.json"
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
FROZEN_NAME = "compound"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78168
FROZEN_PARENT = "count_walk"
FROZEN_CANDIDATES = ("replay_compound",)
MODEL_ORDER = ("base", FROZEN_PARENT, "replay_compound")
TREATED_ARMS = (FROZEN_PARENT, "replay_compound")
# The frozen per-family compounding tolerance: every family independently
# gets at most one episode (0.1) of slack below the parent; the rule caps
# depth per family, not the number of families using slack. The epsilon only
# absorbs float error at the exact lattice boundary (k/10 and k/60 lattices).
PER_FAMILY_SLACK = 0.1
SLACK_EPSILON = 1e-9
# The frozen aggregate tie guard: the gateway reports float aggregates, and
# distinct per-family multisets with exactly equal RATIONAL aggregates can
# render one ulp apart (demonstrated: 0.45999999999999996 vs
# 0.46000000000000008), which would flip BOUNDED to COMPOUNDED under a bare
# ``>``. Strictly-above therefore means ``(candidate - other) >
# AGG_TIE_EPSILON``; ``|delta| <= AGG_TIE_EPSILON`` is a tie and a tie is
# NOT strictly above (the BOUNDED path). Real aggregate differences on the
# k/10-per-family lattices are >= ~1.7e-3, nine orders of magnitude above
# the guard, so 1e-12 cleanly separates ulp noise from any true win.
AGG_TIE_EPSILON = 1e-12
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_count_dont_walk_enumeration"
        / "merged" / "count_walk"
    ),
    "replay_compound": (
        ROOT / "large_artifacts" / EXP.name / "merged" / "replay_compound"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Inherited arms are pinned at design time from committed
# receipts; the candidate's pins are TODO-PINs the orchestrator fills after
# its merge publishes. A None pin aborts the event.
FROZEN_TREE_SHA256: dict[str, str | None] = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    FROZEN_PARENT: (
        "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
    ),
    # TODO-PIN(post-merge): runs/merges/replay_compound.json ->
    # output_tree_sha256. Fill by replacing None with the quoted 64-hex ON
    # THIS LINE; check_design's normalized-hash pin canonicalizes only this
    # value.
    "replay_compound": None,
}
FROZEN_WEIGHTS_SHA256: dict[str, str | None] = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    FROZEN_PARENT: (
        "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
    ),
    # TODO-PIN(post-merge): runs/merges/replay_compound.json weight sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE.
    "replay_compound": None,
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# TODO-PIN(post-merge): sha256 of the COMMITTED runs/merges/replay_compound.json.
# Fill by replacing None with the quoted 64-hex ON THIS LINE.
REPLAY_COMPOUND_MERGE_RECEIPT_SHA256 = None
COMMITTED_MERGE_RECEIPTS: dict[str, tuple[str, object]] = {
    "replay_compound": (
        f"experiments/{EXP.name}/runs/merges/replay_compound.json",
        REPLAY_COMPOUND_MERGE_RECEIPT_SHA256,
    ),
}
EXPECTED_MERGE_RECEIPT_NAMES = {
    "replay_compound": "replay_compound",
}
# The parent arm's provenance anchor (fixed at design time, NOT a fill
# slot): this cell's IN-CELL copy at data/provenance/count_walk_merge.json
# is the sha-pinned fail-closed gate. Lifecycle 27's committed sibling
# original is a VERIFICATION AID only — byte-identical when present
# (divergence fails loudly as tamper evidence), skipped with a recorded
# note when absent; it is never the reproduction path (owner's standalone
# directive).
COUNT_WALK_PARENT_MERGE_RECEIPT = (
    ROOT / "experiments" / "qwen35_4b_count_dont_walk_enumeration"
    / "runs" / "merges" / "count_walk.json"
)
COUNT_WALK_PARENT_MERGE_RECEIPT_SHA256 = (
    "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36"
)
COUNT_WALK_PARENT_PROVENANCE_COPY = (
    EXP / "data" / "provenance" / "count_walk_merge.json"
)
# The parent composite's INNER merge_receipt.json (inside the directory).
COUNT_WALK_PARENT_INNER_RECEIPT_SHA256 = (
    "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7"
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
LEDGER = EXP / "runs" / "benchmark_events.jsonl"
COMPOUNDED_CLAIM = (
    "replay compounding holds at stage 8; the composite becomes the program "
    "reference artifact and feeds the raised-floor confirmation."
)
BOUNDED_CLAIM = (
    "the replay-compounding law hits diminishing returns at stage 8 on this "
    "parent; the count_walk composite remains the reference; further "
    "aggregate pushes need a different move class."
)


def require_pin(value, name: str):
    if value is None:
        raise ValueError(
            f"frozen constant {name} is unpinned (TODO-PIN); the orchestrator "
            "must fill it from the committed merge receipt before this event"
        )
    return value


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


def ledger_rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def require_unconsumed_ledger(ledger: Path, opened_record: dict, resume: bool) -> None:
    """Write-ahead one-seed budget: closed refuses forever, opened needs --resume.

    Any record that is not a well-formed ``opened`` record counts as closed
    (fail closed on legacy or malformed rows). A lone matching opened record
    is a crashed event: it may only continue under an explicit ``--resume``,
    never restart silently.
    """
    rows = ledger_rows(ledger)
    if not rows:
        return
    if any(row.get("phase") != "opened" for row in rows):
        raise ValueError(
            "benchmark ledger already has a closed entry; the one-event budget is spent"
        )
    if len(rows) != 1 or rows[0] != opened_record:
        raise ValueError(
            "benchmark ledger opened record does not match the frozen event"
        )
    if not resume:
        raise ValueError(
            "benchmark ledger has an opened (crashed) event; audit the preserved "
            "receipts and use --resume"
        )


def _valid_score(value: object) -> bool:
    """A gateway score must be a finite float in [0, 1]; NaN never passes."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


def require_todo_pins_filled() -> None:
    """Every candidate pin must be a filled sha256; None refuses fail-closed."""
    sha_re = re.compile(r"[0-9a-f]{64}")
    for label, value in (
        ("replay_compound tree", FROZEN_TREE_SHA256["replay_compound"]),
        ("replay_compound weights", FROZEN_WEIGHTS_SHA256["replay_compound"]),
        ("replay_compound merge receipt", REPLAY_COMPOUND_MERGE_RECEIPT_SHA256),
    ):
        if not isinstance(value, str) or sha_re.fullmatch(value) is None:
            raise ValueError(
                f"{label} pin is unfilled (TODO-PIN): merge the trained arm, "
                "commit its merge receipt, fill the three pins from it, and "
                "re-seek review before any seed can be consumed"
            )


def require_count_walk_parent_provenance(model: Path) -> str:
    """The IN-CELL sha-pinned provenance copy must pin exactly the parent arm.

    The in-cell copy is the hard fail-closed gate; the committed lifecycle-27
    sibling original is a verification aid: byte-identical when present
    (divergence fails loudly as tamper evidence), skipped with a recorded
    note when absent. Returns the sibling-original status note.
    """
    if (
        not COUNT_WALK_PARENT_PROVENANCE_COPY.is_file()
        or sha256_file(COUNT_WALK_PARENT_PROVENANCE_COPY)
        != COUNT_WALK_PARENT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            "in-cell count_walk parent provenance copy is absent or changed: "
            f"{COUNT_WALK_PARENT_PROVENANCE_COPY}"
        )
    if COUNT_WALK_PARENT_MERGE_RECEIPT.is_file():
        if (
            COUNT_WALK_PARENT_MERGE_RECEIPT.read_bytes()
            != COUNT_WALK_PARENT_PROVENANCE_COPY.read_bytes()
        ):
            raise ValueError(
                "committed count_walk sibling merge receipt diverged from the "
                f"in-cell provenance pin: {COUNT_WALK_PARENT_MERGE_RECEIPT}"
            )
        sibling_note = "present, byte-identical to the in-cell pin"
    else:
        sibling_note = "absent, in-cell pin authoritative"
        print(
            "[run_benchmark] count_walk sibling original absent; the in-cell "
            "sha-pinned provenance copy is authoritative"
        )
    payload = json.loads(
        COUNT_WALK_PARENT_PROVENANCE_COPY.read_text(encoding="utf-8")
    )
    if (
        payload.get("experiment_id") != "qwen35_4b_count_dont_walk_enumeration"
        or payload.get("name") != "count_walk"
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != model.resolve()
        or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[FROZEN_PARENT]
        or {
            row.get("name"): row.get("sha256")
            for row in payload.get("weight_files", [])
        }
        != {"model.safetensors": FROZEN_WEIGHTS_SHA256[FROZEN_PARENT]}
        or payload.get("merge_receipt_sha256")
        != COUNT_WALK_PARENT_INNER_RECEIPT_SHA256
    ):
        raise ValueError(
            "count_walk parent merge receipt does not describe the frozen parent arm"
        )
    return sibling_note


def authenticate_model_tree(label: str, model: Path) -> dict:
    """Bind the arm to its published bytes: recompute the full tree hash."""
    expected_tree = require_pin(
        FROZEN_TREE_SHA256[label], f"FROZEN_TREE_SHA256[{label!r}]"
    )
    expected_weights = require_pin(
        FROZEN_WEIGHTS_SHA256[label], f"FROZEN_WEIGHTS_SHA256[{label!r}]"
    )
    manifest = merged_tree_manifest(model)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != expected_tree:
        raise ValueError(f"benchmark arm tree changed for {label}: {observed_tree}")
    files = {row["name"]: row for row in manifest}
    weights = files["model.safetensors"]
    if (
        weights["sha256"] != expected_weights
        or weights["size"] != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(f"benchmark arm weights changed for {label}")
    if label in COMMITTED_MERGE_RECEIPTS:
        relative, expected = COMMITTED_MERGE_RECEIPTS[label]
        expected = require_pin(expected, f"COMMITTED_MERGE_RECEIPTS[{label!r}]")
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != EXPECTED_MERGE_RECEIPT_NAMES[label]
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != expected_tree
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": expected_weights}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    elif label == FROZEN_PARENT:
        require_count_walk_parent_provenance(model)
        if (
            files["merge_receipt.json"]["sha256"]
            != COUNT_WALK_PARENT_INNER_RECEIPT_SHA256
        ):
            raise ValueError(
                "count_walk parent composite's inner merge receipt does not "
                "match the committed lifecycle-27 merge receipt"
            )
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def authenticate_local_promotion(candidate: str) -> dict:
    if not PROMOTION_RECEIPT.is_file():
        raise ValueError("local promotion receipt is absent; benchmark stays sealed")
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("screen_seeds") != list(SCREEN_SEEDS)
        or payload.get("aggregate_seed") != FROZEN_SEED
        or payload.get("aggregate_seed_open") is not True
        or payload.get("benchmark_data_read") is not False
        or payload.get("promoted") != candidate
        or not LOCAL_RECEIPT.is_file()
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
        or payload.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN_RECEIPT)
    ):
        raise ValueError(
            "local promotion receipt does not authorize this candidate benchmark"
        )
    return payload


def load_event(path: Path, model: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != FROZEN_TIER
        or payload.get("think_budget") != FROZEN_THINK_BUDGET
        or payload.get("seed") != FROZEN_SEED
        or payload.get("backend") != "qwen_vllm"
        or Path(payload.get("model", "")).resolve() != model.resolve()
        or payload.get("within_budget") is not True
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


def family_within_slack(candidate_value: float, parent_value: float) -> bool:
    """The frozen per-family compounding tolerance, exact at the boundary.

    ``candidate >= parent - 0.1 - 1e-9``: a family exactly 0.1 below the
    parent (one full episode on the k/10 lattice; six steps on the k/60
    lattice) still passes; 0.10000001 below fails. The epsilon only absorbs
    float representation error at exact lattice multiples of 0.1.
    """
    return candidate_value >= parent_value - PER_FAMILY_SLACK - SLACK_EPSILON


def aggregate_strictly_above(candidate_value: float, other_value: float) -> bool:
    """The frozen strictly-above reading on gateway-reported float aggregates.

    ``(candidate - other) > AGG_TIE_EPSILON``: a TRUE rational tie whose two
    float renderings differ by one ulp is a tie, and a tie is never strictly
    above (BOUNDED). Real aggregate differences (>= ~1.7e-3) clear the
    1e-12 guard by nine orders of magnitude.
    """
    return (candidate_value - other_value) > AGG_TIE_EPSILON


def consequence_reading(events: dict[str, dict], candidate: str) -> dict:
    """The frozen two-directional consequence: COMPOUNDED or BOUNDED only."""
    candidate_event = events[candidate]
    parent_event = events[FROZEN_PARENT]
    base_event = events["base"]
    beats_parent = aggregate_strictly_above(
        candidate_event["aggregate"], parent_event["aggregate"]
    )
    beats_base = aggregate_strictly_above(
        candidate_event["aggregate"], base_event["aggregate"]
    )
    family_table = {
        family: {
            "candidate": candidate_event["per_family"][family],
            "parent": parent_event["per_family"][family],
            "delta": candidate_event["per_family"][family]
            - parent_event["per_family"][family],
            "within_slack": family_within_slack(
                candidate_event["per_family"][family],
                parent_event["per_family"][family],
            ),
        }
        for family in sorted(PUBLIC_FAMILIES)
    }
    families_within_slack = all(
        row["within_slack"] for row in family_table.values()
    )
    compounded = beats_parent and families_within_slack and beats_base
    return {
        "candidate": candidate,
        "parent": FROZEN_PARENT,
        "aggregate_strictly_beats_parent": beats_parent,
        "aggregate_strictly_beats_base": beats_base,
        "no_family_below_parent_by_more_than_slack": families_within_slack,
        "per_family_slack": PER_FAMILY_SLACK,
        "slack_epsilon": SLACK_EPSILON,
        "aggregate_tie_epsilon": AGG_TIE_EPSILON,
        "family_table": family_table,
        "families_below_slack": sorted(
            family
            for family, row in family_table.items()
            if not row["within_slack"]
        ),
        "verdict": "COMPOUNDED" if compounded else "BOUNDED",
        "frozen_claim": COMPOUNDED_CLAIM if compounded else BOUNDED_CLAIM,
        "no_third_state": True,
    }


def goal_gate_reading(events: dict[str, dict]) -> dict:
    """Strict wins / ties / losses vs base per treated arm; pass = ten wins.

    Recorded DESCRIPTIVELY either way; never part of the frozen consequence."""
    base = events["base"]["per_family"]
    table = {}
    for arm in TREATED_ARMS:
        values = events[arm]["per_family"]
        wins = sorted(f for f in PUBLIC_FAMILIES if values[f] > base[f])
        losses = sorted(f for f in PUBLIC_FAMILIES if values[f] < base[f])
        ties = sorted(f for f in PUBLIC_FAMILIES if values[f] == base[f])
        table[arm] = {
            "strict_wins": len(wins),
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "goal_gate_pass": len(wins) == len(PUBLIC_FAMILIES),
        }
    return {
        "per_arm": table,
        "included_in_consequence": False,
        "recorded_either_way": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument(
        "--candidate", choices=FROZEN_CANDIDATES, required=True,
        help="the single locally promoted candidate label",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or args.seed != FROZEN_SEED
        or args.think_budget != FROZEN_THINK_BUDGET
    ):
        parser.error("benchmark event differs from the preregistered event")
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
            "models must be exactly base, count_walk, and replay_compound"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    opened_record = {
        "name": args.name,
        "phase": "opened",
        "seed": args.seed,
        "think_budget": args.think_budget,
        "tier": args.tier,
    }
    try:
        require_todo_pins_filled()
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        promotion = authenticate_local_promotion(args.candidate)
        require_clean_pushed_main(
            [
                LOCAL_DESIGN_RECEIPT,
                BENCH_REVIEW,
                PROMOTION_RECEIPT,
                LOCAL_RECEIPT,
                COUNT_WALK_PARENT_PROVENANCE_COPY,
            ]
            # The sibling original is a verification aid: HEAD-checked when
            # present, skipped when absent (the in-cell pin is the gate).
            + (
                [COUNT_WALK_PARENT_MERGE_RECEIPT]
                if COUNT_WALK_PARENT_MERGE_RECEIPT.is_file()
                else []
            )
            + [
                ROOT / relative
                for relative, _ in COMMITTED_MERGE_RECEIPTS.values()
            ]
        )
        # Re-verify the frozen design at the seed-consuming boundary: the
        # local design receipt pins the gate/harness/runner code set and
        # check_design pins this runner's normalized hash, so a committed
        # drift of either fails here before any gateway call.
        for check_script in ("check_design.py", "gen_local_gate.py"):
            subprocess.run(
                [str(PYTHON), "-B", str(SCRIPTS / check_script), "--check"],
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.DEVNULL,
                check=True,
            )
        require_unconsumed_ledger(LEDGER, opened_record, args.resume)
        model_hashes = {
            label: authenticate_model_tree(label, models[label])
            for label in MODEL_ORDER
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    budget_label = str(args.think_budget) if args.think_budget is not None else "native"
    output_dir = (
        EXP / "runs" / "benchmark"
        / f"{args.tier}_tb{budget_label}_seed{args.seed}_{args.name}"
    )
    if output_dir.exists() and not args.resume:
        parser.error("partial event exists; use --resume after auditing it")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write-ahead record: the seed is spent the moment the first gateway
    # call can start, so a mid-event crash leaves a permanent trace.
    if not ledger_rows(LEDGER):
        LEDGER.parent.mkdir(parents=True, exist_ok=True)
        with LEDGER.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(opened_record, sort_keys=True) + "\n")

    events = {}
    for label in MODEL_ORDER:
        model = models[label]
        output = output_dir / f"{label}.json"
        failure = output_dir / f"{label}.failure.json"
        if failure.exists():
            parser.error(f"preserved failure exists for {label}; audit before retrying")
        if not output.exists():
            command = [
                str(PYTHON), str(GATEWAY), "--tier", args.tier, "--seed", str(args.seed),
                "--model", str(model), "--out", str(output),
                "--think-budget", str(args.think_budget),
            ]
            completed = subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, check=False,
            )
            if completed.returncode != 0:
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
                        "seed": args.seed,
                        "arm": label,
                        "model": str(model),
                        "model_merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
                        "gateway_exit_code": completed.returncode,
                        "safe_diagnostic": diagnostic,
                        "score_emitted": False,
                        "raw_streams_exposed": False,
                        "benchmark_output_exposed": False,
                    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    f"aggregate gateway failed for {label} with exit {completed.returncode} "
                    f"({diagnostic}); "
                    "private output remained suppressed"
                )
        events[label] = load_event(output, model)

    signatures = {
        (
            event["benchmark_runner_sha256"],
            event["benchmark_source_inventory_sha256"],
            event["benchmark_source_file_count"],
        )
        for event in events.values()
    }
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between paired arms")
    runner_sha, inventory_sha, file_count = next(iter(signatures))

    base = events["base"]
    comparisons = {}
    for label in TREATED_ARMS:
        event = events[label]
        family_delta = {
            family: event["per_family"][family] - base["per_family"][family]
            for family in sorted(PUBLIC_FAMILIES)
        }
        comparisons[f"{label}_minus_base"] = {
            "aggregate_delta": event["aggregate"] - base["aggregate"],
            "per_family_delta": family_delta,
            "positive_families": sum(value > 0 for value in family_delta.values()),
            "nonnegative_families": sum(value >= 0 for value in family_delta.values()),
            "minimum_family_delta": min(family_delta.values()),
        }
    candidate = events[args.candidate]
    parent = events[FROZEN_PARENT]
    candidate_comparisons = {
        f"{args.candidate}_minus_{FROZEN_PARENT}": {
            "aggregate_delta": candidate["aggregate"] - parent["aggregate"],
            "per_family_delta": {
                family: candidate["per_family"][family] - parent["per_family"][family]
                for family in sorted(PUBLIC_FAMILIES)
            },
        }
    }
    consequence = consequence_reading(events, args.candidate)
    goal_gate = goal_gate_reading(events)
    # The summary payload is a PURE FUNCTION of the three gateway receipts
    # and the committed prerequisites (no wall-clock, no environment) — the
    # byte-equal crash reconciliation below depends on this.
    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": args.seed,
        "gateway_sha256": GATEWAY_SHA256,
        "local_design_receipt_sha256": sha256_file(LOCAL_DESIGN_RECEIPT),
        "count_walk_parent_merge_receipt_sha256": (
            COUNT_WALK_PARENT_MERGE_RECEIPT_SHA256
        ),
        "candidate": args.candidate,
        "local_promotion_receipt": str(PROMOTION_RECEIPT.resolve()),
        "local_promotion_receipt_sha256": sha256_file(PROMOTION_RECEIPT),
        "local_promotion_promoted": promotion["promoted"],
        "model_order": list(MODEL_ORDER),
        "models": {label: str(path) for label, path in models.items()},
        "model_tree_sha256s": {
            label: hashes["tree_sha256"] for label, hashes in model_hashes.items()
        },
        "model_weight_sha256s": {
            label: hashes["weights_sha256"] for label, hashes in model_hashes.items()
        },
        "benchmark_implementation": {
            "runner_sha256": runner_sha,
            "source_inventory_sha256": inventory_sha,
            "source_file_count": file_count,
        },
        "scores": {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        },
        "comparisons": comparisons,
        "candidate_comparisons": candidate_comparisons,
        "consequence": consequence,
        "goal_gate": goal_gate,
        "benchmark_data_read": False,
    }
    rendered = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    result = output_dir / "summary.json"
    if result.exists():
        # BYTE-EQUAL CRASH RECONCILIATION: a crash between the summary write
        # and the ledger close leaves a summary without a closed record. The
        # deterministic payload recomputed from the preserved receipts must
        # equal the preserved bytes exactly; then the closed record is
        # appended. Any divergence refuses forever.
        if result.read_bytes() != rendered:
            parser.error(
                "preserved event summary does not reconcile byte-identically "
                "with the receipts; refusing to close the ledger"
            )
    else:
        result.write_bytes(rendered)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "name": args.name, "phase": "closed", "tier": args.tier,
            "think_budget": args.think_budget, "seed": args.seed,
            "summary": str(result), "summary_sha256": sha256_file(result),
        }, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
