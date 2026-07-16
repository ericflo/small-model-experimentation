#!/usr/bin/env python3
"""Run the one conditional four-model medium pilot through the trusted gateway.

CONDITIONAL on local promotion: this event exists only if ``feedloop_scale``
passed the frozen seed-88037 local gate. Four composites (base, the
hygiene_explore parent, the replay_ctl3 control, and the promoted candidate)
are evaluated sequentially on the same sealed fresh seed 78158 at tier
medium, think budget 1024, in that frozen order. Discipline copied from the
hardened pilot runner PLUS the confirmation cell's receipt-pinned
closed-ledger pattern (that fix class is now standard):

- the PASS_BENCHMARK_EVENT review verdict and the frozen design receipts are
  enforced HERE, at the seed-consuming boundary (check_design --check and
  gen_local_gate --check re-run as subprocesses), not only in the harness: a
  direct invocation cannot consume the seed with unreviewed or drifted code;
- clean pushed main plus the committed-at-HEAD design receipt, local design
  receipt, benchmark design review, promotion receipt, local receipt, and
  every arm's committed merge receipt are hard prerequisites;
- every arm authenticates by recomputing its full on-disk tree sha256 (which
  covers the 9GB weights) against its pin — the two trained arms' pins are
  orchestrator-filled TODO-PINs that abort while None;
- one-seed WRITE-AHEAD ledger with RECEIPT-SHA-PINNED close: an ``opened``
  record is appended before the first gateway call; the ``closed`` record
  appended after the summary sha-pins the summary AND all four per-arm
  gateway receipts, so every verdict input is provenance-anchored at close
  time. Any closed record refuses forever; a crashed event forces recovery
  through ``--resume`` whose opened record must match exactly; an UNOPENED
  event refuses pre-existing receipt/failure/summary files unconditionally;
  a crash in the window between the summary write and the closed-record
  append regenerates the summary deterministically from the authenticated
  receipts and requires BYTE-IDENTICAL equality before closing;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a NaN
  can never silently drop a family from the strict-win partition.

PILOT GATE (recorded in the summary): candidate aggregate strictly > base
AND > replay_ctl3 AND > hygiene_explore_parent. The GOAL GATE (all ten
families strictly > base) is recorded from the same event either way, with
the frozen power statement. The benchmark suite directory is never read;
only scripts/run_benchmark_aggregate.py runs.
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
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
LOCAL_DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
LOCAL_SEED = 88037
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}_promotion.json"
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
FROZEN_NAME = "pilot"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78158
FROZEN_PARENT = "hygiene_explore_parent"
FROZEN_REPLAY_CONTROL = "replay_ctl3"
FROZEN_CANDIDATES = ("feedloop_scale",)
MODEL_ORDER = ("base", FROZEN_PARENT, FROZEN_REPLAY_CONTROL, "feedloop_scale")
TREATED_ARMS = (FROZEN_PARENT, FROZEN_REPLAY_CONTROL, "feedloop_scale")
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
    FROZEN_REPLAY_CONTROL: (
        ROOT / "large_artifacts" / EXP.name / "merged" / "replay_ctl3"
    ),
    "feedloop_scale": ROOT / "large_artifacts" / EXP.name / "merged" / "feedloop_scale",
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Inherited arms are pinned at design time from committed
# receipts; the two trained arms are TODO-PINs the orchestrator fills after
# their merges publish. A None pin aborts the event.
FROZEN_TREE_SHA256: dict[str, str | None] = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    FROZEN_PARENT: (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
    FROZEN_REPLAY_CONTROL: None,
    "feedloop_scale": None,
}
FROZEN_WEIGHTS_SHA256: dict[str, str | None] = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    FROZEN_PARENT: (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
    FROZEN_REPLAY_CONTROL: None,
    "feedloop_scale": None,
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# The parent arm carries a committed merge receipt at its source experiment;
# the two trained arms carry committed receipts here (sha pinned by the
# orchestrator after merge); base's reserialization receipt lives inside the
# composite.
COMMITTED_MERGE_RECEIPTS: dict[str, tuple[str, str | None]] = {
    FROZEN_PARENT: (
        "experiments/qwen35_4b_hygiene_explore_destack_medium"
        "/runs/merges/hygiene_explore.json",
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
    ),
    FROZEN_REPLAY_CONTROL: (
        f"experiments/{EXP.name}/runs/merges/replay_ctl3.json",
        None,
    ),
    "feedloop_scale": (
        f"experiments/{EXP.name}/runs/merges/feedloop_scale.json",
        None,
    ),
}
EXPECTED_MERGE_RECEIPT_NAMES = {
    FROZEN_PARENT: "hygiene_explore",
    FROZEN_REPLAY_CONTROL: "replay_ctl3",
    "feedloop_scale": "feedloop_scale",
}
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
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
CLOSED_RECORD_KEYS = frozenset(
    {
        "name", "phase", "tier", "think_budget", "seed",
        "summary", "summary_sha256", "receipts",
    }
)
POWER_STATEMENT = (
    "menders alone gates the all-families goal (nine families hold vs base "
    "on every sealed seed; the ties are 0-margin); three small-dose "
    "pedagogies failed at it and dose scale is the one permitted mechanism "
    "class, so menders > 0 for the candidate on this seed is the reading of "
    "consequence; any 10/10 feeds a fresh confirmation cell (independent "
    "seeds + matched compute) before any claim"
)


def require_pin(value, name: str):
    if value is None:
        raise ValueError(
            f"frozen constant {name} is unpinned (TODO-PIN); the orchestrator "
            "must fill it from the committed merge receipts before this event"
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


def opened_record() -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": FROZEN_SEED,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def is_closed_record(row: object) -> bool:
    """A well-formed receipt-pinned closed record; anything else fails closed.

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
        and row["seed"] == FROZEN_SEED
        and row["summary"] == str(EVENT_DIR / "summary.json")
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


def ledger_plan(rows: list[object], resume: bool) -> str:
    """One-seed write-ahead budget with a receipt-pinned close.

    The only valid histories are the empty ledger (fresh), exactly the
    frozen opened record (a crashed event, which may only continue under an
    explicit ``--resume``), or opened followed by a well-formed closed
    record (the budget is spent and every new invocation refuses, resume or
    not). Anything else — legacy rows, malformed rows, trailing rows — fails
    closed.
    """
    if not rows:
        return "fresh"
    if rows[0] != opened_record():
        raise ValueError(
            "benchmark ledger row 1 does not match the frozen opened record"
        )
    if len(rows) == 1:
        if not resume:
            raise ValueError(
                "benchmark ledger has an opened (crashed) event; audit the "
                "preserved receipts and use --resume"
            )
        return "crashed"
    if len(rows) == 2 and is_closed_record(rows[1]):
        raise ValueError(
            "benchmark ledger already has a closed entry; the one-event "
            "budget is spent"
        )
    raise ValueError("benchmark ledger history is not a valid one-event record")


def stale_event_files(output_dir: Path) -> list[str]:
    """Event files that must never predate the opened record.

    An UNOPENED event requires a clean slate: receipt, failure, or summary
    files already present in its event directory are refused unconditionally
    — only a crashed (opened) event may reuse preserved receipts, whose shas
    the closed record will pin.
    """
    if not output_dir.is_dir():
        return []
    names = [f"{label}.json" for label in MODEL_ORDER]
    names += [f"{label}.failure.json" for label in MODEL_ORDER]
    names.append("summary.json")
    return sorted(name for name in names if (output_dir / name).exists())


def reconcile_crashed_summary(summary_path: Path, rendered: bytes) -> None:
    """Recover the crash window between summary write and closed append.

    The summary is a pure function of the authenticated receipts and the
    frozen pins, so an honest crash in that window regenerates it
    BYTE-IDENTICALLY; equality lets the closed record be appended and the
    event finish. Any divergence is tampering or drift and refuses loudly
    with both digests.
    """
    existing = summary_path.read_bytes()
    if existing != rendered:
        raise ValueError(
            "crashed event has a summary that does not match its "
            "deterministic regeneration (existing sha256 "
            f"{hashlib.sha256(existing).hexdigest()}, regenerated sha256 "
            f"{hashlib.sha256(rendered).hexdigest()}); audit the event "
            "directory before any recovery"
        )


def _valid_score(value: object) -> bool:
    """A gateway score must be a finite float in [0, 1]; NaN never passes."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 1.0
    )


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
        or payload.get("seed") != LOCAL_SEED
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


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def pilot_gate(events: dict[str, dict], candidate: str) -> dict:
    """Frozen pilot promotion reading: strict aggregate wins over all three."""
    scores = {label: events[label]["aggregate"] for label in MODEL_ORDER}
    gate = {
        "strictly_beats_base_aggregate": scores[candidate] > scores["base"],
        "strictly_beats_replay_control_aggregate": (
            scores[candidate] > scores[FROZEN_REPLAY_CONTROL]
        ),
        "strictly_beats_immediate_parent_aggregate": (
            scores[candidate] > scores[FROZEN_PARENT]
        ),
    }
    gate["passes_pilot_gate"] = all(gate.values())
    return gate


def goal_gate_reading(events: dict[str, dict]) -> dict:
    """Strict wins / ties / losses vs base per treated arm; pass = ten wins."""
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
            "menders_margin_vs_base": values["menders"] - base["menders"],
        }
    return {
        "per_arm": table,
        "included_in_pilot_gate": False,
        "recorded_either_way": True,
        "power_statement": POWER_STATEMENT,
    }


def render_summary(
    events: dict[str, dict],
    models: dict[str, Path],
    model_hashes: dict[str, dict],
    promotion: dict,
    candidate: str,
    think_budget: int,
) -> bytes:
    """Deterministic summary bytes: a pure function of receipts + pins."""
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
    candidate_event = events[candidate]
    candidate_comparisons = {candidate: {}}
    for control_label in (FROZEN_PARENT, FROZEN_REPLAY_CONTROL):
        control = events[control_label]
        candidate_comparisons[candidate][f"{candidate}_minus_{control_label}"] = {
            "aggregate_delta": candidate_event["aggregate"] - control["aggregate"],
            "per_family_delta": {
                family: candidate_event["per_family"][family]
                - control["per_family"][family]
                for family in sorted(PUBLIC_FAMILIES)
            },
        }
    payload = {
        "schema_version": 1,
        "name": FROZEN_NAME,
        "tier": FROZEN_TIER,
        "think_budget": think_budget,
        "seed": FROZEN_SEED,
        "gateway_sha256": GATEWAY_SHA256,
        "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "local_design_receipt_sha256": sha256_file(LOCAL_DESIGN_RECEIPT),
        "candidate": candidate,
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
        "gateway_receipt_sha256s": {
            label: sha256_file(EVENT_DIR / f"{label}.json") for label in MODEL_ORDER
        },
        "scores": {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        },
        "comparisons": comparisons,
        "candidate_comparisons": candidate_comparisons,
        "promotions": {candidate: pilot_gate(events, candidate)},
        "goal_gate": goal_gate_reading(events),
        "benchmark_data_read": False,
    }
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


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
        parser.error("benchmark event differs from the preregistered pilot")
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
            "models must be exactly base, hygiene_explore_parent, replay_ctl3, "
            "and feedloop_scale"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    try:
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        promotion = authenticate_local_promotion(args.candidate)
        require_clean_pushed_main(
            [
                DESIGN_RECEIPT,
                LOCAL_DESIGN_RECEIPT,
                BENCH_REVIEW,
                PROMOTION_RECEIPT,
                LOCAL_RECEIPT,
            ]
            + [
                ROOT / relative
                for relative, _ in COMMITTED_MERGE_RECEIPTS.values()
            ]
        )
        # Re-verify the frozen design at the seed-consuming boundary: the
        # local design receipt pins the gate/harness/runner code set and the
        # design receipt pins the construction, so a committed drift of
        # either fails here before any gateway call.
        for check_script in ("check_design.py", "gen_local_gate.py"):
            subprocess.run(
                [str(PYTHON), "-B", str(SCRIPTS / check_script), "--check"],
                cwd=ROOT,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=subprocess.DEVNULL,
                check=True,
            )
        plan = ledger_plan(ledger_rows(LEDGER), args.resume)
        if plan == "fresh":
            stale = stale_event_files(EVENT_DIR)
            if stale:
                raise ValueError(
                    "the event was never opened but its directory already "
                    f"contains {stale}; an unopened event requires a clean "
                    "slate (only a crashed, opened event may reuse preserved "
                    "receipts)"
                )
        model_hashes = {
            label: authenticate_model_tree(label, models[label])
            for label in MODEL_ORDER
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    # Write-ahead record: the seed is spent the moment the first gateway
    # call can start, so a mid-event crash leaves a permanent trace. A
    # crashed event already has its opened record and must not open twice.
    if plan == "fresh":
        append_ledger(opened_record())

    events = {}
    for label in MODEL_ORDER:
        model = models[label]
        output = EVENT_DIR / f"{label}.json"
        failure = EVENT_DIR / f"{label}.failure.json"
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

    rendered = render_summary(
        events, models, model_hashes, promotion, args.candidate, args.think_budget
    )
    summary_path = EVENT_DIR / "summary.json"
    if summary_path.exists():
        # Crash window between the summary write and the closed-record
        # append (only a crashed event can reach here; unopened events
        # refused above): the deterministic regeneration must match
        # byte-for-byte, then the close proceeds.
        try:
            reconcile_crashed_summary(summary_path, rendered)
        except ValueError as error:
            parser.error(str(error))
    else:
        summary_path.write_bytes(rendered)
    # The closed record pins every verdict input: the sealed summary AND all
    # four per-arm gateway receipts, by sha256.
    append_ledger({
        "name": args.name, "phase": "closed", "tier": args.tier,
        "think_budget": args.think_budget, "seed": args.seed,
        "summary": str(summary_path), "summary_sha256": sha256_file(summary_path),
        "receipts": {
            label: sha256_file(EVENT_DIR / f"{label}.json")
            for label in MODEL_ORDER
        },
    })
    print(rendered.decode("utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
