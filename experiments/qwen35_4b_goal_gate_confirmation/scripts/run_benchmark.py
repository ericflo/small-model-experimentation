#!/usr/bin/env python3
"""Run the three sealed per-seed CONFIRMATION events through the trusted gateway.

MEASUREMENT INTAKE, not a promotion: there is no candidate, no local
gate, and no pass bar anywhere in this file. The recorded seed-78154
goal-gate pass (hygiene_explore over base on ALL TEN public families) is
replicated on THREE independent sealed fresh seeds — 78155, 78156,
78157 — at tier medium, think budget 1024, TWO arms per seed in the
frozen order base then hygiene_explore, seed-major (a seed's two arms
complete and close before the next seed opens). Six gateway runs total;
the process exits 0 on any complete three-seed event. Discipline copied
from the hardened budget-probe measurement runner:

- the PASS_BENCHMARK_EVENT review verdict and the design receipt's code
  pins are enforced HERE, at the seed-consuming boundary (gen_design_receipt
  --check re-runs as a subprocess), not only in the harness: a direct
  invocation cannot consume any seed with unreviewed or drifted code;
- clean pushed main plus the committed-at-HEAD design receipt,
  preregistration, benchmark design review, the committed hygiene_explore
  merge receipt, AND the committed discovery-seed summary (sha256-pinned)
  are hard prerequisites;
- every arm authenticates by recomputing its full on-disk tree sha256
  (which covers the 9GB weights) against the design-time pin, base
  additionally against its frozen reserialized-weights hash;
- K-SEED WRITE-AHEAD ledger: each seed gets its own ``opened`` record
  appended before its first gateway call and its own ``closed`` record
  after its per-seed summary. The closed record sha-pins the summary AND
  BOTH per-arm gateway receipts, so every verdict input is provenance-
  anchored at close time (check_benchmark refuses receipts that do not
  match the sealed pins). The only valid ledger history is a prefix of
  the canonical seed-major sequence; a closed record refuses its seed
  forever (completed seeds are never re-run); a crash mid-seed leaves a
  permanent opened record that forces recovery through ``--resume`` with
  the preserved receipts, and ``--resume`` must match the per-seed opened
  record exactly. RECOVERY SEMANTICS: an UNOPENED seed requires a clean
  slate — pre-existing receipt/failure/summary files in a never-opened
  seed's event directory refuse unconditionally; only a crashed (opened)
  seed may reuse its preserved receipts. If the crash landed in the
  window between the summary write and the closed-record append, the
  summary is regenerated deterministically from the authenticated
  receipts and must compare BYTE-IDENTICAL to the file on disk — a match
  appends the closed record and continues, a mismatch refuses loudly
  with both digests. The OVERALL event completes only when all three
  seeds close; once they have, every new invocation refuses, resume or
  not;
- implementation-signature integrity per seed: the two receipts must
  share one (runner sha256, source inventory sha256, file count)
  signature AND match the pinned discovery summary's block, before the
  seed's summary is written — all six receipts are thereby anchored to
  the discovery event, fail closed;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score in a gateway receipt must be a finite float in [0, 1]; a NaN
  can never silently drop a family from the strict-win partition;
- NO local wall-time cap: the gateway owns budget policy; this runner
  records ``within_budget`` and ``wall_seconds`` exactly as returned (a
  strict bool / finite non-negative number) and never gates on them —
  the budget_integrity reading in check_benchmark.py scopes the paired
  comparison instead.

After the third seed closes, check_benchmark.py is invoked to write the
preregistered confirmation readout. The benchmark suite directory is
never read; only scripts/run_benchmark_aggregate.py runs.
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
SEED_ORDER = (78155, 78156, 78157)
MODEL_ORDER = ("base", "hygiene_explore")
TREATED_ARM = "hygiene_explore"
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Identical to the discovery event's pins.
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
}
WEIGHTS_SIZE_BYTES = 9_078_620_536
# The treated arm carries a committed merge receipt at its source
# experiment; base's reserialization receipt lives inside the composite.
COMMITTED_MERGE_RECEIPTS = {
    "hygiene_explore": (
        "experiments/qwen35_4b_hygiene_explore_destack_medium"
        "/runs/merges/hygiene_explore.json",
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
    ),
}
BASE_MERGE_RECEIPT_SHA256 = (
    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
)
# The committed discovery event at seed 78154 whose 10/10 pass this cell
# replicates. Pinned here so a drifted discovery source fails BEFORE any
# seed is consumed; reported by check_benchmark.py, never counted.
DISCOVERY_SUMMARY = (
    "experiments/qwen35_4b_statechain_only_dose"
    "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json"
)
DISCOVERY_SUMMARY_SHA256 = (
    "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa"
)
# Every receipt of the six must carry exactly this signature (the
# discovery summary's benchmark_implementation block) or the event fails
# closed at its seed's summary boundary.
DISCOVERY_IMPLEMENTATION = {
    "runner_sha256": (
        "a3beecd8b5c89ccfd99a172a6d85321d39b9feb6c29d12f10b2f4d7499e273cb"
    ),
    "source_inventory_sha256": (
        "218b8615a95f24da962c931e9cd2dba58d853a7bdcd2847cd8e2c42fc2c05f42"
    ),
    "source_file_count": 56,
}
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


def require_discovery_reference() -> None:
    """Fail closed if the pinned discovery-seed summary drifted."""
    summary = ROOT / DISCOVERY_SUMMARY
    if not summary.is_file() or sha256_file(summary) != DISCOVERY_SUMMARY_SHA256:
        raise ValueError(
            f"pinned discovery-seed summary is absent or changed: {summary}"
        )


def ledger_rows(ledger: Path) -> list[dict]:
    if not ledger.exists():
        return []
    return [
        json.loads(line)
        for line in ledger.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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
    sequence opened(78155), closed(78155), opened(78156), closed(78156),
    opened(78157), closed(78157). Anything else — legacy rows, malformed
    rows, out-of-order rows, duplicate opened records, an opened record
    for a later seed before an earlier seed closed — fails closed. A
    closed seed is NEVER re-run; when all three seeds are closed the
    confirmation budget is spent and every new event refuses, resume or
    not. A trailing opened record is a crashed seed: the whole event may
    only continue under an explicit ``--resume``, and the opened record
    must match the frozen per-seed record exactly.
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
            "benchmark ledger has rows beyond the frozen three-seed event"
        )
    if all(entry["status"] == "closed" for entry in plan.values()):
        raise ValueError(
            "all three confirmation seeds are closed; the k-seed budget is spent"
        )
    if not resume:
        raise ValueError(
            "benchmark ledger has prior per-seed records; audit the preserved "
            "receipts and use --resume"
        )
    return plan


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
            payload.get("name") != label
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[label]
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def load_event(path: Path, model: Path, seed: int) -> dict:
    """Authenticate one aggregate-gateway receipt against its frozen seed.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: an over-budget arm keeps its scores and is scoped by the
    budget_integrity reading (paired_comparison_valid: false) instead of
    being rejected here. The gateway owns budget policy.
    """
    if seed not in SEED_ORDER:
        raise ValueError(f"receipt seed is not one of the frozen three: {seed}")
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


def append_ledger(record: dict) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def require_closed_seed_intact(seed: int, closed: dict, models: dict) -> None:
    """A closed seed is never re-run: its preserved artifacts must verify.

    Every verdict input is checked against the shas the closed record
    pinned at close time — the summary AND both per-arm receipts.
    """
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
        help="repeat three times in the frozen order",
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
        parser.error("models must be exactly base and hygiene_explore")
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    try:
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        require_discovery_reference()
        require_clean_pushed_main(
            [DESIGN_RECEIPT, PREREGISTRATION, BENCH_REVIEW, ROOT / DISCOVERY_SUMMARY]
            + [ROOT / relative for relative, _ in COMMITTED_MERGE_RECEIPTS.values()]
        )
        # Re-verify the design receipt at the seed-consuming boundary: its
        # code pins cover this script and check_benchmark.py, so a committed
        # drift of either fails here before any gateway call.
        subprocess.run(
            [str(PYTHON), "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"],
            cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            stdout=subprocess.DEVNULL,
            check=True,
        )
        plan = ledger_plan(ledger_rows(LEDGER), args.resume)
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
            raise ValueError(
                f"benchmark implementation changed between paired arms at seed {seed}"
            )
        runner_sha, inventory_sha, file_count = next(iter(signatures))
        implementation = {
            "runner_sha256": runner_sha,
            "source_inventory_sha256": inventory_sha,
            "source_file_count": file_count,
        }
        if implementation != DISCOVERY_IMPLEMENTATION:
            raise ValueError(
                f"benchmark implementation at seed {seed} differs from the "
                f"pinned discovery event ({implementation} != "
                f"{DISCOVERY_IMPLEMENTATION}); the confirmation is not comparable"
            )

        payload = {
            "schema_version": 1,
            "name": args.name,
            "tier": args.tier,
            "think_budget": args.think_budget,
            "seed": seed,
            "gateway_sha256": GATEWAY_SHA256,
            "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
            "discovery_summary_sha256": DISCOVERY_SUMMARY_SHA256,
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
        # AND both per-arm gateway receipts, by sha256.
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

    # All three seeds are closed: write (or verify) the terminal readout.
    check_command = [
        str(PYTHON), "-B", str(SCRIPTS / "check_benchmark.py"),
    ]
    if not READOUT.exists():
        check_command.extend(("--out", str(READOUT)))
    subprocess.run(
        check_command, cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=True,
    )
    print(json.dumps({
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seeds": list(SEED_ORDER),
        "summaries_sha256": {str(seed): summaries[seed] for seed in SEED_ORDER},
        "readout": str(READOUT),
        "verdict": json.loads(READOUT.read_text(encoding="utf-8"))["verdict"],
    }, indent=1, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
