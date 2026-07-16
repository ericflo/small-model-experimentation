#!/usr/bin/env python3
"""Run the ONE sealed single-seed three-arm zero-root event through the gateway.

MEASUREMENT of a provenance question, not a promotion: there is no
candidate, no local gate, and no pass bar anywhere in this file. The six
documented lineage stages, replayed from a fresh zero-initialized root
(no undocumented C53-era blend adapter), are measured ONCE against the
original blend-rooted composite and the untouched base: tier medium,
think budget 1024, ONE fresh sealed seed 78159, THREE arms in the frozen
order base, hygiene_explore_original, zero_root_hygiene_explore. Three
gateway runs total; the process exits 0 on the complete event.
Discipline copied from the hardened goal-gate confirmation runner:

- the PASS_BENCHMARK_EVENT review verdict and the design receipt's code
  pins are enforced HERE, at the seed-consuming boundary
  (gen_design_receipt --check re-runs as a subprocess), not only in the
  harness: a direct invocation cannot consume the seed with unreviewed
  or drifted code;
- clean pushed main plus the committed-at-HEAD design receipt, benchmark
  design review, AND the committed zero-root rebuild receipts (six stage
  receipts + the merge receipt, sha-pinned) are hard prerequisites;
- the zero-root arm's tree/weights/receipt pins are TODO-PINs filled
  only after the merge exists; every pin refuses unfilled (fail closed).
  This file is frozen by a NORMALIZED-HASH code pin in the design
  receipt: exactly the three pin VALUE slots are canonicalized to a
  fixed placeholder before hashing, so EVERY OTHER BYTE of this runner
  — including every guard call site below — is byte-frozen pre- and
  post-fill, and any drift fails ``gen_design_receipt --check`` at the
  seed-consuming boundary;
- every arm authenticates by recomputing its full on-disk tree sha256
  (which covers the 9GB weights) against its pin;
- SINGLE-SEED WRITE-AHEAD ledger: an ``opened`` record is appended
  before the first gateway call and a ``closed`` record after the sealed
  summary. The closed record sha-pins the summary AND ALL THREE per-arm
  gateway receipts, so every verdict input is provenance-anchored at
  close time. A closed seed refuses forever; a crash mid-event leaves a
  permanent opened record that forces recovery through ``--resume`` with
  the preserved receipts. An UNOPENED seed requires a clean slate. A
  crash between the summary write and the closed append regenerates the
  summary deterministically and requires BYTE-IDENTICAL equality;
- implementation-signature integrity: the three receipts must share one
  (runner sha256, source inventory sha256, file count) signature AND
  match the pinned reference block (the signature of the discovery and
  confirmation events under which the original composite recorded its
  two 10/10 sweeps) — fail closed, or the contrast is not comparable;
- gateway failures leave a safe failure receipt with the sanitized
  diagnostic only; child stdout/stderr never surface here;
- every score must be a finite float in [0, 1]; a NaN can never silently
  drop a family from the strict-win partition (finiteness guards);
- NO local wall-time cap: the gateway owns budget policy; this runner
  records ``within_budget`` and ``wall_seconds`` exactly as returned and
  never gates on them — budget_integrity in check_benchmark.py scopes
  the paired comparison instead.

After the seed closes, check_benchmark.py is invoked to write the frozen
readout. The benchmark suite directory is never read; only
scripts/run_benchmark_aggregate.py runs.
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
FROZEN_NAME = "zero_root"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78159
MODEL_ORDER = ("base", "hygiene_explore_original", "zero_root_hygiene_explore")
ORIGINAL_ARM = "hygiene_explore_original"
ZERO_ROOT_ARM = "zero_root_hygiene_explore"
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    "hygiene_explore_original": (
        ROOT / "large_artifacts" / "qwen35_4b_hygiene_explore_destack_medium"
        / "merged" / "hygiene_explore"
    ),
    "zero_root_hygiene_explore": (
        ROOT / "large_artifacts" / "qwen35_4b_zero_root_lineage_rebuild"
        / "merged" / "zero_root_hygiene_explore"
    ),
}
# Full on-disk tree hashes, recomputed at event time; the tree manifest
# covers every file including model.safetensors, so a tree match implies a
# weights match. Base and the original composite carry the pins recorded by
# the goal-gate discovery/confirmation events. The zero-root arm does not
# exist until --stage rebuild completes: its three pins are TODO-PINs
# filled from the committed runs/lineage/merge.json, and every unfilled pin
# refuses fail-closed below.
FROZEN_TREE_SHA256 = {
    "base": "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677",
    "hygiene_explore_original": (
        "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971"
    ),
    # TODO-PIN(post-merge): runs/lineage/merge.json -> output_tree_sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE; the
    # design receipt's normalized-hash pin canonicalizes only this value.
    "zero_root_hygiene_explore": None,
}
FROZEN_WEIGHTS_SHA256 = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "hygiene_explore_original": (
        "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f"
    ),
    # TODO-PIN(post-merge): runs/lineage/merge.json -> weights_sha256.
    # Fill by replacing None with the quoted 64-hex ON THIS LINE; the
    # design receipt's normalized-hash pin canonicalizes only this value.
    "zero_root_hygiene_explore": None,
}
# TODO-PIN(post-merge): sha256 of the COMMITTED runs/lineage/merge.json.
# Fill by replacing None with the quoted 64-hex ON THIS LINE; the design
# receipt's normalized-hash pin canonicalizes only this value.
ZERO_ROOT_MERGE_RECEIPT_SHA256 = None
ZERO_ROOT_MERGE_RECEIPT = EXP / "runs" / "lineage" / "merge.json"
ZERO_ROOT_STAGE_RECEIPTS = tuple(
    f"runs/lineage/{name}.json"
    for name in (
        "stage01_replay_refresh",
        "stage02_designed160",
        "stage03_close_xi",
        "stage04_replay_after_close",
        "stage05_designed_fresh",
        "stage06_hygiene_explore",
    )
)
WEIGHTS_SIZE_BYTES = 9_078_620_536
# The original composite's committed merge receipt at its source experiment;
# base's reserialization receipt lives inside the composite. The receipt's
# recorded name is "hygiene_explore" (the arm label here adds _original).
COMMITTED_MERGE_RECEIPTS = {
    "hygiene_explore_original": (
        "experiments/qwen35_4b_hygiene_explore_destack_medium"
        "/runs/merges/hygiene_explore.json",
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
        "hygiene_explore",
    ),
}
BASE_MERGE_RECEIPT_SHA256 = (
    "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
)
# The benchmark-implementation signature of the discovery (seed 78154) and
# confirmation (78155-78157) events under which the original composite
# recorded its two 10/10 goal-gate sweeps. All three receipts of this event
# must carry exactly this signature or the cross-event framing fails closed.
REFERENCE_IMPLEMENTATION = {
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
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
READOUT = EXP / "runs" / "benchmark" / "zero_root_readout.json"
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


def require_todo_pins_filled() -> None:
    """Every zero-root pin must be a filled sha256; None refuses fail-closed."""
    sha_re = re.compile(r"[0-9a-f]{64}")
    for label, value in (
        ("zero_root tree", FROZEN_TREE_SHA256[ZERO_ROOT_ARM]),
        ("zero_root weights", FROZEN_WEIGHTS_SHA256[ZERO_ROOT_ARM]),
        ("zero_root merge receipt", ZERO_ROOT_MERGE_RECEIPT_SHA256),
    ):
        if not isinstance(value, str) or sha_re.fullmatch(value) is None:
            raise ValueError(
                f"{label} pin is unfilled (TODO-PIN): run --stage rebuild, "
                "commit runs/lineage/merge.json, fill the three pins from it, "
                "and re-seek review before any seed can be consumed"
            )


def require_zero_root_provenance() -> None:
    """The committed merge receipt must pin exactly the frozen zero-root arm."""
    if (
        not ZERO_ROOT_MERGE_RECEIPT.is_file()
        or sha256_file(ZERO_ROOT_MERGE_RECEIPT) != ZERO_ROOT_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            "committed zero-root merge receipt is absent or changed: "
            f"{ZERO_ROOT_MERGE_RECEIPT}"
        )
    payload = json.loads(ZERO_ROOT_MERGE_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("stage") != "merge"
        or payload.get("name") != ZERO_ROOT_ARM
        or payload.get("base_model", {}).get("id") != MODEL_ID
        or payload.get("base_model", {}).get("revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve()
        != FROZEN_MODEL_PATHS[ZERO_ROOT_ARM].resolve()
        or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[ZERO_ROOT_ARM]
        or payload.get("weights_sha256") != FROZEN_WEIGHTS_SHA256[ZERO_ROOT_ARM]
        or payload.get("weights_size_bytes") != WEIGHTS_SIZE_BYTES
    ):
        raise ValueError(
            "zero-root merge receipt does not describe the frozen zero-root arm"
        )


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
    """A well-formed closed record; anything else fails closed.

    Besides the summary sha, the closed record must sha-pin ALL THREE
    per-arm gateway receipts: the verdict inputs are provenance-anchored
    at close time and any later receipt swap fails against these pins.
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


def ledger_plan(rows: list[object], resume: bool) -> dict:
    """Single-seed write-ahead budget: parse the ledger into a plan.

    The only valid history is a prefix of the canonical sequence
    opened(78159), closed(78159). Anything else — legacy rows, malformed
    rows, extra rows — fails closed. A closed seed is NEVER re-run: once
    both rows exist the event budget is spent and every new invocation
    refuses, resume or not. A trailing opened record is a crashed event:
    it may only continue under an explicit ``--resume``, and the opened
    record must match the frozen record exactly.
    """
    if not rows:
        return {"status": "fresh", "closed": None}
    if rows[0] != opened_record():
        raise ValueError(
            "benchmark ledger row 1 does not match the frozen opened record "
            f"for seed {FROZEN_SEED}"
        )
    if len(rows) == 1:
        if not resume:
            raise ValueError(
                "benchmark ledger has a crashed opened event; audit the "
                "preserved receipts and use --resume"
            )
        return {"status": "crashed", "closed": None}
    if not is_closed_record(rows[1]):
        raise ValueError(
            "benchmark ledger row 2 is not the closed record for seed "
            f"{FROZEN_SEED}"
        )
    if len(rows) != 2:
        raise ValueError(
            "benchmark ledger has rows beyond the frozen single-seed event"
        )
    raise ValueError(
        f"seed {FROZEN_SEED} is closed; the single-seed budget is spent and "
        "the event never re-runs"
    )


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
        relative, expected, receipt_name = COMMITTED_MERGE_RECEIPTS[label]
        receipt_path = ROOT / relative
        if not receipt_path.is_file() or sha256_file(receipt_path) != expected:
            raise ValueError(f"committed merge receipt is absent or changed: {relative}")
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
        if (
            payload.get("name") != receipt_name
            or payload.get("model_id") != MODEL_ID
            or payload.get("model_revision") != MODEL_REVISION
            or Path(payload.get("merged", "")).resolve() != model.resolve()
            or payload.get("output_tree_sha256") != FROZEN_TREE_SHA256[label]
            or {row.get("name"): row.get("sha256") for row in payload.get("weight_files", [])}
            != {"model.safetensors": FROZEN_WEIGHTS_SHA256[label]}
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
    elif label == ZERO_ROOT_ARM:
        require_zero_root_provenance()
        receipt = json.loads(ZERO_ROOT_MERGE_RECEIPT.read_text(encoding="utf-8"))
        if files["merge_receipt.json"]["sha256"] != receipt.get(
            "inner_merge_receipt_sha256"
        ):
            raise ValueError(
                "zero-root composite's inner merge receipt does not match the "
                "committed rebuild merge receipt"
            )
    else:
        if files["merge_receipt.json"]["sha256"] != BASE_MERGE_RECEIPT_SHA256:
            raise ValueError("base reserialization receipt changed")
    return {"tree_sha256": observed_tree, "weights_sha256": weights["sha256"]}


def load_event(path: Path, model: Path) -> dict:
    """Authenticate one aggregate-gateway receipt against the frozen seed.

    ``within_budget`` must be a strict bool but is RECORDED, never required
    to be true: an over-budget arm keeps its scores and is scoped by the
    budget_integrity reading instead of being rejected here. The gateway
    owns budget policy.
    """
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


def stale_event_files(output_dir: Path) -> list[str]:
    """Event files that must never predate the opened record.

    An UNOPENED seed requires a clean slate: receipt, failure, or summary
    files already present in the event directory are refused
    unconditionally — only a crashed (opened) event may reuse preserved
    receipts, whose shas the closed record will pin.
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


def run_gateway_arm(
    args: argparse.Namespace, label: str, model: Path, output: Path
) -> None:
    """One gateway call; on failure preserve a sanitized receipt and stop."""
    failure = output.parent / f"{label}.failure.json"
    command = [
        str(PYTHON), str(GATEWAY), "--tier", args.tier, "--seed", str(FROZEN_SEED),
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
            "seed": FROZEN_SEED,
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
        f"aggregate gateway failed for arm {label} with exit "
        f"{completed.returncode} ({diagnostic}); "
        "private output remained suppressed"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or args.seed != FROZEN_SEED
        or args.think_budget != FROZEN_THINK_BUDGET
    ):
        parser.error("benchmark event differs from the frozen zero-root design")
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
            "models must be exactly base, hygiene_explore_original, and "
            "zero_root_hygiene_explore"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    try:
        require_todo_pins_filled()
        if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
            raise ValueError("trusted gateway is absent or changed")
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
        require_clean_pushed_main(
            [DESIGN_RECEIPT, BENCH_REVIEW, ZERO_ROOT_MERGE_RECEIPT]
            + [EXP / relative for relative in ZERO_ROOT_STAGE_RECEIPTS]
            + [
                ROOT / relative
                for relative, _, _ in COMMITTED_MERGE_RECEIPTS.values()
            ]
        )
        # Re-verify the design receipt at the seed-consuming boundary: its
        # code pins cover the harness, the rebuild script, and
        # check_benchmark.py, and its NORMALIZED-HASH pin covers every byte
        # of THIS file except the three canonicalized pin value slots — so a
        # committed drift (including any deleted guard call above or below)
        # fails here before any gateway call.
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

    output_dir = EVENT_DIR
    summary_path = output_dir / "summary.json"
    if plan["status"] == "fresh":
        stale = stale_event_files(output_dir)
        if stale:
            parser.error(
                f"seed {FROZEN_SEED} was never opened but its event directory "
                f"already contains {stale}; an unopened seed requires a clean "
                "slate (only a crashed, opened event may reuse preserved "
                "receipts)"
            )
        if output_dir.exists() and not args.resume:
            parser.error(
                "partial event directory exists; use --resume after auditing it"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write-ahead record: the seed is spent the moment the first gateway
    # call can start, so a mid-event crash leaves a permanent trace. A
    # crashed event already has its opened record (matched exactly by the
    # ledger plan) and must not open twice.
    if plan["status"] == "fresh":
        append_ledger(opened_record())
    events = {}
    for label in MODEL_ORDER:
        model = models[label]
        output = output_dir / f"{label}.json"
        failure = output_dir / f"{label}.failure.json"
        if failure.exists():
            parser.error(
                f"preserved failure exists for arm {label}; audit before retrying"
            )
        if not output.exists():
            run_gateway_arm(args, label, model, output)
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
        raise ValueError(
            "benchmark implementation changed between the three paired arms"
        )
    runner_sha, inventory_sha, file_count = next(iter(signatures))
    implementation = {
        "runner_sha256": runner_sha,
        "source_inventory_sha256": inventory_sha,
        "source_file_count": file_count,
    }
    if implementation != REFERENCE_IMPLEMENTATION:
        raise ValueError(
            "benchmark implementation differs from the pinned reference "
            f"events ({implementation} != {REFERENCE_IMPLEMENTATION}); the "
            "cross-event framing is not comparable"
        )

    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": FROZEN_SEED,
        "gateway_sha256": GATEWAY_SHA256,
        "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "zero_root_merge_receipt_sha256": ZERO_ROOT_MERGE_RECEIPT_SHA256,
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
        # append (only a crashed event can reach here; unopened events
        # refused above): the deterministic regeneration must match
        # byte-for-byte, then the close proceeds.
        try:
            reconcile_crashed_summary(summary_path, rendered)
        except ValueError as error:
            parser.error(str(error))
    else:
        summary_path.write_bytes(rendered)
    summary_sha = sha256_file(summary_path)
    # The closed record pins every verdict input: the sealed summary AND
    # all three per-arm gateway receipts, by sha256.
    append_ledger({
        "name": args.name, "phase": "closed", "tier": args.tier,
        "think_budget": args.think_budget, "seed": FROZEN_SEED,
        "summary": str(summary_path), "summary_sha256": summary_sha,
        "receipts": {
            label: sha256_file(output_dir / f"{label}.json")
            for label in MODEL_ORDER
        },
    })

    # The seed is closed: write (or verify) the terminal readout.
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
        "seed": FROZEN_SEED,
        "summary_sha256": summary_sha,
        "readout": str(READOUT),
        "consequence": json.loads(READOUT.read_text(encoding="utf-8"))["consequence"],
    }, indent=1, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
