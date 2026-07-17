#!/usr/bin/env python3
"""Checkpointed harness for the six-seed state_track confirmation.

EVAL-ONLY MEASUREMENT INTAKE (lifecycle 31): lifecycle 30's sealed
seed-78169 INSTALLED_TRANSFER — the ``state_track`` composite scored
aggregate 0.3260 versus the ``count_walk`` parent's 0.3004 (a paired lift
of +0.0256) — is tested for replication on six fresh sealed seeds (78170,
78171, 78172, 78173, 78174, 78175) at tier medium, think budget 1024, two
arms per seed in the frozen order ``count_walk`` then ``state_track``,
seed-major. This experiment trains nothing, merges nothing, builds no
corpus, and promotes nothing; there is no local gate and no adapter. The
only stage is ``benchmark`` and the preregistered confirmation readout
(CONFIRMED / NOT_CONFIRMED / AMBIGUOUS, no fourth state) is the terminal
artifact.

``--smoke`` verifies every pin (both in-cell provenance copies of the
committed merge receipts, the sha-pinned prior-event summary, the trusted
gateway, the preregistered power arithmetic via ``power_analysis.py
--check``, and the complete in-cell standalone stage 1-9 lineage package
via ``rebuild_lineage.py --verify-inputs``), authenticates any published
per-seed event artifacts against the k-seed ledger (including requiring
the terminal confirmation readout once all six seeds are closed,
byte-verified via ``check_benchmark.py``), compiles every script, and
runs the unit tests. It runs no model event and does NOT require the
benchmark design review to exist.

``--stage benchmark`` requires the committed-at-HEAD preregistration, the
adversarial benchmark design review carrying the literal verdict line,
the committed provenance copies and their committed sources, and clean
pushed green main, then runs the frozen six-seed event. ``--resume`` is
forwarded to run_benchmark.py only when explicitly passed by the operator.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
EXP_REL = f"experiments/{EXP.name}"
FROZEN_NAME = "confirmation"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
SEED_ORDER = (78170, 78171, 78172, 78173, 78174, 78175)
MODEL_ORDER = ("count_walk", "state_track")
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
PREREGISTRATION = EXP / "reports" / "preregistration.md"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
EVENT_DIRS = {
    seed: (
        EXP / "runs" / "benchmark"
        / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{seed}_{FROZEN_NAME}"
    )
    for seed in SEED_ORDER
}
READOUT = EXP / "runs" / "benchmark" / "confirmation_readout.json"
LEDGER = EXP / "runs" / "benchmark_events.jsonl"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"not a JSON object: {path}")
    return payload


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def load_run_benchmark():
    spec = importlib.util.spec_from_file_location(
        "state_track_confirmation_run_benchmark_harness", SCRIPTS / "run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def require_pushed_checkpoint(relative_path: str) -> None:
    path = ROOT / relative_path
    status = output(["git", "status", "--short"])
    branch = output(["git", "branch", "--show-current"])
    head = output(["git", "rev-parse", "HEAD"])
    origin = output(["git", "rev-parse", "origin/main"])
    if status or branch != "main" or head != origin:
        raise SystemExit("stage requires a clean pushed main checkpoint")
    run(["git", "cat-file", "-e", f"HEAD:{relative_path}"])
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if not path.is_file() or path.read_bytes() != committed:
        raise SystemExit(f"stage prerequisite differs from HEAD: {relative_path}")


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"{description} has not been authorized: {path}")


def opened_record(seed: int) -> dict:
    return {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": seed,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }


def authenticate_closed_seed(bench, seed: int, closed: dict) -> None:
    """One closed seed: summary AND both receipts must match the pins."""
    summary_path = EVENT_DIRS[seed] / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"published summary is absent for closed seed {seed}")
    receipt_shas = {}
    for label in MODEL_ORDER:
        receipt = EVENT_DIRS[seed] / f"{label}.json"
        if not receipt.is_file():
            raise SystemExit(
                f"published gateway receipt is absent for seed {seed} arm {label}"
            )
        receipt_shas[label] = sha256_file(receipt)
    if closed != {
        "name": FROZEN_NAME,
        "phase": "closed",
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": seed,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "receipts": receipt_shas,
    }:
        raise SystemExit(
            f"published benchmark ledger failed smoke authentication at seed "
            f"{seed}: the on-disk summary/receipt sha256s do not match the "
            "closed record's pins"
        )
    summary = load_json(summary_path)
    if (
        summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != seed
        or summary.get("model_order") != list(MODEL_ORDER)
        or set(summary.get("scores", {})) != set(MODEL_ORDER)
        or set(summary.get("budget", {})) != set(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("prior_summary_sha256") != bench.PRIOR_EVENT["summary_sha256"]
        or summary.get("benchmark_implementation") != bench.PRIOR_IMPLEMENTATION
    ):
        raise SystemExit(
            f"published benchmark summary failed smoke authentication at seed {seed}"
        )


def smoke_event_receipts(bench) -> None:
    """Authenticate any published event artifacts without running anything."""
    if not LEDGER.exists():
        if READOUT.exists() or any(EVENT_DIRS[seed].exists() for seed in SEED_ORDER):
            raise SystemExit("published event artifacts exist without a ledger")
        return
    rows = [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    closed_seeds = []
    index = 0
    for seed in SEED_ORDER:
        if index == len(rows):
            break
        if rows[index] != opened_record(seed):
            raise SystemExit(
                f"published benchmark ledger row {index + 1} does not match "
                f"the frozen opened record for seed {seed}"
            )
        index += 1
        if index == len(rows):
            raise SystemExit(
                f"benchmark ledger has a crashed opened event for seed {seed}; "
                "audit the preserved receipts and resume before smoke can pass"
            )
        authenticate_closed_seed(bench, seed, rows[index])
        closed_seeds.append(seed)
        index += 1
    if index != len(rows):
        raise SystemExit("published benchmark ledger has unexpected trailing rows")
    for seed in SEED_ORDER:
        if seed not in closed_seeds and EVENT_DIRS[seed].exists():
            raise SystemExit(
                f"event directory exists for seed {seed} without its ledger records"
            )
    if len(closed_seeds) == len(SEED_ORDER):
        if not READOUT.is_file():
            raise SystemExit(
                "published benchmark event lacks its terminal confirmation readout"
            )
        run([sys.executable, "-B", str(SCRIPTS / "check_benchmark.py")])
    elif READOUT.exists():
        raise SystemExit(
            "confirmation readout exists before all six seeds closed"
        )


def smoke() -> None:
    bench = load_run_benchmark()
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise SystemExit("trusted gateway is absent or changed")
    # The sha-pinned prior event (fail closed on drift) plus both in-cell
    # provenance copies (byte-identical to their pins; siblings verification
    # aids).
    try:
        bench.load_prior_reference()
        bench.require_provenance_copies()
    except ValueError as error:
        raise SystemExit(str(error))
    # The preregistered power arithmetic must recompute exactly.
    run([sys.executable, "-B", str(SCRIPTS / "power_analysis.py"), "--check"])
    # The complete in-cell standalone stage 1-9 lineage package must
    # authenticate against the extended manifest.
    run([sys.executable, "-B", str(SCRIPTS / "rebuild_lineage.py"), "--verify-inputs"])
    if BENCH_REVIEW.exists():
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    smoke_event_receipts(bench)
    with tempfile.TemporaryDirectory() as scratch:
        script_paths = (
            sorted(SCRIPTS.glob("*.py"))
            + sorted((SCRIPTS / "lineage_trainers").glob("*.py"))
            + sorted((SCRIPTS / "stage7_wrappers").glob("*.py"))
        )
        for path in script_paths:
            try:
                py_compile.compile(
                    str(path),
                    cfile=str(Path(scratch) / (path.name + "c")),
                    doraise=True,
                )
            except py_compile.PyCompileError as error:
                raise SystemExit(f"compile check failed: {error}")
    tests_dir = EXP / "tests"
    if tests_dir.is_dir():
        run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests_dir), "-v"])
    print(
        "PASS: eval-only six-seed confirmation design — two pre-existing "
        "authenticated composites (count_walk parent, state_track "
        "candidate), sealed fresh seeds 78170/78171/78172/78173/78174/78175, "
        "the sha-pinned prior 78169 INSTALLED_TRANSFER event (never pooled), "
        "byte-identical in-cell provenance copies, the complete in-cell "
        "standalone stage 1-9 lineage package (rebuild_lineage.py "
        "--verify-inputs), the k-seed write-ahead ledger, the frozen paired "
        "replication rule (mean_d > 0 AND wins >= 4, 1e-12 tie guard), and "
        "the preregistered power arithmetic"
    )


def benchmark_stage(resume: bool) -> None:
    bench = load_run_benchmark()
    checkpoints = [
        f"{EXP_REL}/reports/preregistration.md",
        f"{EXP_REL}/reports/benchmark_design_review.md",
        f"{EXP_REL}/{bench.PRIOR_EVENT['summary_copy_relative']}",
        *[
            f"{EXP_REL}/{bench.ARM_PROVENANCE[label]['copy_relative']}"
            for label in MODEL_ORDER
        ],
    ]
    # Sibling originals are verification aids: HEAD-checked only when present.
    for label in MODEL_ORDER:
        sibling = bench.ARM_PROVENANCE[label]["sibling_original"]
        if sibling.is_file():
            checkpoints.append(bench.ARM_PROVENANCE[label]["sibling_relative"])
    if bench.PRIOR_EVENT["summary_sibling"].is_file():
        checkpoints.append(bench.PRIOR_EVENT["summary_sibling_relative"])
    for relative in checkpoints:
        require_pushed_checkpoint(relative)
    require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    command = [
        sys.executable,
        "-B",
        str(SCRIPTS / "run_benchmark.py"),
        "--name",
        FROZEN_NAME,
        "--tier",
        FROZEN_TIER,
        "--think-budget",
        str(FROZEN_THINK_BUDGET),
    ]
    for seed in SEED_ORDER:
        command.extend(("--seed", str(seed)))
    for label in MODEL_ORDER:
        command.extend(("--model", f"{label}={bench.FROZEN_MODEL_PATHS[label]}"))
    if resume:
        command.append("--resume")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("benchmark",))
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a crashed benchmark event after auditing its "
        "preserved receipts (forwarded to run_benchmark.py)",
    )
    args = parser.parse_args()
    if args.smoke:
        if args.resume:
            parser.error("--resume only applies to --stage benchmark")
        smoke()
        return 0
    if args.stage == "benchmark":
        benchmark_stage(args.resume)
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
