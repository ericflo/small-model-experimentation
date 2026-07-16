#!/usr/bin/env python3
"""Checkpointed harness for the medium budget-probe measurement.

Eval-only MEASUREMENT INTAKE: the same four inherited, externally
published composites as the seed-78150 medium event are measured once at
tier medium, think budget 8192 (the probe lever), on one sealed fresh
seed (78152). This experiment trains nothing, merges nothing, and
promotes nothing; there is no local gate and no adapter. The only stage
is ``benchmark`` and the preregistered measurement readout is the
terminal artifact.

``--smoke`` verifies every pin (external merge receipts, the tb1024
contrast-source summary, trusted gateway, design receipt + code hashes
via ``gen_design_receipt.py --check``), authenticates any published event
artifacts (including requiring the terminal readout when a summary
exists), compiles every script, and runs the unit tests. It runs no
model event and does NOT require the benchmark design review to exist.

``--stage benchmark`` requires the committed-at-HEAD design receipt,
preregistration, contrast-source summary, the adversarial benchmark
design review carrying the literal verdict line, and clean pushed green
main, then runs the single frozen event. ``--resume`` is forwarded to
run_benchmark.py only when explicitly passed by the operator.
"""

from __future__ import annotations

import argparse
import hashlib
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
FROZEN_NAME = "measurement"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 8192
FROZEN_SEED = 78152
MODEL_ORDER = ("base", "designed_fresh", "replay_repeat", "hygiene_explore")
# Every arm is an inherited, externally published composite: the committed
# external merge receipts are smoke-verified prerequisites (fail-closed
# hash pins). Base's reserialization receipt lives inside the composite
# and is pinned by the design receipt instead.
EXTERNAL_MERGE_RECEIPTS = {
    "experiments/qwen35_4b_universal_fresh_surface_budget_commit_target_match/runs/merges/designed_fresh.json": (
        "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2"
    ),
    "experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/runs/merges/replay_repeat.json": (
        "22384463d7825ec2a0b95faeaeb273264d7331f4584f8b7e9e58a60545398af1"
    ),
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/hygiene_explore.json": (
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
    ),
}
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
TB1024_SUMMARY = (
    "experiments/qwen35_4b_universal_medium_tier_measurement"
    "/runs/benchmark/medium_tb1024_seed78150_measurement/summary.json"
)
TB1024_SUMMARY_SHA256 = (
    "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43"
)
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{FROZEN_SEED}_{FROZEN_NAME}"
)
SUMMARY = EVENT_DIR / "summary.json"
READOUT = EVENT_DIR / "measurement_readout.json"
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


def smoke_event_receipts() -> None:
    """Authenticate any published event artifacts without running anything."""
    if not LEDGER.exists() and not SUMMARY.exists():
        return
    if not LEDGER.exists() or not SUMMARY.exists():
        raise SystemExit("published benchmark event is incomplete (ledger/summary)")
    rows = [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if rows != [
        {
            "name": FROZEN_NAME,
            "phase": "opened",
            "seed": FROZEN_SEED,
            "think_budget": FROZEN_THINK_BUDGET,
            "tier": FROZEN_TIER,
        },
        {
            "name": FROZEN_NAME,
            "phase": "closed",
            "tier": FROZEN_TIER,
            "think_budget": FROZEN_THINK_BUDGET,
            "seed": FROZEN_SEED,
            "summary": str(SUMMARY),
            "summary_sha256": sha256_file(SUMMARY),
        },
    ]:
        raise SystemExit("published benchmark ledger failed smoke authentication")
    summary = load_json(SUMMARY)
    if (
        summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != FROZEN_SEED
        or summary.get("model_order") != list(MODEL_ORDER)
        or set(summary.get("scores", {})) != set(MODEL_ORDER)
        or set(summary.get("budget", {})) != set(MODEL_ORDER)
        or summary.get("promoted") is not None
        or summary.get("benchmark_data_read") is not False
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("tb1024_reference_summary_sha256") != TB1024_SUMMARY_SHA256
        or not DESIGN_RECEIPT.is_file()
        or summary.get("design_receipt_sha256") != sha256_file(DESIGN_RECEIPT)
    ):
        raise SystemExit("published benchmark summary failed smoke authentication")
    for label in MODEL_ORDER:
        if not (EVENT_DIR / f"{label}.json").is_file():
            raise SystemExit(f"published gateway receipt is absent for {label}")
    if not READOUT.is_file():
        raise SystemExit(
            "published benchmark event lacks its terminal measurement readout"
        )
    # Recomputes the readout byte-identically from the receipts and the
    # pinned tb1024 contrast source (verify mode; writes nothing).
    run([sys.executable, "-B", str(SCRIPTS / "check_benchmark.py")])


def smoke() -> None:
    for relative, expected in sorted(EXTERNAL_MERGE_RECEIPTS.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published external merge receipt changed: {path}")
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise SystemExit("trusted gateway is absent or changed")
    tb1024 = ROOT / TB1024_SUMMARY
    if not tb1024.is_file() or sha256_file(tb1024) != TB1024_SUMMARY_SHA256:
        raise SystemExit("pinned tb1024 contrast-source summary is absent or changed")
    if DESIGN_RECEIPT.exists():
        # Re-verifies every design pin (seed-freshness audit, receipts,
        # contrast source, code hashes) and the receipt bytes.
        run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
        if BENCH_REVIEW.exists():
            require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    smoke_event_receipts()
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")):
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
        "PASS: eval-only budget-probe design, four inherited-composite pins, "
        "sealed fresh seed 78152, pinned tb1024 contrast source, and the "
        "frozen one-event medium/tb8192 measurement contracts"
    )


def benchmark_stage(resume: bool) -> None:
    for relative in (
        f"{EXP_REL}/data/design_receipt.json",
        f"{EXP_REL}/reports/preregistration.md",
        f"{EXP_REL}/reports/benchmark_design_review.md",
        TB1024_SUMMARY,
        *sorted(EXTERNAL_MERGE_RECEIPTS),
    ):
        require_pushed_checkpoint(relative)
    require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
    sys.path.insert(0, str(SCRIPTS))
    from run_benchmark import FROZEN_MODEL_PATHS  # noqa: PLC0415

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
        "--seed",
        str(FROZEN_SEED),
    ]
    for label in MODEL_ORDER:
        command.extend(("--model", f"{label}={FROZEN_MODEL_PATHS[label]}"))
    # The audit-before-resume interlock lives in run_benchmark.py; only an
    # explicit operator --resume may be forwarded, never an automatic one.
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
