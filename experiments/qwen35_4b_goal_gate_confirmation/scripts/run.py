#!/usr/bin/env python3
"""Checkpointed harness for the three-seed goal-gate confirmation measurement.

Eval-only MEASUREMENT INTAKE: the recorded seed-78154 goal-gate pass
(hygiene_explore strictly over base on ALL TEN public families) is
replicated on three sealed fresh seeds (78155, 78156, 78157) at tier
medium, think budget 1024, two arms per seed in the frozen order base
then hygiene_explore, seed-major. This experiment trains nothing, merges
nothing, and promotes nothing; there is no local gate and no adapter.
The only stage is ``benchmark`` and the preregistered confirmation
readout is the terminal artifact.

``--smoke`` verifies every pin (the committed hygiene_explore merge
receipt, the committed discovery-seed summary, trusted gateway, design
receipt + code hashes via ``gen_design_receipt.py --check``),
authenticates any published per-seed event artifacts against the k-seed
ledger (including requiring the terminal confirmation readout once all
three seeds are closed), authenticates the standalone lineage package
via ``rebuild_lineage.py --verify-inputs`` (datasets, trainers, merger,
vendored root adapter, manifest — no GPU), compiles every script, and
runs the unit tests. It runs no model event and does NOT require the
benchmark design review to exist.

``--stage benchmark`` requires the committed-at-HEAD design receipt,
preregistration, discovery-seed summary, hygiene_explore merge receipt,
the adversarial benchmark design review carrying the literal verdict
line, and clean pushed green main, then runs the frozen three-seed
event. ``--resume`` is forwarded to run_benchmark.py only when
explicitly passed by the operator.
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
FROZEN_NAME = "confirmation"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
SEED_ORDER = (78155, 78156, 78157)
MODEL_ORDER = ("base", "hygiene_explore")
# The treated arm is an inherited, externally published composite: its
# committed external merge receipt is a smoke-verified prerequisite
# (fail-closed hash pin). Base's reserialization receipt lives inside the
# composite and is pinned by the design receipt instead.
EXTERNAL_MERGE_RECEIPTS = {
    "experiments/qwen35_4b_hygiene_explore_destack_medium/runs/merges/hygiene_explore.json": (
        "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a"
    ),
}
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"
DISCOVERY_SUMMARY = (
    "experiments/qwen35_4b_statechain_only_dose"
    "/runs/benchmark/medium_tb1024_seed78154_pilot/summary.json"
)
DISCOVERY_SUMMARY_SHA256 = (
    "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa"
)
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
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


def authenticate_closed_seed(seed: int, closed: dict) -> None:
    """One closed seed: summary AND both receipts must match the ledger pins.

    The closed record sha-pins every verdict input at close time; smoke
    recomputes each file's sha256 from disk and requires exact record
    equality — a swapped receipt or summary fails here, not just at the
    readout.
    """
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
        or summary.get("discovery_summary_sha256") != DISCOVERY_SUMMARY_SHA256
        or not DESIGN_RECEIPT.is_file()
        or summary.get("design_receipt_sha256") != sha256_file(DESIGN_RECEIPT)
    ):
        raise SystemExit(
            f"published benchmark summary failed smoke authentication at seed {seed}"
        )


def smoke_event_receipts() -> None:
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
        authenticate_closed_seed(seed, rows[index])
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
        # Recomputes the readout byte-identically from the six receipts and
        # the pinned discovery summary (verify mode; writes nothing).
        run([sys.executable, "-B", str(SCRIPTS / "check_benchmark.py")])
    elif READOUT.exists():
        raise SystemExit(
            "confirmation readout exists before all three seeds closed"
        )


def smoke() -> None:
    for relative, expected in sorted(EXTERNAL_MERGE_RECEIPTS.items()):
        path = ROOT / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published external merge receipt changed: {path}")
    if not GATEWAY.is_file() or sha256_file(GATEWAY) != GATEWAY_SHA256:
        raise SystemExit("trusted gateway is absent or changed")
    discovery = ROOT / DISCOVERY_SUMMARY
    if not discovery.is_file() or sha256_file(discovery) != DISCOVERY_SUMMARY_SHA256:
        raise SystemExit("pinned discovery-seed summary is absent or changed")
    if DESIGN_RECEIPT.exists():
        # Re-verifies every design pin (the three seed-freshness audits,
        # receipts, discovery source, code hashes) and the receipt bytes.
        run([sys.executable, "-B", str(SCRIPTS / "gen_design_receipt.py"), "--check"])
        if BENCH_REVIEW.exists():
            require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    smoke_event_receipts()
    # Standalone-reproducibility gate: authenticate the lineage package
    # (manifest schema and warm-start chaining, the six copied stage
    # datasets, the trainer and merger copies, the vendored frozen root
    # adapter) without touching any model. Fast, no GPU, writes nothing.
    run([sys.executable, "-B", str(SCRIPTS / "rebuild_lineage.py"), "--verify-inputs"])
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")) + sorted(
            (SCRIPTS / "lineage_trainers").glob("*.py")
        ):
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
        "PASS: eval-only three-seed confirmation design, two inherited-"
        "composite pins, sealed fresh seeds 78155/78156/78157, pinned "
        "discovery-seed summary, k-seed write-ahead ledger, and the frozen "
        "medium/tb1024 confirmation contracts"
    )


def benchmark_stage(resume: bool) -> None:
    for relative in (
        f"{EXP_REL}/data/design_receipt.json",
        f"{EXP_REL}/reports/preregistration.md",
        f"{EXP_REL}/reports/benchmark_design_review.md",
        DISCOVERY_SUMMARY,
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
    ]
    for seed in SEED_ORDER:
        command.extend(("--seed", str(seed)))
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
