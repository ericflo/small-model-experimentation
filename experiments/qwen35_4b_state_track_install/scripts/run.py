#!/usr/bin/env python3
"""Checkpointed harness for the stage-9 state_track installation trial.

Lifecycle 30 — stage 9 of the documented zero-root chain: a DIVERGENT
single-kind installation dose of a NEW transferable skill — STATE-TRACKING
UNDER DECLARATIVE UPDATES — onto the count_walk composite. ONE fresh
rank-32/alpha-64 adapter trains on the FRESH 160-row single-kind
state-tracking curriculum (``data/sft_state_track.jsonl``) from the
count_walk composite parent via the trainer's ``--model-path`` at the fixed
fresh seed 87, merges through the external composite merger, must pass a
two-arm three-screen pooled_k3 retention non-drift gate (fresh seeds
88063/88064/88065), and only a locally promoted candidate may open the
sealed medium aggregate seed 78169 (three arms: base, count_walk,
state_track; frozen two-directional consequence INSTALLED_TRANSFER /
BOUNDED).

Every stage requires clean pushed green main and its staged review verdict:
``--stage train`` needs ``reports/compute_review.md`` carrying
PASS_CONTROL_TRAINING; ``--stage merge`` needs
``reports/local_design_review.md`` carrying PASS_CONTROL_MERGE;
``--stage local`` needs the same file carrying PASS_LOCAL_EVENT;
``--stage benchmark`` needs ``reports/benchmark_design_review.md`` carrying
PASS_BENCHMARK_EVENT plus the committed local promotion receipt. Receipts
are committed between stages; the candidate's three published-hash TODO
pins fail closed while unfilled.
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
SCREEN_SEEDS = (88063, 88064, 88065)
AGGREGATE_SEED = 78169
FROZEN_NAME = "install"
FROZEN_TIER = "medium"
FROZEN_THINK_BUDGET = 1024
ROWS_PER_ARM = 312
ARM = "state_track"
PARENT_EVAL_LABEL = "count_walk"
LOCAL_LABELS = (PARENT_EVAL_LABEL, ARM)
MODEL_ORDER = ("base", PARENT_EVAL_LABEL, ARM)
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
LOCAL_DESIGN = EXP / "data" / "local_design_receipt.json"
LOCAL_REVIEW = EXP / "reports" / "local_design_review.md"
BENCH_REVIEW = EXP / "reports" / "benchmark_design_review.md"
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{SCREEN_SEEDS[0]}.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / f"seed{SCREEN_SEEDS[0]}_promotion.json"
TRAIN_VERDICT = "**Verdict:** `PASS_CONTROL_TRAINING`."
MERGE_VERDICT = "**Verdict:** `PASS_CONTROL_MERGE`."
LOCAL_VERDICT = "**Verdict:** `PASS_LOCAL_EVENT`."
BENCH_VERDICT = "**Verdict:** `PASS_BENCHMARK_EVENT`."
# Training base for the fresh rank-32 adapter: the COUNT_WALK merged
# COMPOSITE (not an adapter; no warm start exists in this cell).
MODEL_PATH = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_count_dont_walk_enumeration"
    / "merged"
    / "count_walk"
)
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_ROOT = ROOT / "large_artifacts" / EXP.name / "merged"
EVENT_DIR = (
    EXP / "runs" / "benchmark"
    / f"{FROZEN_TIER}_tb{FROZEN_THINK_BUDGET}_seed{AGGREGATE_SEED}_{FROZEN_NAME}"
)
LEDGER = EXP / "runs" / "benchmark_events.jsonl"
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
GATEWAY_SHA256 = "53cf6533dbd710eb167503363c39f73dbf7559a0d91f40a00436a3c218a01c17"


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
    # A prerequisite that is not committed at HEAD must refuse with a
    # one-line message, never a raw git traceback.
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative_path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if probe.returncode != 0:
        raise SystemExit(
            f"stage prerequisite is not committed at HEAD: {relative_path}"
        )
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


def smoke_local_receipts() -> None:
    if not LOCAL_RECEIPT.exists():
        if PROMOTION_RECEIPT.exists():
            raise SystemExit("promotion receipt exists without its local receipt")
        return
    local = load_json(LOCAL_RECEIPT)
    rows = local.get("rows")
    if (
        local.get("experiment_id") != EXP.name
        or local.get("screen_seeds") != list(SCREEN_SEEDS)
        or local.get("rows_per_arm") != ROWS_PER_ARM
        or local.get("labels") != list(LOCAL_LABELS)
        or not isinstance(rows, list)
        or len(rows) != ROWS_PER_ARM * len(LOCAL_LABELS)
        or local.get("benchmark_data_read") is not False
        or not LOCAL_DESIGN.is_file()
        or local.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN)
    ):
        raise SystemExit("published local receipt failed smoke authentication")
    raw_artifacts = local.get("raw_artifacts") or {}
    expected_keys = {
        f"{label}_seed{seed}" for label in LOCAL_LABELS for seed in SCREEN_SEEDS
    }
    if set(raw_artifacts) != expected_keys:
        raise SystemExit("published local raw-artifact key set changed")
    for label, artifact in sorted(raw_artifacts.items()):
        for key in ("output", "metadata", "log"):
            expected = artifact.get(f"{key}_sha256")
            if expected is None:
                continue
            path = Path(artifact.get(key, ""))
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published local artifact changed for {label}: {path}")
    if PROMOTION_RECEIPT.exists():
        promotion = load_json(PROMOTION_RECEIPT)
        promoted = promotion.get("promoted")
        if (
            promotion.get("screen_seeds") != list(SCREEN_SEEDS)
            or promotion.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
            or promotion.get("benchmark_data_read") is not False
            or promotion.get("aggregate_seed") != AGGREGATE_SEED
            or promotion.get("aggregate_seed_open") is not (promoted is not None)
            or promoted not in (None, ARM)
        ):
            raise SystemExit(
                "published local promotion receipt failed smoke authentication"
            )


def smoke_benchmark_receipts() -> None:
    """Authenticate any published sealed-event artifacts without running."""
    if not LEDGER.exists():
        if EVENT_DIR.exists():
            raise SystemExit("published event artifacts exist without a ledger")
        return
    rows = [
        json.loads(line)
        for line in LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    opened = {
        "name": FROZEN_NAME,
        "phase": "opened",
        "seed": AGGREGATE_SEED,
        "think_budget": FROZEN_THINK_BUDGET,
        "tier": FROZEN_TIER,
    }
    if not rows or rows[0] != opened:
        raise SystemExit("benchmark ledger row 1 does not match the frozen opened record")
    if len(rows) == 1:
        raise SystemExit(
            "benchmark ledger has a crashed opened event; audit the preserved "
            "receipts and resume before smoke can pass"
        )
    if len(rows) != 2:
        raise SystemExit("published benchmark ledger has unexpected trailing rows")
    closed = rows[1]
    summary_path = EVENT_DIR / "summary.json"
    if not summary_path.is_file():
        raise SystemExit("published benchmark event lacks its sealed summary")
    if closed != {
        "name": FROZEN_NAME,
        "phase": "closed",
        "tier": FROZEN_TIER,
        "think_budget": FROZEN_THINK_BUDGET,
        "seed": AGGREGATE_SEED,
        "summary": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
    }:
        raise SystemExit(
            "published benchmark ledger closed record does not sha-pin the "
            "on-disk summary"
        )
    summary = load_json(summary_path)
    consequence = summary.get("consequence") or {}
    if (
        summary.get("name") != FROZEN_NAME
        or summary.get("tier") != FROZEN_TIER
        or summary.get("think_budget") != FROZEN_THINK_BUDGET
        or summary.get("seed") != AGGREGATE_SEED
        or summary.get("model_order") != list(MODEL_ORDER)
        or set(summary.get("scores", {})) != set(MODEL_ORDER)
        or summary.get("gateway_sha256") != GATEWAY_SHA256
        or summary.get("benchmark_data_read") is not False
        or consequence.get("verdict") not in ("COMPOUNDED", "BOUNDED")
        or consequence.get("no_third_state") is not True
    ):
        raise SystemExit("published benchmark summary failed smoke authentication")
    for label in MODEL_ORDER:
        receipt = EVENT_DIR / f"{label}.json"
        if not receipt.is_file():
            raise SystemExit(f"published gateway receipt is absent for arm {label}")


def smoke() -> None:
    # The frozen-design contracts: the three-slot normalized runner pin, the
    # gateway sha, the copied frozen corpora, and the no-benchmark-reads
    # audit.
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    # Standalone clean-chain gate: authenticate the complete copied lineage
    # package (byte-pinned extended manifest, seven stage datasets + the
    # stage-8 replay pool + the stage-9 state_track curriculum, trainers,
    # merger, wrappers, lifecycle 22's provenance receipts) and the
    # deliberate ABSENCE of any blend root. Fast, no GPU, no writes.
    run([
        sys.executable,
        "-B",
        str(SCRIPTS / "rebuild_lineage.py"),
        "--verify-inputs",
    ])
    if LOCAL_DESIGN.exists():
        run([sys.executable, "-B", str(SCRIPTS / "gen_local_gate.py"), "--check"])
    if COMPUTE_REVIEW.exists():
        require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "second adversarial compute review")
    if LOCAL_REVIEW.exists():
        require_verdict(LOCAL_REVIEW, MERGE_VERDICT, "local adversarial review")
    training_receipt = EXP / "runs" / "training" / f"{ARM}.json"
    if training_receipt.exists():
        sys.path.insert(0, str(SCRIPTS))
        from train_trial import validate_published_arm  # noqa: PLC0415

        validate_published_arm(ARM, require_committed=False)
    merge_receipt = EXP / "runs" / "merges" / f"{ARM}.json"
    if merge_receipt.exists():
        sys.path.insert(0, str(SCRIPTS))
        from merge_trained_arm import validate_published_merge  # noqa: PLC0415

        validate_published_merge(ARM, require_committed=False)
    smoke_local_receipts()
    if BENCH_REVIEW.exists():
        require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    smoke_benchmark_receipts()
    with tempfile.TemporaryDirectory() as scratch:
        sources = (
            sorted(SCRIPTS.glob("*.py"))
            + sorted((SCRIPTS / "lineage_trainers").glob("*.py"))
            + sorted((SCRIPTS / "stage7_wrappers").glob("*.py"))
        )
        for path in sources:
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
        "PASS: stage-9 state_track installation design — the fresh 160-row "
        "single-kind state-tracking curriculum (sha-pinned) onto the "
        "authenticated count_walk parent (fresh seed 87, the chain's frozen "
        "QLoRA recipe), the extended in-cell standalone lineage package "
        "(rebuild_lineage.py --verify-inputs over stages 1-9), the two-arm "
        "three-screen pooled_k3 retention non-drift gate (fresh seeds "
        "88063/88064/88065, two-sided bands on integer screen sums), the "
        "three-slot normalized-hash runner pin, the one-seed write-ahead "
        "ledger with byte-equal crash reconciliation, and the frozen "
        "two-directional INSTALLED_TRANSFER/BOUNDED consequence at sealed "
        "seed 78169"
    )


def train_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/data/local_design_receipt.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/compute_review.md")
    require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "second adversarial compute review")
    run([
        sys.executable,
        "-B",
        str(SCRIPTS / "train_trial.py"),
        "--name",
        ARM,
        "--train",
        str(EXP / "data" / "sft_state_track.jsonl"),
        "--out",
        str(ADAPTER_ROOT / ARM),
        "--model-path",
        str(MODEL_PATH),
        "--epochs",
        "1",
        "--lr",
        "1e-5",
        "--rank",
        "32",
        "--alpha",
        "64",
        "--batch-size",
        "1",
        "--grad-accum",
        "8",
        "--max-length",
        "4096",
        "--w-think",
        "0.2",
        "--w-close",
        "0.2",
        "--seed",
        "87",
    ])


def merge_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/training/{ARM}.json")
    require_pushed_checkpoint(f"{EXP_REL}/data/local_design_receipt.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/local_design_review.md")
    require_verdict(LOCAL_REVIEW, MERGE_VERDICT, "local adversarial review")
    sys.path.insert(0, str(SCRIPTS))
    from merge_trained_arm import validate_published_merge  # noqa: PLC0415

    if (EXP / "runs" / "merges" / f"{ARM}.json").exists():
        validate_published_merge(ARM, require_committed=False)
        print(f"[run] {ARM} merge already published and authenticated; skipping")
        return
    run([
        sys.executable,
        "-B",
        str(SCRIPTS / "merge_trained_arm.py"),
        "--name",
        ARM,
        "--adapter",
        str(ADAPTER_ROOT / ARM),
        "--out",
        str(MERGED_ROOT / ARM),
    ])


def local_stage() -> None:
    for relative in (
        "data/local_design_receipt.json",
        "reports/local_design_review.md",
        f"runs/merges/{ARM}.json",
    ):
        require_pushed_checkpoint(f"{EXP_REL}/{relative}")
    require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    run([sys.executable, "-B", str(SCRIPTS / "eval_local_vllm.py")])
    if not LOCAL_RECEIPT.is_file():
        raise SystemExit("local evaluation did not produce the frozen local receipt")
    if PROMOTION_RECEIPT.exists():
        run([sys.executable, "-B", str(SCRIPTS / "check_local.py"), str(LOCAL_RECEIPT)])
    else:
        run([
            sys.executable,
            "-B",
            str(SCRIPTS / "check_local.py"),
            str(LOCAL_RECEIPT),
            "--out",
            str(PROMOTION_RECEIPT),
        ])


def benchmark_stage(resume: bool) -> None:
    require_pushed_checkpoint(
        f"{EXP_REL}/runs/local/seed{SCREEN_SEEDS[0]}_promotion.json"
    )
    require_pushed_checkpoint(f"{EXP_REL}/reports/benchmark_design_review.md")
    require_verdict(BENCH_REVIEW, BENCH_VERDICT, "benchmark design review")
    promotion = load_json(PROMOTION_RECEIPT)
    if promotion.get("promoted") != ARM:
        raise SystemExit(
            "no locally promoted candidate; the aggregate seed stays sealed"
        )
    if not (MERGED_ROOT / ARM / "merge_receipt.json").is_file():
        raise SystemExit(f"promoted candidate composite is incomplete: {ARM}")
    base_model = (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    )
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
        str(AGGREGATE_SEED),
        "--candidate",
        ARM,
        "--model",
        f"base={base_model}",
        "--model",
        f"{PARENT_EVAL_LABEL}={MODEL_PATH}",
        "--model",
        f"{ARM}={MERGED_ROOT / ARM}",
    ]
    # The audit-before-resume interlock lives in run_benchmark.py; only an
    # explicit operator --resume may be forwarded, never an automatic one.
    if resume:
        command.append("--resume")
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument(
        "--stage",
        choices=("train", "merge", "local", "benchmark"),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a crashed benchmark event after auditing its "
        "preserved receipts (forwarded to run_benchmark.py)",
    )
    args = parser.parse_args()
    if args.resume and args.stage != "benchmark":
        parser.error("--resume only applies to --stage benchmark")
    if args.smoke:
        smoke()
        return 0
    if args.stage == "train":
        train_stage()
        return 0
    if args.stage == "merge":
        merge_stage()
        return 0
    if args.stage == "local":
        local_stage()
        return 0
    if args.stage == "benchmark":
        benchmark_stage(args.resume)
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
