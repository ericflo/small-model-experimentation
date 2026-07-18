#!/usr/bin/env python3
"""Checkpointed harness for the lifecycle-33 self-repair installation trial.

Lifecycle 33 — the SECOND curriculum bet of the cognitive-core coding program,
and cognitive-core coding bet #2. Bet #1 (execution-tracing) came back NULL: it
reshuffled which coding tasks the 4B solves without raising the count on
HumanEval, MBPP, or the agentic duet-eval (8/35 -> 8/35). This cell installs the
CHECK-AND-REPAIR loop directly: a fresh rank-32/alpha-64 LoRA trains on 504
buggy->fixed debugging episodes (bug injected by AST mutation, so the fix is
KNOWN; every buggy/corrected pair execution-verified against concrete tests —
the signal is SELF-GENERATED and EXECUTION-VERIFIED, respecting provenance),
merges onto base, and is measured for TRANSFER + RETENTION on the shared
HumanEval + MBPP fitness harness under a frozen, TIGHTENED two-directional
consequence (INSTALLED_CODING requires a >= 3-problem gain).

This cell trains from BASE in ONE stage; the standalone lineage package is the
in-cell base_reserialized provenance copy + the self-repair curriculum + the
fixed-seed recipe + the vendored trainer/merger.

Stages (fail-closed; each needs clean pushed green main + its staged review):
  --smoke                 model-free design gate (no GPU, no writes)
  --stage train           needs reports/compute_review.md PASS_CONTROL_TRAINING
  --stage merge           needs reports/merge_review.md   PASS_CONTROL_MERGE
  --stage measure         needs reports/measure_review.md PASS_MEASURE
"""

from __future__ import annotations

import argparse
import hashlib
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"
EXP_REL = f"experiments/{EXP.name}"

ARM = "self_repair"
CORPUS = EXP / "data" / "sft_self_repair.jsonl"
RECEIPT = EXP / "data" / "curriculum_receipt.json"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
MERGE_REVIEW = EXP / "reports" / "merge_review.md"
MEASURE_REVIEW = EXP / "reports" / "measure_review.md"
TRAIN_VERDICT = "**Verdict:** `PASS_CONTROL_TRAINING`."
MERGE_VERDICT = "**Verdict:** `PASS_CONTROL_MERGE`."
MEASURE_VERDICT = "**Verdict:** `PASS_MEASURE`."

BASE_MODEL = ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum" / "merged" / "base_reserialized"
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_ROOT = ROOT / "large_artifacts" / EXP.name / "merged"
TRAINING_RECEIPT = EXP / "runs" / "training" / f"{ARM}.json"
MERGE_RECEIPT = EXP / "runs" / "merges" / f"{ARM}.json"


def output(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_pushed_checkpoint(relative_path: str) -> None:
    status = output(["git", "status", "--short"])
    branch = output(["git", "branch", "--show-current"])
    head = output(["git", "rev-parse", "HEAD"])
    origin = output(["git", "rev-parse", "origin/main"])
    if status or branch != "main" or head != origin:
        raise SystemExit("stage requires a clean pushed main checkpoint")
    probe = subprocess.run(
        ["git", "cat-file", "-e", f"HEAD:{relative_path}"], cwd=ROOT, check=False, capture_output=True
    )
    if probe.returncode != 0:
        raise SystemExit(f"stage prerequisite is not committed at HEAD: {relative_path}")
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout
    path = ROOT / relative_path
    if not path.is_file() or path.read_bytes() != committed:
        raise SystemExit(f"stage prerequisite differs from HEAD: {relative_path}")


def require_verdict(path: Path, verdict: str, description: str) -> None:
    if not path.is_file() or verdict not in path.read_text(encoding="utf-8"):
        raise SystemExit(f"{description} has not been authorized: {path}")


def smoke() -> None:
    # 1. compile every script.
    with tempfile.TemporaryDirectory() as scratch:
        for path in sorted(SCRIPTS.glob("*.py")):
            try:
                py_compile.compile(str(path), cfile=str(Path(scratch) / (path.name + "c")), doraise=True)
            except py_compile.PyCompileError as error:
                raise SystemExit(f"compile check failed: {error}")

    # 2. the committed base-provenance copy pins exactly the base composite.
    sys.path.insert(0, str(SCRIPTS))
    from train_trial import (  # noqa: PLC0415
        BASE_PROVENANCE_COPY,
        MODEL_PATH_RECEIPT_SHA256,
        check_base_provenance,
    )

    if not BASE_PROVENANCE_COPY.is_file() or sha256_file(BASE_PROVENANCE_COPY) != MODEL_PATH_RECEIPT_SHA256:
        raise SystemExit("in-cell base provenance copy is absent or changed")
    check_base_provenance()

    # 3. the committed corpus independently re-executes + matches its receipt sha.
    run([sys.executable, "-B", str(SCRIPTS / "gen_self_repair_curriculum.py"), "--verify-corpus"])

    # 4. published downstream receipts authenticate without running (if present).
    if COMPUTE_REVIEW.exists():
        require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "adversarial compute review")
    if TRAINING_RECEIPT.exists():
        from train_trial import validate_published_arm  # noqa: PLC0415

        validate_published_arm(ARM, require_committed=False)
    if MERGE_REVIEW.exists():
        require_verdict(MERGE_REVIEW, MERGE_VERDICT, "adversarial merge review")
    if MEASURE_REVIEW.exists():
        require_verdict(MEASURE_REVIEW, MEASURE_VERDICT, "adversarial measure review")

    # 5. unit tests.
    tests_dir = EXP / "tests"
    if tests_dir.is_dir():
        run([sys.executable, "-B", "-m", "unittest", "discover", "-s", str(tests_dir), "-v"])

    print(
        "PASS: lifecycle-33 self-repair installation design — the 504-row "
        "self-repair debugging curriculum (bug injected by AST mutation so the "
        "fix is known; triple-verified by real execution: correct passes all "
        "tests, buggy fails >=1 with a wrong value and crashes on none, "
        "correction differs; contamination-clean by whole-word banned-name "
        "audit and code-only distinctive n-gram overlap), trained from the "
        "fail-closed authenticated base_reserialized composite (fresh seed "
        "91331, r32/a64, 1 epoch, 63 optimizer steps), merged via the vendored "
        "external merger, and scored for TRANSFER + RETENTION on the shared "
        "HumanEval+MBPP fitness harness under the frozen, TIGHTENED "
        "INSTALLED_CODING (>=3 problems) / RETENTION_FAIL / NULL consequence"
    )


def train_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/data/sft_self_repair.jsonl")
    require_pushed_checkpoint(f"{EXP_REL}/reports/compute_review.md")
    require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "adversarial compute review")
    run([
        sys.executable, "-B", str(SCRIPTS / "train_trial.py"),
        "--name", ARM,
        "--train", str(CORPUS),
        "--out", str(ADAPTER_ROOT / ARM),
        "--model-path", str(BASE_MODEL),
        "--epochs", "1",
        "--lr", "1e-5",
        "--rank", "32",
        "--alpha", "64",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--max-length", "4096",
        "--w-think", "0.2",
        "--w-close", "0.2",
        "--seed", "91331",
    ])


def merge_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/training/{ARM}.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/merge_review.md")
    require_verdict(MERGE_REVIEW, MERGE_VERDICT, "adversarial merge review")
    if (MERGED_ROOT / ARM / "merge_receipt.json").is_file():
        print(f"[run] {ARM} merge already published; skipping")
        return
    run([
        sys.executable, "-B", str(SCRIPTS / "merge_adapter.py"),
        "--adapter", str(ADAPTER_ROOT / ARM),
        "--out", str(MERGED_ROOT / ARM),
        "--base-model", str(BASE_MODEL),
    ])


def measure_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/merges/{ARM}.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/measure_review.md")
    require_verdict(MEASURE_REVIEW, MEASURE_VERDICT, "adversarial measure review")
    if not (MERGED_ROOT / ARM / "merge_receipt.json").is_file():
        raise SystemExit(f"self_repair composite is incomplete: {MERGED_ROOT / ARM}")
    run([sys.executable, "-B", str(SCRIPTS / "measure_transfer.py"), "--run"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("train", "merge", "measure"))
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "train":
        train_stage()
        return 0
    if args.stage == "merge":
        merge_stage()
        return 0
    if args.stage == "measure":
        measure_stage()
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
