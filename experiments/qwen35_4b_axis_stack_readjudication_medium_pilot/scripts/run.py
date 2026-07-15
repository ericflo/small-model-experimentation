#!/usr/bin/env python3
"""Checkpointed harness for the training-free axis-stack re-adjudication (medium pilot)."""

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
LOCAL_SEED = 88016
AGGREGATE_SEED = 78146
LOCAL_ROWS = 144
PARENT_EVAL_LABEL = "replay_parent"
CANDIDATE_ARMS = ("axis_on_replay",)
LOCAL_LABELS = (PARENT_EVAL_LABEL, "replay_squared", "axis_on_replay")
# Every arm is an inherited, externally published composite: this experiment
# trains nothing and merges nothing. Frozen source corpora (copied
# byte-identically for the gate's overlap receipt) and the committed external
# merge receipts are the smoke-verified prerequisites.
CORPUS_HASHES = {
    "sft_blend.jsonl": "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
    "sft_axis160.jsonl": "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
}
EXTERNAL_MERGE_RECEIPTS = (
    "experiments/qwen35_4b_goal_gap_axis_curriculum_target_match/runs/merges/replay_repeat.json",
    "experiments/qwen35_4b_axis_replay_stack_medium_target_match/runs/merges/replay_squared.json",
    "experiments/qwen35_4b_axis_replay_stack_medium_target_match/runs/merges/axis_on_replay.json",
)
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
LOCAL_DESIGN = EXP / "data" / "local_design_receipt.json"
LOCAL_REVIEW = EXP / "reports" / "local_design_review.md"
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}_promotion.json"
LOCAL_VERDICT = "**Verdict:** `PASS_LOCAL_EVENT`."
MERGED_ROOT = ROOT / "large_artifacts" / "qwen35_4b_axis_replay_stack_medium_target_match" / "merged"


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


def promoted_candidate(promotion: dict) -> str | None:
    # Fail closed: only the promotion receipt's explicit winner counts.
    return promotion.get("promoted")


def smoke_local_receipts() -> None:
    if not LOCAL_RECEIPT.exists():
        return
    local = load_json(LOCAL_RECEIPT)
    rows = local.get("rows")
    if (
        local.get("experiment_id") != EXP.name
        or local.get("seed") != LOCAL_SEED
        or local.get("rows_per_arm") != LOCAL_ROWS
        or set(local.get("labels") or []) != set(LOCAL_LABELS)
        or not isinstance(rows, list)
        or len(rows) != LOCAL_ROWS * len(LOCAL_LABELS)
        or local.get("benchmark_data_read") is not False
        or not LOCAL_DESIGN.is_file()
        or local.get("design_receipt_sha256") != sha256_file(LOCAL_DESIGN)
    ):
        raise SystemExit("published local receipt failed smoke authentication")
    for label, artifact in sorted((local.get("raw_artifacts") or {}).items()):
        for key in ("output", "metadata", "log"):
            expected = artifact.get(f"{key}_sha256")
            if expected is None:
                continue
            path = Path(artifact.get(key, ""))
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published local artifact changed for {label}: {path}")
    if PROMOTION_RECEIPT.exists():
        promotion = load_json(PROMOTION_RECEIPT)
        promoted = promoted_candidate(promotion)
        if (
            promotion.get("seed") != LOCAL_SEED
            or promotion.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
            or promotion.get("benchmark_data_read") is not False
            or promoted not in (None, *CANDIDATE_ARMS)
            or (promoted is not None and promotion.get("outcome") != "PROMOTED")
        ):
            raise SystemExit("published local promotion receipt failed smoke authentication")


def smoke() -> None:
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    for name, expected in CORPUS_HASHES.items():
        path = EXP / "data" / name
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published corpus artifact changed: {path}")
    if LOCAL_DESIGN.exists():
        # Recomputes all three inherited composite tree manifests against the
        # frozen pins in addition to regenerating the gate byte-identically.
        run([sys.executable, "-B", str(SCRIPTS / "gen_local_gate.py"), "--check"])
        if LOCAL_REVIEW.exists():
            require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    smoke_local_receipts()
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
        "PASS: training-free re-adjudication design, inherited-composite pins, "
        "and frozen local-gate and conditional-pilot contracts"
    )


def local_stage() -> None:
    for relative in (
        f"{EXP_REL}/data/design_receipt.json",
        f"{EXP_REL}/data/local_design_receipt.json",
        f"{EXP_REL}/reports/local_design_review.md",
        *EXTERNAL_MERGE_RECEIPTS,
    ):
        require_pushed_checkpoint(relative)
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


def benchmark_stage() -> None:
    require_pushed_checkpoint(f"{EXP_REL}/runs/local/seed{LOCAL_SEED}_promotion.json")
    promotion = load_json(PROMOTION_RECEIPT)
    promoted = promoted_candidate(promotion)
    if promoted not in CANDIDATE_ARMS:
        raise SystemExit(
            "no locally promoted candidate; the aggregate seed stays sealed"
        )
    if not (MERGED_ROOT / promoted / "merge_receipt.json").is_file():
        raise SystemExit(f"promoted candidate composite is incomplete: {promoted}")
    base_model = (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    )
    parent_model = (
        ROOT / "large_artifacts"
        / "qwen35_4b_goal_gap_axis_curriculum_target_match"
        / "merged" / "replay_repeat"
    )
    run([
        sys.executable,
        "-B",
        str(SCRIPTS / "run_benchmark.py"),
        "--name",
        "pilot",
        "--tier",
        "medium",
        "--think-budget",
        "1024",
        "--seed",
        str(AGGREGATE_SEED),
        "--candidate",
        promoted,
        "--model",
        f"base={base_model}",
        "--model",
        f"{PARENT_EVAL_LABEL}={parent_model}",
        "--model",
        f"replay_squared={MERGED_ROOT / 'replay_squared'}",
        "--model",
        f"{promoted}={MERGED_ROOT / promoted}",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument(
        "--stage",
        choices=(
            "local",
            "benchmark",
        ),
    )
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "local":
        local_stage()
        return 0
    if args.stage == "benchmark":
        benchmark_stage()
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
