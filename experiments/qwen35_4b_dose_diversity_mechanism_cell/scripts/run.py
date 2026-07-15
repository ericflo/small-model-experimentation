#!/usr/bin/env python3
"""Checkpointed harness for the dose-diversity mechanism cell.

Mechanism study: ONE new training arm (axis160_direct), then a four-arm local
gate that reads it against three already-published composites. There is NO
benchmark stage and NO aggregate seed; the local mechanism receipt is the
terminal artifact.
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
LOCAL_SEED = 88020
ROWS_PER_ARM = 1520
LOCAL_ROWS = 144
ARMS = ("axis160_direct",)
CANDIDATE_ARMS = ("axis160_direct",)
PARENT_EVAL_LABEL = "clean_parent"
LOCAL_LABELS = (
    PARENT_EVAL_LABEL,
    "hygiene_explore_direct",
    "replay_clean",
    "axis160_direct",
)
TRAIN_STAGE_ARMS = {
    "train-candidate": "axis160_direct",
}
MATCH_AXES = ("forward", "nonzero_target", "absolute_loss_mass_x5")
CORPUS_HASHES = {
    "sft_blend.jsonl": "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
    "sft_axis160.jsonl": "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e",
    "corpus_manifest.json": "b029c71389ec98314bf5ee8c7e92c828ac996899beaeeb0e6cd469c6ed9eec1f",
}
VERDICTS = ("SUPPORTED", "REFUTED_INTRINSIC", "SCREEN_FORTUNE_SUSPECT")
TOKEN_RECEIPT = EXP / "data" / "stream_token_receipt.json"
STREAM_MANIFEST = EXP / "data" / "stream_manifest.json"
SOURCE_TOKENS = EXP / "data" / "source_token_lengths.json"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
LOCAL_DESIGN = EXP / "data" / "local_design_receipt.json"
LOCAL_REVIEW = EXP / "reports" / "local_design_review.md"
LOCAL_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}.json"
MECHANISM_RECEIPT = EXP / "runs" / "local" / f"seed{LOCAL_SEED}_mechanism.json"
TRAIN_VERDICT = "**Verdict:** `PASS_CONTROL_TRAINING`."
LOCAL_VERDICT = "**Verdict:** `PASS_LOCAL_EVENT`."
PARENT_ADAPTER = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
    / "adapters"
    / "designed_fresh"
)
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MERGED_ROOT = ROOT / "large_artifacts" / EXP.name / "merged"


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


def smoke_token_receipt() -> None:
    receipt = load_json(TOKEN_RECEIPT)
    files = receipt.get("files") or {}
    if set(files) != set(ARMS):
        raise SystemExit("published exact-exposure receipt arm set changed")
    arm = files[ARMS[0]]
    spans = arm.get("spans_per_epoch") or {}
    exposure = tuple(spans.get(axis) for axis in MATCH_AXES)
    if (
        receipt.get("rows_per_arm") != ROWS_PER_ARM
        or receipt.get("skipped_rows") != 0
        or receipt.get("training_authorized") is not False
        or (arm.get("rows"), arm.get("encoded_rows"), arm.get("skipped_rows"))
        != (ROWS_PER_ARM, ROWS_PER_ARM, 0)
        or any(value is None for value in exposure)
        or exposure
        != (
            receipt.get("forward_tokens_per_arm"),
            receipt.get("nonzero_target_tokens_per_arm"),
            receipt.get("absolute_loss_mass_x5_per_arm"),
        )
        or (receipt.get("notional_control_block") or {}).get(
            "equals_candidate_variable_block_on_match_axes"
        )
        is not True
    ):
        raise SystemExit("published exact-exposure receipt failed smoke authentication")


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
    if MECHANISM_RECEIPT.exists():
        mechanism = load_json(MECHANISM_RECEIPT)
        readings = mechanism.get("readings") or {}
        if (
            mechanism.get("seed") != LOCAL_SEED
            or mechanism.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
            or mechanism.get("benchmark_data_read") is not False
            or mechanism.get("aggregate_seed") is not None
            or mechanism.get("aggregate_seed_open") is not False
            or mechanism.get("promoted") is not None
            or mechanism.get("diversity_mechanism") not in VERDICTS
            or readings.get("diversity_mechanism") != mechanism.get("diversity_mechanism")
            or not isinstance(readings.get("retention_delta_axis160"), int)
            or not isinstance(readings.get("retention_delta_hygexp"), int)
        ):
            raise SystemExit("published local mechanism receipt failed smoke authentication")


def smoke() -> None:
    run([sys.executable, "-B", str(SCRIPTS / "build_corpus.py"), "--check"])
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    for name, expected in CORPUS_HASHES.items():
        if expected is None:
            raise SystemExit(f"corpus artifact pin is unfilled (TODO-PIN): {name}")
        path = EXP / "data" / name
        if not path.is_file() or sha256_file(path) != expected:
            raise SystemExit(f"published corpus artifact changed: {path}")
    if SOURCE_TOKENS.exists():
        run([sys.executable, "-B", str(SCRIPTS / "measure_source_tokens.py"), "--check"])
    if STREAM_MANIFEST.exists():
        run([sys.executable, "-B", str(SCRIPTS / "materialize_streams.py"), "--check"])
    if TOKEN_RECEIPT.exists():
        run([sys.executable, "-B", str(SCRIPTS / "validate_streams.py"), "--check"])
        smoke_token_receipt()
    if COMPUTE_REVIEW.exists():
        require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "second adversarial compute review")
    training_dir = EXP / "runs" / "training"
    if any((training_dir / f"{arm}.json").exists() for arm in ARMS):
        sys.path.insert(0, str(SCRIPTS))
        from train_trial import validate_published_arm  # noqa: PLC0415

        for arm in ARMS:
            if (training_dir / f"{arm}.json").exists():
                validate_published_arm(arm, require_committed=False)
    if LOCAL_DESIGN.exists():
        run([sys.executable, "-B", str(SCRIPTS / "gen_local_gate.py"), "--check"])
        if LOCAL_REVIEW.exists():
            require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
        merge_dir = EXP / "runs" / "merges"
        if any((merge_dir / f"{arm}.json").exists() for arm in ARMS):
            sys.path.insert(0, str(SCRIPTS))
            from merge_trained_arm import validate_published_merge  # noqa: PLC0415

            for arm in ARMS:
                if (merge_dir / f"{arm}.json").exists():
                    validate_published_merge(arm, require_committed=False)
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
        "PASS: model-free construction, single exact-exposure stream, "
        "single-arm training, and frozen four-composite mechanism-gate contracts"
    )


def train_stage(arm: str) -> None:
    require_pushed_checkpoint(f"{EXP_REL}/data/stream_token_receipt.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/compute_review.md")
    require_verdict(COMPUTE_REVIEW, TRAIN_VERDICT, "second adversarial compute review")
    run([
        sys.executable,
        "-B",
        str(SCRIPTS / "train_trial.py"),
        "--name",
        arm,
        "--train",
        str(EXP / "data" / f"{arm}.jsonl"),
        "--token-receipt",
        str(TOKEN_RECEIPT),
        "--out",
        str(ADAPTER_ROOT / arm),
        "--warm-start",
        str(PARENT_ADAPTER),
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
        "57",
    ])


def merge_candidate_stage() -> None:
    for arm in ARMS:
        require_pushed_checkpoint(f"{EXP_REL}/runs/training/{arm}.json")
    require_pushed_checkpoint(f"{EXP_REL}/data/local_design_receipt.json")
    require_pushed_checkpoint(f"{EXP_REL}/reports/local_design_review.md")
    require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    sys.path.insert(0, str(SCRIPTS))
    from merge_trained_arm import validate_published_merge  # noqa: PLC0415

    for arm in ARMS:
        if (EXP / "runs" / "merges" / f"{arm}.json").exists():
            validate_published_merge(arm, require_committed=False)
            print(f"[run] {arm} merge already published and authenticated; skipping")
            continue
        run([
            sys.executable,
            "-B",
            str(SCRIPTS / "merge_trained_arm.py"),
            "--name",
            arm,
            "--adapter",
            str(ADAPTER_ROOT / arm),
            "--out",
            str(MERGED_ROOT / arm),
        ])


def local_stage() -> None:
    for relative in (
        "data/local_design_receipt.json",
        "reports/local_design_review.md",
        *(f"runs/merges/{arm}.json" for arm in ARMS),
    ):
        require_pushed_checkpoint(f"{EXP_REL}/{relative}")
    require_verdict(LOCAL_REVIEW, LOCAL_VERDICT, "local adversarial review")
    run([sys.executable, "-B", str(SCRIPTS / "eval_local_vllm.py")])
    if not LOCAL_RECEIPT.is_file():
        raise SystemExit("local evaluation did not produce the frozen local receipt")
    if MECHANISM_RECEIPT.exists():
        run([sys.executable, "-B", str(SCRIPTS / "check_local.py"), str(LOCAL_RECEIPT)])
    else:
        run([
            sys.executable,
            "-B",
            str(SCRIPTS / "check_local.py"),
            str(LOCAL_RECEIPT),
            "--out",
            str(MECHANISM_RECEIPT),
        ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument(
        "--stage",
        choices=(
            "train-candidate",
            "merge-candidate",
            "local",
        ),
    )
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage in TRAIN_STAGE_ARMS:
        train_stage(TRAIN_STAGE_ARMS[args.stage])
        return 0
    if args.stage == "merge-candidate":
        merge_candidate_stage()
        return 0
    if args.stage == "local":
        local_stage()
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
