#!/usr/bin/env python3
"""Fail-closed staged harness for on-policy failure-prefix correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPTS = EXP / "scripts"
PARENT_ADAPTER = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_close_weight_token_match"
    / "adapters"
    / "close_xi"
)
MERGED_PARENT = ROOT / "large_artifacts" / EXP.name / "merged" / "close_xi_parent"
MERGE_RECEIPT = EXP / "runs" / "merges" / "close_xi_parent.json"
ROLLOUT_RECEIPT = EXP / "runs" / "parent_rollout" / "seed66113.receipt.json"
DESIGN_RECEIPT = EXP / "data" / "design_receipt.json"
TOKEN_RECEIPT = EXP / "data" / "stream_token_receipt.json"
TOKEN_RECEIPT_SHA256 = "eb08026ffcf82b8780819a26a522f04d69358ffdfd4797dd4c603dd1fbbe0cfc"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
CONTROL_RECEIPT = EXP / "runs" / "training" / "replay_after_close.json"
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PARENT_WEIGHTS_SHA256 = "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179"
PARENT_CONFIG_SHA256 = "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def require_clean_committed_checkpoint(required_paths: tuple[Path, ...] = ()) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise SystemExit("stage requires a clean incrementally committed worktree")
    for path in required_paths:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if (
            committed.returncode != 0
            or not path.is_file()
            or committed.stdout != path.read_bytes()
        ):
            raise SystemExit(f"required checkpoint receipt is not committed at HEAD: {relative}")


def smoke() -> None:
    config = (EXP / "configs" / "default.yaml").read_text(encoding="utf-8")
    required = (
        f"model_id: {MODEL_ID}",
        f"model_revision: {MODEL_REVISION}",
        f"parent_weights_sha256: {PARENT_WEIGHTS_SHA256}",
        f"parent_config_sha256: {PARENT_CONFIG_SHA256}",
        "status: control_trained_candidate_training_next",
        "rows_per_training_arm: 320",
        "forward_tokens_per_training_arm: 304313",
        "optimizer_steps_per_training_arm: 40",
        "control_receipt_sha256: f78f2069fd1c7b37bbd0b13b581df0ce7360de92256323fcf5f3c7b0936ed6de",
    )
    missing = [entry for entry in required if entry not in config]
    if missing:
        raise SystemExit(f"frozen config entries are missing: {missing}")
    if (
        sha256_file(PARENT_ADAPTER / "adapter_model.safetensors")
        != PARENT_WEIGHTS_SHA256
        or sha256_file(PARENT_ADAPTER / "adapter_config.json")
        != PARENT_CONFIG_SHA256
    ):
        raise SystemExit("authenticated close_xi parent identity changed")
    run([str(PYTHON), "-B", str(SCRIPTS / "gen_rollout_tasks.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "check_design.py"), "--check"])
    review = (EXP / "reports" / "design_review.md").read_text(encoding="utf-8")
    if "**Verdict:** `PASS_PARENT_MERGE`." not in review:
        raise SystemExit("adversarial design review has not authorized the parent merge")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    run(
        [
            str(PYTHON),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(EXP / "tests"),
            "-q",
        ]
    )
    run([str(PYTHON), "-B", str(SCRIPTS / "mine_prefix_repairs.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "measure_source_tokens.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "materialize_streams.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "validate_streams.py"), "--check"])
    if sha256_file(TOKEN_RECEIPT) != TOKEN_RECEIPT_SHA256:
        raise SystemExit("frozen stream-token receipt bytes changed")
    token_receipt = json.loads(TOKEN_RECEIPT.read_text(encoding="utf-8"))
    if (
        token_receipt.get("rows_per_arm") != 320
        or token_receipt.get("forward_tokens_per_arm") != 304313
        or token_receipt.get("forward_token_delta") != 0
        or token_receipt.get("skipped_rows") != 0
        or token_receipt.get("shared_position_aligned_rows") != 200
        or token_receipt.get("training", {}).get("optimizer_steps") != 40
    ):
        raise SystemExit("frozen stream-token receipt contract changed")
    compute_review = COMPUTE_REVIEW.read_text(encoding="utf-8")
    if "**Verdict:** `PASS_CONTROL_TRAINING`." not in compute_review:
        raise SystemExit("second adversarial review has not authorized control training")
    control_receipt = None
    if CONTROL_RECEIPT.is_file():
        sys.path.insert(0, str(SCRIPTS))
        from train_trial import validate_control_prerequisite

        control_receipt = validate_control_prerequisite(require_committed=False)
    design = json.loads(DESIGN_RECEIPT.read_text(encoding="utf-8"))
    print(
        "design, prefix-mining, and exact-compute smoke passed: "
        f"{design['rollout_tasks']['rows']} fresh tasks, six balanced failure classes, "
        "60 quota-satisfying masked repairs, 320 rows and 304313 forward tokens per arm"
    )
    if control_receipt is not None:
        print(
            "authenticated replay control: "
            f"receipt {Path(control_receipt['receipt']).name}, "
            f"adapter {control_receipt['adapter_weights_sha256']}"
        )


def merge_parent() -> None:
    run(
        [
            str(PYTHON),
            "-B",
            str(SCRIPTS / "merge_parent.py"),
            "--name",
            "close_xi_parent",
            "--adapter",
            str(PARENT_ADAPTER),
            "--out",
            str(MERGED_PARENT),
        ]
    )


def train_arm(name: str) -> None:
    train_data = EXP / "data" / f"{name}.jsonl"
    run(
        [
            str(PYTHON),
            "-B",
            str(SCRIPTS / "train_trial.py"),
            "--name",
            name,
            "--train",
            str(train_data),
            "--token-receipt",
            str(TOKEN_RECEIPT),
            "--out",
            str(ADAPTER_ROOT / name),
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
            "47",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "merge-parent",
            "collect-parent",
            "mine-prefixes",
            "train-control",
            "train-candidate",
        ),
        help="run exactly one frozen stage",
    )
    args = parser.parse_args()
    if args.smoke:
        if args.stage:
            parser.error("--smoke and --stage are mutually exclusive")
        smoke()
        return 0
    if not args.stage:
        parser.error("choose --smoke or one explicit --stage")
    smoke()
    if args.stage == "merge-parent":
        require_clean_committed_checkpoint((DESIGN_RECEIPT,))
        merge_parent()
    elif args.stage == "collect-parent":
        require_clean_committed_checkpoint((MERGE_RECEIPT,))
        run([str(PYTHON), "-B", str(SCRIPTS / "collect_parent_rollouts.py")])
    elif args.stage == "mine-prefixes":
        require_clean_committed_checkpoint((ROLLOUT_RECEIPT,))
        run([str(PYTHON), "-B", str(SCRIPTS / "mine_prefix_repairs.py")])
    elif args.stage == "train-control":
        require_clean_committed_checkpoint((TOKEN_RECEIPT, COMPUTE_REVIEW))
        train_arm("replay_after_close")
    elif args.stage == "train-candidate":
        require_clean_committed_checkpoint(
            (TOKEN_RECEIPT, COMPUTE_REVIEW, CONTROL_RECEIPT)
        )
        train_arm("prefix_repair_after_close")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
