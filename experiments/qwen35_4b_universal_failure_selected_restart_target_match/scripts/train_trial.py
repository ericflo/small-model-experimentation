#!/usr/bin/env python3
"""Fail-closed wrapper for the frozen exact-exposure QLoRA trial."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
TRAINER = EXP / "scripts" / "train_think.py"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TOKEN_RECEIPT = EXP / "data" / "stream_token_receipt.json"
TOKEN_RECEIPT_SHA256 = "52a761ef8fd37f3eac88abf8f090013f571a47511daeb26820ca030201b1c170"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
PARENT_ADAPTER = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
    / "adapters"
    / "replay_after_close"
)
PARENT_WEIGHTS_SHA256 = "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d"
PARENT_CONFIG_SHA256 = "0dfd9bda8a835926a87337782cc09b1e11e841a36f46b99c83fbae9bc89e120f"
EXPECTED_ROWS = 320
EXPECTED_FORWARD = 297731
EXPECTED_NONZERO = 126796
EXPECTED_MASS_X5 = 138164
ARM_FILES = {
    "replay_control": (
        EXP / "data" / "replay_control.jsonl",
        "7a8d45666000cbb6bffabf6faab8f9d61006bf3a80275a631238a23cd03b5078",
    ),
    "counterfactual_restart_candidate": (
        EXP / "data" / "counterfactual_restart_candidate.jsonl",
        "28deb20e6bfca81f760549b071d0d0df39bfa561c4d09fde0580d81699413190",
    ),
}
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=False, capture_output=True
    )
    return committed.returncode == 0 and committed.stdout == path.read_bytes()


def normalize_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def expected_hyperparameters() -> dict:
    return {
        "epochs": 1.0,
        "lr": 1e-5,
        "rank": 32,
        "alpha": 64,
        "batch_size": 1,
        "grad_accum": 8,
        "max_length": 4096,
        "w_think": 0.2,
        "w_close": 0.2,
        "seed": 48,
        "optimizer_steps": 40,
    }


def validate_published_arm(name: str, *, require_committed: bool = True) -> dict:
    if name not in ARM_FILES:
        raise ValueError(f"unknown arm: {name}")
    receipt_path = EXP / "runs" / "training" / f"{name}.json"
    log_path = EXP / "runs" / "training" / f"{name}.log"
    adapter = ADAPTER_ROOT / name
    if (
        not receipt_path.is_file()
        or not log_path.is_file()
        or (require_committed and not committed_at_head(receipt_path))
        or (require_committed and not committed_at_head(log_path))
    ):
        raise ValueError(f"published {name} receipt/log is absent from HEAD")
    payload = load_json(receipt_path)
    data_path, data_hash = ARM_FILES[name]
    dataset = payload.get("dataset", {})
    warm_start = payload.get("warm_start", {})
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("returncode") != 0
        or payload.get("adapter_complete") is not True
        or payload.get("token_receipt_sha256") != TOKEN_RECEIPT_SHA256
        or payload.get("train_rows") != EXPECTED_ROWS
        or payload.get("forward_tokens_per_epoch") != EXPECTED_FORWARD
        or payload.get("nonzero_target_tokens_per_epoch") != EXPECTED_NONZERO
        or payload.get("absolute_loss_mass_x5_per_epoch") != EXPECTED_MASS_X5
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or payload.get("hyperparameters") != expected_hyperparameters()
        or dataset.get("path") != str(data_path.resolve())
        or dataset.get("sha256") != data_hash
        or dataset.get("rows") != EXPECTED_ROWS
        or warm_start.get("path") != str(PARENT_ADAPTER.resolve())
        or warm_start.get("weights_sha256") != PARENT_WEIGHTS_SHA256
        or warm_start.get("config_sha256") != PARENT_CONFIG_SHA256
        or payload.get("log_sha256") != sha256_file(log_path)
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or not config.is_file()
        or not weights.is_file()
        or payload.get("adapter_config_sha256") != sha256_file(config)
        or payload.get("adapter_weights_sha256") != sha256_file(weights)
        or payload.get("benchmark_data_read") is not False
        or payload.get("aggregate_seed_open") is not False
    ):
        raise ValueError(f"published {name} violates the frozen training contract")
    return {
        "receipt": str(receipt_path.resolve()),
        "receipt_sha256": sha256_file(receipt_path),
        "adapter": str(adapter.resolve()),
        "adapter_config_sha256": payload["adapter_config_sha256"],
        "adapter_weights_sha256": payload["adapter_weights_sha256"],
        "training_git_head": payload["preflight_git_head"],
    }


def preserve_failure(path: Path, common: dict, *, reason: str, returncode: int) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {**common, "returncode": returncode, "failure_reason": reason},
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", choices=tuple(ARM_FILES), required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--token-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--w-close", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=48)
    args = parser.parse_args()

    expected_data, expected_data_hash = ARM_FILES[args.name]
    expected_out = ADAPTER_ROOT / args.name
    if args.train.resolve() != expected_data.resolve():
        parser.error("training data does not match the frozen arm")
    if args.out.resolve() != expected_out.resolve():
        parser.error("adapter output does not match the frozen large-artifact path")
    if args.warm_start.resolve() != PARENT_ADAPTER.resolve():
        parser.error("warm start does not match the frozen replay parent")
    if args.token_receipt.resolve() != TOKEN_RECEIPT.resolve():
        parser.error("token receipt path changed")
    observed_hyperparameters = {
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "max_length": args.max_length,
        "w_think": args.w_think,
        "w_close": args.w_close,
        "seed": args.seed,
        "optimizer_steps": 40,
    }
    if observed_hyperparameters != expected_hyperparameters():
        parser.error("training hyperparameters differ from the frozen review")
    if (
        not TOKEN_RECEIPT.is_file()
        or sha256_file(TOKEN_RECEIPT) != TOKEN_RECEIPT_SHA256
        or not expected_data.is_file()
        or sha256_file(expected_data) != expected_data_hash
        or "**Verdict:** `PASS_CONTROL_TRAINING`." not in COMPUTE_REVIEW.read_text(encoding="utf-8")
    ):
        parser.error("training receipt, stream, or compute-review authorization changed")
    token_receipt = load_json(TOKEN_RECEIPT)
    file_receipt = token_receipt.get("files", {}).get(args.name, {})
    spans = file_receipt.get("spans_per_epoch", {})
    if (
        token_receipt.get("rows_per_arm") != EXPECTED_ROWS
        or token_receipt.get("forward_tokens_per_arm") != EXPECTED_FORWARD
        or token_receipt.get("nonzero_target_tokens_per_arm") != EXPECTED_NONZERO
        or token_receipt.get("absolute_loss_mass_x5_per_arm") != EXPECTED_MASS_X5
        or token_receipt.get("skipped_rows") != 0
        or file_receipt.get("sha256") != expected_data_hash
        or file_receipt.get("rows") != EXPECTED_ROWS
        or file_receipt.get("encoded_rows") != EXPECTED_ROWS
        or file_receipt.get("skipped_rows") != 0
        or spans.get("forward") != EXPECTED_FORWARD
        or spans.get("nonzero_target") != EXPECTED_NONZERO
        or spans.get("absolute_loss_mass_x5") != EXPECTED_MASS_X5
    ):
        parser.error("token receipt does not match the frozen arm exposure")
    parent_config = PARENT_ADAPTER / "adapter_config.json"
    parent_weights = PARENT_ADAPTER / "adapter_model.safetensors"
    if (
        not parent_config.is_file()
        or not parent_weights.is_file()
        or sha256_file(parent_config) != PARENT_CONFIG_SHA256
        or sha256_file(parent_weights) != PARENT_WEIGHTS_SHA256
    ):
        parser.error("authenticated replay-parent adapter changed")
    control_prerequisite = None
    if args.name == "counterfactual_restart_candidate":
        try:
            control_prerequisite = validate_published_arm("replay_control")
        except (OSError, ValueError) as error:
            parser.error(str(error))

    log_path = EXP / "runs" / "training" / f"{args.name}.log"
    receipt_path = EXP / "runs" / "training" / f"{args.name}.json"
    failure_path = EXP / "runs" / "training" / f"{args.name}.failure.json"
    if any(path.exists() for path in (expected_out, log_path, receipt_path, failure_path)):
        parser.error("refusing to overwrite an adapter, log, or receipt")
    branch = run_text(["git", "branch", "--show-current"])
    head = run_text(["git", "rev-parse", "HEAD"])
    origin = run_text(["git", "rev-parse", "origin/main"])
    status = run_text(["git", "status", "--short"])
    if branch != "main" or head != origin or status:
        parser.error("training requires a clean pushed main checkpoint")

    command = [
        str(PYTHON), "-B", str(TRAINER),
        "--train", str(expected_data.resolve()),
        "--out", str(expected_out.resolve()),
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--rank", str(args.rank),
        "--alpha", str(args.alpha),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--max-length", str(args.max_length),
        "--w-think", str(args.w_think),
        "--w-close", str(args.w_close),
        "--seed", str(args.seed),
        "--warm-start", str(PARENT_ADAPTER.resolve()),
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log.write(line)
            log.flush()
        returncode = process.wait()
    elapsed = time.perf_counter() - started
    normalize_log(log_path)
    common = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "name": args.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dataset": {
            "path": str(expected_data.resolve()),
            "sha256": expected_data_hash,
            "rows": EXPECTED_ROWS,
            "spans_per_epoch": spans,
        },
        "token_receipt": str(TOKEN_RECEIPT.resolve()),
        "token_receipt_sha256": TOKEN_RECEIPT_SHA256,
        "train_rows": EXPECTED_ROWS,
        "forward_tokens_per_epoch": EXPECTED_FORWARD,
        "nonzero_target_tokens_per_epoch": EXPECTED_NONZERO,
        "absolute_loss_mass_x5_per_epoch": EXPECTED_MASS_X5,
        "warm_start": {
            "path": str(PARENT_ADAPTER.resolve()),
            "config_sha256": PARENT_CONFIG_SHA256,
            "weights_sha256": PARENT_WEIGHTS_SHA256,
        },
        "hyperparameters": observed_hyperparameters,
        "command": command,
        "wall_seconds": elapsed,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": head,
        "preflight_git_branch": branch,
        "preflight_origin_main": origin,
        "preflight_git_status": status,
        "post_training_git_status": run_text(["git", "status", "--short"]),
        "control_prerequisite": control_prerequisite,
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
    }
    if returncode != 0:
        preserve_failure(failure_path, common, reason="trainer_nonzero_exit", returncode=returncode)
        raise SystemExit(f"trainer failed with exit {returncode}; preserved log/failure receipt")
    log_text = log_path.read_text(encoding="utf-8")
    encoded = re.search(r"\[train_think\] (\d+) examples \((\d+) skipped", log_text)
    loss_matches = re.findall(r"'train_loss':\s*'?([0-9.eE+-]+)'?", log_text)
    config = expected_out / "adapter_config.json"
    weights = expected_out / "adapter_model.safetensors"
    if (
        not encoded
        or int(encoded.group(1)) != EXPECTED_ROWS
        or int(encoded.group(2)) != 0
        or not config.is_file()
        or not weights.is_file()
    ):
        preserve_failure(failure_path, common, reason="zero_skip_or_adapter_contract_failed", returncode=0)
        raise SystemExit("trainer output failed zero-skip or complete-adapter validation")
    if load_json(config).get("base_model_name_or_path") != MODEL_ID:
        preserve_failure(failure_path, common, reason="adapter_base_model_identity_mismatch", returncode=0)
        raise SystemExit("adapter base model identity mismatch")
    packages = {
        package: importlib.metadata.version(package)
        for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
    }
    receipt = {
        **common,
        "returncode": 0,
        "skipped_rows": 0,
        "train_loss": float(loss_matches[-1]) if loss_matches else None,
        "adapter": str(expected_out.resolve()),
        "adapter_complete": True,
        "adapter_config_sha256": sha256_file(config),
        "adapter_weights_sha256": sha256_file(weights),
        "adapter_size_bytes": weights.stat().st_size,
        "packages": packages,
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
