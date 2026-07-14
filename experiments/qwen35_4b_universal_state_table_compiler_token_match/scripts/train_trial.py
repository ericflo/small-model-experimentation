#!/usr/bin/env python3
"""Fail-closed wrapper around the proven QLoRA trainer with a durable receipt."""

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
PARENT_WEIGHTS_SHA256 = "16e9dc75a0e33e182e916600ff6e1d75fc46dfa45e870216e2c149a41253c179"
PARENT_CONFIG_SHA256 = "de953bd57502ff728a12d1627d5aacab6284b045428ec7b83026388afd8c47ff"
TOKEN_RECEIPT = EXP / "data" / "stream_token_receipt.json"
TOKEN_RECEIPT_SHA256 = "163e40a61d0b3f4dc541f56ea32510bacb8ce64f658e00f47e5867da4a45f0b8"
FROZEN_TRAIN_FILES = {
    "replay_after_close": EXP / "data" / "replay_after_close.jsonl",
    "state_table_after_close": EXP / "data" / "state_table_after_close.jsonl",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"not a JSON object: {path}")
    return payload


def run_text(command: list[str]) -> str:
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def normalize_log(path: Path) -> None:
    """Remove progress-renderer trailing blanks before the durable log is hashed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--token-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--w-close", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("--name must be a safe lowercase identifier")
    if args.name not in FROZEN_TRAIN_FILES:
        parser.error("--name is not a preregistered arm")
    if (
        args.token_receipt.resolve() != TOKEN_RECEIPT.resolve()
        or not TOKEN_RECEIPT.is_file()
        or sha256_file(TOKEN_RECEIPT) != TOKEN_RECEIPT_SHA256
    ):
        parser.error("token receipt does not match the frozen experiment receipt")
    expected_train = FROZEN_TRAIN_FILES[args.name].resolve()
    if len(args.train) != 1 or args.train[0].resolve() != expected_train:
        parser.error("training data does not match the preregistered arm")
    resolved_out = args.out.resolve()
    # The repository's established external-artifact convention is its gitignored
    # large_artifacts/ root.  Reject experiment-local adapter directories while
    # allowing that explicit external payload area.
    if resolved_out.is_relative_to(ROOT.resolve()) and not resolved_out.is_relative_to(
        (ROOT / "large_artifacts").resolve()
    ):
        parser.error("adapter output must live outside experiments/ (use large_artifacts/)")
    expected_out = (
        ROOT / "large_artifacts" / EXP.name / "adapters" / args.name
    ).resolve()
    if resolved_out != expected_out:
        parser.error("adapter output does not match the preregistered arm path")
    if args.warm_start is None:
        parser.error("the authenticated close_xi warm start is required")
    frozen_hyperparameters = (
        args.epochs == 1.0
        and args.lr == 1e-5
        and args.rank == 32
        and args.alpha == 64
        and args.batch_size == 1
        and args.grad_accum == 8
        and args.max_length == 4096
        and args.w_think == 0.2
        and args.w_close == 0.2
        and args.seed == 46
    )
    if not frozen_hyperparameters:
        parser.error("training hyperparameters differ from the preregistration")
    log_path = EXP / "runs" / "training" / f"{args.name}.log"
    receipt_path = EXP / "runs" / "training" / f"{args.name}.json"
    failure_path = EXP / "runs" / "training" / f"{args.name}.failure.json"
    if args.out.exists() or log_path.exists() or receipt_path.exists() or failure_path.exists():
        parser.error("refusing to overwrite an existing adapter, log, or receipt")

    token_receipt = load_json(args.token_receipt)
    if (
        token_receipt.get("model_id") != MODEL_ID
        or token_receipt.get("model_revision") != MODEL_REVISION
        or int(token_receipt.get("max_length", -1)) != args.max_length
        or int(token_receipt.get("skipped_rows", -1)) != 0
    ):
        parser.error("token receipt does not match the frozen model/length/zero-skip contract")
    recorded = {
        Path(row["path"]).resolve(): row for row in token_receipt.get("files", [])
    }
    train_rows = 0
    datasets = []
    for path in args.train:
        resolved = path.resolve()
        row = recorded.get(resolved)
        if (
            row is None
            or row.get("sha256") != sha256_file(resolved)
            or int(row.get("rows", -1)) != 320
            or int(row.get("total_forward_tokens_per_epoch", -1)) != 286814
        ):
            parser.error(f"training file is absent or changed relative to token receipt: {path}")
        train_rows += int(row["rows"])
        datasets.append({"path": str(resolved), "sha256": row["sha256"], "rows": row["rows"]})

    warm_start = None
    if args.warm_start:
        config = args.warm_start / "adapter_config.json"
        weights = args.warm_start / "adapter_model.safetensors"
        if not config.is_file() or not weights.is_file():
            parser.error("warm-start adapter is incomplete")
        warm_start = {
            "path": str(args.warm_start.resolve()),
            "config_sha256": sha256_file(config),
            "weights_sha256": sha256_file(weights),
        }
        if (
            warm_start["config_sha256"] != PARENT_CONFIG_SHA256
            or warm_start["weights_sha256"] != PARENT_WEIGHTS_SHA256
        ):
            parser.error("warm-start adapter does not match the authenticated close_xi parent")

    command = [
        str(PYTHON), str(TRAINER), "--train", *(str(path.resolve()) for path in args.train),
        "--out", str(args.out.resolve()), "--epochs", str(args.epochs), "--lr", str(args.lr),
        "--rank", str(args.rank), "--alpha", str(args.alpha), "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum), "--max-length", str(args.max_length),
        "--w-think", str(args.w_think), "--w-close", str(args.w_close),
        "--seed", str(args.seed),
    ]
    if args.warm_start:
        command.extend(("--warm-start", str(args.warm_start.resolve())))

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
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
    if returncode != 0:
        failure_path.write_text(
            json.dumps({
                "schema_version": 1,
                "experiment_id": EXP.name,
                "name": args.name,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "datasets": datasets,
                "token_receipt": str(args.token_receipt.resolve()),
                "token_receipt_sha256": sha256_file(args.token_receipt),
                "train_rows": train_rows,
                "warm_start": warm_start,
                "hyperparameters": {
                    "epochs": args.epochs, "lr": args.lr, "rank": args.rank,
                    "alpha": args.alpha, "batch_size": args.batch_size,
                    "grad_accum": args.grad_accum, "max_length": args.max_length,
                    "w_think": args.w_think, "w_close": args.w_close,
                    "seed": args.seed,
                },
                "command": command,
                "returncode": returncode,
                "wall_seconds": elapsed,
                "log": str(log_path.resolve()),
                "log_sha256": sha256_file(log_path),
                "adapter_complete": False,
            }, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        raise SystemExit(f"trainer failed with exit code {returncode}; preserved {log_path}")

    log_text = log_path.read_text(encoding="utf-8")
    encoded_match = re.search(r"\[train_think\] (\d+) examples \((\d+) skipped", log_text)
    loss_matches = re.findall(r"'train_loss': '([^']+)'", log_text)
    if not encoded_match or int(encoded_match.group(1)) != train_rows or int(encoded_match.group(2)) != 0:
        raise SystemExit("trainer row count diverged from the frozen zero-skip token receipt")
    adapter_config = args.out / "adapter_config.json"
    adapter_weights = args.out / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        raise SystemExit("trainer returned success without a complete adapter")
    config_payload = load_json(adapter_config)
    if config_payload.get("base_model_name_or_path") != MODEL_ID:
        raise SystemExit("adapter base model identity mismatch")

    packages = {}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "accelerate"):
        packages[package] = importlib.metadata.version(package)
    receipt = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "name": args.name,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "datasets": datasets,
        "token_receipt": str(args.token_receipt.resolve()),
        "token_receipt_sha256": sha256_file(args.token_receipt),
        "train_rows": train_rows,
        "skipped_rows": 0,
        "warm_start": warm_start,
        "hyperparameters": {
            "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
            "max_length": args.max_length, "w_think": args.w_think, "seed": args.seed,
            "w_close": args.w_close,
        },
        "command": command,
        "wall_seconds": elapsed,
        "train_loss": float(loss_matches[-1]) if loss_matches else None,
        "adapter": str(args.out.resolve()),
        "adapter_config_sha256": sha256_file(adapter_config),
        "adapter_weights_sha256": sha256_file(adapter_weights),
        "adapter_size_bytes": adapter_weights.stat().st_size,
        "packages": packages,
        "git_head": run_text(["git", "rev-parse", "HEAD"]),
        "git_status": run_text(["git", "status", "--short"]),
    }
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
