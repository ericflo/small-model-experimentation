#!/usr/bin/env python3
"""Fail-closed wrapper around the frozen QLoRA trial with a durable receipt."""

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
TOKEN_RECEIPT_SHA256 = "eb08026ffcf82b8780819a26a522f04d69358ffdfd4797dd4c603dd1fbbe0cfc"
EXPECTED_FORWARD_TOKENS = 304313
EXPECTED_ROWS = 320
CONTROL_DATA_SHA256 = "541805df2d817707c1e76213e50c8f08fd9caff10d0a3887e1196424b6820be6"
CONTROL_RECEIPT_SHA256 = "f78f2069fd1c7b37bbd0b13b581df0ce7360de92256323fcf5f3c7b0936ed6de"
CONTROL_CONFIG_SHA256 = "0dfd9bda8a835926a87337782cc09b1e11e841a36f46b99c83fbae9bc89e120f"
CONTROL_WEIGHTS_SHA256 = "bb59d3bd9273ae3bb3dffe54e983590dada69e6e1bdba571009ffedbba05154d"
CONTROL_RECEIPT = EXP / "runs" / "training" / "replay_after_close.json"
CONTROL_ADAPTER = (
    ROOT / "large_artifacts" / EXP.name / "adapters" / "replay_after_close"
)
FROZEN_TRAIN_FILES = {
    "replay_after_close": EXP / "data" / "replay_after_close.jsonl",
    "prefix_repair_after_close": EXP / "data" / "prefix_repair_after_close.jsonl",
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
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def normalize_log(path: Path) -> None:
    """Remove progress-renderer trailing blanks before the durable log is hashed."""
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def frozen_hyperparameters(args: argparse.Namespace) -> bool:
    return (
        args.epochs == 1.0
        and args.lr == 1e-5
        and args.rank == 32
        and args.alpha == 64
        and args.batch_size == 1
        and args.grad_accum == 8
        and args.max_length == 4096
        and args.w_think == 0.2
        and args.w_close == 0.2
        and args.seed == 47
    )


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return committed.returncode == 0 and committed.stdout == path.read_bytes()


def validate_control_prerequisite(*, require_committed: bool = True) -> dict:
    """Authenticate the published replay control before candidate training."""
    if not CONTROL_RECEIPT.is_file() or (
        require_committed and not committed_at_head(CONTROL_RECEIPT)
    ):
        raise ValueError("candidate requires the control receipt committed at HEAD")
    if sha256_file(CONTROL_RECEIPT) != CONTROL_RECEIPT_SHA256:
        raise ValueError("published control receipt bytes changed")
    payload = load_json(CONTROL_RECEIPT)
    datasets = payload.get("datasets", [])
    warm_start = payload.get("warm_start", {})
    log = Path(payload.get("log", ""))
    expected_hyperparameters = {
        "epochs": 1.0,
        "lr": 1e-5,
        "rank": 32,
        "alpha": 64,
        "batch_size": 1,
        "grad_accum": 8,
        "max_length": 4096,
        "w_think": 0.2,
        "w_close": 0.2,
        "seed": 47,
        "optimizer_steps": 40,
    }
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("name") != "replay_after_close"
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("returncode") != 0
        or payload.get("adapter_complete") is not True
        or payload.get("adapter_config_sha256") != CONTROL_CONFIG_SHA256
        or payload.get("adapter_weights_sha256") != CONTROL_WEIGHTS_SHA256
        or payload.get("token_receipt_sha256") != TOKEN_RECEIPT_SHA256
        or payload.get("train_rows") != EXPECTED_ROWS
        or payload.get("forward_tokens_per_epoch") != EXPECTED_FORWARD_TOKENS
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or not re.fullmatch(r"[0-9a-f]{40}", str(payload.get("preflight_git_head", "")))
        or payload.get("hyperparameters") != expected_hyperparameters
        or len(datasets) != 1
        or datasets[0].get("sha256") != CONTROL_DATA_SHA256
        or datasets[0].get("rows") != EXPECTED_ROWS
        or datasets[0].get("forward_tokens_per_epoch") != EXPECTED_FORWARD_TOKENS
        or Path(datasets[0].get("path", "")).resolve()
        != FROZEN_TRAIN_FILES["replay_after_close"].resolve()
        or warm_start.get("weights_sha256") != PARENT_WEIGHTS_SHA256
        or warm_start.get("config_sha256") != PARENT_CONFIG_SHA256
        or Path(payload.get("adapter", "")).resolve() != CONTROL_ADAPTER.resolve()
    ):
        raise ValueError("published control receipt violates the frozen contract")
    config = CONTROL_ADAPTER / "adapter_config.json"
    weights = CONTROL_ADAPTER / "adapter_model.safetensors"
    if (
        not config.is_file()
        or not weights.is_file()
        or sha256_file(config) != CONTROL_CONFIG_SHA256
        or sha256_file(weights) != CONTROL_WEIGHTS_SHA256
    ):
        raise ValueError("published control adapter is absent or changed")
    if (
        not log.is_file()
        or (require_committed and not committed_at_head(log))
        or payload.get("log_sha256") != sha256_file(log)
    ):
        raise ValueError("published control log is absent or changed")
    return {
        "receipt": str(CONTROL_RECEIPT.resolve()),
        "receipt_sha256": sha256_file(CONTROL_RECEIPT),
        "adapter": str(CONTROL_ADAPTER.resolve()),
        "adapter_config_sha256": payload["adapter_config_sha256"],
        "adapter_weights_sha256": payload["adapter_weights_sha256"],
        "training_git_head": payload["preflight_git_head"],
    }


def preserve_failure(
    path: Path,
    common: dict,
    *,
    reason: str,
    returncode: int,
    adapter_complete: bool,
) -> None:
    failure = {
        **common,
        "returncode": returncode,
        "failure_reason": reason,
        "adapter_complete": adapter_complete,
    }
    with path.open("x", encoding="utf-8") as handle:
        handle.write(
            json.dumps(failure, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--token-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--w-close", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=47)
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
    if not frozen_hyperparameters(args):
        parser.error("training hyperparameters differ from the preregistration")
    control_prerequisite = None
    if args.name == "prefix_repair_after_close":
        try:
            control_prerequisite = validate_control_prerequisite()
        except (OSError, ValueError) as error:
            parser.error(str(error))

    log_path = EXP / "runs" / "training" / f"{args.name}.log"
    receipt_path = EXP / "runs" / "training" / f"{args.name}.json"
    failure_path = EXP / "runs" / "training" / f"{args.name}.failure.json"
    if args.out.exists() or log_path.exists() or receipt_path.exists() or failure_path.exists():
        parser.error("refusing to overwrite an existing adapter, log, or receipt")

    token_receipt = load_json(args.token_receipt)
    training = token_receipt.get("training", {})
    if (
        token_receipt.get("model_id") != MODEL_ID
        or token_receipt.get("model_revision") != MODEL_REVISION
        or int(token_receipt.get("max_length", -1)) != args.max_length
        or int(token_receipt.get("rows_per_arm", -1)) != EXPECTED_ROWS
        or int(token_receipt.get("forward_tokens_per_arm", -1))
        != EXPECTED_FORWARD_TOKENS
        or int(token_receipt.get("skipped_rows", -1)) != 0
        or int(token_receipt.get("shared_position_aligned_rows", -1)) != 200
        or training
        != {
            "batch_size": 1,
            "close_weight": 0.2,
            "epochs": 1,
            "forward_tokens_per_arm": EXPECTED_FORWARD_TOKENS,
            "gradient_accumulation": 8,
            "learning_rate": 1e-5,
            "max_length": 4096,
            "optimizer_steps": 40,
            "rows_per_arm": EXPECTED_ROWS,
            "seed": 47,
            "thought_weight": 0.2,
        }
    ):
        parser.error("token receipt does not match the frozen training contract")

    recorded = {
        (ROOT / row["path"]).resolve(): row for row in token_receipt.get("files", [])
    }
    train_rows = 0
    datasets = []
    for path in args.train:
        resolved = path.resolve()
        row = recorded.get(resolved)
        spans = row.get("spans_per_epoch", {}) if row else {}
        if (
            row is None
            or row.get("sha256") != sha256_file(resolved)
            or int(row.get("rows", -1)) != EXPECTED_ROWS
            or int(row.get("encoded_rows", -1)) != EXPECTED_ROWS
            or int(row.get("skipped_rows", -1)) != 0
            or int(spans.get("forward", -1)) != EXPECTED_FORWARD_TOKENS
        ):
            parser.error(f"training file is absent or changed relative to token receipt: {path}")
        train_rows += int(row["rows"])
        datasets.append(
            {
                "path": str(resolved),
                "sha256": row["sha256"],
                "rows": row["rows"],
                "forward_tokens_per_epoch": spans["forward"],
                "target_tokens_per_epoch": spans["target_span"],
                "nonzero_weight_tokens_per_epoch": row["nonzero_weight_tokens_per_epoch"],
                "absolute_weight_mass_per_epoch": row["absolute_weight_mass_per_epoch"],
            }
        )

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

    preflight_git_head = run_text(["git", "rev-parse", "HEAD"])
    preflight_git_status = run_text(["git", "status", "--short"])
    if preflight_git_status:
        parser.error("training requires a clean incrementally committed worktree")

    command = [
        str(PYTHON),
        "-B",
        str(TRAINER),
        "--train",
        *(str(path.resolve()) for path in args.train),
        "--out",
        str(args.out.resolve()),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--batch-size",
        str(args.batch_size),
        "--grad-accum",
        str(args.grad_accum),
        "--max-length",
        str(args.max_length),
        "--w-think",
        str(args.w_think),
        "--w-close",
        str(args.w_close),
        "--seed",
        str(args.seed),
        "--warm-start",
        str(args.warm_start.resolve()),
    ]
    hyperparameters = {
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
        "datasets": datasets,
        "token_receipt": str(args.token_receipt.resolve()),
        "token_receipt_sha256": sha256_file(args.token_receipt),
        "train_rows": train_rows,
        "forward_tokens_per_epoch": EXPECTED_FORWARD_TOKENS,
        "warm_start": warm_start,
        "hyperparameters": hyperparameters,
        "command": command,
        "wall_seconds": elapsed,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": preflight_git_head,
        "preflight_git_status": preflight_git_status,
        "post_training_git_status": run_text(["git", "status", "--short"]),
        "control_prerequisite": control_prerequisite,
    }
    if returncode != 0:
        preserve_failure(
            failure_path,
            common,
            reason="trainer_nonzero_exit",
            returncode=returncode,
            adapter_complete=False,
        )
        raise SystemExit(f"trainer failed with exit code {returncode}; preserved {log_path}")

    log_text = log_path.read_text(encoding="utf-8")
    encoded_match = re.search(r"\[train_think\] (\d+) examples \((\d+) skipped", log_text)
    loss_matches = re.findall(r"'train_loss':\s*'?([0-9.eE+-]+)'?", log_text)
    if (
        not encoded_match
        or int(encoded_match.group(1)) != train_rows
        or int(encoded_match.group(2)) != 0
    ):
        preserve_failure(
            failure_path,
            common,
            reason="encoded_row_count_or_skip_mismatch",
            returncode=0,
            adapter_complete=(args.out / "adapter_model.safetensors").is_file(),
        )
        raise SystemExit("trainer row count diverged from the frozen zero-skip token receipt")
    adapter_config = args.out / "adapter_config.json"
    adapter_weights = args.out / "adapter_model.safetensors"
    if not adapter_config.is_file() or not adapter_weights.is_file():
        preserve_failure(
            failure_path,
            common,
            reason="incomplete_adapter",
            returncode=0,
            adapter_complete=False,
        )
        raise SystemExit("trainer returned success without a complete adapter")
    try:
        config_payload = load_json(adapter_config)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        preserve_failure(
            failure_path,
            common,
            reason=f"invalid_adapter_config: {error}",
            returncode=0,
            adapter_complete=False,
        )
        raise SystemExit("trainer returned an invalid adapter config") from error
    if config_payload.get("base_model_name_or_path") != MODEL_ID:
        preserve_failure(
            failure_path,
            common,
            reason="adapter_base_model_identity_mismatch",
            returncode=0,
            adapter_complete=False,
        )
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
        "adapter": str(args.out.resolve()),
        "adapter_complete": True,
        "adapter_config_sha256": sha256_file(adapter_config),
        "adapter_weights_sha256": sha256_file(adapter_weights),
        "adapter_size_bytes": adapter_weights.stat().st_size,
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
