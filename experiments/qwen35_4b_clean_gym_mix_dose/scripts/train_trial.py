#!/usr/bin/env python3
"""Fail-closed wrapper for the frozen gym-mix QLoRA trials.

Two arms train, control FIRST: ``replay_ctl5`` then ``gym_mix``. Each is a
FRESH rank-32/alpha-64 adapter — NO warm start exists in this cell — with
the model weights loaded from the frozen ``zero_root_hygiene_explore`` merged
composite via the per-experiment trainer's ``--model-path`` argument. The
tokenizer stays hub-pinned; the composite's tokenizer files are hash-verified
against the hub-identical pins below.
"""

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
STREAM_TOKEN_RECEIPT_SHA256 = "fcfc905242fcfd9cb0656f3cc66193d91adcb68e287d1c00692ca4bb8685977e"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"
# Training base: the ZERO-ROOT MERGED COMPOSITE (a directory of full
# bf16 weights), NOT an adapter. Both fresh rank-32 adapters train on top of
# it and no warm start is accepted anywhere in this cell.
MODEL_PATH = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_zero_root_lineage_rebuild"
    / "merged"
    / "zero_root_hygiene_explore"
)
MODEL_PATH_WEIGHTS_SHA256 = "6e9aad251465ca2713fda0238a34aa9f46262053860b867f80189d65c9ee3932"
MODEL_PATH_RECEIPT_SHA256 = "f8981f4638d901471eb41aff0ffd0bfac88aebd6e3e4d4db1e1c733be16709c0"
MODEL_PATH_TREE_SHA256 = "414f582950bf60fed2fe462cd141ab98d0f772087b4f9c6bc5aa12f03f379e7d"
MODEL_PATH_WEIGHTS_SIZE_BYTES = 9_078_620_536
# The composite carries the hub-identical tokenizer; the tokenizer itself
# still loads from the pinned hub id inside the trainer (identity preferred).
MODEL_PATH_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
MODEL_PATH_TOKENIZER_CONFIG_SHA256 = "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d"
LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = (
    "down_proj",
    "gate_proj",
    "k_proj",
    "o_proj",
    "q_proj",
    "up_proj",
    "v_proj",
)
EXPECTED_ROWS = 1520
EXPECTED_FORWARD = 1359192
EXPECTED_NONZERO = 567805
EXPECTED_MASS_X5 = 621517
ARM_FILES = {
    "replay_ctl5": (
        EXP / "data" / "replay_ctl5.jsonl",
        "6deebd63cf8309aaf1e691729214bf122df8e8c1108c69a1025f936391ba7247",
    ),
    "gym_mix": (
        EXP / "data" / "gym_mix.jsonl",
        "979cf066115bd37b8b411944e9d3625a5e64f90a7d19a6f79d2619afc7780896",
    ),
}
ARM_PREREQUISITES = {
    "replay_ctl5": (),
    "gym_mix": ("replay_ctl5",),
}
PUBLISHED_ARM_HASHES = {
    # TODO-PIN: after each arm trains and its receipt/log/adapter are
    # published, the orchestrator replaces the None with the four published
    # sha256 values {"receipt": ..., "log": ..., "adapter_config": ...,
    # "adapter_weights": ...}.
    "replay_ctl5": {
        "receipt": "da265238e26a8a50f62a90ecd8f5181233d6e811dcd155805c8dd45903931707",
        "log": "1916286981de472a50360986db3321a4d46e215ce3d4bdd2b7a5be63a7f6710f",
        "adapter_config": "fcafc713abc543c31fcdce4c8c374eeb4fe6664d834eb1f15fc93c5078e6bd97",
        "adapter_weights": "21ee5d03fbc4b87c113f65ba0b6adbd964d0fb44ec1004a4bd7f988cd6ec0893",
    },
    "gym_mix": {
        "receipt": "d3710cc698459fce3e51efa41077129558595a317354816fb45bdbdfbb72d480",
        "log": "85cd6bf0650feebc07f5c9d4180238390a1f4a2269d5e500239e6f71db5195b9",
        "adapter_config": "2ffb7d0f00ebd80f54d59284fe60743a4941f0a23dac0105959d503f5a2bba95",
        "adapter_weights": "28d41601dfefbc390766615ae81d618548b8c60d2538e6d8b8efa73cb0485004",
    },
}
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"


def require_pin(value, name: str):
    if value is None:
        raise SystemExit(
            f"frozen constant {name} is unpinned (TODO-PIN); the orchestrator must "
            "fill it from the committed stream receipt before this stage runs"
        )
    return value


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
        "seed": 79,
        "optimizer_steps": 190,
    }


def frozen_exposure() -> tuple[str, int, int, int]:
    return (
        require_pin(STREAM_TOKEN_RECEIPT_SHA256, "STREAM_TOKEN_RECEIPT_SHA256"),
        require_pin(EXPECTED_FORWARD, "EXPECTED_FORWARD"),
        require_pin(EXPECTED_NONZERO, "EXPECTED_NONZERO"),
        require_pin(EXPECTED_MASS_X5, "EXPECTED_MASS_X5"),
    )


def check_base_composite_cheap() -> None:
    """Receipt-transitive base authentication (no 9 GB hash)."""
    receipt = MODEL_PATH / "merge_receipt.json"
    weights = MODEL_PATH / "model.safetensors"
    tokenizer = MODEL_PATH / "tokenizer.json"
    tokenizer_config = MODEL_PATH / "tokenizer_config.json"
    if (
        not receipt.is_file()
        or sha256_file(receipt) != MODEL_PATH_RECEIPT_SHA256
        or not weights.is_file()
        or weights.stat().st_size != MODEL_PATH_WEIGHTS_SIZE_BYTES
        or not tokenizer.is_file()
        or sha256_file(tokenizer) != MODEL_PATH_TOKENIZER_SHA256
        or not tokenizer_config.is_file()
        or sha256_file(tokenizer_config) != MODEL_PATH_TOKENIZER_CONFIG_SHA256
    ):
        raise ValueError("zero-root training-base composite changed")


def validate_adapter_config(config: dict) -> bool:
    """The fresh-adapter rank/alpha/base-identity contract."""
    return (
        config.get("r") == LORA_RANK
        and config.get("lora_alpha") == LORA_ALPHA
        and config.get("base_model_name_or_path") == str(MODEL_PATH.resolve())
        and sorted(config.get("target_modules") or ()) == list(LORA_TARGET_MODULES)
    )


def validate_published_arm(name: str, *, require_committed: bool = True) -> dict:
    if name not in ARM_FILES:
        raise ValueError(f"unknown arm: {name}")
    token_sha, forward, nonzero, mass_x5 = frozen_exposure()
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
    published = PUBLISHED_ARM_HASHES.get(name)
    if published is None and require_committed:
        # Fail closed like the eval's tree pins: once an arm's receipt is being
        # relied on as a committed prerequisite, its published hashes must be
        # pinned here; a forgotten pin must abort, never silently skip.
        raise ValueError(
            f"PUBLISHED_ARM_HASHES[{name!r}] is unfilled (TODO-PIN); pin the "
            "published receipt/log/adapter hashes before the next stage"
        )
    data_path, data_hash = ARM_FILES[name]
    data_hash = require_pin(data_hash, f"ARM_FILES[{name!r}]")
    dataset = payload.get("dataset", {})
    model_path = payload.get("model_path", {})
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("returncode") != 0
        or payload.get("adapter_complete") is not True
        or payload.get("token_receipt_sha256") != token_sha
        or payload.get("train_rows") != EXPECTED_ROWS
        or payload.get("forward_tokens_per_epoch") != forward
        or payload.get("nonzero_target_tokens_per_epoch") != nonzero
        or payload.get("absolute_loss_mass_x5_per_epoch") != mass_x5
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or payload.get("hyperparameters") != expected_hyperparameters()
        or dataset.get("path") != str(data_path.resolve())
        or dataset.get("sha256") != data_hash
        or dataset.get("rows") != EXPECTED_ROWS
        or model_path.get("path") != str(MODEL_PATH.resolve())
        or model_path.get("weights_sha256") != MODEL_PATH_WEIGHTS_SHA256
        or model_path.get("merge_receipt_sha256") != MODEL_PATH_RECEIPT_SHA256
        or model_path.get("tree_sha256") != MODEL_PATH_TREE_SHA256
        or payload.get("fresh_adapter") is not True
        or payload.get("log_sha256") != sha256_file(log_path)
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or not config.is_file()
        or not weights.is_file()
        or payload.get("adapter_config_sha256") != sha256_file(config)
        or payload.get("adapter_weights_sha256") != sha256_file(weights)
        or not validate_adapter_config(load_json(config))
        or payload.get("benchmark_data_read") is not False
        or payload.get("aggregate_seed_open") is not False
        or (
            published is not None
            and (
                sha256_file(receipt_path) != published["receipt"]
                or sha256_file(log_path) != published["log"]
                or sha256_file(config) != published["adapter_config"]
                or sha256_file(weights) != published["adapter_weights"]
            )
        )
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
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--w-close", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=79)
    args = parser.parse_args()

    token_sha, forward, nonzero, mass_x5 = frozen_exposure()
    expected_data, expected_data_hash = ARM_FILES[args.name]
    expected_data_hash = require_pin(expected_data_hash, f"ARM_FILES[{args.name!r}]")
    expected_out = ADAPTER_ROOT / args.name
    if args.train.resolve() != expected_data.resolve():
        parser.error("training data does not match the frozen arm")
    if args.out.resolve() != expected_out.resolve():
        parser.error("adapter output does not match the frozen large-artifact path")
    if args.model_path.resolve() != MODEL_PATH.resolve():
        parser.error("model path does not match the frozen zero-root composite")
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
        "optimizer_steps": 190,
    }
    if observed_hyperparameters != expected_hyperparameters():
        parser.error("training hyperparameters differ from the frozen review")
    if (
        not TOKEN_RECEIPT.is_file()
        or sha256_file(TOKEN_RECEIPT) != token_sha
        or not expected_data.is_file()
        or sha256_file(expected_data) != expected_data_hash
        or not COMPUTE_REVIEW.is_file()
        or "**Verdict:** `PASS_CONTROL_TRAINING`."
        not in COMPUTE_REVIEW.read_text(encoding="utf-8")
    ):
        parser.error("training receipt, stream, or compute-review authorization changed")
    token_receipt = load_json(TOKEN_RECEIPT)
    file_receipt = token_receipt.get("files", {}).get(args.name, {})
    spans = file_receipt.get("spans_per_epoch", {})
    if (
        token_receipt.get("rows_per_arm") != EXPECTED_ROWS
        or token_receipt.get("forward_tokens_per_arm") != forward
        or token_receipt.get("nonzero_target_tokens_per_arm") != nonzero
        or token_receipt.get("absolute_loss_mass_x5_per_arm") != mass_x5
        or token_receipt.get("skipped_rows") != 0
        or file_receipt.get("sha256") != expected_data_hash
        or file_receipt.get("rows") != EXPECTED_ROWS
        or file_receipt.get("encoded_rows") != EXPECTED_ROWS
        or file_receipt.get("skipped_rows") != 0
        or spans.get("forward") != forward
        or spans.get("nonzero_target") != nonzero
        or spans.get("absolute_loss_mass_x5") != mass_x5
        # Bind the trainer/encoder bytes at train time: the exposure receipt was
        # certified with exactly this encoder, and pin-filling commits happen
        # between arm stages, so an intervening encoder edit must abort.
        or token_receipt.get("encoder_sha256") != sha256_file(TRAINER)
    ):
        parser.error("token receipt does not match the frozen arm exposure")
    try:
        check_base_composite_cheap()
    except ValueError as error:
        parser.error(str(error))
    # The one expensive preflight: the full 9 GB weight hash of the training
    # base. A composite swapped after the cheap receipt check must abort here.
    if sha256_file(MODEL_PATH / "model.safetensors") != MODEL_PATH_WEIGHTS_SHA256:
        parser.error("zero-root training-base composite weights changed")
    prerequisite_arms = None
    if ARM_PREREQUISITES[args.name]:
        try:
            prerequisite_arms = {
                prerequisite: validate_published_arm(prerequisite)
                for prerequisite in ARM_PREREQUISITES[args.name]
            }
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
        "--model-path", str(MODEL_PATH.resolve()),
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
        "token_receipt_sha256": token_sha,
        "train_rows": EXPECTED_ROWS,
        "forward_tokens_per_epoch": forward,
        "nonzero_target_tokens_per_epoch": nonzero,
        "absolute_loss_mass_x5_per_epoch": mass_x5,
        "model_path": {
            "path": str(MODEL_PATH.resolve()),
            "weights_sha256": MODEL_PATH_WEIGHTS_SHA256,
            "merge_receipt_sha256": MODEL_PATH_RECEIPT_SHA256,
            "tree_sha256": MODEL_PATH_TREE_SHA256,
        },
        "fresh_adapter": True,
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
        "prerequisite_arms": prerequisite_arms,
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
    if not validate_adapter_config(load_json(config)):
        preserve_failure(failure_path, common, reason="adapter_rank_alpha_or_base_identity_mismatch", returncode=0)
        raise SystemExit("adapter rank/alpha/base identity mismatch")
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
