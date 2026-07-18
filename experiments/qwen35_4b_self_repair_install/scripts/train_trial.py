#!/usr/bin/env python3
"""Fail-closed wrapper for the lifecycle-33 self-repair installation trial.

ONE arm trains: ``self_repair`` — a FRESH rank-32/alpha-64 QLoRA adapter (NO
warm start) with the model weights loaded from the pinned ``base_reserialized``
composite (the reserialized base Qwen/Qwen3.5-4B, tree ``26d8ee48…``, weights
``b654e033…``) via the vendored trainer's ``--model-path``, over the FRESH
504-row self-repair curriculum ``data/sft_self_repair.jsonl`` (sha
``920cb228…``) with the frozen recipe (epochs 1, lr 1e-5, batch 1, grad-accum
8, max-length 4096, w_think 0.2, w_close 0.2) at the fixed training seed 91331 —
63 optimizer steps. The tokenizer stays hub-pinned; the composite's tokenizer
files are hash-verified against the pins below.

This cell trains from BASE in ONE stage, so the lineage package is just the
base_reserialized provenance (the IN-CELL sha-pinned copy
``data/provenance/base_reserialized.json``) + the self-repair curriculum + the
fixed-seed recipe + the vendored trainer/merger. The base composite
authenticates FAIL-CLOSED pre-training: the in-cell provenance copy, the cheap
per-file receipt/tokenizer checks, the full-tree manifest sha, AND the full
9 GB weights hash.
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
TRAINER_SHA256 = "e0eca2a230dae5d109d418dcb4cc19af05882a770af14350ffd741a8d5e90f01"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
COMPUTE_REVIEW = EXP / "reports" / "compute_review.md"

# Training base: the reserialized BASE composite (a directory of full bf16
# weights), NOT an adapter. The one fresh rank-32 adapter trains on top of it.
MODEL_PATH = (
    ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum" / "merged" / "base_reserialized"
)
MODEL_PATH_TREE_SHA256 = "26d8ee48583adb0fb557d0ff668664949adff0068fa5baafe6f0af68e22fb677"
MODEL_PATH_WEIGHTS_SHA256 = "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db"
MODEL_PATH_WEIGHTS_SIZE_BYTES = 9_078_620_536
MODEL_PATH_RECEIPT_SHA256 = "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f"
MODEL_PATH_TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
# Complete flat file set of the merged composite (identical algorithm to the
# shared benchmark runner's merged_tree_manifest -> tree_manifest_sha256).
MERGED_FILE_NAMES = frozenset(
    {
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "merge_receipt.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
)
MERGED_FILE_SHA256 = {
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "config.json": "a1c80f0efa6f83f631eaa9c25ffa166e3b1f9db395cc3b14374dfc0962261f60",
    "generation_config.json": "0c46d8aa4f0ae5e611c961f70b87c83fb696043c1e319337708e96f882180de1",
    "merge_receipt.json": "25aee794cfffe4d58110defc61177edef1f5324e47deb28fbd3cb7ccd61ae54f",
    "model.safetensors": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
    "tokenizer_config.json": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
}
# The IN-CELL sha-pinned base provenance copy (the hard fail-closed gate; the
# composite's own inner merge_receipt.json, copied here for standalone
# reproducibility).
BASE_PROVENANCE_COPY = EXP / "data" / "provenance" / "base_reserialized.json"

LORA_RANK = 32
LORA_ALPHA = 64
LORA_TARGET_MODULES = (
    "down_proj", "gate_proj", "k_proj", "o_proj", "q_proj", "up_proj", "v_proj",
)
EXPECTED_ROWS = 504
TRAINING_SEED = 91331
OPTIMIZER_STEPS = 63  # 504 rows / (batch 1 x grad-accum 8)
ARM_FILES = {
    "self_repair": (
        EXP / "data" / "sft_self_repair.jsonl",
        "920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb",
    ),
}
PUBLISHED_ARM_HASHES = {
    # TODO-PIN: after the arm trains and its receipt/log/adapter are published,
    # the orchestrator replaces the None below with the SINGLE-LINE sorted-key
    # dict of the four published sha256 values:
    # {"adapter_config": "<64hex>", "adapter_weights": "<64hex>",
    #  "log": "<64hex>", "receipt": "<64hex>"}.
    "self_repair": {"adapter_config": "702360585faa42545551734b58bc8c0b0347f5ad92a9b1ec27c96b1745dae686", "adapter_weights": "fe84c983c2188483ebb99ae1887c1f2fbe87065c17fdf5d0b6b0d66849bee91f", "log": "d6b469d2bd9c23b55a40310898fd1d2f865c6ab1c8b63f1fcd7079a4bbcde0cc", "receipt": "90b33c8e1ce528f10a0d0dc72da07ae510d96a97f50fe7d3f6e82efe1b67b66a"},
}
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"


def require_pin(value, name: str):
    if value is None:
        raise SystemExit(
            f"frozen constant {name} is unpinned (TODO-PIN); the orchestrator must "
            "fill it before this stage runs"
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
    return subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


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
        "seed": TRAINING_SEED,
        "optimizer_steps": OPTIMIZER_STEPS,
    }


def merged_tree_manifest(output: Path) -> list[dict]:
    """Hash the complete flat merged composite tree; reject surprises."""
    if not output.is_dir() or output.is_symlink():
        raise ValueError(f"base composite is not a real directory: {output}")
    children = sorted(output.iterdir(), key=lambda p: p.name)
    if any(p.is_symlink() or not p.is_file() for p in children):
        raise ValueError("base composite contains a symlink or non-file entry")
    names = {p.name for p in children}
    if names != MERGED_FILE_NAMES:
        raise ValueError(
            f"base composite file set changed: missing={sorted(MERGED_FILE_NAMES - names)}, "
            f"unexpected={sorted(names - MERGED_FILE_NAMES)}"
        )
    return [{"name": p.name, "size": p.stat().st_size, "sha256": sha256_file(p)} for p in children]


def tree_manifest_sha256(manifest: list[dict]) -> str:
    rendered = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(rendered).hexdigest()


def check_base_provenance() -> None:
    """The IN-CELL sha-pinned base provenance copy must pin exactly the base."""
    if (
        not BASE_PROVENANCE_COPY.is_file()
        or sha256_file(BASE_PROVENANCE_COPY) != MODEL_PATH_RECEIPT_SHA256
    ):
        raise ValueError(f"in-cell base provenance copy is absent or changed: {BASE_PROVENANCE_COPY}")
    payload = load_json(BASE_PROVENANCE_COPY)
    weight_files = {row.get("name"): (row.get("sha256"), row.get("size_bytes")) for row in payload.get("weight_files", [])}
    if (
        payload.get("method") != "pinned_base_composite_reserialization"
        or payload.get("model_lineage") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("tokenizer_sha256") != MODEL_PATH_TOKENIZER_SHA256
        or weight_files.get("model.safetensors") != (MODEL_PATH_WEIGHTS_SHA256, MODEL_PATH_WEIGHTS_SIZE_BYTES)
    ):
        raise ValueError("base provenance copy does not describe the frozen base_reserialized composite")


def check_base_composite_cheap() -> None:
    """Per-file receipt/tokenizer/size checks (no full weights hash)."""
    for name, expected in MERGED_FILE_SHA256.items():
        path = MODEL_PATH / name
        if not path.is_file():
            raise ValueError(f"base composite missing {name}")
        if name == "model.safetensors":
            if path.stat().st_size != MODEL_PATH_WEIGHTS_SIZE_BYTES:
                raise ValueError("base composite weights size changed")
            continue
        if sha256_file(path) != expected:
            raise ValueError(f"base composite {name} changed")


def authenticate_base_fail_closed(*, full_weights: bool) -> dict:
    """Fail-closed base authentication. ``full_weights`` runs the 9 GB hash."""
    check_base_provenance()
    check_base_composite_cheap()
    manifest = merged_tree_manifest(MODEL_PATH)
    observed_tree = tree_manifest_sha256(manifest)
    if observed_tree != MODEL_PATH_TREE_SHA256:
        raise ValueError(f"base composite tree changed: {observed_tree}")
    if full_weights:
        if sha256_file(MODEL_PATH / "model.safetensors") != MODEL_PATH_WEIGHTS_SHA256:
            raise ValueError("base composite weights hash changed")
    return {
        "path": str(MODEL_PATH.resolve()),
        "tree_sha256": observed_tree,
        "weights_sha256": MODEL_PATH_WEIGHTS_SHA256,
        "receipt_sha256": MODEL_PATH_RECEIPT_SHA256,
        "provenance_copy_sha256": MODEL_PATH_RECEIPT_SHA256,
    }


def validate_adapter_config(config: dict) -> bool:
    return (
        config.get("r") == LORA_RANK
        and config.get("lora_alpha") == LORA_ALPHA
        and config.get("base_model_name_or_path") == str(MODEL_PATH.resolve())
        and sorted(config.get("target_modules") or ()) == list(LORA_TARGET_MODULES)
    )


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
    published = PUBLISHED_ARM_HASHES.get(name)
    if published is None and require_committed:
        raise ValueError(
            f"PUBLISHED_ARM_HASHES[{name!r}] is unfilled (TODO-PIN); pin the published "
            "receipt/log/adapter hashes before the next stage"
        )
    data_path, data_hash = ARM_FILES[name]
    dataset = payload.get("dataset", {})
    base = payload.get("base_composite", {})
    config = adapter / "adapter_config.json"
    weights = adapter / "adapter_model.safetensors"
    if (
        payload.get("experiment_id") != EXP.name
        or payload.get("name") != name
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("returncode") != 0
        or payload.get("adapter_complete") is not True
        or payload.get("trainer_sha256") != TRAINER_SHA256
        or payload.get("train_rows") != EXPECTED_ROWS
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or payload.get("hyperparameters") != expected_hyperparameters()
        or dataset.get("path") != str(data_path.resolve())
        or dataset.get("sha256") != data_hash
        or dataset.get("rows") != EXPECTED_ROWS
        or base.get("path") != str(MODEL_PATH.resolve())
        or base.get("tree_sha256") != MODEL_PATH_TREE_SHA256
        or base.get("weights_sha256") != MODEL_PATH_WEIGHTS_SHA256
        or base.get("receipt_sha256") != MODEL_PATH_RECEIPT_SHA256
        or payload.get("fresh_adapter") is not True
        or payload.get("log_sha256") != sha256_file(log_path)
        or Path(payload.get("adapter", "")).resolve() != adapter.resolve()
        or not config.is_file()
        or not weights.is_file()
        or payload.get("adapter_config_sha256") != sha256_file(config)
        or payload.get("adapter_weights_sha256") != sha256_file(weights)
        or not validate_adapter_config(load_json(config))
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
            json.dumps({**common, "returncode": returncode, "failure_reason": reason},
                       indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--name", choices=tuple(ARM_FILES), required=True)
    parser.add_argument("--train", type=Path, required=True)
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
    parser.add_argument("--seed", type=int, default=TRAINING_SEED)
    args = parser.parse_args()

    expected_data, expected_data_hash = ARM_FILES[args.name]
    expected_out = ADAPTER_ROOT / args.name
    if args.train.resolve() != expected_data.resolve():
        parser.error("training data does not match the frozen arm")
    if args.out.resolve() != expected_out.resolve():
        parser.error("adapter output does not match the frozen large-artifact path")
    if args.model_path.resolve() != MODEL_PATH.resolve():
        parser.error("model path does not match the frozen base_reserialized composite")
    observed = {
        "epochs": args.epochs, "lr": args.lr, "rank": args.rank, "alpha": args.alpha,
        "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_length": args.max_length, "w_think": args.w_think, "w_close": args.w_close,
        "seed": args.seed, "optimizer_steps": OPTIMIZER_STEPS,
    }
    if observed != expected_hyperparameters():
        parser.error("training hyperparameters differ from the frozen review")
    rows = sum(1 for line in expected_data.read_text(encoding="utf-8").splitlines() if line.strip())
    if (
        not expected_data.is_file()
        or sha256_file(expected_data) != expected_data_hash
        or rows != EXPECTED_ROWS
        or not TRAINER.is_file()
        or sha256_file(TRAINER) != TRAINER_SHA256
        or not COMPUTE_REVIEW.is_file()
        or "**Verdict:** `PASS_CONTROL_TRAINING`." not in COMPUTE_REVIEW.read_text(encoding="utf-8")
    ):
        parser.error("training pool, trainer, or compute-review authorization changed")
    try:
        base_auth = authenticate_base_fail_closed(full_weights=True)
    except ValueError as error:
        parser.error(str(error))

    log_path = EXP / "runs" / "training" / f"{args.name}.log"
    receipt_path = EXP / "runs" / "training" / f"{args.name}.json"
    failure_path = EXP / "runs" / "training" / f"{args.name}.failure.json"
    if any(p.exists() for p in (expected_out, log_path, receipt_path, failure_path)):
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
            command, cwd=ROOT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                 "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
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
        "dataset": {"path": str(expected_data.resolve()), "sha256": expected_data_hash, "rows": EXPECTED_ROWS},
        "train_rows": EXPECTED_ROWS,
        "trainer_sha256": TRAINER_SHA256,
        "base_composite": base_auth,
        "fresh_adapter": True,
        "hyperparameters": observed,
        "command": command,
        "wall_seconds": elapsed,
        "log": str(log_path.resolve()),
        "log_sha256": sha256_file(log_path),
        "preflight_git_head": head,
        "preflight_git_branch": branch,
        "preflight_origin_main": origin,
        "preflight_git_status": status,
        "post_training_git_status": run_text(["git", "status", "--short"]),
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
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
