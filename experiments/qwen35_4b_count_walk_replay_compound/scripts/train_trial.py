#!/usr/bin/env python3
"""Fail-closed wrapper for the frozen stage-8 replay-compound QLoRA trial.

ONE arm trains: ``replay_compound`` — a FRESH rank-32/alpha-64 adapter (NO
warm start exists in this cell) with the model weights loaded from the
frozen ``count_walk`` merged composite (lifecycle 27's stage-7 candidate,
tree ``d5fdc55c…``, weights ``ddd7bc4b…``) via the per-experiment trainer's
``--model-path`` argument, over the FULL 2,240-row replay pool
``data/sft_blend.jsonl`` (sha ``25a9595f…``) with the chain's established
replay-refresh recipe (epochs 1, lr 1e-5, batch 1, grad-accum 8, max-length
4096, w_think 0.2, w_close 0.2) at the fixed fresh training seed 86. The
tokenizer stays hub-pinned; the composite's tokenizer files are
hash-verified against the pins below. The parent composite authenticates
FAIL-CLOSED pre-training: the IN-CELL sha-pinned provenance copy of
lifecycle 27's merge receipt (the committed sibling original is a
verification aid — byte-identical when present, skipped with a recorded
note when absent), the cheap receipt/tokenizer/size checks, and finally
the full 9 GB weights hash.
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
# Training base: the COUNT_WALK MERGED COMPOSITE (a directory of full bf16
# weights), NOT an adapter. The one fresh rank-32 adapter trains on top of
# it and no warm start is accepted anywhere in this cell.
MODEL_PATH = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_count_dont_walk_enumeration"
    / "merged"
    / "count_walk"
)
MODEL_PATH_WEIGHTS_SHA256 = (
    "ddd7bc4b5b8f4f2393996148bcb1b411a8be4d7f03430babe789b3534b9850a3"
)
# The composite's INNER merge_receipt.json (inside the directory).
MODEL_PATH_RECEIPT_SHA256 = (
    "3c432f110fe96a508d6a75ab34e4a649671a3d7b2d942f3346cab609bef437d7"
)
MODEL_PATH_TREE_SHA256 = (
    "d5fdc55c0238ffbe2465bd73a5f9d63f442ad4083ff9eb477c9887e15e3da6b1"
)
MODEL_PATH_WEIGHTS_SIZE_BYTES = 9_078_620_536
MODEL_PATH_TOKENIZER_SHA256 = (
    "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
)
MODEL_PATH_TOKENIZER_CONFIG_SHA256 = (
    "bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87"
)
# The parent's lineage anchor: this cell's IN-CELL sha-pinned provenance
# copy is the hard fail-closed gate. The COMMITTED lifecycle-27 sibling
# receipt is a verification aid only — byte-identical when present, skipped
# with a recorded note when absent; the reproduction path is this cell's
# own lineage package.
PARENT_COMMITTED_MERGE_RECEIPT = (
    ROOT / "experiments" / "qwen35_4b_count_dont_walk_enumeration"
    / "runs" / "merges" / "count_walk.json"
)
PARENT_COMMITTED_MERGE_RECEIPT_SHA256 = (
    "840edca0638b9e291bb34fde28b4b530df8743faf9b7b18b7f2358ce55ec4c36"
)
PARENT_PROVENANCE_COPY = EXP / "data" / "provenance" / "count_walk_merge.json"
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
EXPECTED_ROWS = 2240
TRAINING_SEED = 86
OPTIMIZER_STEPS = 280  # 2240 rows / (batch 1 x grad-accum 8)
# Documented exposure of the full replay pool under this exact encoder
# (measured by lifecycle 27's committed data/source_token_lengths.json with
# encoder sha e0eca2a2… — identical to TRAINER_SHA256; max forward 3,193
# tokens < max-length 4,096, so zero skips are expected and enforced).
DOCUMENTED_EXPOSURE = {
    "forward_tokens_per_epoch": 2_136_805,
    "nonzero_target_tokens_per_epoch": 893_282,
    "absolute_loss_mass_x5_per_epoch": 973_034,
    "max_forward_tokens": 3_193,
    "measured_by": (
        "experiments/qwen35_4b_count_dont_walk_enumeration/data/"
        "source_token_lengths.json (encoder sha e0eca2a2…, identical to this "
        "cell's scripts/train_think.py)"
    ),
}
ARM_FILES = {
    "replay_compound": (
        EXP / "data" / "sft_blend.jsonl",
        "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
    ),
}
PUBLISHED_ARM_HASHES = {
    # TODO-PIN: after the arm trains and its receipt/log/adapter are
    # published, the orchestrator replaces the None ON THIS LINE with the
    # SINGLE-LINE sorted-key dict of the four published sha256 values:
    # {"adapter_config": "<64hex>", "adapter_weights": "<64hex>",
    # "log": "<64hex>", "receipt": "<64hex>"} — check_design.py's
    # normalized-hash pin canonicalizes exactly this value slot.
    "replay_compound": {"adapter_config": "d1625592e38be2776e95b1e96ab308676222b02717dda271012828a8f189009f", "adapter_weights": "259c460bd2531a09f28099624fe1792908f09136045cb2752caebf48760849c9", "log": "a5effabe9d5c849f92d6746ca1396b3162241db94bef547466c88a48afafd0f0", "receipt": "6967a0becddb1d14fba04655cbc8a939f217c4716ebda504d3a0c4c0bd2bf9cc"},
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
        "seed": TRAINING_SEED,
        "optimizer_steps": OPTIMIZER_STEPS,
    }


def check_parent_provenance() -> str:
    """The IN-CELL sha-pinned provenance copy must pin exactly the parent.

    The in-cell copy is the hard fail-closed gate. The committed lifecycle-27
    sibling original is a VERIFICATION AID only: byte-identical when present
    (divergence fails loudly as tamper evidence), skipped with a recorded
    note when absent — never the reproduction path (that is this cell's own
    lineage package, rebuild_lineage.py). Returns the sibling status note
    recorded in the training receipt.
    """
    if (
        not PARENT_PROVENANCE_COPY.is_file()
        or sha256_file(PARENT_PROVENANCE_COPY)
        != PARENT_COMMITTED_MERGE_RECEIPT_SHA256
    ):
        raise ValueError(
            "in-cell count_walk parent provenance copy is absent or changed: "
            f"{PARENT_PROVENANCE_COPY}"
        )
    if PARENT_COMMITTED_MERGE_RECEIPT.is_file():
        if (
            PARENT_COMMITTED_MERGE_RECEIPT.read_bytes()
            != PARENT_PROVENANCE_COPY.read_bytes()
        ):
            raise ValueError(
                "committed count_walk sibling merge receipt diverged from the "
                f"in-cell provenance pin: {PARENT_COMMITTED_MERGE_RECEIPT}"
            )
        sibling_note = "present, byte-identical to the in-cell pin"
    else:
        sibling_note = "absent, in-cell pin authoritative"
    payload = load_json(PARENT_PROVENANCE_COPY)
    if (
        payload.get("experiment_id") != "qwen35_4b_count_dont_walk_enumeration"
        or payload.get("name") != "count_walk"
        or payload.get("model_id") != MODEL_ID
        or payload.get("model_revision") != MODEL_REVISION
        or Path(payload.get("merged", "")).resolve() != MODEL_PATH.resolve()
        or payload.get("output_tree_sha256") != MODEL_PATH_TREE_SHA256
        or {
            row.get("name"): row.get("sha256")
            for row in payload.get("weight_files", [])
        }
        != {"model.safetensors": MODEL_PATH_WEIGHTS_SHA256}
        or payload.get("merge_receipt_sha256") != MODEL_PATH_RECEIPT_SHA256
    ):
        raise ValueError(
            "count_walk parent merge receipt does not describe the frozen parent arm"
        )
    return sibling_note


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
        raise ValueError("count_walk training-base composite changed")


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
        # Fail closed like the eval's tree pins: once the arm's receipt is
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
        or payload.get("trainer_sha256") != TRAINER_SHA256
        or payload.get("train_rows") != EXPECTED_ROWS
        or payload.get("skipped_rows") != 0
        or payload.get("preflight_git_status") != ""
        or payload.get("hyperparameters") != expected_hyperparameters()
        or payload.get("documented_exposure") != DOCUMENTED_EXPOSURE
        or dataset.get("path") != str(data_path.resolve())
        or dataset.get("sha256") != data_hash
        or dataset.get("rows") != EXPECTED_ROWS
        or model_path.get("path") != str(MODEL_PATH.resolve())
        or model_path.get("weights_sha256") != MODEL_PATH_WEIGHTS_SHA256
        or model_path.get("merge_receipt_sha256") != MODEL_PATH_RECEIPT_SHA256
        or model_path.get("tree_sha256") != MODEL_PATH_TREE_SHA256
        or model_path.get("committed_merge_receipt_sha256")
        != PARENT_COMMITTED_MERGE_RECEIPT_SHA256
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
    expected_data_hash = require_pin(expected_data_hash, f"ARM_FILES[{args.name!r}]")
    expected_out = ADAPTER_ROOT / args.name
    if args.train.resolve() != expected_data.resolve():
        parser.error("training data does not match the frozen arm")
    if args.out.resolve() != expected_out.resolve():
        parser.error("adapter output does not match the frozen large-artifact path")
    if args.model_path.resolve() != MODEL_PATH.resolve():
        parser.error("model path does not match the frozen count_walk composite")
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
        "optimizer_steps": OPTIMIZER_STEPS,
    }
    if observed_hyperparameters != expected_hyperparameters():
        parser.error("training hyperparameters differ from the frozen review")
    rows = sum(
        1
        for line in expected_data.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if (
        not expected_data.is_file()
        or sha256_file(expected_data) != expected_data_hash
        or rows != EXPECTED_ROWS
        or not TRAINER.is_file()
        or sha256_file(TRAINER) != TRAINER_SHA256
        or not COMPUTE_REVIEW.is_file()
        or "**Verdict:** `PASS_CONTROL_TRAINING`."
        not in COMPUTE_REVIEW.read_text(encoding="utf-8")
    ):
        parser.error(
            "training pool, trainer, or compute-review authorization changed"
        )
    try:
        parent_sibling_note = check_parent_provenance()
        check_base_composite_cheap()
    except ValueError as error:
        parser.error(str(error))
    # The one expensive preflight: the full 9 GB weight hash of the training
    # base. A composite swapped after the cheap receipt check must abort here.
    if sha256_file(MODEL_PATH / "model.safetensors") != MODEL_PATH_WEIGHTS_SHA256:
        parser.error("count_walk training-base composite weights changed")

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
        },
        "train_rows": EXPECTED_ROWS,
        "documented_exposure": DOCUMENTED_EXPOSURE,
        "trainer_sha256": TRAINER_SHA256,
        "model_path": {
            "path": str(MODEL_PATH.resolve()),
            "weights_sha256": MODEL_PATH_WEIGHTS_SHA256,
            "merge_receipt_sha256": MODEL_PATH_RECEIPT_SHA256,
            "tree_sha256": MODEL_PATH_TREE_SHA256,
            "committed_merge_receipt_sha256": (
                PARENT_COMMITTED_MERGE_RECEIPT_SHA256
            ),
            "sibling_original": parent_sibling_note,
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
