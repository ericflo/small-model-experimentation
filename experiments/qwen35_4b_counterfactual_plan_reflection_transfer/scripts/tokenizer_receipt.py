#!/usr/bin/env python3
"""Render every training arm and emit the prerequisite token/mask parity receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import yaml
from transformers import Qwen2Tokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402
from runtime_contract import require_detached_execution_worktree  # noqa: E402
from load_window_guard import LoadWindowGuard  # noqa: E402
from tokenizer_lineage import (  # noqa: E402
    authenticate_closed_tokenizer_view,
    authenticate_tokenizer_snapshot,
    ensure_closed_tokenizer_view,
)

install_benchmark_firewall(EXP.parents[1])

from records import (  # noqa: E402
    TRAINING_ARMS,
    build_training_records,
    encode_training_record,
    validate_tokenized_parity,
)
from taskgen import build_corpus  # noqa: E402


def _sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _write_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["tokenizer"] is not True:
        raise SystemExit("tokenizer stage is not authorized by the committed config")
    worktree = require_detached_execution_worktree(EXP.parents[1])
    construction = config["construction"]
    counts = {
        split: int(construction["per_family"][split])
        for split in ("train", "calibration", "qualification", "confirmation")
    }
    corpus = build_corpus(counts, int(construction["seed"]))
    schedule = config["training"]["schedule"]
    arms, record_receipt = build_training_records(
        corpus["train"],
        shuffle_seed=int(construction["shuffle_seed"]),
        schedule_seed=int(construction["schedule_seed"]),
        per_family_per_step=int(schedule["per_family_per_optimizer_group"]),
    )
    model = config["model"]
    tokenizer_path, tokenizer_snapshot = ensure_closed_tokenizer_view(
        ensure_downloaded=True
    )
    with LoadWindowGuard([tokenizer_path]) as load_guard:
        tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=False,
            local_files_only=True,
        )
    tokenizer_load_guard = load_guard.receipt
    if tokenizer_load_guard is None:
        raise RuntimeError("tokenizer load-window guard emitted no receipt")
    if (
        authenticate_tokenizer_snapshot() != tokenizer_snapshot
        or authenticate_closed_tokenizer_view(tokenizer_path) != tokenizer_snapshot
    ):
        raise ValueError("tokenizer files changed across tokenizer initialization")
    if int(tokenizer.eos_token_id) != 248046:
        raise ValueError(f"unexpected tokenizer EOS: {tokenizer.eos_token_id}")
    recipe = config["training"]["recipe"]
    weights = config["training"]["loss_weights"]
    encoded = {
        arm: [
            encode_training_record(
                row,
                tokenizer,
                max_length=int(recipe["max_length"]),
                think_weight=float(weights["thought"]),
                close_weight=float(weights["autonomous_close"]),
            )
            for row in arms[arm]
        ]
        for arm in TRAINING_ARMS
    }
    parity = validate_tokenized_parity(arms, encoded)
    row_receipts = {
        arm: [
            {
                "task_id": record["task_id"],
                "optimizer_group": record["optimizer_group"],
                "prompt_tokens": tokenized["prompt_tokens"],
                "target_tokens": tokenized["target_tokens"],
                "think_target_tokens": tokenized["think_target_tokens"],
                "close_target_tokens": tokenized["close_target_tokens"],
                "answer_target_tokens": tokenized["answer_target_tokens"],
                "input_ids_sha256": tokenized["input_ids_sha256"],
                "target_ids_sha256": tokenized["target_ids_sha256"],
                "mask_sha256": tokenized["mask_sha256"],
            }
            for record, tokenized in zip(arms[arm], encoded[arm], strict=True)
        ]
        for arm in TRAINING_ARMS
    }
    receipt = {
        "schema_version": 3,
        "experiment_id": config["experiment_id"],
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "model_id": model["id"],
        "model_revision": model["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_eos_token_id": int(tokenizer.eos_token_id),
        "trust_remote_code": False,
        "tokenizer_snapshot": tokenizer_snapshot,
        "load_window_guard": tokenizer_load_guard,
        "worktree": worktree,
        "record_receipt": record_receipt,
        "parity": parity,
        "rows": row_receipts,
        "rows_sha256": _sha256(row_receipts),
        "model_calls": 0,
        "gpu_events": 0,
        "benchmark_reads": 0,
    }
    _write_exclusive(args.output, receipt)
    print(json.dumps({key: receipt[key] for key in ("rows_sha256", "parity")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
