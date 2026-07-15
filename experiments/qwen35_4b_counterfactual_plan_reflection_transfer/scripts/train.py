#!/usr/bin/env python3
"""Frozen QLoRA trainer for counterfactual plan-reflection arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import time
from pathlib import Path

if sys.flags.no_site != 1:
    raise SystemExit("training must start with the pinned interpreter and -I -B -S")

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from runtime_contract import (  # noqa: E402
    bind_active_cuda_identity,
    bootstrap_runtime_environment,
    require_detached_execution_worktree,
    runtime_metadata,
)

bootstrap_runtime_environment(EXP.parents[1], "training")

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Qwen2Tokenizer,
    Trainer,
    TrainingArguments,
)

from firewall import install_benchmark_firewall  # noqa: E402

install_benchmark_firewall(EXP.parents[1])

from records import (  # noqa: E402
    TRAINING_ARMS,
    build_training_records,
    encode_training_record,
    validate_tokenized_parity,
)
from taskgen import build_corpus  # noqa: E402
from stages import read_and_validate_stage_receipt  # noqa: E402
from merge_replay import authenticate_base_snapshot, base_snapshot_commitment  # noqa: E402
from load_window_guard import LoadWindowGuard, validate_load_window_receipt  # noqa: E402
from provenance import (  # noqa: E402
    validate_runtime_bootstrap,
    validate_runtime_packages,
)
from tokenizer_lineage import (  # noqa: E402
    authenticate_closed_tokenizer_view,
    authenticate_tokenizer_snapshot,
    ensure_closed_tokenizer_view,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_value(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_tree(path: Path, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


class EncodedDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, rows: list[dict]) -> dict[str, torch.Tensor]:
        length = max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] + [self.pad_token_id] * (length - len(row["input_ids"])) for row in rows]
            ),
            "attention_mask": torch.tensor(
                [row["attention_mask"] + [0] * (length - len(row["attention_mask"])) for row in rows]
            ),
            "labels": torch.tensor(
                [row["labels"] + [-100] * (length - len(row["labels"])) for row in rows]
            ),
            "loss_weights": torch.tensor(
                [row["loss_weights"] + [0.0] * (length - len(row["loss_weights"])) for row in rows]
            ),
        }


class FixedOrderTrainer(Trainer):
    def _get_train_sampler(self, *unused_args, **unused_kwargs):
        return SequentialSampler(self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, **unused_kwargs):
        weights = inputs.pop("loss_weights")
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        shift_labels = labels[:, 1:].contiguous()
        shift_weights = weights[:, 1:].contiguous()
        losses = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            shift_labels.reshape(-1).clamp(min=0),
            reduction="none",
        ).view_as(shift_labels)
        mask = (shift_labels != -100).float() * shift_weights
        loss = (losses * mask).sum() / mask.sum().clamp(min=1.0)
        return (loss, outputs) if return_outputs else loss


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=TRAINING_ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--stage-receipt", type=Path, required=True)
    parser.add_argument("--tokenizer-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["training"] is not True:
        raise SystemExit("training is not authorized by the committed config")
    allowed_seeds = set(config["training"]["staged_seeds"].values())
    if args.seed not in allowed_seeds:
        raise SystemExit(f"seed {args.seed} is not preregistered")
    expected_stage = (
        "screen_training"
        if args.seed == config["training"]["staged_seeds"]["screen"]
        else "replication_training"
    )
    read_and_validate_stage_receipt(
        args.stage_receipt,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    positive = config["training"]["positive_control"]["arm"]
    if args.arm == positive and args.seed != config["training"]["staged_seeds"]["screen"]:
        raise SystemExit("the noncomparable positive control has no replication-seed authorization")
    if args.output.exists():
        raise SystemExit(f"output already exists: {args.output}")
    worktree = require_detached_execution_worktree(EXP.parents[1])
    trainer_git_commit = worktree["git_commit"]
    lock_path = EXP.parents[1] / "requirements-training.lock.txt"
    base_root, _base_index, _base_structure = authenticate_base_snapshot()
    base_snapshot = base_snapshot_commitment(base_root)
    tokenizer_path, tokenizer_snapshot = ensure_closed_tokenizer_view()

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
    model_config = config["model"]
    expected_tokenizer_content = {"tokenizer": tokenizer_snapshot}
    with LoadWindowGuard(
        [tokenizer_path], expected_content=expected_tokenizer_content
    ) as tokenizer_guard:
        before_tokenizer_content = {
            "tokenizer": authenticate_closed_tokenizer_view(tokenizer_path)
        }
        tokenizer = Qwen2Tokenizer.from_pretrained(
            str(tokenizer_path),
            trust_remote_code=False,
            local_files_only=True,
        )
        after_tokenizer_content = {
            "tokenizer": authenticate_closed_tokenizer_view(tokenizer_path)
        }
        tokenizer_guard.bind_authenticated_content(
            before_tokenizer_content, after_tokenizer_content
        )
    tokenizer_load_guard = tokenizer_guard.receipt
    if tokenizer_load_guard is None:
        raise RuntimeError("training tokenizer load-window guard emitted no receipt")
    if (
        authenticate_tokenizer_snapshot() != tokenizer_snapshot
        or authenticate_closed_tokenizer_view(tokenizer_path) != tokenizer_snapshot
    ):
        raise ValueError("tokenizer files changed across training tokenizer initialization")
    if int(tokenizer.eos_token_id) != 248046:
        raise ValueError(f"unexpected tokenizer EOS: {tokenizer.eos_token_id}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    recipe = config["training"]["recipe"]
    loss = config["training"]["loss_weights"]
    encoded = {
        arm: [
            encode_training_record(
                row,
                tokenizer,
                max_length=int(recipe["max_length"]),
                think_weight=float(loss["thought"]),
                close_weight=float(loss["autonomous_close"]),
            )
            for row in rows
        ]
        for arm, rows in arms.items()
    }
    parity = validate_tokenized_parity(arms, encoded)
    live_row_receipts = {
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
    expected_tokenizer_receipt = json.loads(args.tokenizer_receipt.read_text())
    tokenizer_receipt_keys = {
        "schema_version", "experiment_id", "config_sha256", "runner_sha256",
        "model_id", "model_revision", "tokenizer_class", "tokenizer_eos_token_id",
        "trust_remote_code", "tokenizer_snapshot", "worktree", "record_receipt",
        "parity", "rows", "rows_sha256", "model_calls", "gpu_events",
        "benchmark_reads", "load_window_guard", "runtime_bootstrap",
    }
    if (
        set(expected_tokenizer_receipt) != tokenizer_receipt_keys
        or expected_tokenizer_receipt.get("schema_version") != 5
        or expected_tokenizer_receipt.get("experiment_id") != config["experiment_id"]
        or expected_tokenizer_receipt.get("config_sha256") != _sha256_file(config_path)
        or expected_tokenizer_receipt.get("runner_sha256")
        != _sha256_file(EXP / "scripts" / "tokenizer_receipt.py")
        or expected_tokenizer_receipt.get("model_id") != model_config["id"]
        or expected_tokenizer_receipt.get("model_revision") != model_config["revision"]
        or expected_tokenizer_receipt.get("tokenizer_eos_token_id") != 248046
        or expected_tokenizer_receipt.get("tokenizer_class") != "Qwen2Tokenizer"
        or expected_tokenizer_receipt.get("trust_remote_code") is not False
        or expected_tokenizer_receipt.get("tokenizer_snapshot") != tokenizer_snapshot
        or expected_tokenizer_receipt.get("load_window_guard") is None
        or expected_tokenizer_receipt.get("worktree") != worktree
        or expected_tokenizer_receipt.get("model_calls") != 0
        or expected_tokenizer_receipt.get("gpu_events") != 0
        or expected_tokenizer_receipt.get("benchmark_reads") != 0
    ):
        raise ValueError("tokenizer receipt identity is invalid")
    validate_load_window_receipt(
        expected_tokenizer_receipt["load_window_guard"],
        [tokenizer_path],
        expected_content=expected_tokenizer_content,
    )
    validate_runtime_bootstrap(
        {
            "bootstrap": expected_tokenizer_receipt["runtime_bootstrap"],
            "worktree": worktree,
        },
        EXP.parents[1],
        "training",
    )
    if expected_tokenizer_receipt["parity"] != parity:
        raise ValueError("live token/mask parity differs from the prerequisite receipt")
    if expected_tokenizer_receipt["record_receipt"] != record_receipt:
        raise ValueError("live training records differ from the prerequisite receipt")
    if (
        expected_tokenizer_receipt["rows"] != live_row_receipts
        or expected_tokenizer_receipt["rows_sha256"]
        != _sha256_value(live_row_receipts)
    ):
        raise ValueError("live input/target/mask rows differ from tokenizer receipt")

    args.output.mkdir(parents=True, exist_ok=False)
    copied_tokenizer_receipt = args.output / "source_tokenizer_receipt.json"
    copied_stage_receipt = args.output / "source_stage_receipt.json"
    shutil.copyfile(args.tokenizer_receipt, copied_tokenizer_receipt)
    shutil.copyfile(args.stage_receipt, copied_stage_receipt)
    (args.output / "STARTED.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "arm": args.arm,
                "seed": args.seed,
                "config_sha256": _sha256_file(config_path),
                "tokenizer_receipt_sha256": _sha256_file(args.tokenizer_receipt),
                "stage_receipt_sha256": _sha256_file(args.stage_receipt),
                "trainer_git_commit": trainer_git_commit,
                "trainer_sha256": _sha256_file(Path(__file__).resolve()),
                "worktree": worktree,
                "runtime_pending": True,
                "base_snapshot": base_snapshot,
                "tokenizer_snapshot": tokenizer_snapshot,
                "tokenizer_load_window_guard": tokenizer_load_guard,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    load_started = time.perf_counter()
    expected_model_content = {"base": base_snapshot}
    with LoadWindowGuard(
        [base_root], expected_content=expected_model_content
    ) as model_guard:
        before_model_content = {"base": base_snapshot_commitment(base_root)}
        model = AutoModelForCausalLM.from_pretrained(
            str(base_root),
            local_files_only=True,
            trust_remote_code=False,
            device_map="cuda",
            dtype=torch.bfloat16,
            quantization_config=quantization,
            attn_implementation="sdpa",
        )
        after_model_content = {"base": base_snapshot_commitment(base_root)}
        model_guard.bind_authenticated_content(
            before_model_content, after_model_content
        )
    model_load_guard = model_guard.receipt
    if model_load_guard is None:
        raise RuntimeError("training model load-window guard emitted no receipt")
    torch.cuda.synchronize()
    model_load_seconds = time.perf_counter() - load_started
    if base_snapshot_commitment(base_root) != base_snapshot:
        raise ValueError("base checkpoint files changed across training model initialization")
    active_gpu_identity = bind_active_cuda_identity(EXP.parents[1], torch)
    model.config._name_or_path = model_config["id"]
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(recipe["lora_rank"]),
            lora_alpha=int(recipe["lora_alpha"]),
            lora_dropout=float(recipe["lora_dropout"]),
            bias=str(recipe["lora_bias"]),
            task_type="CAUSAL_LM",
            target_modules=list(recipe["target_modules"]),
        ),
    )
    model.config.use_cache = False
    arguments = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=float(recipe["epochs"]),
        per_device_train_batch_size=int(recipe["per_device_batch"]),
        gradient_accumulation_steps=int(recipe["gradient_accumulation"]),
        learning_rate=float(recipe["learning_rate"]),
        lr_scheduler_type=str(recipe["scheduler"]),
        warmup_ratio=float(recipe["warmup_ratio"]),
        weight_decay=float(recipe["weight_decay"]),
        max_grad_norm=float(recipe["max_grad_norm"]),
        bf16=True,
        tf32=False,
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        optim=str(recipe["optimizer"]),
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_drop_last=False,
    )
    trainer = FixedOrderTrainer(
        model=model,
        args=arguments,
        train_dataset=EncodedDataset(encoded[args.arm]),
        data_collator=Collator(int(tokenizer.pad_token_id)),
    )
    torch.cuda.synchronize()
    training_started = time.perf_counter()
    outcome = trainer.train()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - training_started
    expected_steps = int(schedule["optimizer_steps_total"])
    if int(outcome.global_step) != expected_steps:
        raise ValueError(f"expected {expected_steps} optimizer steps, got {outcome.global_step}")
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    training_runtime = runtime_metadata(
        EXP.parents[1], lock_path, active_gpu_identity
    )
    validate_runtime_packages(
        training_runtime, lock_path, required_backend="training"
    )
    epochs = int(recipe["epochs"])
    forward_tokens = sum(len(row["input_ids"]) for row in encoded[args.arm]) * epochs
    training_receipt = {
        "schema_version": 6,
        "experiment_id": config["experiment_id"],
        "arm": args.arm,
        "seed": args.seed,
        "model_id": model_config["id"],
        "model_revision": model_config["revision"],
        "optimizer_steps": int(outcome.global_step),
        "train_loss": float(outcome.training_loss),
        "config_sha256": _sha256_file(config_path),
        "tokenizer_receipt_sha256": _sha256_file(args.tokenizer_receipt),
        "stage_receipt_sha256": _sha256_file(args.stage_receipt),
        "copied_tokenizer_receipt_sha256": _sha256_file(copied_tokenizer_receipt),
        "copied_stage_receipt_sha256": _sha256_file(copied_stage_receipt),
        "trainer_git_commit": trainer_git_commit,
        "trainer_sha256": _sha256_file(Path(__file__).resolve()),
        "recipe_sha256": hashlib.sha256(
            json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "record_receipt_sha256": hashlib.sha256(
            json.dumps(record_receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "parity_sha256": hashlib.sha256(
            json.dumps(parity, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "adapter_tree_excluding_training_receipt_sha256": _sha256_tree(args.output),
        "worktree": worktree,
        "runtime": training_runtime,
        "base_snapshot": base_snapshot,
        "tokenizer_snapshot": tokenizer_snapshot,
        "load_window_guards": {
            "tokenizer": tokenizer_load_guard,
            "model": model_load_guard,
        },
        "compute": {
            "schema_version": 1,
            "amortization_horizon": "full_training_charged_to_each_confirmation_split",
            "epochs": epochs,
            "forward_tokens": forward_tokens,
            # Conservative checkpoint-aware charge: one original forward,
            # two backward-equivalent passes, and one recomputed forward.
            "forward_backward_multiplier": 4,
            "token_forward_equivalents": forward_tokens * 4,
            "model_load_seconds": model_load_seconds,
            "training_seconds": training_seconds,
            "gpu_phase_wall_seconds": model_load_seconds + training_seconds,
        },
    }
    receipt_path = args.output / "training_receipt.json"
    receipt_path.write_text(json.dumps(training_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(training_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
