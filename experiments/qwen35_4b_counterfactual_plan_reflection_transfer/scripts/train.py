#!/usr/bin/env python3
"""Frozen QLoRA trainer for counterfactual plan-reflection arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset, SequentialSampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    git_status = subprocess.run(
        ["git", "status", "--porcelain"], check=True, capture_output=True, text=True
    ).stdout
    if git_status:
        raise SystemExit("training requires a clean worktree")
    trainer_git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        trust_remote_code=True,
        use_fast=True,
    )
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
    expected_tokenizer_receipt = json.loads(args.tokenizer_receipt.read_text())
    if (
        expected_tokenizer_receipt.get("experiment_id") != config["experiment_id"]
        or expected_tokenizer_receipt.get("model_id") != model_config["id"]
        or expected_tokenizer_receipt.get("model_revision") != model_config["revision"]
        or expected_tokenizer_receipt.get("tokenizer_eos_token_id") != 248046
        or expected_tokenizer_receipt.get("model_calls") != 0
        or expected_tokenizer_receipt.get("gpu_events") != 0
        or expected_tokenizer_receipt.get("benchmark_reads") != 0
    ):
        raise ValueError("tokenizer receipt identity is invalid")
    if expected_tokenizer_receipt["parity"] != parity:
        raise ValueError("live token/mask parity differs from the prerequisite receipt")
    if expected_tokenizer_receipt["record_receipt"] != record_receipt:
        raise ValueError("live training records differ from the prerequisite receipt")

    args.output.mkdir(parents=True, exist_ok=False)
    copied_tokenizer_receipt = args.output / "source_tokenizer_receipt.json"
    copied_stage_receipt = args.output / "source_stage_receipt.json"
    shutil.copyfile(args.tokenizer_receipt, copied_tokenizer_receipt)
    shutil.copyfile(args.stage_receipt, copied_stage_receipt)
    (args.output / "STARTED.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "arm": args.arm,
                "seed": args.seed,
                "config_sha256": _sha256_file(config_path),
                "tokenizer_receipt_sha256": _sha256_file(args.tokenizer_receipt),
                "stage_receipt_sha256": _sha256_file(args.stage_receipt),
                "trainer_git_commit": trainer_git_commit,
                "trainer_sha256": _sha256_file(Path(__file__).resolve()),
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
    model = AutoModelForCausalLM.from_pretrained(
        model_config["id"],
        revision=model_config["revision"],
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        attn_implementation="sdpa",
    )
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
    outcome = trainer.train()
    expected_steps = int(schedule["optimizer_steps_total"])
    if int(outcome.global_step) != expected_steps:
        raise ValueError(f"expected {expected_steps} optimizer steps, got {outcome.global_step}")
    model.save_pretrained(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    training_receipt = {
        "schema_version": 2,
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
    }
    receipt_path = args.output / "training_receipt.json"
    receipt_path.write_text(json.dumps(training_receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(training_receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
