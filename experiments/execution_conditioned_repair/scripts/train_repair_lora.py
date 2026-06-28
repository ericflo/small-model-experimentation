#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from repair_experiment.modeling import (  # noqa: E402
    CausalCollator,
    RepairSftDataset,
    attach_lora,
    load_base_model,
    load_jsonl,
    load_tokenizer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--eval", type=Path)
    parser.add_argument("--mode", choices=["final_patch", "no_trace", "trace"], required=True)
    parser.add_argument("--shuffle-traces", action="store_true")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--max-train-records", type=int)
    args = parser.parse_args()

    train_records = load_jsonl(args.train)
    if args.max_train_records:
        train_records = train_records[: args.max_train_records]
    eval_records = load_jsonl(args.eval) if args.eval and args.eval.exists() else []

    tokenizer = load_tokenizer(args.model_id, args.revision)
    model = load_base_model(args.model_id, args.revision, load_in_4bit=True, for_training=True)
    model = attach_lora(model, rank=args.rank, alpha=args.alpha, dropout=args.dropout)
    model.print_trainable_parameters()

    train_dataset = RepairSftDataset(
        train_records,
        tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        shuffle_traces=args.shuffle_traces,
    )
    eval_dataset = RepairSftDataset(
        eval_records,
        tokenizer,
        mode=args.mode,
        max_length=args.max_length,
        shuffle_traces=args.shuffle_traces,
    ) if eval_records else None

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=1,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        remove_unused_columns=False,
        lr_scheduler_type="cosine",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCollator(tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    metadata = {
        "model_id": args.model_id,
        "revision": args.revision,
        "mode": args.mode,
        "shuffle_traces": args.shuffle_traces,
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "lora_rank": args.rank,
        "lora_alpha": args.alpha,
        "lora_dropout": args.dropout,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "gradient_accumulation_steps": args.grad_accum,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "experiment_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
