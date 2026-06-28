#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.modeling import CausalCollator, CompilerDataset, attach_lora, load_jsonl, load_model, load_tokenizer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/workspace/.cache/huggingface/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data" / "validation.jsonl")
    parser.add_argument("--out", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_transform_abi_compiler_pilot/lora"))
    parser.add_argument("--steps", type=int, default=90)
    parser.add_argument("--max-length", type=int, default=1536)
    args = parser.parse_args()

    set_seed(17)
    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path, for_training=True, load_in_4bit=True)
    model = attach_lora(model, rank=8, alpha=16, dropout=0.05)
    model.print_trainable_parameters()

    train_records = load_jsonl(args.train)
    val_records = load_jsonl(args.validation)
    train_ds = CompilerDataset(train_records, tokenizer, max_length=args.max_length)
    val_ds = CompilerDataset(val_records, tokenizer, max_length=args.max_length)
    collator = CausalCollator(tokenizer)

    training_args = TrainingArguments(
        output_dir=str(args.out.parent / "trainer_tmp"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=args.steps,
        learning_rate=2e-4,
        warmup_steps=5,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=30,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to=[],
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
    )
    result = trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    metrics = {"train_result": result.metrics, "train_records": len(train_records), "validation_records": len(val_records)}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "train_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
