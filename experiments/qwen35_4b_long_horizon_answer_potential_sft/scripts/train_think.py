#!/usr/bin/env python3
"""Exact-token, weighted long-horizon QLoRA SFT for one frozen arm."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BitsAndBytesConfig,
    Qwen3_5ForCausalLM,
    Trainer,
    TrainingArguments,
)

EXP = Path(__file__).resolve().parents[1]
CONFIG_PATH = EXP / "configs" / "default.yaml"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def exact_encode(rec: dict[str, Any], *, max_length: int, w_think: float) -> dict[str, Any]:
    prompt = [int(value) for value in rec["prompt_token_ids"]]
    trace = [int(value) for value in rec["trace_token_ids"]]
    boundary = [int(value) for value in rec["answer_boundary_token_ids"]]
    answer = [int(value) for value in rec["answer_token_ids"]]
    eos = [int(rec["eos_token_id"])]
    ids = [*prompt, *trace, *boundary, *answer, *eos]
    if len(ids) != int(rec["total_tokens"]):
        raise ValueError(f"stored token count mismatch for {rec['record_id']}")
    if len(ids) > max_length:
        raise ValueError(
            f"selected target exceeds max_length and may not be truncated: "
            f"{rec['record_id']} {len(ids)} > {max_length}"
        )
    weights = (
        [0.0] * len(prompt)
        + [w_think] * len(trace)
        + [1.0] * (len(boundary) + len(answer) + 1)
    )
    labels = [-100 if weight == 0 else token for token, weight in zip(ids, weights)]
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": labels,
        "loss_weights": weights,
    }


class ThinkSftData(Dataset):
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


class Collator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in features)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"] + [self.pad_token_id] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "attention_mask": torch.tensor(
                [row["attention_mask"] + [0] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "labels": torch.tensor(
                [row["labels"] + [-100] * (maximum - len(row["input_ids"])) for row in features]
            ),
            "loss_weights": torch.tensor(
                [row["loss_weights"] + [0.0] * (maximum - len(row["input_ids"])) for row in features],
                dtype=torch.float32,
            ),
        }


def text_checkpoint_key_mapping() -> dict[str, str]:
    """Map the composite checkpoint's language tower into Qwen3_5ForCausalLM."""
    index_path = Path(
        hf_hub_download(
            MODEL_ID,
            "model.safetensors.index.json",
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    )
    keys = json.loads(index_path.read_text(encoding="utf-8"))["weight_map"]
    prefix = "model.language_model."
    mapping = {
        key: f"model.{key[len(prefix):]}"
        for key in keys
        if key.startswith(prefix)
    }
    if len(mapping) < 400:
        raise RuntimeError(f"unexpectedly small Qwen3.5 text key map: {len(mapping)}")
    return mapping


def load_text_model(rank: int, alpha: int, dropout: float) -> Any:
    outer = AutoConfig.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = Qwen3_5ForCausalLM.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        config=outer.text_config,
        key_mapping=text_checkpoint_key_mapping(),
        trust_remote_code=True,
        local_files_only=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET,
        ),
    )
    model.config.use_cache = False
    model.enable_input_require_grads()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if args.arm not in config["sft"]["arms"]:
        raise ValueError(f"unknown frozen arm: {args.arm}")
    dataset = args.dataset or (
        Path(config["artifacts"]["external_root"]) / "sft" / f"{args.arm}.jsonl.gz"
    )
    records = read_jsonl_gz(dataset)
    random.Random(args.seed).shuffle(records)
    epochs = float(config["sft"]["epochs"])
    if args.smoke:
        records = records[:2]
        epochs = 1.0

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = [
        exact_encode(
            row,
            max_length=int(config["sft"]["max_length"]),
            w_think=float(config["sft"]["weight_think"]),
        )
        for row in records
    ]
    # With batch size one, deterministic global length ordering minimizes
    # allocator churn while preserving exactly the same optimizer examples.
    encoded.sort(key=lambda row: len(row["input_ids"]))
    print(
        f"[train] arm={args.arm} rows={len(encoded)} "
        f"tokens={sum(len(row['input_ids']) for row in encoded)} max={max(len(row['input_ids']) for row in encoded)}",
        flush=True,
    )

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = load_text_model(
        int(config["sft"]["rank"]),
        int(config["sft"]["alpha"]),
        float(config["sft"]["dropout"]),
    )
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out),
        num_train_epochs=epochs,
        per_device_train_batch_size=int(config["sft"]["batch_size"]),
        gradient_accumulation_steps=int(config["sft"]["gradient_accumulation"]),
        learning_rate=float(config["sft"]["learning_rate"]),
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    class ExactWeightedTrainer(Trainer):
        def _get_train_sampler(self, *unused_args: Any, **unused_kwargs: Any) -> Any:
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

        def compute_loss(
            self,
            peft_model: Any,
            inputs: dict[str, torch.Tensor],
            return_outputs: bool = False,
            **unused_kwargs: Any,
        ) -> Any:
            weights = inputs.pop("loss_weights")[:, 1:].contiguous()
            labels = inputs.pop("labels")[:, 1:].contiguous()
            causal_model = peft_model.get_base_model()
            outputs = causal_model.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                use_cache=False,
                return_dict=True,
            )
            hidden = outputs.last_hidden_state[:, :-1, :]
            mask = (labels != -100).float() * weights
            denominator = mask.sum().clamp(min=1.0)

            def chunk_numerator(
                chunk_hidden: torch.Tensor,
                chunk_labels: torch.Tensor,
                chunk_mask: torch.Tensor,
            ) -> torch.Tensor:
                logits = causal_model.lm_head(chunk_hidden).float()
                losses = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    chunk_labels.reshape(-1).clamp(min=0),
                    reduction="none",
                ).view_as(chunk_labels)
                return (losses * chunk_mask).sum()

            numerator = hidden.new_zeros((), dtype=torch.float32)
            for start in range(0, hidden.shape[1], 128):
                stop = min(hidden.shape[1], start + 128)
                numerator = numerator + checkpoint(
                    chunk_numerator,
                    hidden[:, start:stop, :],
                    labels[:, start:stop],
                    mask[:, start:stop],
                    use_reentrant=False,
                )
            loss = numerator / denominator
            return (loss, outputs) if return_outputs else loss

    trainer = ExactWeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ThinkSftData(encoded),
        data_collator=Collator(int(tokenizer.pad_token_id)),
    )
    train_result = trainer.train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    artifacts = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
        for path in sorted(args.out.iterdir())
        if path.is_file()
    }
    receipt = {
        "schema_version": 1,
        "arm": args.arm,
        "seed": args.seed,
        "smoke": args.smoke,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "rows": len(encoded),
        "forward_tokens": sum(len(row["input_ids"]) for row in encoded),
        "supervised_weighted_tokens": sum(sum(row["loss_weights"]) for row in encoded),
        "skipped_rows": 0,
        "epochs": epochs,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "train_metrics": train_result.metrics,
        "artifacts": artifacts,
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[train] saved {args.arm} to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
