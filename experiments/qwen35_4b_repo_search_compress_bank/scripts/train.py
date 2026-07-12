#!/usr/bin/env python3
"""Matched-step QLoRA for C54 replay plus operator-balanced repository rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

# The long-target Qwen logits allocation otherwise leaves large unusable CUDA
# fragments after checkpoint preparation on 48 GB devices.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
OPERATORS = ("INSPECT", "PATCH", "VERIFY", "COMMIT")


def checkpointed_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    signed_weights: torch.Tensor,
    chunk_positions: int,
) -> torch.Tensor:
    """Exact weighted CE without retaining a full-vocabulary FP32 temporary."""
    if chunk_positions < 1:
        raise ValueError("chunk_positions must be positive")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_weights = signed_weights.reshape(-1)

    def chunk_loss(
        chunk_logits: torch.Tensor,
        chunk_labels: torch.Tensor,
        chunk_weights: torch.Tensor,
    ) -> torch.Tensor:
        losses = torch.nn.functional.cross_entropy(
            chunk_logits, chunk_labels.clamp(min=0), reduction="none"
        )
        return (losses * chunk_weights).sum()

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_logits.size(0), chunk_positions):
        end = min(start + chunk_positions, flat_logits.size(0))
        # Reentrant checkpoint runs the expensive softmax under no_grad and
        # recomputes it chunk-by-chunk during backward, avoiding the 9.5 GiB
        # all-position FP32 allocation that exceeded the 48 GB device.
        total = total + checkpoint(
            chunk_loss,
            flat_logits[start:end],
            flat_labels[start:end],
            flat_weights[start:end],
            use_reentrant=True,
        )
    return total


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def encode_row(record: dict, tokenizer, max_length: int, default_think_weight: float) -> dict | None:
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    think_part = record["think"].strip() + "\n</think>\n\n"
    answer_part = record["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    think_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        return None
    if full_ids[: len(prompt_ids)] != prompt_ids or full_ids[: len(think_ids)] != think_ids:
        return None

    kind = str(record.get("kind", "atom"))
    row_weight = float(record.get("row_weight", 1.0))
    think_weight = float(record.get("think_weight", default_think_weight))
    if kind.endswith("_fc") or row_weight < 0:
        think_weight = 0.0
    weights = (
        [0.0] * len(prompt_ids)
        + [think_weight * abs(row_weight)] * (len(think_ids) - len(prompt_ids))
        + [row_weight] * (len(full_ids) - len(think_ids))
    )
    labels = [token if weight != 0.0 else -100 for token, weight in zip(full_ids, weights)]
    answer_mask = (
        [0.0] * len(think_ids)
        + [1.0] * (len(full_ids) - len(think_ids))
    )
    source = "repo" if kind.startswith("repo_") else "apex"
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "loss_weights": weights,
        "answer_mask": answer_mask,
        "source_code": 1 if source == "repo" else 0,
        "source": source,
        "operator": record.get("operator"),
        "task_id": record.get("task_id"),
        "row_id": record.get("id"),
        "target_tokens": sum(label != -100 for label in labels),
        "absolute_weight_mass": sum(abs(weight) for weight in weights),
    }


def make_batches(
    encoded: list[dict], batch_size: int, gradient_accumulation_steps: int, seed: int
) -> tuple[list[dict], dict]:
    """Keep source-homogeneous microbatches and complete operator task blocks."""
    if batch_size not in (2, 4):
        raise ValueError("supported registered microbatch sizes are 2 and 4")
    apex = [row for row in encoded if row["source"] == "apex"]
    repo = [row for row in encoded if row["source"] == "repo"]
    apex.sort(key=lambda row: len(row["input_ids"]))
    effective_batch = batch_size * gradient_accumulation_steps
    apex_padding = (-len(apex)) % effective_batch
    if apex and apex_padding:
        apex.extend(copy.deepcopy(apex[index % len(apex)]) for index in range(apex_padding))
    chunks = [apex[index:index + batch_size] for index in range(0, len(apex), batch_size)]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in repo:
        by_task[str(row["task_id"])].append(row)
    for task_id in sorted(by_task):
        group = by_task[task_id]
        operators = Counter(row["operator"] for row in group)
        if len(group) != 4 or operators != Counter(OPERATORS):
            raise ValueError(
                f"repository task {task_id} is not one-row-per-operator: {dict(operators)}"
            )
        ordered = [next(row for row in group if row["operator"] == operator) for operator in OPERATORS]
        chunks.extend(
            ordered[index:index + batch_size]
            for index in range(0, len(ordered), batch_size)
        )
    random.Random(seed).shuffle(chunks)
    flattened = [row for chunk in chunks for row in chunk]
    return flattened, {
        "apex_rows": len(apex) - apex_padding,
        "apex_padding_duplicates": apex_padding,
        "repository_rows": len(repo),
        "repository_tasks": len(by_task),
        "microbatches_per_epoch": len(chunks),
        "repository_microbatches_per_epoch": len(repo) // batch_size,
        "homogeneous_batches": True,
        "four_operator_repository_task_blocks": True,
        "effective_batch_size": effective_batch,
    }


class WeightedDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in features)
        pad = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([
                row["input_ids"] + [pad] * (maximum - len(row["input_ids"])) for row in features
            ]),
            "attention_mask": torch.tensor([
                row["attention_mask"] + [0] * (maximum - len(row["input_ids"])) for row in features
            ]),
            "labels": torch.tensor([
                row["labels"] + [-100] * (maximum - len(row["input_ids"])) for row in features
            ]),
            "loss_weights": torch.tensor([
                row["loss_weights"] + [0.0] * (maximum - len(row["input_ids"])) for row in features
            ], dtype=torch.float32),
            "answer_mask": torch.tensor([
                row["answer_mask"] + [0.0] * (maximum - len(row["input_ids"])) for row in features
            ], dtype=torch.float32),
            "source_code": torch.tensor([row["source_code"] for row in features]),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["apex_replay", "action_only", "compact"], required=True)
    parser.add_argument("--train", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=584)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--loss-chunk-positions", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--w-think", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stress-longest", action="store_true")
    args = parser.parse_args()

    records = []
    source_files = []
    for path in args.train:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        records.extend(rows)
        source_files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(rows)})
    if args.arm == "apex_replay" and any(str(row.get("kind", "")).startswith("repo_") for row in records):
        raise SystemExit("apex_replay cannot contain repository rows")
    if args.arm in ("compact", "action_only") and not any(
        str(row.get("kind", "")).startswith("repo_") for row in records
    ):
        raise SystemExit(f"{args.arm} requires repository rows")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    if not probe.endswith("<think>\n"):
        raise SystemExit(f"unexpected thinking template tail: {probe[-40:]!r}")

    encoded = []
    skipped = Counter()
    for record in records:
        row = encode_row(record, tokenizer, args.max_length, args.w_think)
        source = "repo" if str(record.get("kind", "")).startswith("repo_") else "apex"
        if row is None:
            skipped[source] += 1
        else:
            encoded.append(row)
    if skipped["repo"]:
        raise SystemExit(f"repository rows would truncate or merge at token boundary: {skipped['repo']}")
    if not encoded:
        raise SystemExit("no trainable rows")
    ordered, batch_receipt = make_batches(
        encoded, args.batch_size, args.grad_accum, args.seed
    )
    microbatches_required = args.max_steps * args.grad_accum
    effective_epochs = microbatches_required / batch_receipt["microbatches_per_epoch"]
    encoding_receipt = {
        "schema_version": 1,
        "arm": args.arm,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "training_files": source_files,
        "input_rows": len(records),
        "encoded_rows": len(encoded),
        "skipped_by_source": dict(skipped),
        "max_encoded_tokens": max(len(row["input_ids"]) for row in encoded),
        "target_tokens_by_source": {
            source: sum(row["target_tokens"] for row in encoded if row["source"] == source)
            for source in ("apex", "repo")
        },
        "absolute_weight_mass_by_source": {
            source: sum(row["absolute_weight_mass"] for row in encoded if row["source"] == source)
            for source in ("apex", "repo")
        },
        "repository_operator_rows": dict(Counter(
            row["operator"] for row in encoded if row["source"] == "repo"
        )),
        "max_steps": args.max_steps,
        "gradient_accumulation_steps": args.grad_accum,
        "microbatches_required": microbatches_required,
        "effective_dataset_epochs": effective_epochs,
        **batch_receipt,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "encoding_receipt.json").write_text(
        json.dumps(encoding_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(encoding_receipt, indent=2), flush=True)
    if args.encode_only:
        return 0
    if args.stress_longest:
        ordered = sorted(encoded, key=lambda row: len(row["input_ids"]), reverse=True)[:8]
        if len(ordered) != 8 or len({row["source"] for row in ordered}) != 1:
            raise SystemExit("long-target stress requires eight rows from one source")
        args.max_steps = 2
        args.grad_accum = 1
    elif args.smoke:
        ordered = ordered[:8]
        args.max_steps = 2
        args.grad_accum = 1

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, device_map="cuda",
        dtype=torch.bfloat16, quantization_config=quantization, attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05, bias="none",
            task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        remove_unused_columns=False,
    )

    class OperatorBalancedTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            weights = inputs.pop("loss_weights")
            answer_mask = inputs.pop("answer_mask")
            labels = inputs.pop("labels")
            source_codes = inputs.pop("source_code")
            if torch.unique(source_codes).numel() != 1:
                raise RuntimeError("mixed apex/repository microbatch violates registered normalization")
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:].contiguous()
            shifted_weights = weights[:, 1:].contiguous()
            shifted_answer_mask = answer_mask[:, 1:].contiguous()
            active = (shifted_labels != -100).float()
            signed = active * shifted_weights
            numerator = checkpointed_weighted_cross_entropy(
                logits, shifted_labels, signed, args.loss_chunk_positions
            )
            if int(source_codes[0].item()) == 0:
                denominator = signed.abs().sum().clamp(min=1.0)  # exact C54 objective
            else:
                # The compact and action-only controls share the same action
                # denominator. Compact plan supervision is the only added
                # gradient, rather than accidentally diluting action dose.
                denominator = shifted_answer_mask.sum().clamp(min=1.0)
            loss = numerator / denominator
            return (loss, outputs) if return_outputs else loss

    started = time.perf_counter()
    result = OperatorBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=WeightedDataset(ordered),
        data_collator=Collator(tokenizer),
    ).train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    receipt = {
        **encoding_receipt,
        "method": "operator_balanced_repo_plus_c54_qlora",
        "learning_rate": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps_actual": args.grad_accum,
        "default_think_weight": args.w_think,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "optimizer_steps": int(result.global_step),
        "training_loss": float(result.training_loss),
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "normalization": {
            "apex": "absolute signed loss-weight mass (C54 replay)",
            "repository": "unweighted answer-token count (same action denominator in both controls)",
        },
        "loss_chunk_positions": args.loss_chunk_positions,
        "loss_implementation": "exact_sequence_chunked_checkpointed_cross_entropy",
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[train] saved {args.arm} adapter to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
