#!/usr/bin/env python3
"""Low-dose warm-start QLoRA on transition-balanced repository rows."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

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
TRANSITIONS = (
    "start_to_inspect",
    "inspect_to_patch",
    "rejected_patch_to_changed_patch",
    "failed_test_to_diagnose",
    "diagnosis_to_changed_patch",
    "patch_ok_to_verify",
    "passed_test_to_commit",
)


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


def validate_start_checkpoint(path: Path, expected_weight_sha256: str) -> dict:
    config_path = path / "config.json"
    merge_path = path / "merge_receipt.json"
    weights_path = path / "model.safetensors"
    if not (config_path.is_file() and merge_path.is_file() and weights_path.is_file()):
        raise SystemExit(f"incomplete merged start checkpoint: {path}")
    config = json.loads(config_path.read_text())
    text = config.get("text_config", {})
    fingerprint = (
        config.get("model_type") == "qwen3_5"
        and text.get("model_type") == "qwen3_5_text"
        and text.get("vocab_size") == 248320
        and text.get("hidden_size") == 2560
        and text.get("num_hidden_layers") == 32
    )
    merge = json.loads(merge_path.read_text())
    lineage = (
        merge.get("model_lineage", merge.get("base_model")) == MODEL_ID
        and merge.get("model_revision", merge.get("base_revision")) == MODEL_REVISION
    )
    if not fingerprint or not lineage:
        raise SystemExit("start checkpoint is not the registered Qwen/Qwen3.5-4B lineage")
    recorded_weights = {
        item.get("name"): item.get("sha256") for item in merge.get("weight_files", [])
    }
    if recorded_weights.get("model.safetensors") != expected_weight_sha256:
        raise SystemExit("start merge receipt does not name the registered frozen weight hash")
    observed_weight_sha256 = sha256_file(weights_path)
    if observed_weight_sha256 != expected_weight_sha256:
        raise SystemExit(f"start weight hash mismatch: {observed_weight_sha256}")
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256_file(config_path),
        "merge_receipt_sha256": sha256_file(merge_path),
        "weight_sha256": observed_weight_sha256,
        "recorded_weight_files": merge.get("weight_files", []),
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def encode_row(record: dict, tokenizer, max_length: int) -> dict | None:
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
    if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[:len(think_ids)] != think_ids:
        return None
    row_weight = float(record.get("row_weight", 1.0))
    think_weight = float(record.get("think_weight", 0.0))
    weights = (
        [0.0] * len(prompt_ids)
        + [think_weight * abs(row_weight)] * (len(think_ids) - len(prompt_ids))
        + [row_weight] * (len(full_ids) - len(think_ids))
    )
    labels = [token if weight != 0.0 else -100 for token, weight in zip(full_ids, weights)]
    answer_mask = [0.0] * len(think_ids) + [1.0] * (len(full_ids) - len(think_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "loss_weights": weights,
        "answer_mask": answer_mask,
        "operator": record["operator"],
        "transition": record["transition"],
        "family": record["family"],
        "task_id": record["task_id"],
        "row_id": record["id"],
        "think_tokens": len(think_ids) - len(prompt_ids),
        "answer_tokens": len(full_ids) - len(think_ids),
        "weighted_plan_mass": (
            (len(think_ids) - len(prompt_ids)) * think_weight * abs(row_weight)
        ),
        "weighted_action_mass": (len(full_ids) - len(think_ids)) * abs(row_weight),
    }


def make_batches(
    encoded: list[dict], batch_size: int, gradient_accumulation_steps: int, seed: int
) -> tuple[list[dict], dict]:
    """Create complete transition supercycles, padding only whole task blocks."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if gradient_accumulation_steps != len(TRANSITIONS):
        raise ValueError("registered accumulation must equal the seven transition strata")
    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in encoded:
        by_task[str(row["task_id"])].append(row)
    for task_id, rows in by_task.items():
        observed = Counter(row["transition"] for row in rows)
        if len(rows) != len(TRANSITIONS) or observed != Counter(TRANSITIONS):
            raise ValueError(f"task {task_id} does not contain exactly one row per transition")

    original_task_ids = sorted(by_task)
    padding_tasks = (-len(original_task_ids)) % batch_size
    for index in range(padding_tasks):
        source_id = original_task_ids[index % len(original_task_ids)]
        padded_id = f"{source_id}::whole-task-pad-{index}"
        copied = copy.deepcopy(by_task[source_id])
        for row in copied:
            row["task_id"] = padded_id
            row["row_id"] = f"{row['row_id']}::whole-task-pad-{index}"
        by_task[padded_id] = copied

    rng = random.Random(seed)
    chunks_by_transition: dict[str, list[list[dict]]] = {}
    for transition in TRANSITIONS:
        rows = [
            next(row for row in by_task[task_id] if row["transition"] == transition)
            for task_id in sorted(by_task)
        ]
        rng.shuffle(rows)
        transition_chunks = [
            rows[index:index + batch_size]
            for index in range(0, len(rows), batch_size)
        ]
        if any(len(chunk) != batch_size for chunk in transition_chunks):
            raise AssertionError("whole-task padding did not form full microbatches")
        chunks_by_transition[transition] = transition_chunks
    supercycles = []
    chunks_per_transition = len(next(iter(chunks_by_transition.values())))
    for index in range(chunks_per_transition):
        cycle = [chunks_by_transition[transition][index] for transition in TRANSITIONS]
        rng.shuffle(cycle)
        supercycles.append(cycle)
    rng.shuffle(supercycles)
    chunks = [chunk for cycle in supercycles for chunk in cycle]
    if len(chunks) % gradient_accumulation_steps:
        raise AssertionError("transition supercycle is not optimizer-step aligned")
    flattened = [row for chunk in chunks for row in chunk]
    return flattened, {
        "original_tasks": len(original_task_ids),
        "whole_task_padding_duplicates": padding_tasks,
        "effective_tasks_per_epoch": len(by_task),
        "rows_per_epoch": len(flattened),
        "microbatches_per_epoch": len(chunks),
        "optimizer_steps_per_epoch": len(chunks) // gradient_accumulation_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "complete_transition_supercycle": True,
        "every_optimizer_step_contains_all_transitions": True,
        "transition_exposures_per_epoch": {
            transition: len(by_task) for transition in TRANSITIONS
        },
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
                row["input_ids"] + [pad] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "attention_mask": torch.tensor([
                row["attention_mask"] + [0] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "labels": torch.tensor([
                row["labels"] + [-100] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "loss_weights": torch.tensor([
                row["loss_weights"] + [0.0] * (maximum - len(row["input_ids"]))
                for row in features
            ], dtype=torch.float32),
            "answer_mask": torch.tensor([
                row["answer_mask"] + [0.0] * (maximum - len(row["input_ids"]))
                for row in features
            ], dtype=torch.float32),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=["transaction_replay", "replay_only"],
        required=True,
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-weight-sha256", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=7)
    parser.add_argument("--loss-chunk-positions", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    base_receipt = validate_start_checkpoint(
        args.base_model, args.expected_base_weight_sha256
    )
    rows = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("kind") != f"repo_{args.arm}" for row in rows):
        raise SystemExit(f"{args.train} is not a pure {args.arm} bank")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
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

    encoded = [encode_row(record, tokenizer, args.max_length) for record in rows]
    if any(row is None for row in encoded):
        raise SystemExit("a registered row truncates or merges at a token boundary")
    encoded = [row for row in encoded if row is not None]
    ordered, batch_receipt = make_batches(
        encoded, args.batch_size, args.grad_accum, args.seed
    )
    steps_per_epoch = int(batch_receipt["optimizer_steps_per_epoch"])
    max_steps = steps_per_epoch * args.epochs
    encoding_receipt = {
        "schema_version": 1,
        "arm": args.arm,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "start_checkpoint": base_receipt,
        "training_file": {
            "path": str(args.train.resolve()),
            "sha256": sha256_file(args.train),
            "rows": len(rows),
        },
        "encoded_rows": len(encoded),
        "max_encoded_tokens": max(len(row["input_ids"]) for row in encoded),
        "think_tokens": sum(row["think_tokens"] for row in encoded),
        "answer_tokens": sum(row["answer_tokens"] for row in encoded),
        "weighted_plan_mass": sum(row["weighted_plan_mass"] for row in encoded),
        "weighted_action_mass": sum(row["weighted_action_mass"] for row in encoded),
        "operator_rows": dict(Counter(row["operator"] for row in encoded)),
        "transition_rows": dict(Counter(row["transition"] for row in encoded)),
        "epochs": args.epochs,
        "max_steps": max_steps,
        **batch_receipt,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "encoding_receipt.json").write_text(
        json.dumps(encoding_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(encoding_receipt, indent=2), flush=True)
    if args.encode_only:
        return 0
    if args.smoke:
        needed = args.batch_size * args.grad_accum * 2
        ordered = (ordered * ((needed + len(ordered) - 1) // len(ordered)))[:needed]
        max_steps = 2

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
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
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.out),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=max(1, min(10, max_steps // 4)),
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    class TransitionBalancedTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            weights = inputs.pop("loss_weights")
            answer_mask = inputs.pop("answer_mask")
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits[:, :-1, :]
            shifted_labels = labels[:, 1:].contiguous()
            shifted_weights = weights[:, 1:].contiguous()
            shifted_answer_mask = answer_mask[:, 1:].contiguous()
            signed = (shifted_labels != -100).float() * shifted_weights
            numerator = checkpointed_weighted_cross_entropy(
                logits, shifted_labels, signed, args.loss_chunk_positions
            )
            # All arms use the same unweighted action-token denominator.  The
            # reason arm adds only its registered 5% plan numerator.
            denominator = shifted_answer_mask.sum().clamp(min=1.0)
            loss = numerator / denominator
            return (loss, outputs) if return_outputs else loss

    started = time.perf_counter()
    result = TransitionBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=WeightedDataset(ordered),
        data_collator=Collator(tokenizer),
    ).train()
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    receipt = {
        **encoding_receipt,
        "method": "warm_start_transition_balanced_qlora",
        "learning_rate": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps_actual": args.grad_accum,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "optimizer_steps": int(result.global_step),
        "training_loss": float(result.training_loss),
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "normalization": "unweighted action-token count in every arm",
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
