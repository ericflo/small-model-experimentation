#!/usr/bin/env python3
"""QLoRA think-channel SFT on verified gauntlet examples (multi-turn aware).

Recipe copied from qwen35_4b_hypothesize_verify_wall/scripts/train_think.py
(C45/C48 line): r32/a64/dropout .05 on all 7 proj modules, nf4 double-quant,
lr 2e-4 cosine, bs 2 x grad-accum 8, paged_adamw_8bit, loss on the assistant
target only. Adapted here to multi-turn chat contexts: each JSONL row carries
`messages` (context ending with a user turn) plus `think`/`answer` targets.
Rows that would truncate at max_length are SKIPPED (never silently cut a
target mid-answer). Run under the repo .venv.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"  # matches src/vllm_runner.py pin
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode_row(rec: dict, tok, max_length: int, w_think: float) -> dict | None:
    prompt = tok.apply_chat_template(
        rec["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    think_part = rec["think"].strip() + "\n</think>\n\n"
    answer_part = rec["answer"].strip() + tok.eos_token
    pid = tok(prompt, add_special_tokens=False)["input_ids"]
    mid = tok(prompt + think_part, add_special_tokens=False)["input_ids"]
    fid = tok(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
    if len(fid) > max_length:
        return None
    if fid[: len(pid)] != pid or fid[: len(mid)] != mid:
        return None  # tokenizer merge across a boundary; skip rather than mislabel
    # Forced-close recovery examples condition on a truncated chain; the chain
    # is context, not target behavior.
    row_w_think = 0.0 if rec["kind"].endswith("_fc") else w_think
    weights = (
        [0.0] * len(pid)
        + [row_w_think] * (len(mid) - len(pid))
        + [1.0] * (len(fid) - len(mid))
    )
    labels = [
        -100 if weight == 0.0 else token
        for token, weight in zip(fid, weights)
    ]
    return {
        "input_ids": fid,
        "attention_mask": [1] * len(fid),
        "labels": labels,
        "loss_weights": weights,
    }


class ThinkSftData(Dataset):
    def __init__(self, encoded):
        self.rows = encoded

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Collator:
    def __init__(self, tok):
        self.tok = tok

    def __call__(self, feats):
        m = max(len(f["input_ids"]) for f in feats)
        pad = self.tok.pad_token_id
        return {
            "input_ids": torch.tensor([f["input_ids"] + [pad] * (m - len(f["input_ids"])) for f in feats]),
            "attention_mask": torch.tensor([f["attention_mask"] + [0] * (m - len(f["input_ids"])) for f in feats]),
            "labels": torch.tensor([f["labels"] + [-100] * (m - len(f["input_ids"])) for f in feats]),
            "loss_weights": torch.tensor(
                [f["loss_weights"] + [0.0] * (m - len(f["input_ids"])) for f in feats]
            ),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train", type=Path, nargs="+",
                    default=[EXP / "data" / "sft_round1.jsonl"],
                    help="one or more SFT JSONL files (later rounds train from "
                         "BASE on the union of all rounds' data)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=3072)
    ap.add_argument("--w-think", type=float, default=0.2,
                    help="loss weight on think tokens (answer/action tokens are 1.0; "
                         "recovery-arm think is always 0.0)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    recs = []
    for train_path in args.train:
        recs += [json.loads(l) for l in train_path.read_text().splitlines() if l.strip()]
    random.Random(args.seed).shuffle(recs)
    if args.smoke:
        recs = recs[:8]
        args.epochs = 1.0

    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                        trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
    assert probe.endswith("<think>\n"), f"unexpected think-template tail: {probe[-40:]!r}"

    encoded, skipped = [], 0
    for rec in recs:
        rec.setdefault("kind", "atom")
        row = encode_row(rec, tok, args.max_length, args.w_think)
        if row is None:
            skipped += 1
        else:
            encoded.append(row)
    print(f"[train_think] {len(encoded)} examples ({skipped} skipped as over-length)", flush=True)
    if not encoded:
        raise SystemExit("no trainable examples")

    # Length-bucketed batching (this transformers version dropped
    # group_by_length): sort by length, chunk into microbatches, shuffle the
    # chunk order deterministically, then train with a sequential sampler.
    # Without this, mixed-length rounds pad most microbatches to ~max_length
    # and step time triples.
    encoded.sort(key=lambda row: len(row["input_ids"]))
    chunks = [encoded[i:i + args.batch_size] for i in range(0, len(encoded), args.batch_size)]
    random.Random(args.seed).shuffle(chunks)
    encoded = [row for chunk in chunks for row in chunk]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True, device_map="cuda",
        dtype=torch.bfloat16, quantization_config=bnb, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGET))
    model.config.use_cache = False
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr,
        lr_scheduler_type="cosine", warmup_ratio=0.03, bf16=True, logging_steps=10,
        save_strategy="no", report_to=[], gradient_checkpointing=True,
        optim="paged_adamw_8bit", seed=args.seed,
        remove_unused_columns=False)  # keep loss_weights for the collator

    class BucketOrderTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            from torch.utils.data import SequentialSampler

            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
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

    BucketOrderTrainer(model=model, args=targs, train_dataset=ThinkSftData(encoded),
                       data_collator=Collator(tok)).train()
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[train_think] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
