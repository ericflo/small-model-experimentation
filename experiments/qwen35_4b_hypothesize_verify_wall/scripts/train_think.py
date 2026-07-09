#!/usr/bin/env python3
"""QLoRA reasoning-SFT of hypothesize-and-verify traces into the THINK channel (dsl_sft arm).

Data: traces.jsonl of {"prompt", "think", "answer"} -- the trace trains inside <think>...</think>
and the final ```python code block is the post-think answer (bank_the_thoughts train_lora_think
pattern; a stated recipe change vs C45, which trained the no-think content channel). The qwen3_5
chat template with enable_thinking=True already ends the generation prompt with '<think>\\n', so
the target is: trace + '\\n</think>\\n\\n' + answer + EOS. C45 recipe: r32/a64, bs 2, grad-accum 8.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                          Trainer, TrainingArguments)

EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class ThinkSftData(Dataset):
    def __init__(self, records, tok, max_length=2048):
        self.r, self.tok, self.max = records, tok, max_length

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        rec = self.r[i]
        msgs = [{"role": "user", "content": rec["prompt"]}]
        prompt = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=True)
        target = rec["think"].strip() + "\n</think>\n\n" + rec["answer"].strip() + self.tok.eos_token
        pid = self.tok(prompt, add_special_tokens=False, truncation=True, max_length=self.max)["input_ids"]
        fid = self.tok(prompt + target, add_special_tokens=False, truncation=True, max_length=self.max)["input_ids"]
        labels = fid.copy()
        labels[: min(len(pid), len(labels))] = [-100] * min(len(pid), len(labels))
        return {"input_ids": fid, "attention_mask": [1] * len(fid), "labels": labels}


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
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, default=EXP / "data" / "traces.jsonl")
    ap.add_argument("--out", type=Path, default=EXP / "runs" / "lora_dsl")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in args.train.read_text().splitlines() if l.strip()]
    if args.smoke:
        recs = recs[:8]
        args.epochs = 1.0
    print(f"[train_think] training on {len(recs)} think-channel traces", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    probe = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                    add_generation_prompt=True, enable_thinking=True)
    assert probe.endswith("<think>\n"), f"unexpected think-template tail: {probe[-40:]!r}"

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="cuda", dtype=torch.bfloat16,
        quantization_config=bnb, attn_implementation="sdpa")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.05,
                                             bias="none", task_type="CAUSAL_LM", target_modules=TARGET))
    model.config.use_cache = False
    model.print_trainable_parameters()

    targs = TrainingArguments(
        output_dir=str(args.out), num_train_epochs=args.epochs, per_device_train_batch_size=2,
        gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, lr_scheduler_type="cosine",
        warmup_ratio=0.03, bf16=True, logging_steps=5, save_strategy="no", report_to=[],
        gradient_checkpointing=True, optim="paged_adamw_8bit", seed=args.seed)
    Trainer(model=model, args=targs, train_dataset=ThinkSftData(recs, tok, args.max_length),
            data_collator=Collator(tok)).train()
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[train_think] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
