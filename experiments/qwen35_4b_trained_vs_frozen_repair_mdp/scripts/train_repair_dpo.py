#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, attach_new_lora, load_quant_model, load_tokenizer  # noqa: E402


class PreferenceDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def encode(self, prompt: str, response: str) -> dict[str, list[int]]:
        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        target_ids = self.tokenizer(response + self.tokenizer.eos_token, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        if len(input_ids) > self.max_length:
            overflow = len(input_ids) - self.max_length
            input_ids = input_ids[overflow:]
            labels = labels[overflow:]
        return {"input_ids": input_ids, "labels": labels}

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {
            "chosen": self.encode(row["prompt"], row["chosen"]),
            "rejected": self.encode(row["prompt"], row["rejected"]),
        }


def pad_rows(rows: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in rows)
    input_ids = []
    labels = []
    attention_mask = []
    for row in rows:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        attention_mask.append([1] * len(row["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "chosen": pad_rows([row["chosen"] for row in batch], pad_token_id),
        "rejected": pad_rows([row["rejected"] for row in batch], pad_token_id),
    }


def sequence_logprob(model: Any, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    logits = out.logits[:, :-1, :]
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_logps = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    token_logps = token_logps * mask
    denom = mask.sum(dim=1).clamp_min(1)
    return token_logps.sum(dim=1) / denom


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-out", type=Path, required=True)
    parser.add_argument("--method", type=str, default="repair_dpo")
    parser.add_argument("--base-adapter-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.loss_out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.train)
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no DPO rows")

    tokenizer = load_tokenizer(args.model_path)
    model = load_quant_model(args.model_path, for_training=True)
    if args.base_adapter_dir:
        model = attach_existing_lora(model, args.base_adapter_dir, is_trainable=True)
    else:
        model = attach_new_lora(model)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    dataset = PreferenceDataset(rows, tokenizer, args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))

    losses: list[dict[str, float]] = []
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc=args.method)
    model.train()
    while step < args.max_steps:
        for batch in loader:
            chosen = {key: value.to(model.device) for key, value in batch["chosen"].items()}
            rejected = {key: value.to(model.device) for key, value in batch["rejected"].items()}
            chosen_logp = sequence_logprob(model, chosen)
            rejected_logp = sequence_logprob(model, rejected)
            margin = chosen_logp - rejected_logp
            loss = -F.logsigmoid(args.beta * margin).mean() / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu()) * args.grad_accum
                margin_value = float(margin.detach().mean().cpu())
                losses.append({"step": step, "loss": loss_value, "margin": margin_value})
                progress.update(1)
                if step % 20 == 0:
                    progress.write(json.dumps({"step": step, "loss": round(loss_value, 4), "margin": round(margin_value, 4)}))
                if step >= args.max_steps:
                    break
    progress.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "method": args.method,
        "train_examples": len(rows),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "base_adapter_dir": str(args.base_adapter_dir) if args.base_adapter_dir else None,
        "losses": losses,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    args.loss_out.write_text(json.dumps(losses, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
