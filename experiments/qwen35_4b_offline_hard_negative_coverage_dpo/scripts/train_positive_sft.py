#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_new_lora, load_quant_model, load_tokenizer  # noqa: E402


class SFTDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, list[int]]] = []
        seen: set[tuple[str, str]] = set()
        eos = tokenizer.eos_token or ""
        for row in rows:
            key = (row["record_id"], row["chosen"])
            if key in seen:
                continue
            seen.add(key)
            prompt_ids = tokenizer(row["prompt"], add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(row["chosen"].rstrip() + "\n" + eos, add_special_tokens=False)["input_ids"]
            if len(target_ids) > max_length - 8:
                target_ids = target_ids[: max_length - 8]
            prompt_ids = prompt_ids[-(max_length - len(target_ids)) :]
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids
            if not input_ids or all(label == -100 for label in labels):
                continue
            self.items.append({"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


def collate(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = []
    labels = []
    attention_mask = []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        labels.append(row["labels"] + [-100] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.pairs)
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no SFT rows")

    tokenizer = load_tokenizer(args.model_path, padding_side="right")
    model = attach_new_lora(
        load_quant_model(args.model_path, for_training=True),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()
    dataset = SFTDataset(rows, tokenizer, args.max_length)
    if not dataset:
        raise RuntimeError("empty tokenized SFT dataset")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses: list[dict[str, float]] = []
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc=args.run_name)
    model.train()
    while step < args.max_steps:
        for batch in loader:
            batch = {key: value.to(model.device) for key, value in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                loss_value = float(loss.detach().cpu()) * args.grad_accum
                losses.append({"step": step, "loss": loss_value})
                progress.update(1)
                if step % 10 == 0:
                    progress.write(json.dumps({"step": step, "loss": round(loss_value, 4)}))
                if step >= args.max_steps:
                    break
    progress.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "run_name": args.run_name,
        "pairs_in": str(args.pairs),
        "pair_rows": len(rows),
        "tokenized_rows": len(dataset),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": args.max_length,
        "losses": losses,
    }
    write_json(args.metrics_out, metadata)
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

