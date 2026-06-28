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
from src.model_utils import DEFAULT_MODEL_PATH, action_token_ids, attach_new_lora, last_token_action_logits, load_quant_model, load_tokenizer  # noqa: E402


class ActionDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return {"prompt": row["prompt"], "label": int(row["label_index"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--loss-out", type=Path, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--letters", type=str, default="A,B")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.loss_out.parent.mkdir(parents=True, exist_ok=True)
    letters = [item.strip() for item in args.letters.split(",") if item.strip()]

    rows = load_jsonl(args.train)
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise RuntimeError("no training rows")

    tokenizer = load_tokenizer(args.model_path)
    model = attach_new_lora(load_quant_model(args.model_path, for_training=True))
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loader = DataLoader(ActionDataset(rows), batch_size=args.batch_size, shuffle=True)

    losses: list[dict[str, float]] = []
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc=args.method)
    model.train()
    while step < args.max_steps:
        for batch in loader:
            prompts = list(batch["prompt"])
            labels = torch.as_tensor(batch["label"], dtype=torch.long, device=model.device)
            logits = last_token_action_logits(model, tokenizer, prompts, letters=letters, max_length=args.max_length)
            loss = F.cross_entropy(logits.float(), labels) / args.grad_accum
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
                if step % 20 == 0:
                    progress.write(json.dumps({"step": step, "loss": round(loss_value, 4)}))
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
        "max_length": args.max_length,
        "letters": letters,
        "action_token_ids": dict(zip(letters, action_token_ids(tokenizer, letters=letters))),
        "losses": losses,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    args.loss_out.write_text(json.dumps(losses, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

