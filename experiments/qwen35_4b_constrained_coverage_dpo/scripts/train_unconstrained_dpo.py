#!/usr/bin/env python3
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

from src.jsonl import load_jsonl, write_json  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_new_lora, load_quant_model, load_tokenizer  # noqa: E402


class PairDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.items: list[dict[str, Any]] = []
        for row in rows:
            chosen = self.encode(tokenizer, row["prompt"], row["chosen"], max_length)
            rejected = self.encode(tokenizer, row["prompt"], row["rejected"], max_length)
            if chosen is None or rejected is None:
                continue
            self.items.append({"chosen": chosen, "rejected": rejected, "pair_id": row["pair_id"], "record_id": row["record_id"]})

    @staticmethod
    def encode(tokenizer: Any, prompt: str, response: str, max_length: int) -> dict[str, list[int]] | None:
        eos = tokenizer.eos_token or ""
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response.rstrip() + "\n" + eos, add_special_tokens=False)["input_ids"]
        if not prompt_ids or not response_ids:
            return None
        if len(response_ids) > max_length - 8:
            response_ids = response_ids[: max_length - 8]
        prompt_budget = max_length - len(response_ids)
        if prompt_budget <= 0:
            return None
        prompt_ids = prompt_ids[-prompt_budget:]
        input_ids = prompt_ids + response_ids
        response_mask = [0] * max(len(input_ids) - 1, 0)
        start = max(len(prompt_ids) - 1, 0)
        for index in range(start, len(response_mask)):
            response_mask[index] = 1
        if not any(response_mask):
            return None
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "response_mask": response_mask}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def pad_side(batch: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(len(row["input_ids"]) for row in batch)
    input_ids = []
    attention_mask = []
    response_mask = []
    for row in batch:
        pad = max_len - len(row["input_ids"])
        input_ids.append(row["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(row["attention_mask"] + [0] * pad)
        response_mask.append(row["response_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "response_mask": torch.tensor(response_mask, dtype=torch.float32),
    }


def collate(batch: list[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    return {
        "chosen": pad_side([row["chosen"] for row in batch], pad_token_id),
        "rejected": pad_side([row["rejected"] for row in batch], pad_token_id),
        "pair_ids": [row["pair_id"] for row in batch],
        "record_ids": [row["record_id"] for row in batch],
    }


def response_logps(model: Any, batch: dict[str, torch.Tensor], normalize: bool) -> torch.Tensor:
    input_ids = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    response_mask = batch["response_mask"].to(model.device)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    mask = response_mask[:, : labels.shape[1]] * attention_mask[:, 1:].float()
    logps = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    summed = (logps * mask).sum(dim=1)
    if normalize:
        return summed / mask.sum(dim=1).clamp_min(1.0)
    return summed


def dpo_step(model: Any, batch: dict[str, Any], beta: float, normalize: bool) -> tuple[torch.Tensor, dict[str, float]]:
    pi_chosen = response_logps(model, batch["chosen"], normalize)
    pi_rejected = response_logps(model, batch["rejected"], normalize)
    with torch.no_grad():
        with model.disable_adapter():
            ref_chosen = response_logps(model, batch["chosen"], normalize)
            ref_rejected = response_logps(model, batch["rejected"], normalize)
    pi_margin = pi_chosen - pi_rejected
    ref_margin = ref_chosen - ref_rejected
    logits = beta * (pi_margin - ref_margin)
    loss = -F.logsigmoid(logits).mean()
    chosen_win = (pi_margin > 0).float().mean()
    reward_margin = (pi_margin - ref_margin).mean()
    return loss, {
        "loss": float(loss.detach().cpu()),
        "pi_margin": float(pi_margin.mean().detach().cpu()),
        "ref_margin": float(ref_margin.mean().detach().cpu()),
        "reward_margin": float(reward_margin.detach().cpu()),
        "chosen_win_rate": float(chosen_win.detach().cpu()),
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
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--sum-logps", action="store_true", help="Use unnormalized sequence log-probabilities instead of mean-token log-probabilities.")
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
        raise RuntimeError("no preference pairs")

    tokenizer = load_tokenizer(args.model_path, padding_side="right")
    model = attach_new_lora(
        load_quant_model(args.model_path, for_training=True),
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    model.print_trainable_parameters()
    dataset = PairDataset(rows, tokenizer, args.max_length)
    if not dataset:
        raise RuntimeError("empty tokenized preference dataset")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    metrics: list[dict[str, float]] = []
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    progress = tqdm(total=args.max_steps, desc=args.run_name)
    model.train()
    normalize = not args.sum_logps
    while step < args.max_steps:
        for batch in loader:
            loss, row = dpo_step(model, batch, args.beta, normalize)
            (loss / args.grad_accum).backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row["step"] = step
                metrics.append(row)
                progress.update(1)
                if step % 10 == 0:
                    progress.write(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
                if step >= args.max_steps:
                    break
    progress.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "run_name": args.run_name,
        "pairs_in": str(args.pairs),
        "pair_rows": len(rows),
        "tokenized_pairs": len(dataset),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "max_length": args.max_length,
        "logprob_normalization": "mean_token" if normalize else "sum_sequence",
        "metrics": metrics,
    }
    write_json(args.metrics_out, metadata)
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

