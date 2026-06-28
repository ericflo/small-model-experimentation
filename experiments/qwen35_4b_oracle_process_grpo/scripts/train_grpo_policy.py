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

from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402
from src.operator_env import LETTERS, load_jsonl  # noqa: E402


class RewardDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]], ref_probs: list[list[float]]) -> None:
        self.rows = rows
        self.ref_probs = ref_probs

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        rewards = [float(row["reward_by_action"][letter]) for letter in LETTERS]
        return {"prompt": row["prompt"], "rewards": rewards, "ref_probs": self.ref_probs[idx]}


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt": [item["prompt"] for item in batch],
        "rewards": torch.tensor([item["rewards"] for item in batch], dtype=torch.float32),
        "ref_probs": torch.tensor([item["ref_probs"] for item in batch], dtype=torch.float32),
    }


@torch.no_grad()
def compute_ref_probs(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_length: int, batch_size: int) -> list[list[float]]:
    model.eval()
    probs: list[list[float]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="reference-probs"):
        prompts = [row["prompt"] for row in rows[start : start + batch_size]]
        logits = last_token_action_logits(model, tokenizer, prompts, max_length=max_length)
        probs.extend(torch.softmax(logits.float(), dim=-1).detach().cpu().tolist())
    return probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "process_train_states.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft-adapter", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/sft_process_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_oracle_process_grpo/models/grpo_process_lora"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--kl-beta", type=float, default=0.02)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shuffle-rewards", action="store_true")
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_jsonl(args.train)
    random.shuffle(rows)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = load_tokenizer(args.model_path)
    model = attach_existing_lora(load_base_model(args.model_path, for_training=True), args.sft_adapter, is_trainable=True)
    ref_probs = compute_ref_probs(model, tokenizer, rows, args.max_length, args.batch_size)
    if args.shuffle_rewards:
        rng = random.Random(args.seed + 31)
        for row in rows:
            values = [row["reward_by_action"][letter] for letter in LETTERS]
            rng.shuffle(values)
            row["reward_by_action"] = dict(zip(LETTERS, values))
    dataset = RewardDataset(rows, ref_probs)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    progress = tqdm(total=args.max_steps, desc="grpo-process-policy")
    model.train()
    while step < args.max_steps:
        for batch in loader:
            prompts = batch["prompt"]
            rewards = batch["rewards"].to(model.device)
            ref = batch["ref_probs"].to(model.device).clamp_min(1e-6)
            logits = last_token_action_logits(model, tokenizer, prompts, max_length=args.max_length).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            probs = torch.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=args.group_size, replacement=True)
            sampled_rewards = rewards.gather(1, sampled)
            centered = sampled_rewards - sampled_rewards.mean(dim=1, keepdim=True)
            scale = sampled_rewards.std(dim=1, keepdim=True).clamp_min(1e-4)
            advantages = centered / scale
            sampled_log_probs = log_probs.gather(1, sampled)
            policy_loss = -(advantages.detach() * sampled_log_probs).mean()
            kl = (probs * (log_probs - torch.log(ref))).sum(dim=-1).mean()
            loss = (policy_loss + args.kl_beta * kl) / args.grad_accum
            loss.backward()
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                row = {
                    "step": step,
                    "loss": float(loss.detach().cpu()) * args.grad_accum,
                    "policy_loss": float(policy_loss.detach().cpu()),
                    "kl": float(kl.detach().cpu()),
                    "sampled_reward_mean": float(sampled_rewards.mean().detach().cpu()),
                    "sampled_reward_max": float(sampled_rewards.max().detach().cpu()),
                    "entropy": float((-(probs * log_probs).sum(dim=-1).mean()).detach().cpu()),
                }
                logs.append(row)
                progress.update(1)
                if step % 20 == 0:
                    progress.write(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()}))
                if step >= args.max_steps:
                    break
    progress.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "method": "group_relative_policy_optimization_over_verifier_rewards",
        "train_states": len(rows),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "group_size": args.group_size,
        "learning_rate": args.learning_rate,
        "kl_beta": args.kl_beta,
        "shuffle_rewards": args.shuffle_rewards,
        "max_length": args.max_length,
        "logs": logs,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (ROOT / "reports" / "grpo_training_logs.json").write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
