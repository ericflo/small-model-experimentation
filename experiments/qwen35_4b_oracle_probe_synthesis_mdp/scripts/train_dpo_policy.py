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


class PreferenceDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        ref_log_probs: list[list[float]],
        seed: int,
        shuffle_rewards: bool,
    ) -> None:
        rng = random.Random(seed)
        self.items: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            rewards = [float(row["reward_by_action"][letter]) for letter in LETTERS]
            if shuffle_rewards:
                rewards = list(rewards)
                rng.shuffle(rewards)
            order = sorted(range(len(LETTERS)), key=lambda i: (rewards[i], -i), reverse=True)
            chosen = order[0]
            # Hard negative: best non-oracle action with meaningfully lower reward.
            rejected = None
            for candidate in order[1:]:
                if rewards[chosen] > rewards[candidate] + 1e-6:
                    rejected = candidate
                    break
            if rejected is None:
                continue
            self.items.append(
                {
                    "prompt": row["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_reward": rewards[chosen],
                    "rejected_reward": rewards[rejected],
                    "reward_gap": rewards[chosen] - rewards[rejected],
                    "ref_chosen": ref_log_probs[idx][chosen],
                    "ref_rejected": ref_log_probs[idx][rejected],
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.items[idx]


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "prompt": [item["prompt"] for item in batch],
        "chosen": torch.tensor([item["chosen"] for item in batch], dtype=torch.long),
        "rejected": torch.tensor([item["rejected"] for item in batch], dtype=torch.long),
        "chosen_reward": torch.tensor([item["chosen_reward"] for item in batch], dtype=torch.float32),
        "rejected_reward": torch.tensor([item["rejected_reward"] for item in batch], dtype=torch.float32),
        "reward_gap": torch.tensor([item["reward_gap"] for item in batch], dtype=torch.float32),
        "ref_chosen": torch.tensor([item["ref_chosen"] for item in batch], dtype=torch.float32),
        "ref_rejected": torch.tensor([item["ref_rejected"] for item in batch], dtype=torch.float32),
    }


@torch.no_grad()
def compute_ref_log_probs(model: Any, tokenizer: Any, rows: list[dict[str, Any]], max_length: int, batch_size: int) -> list[list[float]]:
    model.eval()
    out_rows: list[list[float]] = []
    for start in tqdm(range(0, len(rows), batch_size), desc="reference-logprobs"):
        prompts = [row["prompt"] for row in rows[start : start + batch_size]]
        logits = last_token_action_logits(model, tokenizer, prompts, max_length=max_length)
        out_rows.extend(torch.log_softmax(logits.float(), dim=-1).detach().cpu().tolist())
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "process_train_states.jsonl")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--sft-adapter", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/sft_process_lora"))
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/large_artifacts/qwen35_4b_oracle_probe_synthesis_mdp/models/dpo_process_lora"))
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--beta", type=float, default=0.2)
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
    ref_log_probs = compute_ref_log_probs(model, tokenizer, rows, args.max_length, args.batch_size)
    dataset = PreferenceDataset(rows, ref_log_probs, args.seed + 17, args.shuffle_rewards)
    if not dataset:
        raise RuntimeError("no preference pairs after tie filtering")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    logs: list[dict[str, float]] = []
    optimizer.zero_grad(set_to_none=True)
    step = 0
    micro = 0
    progress = tqdm(total=args.max_steps, desc="dpo-process-policy")
    model.train()
    while step < args.max_steps:
        for batch in loader:
            prompts = batch["prompt"]
            chosen = batch["chosen"].to(model.device)
            rejected = batch["rejected"].to(model.device)
            ref_chosen = batch["ref_chosen"].to(model.device)
            ref_rejected = batch["ref_rejected"].to(model.device)
            reward_gap = batch["reward_gap"].to(model.device)
            logits = last_token_action_logits(model, tokenizer, prompts, max_length=args.max_length).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            pi_chosen = log_probs.gather(1, chosen[:, None]).squeeze(1)
            pi_rejected = log_probs.gather(1, rejected[:, None]).squeeze(1)
            pi_delta = pi_chosen - pi_rejected
            ref_delta = ref_chosen - ref_rejected
            loss_vec = -F.logsigmoid(args.beta * (pi_delta - ref_delta))
            loss = loss_vec.mean() / args.grad_accum
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
                    "pi_delta": float(pi_delta.mean().detach().cpu()),
                    "ref_delta": float(ref_delta.mean().detach().cpu()),
                    "reward_gap": float(reward_gap.mean().detach().cpu()),
                    "pair_accuracy": float((pi_delta > 0).float().mean().detach().cpu()),
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
        "method": "process_dpo_over_verifier_preference_pairs",
        "shuffle_rewards": args.shuffle_rewards,
        "raw_train_states": len(rows),
        "preference_pairs": len(dataset),
        "max_steps": args.max_steps,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "learning_rate": args.learning_rate,
        "beta": args.beta,
        "max_length": args.max_length,
        "logs": logs,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    name = "dpo_shuffled_training_logs.json" if args.shuffle_rewards else "dpo_training_logs.json"
    (ROOT / "reports" / name).write_text(json.dumps(logs, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

