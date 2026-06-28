#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import write_json
from src.mbpp_env import candidate_from_completion, load_mbpp_records
from src.model_utils import DEFAULT_MODEL_PATH, attach_new_lora, load_quant_model, load_tokenizer, sample_prompt


def passk_utility(n: int, m: int, k: int) -> float:
    if m <= 0:
        return 0.0
    if k >= n:
        return 1.0
    if n - m < k:
        return 1.0
    return 1.0 - (math.comb(n - m, k) / math.comb(n, k))


def assign_advantages(candidates: list[dict[str, Any]], target_k: int, negative_weight: float) -> tuple[list[float], dict[str, Any]]:
    n = len(candidates)
    m = sum(1 for cand in candidates if cand["full_pass"])
    utility = passk_utility(n, m, min(target_k, n))
    advantages = [0.0] * n
    if m:
        positive_adv = utility / m
        for idx, cand in enumerate(candidates):
            if cand["full_pass"]:
                advantages[idx] = positive_adv
            elif cand["visible_all_pass"]:
                advantages[idx] = -negative_weight * utility / max(1, n - m)
    else:
        for idx, cand in enumerate(candidates):
            if cand["visible_all_pass"]:
                advantages[idx] = -negative_weight
    return advantages, {"group_size": n, "positive_count": m, "passk_utility": utility}


def sequence_logprob(
    model: Any,
    tokenizer: Any,
    prompt: str,
    completions: list[str],
    max_length: int,
    disable_adapter: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    texts = [prompt + completion + tokenizer.eos_token for completion in completions]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length, add_special_tokens=False)
    prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc["attention_mask"].to(model.device)
    labels = input_ids[:, 1:].contiguous()
    label_mask = attention_mask[:, 1:].contiguous().bool()
    positions = torch.arange(labels.shape[1], device=model.device).unsqueeze(0)
    gen_mask = label_mask & (positions >= max(0, prompt_len - 1))
    ctx = model.disable_adapter() if disable_adapter and hasattr(model, "disable_adapter") else torch.enable_grad()
    with ctx:
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[:, :-1, :].contiguous()
        logprobs = torch.log_softmax(logits, dim=-1)
        token_logprobs = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    lengths = gen_mask.sum(dim=1).clamp(min=1)
    seq_logprob = (token_logprobs * gen_mask).sum(dim=1) / lengths
    return seq_logprob, lengths


def train_step(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    args: argparse.Namespace,
    step_seed: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    model.eval()
    completions = sample_prompt(
        model,
        tokenizer,
        record["prompt"],
        count=args.rollouts_per_task,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.generation_batch_size,
        seed=step_seed,
    )
    candidates = [
        candidate_from_completion(comp, record, source=f"train_t{args.temperature:g}_{idx}", order=idx, tokenizer=tokenizer, prompt=record["prompt"])
        for idx, comp in enumerate(completions)
    ]
    advantages, stats = assign_advantages(candidates, args.target_k, args.negative_weight)
    adv = torch.tensor(advantages, dtype=torch.float32, device=model.device)
    model.train()
    seq_logp, lengths = sequence_logprob(model, tokenizer, record["prompt"], completions, args.max_length, disable_adapter=False)
    with torch.no_grad():
        ref_logp, _ = sequence_logprob(model, tokenizer, record["prompt"], completions, args.max_length, disable_adapter=True)
    pg_loss = -(adv * seq_logp).mean()
    kl_proxy = ((seq_logp - ref_logp.detach()) ** 2).mean()
    loss = pg_loss + args.kl_coef * kl_proxy
    stats.update(
        {
            "task_id": record["task_id"],
            "loss": float(loss.detach().cpu()),
            "pg_loss": float(pg_loss.detach().cpu()),
            "kl_proxy": float(kl_proxy.detach().cpu()),
            "adv_mean": float(adv.mean().detach().cpu()),
            "adv_abs_mean": float(adv.abs().mean().detach().cpu()),
            "visible_count": sum(1 for cand in candidates if cand["visible_all_pass"]),
            "mean_completion_tokens": sum(cand["completion_tokens"] for cand in candidates) / max(1, len(candidates)),
            "mean_gen_tokens_for_logprob": float(lengths.float().mean().detach().cpu()),
        }
    )
    return loss, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "validation", "test", "prompt"], default="train")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=5.0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-attempts", type=int, default=0)
    parser.add_argument("--rollouts-per-task", type=int, default=4)
    parser.add_argument("--target-k", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--generation-batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--kl-coef", type=float, default=0.05)
    parser.add_argument("--negative-weight", type=float, default=0.2)
    parser.add_argument("--skip-zero-positive", action="store_true")
    parser.add_argument("--skip-saturated-positive", action="store_true")
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    records = load_mbpp_records(args.split, args.count, args.offset, args.visible_tests, args.timeout_s, tokenizer=tokenizer)
    model = attach_new_lora(load_quant_model(args.model_path, for_training=True), r=16, alpha=32, dropout=0.05)
    model.print_trainable_parameters()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    metrics: list[dict[str, Any]] = []
    max_attempts = args.max_attempts or args.max_steps
    progress = tqdm(total=args.max_steps, desc="passk-rl")
    start = time.time()
    updates = 0
    attempts = 0
    while updates < args.max_steps and attempts < max_attempts:
        record = records[attempts % len(records)]
        optimizer.zero_grad(set_to_none=True)
        loss, stats = train_step(model, tokenizer, record, args, args.seed + attempts * 10007)
        attempts += 1
        stats["attempt"] = attempts
        if args.skip_zero_positive and stats["positive_count"] == 0:
            stats["skipped"] = True
            stats["skip_reason"] = "zero_positive"
            stats["step"] = updates
            metrics.append(stats)
            progress.set_postfix({"skip": attempts, "pos": 0})
            continue
        if args.skip_saturated_positive and stats["positive_count"] >= stats["group_size"]:
            stats["skipped"] = True
            stats["skip_reason"] = "saturated_positive"
            stats["step"] = updates
            metrics.append(stats)
            progress.set_postfix({"skip": attempts, "pos": stats["positive_count"]})
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        updates += 1
        stats["step"] = updates
        stats["skipped"] = False
        stats["skip_reason"] = ""
        metrics.append(stats)
        progress.update(1)
        progress.set_postfix({"loss": f"{stats['loss']:.3f}", "pos": stats["positive_count"], "u": f"{stats['passk_utility']:.2f}", "try": attempts})
        if updates % 5 == 0:
            print(json.dumps(stats, sort_keys=True), flush=True)
    if updates < args.max_steps:
        print(json.dumps({"warning": "ended_before_requested_updates", "updates": updates, "attempts": attempts}, sort_keys=True), flush=True)
    progress.close()

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    summary = {
        "experiment": "qwen35_4b_passk_coverage_rl",
        "method": "online_passk_rl",
        "train_split": args.split,
        "train_count": args.count,
        "max_steps": args.max_steps,
        "max_attempts": max_attempts,
        "updates": updates,
        "attempts": attempts,
        "rollouts_per_task": args.rollouts_per_task,
        "target_k": args.target_k,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "learning_rate": args.learning_rate,
        "kl_coef": args.kl_coef,
        "negative_weight": args.negative_weight,
        "skip_zero_positive": args.skip_zero_positive,
        "skip_saturated_positive": args.skip_saturated_positive,
        "elapsed_s": round(time.time() - start, 3),
        "metrics": metrics,
    }
    (args.output_dir / "training_metadata.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    write_json(args.metrics_out, summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "metrics"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
