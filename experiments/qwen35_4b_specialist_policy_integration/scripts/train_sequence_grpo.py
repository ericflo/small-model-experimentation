#!/usr/bin/env python3
"""Guarded sequence-GRPO over complete interactive trajectories.

This is deliberately small and auditable: trajectories are collected by the
current merged checkpoint, advantages are normalized only among sibling
rollouts of the same initial episode, and the new LoRA starts at exactly zero
delta on that checkpoint.  Injected force-close tokens are masked from policy
loss.  ``--shuffle-advantages`` is the mechanism control.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, read_jsonl, sha256_file, training_seed  # noqa: E402


TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def _shuffle_group_advantages(trajectories: list[dict[str, Any]], seed: int) -> str:
    """Permute complete advantage vectors across groups within family/level."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        groups[row["episode_key"]].append(row)
    cells: dict[tuple[str, int], list[str]] = defaultdict(list)
    for key, rows in groups.items():
        cells[(rows[0]["family"], int(rows[0]["level"]))].append(key)
    rng = random.Random(seed)
    mapping: dict[str, str] = {}
    for _, keys in sorted(cells.items()):
        keys = sorted(keys)
        donors = list(keys)
        rng.shuffle(donors)
        mapping.update(dict(zip(keys, donors)))
    original = {
        key: [float(row["advantage"]) for row in sorted(rows, key=lambda x: x["rollout"])]
        for key, rows in groups.items()
    }
    for key, rows in groups.items():
        values = original[mapping[key]]
        for row, value in zip(sorted(rows, key=lambda x: x["rollout"]), values):
            row["advantage"] = value
            row["advantage_active"] = abs(value) > 0.0
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _prompt_ids(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    if not rendered.endswith("<think>\n"):
        raise ValueError(f"unexpected thinking prompt tail: {rendered[-50:]!r}")
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def _policy_sample(
    tokenizer: Any,
    trajectory: dict[str, Any],
    turn: dict[str, Any],
    max_length: int,
    think_weight: float,
) -> dict[str, Any] | None:
    policy = turn["policy"]
    prompt = _prompt_ids(tokenizer, turn["messages_before"])
    recorded_prompt = int(policy["n_stage1_prompt_tokens"])
    if len(prompt) != recorded_prompt:
        raise ValueError(
            f"prompt-token mismatch {trajectory['rid']} t{turn['turn']}: "
            f"trainer={len(prompt)} rollout={recorded_prompt}"
        )
    completion = [int(token) for token in policy.get("token_ids") or []]
    if not completion or len(prompt) + len(completion) > max_length:
        return None
    weights = [0.0] * len(completion)
    if policy.get("forced_close"):
        n_think = len(policy.get("retained_thinking_token_ids") or [])
        n_injected = len(policy.get("injected_token_ids") or [])
        for index in range(min(n_think, len(weights))):
            weights[index] = think_weight
        for index in range(n_think + n_injected, len(weights)):
            weights[index] = 1.0
    else:
        close_id = int(tokenizer.convert_tokens_to_ids("</think>"))
        close_index = completion.index(close_id) if close_id in completion else len(completion) - 1
        for index in range(close_index + 1):
            weights[index] = think_weight
        for index in range(close_index + 1, len(weights)):
            weights[index] = 1.0
    if sum(weights) <= 0.0:
        return None
    return {
        "id": f"{trajectory['rid']}-t{turn['turn']}",
        "family": trajectory["family"],
        "level": int(trajectory["level"]),
        "episode_key": trajectory["episode_key"],
        "input_ids": prompt + completion,
        "completion_ids": completion,
        "completion_weights": weights,
        "advantage": float(trajectory["advantage"]),
        "trajectory_turn_weight": 1.0 / max(1, len(trajectory["turns"])),
        "forced_close": bool(policy.get("forced_close")),
    }


def _anchor_sample(
    tokenizer: Any,
    row: dict[str, Any],
    max_length: int,
    think_weight: float,
) -> dict[str, Any] | None:
    prompt = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    think_part = row["think"].strip() + "\n</think>\n\n"
    answer_part = row["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    middle_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
    final_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
    if len(final_ids) > max_length:
        return None
    if final_ids[: len(prompt_ids)] != prompt_ids or final_ids[: len(middle_ids)] != middle_ids:
        return None
    completion = final_ids[len(prompt_ids) :]
    n_think = len(middle_ids) - len(prompt_ids)
    weights = [think_weight] * n_think + [1.0] * (len(completion) - n_think)
    return {
        "id": row.get("id", "anchor"),
        "input_ids": final_ids,
        "completion_ids": completion,
        "completion_weights": weights,
    }


def _completion_log_probs(model: Any, sample: dict[str, Any]) -> torch.Tensor:
    input_ids = torch.tensor([sample["input_ids"]], dtype=torch.long, device=model.device)
    attention = torch.ones_like(input_ids)
    n_completion = len(sample["completion_ids"])
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention,
        logits_to_keep=n_completion + 1,
        use_cache=False,
    )
    logits = outputs.logits
    if logits.shape[1] < n_completion + 1:
        raise RuntimeError(
            f"logits_to_keep returned {logits.shape[1]} positions for {n_completion} targets"
        )
    prediction_logits = logits[:, -(n_completion + 1) : -1, :][0]
    targets = torch.tensor(sample["completion_ids"], dtype=torch.long, device=model.device)
    pieces = []
    for start in range(0, n_completion, 64):
        chunk = prediction_logits[start : start + 64].float()
        target = targets[start : start + 64]
        pieces.append(torch.log_softmax(chunk, dim=-1).gather(1, target[:, None])[:, 0])
    return torch.cat(pieces)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--trajectories", type=Path, default=EXP / "runs" / "rl_collection" / "trajectories.jsonl.gz")
    parser.add_argument("--anchors", type=Path, default=EXP / "data" / "rl_anchor.jsonl")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--run-tag")
    parser.add_argument("--shuffle-advantages", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    train_cfg = config["rl_train"]
    max_steps = int(args.max_steps or train_cfg["max_steps"])
    if args.smoke:
        max_steps = min(max_steps, 2)
    if not (args.model / "config.json").exists():
        raise SystemExit(f"source merged checkpoint missing config.json: {args.model}")

    trajectories = read_jsonl(args.trajectories)
    shuffle_mapping_sha = None
    if args.shuffle_advantages:
        shuffle_mapping_sha = _shuffle_group_advantages(
            trajectories, int(config["seeds"]["shuffled_reward"])
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(train_cfg["max_length"])
    think_weight = float(train_cfg["think_loss_weight"])
    samples = []
    skipped = 0
    for trajectory in trajectories:
        if not trajectory.get("advantage_active") or abs(float(trajectory["advantage"])) < 1e-12:
            continue
        for turn in trajectory["turns"]:
            sample = _policy_sample(tokenizer, trajectory, turn, max_length, think_weight)
            if sample is None:
                skipped += 1
            else:
                samples.append(sample)
    if not samples:
        raise SystemExit("no active policy samples after tokenization")
    skip_rate = skipped / (skipped + len(samples))
    if skip_rate > 0.15:
        raise SystemExit(f"policy sample over-length/invalid skip rate {skip_rate:.3f} > 0.15")

    anchors = []
    for row in read_jsonl(args.anchors):
        sample = _anchor_sample(tokenizer, row, max_length, think_weight)
        if sample is not None:
            anchors.append(sample)
    if not anchors:
        raise SystemExit("no trainable supervised anchors")

    seed = int(args.seed if args.seed is not None else training_seed(config))
    rng = random.Random(seed + (1 if args.shuffle_advantages else 0))
    rng.shuffle(samples)
    rng.shuffle(anchors)

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=bnb,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(train_cfg["rank"]),
            lora_alpha=int(train_cfg["alpha"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(train_cfg["learning_rate"]))

    grad_accum = int(train_cfg["grad_accum"])
    clip_epsilon = float(train_cfg["clip_epsilon"])
    kl_beta = float(train_cfg["kl_beta"])
    anchor_coef = float(train_cfg["anchor_coef"])
    anchor_every = int(train_cfg["anchor_every"])
    max_grad_norm = float(train_cfg["max_grad_norm"])
    max_kl = float(train_cfg["max_mean_sampled_token_kl"])
    logs: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    micro_step = 0
    optimizer_step = 0
    sample_index = 0
    anchor_index = 0
    stopped_reason = None
    compute_ledger = {
        "reference_forward_input_tokens": 0,
        "policy_forward_input_tokens": 0,
        "anchor_forward_input_tokens": 0,
        "policy_target_tokens": 0,
        "anchor_target_tokens": 0,
        "reference_forwards": 0,
        "policy_forwards": 0,
        "anchor_forwards": 0,
    }
    model.train()

    while optimizer_step < max_steps:
        sample = samples[sample_index % len(samples)]
        sample_index += 1
        sample_input_tokens = len(sample["input_ids"])
        compute_ledger["reference_forward_input_tokens"] += sample_input_tokens
        compute_ledger["policy_forward_input_tokens"] += sample_input_tokens
        compute_ledger["policy_target_tokens"] += len(sample["completion_ids"])
        compute_ledger["reference_forwards"] += 1
        compute_ledger["policy_forwards"] += 1
        with torch.no_grad(), model.disable_adapter():
            reference_log_probs = _completion_log_probs(model, sample).detach()
        policy_log_probs = _completion_log_probs(model, sample)
        weights = torch.tensor(
            sample["completion_weights"], dtype=torch.float32, device=model.device
        )
        log_ratio = policy_log_probs - reference_log_probs
        ratio = torch.exp(log_ratio.clamp(-10.0, 10.0))
        advantage = torch.tensor(float(sample["advantage"]), device=model.device)
        objective = torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage,
        )
        policy_loss = -_weighted_mean(objective, weights)
        approx_kl = _weighted_mean(0.5 * log_ratio.square(), weights)
        loss = (policy_loss + kl_beta * approx_kl) * float(sample["trajectory_turn_weight"])

        anchor_loss_value = None
        if micro_step % anchor_every == 0:
            anchor = anchors[anchor_index % len(anchors)]
            anchor_index += 1
            compute_ledger["anchor_forward_input_tokens"] += len(anchor["input_ids"])
            compute_ledger["anchor_target_tokens"] += len(anchor["completion_ids"])
            compute_ledger["anchor_forwards"] += 1
            anchor_log_probs = _completion_log_probs(model, anchor)
            anchor_weights = torch.tensor(
                anchor["completion_weights"], dtype=torch.float32, device=model.device
            )
            anchor_loss = -_weighted_mean(anchor_log_probs, anchor_weights)
            loss = loss + anchor_coef * anchor_loss
            anchor_loss_value = float(anchor_loss.detach().cpu())

        if not torch.isfinite(loss):
            stopped_reason = f"non-finite loss at micro-step {micro_step}"
            break
        (loss / grad_accum).backward()
        micro_step += 1
        if micro_step % grad_accum:
            continue

        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, max_grad_norm).detach().cpu())
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        optimizer_step += 1
        row = {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "sample_id": sample["id"],
            "family": sample["family"],
            "level": sample["level"],
            "advantage": sample["advantage"],
            "policy_loss": float(policy_loss.detach().cpu()),
            "approx_kl": float(approx_kl.detach().cpu()),
            "mean_abs_log_ratio": float(_weighted_mean(log_ratio.abs(), weights).detach().cpu()),
            "anchor_loss": anchor_loss_value,
            "grad_norm": grad_norm,
            "forced_close": sample["forced_close"],
        }
        logs.append(row)
        if optimizer_step == 1 or optimizer_step % 10 == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if optimizer_step >= 10:
            recent_kl = sum(item["approx_kl"] for item in logs[-10:]) / min(10, len(logs))
            if recent_kl > max_kl:
                stopped_reason = f"rolling mean sampled-token KL {recent_kl:.6f} > {max_kl}"
                break

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    receipt = {
        "method": "guarded_sequence_grpo",
        "source_model": str(args.model.resolve()),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "trajectory_file": str(args.trajectories.resolve()),
        "trajectory_sha256": sha256_file(args.trajectories),
        "anchor_file": str(args.anchors.resolve()),
        "anchor_sha256": sha256_file(args.anchors),
        "policy_samples": len(samples),
        "skipped_policy_samples": skipped,
        "policy_skip_rate": skip_rate,
        "anchor_samples": len(anchors),
        "requested_steps": max_steps,
        "completed_steps": optimizer_step,
        "micro_steps": micro_step,
        "compute_ledger": compute_ledger,
        "stopped_reason": stopped_reason,
        "shuffle_advantages": bool(args.shuffle_advantages),
        "shuffle_mapping_sha256": shuffle_mapping_sha,
        "seed": seed,
        "training_environment": {
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "peft": __import__("peft").__version__,
            "bitsandbytes": __import__("bitsandbytes").__version__,
            "lock_path": str(REPO / "requirements-training.lock.txt"),
            "lock_sha256": sha256_file(REPO / "requirements-training.lock.txt"),
            "gpu": torch.cuda.get_device_name(0),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        },
        "hyperparameters": {
            key: train_cfg[key]
            for key in (
                "learning_rate",
                "rank",
                "alpha",
                "grad_accum",
                "max_length",
                "clip_epsilon",
                "kl_beta",
                "max_mean_sampled_token_kl",
                "max_grad_norm",
                "think_loss_weight",
                "anchor_coef",
                "anchor_every",
            )
        },
        "logs": logs,
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_name = (
        f"{args.run_tag}.json"
        if args.run_tag
        else ("shuffled_grpo_training.json" if args.shuffle_advantages else "grpo_training.json")
    )
    (EXP / "runs").mkdir(parents=True, exist_ok=True)
    (EXP / "runs" / report_name).write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "logs"}, indent=2, sort_keys=True))
    return 0 if stopped_reason is None else 3


if __name__ == "__main__":
    raise SystemExit(main())
