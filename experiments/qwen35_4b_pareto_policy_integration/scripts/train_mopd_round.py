#!/usr/bin/env python3
"""Train one consume-once MOPD round from cached exact-prefix teacher targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file  # noqa: E402
from mopd_loss import sparse_teacher_topk_reverse_kl  # noqa: E402


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def _training_units(samples: list[dict], quick_fraction: float, micro_steps: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    pools: dict[str, list[dict]] = {"quick": [], "deep": []}
    for sample in samples:
        stratum = str(sample["meta"]["stratum"])
        active = torch.nonzero(sample["policy_mask"], as_tuple=False).flatten().tolist()
        if not active:
            continue
        # Each cached sample is a distinct student trajectory. Consume one
        # target span from it exactly once; never manufacture extra units by
        # splitting or replaying a rollout.
        chunks = [active[-256:]]
        for chunk_index, positions in enumerate(chunks):
            if positions:
                pools[stratum].append({
                    "sample": sample, "positions": positions, "chunk": chunk_index,
                })
    for values in pools.values():
        rng.shuffle(values)
    quick_n = round(micro_steps * quick_fraction)
    deep_n = micro_steps - quick_n
    if len(pools["quick"]) < quick_n or len(pools["deep"]) < deep_n:
        raise ValueError(
            f"insufficient consume-once units: quick {len(pools['quick'])}/{quick_n}, "
            f"deep {len(pools['deep'])}/{deep_n}"
        )
    units = pools["quick"][:quick_n] + pools["deep"][:deep_n]
    rng.shuffle(units)
    return units


def _unit_loss(model, unit: dict) -> tuple[torch.Tensor, int, int]:
    """Evaluate one rollout span against its cached full-softmax teacher top-k."""
    sample = unit["sample"]
    positions = unit["positions"]
    end = max(positions) + 1
    prompt = sample["prompt_ids"].to(dtype=torch.long).tolist()
    completion = sample["completion_ids"].to(dtype=torch.long).tolist()[:end]
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=model.device)
    outputs = model(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        logits_to_keep=end + 1, use_cache=False,
    )
    prediction = outputs.logits[0, -(end + 1):-1]
    pos = torch.tensor(positions, dtype=torch.long, device=model.device)
    selected_logits = prediction.index_select(0, pos)
    cpu_pos = torch.tensor(positions, dtype=torch.long)
    teacher_indices = sample["teacher_indices"].index_select(0, cpu_pos)
    teacher_log_probs = sample["teacher_log_probs"].index_select(0, cpu_pos)
    loss = sparse_teacher_topk_reverse_kl(
        selected_logits, teacher_indices, teacher_log_probs, reduction="mean"
    )
    return loss, int(ids.shape[1]), len(positions)


def _matched_loss_scale(initial_loss: float, target_loss: float | None) -> float:
    if not math.isfinite(initial_loss) or initial_loss <= 0.0:
        raise ValueError(f"initial loss must be finite and positive, got {initial_loss}")
    if target_loss is None:
        return 1.0
    if not math.isfinite(target_loss) or target_loss <= 0.0:
        raise ValueError(f"target loss must be finite and positive, got {target_loss}")
    scale = target_loss / initial_loss
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid matched loss scale: {scale}")
    return scale


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--teacher-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--target-initial-loss", type=float)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not (args.base_model / "config.json").is_file():
        raise SystemExit("base merged composite is incomplete")
    receipt_path = args.teacher_cache.with_suffix(args.teacher_cache.suffix + ".receipt.json")
    cache_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if cache_receipt["cache_sha256"] != sha256_file(args.teacher_cache):
        raise SystemExit("teacher cache checksum mismatch")
    payload = torch.load(args.teacher_cache, map_location="cpu", weights_only=False)
    samples = payload["samples"]
    cfg = config["mopd"]
    updates = 1 if args.smoke else int(
        args.updates if args.updates is not None else cfg["updates_per_round"]
    )
    if updates < 1 or updates > int(cfg["updates_per_round"]):
        raise SystemExit(
            f"updates must be in [1, {int(cfg['updates_per_round'])}], got {updates}"
        )
    grad_accum = 1 if args.smoke else int(cfg["grad_accum"])
    units = _training_units(
        samples, float(cfg["retention_fraction"]), updates * grad_accum, args.seed
    )
    if len({unit["sample"]["id"] for unit in units}) != len(units):
        raise RuntimeError("consume-once violation: a rollout was selected more than once")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True,
        device_map="cuda", dtype=torch.bfloat16, quantization_config=bnb,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(cfg["rank"]), lora_alpha=int(cfg["alpha"]), lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    probe_quick_n = round(grad_accum * float(cfg["retention_fraction"]))
    probe_deep_n = grad_accum - probe_quick_n
    probe_units = (
        sorted(
            [unit for unit in units if unit["sample"]["meta"]["stratum"] == "quick"],
            key=lambda unit: unit["sample"]["id"],
        )[:probe_quick_n]
        + sorted(
            [unit for unit in units if unit["sample"]["meta"]["stratum"] == "deep"],
            key=lambda unit: unit["sample"]["id"],
        )[:probe_deep_n]
    )
    model.eval()
    with torch.inference_mode():
        initial_probe_losses = [
            float(_unit_loss(model, unit)[0].detach().cpu()) for unit in probe_units
        ]
    initial_probe_loss = sum(initial_probe_losses) / len(initial_probe_losses)
    loss_scale = _matched_loss_scale(initial_probe_loss, args.target_initial_loss)
    print(json.dumps({
        "initial_probe_loss": initial_probe_loss,
        "target_initial_loss": args.target_initial_loss,
        "loss_scale": loss_scale,
        "probe_units": len(probe_units),
    }, sort_keys=True), flush=True)
    logs = []
    started = time.perf_counter()
    token_ledger = {"forward_input_tokens": 0, "distilled_positions": 0}
    unsafe_reason = None
    completed_updates = 0
    model.train()
    for micro_step, unit in enumerate(units, 1):
        sample = unit["sample"]
        positions = unit["positions"]
        loss, input_tokens, distilled_positions = _unit_loss(model, unit)
        token_ledger["forward_input_tokens"] += input_tokens
        token_ledger["distilled_positions"] += distilled_positions
        row = {
            "micro_step": micro_step, "sample_id": sample["id"],
            "stratum": sample["meta"]["stratum"], "chunk": unit["chunk"],
            "positions": len(positions),
            "loss": float(loss.detach().cpu()) if torch.isfinite(loss) else None,
            "scaled_loss": (
                float((loss.detach() * loss_scale).cpu())
                if torch.isfinite(loss) else None
            ),
        }
        if not torch.isfinite(loss):
            unsafe_reason = f"non-finite loss at micro-step {micro_step}"
            row["unsafe_stop"] = unsafe_reason
            logs.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
            break
        (loss * loss_scale / grad_accum).backward()
        if micro_step % grad_accum == 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"])).detach().cpu()
            )
            row["optimizer_step"] = micro_step // grad_accum
            row["grad_norm"] = grad_norm
            if not math.isfinite(grad_norm):
                unsafe_reason = f"non-finite gradient norm at micro-step {micro_step}"
                row["unsafe_stop"] = unsafe_reason
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed_updates += 1
            print(json.dumps(row, sort_keys=True), flush=True)
        logs.append(row)
        if unsafe_reason is not None:
            break

    finite_losses = [float(row["loss"]) for row in logs if row["loss"] is not None]
    mean_loss = sum(finite_losses) / len(finite_losses) if finite_losses else None
    gate_passed = (
        unsafe_reason is None
        and completed_updates == updates
        and mean_loss is not None
        and mean_loss <= float(cfg["maximum_round_mean_kl"])
    )
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    receipt = {
        "method": "on_policy_mopd_round",
        "round": args.round,
        "base_model": str(args.base_model.resolve()),
        "base_config_sha256": sha256_file(args.base_model / "config.json"),
        "teacher_cache": str(args.teacher_cache.resolve()),
        "teacher_cache_sha256": sha256_file(args.teacher_cache),
        "routing": payload["routing"],
        "config": str(config_path), "config_sha256": sha256_file(config_path),
        "seed": args.seed, "smoke": bool(args.smoke),
        "requested_optimizer_steps": updates,
        "optimizer_steps": completed_updates,
        "requested_micro_steps": len(units), "micro_steps": len(logs),
        "quick_units": sum(unit["sample"]["meta"]["stratum"] == "quick" for unit in units),
        "deep_units": sum(unit["sample"]["meta"]["stratum"] == "deep" for unit in units),
        "unique_rollouts": len({unit["sample"]["id"] for unit in units}),
        "consume_once_verified": (
            len({unit["sample"]["id"] for unit in units}) == len(units)
        ),
        "mean_corrected_topk_loss": mean_loss,
        "initial_probe": {
            "units": len(probe_units),
            "mean_corrected_topk_loss": initial_probe_loss,
            "unit_losses": initial_probe_losses,
            "target_mean_corrected_topk_loss": args.target_initial_loss,
            "backward_loss_scale": loss_scale,
        },
        "round_loss_gate": {
            "passed": gate_passed,
            "maximum": float(cfg["maximum_round_mean_kl"]),
            "unsafe_reason": unsafe_reason,
            "completed_all_updates": completed_updates == updates,
        },
        "token_ledger": token_ledger,
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "hyperparameters": {
            key: cfg[key] for key in (
                "top_k", "learning_rate", "rank", "alpha", "grad_accum",
                "max_length", "max_grad_norm", "retention_fraction",
                "maximum_round_mean_kl",
            )
        },
        "logs": logs,
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "logs"}, indent=2))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
