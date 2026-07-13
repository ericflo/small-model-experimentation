#!/usr/bin/env python3
"""Train one matched off-policy best-teacher-continuation SFT control round."""

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


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import load_config, sha256_file  # noqa: E402
from training_units import (  # noqa: E402
    fit_prompt_around_completion,
    offpolicy_prompt_and_completion,
    prompt_and_student_completion,
)


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]


def _loss_scale(initial_loss: float, target_loss: float | None) -> float:
    if not math.isfinite(initial_loss) or initial_loss <= 0.0:
        raise ValueError("initial SFT loss must be finite and positive")
    if target_loss is None:
        return 1.0
    if not math.isfinite(target_loss) or target_loss <= 0.0:
        raise ValueError("matched target pressure must be finite and positive")
    return target_loss / initial_loss


def _make_units(manifest: dict, tokenizer, *, max_positions: int, max_length: int) -> list[dict]:
    units = []
    for row in manifest["units"]:
        if row["role"] == "capability":
            prompt, completion, active = offpolicy_prompt_and_completion(row, tokenizer)
            target_policy = str(row["offpolicy_target"]["policy"])
            terminal_score = float(row["offpolicy_target"]["terminal_score"])
        elif row["role"] == "anchor":
            # The loss family has no dense soup target.  Matching the exact
            # successful current-student trajectory provides the closest
            # consume-once retention analogue without teacher generation.
            prompt, completion, active = prompt_and_student_completion(row, tokenizer)
            target_policy = "student_anchor"
            terminal_score = float(row["state"]["student_terminal_score"])
        else:
            raise ValueError(f"unknown training role: {row['role']}")
        positions = active[-int(max_positions):]
        prompt, prompt_tokens_truncated = fit_prompt_around_completion(
            prompt,
            completion,
            max_length=max_length,
            state_id=str(row["state_id"]),
        )
        units.append(
            {
                "id": str(row["state_id"]),
                "role": str(row["role"]),
                "kind": str(row["kind"]),
                "level": int(row["level"]),
                "target_policy": target_policy,
                "target_terminal_score": terminal_score,
                "prompt_tokens_truncated": prompt_tokens_truncated,
                "prompt_ids": prompt,
                "completion_ids": completion,
                "positions": positions,
            }
        )
    if len({unit["id"] for unit in units}) != len(units):
        raise ValueError("off-policy units are not consume-once unique")
    return units


def _unit_loss(model, unit: dict) -> tuple[torch.Tensor, float, int, int]:
    positions = [int(value) for value in unit["positions"]]
    first = min(positions)
    end = max(positions) + 1
    completion = unit["completion_ids"][:end]
    ids = torch.tensor(
        [unit["prompt_ids"] + completion], dtype=torch.long, device=model.device
    )
    tail = end - first
    logits = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=tail + 1,
        use_cache=False,
    ).logits[0, -(tail + 1):-1]
    relative = torch.tensor(
        [position - first for position in positions], dtype=torch.long, device=model.device
    )
    selected = logits.index_select(0, relative).float()
    labels = torch.tensor(
        [completion[position] for position in positions],
        dtype=torch.long,
        device=model.device,
    )
    loss = torch.nn.functional.cross_entropy(selected, labels, reduction="mean")
    with torch.no_grad():
        probs = torch.softmax(selected, dim=-1)
        entropy = float(
            -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1).mean().item()
        )
    return loss, entropy, int(ids.shape[1]), len(positions)


def _probe(model, units: list[dict]) -> dict:
    model.eval()
    losses, entropies = [], []
    with torch.inference_mode():
        for unit in units:
            loss, entropy, _, _ = _unit_loss(model, unit)
            losses.append(float(loss.detach().cpu()))
            entropies.append(entropy)
    return {
        "mean_loss": sum(losses) / len(losses),
        "mean_entropy": sum(entropies) / len(entropies),
        "unit_count": len(units),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--round-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-initial-loss", type=float)
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    config, config_path = load_config(args.config)
    cfg = config["mopd"]
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not (args.base_model / "merge_receipt.json").is_file():
        raise SystemExit("off-policy base must be an explicitly merged composite")
    manifest = json.loads(args.round_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "online_advantage_training_round":
        raise SystemExit("invalid off-policy round manifest")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise SystemExit("off-policy round/config mismatch")
    if int(manifest.get("round", -1)) != args.round:
        raise SystemExit("off-policy round index mismatch")
    expected = int(cfg["updates_per_round"]) * int(cfg["grad_accum"])
    if len(manifest.get("units") or []) != expected:
        raise SystemExit("off-policy manifest does not contain the matched unit count")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    units = _make_units(
        manifest,
        tokenizer,
        max_positions=int(cfg["max_target_positions"]),
        max_length=int(cfg["max_length"]),
    )
    random.Random(args.seed + args.round * 1000).shuffle(units)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
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
            r=int(cfg["rank"]),
            lora_alpha=int(cfg["alpha"]),
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    probe_units = sorted(units, key=lambda value: value["id"])[:8]
    initial_probe = _probe(model, probe_units)
    scale = _loss_scale(initial_probe["mean_loss"], args.target_initial_loss)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    grad_accum = int(cfg["grad_accum"])
    completed_updates = 0
    unsafe_reason = None
    losses, entropies, logs = [], [], []
    token_ledger = {"forward_input_tokens": 0, "target_positions": 0}
    started = time.perf_counter()
    model.train()
    for micro_index, unit in enumerate(units):
        loss, entropy, input_tokens, positions = _unit_loss(model, unit)
        raw = float(loss.detach().cpu())
        if not math.isfinite(raw) or not math.isfinite(entropy):
            unsafe_reason = "non_finite_loss_or_entropy"
            break
        (loss * scale / grad_accum).backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable
        ):
            unsafe_reason = "non_finite_gradient"
            break
        losses.append(raw)
        entropies.append(entropy)
        token_ledger["forward_input_tokens"] += input_tokens
        token_ledger["target_positions"] += positions
        if (micro_index + 1) % grad_accum == 0:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"]))
            )
            if not math.isfinite(grad_norm):
                unsafe_reason = "non_finite_gradient_norm"
                break
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_updates += 1
            logs.append(
                {
                    "update": completed_updates,
                    "mean_raw_ce": sum(losses[-grad_accum:]) / grad_accum,
                    "mean_entropy": sum(entropies[-grad_accum:]) / grad_accum,
                    "gradient_norm": grad_norm,
                }
            )
            print(json.dumps(logs[-1], sort_keys=True), flush=True)
    final_probe = _probe(model, probe_units)
    expected_updates = int(cfg["updates_per_round"])
    gate_passed = unsafe_reason is None and completed_updates == expected_updates
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    unit_ledger = [
        {
            "micro_step": index + 1,
            "sample_id": unit["id"],
            "role": unit["role"],
            "kind": unit["kind"],
            "target_policy": unit["target_policy"],
            "prompt_tokens_truncated": int(unit["prompt_tokens_truncated"]),
            "target_positions": len(unit["positions"]),
            "target_terminal_score": unit["target_terminal_score"],
        }
        for index, unit in enumerate(units)
    ]
    receipt = {
        "schema_version": 1,
        "method": "offpolicy_best_selection_continuation_sft",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_model": str(args.base_model.resolve()),
        "base_merge_receipt_sha256": sha256_file(args.base_model / "merge_receipt.json"),
        "round_manifest": str(args.round_manifest.resolve()),
        "round_manifest_sha256": sha256_file(args.round_manifest),
        "round": args.round,
        "seed": args.seed,
        "requested_updates": expected_updates,
        "completed_updates": completed_updates,
        "consume_once_units": len(units),
        "consume_once_verified": len({row["sample_id"] for row in unit_ledger}) == len(unit_ledger),
        "assignment_sha256": hashlib.sha256(
            json.dumps(unit_ledger, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "initial_probe": initial_probe,
        "final_probe": final_probe,
        "target_initial_loss": args.target_initial_loss,
        "backward_loss_scale": scale,
        "mean_cross_entropy": sum(losses) / len(losses) if losses else None,
        "mean_entropy_during_training": sum(entropies) / len(entropies) if entropies else None,
        "round_gate": {
            "passed": gate_passed,
            "unsafe_reason": unsafe_reason,
            "completed_all_updates": completed_updates == expected_updates,
        },
        "token_ledger": token_ledger,
        "unit_ledger": unit_ledger,
        "logs": logs,
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "hyperparameters": {
            key: cfg[key]
            for key in (
                "learning_rate", "rank", "alpha", "grad_accum", "max_length",
                "max_target_positions", "max_grad_norm",
            )
        },
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key not in {"logs", "unit_ledger"}}, indent=2))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
