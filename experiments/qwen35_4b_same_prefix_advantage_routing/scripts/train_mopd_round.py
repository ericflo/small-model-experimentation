#!/usr/bin/env python3
"""Train one matched advantage-routed MOPD or dense-routing control round."""

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
from mopd_loss import sparse_teacher_topk_reverse_kl  # noqa: E402


TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
ARMS = ("primary", "shuffled", "coarse", "fixed_deep")


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


def _target_assignments(samples: list[dict], arm: str, seed: int, quick_levels=(1, 2)) -> dict[str, str]:
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm}")
    assignments = {}
    capability = [sample for sample in samples if sample["meta"]["role"] == "capability"]
    if arm == "shuffled":
        # Break the state/teacher relation without changing either teacher's
        # atom/episode quota.  Mixing across kinds would give the shuffled arm
        # a different task geometry from the primary arm.
        shuffled_map = {}
        rng = random.Random(seed)
        for kind in sorted({str(sample["meta"]["kind"]) for sample in capability}):
            ordered = sorted(
                [sample for sample in capability if str(sample["meta"]["kind"]) == kind],
                key=lambda sample: sample["id"],
            )
            labels = [str(sample["meta"]["primary_teacher"]) for sample in ordered]
            shuffled = list(labels)
            rng.shuffle(shuffled)
            if len(set(labels)) > 1 and shuffled == labels:
                shuffled = shuffled[1:] + shuffled[:1]
            shuffled_map.update(
                {sample["id"]: label for sample, label in zip(ordered, shuffled)}
            )
        original = {
            sample["id"]: str(sample["meta"]["primary_teacher"])
            for sample in capability
        }
        if shuffled_map == original:
            raise ValueError(
                "shuffled-route control cannot break routing while preserving state-kind quotas"
            )
    else:
        shuffled_map = {}
    for sample in samples:
        if sample["meta"]["role"] == "anchor":
            target = "soup"
        elif arm == "primary":
            target = str(sample["meta"]["primary_teacher"])
        elif arm == "shuffled":
            target = shuffled_map[sample["id"]]
        elif arm == "coarse":
            target = (
                "quick"
                if sample["meta"]["kind"] == "atom"
                and int(sample["meta"]["level"]) in set(quick_levels)
                else "deep"
            )
        else:
            target = "deep"
        if target not in sample["targets"]:
            raise ValueError(f"target {target} absent for {sample['id']}")
        assignments[sample["id"]] = target
    return assignments


def _training_units(
    samples: list[dict],
    assignments: dict[str, str],
    seed: int,
    required: int | None = None,
) -> list[dict]:
    if len({sample["id"] for sample in samples}) != len(samples):
        raise ValueError("training samples must be consume-once unique")
    if set(assignments) != {sample["id"] for sample in samples}:
        raise ValueError("target assignments do not cover exact samples")
    units = [{"sample": sample, "target": assignments[sample["id"]]} for sample in samples]
    required = len(units) if required is None else int(required)
    if not 1 <= required <= len(units):
        raise ValueError(f"required units must be in [1, {len(units)}], got {required}")
    rng = random.Random(seed)
    if required == len(units):
        rng.shuffle(units)
        return units

    # A short locality run remains a proportional miniature of the full
    # quick/deep/anchor mixture instead of taking an accidental prefix after a
    # global shuffle.
    buckets = {
        target: [unit for unit in units if unit["target"] == target]
        for target in ("quick", "deep", "soup")
    }
    total = len(units)
    exact = {target: required * len(values) / total for target, values in buckets.items()}
    counts = {target: math.floor(value) for target, value in exact.items()}
    remaining = required - sum(counts.values())
    order = sorted(
        buckets,
        key=lambda target: (exact[target] - counts[target], target == "quick", target),
        reverse=True,
    )
    for target in order[:remaining]:
        counts[target] += 1
    selected = []
    for target, values in buckets.items():
        rng.shuffle(values)
        if counts[target] > len(values):
            raise ValueError(f"insufficient {target} units for proportional subset")
        selected.extend(values[:counts[target]])
    rng.shuffle(selected)
    if len(selected) != required:
        raise AssertionError("proportional unit selection count mismatch")
    return selected


def _unit_metrics(model, unit: dict, top_k: int) -> tuple[torch.Tensor, float, float, int, int]:
    sample = unit["sample"]
    target = sample["targets"][unit["target"]]
    positions = sample["positions"].to(dtype=torch.long).tolist()
    first = min(positions)
    end = max(positions) + 1
    prompt = sample["prompt_ids"].to(dtype=torch.long).tolist()
    completion = sample["completion_ids"].to(dtype=torch.long).tolist()[:end]
    ids = torch.tensor([prompt + completion], dtype=torch.long, device=model.device)
    tail = end - first
    outputs = model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=tail + 1,
        use_cache=False,
    )
    prediction = outputs.logits[0, -(tail + 1):-1]
    relative = torch.tensor(
        [position - first for position in positions], dtype=torch.long, device=model.device
    )
    selected = prediction.index_select(0, relative)
    teacher_indices = target["indices"]
    teacher_log_probs = target["log_probs"]
    loss = sparse_teacher_topk_reverse_kl(
        selected, teacher_indices, teacher_log_probs, reduction="mean"
    )
    with torch.no_grad():
        student_indices = torch.topk(selected.float(), k=top_k, dim=-1).indices.cpu()
        teacher_cpu = teacher_indices.to(dtype=torch.long)
        overlap = (
            (student_indices.unsqueeze(-1) == teacher_cpu.unsqueeze(-2))
            .any(dim=-1)
            .float()
            .mean()
        )
        probs = torch.softmax(selected.float(), dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-30))).sum(dim=-1).mean()
    return loss, float(overlap), float(entropy), int(ids.shape[1]), len(positions)


def _probe_units(units: list[dict]) -> list[dict]:
    buckets = {"quick": [], "deep": [], "soup": []}
    for unit in sorted(units, key=lambda value: value["sample"]["id"]):
        buckets[unit["target"]].append(unit)
    # The exact mix is deterministic and includes every target class that is
    # present. It is a locality/dynamics probe, not a performance estimator.
    selected = buckets["quick"][:3] + buckets["deep"][:3] + buckets["soup"][:2]
    if not selected:
        raise ValueError("no probe units")
    return selected


def _evaluate_probe(model, units: list[dict], top_k: int) -> dict:
    model.eval()
    losses, overlaps, entropies = [], [], []
    with torch.inference_mode():
        for unit in units:
            loss, overlap, entropy, _, _ = _unit_metrics(model, unit, top_k)
            losses.append(float(loss.detach().cpu()))
            overlaps.append(overlap)
            entropies.append(entropy)
    return {
        "mean_loss": sum(losses) / len(losses),
        "mean_topk_overlap": sum(overlaps) / len(overlaps),
        "mean_entropy": sum(entropies) / len(entropies),
        "unit_losses": losses,
        "unit_count": len(units),
    }


def main() -> int:
    # Keep the pure assignment/loss helpers importable in the repository's CPU
    # validation environment.  PEFT is required only by the actual GPU stage.
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--updates", type=int)
    parser.add_argument("--target-initial-loss", type=float)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not (args.base_model / "merge_receipt.json").is_file():
        raise SystemExit("round base must be an explicitly merged composite")
    receipt_path = args.target_cache.with_suffix(args.target_cache.suffix + ".receipt.json")
    cache_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if cache_receipt["cache_sha256"] != sha256_file(args.target_cache):
        raise SystemExit("target cache checksum mismatch")
    payload = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    if int(payload["round"]) != args.round:
        raise SystemExit("target cache round mismatch")
    samples = payload["samples"]
    cfg = config["mopd"]
    updates = int(args.updates if args.updates is not None else cfg["updates_per_round"])
    if not 1 <= updates <= int(cfg["updates_per_round"]):
        raise SystemExit("updates outside frozen round range")
    grad_accum = int(cfg["grad_accum"])
    required = updates * grad_accum
    if len(samples) < required:
        raise SystemExit(f"only {len(samples)} samples for {required} consume-once units")
    assignments = _target_assignments(
        samples,
        args.arm,
        args.seed + args.round * 1000,
        tuple(int(value) for value in config["strata"]["quick_atom_levels"]),
    )
    units = _training_units(
        samples, assignments, args.seed + args.round * 1000, required=required
    )
    if len({unit["sample"]["id"] for unit in units}) != len(units):
        raise SystemExit("consume-once violation")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
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
    top_k = int(cfg["top_k"])
    probe_units = _probe_units(units)
    initial_probe = _evaluate_probe(model, probe_units, top_k)
    loss_scale = _matched_loss_scale(initial_probe["mean_loss"], args.target_initial_loss)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]))
    optimizer.zero_grad(set_to_none=True)
    logs = []
    raw_losses = []
    overlaps = []
    entropies = []
    token_ledger = {"forward_input_tokens": 0, "target_positions": 0}
    unsafe_reason = None
    completed_updates = 0
    started = time.perf_counter()
    model.train()
    for micro_index, unit in enumerate(units):
        loss, overlap, entropy, forward_tokens, positions = _unit_metrics(model, unit, top_k)
        raw = float(loss.detach().cpu())
        if not math.isfinite(raw) or not math.isfinite(overlap) or not math.isfinite(entropy):
            unsafe_reason = "non_finite_loss_or_metric"
            break
        scaled = loss * loss_scale / grad_accum
        scaled.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in trainable
        ):
            unsafe_reason = "non_finite_gradient"
            break
        raw_losses.append(raw)
        overlaps.append(overlap)
        entropies.append(entropy)
        token_ledger["forward_input_tokens"] += forward_tokens
        token_ledger["target_positions"] += positions
        if (micro_index + 1) % grad_accum == 0:
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, float(cfg["max_grad_norm"])))
            if not math.isfinite(grad_norm):
                unsafe_reason = "non_finite_gradient_norm"
                break
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            completed_updates += 1
            logs.append(
                {
                    "update": completed_updates,
                    "mean_raw_loss": sum(raw_losses[-grad_accum:]) / grad_accum,
                    "mean_topk_overlap": sum(overlaps[-grad_accum:]) / grad_accum,
                    "mean_entropy": sum(entropies[-grad_accum:]) / grad_accum,
                    "gradient_norm": grad_norm,
                }
            )
            print(json.dumps(logs[-1], sort_keys=True), flush=True)
    final_probe = _evaluate_probe(model, probe_units, top_k)
    mean_loss = sum(raw_losses) / len(raw_losses) if raw_losses else float("inf")
    overlap_passed = (
        final_probe["mean_topk_overlap"] + 1e-7 >= initial_probe["mean_topk_overlap"]
        if bool(cfg["require_non_decreasing_student_teacher_topk_overlap"])
        else True
    )
    gate_passed = (
        unsafe_reason is None
        and completed_updates == updates
        and math.isfinite(mean_loss)
        and mean_loss <= float(cfg["maximum_round_mean_corrected_topk_loss"])
        and overlap_passed
    )
    if unsafe_reason is None and not overlap_passed:
        unsafe_reason = "student_teacher_topk_overlap_decreased"
    if unsafe_reason is None and mean_loss > float(cfg["maximum_round_mean_corrected_topk_loss"]):
        unsafe_reason = "mean_corrected_topk_loss_exceeded"
    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    assignment_payload = sorted(assignments.items())
    unit_ledger = [
        {
            "micro_step": index + 1,
            "sample_id": unit["sample"]["id"],
            "target": unit["target"],
            "role": unit["sample"]["meta"]["role"],
            "kind": unit["sample"]["meta"]["kind"],
            "level": int(unit["sample"]["meta"]["level"]),
            "target_positions": int(unit["sample"]["positions"].numel()),
        }
        for index, unit in enumerate(units)
    ]
    receipt = {
        "schema_version": 2,
        "method": "advantage_routed_corrected_teacher_topk_reverse_kl",
        "arm": args.arm,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "base_model": str(args.base_model.resolve()),
        "base_merge_receipt_sha256": sha256_file(args.base_model / "merge_receipt.json"),
        "target_cache": str(args.target_cache.resolve()),
        "target_cache_sha256": sha256_file(args.target_cache),
        "round": args.round,
        "seed": args.seed,
        "requested_updates": updates,
        "completed_updates": completed_updates,
        "consume_once_units": len(units),
        "consume_once_verified": (
            len({row["sample_id"] for row in unit_ledger}) == len(unit_ledger)
        ),
        "assignment_sha256": hashlib.sha256(
            json.dumps(assignment_payload, separators=(",", ":")).encode()
        ).hexdigest(),
        "target_counts": {
            policy: sum(unit["target"] == policy for unit in units)
            for policy in ("quick", "deep", "soup")
        },
        "initial_probe": initial_probe,
        "final_probe": final_probe,
        "target_initial_loss": args.target_initial_loss,
        "backward_loss_scale": loss_scale,
        "mean_corrected_topk_loss": mean_loss,
        "mean_topk_overlap_during_training": (
            sum(overlaps) / len(overlaps) if overlaps else None
        ),
        "mean_entropy_during_training": (
            sum(entropies) / len(entropies) if entropies else None
        ),
        "round_gate": {
            "passed": gate_passed,
            "unsafe_reason": unsafe_reason,
            "maximum_mean_loss": float(cfg["maximum_round_mean_corrected_topk_loss"]),
            "overlap_non_decreasing": overlap_passed,
            "completed_all_updates": completed_updates == updates,
        },
        "token_ledger": token_ledger,
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "hyperparameters": {
            key: cfg[key]
            for key in (
                "top_k", "learning_rate", "rank", "alpha", "grad_accum",
                "max_length", "max_target_positions", "max_grad_norm",
                "maximum_round_mean_corrected_topk_loss",
            )
        },
        "unit_ledger": unit_ledger,
        "logs": logs,
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in receipt.items() if key != "logs"}, indent=2))
    return 0 if gate_passed else 3


if __name__ == "__main__":
    raise SystemExit(main())
