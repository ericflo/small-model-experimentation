#!/usr/bin/env python3
"""Train a Qwen compiler with on-policy local-repair targets.

The experiment starts from a QLoRA-attached Qwen numeric compiler, runs the
compiler on its own training prompts, enumerates local program repairs around
those on-policy predictions, and fine-tunes the same compiler toward verified
repaired programs. This tests whether repair headroom can become a better
compiler policy rather than remaining a target-aware search result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from qwen_onpolicy_repair_compiler_core import (
    DirectAnswerHead,
    ExampleSet,
    ProgramCompiler,
    TextProgramGenerator,
    TransitionExecutor,
    collate_examples,
    dtype_from_string,
    ensure_pad_token,
    evaluate,
    forward_hidden,
    state_trajectory_loss,
    trace_losses,
    verifier_repair_batch,
)


ROOT = Path("experiments/qwen_onpolicy_repair_compiler")
RUNS = ROOT / "runs"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
CHECKPOINT_ROOT = Path("large_artifacts/qwen_onpolicy_repair_compiler/checkpoints")
DEFAULT_COMPILER_CHECKPOINT = CHECKPOINT_ROOT / "fixed_compiler_step00800"


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def clean_render_dataset(
    tokenizer: Any,
    run_args: Dict[str, Any],
    n_examples: int,
    length: int,
    seed: int,
    template_modes: Sequence[str],
) -> ExampleSet:
    gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], seed, "standard")
    modes = list(template_modes)
    if not modes:
        raise ValueError("at least one template mode is required")
    examples = []
    for i in range(n_examples):
        spec = gen.make_spec(length, length)
        examples.append(gen.render_spec(spec, modes[i % len(modes)]))
    return ExampleSet(examples)


def make_datasets(tokenizer: Any, run_args: Dict[str, Any], args: argparse.Namespace) -> Dict[str, ExampleSet]:
    train_modes = [x.strip() for x in args.train_template_modes.split(",") if x.strip()]
    datasets: Dict[str, ExampleSet] = {
        "train_len24": clean_render_dataset(tokenizer, run_args, args.train_examples, args.train_length, args.seed + 11, train_modes),
        "val_len24": clean_render_dataset(tokenizer, run_args, args.val_examples, args.eval_length, args.seed + 29, train_modes),
    }
    for mode_index, mode in enumerate(["standard", "paraphrase"]):
        gen = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], args.eval_seed + 101 * mode_index, mode)
        datasets[f"fresh_{mode}_len{args.eval_length}"] = gen.dataset(args.eval_examples, args.eval_length, args.eval_length)
    gen_pair = TextProgramGenerator(tokenizer, run_args["modulus"], run_args["max_steps"], args.eval_seed + 503, "mixed")
    datasets[f"fresh_paired_len{args.eval_length}"] = gen_pair.paired_dataset(
        args.eval_pairs, args.eval_length, args.eval_length, ["standard", "paraphrase"]
    )
    return datasets


def optimizer_for_trainable(params: Sequence[nn.Parameter], lr: float, weight_decay: float) -> torch.optim.Optimizer:
    trainable = [p for p in params if p.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable parameters found")
    return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)


def load_trainable_compiler(
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[Any, nn.Module, Optional[DirectAnswerHead], ProgramCompiler, Dict[str, Any]]:
    checkpoint_dir = Path(args.compiler_checkpoint)
    heads_path = checkpoint_dir / "heads.pt"
    adapter_dir = checkpoint_dir / "adapter"
    if not heads_path.exists():
        raise FileNotFoundError(heads_path)
    if not adapter_dir.exists():
        raise FileNotFoundError(adapter_dir)

    heads = torch.load(heads_path, map_location=device)
    run_args = dict(heads["args"])
    run_args["hidden_dim"] = int(heads["hidden_dim"])
    run_args["eval_batch_size"] = args.qwen_batch_size
    run_args["repair_topk"] = args.repair_topk
    run_args["repair_max_edits"] = args.repair_max_edits
    run_args["repair_max_pair_arg_slots"] = args.repair_max_pair_arg_slots
    run_args["max_length"] = args.max_length

    tokenizer = AutoTokenizer.from_pretrained(run_args["model_id"], trust_remote_code=True, use_fast=True)
    ensure_pad_token(tokenizer)
    tokenizer.padding_side = "right"

    dtype = dtype_from_string(run_args.get("torch_dtype", "bf16"))
    quantization_config = None
    if run_args.get("load_in_4bit", False):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype if dtype != torch.float32 else torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    common: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "device_map": run_args.get("device_map", "auto") if torch.cuda.is_available() else None,
    }
    if quantization_config is not None:
        common["quantization_config"] = quantization_config
    common = {k: v for k, v in common.items() if v is not None}

    print(f"[load] {run_args['model_id']} + {adapter_dir}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(run_args["model_id"], **common)
    model.config.use_cache = False
    if run_args.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=bool(args.gradient_checkpointing))
    model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    direct = None
    if heads.get("direct") is not None:
        direct = DirectAnswerHead(heads["hidden_dim"], run_args["modulus"], run_args["head_width"]).to(device)
        direct.load_state_dict(heads["direct"])
        direct.eval()
        for p in direct.parameters():
            p.requires_grad_(False)

    compiler = ProgramCompiler(
        heads["hidden_dim"],
        run_args["modulus"],
        run_args["head_width"],
        run_args["max_steps"],
        run_args["rank_temperature"],
        run_args["arg_reader_mode"],
        run_args["arg_window"],
        run_args["arg_distance_temperature"],
    ).to(device)
    compiler.load_state_dict(heads["compiler"])
    compiler.train()
    return tokenizer, model, direct, compiler, run_args


def repair_targets_from_current_policy(
    dataset: ExampleSet,
    tokenizer: Any,
    model: nn.Module,
    compiler: ProgramCompiler,
    run_args: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
    split: str,
) -> Dict[str, Any]:
    model.eval()
    compiler.eval()
    pad_id = int(tokenizer.pad_token_id)
    target_init_rows: List[torch.Tensor] = []
    target_op_rows: List[torch.Tensor] = []
    target_arg_rows: List[torch.Tensor] = []
    found_rows: List[torch.Tensor] = []
    changed_rows: List[torch.Tensor] = []
    candidate_rows: List[torch.Tensor] = []
    verified_rows: List[torch.Tensor] = []
    active_rows: List[torch.Tensor] = []
    source_counts = {"repair": 0, "gold": 0, "gold_fallback": 0, "base_fallback": 0, "skipped": 0}

    for start in range(0, len(dataset), args.qwen_batch_size):
        chunk = dataset.examples[start : start + args.qwen_batch_size]
        batch = collate_examples(chunk, pad_id, int(run_args["max_steps"]), args.max_length, device)
        with torch.no_grad():
            hidden = forward_hidden(model, batch)
            init_logits, op_logits, arg_logits = compiler(
                hidden,
                batch["hidden_mask"],
                batch["num_values"],
                batch["op_values"],
                return_scores=False,
            )
            repaired = verifier_repair_batch(init_logits, op_logits, arg_logits, batch["lengths"], batch["answer"], batch["states"], args)
        base_init = init_logits.argmax(dim=-1)
        base_ops = op_logits.argmax(dim=-1)
        base_args = arg_logits.argmax(dim=-1)
        target_init = repaired["init"].clone()
        target_ops = repaired["ops"].clone()
        target_args = repaired["args"].clone()
        use_mask = torch.ones_like(repaired["found_verified"], dtype=torch.bool)

        for row in range(target_init.shape[0]):
            if args.target_mode == "gold_only":
                target_init[row] = batch["init_value"][row]
                target_ops[row] = torch.where(batch["ops"][row].ne(-100), batch["ops"][row], target_ops[row])
                target_args[row] = torch.where(batch["args"][row].ne(-100), batch["args"][row], target_args[row])
                source_counts["gold"] += 1
                continue
            if bool(repaired["found_verified"][row].item()):
                source_counts["repair"] += 1
                continue
            if args.target_mode == "repair_or_gold":
                target_init[row] = batch["init_value"][row]
                target_ops[row] = torch.where(batch["ops"][row].ne(-100), batch["ops"][row], target_ops[row])
                target_args[row] = torch.where(batch["args"][row].ne(-100), batch["args"][row], target_args[row])
                source_counts["gold_fallback"] += 1
            elif args.target_mode == "repair_or_base":
                target_init[row] = base_init[row]
                target_ops[row] = base_ops[row]
                target_args[row] = base_args[row]
                source_counts["base_fallback"] += 1
            elif args.target_mode == "repair_only":
                use_mask[row] = False
                source_counts["skipped"] += 1
            else:
                raise ValueError(args.target_mode)

        target_init_rows.append(target_init.detach().cpu())
        target_op_rows.append(target_ops.detach().cpu())
        target_arg_rows.append(target_args.detach().cpu())
        found_rows.append(repaired["found_verified"].detach().cpu())
        changed_rows.append(repaired["changed"].detach().cpu())
        candidate_rows.append(repaired["candidate_count"].detach().cpu())
        verified_rows.append(repaired["verified_count"].detach().cpu())
        active_rows.append(use_mask.detach().cpu())
        print(f"[targets] {split} {min(start + len(chunk), len(dataset))}/{len(dataset)}", flush=True)

    return {
        "init": torch.cat(target_init_rows, dim=0),
        "ops": torch.cat(target_op_rows, dim=0),
        "args": torch.cat(target_arg_rows, dim=0),
        "found_verified": torch.cat(found_rows, dim=0),
        "changed": torch.cat(changed_rows, dim=0),
        "candidate_count": torch.cat(candidate_rows, dim=0),
        "verified_count": torch.cat(verified_rows, dim=0),
        "active_mask": torch.cat(active_rows, dim=0),
        "source_counts": source_counts,
    }


def onpolicy_slot_loss(
    init_logits: torch.Tensor,
    op_logits: torch.Tensor,
    arg_logits: torch.Tensor,
    batch: Dict[str, torch.Tensor],
    target_init: torch.Tensor,
    target_ops: torch.Tensor,
    target_args: torch.Tensor,
    target_mask: torch.Tensor,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    target_mask = target_mask.bool()
    active = batch["ops"].ne(-100) & target_mask.view(-1, 1)
    if not bool(target_mask.any().item()):
        zero = init_logits.sum() * 0.0 + op_logits.sum() * 0.0 + arg_logits.sum() * 0.0
        return zero, {"repair_init_loss": 0.0, "repair_op_loss": 0.0, "repair_arg_loss": 0.0}
    init_loss = F.cross_entropy(init_logits[target_mask], target_init[target_mask])
    op_loss = F.cross_entropy(op_logits[active], target_ops[active]) if active.any() else op_logits.sum() * 0.0
    arg_loss = F.cross_entropy(arg_logits[active], target_args[active]) if active.any() else arg_logits.sum() * 0.0
    loss = args.repair_trace_loss_weight * (
        args.init_repair_loss_weight * init_loss
        + args.op_repair_loss_weight * op_loss
        + args.arg_repair_loss_weight * arg_loss
    )
    return loss, {
        "repair_init_loss": float(init_loss.detach().cpu()),
        "repair_op_loss": float(op_loss.detach().cpu()),
        "repair_arg_loss": float(arg_loss.detach().cpu()),
    }


def evaluate_compiler_splits(
    model: nn.Module,
    compiler: ProgramCompiler,
    executor: TransitionExecutor,
    datasets: Dict[str, ExampleSet],
    tokenizer: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    model.eval()
    compiler.eval()
    return {
        split: evaluate("onpolicy_repair_compiler", model, None, compiler, executor, dataset, tokenizer, args, device)
        for split, dataset in datasets.items()
    }


def flatten_metrics(metrics_by_split: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for split, metrics in metrics_by_split.items():
        for key, value in metrics.items():
            if key not in {"variant", "n"}:
                flat[f"{split}_{key}"] = value
    return flat


def rows_from_compiler_metrics(metrics_by_split: Dict[str, Dict[str, float]], run_name: str, phase: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for split, metrics in metrics_by_split.items():
        row = {"run": run_name, "phase": phase, "split": split}
        row.update(metrics)
        rows.append(row)
    return rows


def save_compiler_checkpoint(
    run_dir: Path,
    model: nn.Module,
    compiler: ProgramCompiler,
    run_args: Dict[str, Any],
    args: argparse.Namespace,
    round_index: int,
    metrics: Dict[str, Any],
) -> Dict[str, Any]:
    ckpt_root = CHECKPOINT_ROOT / run_dir.name / f"round{round_index:02d}"
    ckpt_root.mkdir(parents=True, exist_ok=True)
    adapter_dir = ckpt_root / "adapter"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(adapter_dir)
    heads_path = ckpt_root / "heads.pt"
    torch.save(
        {
            "variant": "onpolicy_repair_compiler",
            "args": run_args,
            "experiment_args": vars(args),
            "compiler": compiler.state_dict(),
            "direct": None,
            "hidden_dim": int(run_args["hidden_dim"]),
            "round": int(round_index),
            "metrics": metrics,
        },
        heads_path,
    )
    return {
        "run": run_dir.name,
        "round": round_index,
        "adapter_path": str(adapter_dir),
        "heads_path": str(heads_path),
    }


def train_onpolicy_repair_compiler(
    model: nn.Module,
    compiler: ProgramCompiler,
    executor: TransitionExecutor,
    tokenizer: Any,
    train_set: ExampleSet,
    eval_sets: Dict[str, ExampleSet],
    run_args: Dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    device: torch.device,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, float]], List[Dict[str, Any]]]:
    trainable = [p for p in model.parameters() if p.requires_grad] + [p for p in compiler.parameters() if p.requires_grad]
    opt = optimizer_for_trainable(trainable, args.lr, args.weight_decay)
    rng = random.Random(args.seed + 3300)
    pad_id = int(tokenizer.pad_token_id)
    train_rows: List[Dict[str, Any]] = []
    checkpoint_rows: List[Dict[str, Any]] = []

    metrics_by_split = evaluate_compiler_splits(model, compiler, executor, eval_sets, tokenizer, args, device)
    train_rows.append({"round": 0, "epoch": 0, "phase": "baseline", **flatten_metrics(metrics_by_split)})
    print(
        f"[baseline] val={100.0 * metrics_by_split['val_len24']['executor_accuracy']:.1f}% "
        f"repair={100.0 * metrics_by_split['val_len24']['repair_executor_accuracy']:.1f}%",
        flush=True,
    )

    for round_index in range(1, args.onpolicy_rounds + 1):
        targets = repair_targets_from_current_policy(train_set, tokenizer, model, compiler, run_args, args, device, f"train_round{round_index}")
        order = list(range(len(train_set)))
        for epoch in range(1, args.epochs_per_round + 1):
            rng.shuffle(order)
            totals: Dict[str, float] = {
                "loss": 0.0,
                "repair_init_loss": 0.0,
                "repair_op_loss": 0.0,
                "repair_arg_loss": 0.0,
                "gold_trace_loss": 0.0,
                "executor_loss": 0.0,
                "state_loss": 0.0,
            }
            trained_batches = 0
            model.train()
            compiler.train()

            for start in range(0, len(order), args.train_batch_size):
                idxs = order[start : start + args.train_batch_size]
                examples = [train_set.examples[i] for i in idxs]
                batch = collate_examples(examples, pad_id, int(run_args["max_steps"]), args.max_length, device)
                target_init = targets["init"][idxs].to(device)
                target_ops = targets["ops"][idxs].to(device)
                target_args = targets["args"][idxs].to(device)
                target_mask = targets["active_mask"][idxs].to(device)
                if not bool(target_mask.any().item()):
                    continue

                hidden = forward_hidden(model, batch)
                init_logits, op_logits, arg_logits, _scores = compiler(
                    hidden,
                    batch["hidden_mask"],
                    batch["num_values"],
                    batch["op_values"],
                    return_scores=True,
                )
                loss, parts = onpolicy_slot_loss(init_logits, op_logits, arg_logits, batch, target_init, target_ops, target_args, target_mask, args)

                if args.gold_trace_loss_weight > 0:
                    gold_loss, _gold_parts = trace_losses(
                        init_logits,
                        op_logits,
                        arg_logits,
                        batch,
                        args.init_gold_loss_weight,
                        args.op_gold_loss_weight,
                        args.arg_gold_loss_weight,
                    )
                    loss = loss + args.gold_trace_loss_weight * gold_loss
                    totals["gold_trace_loss"] += float(gold_loss.detach().cpu())

                if args.executor_loss_weight > 0 or args.state_loss_weight > 0:
                    state_probs = executor.soft_trajectory(init_logits, op_logits, arg_logits, batch["lengths"])
                    if args.executor_loss_weight > 0:
                        gather_idx = (batch["lengths"].clamp_min(1) - 1).view(-1, 1, 1).expand(-1, 1, state_probs.shape[-1])
                        answer_probs = state_probs.gather(1, gather_idx).squeeze(1).clamp_min(1e-9)
                        exec_loss = F.nll_loss(answer_probs.log(), batch["answer"])
                        loss = loss + args.executor_loss_weight * exec_loss
                        totals["executor_loss"] += float(exec_loss.detach().cpu())
                    if args.state_loss_weight > 0:
                        st_loss, _st_parts = state_trajectory_loss(state_probs, batch)
                        loss = loss + args.state_loss_weight * st_loss
                        totals["state_loss"] += float(st_loss.detach().cpu())

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
                opt.step()

                totals["loss"] += float(loss.detach().cpu())
                for key in ["repair_init_loss", "repair_op_loss", "repair_arg_loss"]:
                    totals[key] += parts.get(key, 0.0)
                trained_batches += 1

            metrics_by_split = evaluate_compiler_splits(model, compiler, executor, eval_sets, tokenizer, args, device)
            row = {
                "round": round_index,
                "epoch": epoch,
                "phase": "train",
                "train_loss": totals["loss"] / max(1, trained_batches),
                "train_repair_init_loss": totals["repair_init_loss"] / max(1, trained_batches),
                "train_repair_op_loss": totals["repair_op_loss"] / max(1, trained_batches),
                "train_repair_arg_loss": totals["repair_arg_loss"] / max(1, trained_batches),
                "train_gold_trace_loss": totals["gold_trace_loss"] / max(1, trained_batches),
                "train_executor_loss": totals["executor_loss"] / max(1, trained_batches),
                "train_state_loss": totals["state_loss"] / max(1, trained_batches),
                "target_repair_fraction": float(targets["found_verified"].float().mean().item()),
                "target_changed_fraction": float(targets["changed"].float().mean().item()),
                "target_active_fraction": float(targets["active_mask"].float().mean().item()),
                "target_avg_candidates": float(targets["candidate_count"].float().mean().item()),
                "target_avg_verified": float(targets["verified_count"].float().mean().item()),
                **{f"target_source_{k}": v for k, v in targets["source_counts"].items()},
                **flatten_metrics(metrics_by_split),
            }
            train_rows.append(row)
            print(
                f"[train] round={round_index} epoch={epoch} loss={row['train_loss']:.4f} "
                f"val={100.0 * metrics_by_split['val_len24']['executor_accuracy']:.1f}% "
                f"val_repair={100.0 * metrics_by_split['val_len24']['repair_executor_accuracy']:.1f}% "
                f"paired={100.0 * metrics_by_split.get('fresh_paired_len24', metrics_by_split['val_len24'])['executor_accuracy']:.1f}%",
                flush=True,
            )
        checkpoint_rows.append(save_compiler_checkpoint(run_dir, model, compiler, run_args, args, round_index, flatten_metrics(metrics_by_split)))

    return train_rows, metrics_by_split, checkpoint_rows


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a Qwen compiler from on-policy local-repair targets")
    p.add_argument("--run_name", type=str, default="main_onpolicy_repair_s256")
    p.add_argument("--compiler_checkpoint", type=str, default=str(DEFAULT_COMPILER_CHECKPOINT))
    p.add_argument("--train_examples", type=int, default=256)
    p.add_argument("--val_examples", type=int, default=128)
    p.add_argument("--eval_examples", type=int, default=256)
    p.add_argument("--eval_pairs", type=int, default=256)
    p.add_argument("--train_length", type=int, default=24)
    p.add_argument("--eval_length", type=int, default=24)
    p.add_argument("--train_template_modes", type=str, default="standard,paraphrase")
    p.add_argument("--target_mode", type=str, default="repair_or_gold", choices=["repair_or_gold", "repair_or_base", "repair_only", "gold_only"])
    p.add_argument("--onpolicy_rounds", type=int, default=2)
    p.add_argument("--epochs_per_round", type=int, default=1)
    p.add_argument("--train_batch_size", type=int, default=2)
    p.add_argument("--qwen_batch_size", type=int, default=8)
    p.add_argument("--max_length", type=int, default=384)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair_trace_loss_weight", type=float, default=1.0)
    p.add_argument("--init_repair_loss_weight", type=float, default=4.0)
    p.add_argument("--op_repair_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_repair_loss_weight", type=float, default=4.0)
    p.add_argument("--gold_trace_loss_weight", type=float, default=0.15)
    p.add_argument("--init_gold_loss_weight", type=float, default=4.0)
    p.add_argument("--op_gold_loss_weight", type=float, default=1.0)
    p.add_argument("--arg_gold_loss_weight", type=float, default=4.0)
    p.add_argument("--executor_loss_weight", type=float, default=0.2)
    p.add_argument("--state_loss_weight", type=float, default=0.05)
    p.add_argument("--repair_eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair_verifier_mode", type=str, default="state", choices=["state", "answer"])
    p.add_argument("--repair_topk", type=int, default=3)
    p.add_argument("--repair_max_edits", type=int, default=2)
    p.add_argument("--repair_max_pair_arg_slots", type=int, default=24)
    p.add_argument("--repair_include_init", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair_include_ops", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair_include_args", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--repair_same_step_op_arg", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--eval_seed", type=int, default=71001)
    return p


def main() -> None:
    args = build_parser().parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = RUNS / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    tokenizer, model, _direct, compiler, run_args = load_trainable_compiler(args, device)
    args.modulus = int(run_args["modulus"])
    args.max_steps = int(run_args["max_steps"])
    args.eval_batch_size = int(args.qwen_batch_size)

    executor = TransitionExecutor(args.modulus, device)
    datasets = make_datasets(tokenizer, run_args, args)
    train_set = datasets["train_len24"]
    eval_sets = {key: value for key, value in datasets.items() if key != "train_len24"}

    train_rows, metrics_by_split, checkpoint_rows = train_onpolicy_repair_compiler(
        model,
        compiler,
        executor,
        tokenizer,
        train_set,
        eval_sets,
        run_args,
        args,
        run_dir,
        device,
    )

    metric_rows = rows_from_compiler_metrics(metrics_by_split, args.run_name, "final")
    write_csv(run_dir / "train_log.csv", train_rows)
    write_csv(run_dir / "metrics.csv", metric_rows)
    write_csv(run_dir / "checkpoints.csv", checkpoint_rows)
    write_csv(ANALYSIS / "final_metrics.csv", metric_rows)
    write_csv(ROOT / "checkpoint_manifest.csv", checkpoint_rows)

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "args": vars(args),
        "compiler_args": {
            key: run_args.get(key)
            for key in [
                "model_id",
                "modulus",
                "max_steps",
                "head_width",
                "rank_temperature",
                "arg_reader_mode",
                "arg_window",
                "arg_distance_temperature",
            ]
        },
        "train_seconds": time.time() - t0,
    }
    json_dump(
        run_dir / "results.json",
        {
            "metadata": metadata,
            "metrics": metrics_by_split,
            "train_log": train_rows,
            "checkpoints": checkpoint_rows,
        },
    )
    print(f"[done] {run_dir} seconds={metadata['train_seconds']:.1f}", flush=True)


if __name__ == "__main__":
    main()
