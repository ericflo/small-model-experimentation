#!/usr/bin/env python3
"""Train matched sparse thought-pivot objectives.

`demote` is the published pairwise FTPO objective on confident wrong turns.
`uplift` raises only the empirically successful sibling by a bounded amount;
the failed token is treated as a tightly reference-tethered non-target.

  ../../.venv/bin/python scripts/train_sparse.py --arm uplift --out <dir>
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

EXP = Path(__file__).resolve().parents[1]
OOM_ERRORS = (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", RuntimeError))


def load_rows(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def final_logits(backbone, head, row: dict, device) -> torch.Tensor:
    ids = torch.tensor(row["context_ids"], dtype=torch.long, device=device).unsqueeze(0)
    attention = torch.ones_like(ids)
    hidden = backbone(input_ids=ids, attention_mask=attention, use_cache=False).last_hidden_state
    return head(hidden[:, -1, :]).float()


def masks_and_ids(z: torch.Tensor, row: dict, *, include_rejected: bool, device):
    chosen = torch.tensor(row["chosen_ids"], dtype=torch.long, device=device)
    rejected = torch.tensor(int(row["rejected_id"]), dtype=torch.long, device=device)
    target_mask = torch.zeros_like(z, dtype=torch.bool)
    target_mask[0, chosen] = True
    if include_rejected:
        target_mask[0, rejected] = True
    return chosen, rejected, target_mask


def tether(z: torch.Tensor, z_ref: torch.Tensor, target_mask: torch.Tensor,
           cfg: dict) -> tuple[torch.Tensor, float, float]:
    diff = z - z_ref
    nontarget = ~target_mask
    mse_nt = (diff.square() * nontarget).sum() / nontarget.sum()
    excess = torch.clamp((diff * target_mask).abs() - float(cfg["tau_mse_target"]), min=0)
    mse_t = excess.square().sum() / target_mask.sum().clamp(min=1)
    total = float(cfg["lambda_mse"]) * mse_nt + float(cfg["lambda_mse_target"]) * mse_t
    return total, float(mse_nt.detach()), float(mse_t.detach())


def objective(z: torch.Tensor, z_ref: torch.Tensor, row: dict, arm: str,
              cfg: dict, device) -> tuple[torch.Tensor, dict]:
    chosen, rejected, target_mask = masks_and_ids(
        z, row, include_rejected=(arm == "demote"), device=device)
    if arm == "demote":
        epsilon = float(cfg["demote_margin_logits"])
        delta = z[0, chosen] - z[0, rejected]
        weight = torch.clamp((epsilon - delta) / epsilon, 0, 1)
        primary = (F.softplus(epsilon - delta) * weight).mean()
        hit = float((delta > 0).float().mean().detach())
        mean_signal = float(delta.mean().detach())
        signal_name = "chosen_minus_rejected"
    else:
        target = float(cfg["uplift_gain_logits"])
        gain = z[0, chosen] - z_ref[0, chosen]
        weight = torch.clamp((target - gain) / target, 0, 1)
        primary = (F.softplus(target - gain) * weight).mean()
        hit = float((gain >= target).float().mean().detach())
        mean_signal = float(gain.mean().detach())
        signal_name = "chosen_gain_vs_reference"
    tether_loss, mse_nt, mse_t = tether(z, z_ref, target_mask, cfg)
    return primary + tether_loss, {
        "primary_loss": float(primary.detach()), "mse_nt": mse_nt, "mse_t": mse_t,
        "hit": hit, signal_name: mean_signal,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["demote", "uplift", "uplift_shuffled"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke-rows", type=int, default=0)
    args = parser.parse_args()

    cfg_all = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    cfg = cfg_all["train"]
    row_name = "rows_shuffled_selected.jsonl.gz" if args.arm == "uplift_shuffled" \
        else "rows_real_selected.jsonl.gz"
    rows = load_rows(EXP / "data" / row_name)
    if args.smoke_rows:
        rows = rows[:args.smoke_rows]
    elif len(rows) < int(cfg_all["geometry"]["min_rows_gate"]):
        raise SystemExit("P0 geometry gate not met")
    rows = [r for r in rows if len(r["context_ids"]) + 1 <= int(cfg["max_seq_length"])]
    if not rows:
        raise SystemExit("no rows")
    objective_arm = "demote" if args.arm == "demote" else "uplift"

    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = cfg_all["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["id"], revision=model_cfg["revision"], trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora = LoraConfig(
        r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]), bias="none",
        target_modules=list(cfg["target_modules"]))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    base = model.get_base_model()
    backbone, head = base.model, base.lm_head
    device = next(model.parameters()).device

    # Longest-row forward is a fail-fast memory and exact-channel smoke.
    longest = max(rows, key=lambda r: len(r["context_ids"]))
    with torch.no_grad():
        _ = final_logits(backbone, head, longest, device)
    print(f"[train:{args.arm}] rows={len(rows)} max_ctx={len(longest['context_ids'])}", flush=True)

    per_device = 1  # frozen: batching is scientifically invalid on this architecture
    accum = max(1, int(cfg["effective_batch"]) // per_device)
    total_micro = len(rows) * int(cfg["num_epochs"])
    total_steps = math.ceil(total_micro / accum)
    warmup = int(total_steps * float(cfg["warmup_ratio"]))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params, lr=float(cfg["learning_rate"]), weight_decay=float(cfg["weight_decay"]))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(warmup, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    rng = np.random.default_rng(int(cfg["seed"]))
    history: list[dict] = []
    started = time.time()
    step = micro = 0
    hit_window: list[float] = []
    metric_window: list[dict] = []
    stopped_early = False
    optimizer.zero_grad(set_to_none=True)

    def optimizer_step() -> None:
        nonlocal step, stopped_early, hit_window, metric_window
        torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        step += 1
        mean_hit = float(np.mean(hit_window)) if hit_window else 0.0
        if step % int(cfg["log_every_steps"]) == 0 or step == total_steps:
            entry = {
                "step": step, "of": total_steps, "hit_rate": mean_hit,
                "primary_loss": float(np.mean([m["primary_loss"] for m in metric_window])),
                "mse_nt": float(np.mean([m["mse_nt"] for m in metric_window])),
                "mse_t": float(np.mean([m["mse_t"] for m in metric_window])),
                "lr": scheduler.get_last_lr()[0], "elapsed_s": time.time() - started,
            }
            history.append(entry); print(f"[train:{args.arm}] {entry}", flush=True)
        min_step = int(total_steps * float(cfg["early_stop_min_progress"]))
        if mean_hit >= float(cfg["early_stop_hit_rate"]) and step >= min_step:
            stopped_early = True
            print(f"[train:{args.arm}] safety stop at hit_rate={mean_hit:.3f}", flush=True)
        hit_window = []; metric_window = []

    for _epoch in range(int(cfg["num_epochs"])):
        for index in rng.permutation(len(rows)):
            row = rows[int(index)]
            try:
                z = final_logits(backbone, head, row, device)
                with torch.no_grad(), model.disable_adapter():
                    z_ref = final_logits(backbone, head, row, device)
                loss, metrics = objective(z, z_ref, row, objective_arm, cfg, device)
                (loss / accum).backward()
            except OOM_ERRORS as exc:
                raise SystemExit(f"accelerator failure at ctx={len(row['context_ids'])}: {exc}") from exc
            micro += 1; hit_window.append(metrics["hit"]); metric_window.append(metrics)
            if micro % accum == 0:
                optimizer_step()
            if stopped_early:
                break
        if stopped_early:
            break
    if micro % accum and not stopped_early:
        optimizer_step()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out)); tokenizer.save_pretrained(str(args.out))
    summary = {
        "arm": args.arm, "objective": objective_arm, "rows": len(rows),
        "steps_run": step, "total_steps": total_steps, "stopped_early": stopped_early,
        "wall_s": time.time() - started, "history": history, "config": cfg,
    }
    (args.out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    (EXP / "runs").mkdir(parents=True, exist_ok=True)
    (EXP / "runs" / f"train_summary_{args.arm}.json").write_text(json.dumps(summary, indent=2))
    print(f"[train:{args.arm}] saved {args.out} in {summary['wall_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
