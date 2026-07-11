#!/usr/bin/env python3
"""Standalone FTPO trainer (single-position, multi-chosen, two-tier tether).

Run under the repo .venv (torch+peft+transformers):
  ../../.venv/bin/python scripts/train_ftpo.py --arm pivot --out <adapter dir>

Memory/architecture requirements (preregistered):
- final-position logits ONLY: the backbone is called directly for hidden
  states, the last-real-index hidden vector is gathered per row, and lm_head
  is applied to the gathered vectors — full-sequence logits over the 248,320
  vocab are never materialized;
- RIGHT-padded batches (real tokens are a contiguous prefix — safe for the
  hybrid recurrent/linear-attention blocks; left padding is forbidden);
- a padding-equivalence gate asserts batched == batch-of-1 final logits on
  real rows before training (tolerance from config); on failure the trainer
  falls back to batch-of-1 automatically;
- gradient checkpointing on; expandable_segments set in-process.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
import yaml

EXP = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

OOM_ERRORS = (torch.cuda.OutOfMemoryError, getattr(torch, "AcceleratorError", RuntimeError))


def load_rows(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    return rows


def resolve_backbone_and_head(peft_model):
    base = peft_model.get_base_model()
    backbone = getattr(base, "model", None)
    head = getattr(base, "lm_head", None)
    if backbone is None or head is None:
        raise SystemExit("could not resolve backbone/lm_head on the loaded model")
    return backbone, head


def final_logits(backbone, head, batch_rows: list[dict], device) -> torch.Tensor:
    """Right-padded batched forward -> [B, V] final-position logits (float32)."""
    lengths = [len(r["context_ids"]) for r in batch_rows]
    max_len = max(lengths)
    pad_id = 0  # attention-masked; value irrelevant for a right-pad suffix
    input_ids = torch.full((len(batch_rows), max_len), pad_id, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    for i, row in enumerate(batch_rows):
        ids = torch.tensor(row["context_ids"], dtype=torch.long)
        input_ids[i, : ids.numel()] = ids
        attention[i, : ids.numel()] = 1
    input_ids = input_ids.to(device)
    attention = attention.to(device)
    hidden = backbone(input_ids=input_ids, attention_mask=attention,
                      use_cache=False).last_hidden_state
    gather_idx = (torch.tensor(lengths, device=device) - 1).view(-1, 1, 1)
    gathered = hidden.gather(1, gather_idx.expand(-1, 1, hidden.size(-1))).squeeze(1)
    return head(gathered).float()


def ftpo_loss(z: torch.Tensor, z_ref: torch.Tensor, batch_rows: list[dict],
              cfg: dict, device) -> tuple[torch.Tensor, dict]:
    eps = float(cfg["clip_epsilon_logits"])
    lam_nt = float(cfg["lambda_mse"])
    lam_t = float(cfg["lambda_mse_target"])
    tau = float(cfg["tau_mse_target"])

    batch = len(batch_rows)
    max_c = max(len(r["chosen_ids"]) for r in batch_rows)
    chosen = torch.zeros((batch, max_c), dtype=torch.long)
    mask = torch.zeros((batch, max_c), dtype=torch.bool)
    rejected = torch.tensor([r["rejected_id"] for r in batch_rows], dtype=torch.long)
    for i, row in enumerate(batch_rows):
        ids = torch.tensor(row["chosen_ids"], dtype=torch.long)
        chosen[i, : ids.numel()] = ids
        mask[i, : ids.numel()] = True
    chosen, mask, rejected = chosen.to(device), mask.to(device), rejected.to(device)

    z_rej = z.gather(-1, rejected.unsqueeze(-1))
    delta = z.gather(-1, chosen) - z_rej
    weight = torch.clamp((eps - delta) / eps, 0.0, 1.0) * mask
    per_token = F.softplus(eps - delta)
    n_chosen = mask.sum(-1).clamp(min=1)
    pref = ((per_token * weight).sum(-1) / n_chosen).mean()

    diff = z - z_ref
    target_mask = torch.zeros_like(z, dtype=torch.bool)
    rows_idx = torch.arange(batch, device=device).unsqueeze(1).expand_as(chosen)
    target_mask[rows_idx[mask], chosen[mask]] = True
    target_mask.scatter_(1, rejected.unsqueeze(-1), True)
    nontarget = ~target_mask
    mse_nt = (diff.pow(2) * nontarget).sum() / nontarget.sum()
    excess = torch.clamp((diff * target_mask).abs() - tau, min=0.0)
    mse_t = excess.pow(2).sum() / target_mask.sum()
    loss = pref + lam_nt * mse_nt + lam_t * mse_t

    with torch.no_grad():
        logp = F.log_softmax(z, dim=-1)
        lp_rej = logp.gather(-1, rejected.unsqueeze(-1))
        wins = (logp.gather(-1, chosen) > lp_rej) & mask
        chosen_win = (wins.float().sum(-1) / n_chosen).mean().item()
        margin_win = ((delta >= eps) & mask).float().sum().item() / mask.sum().clamp(min=1).item()
    metrics = {"pref_loss": pref.item(), "mse_nt": mse_nt.item(),
               "mse_t": mse_t.item(), "chosen_win": chosen_win,
               "margin_win": margin_win}
    return loss, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["pivot", "shuffled"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", type=Path, default=None)
    parser.add_argument("--smoke-rows", type=int, default=0,
                        help="train on only the first N rows (pipeline smoke)")
    parser.add_argument("--max-rows", type=int, default=0,
                        help="random subsample to N rows (control-arm matching, seed 3407)")
    args = parser.parse_args()

    cfg_all = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    cfg = cfg_all["train"]
    reg = cfg_all["regularization"]
    torch.manual_seed(int(cfg["seed"]))
    np.random.seed(int(cfg["seed"]))

    rows_path = args.rows or EXP / "data" / f"rows_{args.arm}.jsonl.gz"
    rows = load_rows(rows_path)
    if args.smoke_rows:
        rows = rows[: args.smoke_rows]
    if args.max_rows and len(rows) > args.max_rows:
        rng = np.random.default_rng(3407)
        keep = rng.choice(len(rows), size=args.max_rows, replace=False)
        rows = [rows[i] for i in sorted(keep)]
        print(f"[train] downsampled to {len(rows)} rows (control matching)", flush=True)
    elif len(rows) < int(reg["min_train_rows"]):
        raise SystemExit(f"yield gate: {len(rows)} rows < {reg['min_train_rows']}")
    rows = [r for r in rows if len(r["context_ids"]) + 1 <= int(cfg["max_seq_length"])]
    rows.sort(key=lambda r: -len(r["context_ids"]))  # long-first: fail fast on memory
    print(f"[train:{args.arm}] {len(rows)} rows, max ctx "
          f"{max(len(r['context_ids']) for r in rows)}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, trust_remote_code=True,
        dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora = LoraConfig(r=int(cfg["lora_r"]), lora_alpha=int(cfg["lora_alpha"]),
                      lora_dropout=float(cfg["lora_dropout"]), bias="none",
                      target_modules=list(cfg["target_modules"]))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    backbone, head = resolve_backbone_and_head(model)
    device = next(model.parameters()).device

    # --- preregistered padding-equivalence gate (and memory smoke) ----------
    per_device = int(cfg["per_device_batch"])
    tol = float(cfg["padding_equivalence_tolerance"])
    probe = rows[: max(per_device * 2, 4)]
    with torch.no_grad():
        batched = final_logits(backbone, head, probe[:per_device * 2], device)
        singles = torch.cat([final_logits(backbone, head, [r], device)
                             for r in probe[:per_device * 2]])
    max_diff = (batched - singles).abs().max().item()
    print(f"[gate] padding equivalence max|Δlogit| = {max_diff:.4f} (tol {tol})", flush=True)
    if max_diff > tol:
        print("[gate] FAILED — falling back to batch-of-1", flush=True)
        per_device = 1
    del batched, singles

    accum = max(1, int(cfg["effective_batch"]) // per_device)
    steps_per_epoch = math.ceil(len(rows) / (per_device * accum))
    total_steps = steps_per_epoch * int(cfg["num_epochs"])
    warmup = int(total_steps * float(cfg["warmup_ratio"]))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))

    def lr_lambda(step):
        if step < warmup:
            return step / max(warmup, 1)
        return max(0.0, (total_steps - step) / max(total_steps - warmup, 1))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    stop_threshold = float(cfg["early_stopping_chosen_win"])
    log_every = int(cfg["log_every_steps"])
    history: list[dict] = []
    window: list[float] = []
    started = time.time()
    stopped_early = False
    step = 0
    micro = 0
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(int(cfg["num_epochs"])):
        for start in range(0, len(rows), per_device):
            batch_rows = rows[start: start + per_device]
            try:
                z = final_logits(backbone, head, batch_rows, device)
                with torch.no_grad(), model.disable_adapter():
                    z_ref = final_logits(backbone, head, batch_rows, device)
                loss, metrics = ftpo_loss(z, z_ref, batch_rows, cfg, device)
                (loss / accum).backward()
            except OOM_ERRORS as exc:  # noqa: PERF203
                raise SystemExit(
                    f"OOM/accelerator error at ctx {max(len(r['context_ids']) for r in batch_rows)}: {exc}"
                ) from exc
            window.append(metrics["chosen_win"])
            micro += 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0 or step == total_steps:
                    mean_win = sum(window) / len(window)
                    entry = {"step": step, "of": total_steps,
                             "chosen_win": round(mean_win, 4),
                             "pref_loss": round(metrics["pref_loss"], 4),
                             "margin_win": round(metrics["margin_win"], 4),
                             "lr": scheduler.get_last_lr()[0],
                             "elapsed_s": round(time.time() - started, 1)}
                    history.append(entry)
                    print(f"[train:{args.arm}] {entry}", flush=True)
                    min_step = int(total_steps * float(cfg.get("early_stop_min_progress", 0.0)))
                    if mean_win >= stop_threshold and step >= min_step:
                        print(f"[train:{args.arm}] chosen_win {mean_win:.3f} >= "
                              f"{stop_threshold} — early stop", flush=True)
                        stopped_early = True
                    window = []
            if stopped_early:
                break
        if stopped_early:
            break

    if micro % accum != 0 and not stopped_early:
        # flush the trailing partial accumulation
        torch.nn.utils.clip_grad_norm_(params, float(cfg["max_grad_norm"]))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
    if window:
        mean_win = sum(window) / len(window)
        history.append({"step": step, "of": total_steps,
                        "chosen_win": round(mean_win, 4), "final": True})
        print(f"[train:{args.arm}] final: step={step} chosen_win={mean_win:.4f}",
              flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    (args.out / "train_summary.json").write_text(json.dumps({
        "arm": args.arm, "rows": len(rows), "per_device_batch": per_device,
        "accum": accum, "steps_run": step, "total_steps": total_steps,
        "stopped_early": stopped_early,
        "padding_equivalence_max_diff": max_diff,
        "wall_s": time.time() - started, "history": history,
        "config": cfg,
    }, indent=2))
    print(f"[train:{args.arm}] saved adapter to {args.out} "
          f"({time.time() - started:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
