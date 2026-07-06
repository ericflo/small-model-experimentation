#!/usr/bin/env python3
"""Manual cached-reference DPO on QLoRA (trl unavailable). Continue an SFT adapter as the policy; the SFT model
IS the frozen reference (ref logps precomputed once at init). Loss per pair:
  L = -log sigmoid( beta * ((logp_pol(chosen) - logp_ref(chosen)) - (logp_pol(rejected) - logp_ref(rejected))) )
plus a small SFT regularizer on chosen. Completion-token logp (prompt masked), same prompt/target format as SFT.
Tests whether contrastive discrimination (learning from the model's own wrong samples) raises deployable greedy@1
beyond SFT-on-positives."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
from gen_lib import MODEL_ID  # noqa: E402
TARGET = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def build_ids(tok, prompt, code, max_length=1024):
    msgs = [{"role": "user", "content": prompt}]
    ptext = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    target = "```python\n" + code.strip() + "\n```" + tok.eos_token
    pid = tok(ptext, add_special_tokens=False)["input_ids"]
    fid = tok(ptext + target, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    return fid, len(pid)


@torch.no_grad()
def _no_grad_flag():
    return True


def seq_logp(model, ids, plen, device):
    """Sum of completion-token logprobs (tokens from position plen onward). ids: full python list."""
    x = torch.tensor([ids], device=device)
    logits = model(input_ids=x).logits[0]                 # [L, V]
    logp = F.log_softmax(logits[:-1].float(), dim=-1)     # predicts token t from t-1
    tgt = x[0, 1:]                                         # [L-1]
    tok_logp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    comp = tok_logp[plen - 1:]                             # completion tokens only
    return comp.sum(), comp.numel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft-adapter", required=True)
    ap.add_argument("--pairs", type=Path, default=EXP / "data" / "pairs.jsonl")
    ap.add_argument("--out", type=Path, default=EXP / "runs" / "adapter_DPO")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--sft-lambda", type=float, default=0.05)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel, prepare_model_for_kbit_training

    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                             bnb_4bit_compute_dtype=torch.bfloat16)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, device_map="cuda", dtype=torch.bfloat16,
        quantization_config=bnb, attn_implementation="sdpa")
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=True)
    # policy = SFT adapter, trainable
    model = PeftModel.from_pretrained(base, args.sft_adapter, is_trainable=True)
    model.config.use_cache = False
    device = model.device

    pairs = [json.loads(l) for l in args.pairs.read_text().splitlines() if l.strip()]
    print(f"[dpo] {len(pairs)} pairs | beta={args.beta} lr={args.lr} sft_lambda={args.sft_lambda}", flush=True)
    data = []
    for p in pairs:
        cid, cpl = build_ids(tok, p["prompt"], p["chosen"])
        rid, rpl = build_ids(tok, p["prompt"], p["rejected"])
        data.append((cid, cpl, rid, rpl))

    # precompute reference logps (policy == SFT at init, adapter frozen for this pass)
    model.eval()
    ref = []
    with torch.no_grad():
        for (cid, cpl, rid, rpl) in data:
            lc, _ = seq_logp(model, cid, cpl, device)
            lr, _ = seq_logp(model, rid, rpl, device)
            ref.append((lc.item(), lr.item()))
    afc = sum(1 for c, r in ref if c > r) / max(1, len(ref))
    print(f"[dpo] cached reference logps | pre-DPO 2AFC (SFT logp_chosen>logp_rejected): {afc:.3f} "
          f"(this is the discriminability the DPO gradient starts from)", flush=True)

    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rng = random.Random(args.seed)
    steps = int(len(data) * args.epochs)
    order = list(range(len(data)))
    accl = 0.0; accmarg = 0.0; nacc = 0
    opt.zero_grad()
    for step in range(steps):
        if step % len(data) == 0:
            rng.shuffle(order)
        i = order[step % len(data)]
        cid, cpl, rid, rpl = data[i]
        ref_c, ref_r = ref[i]
        pol_c, nc = seq_logp(model, cid, cpl, device)
        pol_r, _ = seq_logp(model, rid, rpl, device)
        logits = args.beta * ((pol_c - ref_c) - (pol_r - ref_r))
        loss = -F.logsigmoid(logits) + args.sft_lambda * (-(pol_c / max(1, nc)))
        (loss / args.grad_accum).backward()
        accl += loss.item(); accmarg += (pol_c - pol_r).item(); nacc += 1
        if (step + 1) % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
        if (step + 1) % 40 == 0:
            print(f"[dpo] step {step+1}/{steps} loss {accl/nacc:.4f} margin(logp_c-logp_r) {accmarg/nacc:.2f}", flush=True)
            accl = accmarg = 0.0; nacc = 0
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"[dpo] saved adapter to {args.out}", flush=True)


if __name__ == "__main__":
    main()
