#!/usr/bin/env python3
"""THE GATE for the looping line: does mid-stack depth STACK with chain-of-thought, or replace it?

Everything measured so far used a forced single-token read -- a probe that deliberately removes the
model's usual recourse to tokens. That is the easiest possible regime in which to find an effect: the
base sits at chance (0.105) because it is denied the serial compute it normally buys with tokens. A
result there is diagnostic, not deployable.

The question that decides whether this line matters:

  base forced      0.105   <- chance; the wall
  base + CoT       0.235   <- what tokens buy (C59)
  loop forced      0.245   <= depth substitutes for tokens (established)
  loop + CoT         ???   <- if this clears 0.235, depth ADDS to reasoning rather than duplicating it

SUBSTITUTES vs STACKS is the whole fork. If loop+CoT lands at ~0.235, looping is a cheaper route to
something generation already achieves -- interesting mechanistically, worth little in deployment, since
a deployed agent always has tokens available. If loop+CoT clears it, the model has both more serial
depth per token AND its reasoning, and that is a lever a real agent can use.

Protocol is copied from C59's `real_cot` arm so the numbers are directly comparable: same substrate,
same chat template with enable_thinking=False, free generation, answer parsed from the trailing
`Answer: <digit>`. Greedy, so any difference is the intervention rather than sampling.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
EXP = HERE.parent

from loop import CacheSafeLoop  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-4B"
REV = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
DIGIT_IDS = [15 + d for d in range(10)]
ANS = re.compile(r"[Aa]nswer:\s*\**\s*(\d)")
C59 = {"forced": 0.090, "real_cot": 0.235}          # claim-ledger anchors, n=200 held-out shift


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def forced(model, tok, eps, bs=16):
    ok = 0
    for s in range(0, len(eps), bs):
        sub = eps[s:s + bs]
        enc = tok([chat(tok, e["prompt"]) + "Answer: " for e in sub], return_tensors="pt",
                  padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, use_cache=False).logits[:, -1, DIGIT_IDS]
        ok += sum(int(str(d) == e["answer"]) for e, d in zip(sub, lg.argmax(-1).tolist()))
    return ok / len(eps)


@torch.no_grad()
def cot(model, tok, eps, bs=16, max_new=512):
    """Free generation, answer parsed from the trailing `Answer: <digit>` (C59's arm)."""
    ok, unparsed, lens = 0, 0, []
    for s in range(0, len(eps), bs):
        sub = eps[s:s + bs]
        enc = tok([chat(tok, e["prompt"]) for e in sub], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for e, row in zip(sub, gen):
            txt = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            lens.append(len(tok(txt, add_special_tokens=False).input_ids))
            m = ANS.findall(txt)
            if not m:
                unparsed += 1                      # never committed -> counts as wrong, tracked
            ok += int((m[-1] if m else "") == e["answer"])
    return ok / len(eps), unparsed / len(eps), sum(lens) / max(1, len(lens))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(EXP / "data" / "heldout_shift.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--block", default="12:16")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--out", default=str(EXP / "reports" / "gen_stack.json"))
    args = ap.parse_args()

    a, b = (int(x) for x in args.block.split(":"))
    eps = [json.loads(l) for l in open(args.data)][: args.n]
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=REV, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=REV, dtype=torch.bfloat16,
                                                 device_map="cuda", attn_implementation="eager")
    model.eval()
    res = {"n": len(eps), "substrate": Path(args.data).name, "block": args.block, "k": args.k,
           "c59_reference": C59, "arms": {}}

    def record(name, acc, extra=None):
        res["arms"][name] = {"acc": round(acc, 4), **(extra or {})}
        print(f"  {name:22s} acc {acc:.4f}" + (f"   {extra}" if extra else ""), flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))

    print(f"n={len(eps)} | block {args.block} k={args.k} | greedy, enable_thinking=False", flush=True)
    record("base_forced", forced(model, tok, eps))
    acc, unp, ln = cot(model, tok, eps, max_new=args.max_new)
    record("base_cot", acc, {"unparsed_frac": round(unp, 3), "mean_gen_tokens": round(ln, 1)})
    with CacheSafeLoop(model, a, b, args.k):
        record("loop_forced", forced(model, tok, eps))
        acc, unp, ln = cot(model, tok, eps, max_new=args.max_new)
        record("loop_cot", acc, {"unparsed_frac": round(unp, 3), "mean_gen_tokens": round(ln, 1)})

    A = res["arms"]
    print("\n=== VERDICT ===", flush=True)
    print(f"  forced: base {A['base_forced']['acc']:.3f} -> loop {A['loop_forced']['acc']:.3f}", flush=True)
    print(f"  CoT   : base {A['base_cot']['acc']:.3f} -> loop {A['loop_cot']['acc']:.3f}", flush=True)
    se = (0.25 / len(eps)) ** 0.5                  # conservative binomial SE at p=0.5
    delta = A["loop_cot"]["acc"] - A["base_cot"]["acc"]
    if delta > 2 * se:
        print(f"  STACKS: depth adds {delta:+.3f} ON TOP of chain-of-thought (2 SE = {2*se:.3f}). "
              f"Depth is a lever an agent with tokens can still use.", flush=True)
    elif A["loop_forced"]["acc"] >= A["base_cot"]["acc"] - 2 * se:
        print(f"  SUBSTITUTES ONLY: looping reaches the CoT number without tokens ({delta:+.3f} on top, "
              f"2 SE = {2*se:.3f}). Mechanistically real, but a deployed agent already has tokens -- "
              f"so this does not by itself buy deployment.", flush=True)
    else:
        print(f"  WEAKER THAN CoT under generation ({delta:+.3f}); the forced-read gain does not "
              f"survive when the model can reason in tokens.", flush=True)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(f"saved -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
