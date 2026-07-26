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
same no-think chat template, free generation, answer parsed from the trailing
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
                                   add_generation_prompt=True,
                                   enable_thinking=False)  # footgun-ok: C59's real_cot arm is no-think; required for comparability


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


# Stop as soon as the model commits, so the token budget stops being a variable at all. Without this
# the budget is a confound in BOTH directions: too low truncates reasoning into a false failure (a 512
# cap made base CoT read 0.040 with 95.5% never committing), and any single fixed cap silently favours
# whichever arm happens to commit sooner -- looping may well change that. docs/model_playbook.md:
# "Budget the CoT generously and check truncation... Do not transfer a budget."
STOPS = [f"Answer: {d}" for d in range(10)]


@torch.no_grad()
def cot(model, tok, eps, bs=16, max_new=3072):
    """Free generation with stop-on-commit; answer parsed from trailing `Answer: <digit>` (C59's arm)."""
    ok, unparsed, lens = 0, 0, []
    for s in range(0, len(eps), bs):
        sub = eps[s:s + bs]
        enc = tok([chat(tok, e["prompt"]) for e in sub], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, use_cache=True,
                             stop_strings=STOPS, tokenizer=tok,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        for e, row in zip(sub, gen):
            txt = tok.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True)
            lens.append(len(tok(txt, add_special_tokens=False).input_ids))
            m = ANS.findall(txt)
            if not m:
                unparsed += 1                      # never committed -> counts as wrong, tracked
            ok += int((m[-1] if m else "") == e["answer"])
    # Report the DISTRIBUTION, not just the mean. A mean well under the cap can still hide a
    # truncated tail: at a 512 cap the mean was 510 (obvious), but a mean of 400 with p99 pinned at
    # the cap looks healthy and is not. frac_at_cap is the statistic that actually answers "is the
    # budget enough" -- together with unparsed_frac it makes the question empirical.
    lens_sorted = sorted(lens)
    def pct(q):
        return lens_sorted[min(len(lens_sorted) - 1, int(q * len(lens_sorted)))] if lens_sorted else 0
    at_cap = sum(1 for L in lens if L >= max_new - 2) / max(1, len(lens))
    return ok / len(eps), unparsed / len(eps), {
        "mean": round(sum(lens) / max(1, len(lens)), 1), "p50": pct(0.50),
        "p90": pct(0.90), "p99": pct(0.99), "max": lens_sorted[-1] if lens_sorted else 0,
        "frac_at_cap": round(at_cap, 3), "cap": max_new}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(EXP / "data" / "heldout_shift.jsonl"))
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--block", default="12:16")
    ap.add_argument("--k", type=int, default=2)
    # 1024, not 512: at 512 the base CoT arm generated a mean of 510 tokens and 95.5% NEVER emitted
    # "Answer: <digit>" -- accuracy read 0.040 purely because the budget cut reasoning off before it
    # could commit. C59 used 768 on this substrate. A truncation-bound arm is not a measurement, and
    # this program has already been burned by exactly this class of artifact.
    # Generous by default AND stop-on-commit (see STOPS): the budget should never be the thing being
    # measured. Cost stays low because only non-committal episodes run to the cap.
    ap.add_argument("--max-new", type=int, default=3072)
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
        warn = ""
        gt = (extra or {}).get("gen_tokens") or {}
        if gt and gt.get("frac_at_cap", 0) > 0.05:
            res["arms"][name]["tail_at_cap"] = True
            print(f"  NOTE {name}: {gt['frac_at_cap']:.1%} of generations hit the {gt['cap']}-token cap "
                  f"(p99={gt['p99']}). The tail is budget-bound even though the parse rate looks fine.",
                  flush=True)
        if extra and extra.get("unparsed_frac", 0) > 0.2:
            # HARD GATE: if the model mostly never commits an answer, the number measures the token
            # budget rather than the model, and must not be quoted as a capability result.
            warn = (f"  <-- TRUNCATION-BOUND ({extra['unparsed_frac']:.0%} never emitted an answer, "
                    f"lengths {extra.get('gen_tokens')}): NOT INTERPRETABLE")
            res["arms"][name]["truncation_bound"] = True
        print(f"  {name:22s} acc {acc:.4f}" + (f"   {extra}" if extra else "") + warn, flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))

    print(f"n={len(eps)} | block {args.block} k={args.k} | greedy, no-think (C59 protocol)", flush=True)
    record("base_forced", forced(model, tok, eps))
    acc, unp, ln = cot(model, tok, eps, max_new=args.max_new)
    record("base_cot", acc, {"unparsed_frac": round(unp, 3), "gen_tokens": ln})
    with CacheSafeLoop(model, a, b, args.k):
        record("loop_forced", forced(model, tok, eps))
        acc, unp, ln = cot(model, tok, eps, max_new=args.max_new)
        record("loop_cot", acc, {"unparsed_frac": round(unp, 3), "gen_tokens": ln})

    A = res["arms"]
    print("\n=== VERDICT ===", flush=True)
    print(f"  forced: base {A['base_forced']['acc']:.3f} -> loop {A['loop_forced']['acc']:.3f}", flush=True)
    print(f"  CoT   : base {A['base_cot']['acc']:.3f} -> loop {A['loop_cot']['acc']:.3f}", flush=True)
    se = (0.25 / len(eps)) ** 0.5                  # conservative binomial SE at p=0.5
    delta = A["loop_cot"]["acc"] - A["base_cot"]["acc"]
    # SATURATION GATE (pre-registered): a CoT comparison is only interpretable if BOTH arms actually
    # commit. If either is truncation-bound the contrast measures the budget, not depth.
    if A["base_cot"].get("truncation_bound") or A["loop_cot"].get("truncation_bound"):
        print("  NOT INTERPRETABLE: a CoT arm is truncation-bound; raise the budget and re-run. "
              "Reporting this contrast would repeat C45's false 0.00.", flush=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
        return
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
