#!/usr/bin/env python3
"""Can thinking breach the lookahead wall? Channel-matched (think->RANK vs no-think->RANK), STEP-1 headline.
Thinking generation is BATCHED across tasks (the per-prompt version was ~80s/task). Captures step-1 traces
for the internal-brute-force (enumerate vs plan) classification."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import gen_lib as GL  # noqa: E402
import decompose as D  # noqa: E402
import think_rank as TR  # noqa: E402

fam = FAM.FAMILIES["list"]


def batched_prefixes(probe, prompts, budget, batch_size=16):
    """Return (prefix_ids_list, thinking_texts). budget==0 -> no-think ranking prefix (just the prompt)."""
    if budget == 0:
        return [probe._ids(probe.prompt(p, enable_thinking=False)) for p in prompts], [""] * len(prompts)
    ptexts = [probe.prompt(p, enable_thinking=True) for p in prompts]
    ptext_ids = [probe._ids(pt) for pt in ptexts]
    outs = probe.gen_sequences(ptexts, think=True, budget=budget, batch_size=batch_size)
    prefixes, thinking = [], []
    for pid, out in zip(ptext_ids, outs):
        gen = out["seq_ids"][len(pid):]
        nt = out["n_think"]
        prefixes.append(pid + gen[:nt] + probe.close_ids)
        thinking.append(probe.tok.decode(gen[:nt]))
    return prefixes, thinking


def measure(probe, tasks, step, budget, K=6, capture=False):
    prompts = []
    gts = []
    for t in tasks:
        states, target = TR.states_at_step(t, step)
        prompts.append(D.propose_prompt(states, target))
        gts.append(TR.gt_ops(t)[step - 1])
    prefixes, thinking = batched_prefixes(probe, prompts, budget)
    ranks, hit1, hitk = [], 0, 0
    traces = []
    for pref, gt, th, t in zip(prefixes, gts, thinking, tasks):
        sc = TR.score_ops_prefix(probe, pref)
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        rank = order.index(gt) + 1
        ranks.append(rank)
        if rank == 1:
            hit1 += 1
        if rank <= K:
            hitk += 1
        if capture:
            traces.append({"task_id": t["task_id"], "gt_op": D.OP_REPRS[gt],
                           "pred_op": D.OP_REPRS[order[0]], "correct": rank == 1, "thinking": th})
    n = len(tasks)
    out = {"step": step, "budget": budget, "n": n, "top1": hit1 / n, f"top{K}": hitk / n,
           "mean_rank": statistics.mean(ranks), "median_rank": statistics.median(ranks)}
    return out, traces


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--budgets", type=int, nargs="+", default=[0, 512, 1024, 2048])
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="base")
    args = ap.parse_args()
    tasks = [json.loads(l) for l in (EXP / "data" / "eval_frozen_d3.jsonl").read_text().splitlines() if l.strip()][:args.n]
    probe = GL.Probe()
    if args.adapter:
        from peft import PeftModel
        probe.model = PeftModel.from_pretrained(probe.model, args.adapter).eval()
        print(f"[{args.tag}] loaded adapter {args.adapter}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    results = []
    print(f"chance top1={1/32:.3f}; n={len(tasks)}", flush=True)
    for budget in args.budgets:
        for step in args.steps:
            t0 = time.time()
            out, traces = measure(probe, tasks, step, budget, capture=(step == 1))
            results.append(out)
            print(f"[B={budget} step={step}] top1 {out['top1']:.3f} | top6 {out['top6']:.3f} | "
                  f"mean rank {out['mean_rank']:.1f} | {time.time()-t0:.0f}s", flush=True)
            (EXP / "runs" / f"results_{args.tag}.json").write_text(json.dumps(results, indent=1))
            if step == 1 and budget > 0:
                (EXP / "runs" / f"traces_{args.tag}_B{budget}.json").write_text(json.dumps(traces, indent=1))
    print("ALLDONE", flush=True)


if __name__ == "__main__":
    main()
