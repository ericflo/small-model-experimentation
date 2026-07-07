#!/usr/bin/env python3
"""Does the multi-step SIMULATION wall (C13) depend on modality? Eval the base model on the successor-chain
traversal in a rendering (linguistic-semantic / linguistic-symbolic / formal-dict), no-think (PRIMARY: forces
mental simulation) or think (externalized transcription). Same tasks across renderings (paired). Reports accuracy
vs depth, off-by-one (simulation slip), a recency baseline, and prompt token counts (length-confound check)."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import reasoning_family as RF  # noqa: E402
ANS = re.compile(r"Answer:\s*([A-Za-z0-9']+)")


def parse_answer(text, entities):
    text = text.split("</think>")[-1] if "</think>" in text else text
    ent = set(entities)
    for m in reversed(ANS.findall(text)):
        cand = m.strip().strip("'\".,)")
        if cand in ent: return cand
    best, bestpos = "", -1
    for e in entities:
        pos = text.rfind(e)
        if pos > bestpos: bestpos, best = pos, e
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=80)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--render", required=True, choices=list(RF.RENDERERS))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--seed", type=int, default=100)
    args = ap.parse_args()
    render = RF.RENDERERS[args.render]

    tasks = [RF.gen_task(d, args.seed * 100000 + d * 1000 + i) for d in args.depths for i in range(args.n_per_depth)]
    print(f"[reason] {len(tasks)} tasks | render={args.render} think={args.think}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    prompts = [p.prompt(render(t), enable_thinking=args.think) for t in tasks]
    ntok = [len(p._ids(pr)) for pr in prompts]
    budget = 1024 if args.think else None
    gg = p.gen_sequences(prompts, think=args.think, budget=budget, greedy=True, answer_max=200, batch_size=48)
    by = {d: [0, 0, 0, 0] for d in args.depths}  # n, correct, offby1, recency-hit
    for t, pr, g in zip(tasks, prompts, gg):
        ans = parse_answer(p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False), t["entities"])
        correct = ans == t["answer"]
        idx = t["main"].index(t["answer"])
        nb = set()
        if idx-1 >= 0: nb.add(t["main"][idx-1])
        if idx+1 < len(t["main"]): nb.add(t["main"][idx+1])
        recency = t["facts"][-1][1]  # last-mentioned entity (a shortcut-seeker's guess)
        c = by[t["depth"]]; c[0] += 1; c[1] += int(correct); c[2] += int((not correct) and ans in nb); c[3] += int(recency == t["answer"])
    out = {"render": args.render, "think": args.think, "n_per_depth": args.n_per_depth,
           "mean_prompt_tokens": round(sum(ntok)/len(ntok)),
           "by_depth": {d: {"n": c[0], "acc": round(c[1]/c[0], 3), "offby1": round(c[2]/c[0], 3),
                            "recency_base": round(c[3]/c[0], 3)} for d, c in by.items()}}
    (EXP / "runs").mkdir(exist_ok=True)
    tag = f"{args.render}_{'think' if args.think else 'nothink'}"
    json.dump(out, open(EXP / "runs" / f"reason_{tag}.json", "w"), indent=1)
    print(f"[reason] {tag} ({out['mean_prompt_tokens']} tok): " +
          " ".join(f"d{d}={out['by_depth'][d]['acc']:.2f}" for d in args.depths) +
          f" | recency-base d{args.depths[-1]}={out['by_depth'][args.depths[-1]]['recency_base']:.2f}", flush=True)
    print(f"[reason] wrote runs/reason_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
