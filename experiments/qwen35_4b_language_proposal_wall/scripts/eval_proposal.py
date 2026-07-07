#!/usr/bin/env python3
"""Does the structure-PROPOSAL wall (C32/C36) exist in language? Eval the base model on relational-composition
INDUCTION (infer a hidden depth-D relation-sequence from examples, apply to a new query), rendered linguistic vs
formal, no-think + think, depths 1-4. Accuracy vs depth; guessing baseline = 1/n_entities. Compare to C37
(simulation: no wall in language) and to the formal-composition proposal wall (C32/C36: base ~0 at depth-3)."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import reasoning_proposal as RP  # noqa: E402
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
    ap.add_argument("--n-per-depth", type=int, default=60)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--render", required=True, choices=list(RP.RENDERERS))
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--n-ent", type=int, default=10)
    ap.add_argument("--seed", type=int, default=200)
    args = ap.parse_args()
    render = RP.RENDERERS[args.render]

    tasks = []
    for d in args.depths:
        i = 0; made = 0
        while made < args.n_per_depth and i < args.n_per_depth * 5:
            t = RP.gen_task(d, args.seed * 100000 + d * 1000 + i, n_ent=args.n_ent)
            i += 1
            if t: tasks.append(t); made += 1
    print(f"[prop] {len(tasks)} well-posed tasks | render={args.render} think={args.think}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    prompts = [p.prompt(render(t), enable_thinking=args.think) for t in tasks]
    budget = 4096 if args.think else None
    bs = 8 if args.think else 32  # 4096-budget think on large KB prompts OOMs at bs=32
    gg = p.gen_sequences(prompts, think=args.think, budget=budget, greedy=True, answer_max=200, batch_size=bs)
    by = {d: [0, 0, 0] for d in args.depths}  # n, correct, finished-thinking
    for t, pr, g in zip(tasks, prompts, gg):
        txt = p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False)
        ans = parse_answer(txt, t["ents"])
        finished = (not args.think) or ("</think>" in txt)
        c = by[t["depth"]]; c[0] += 1; c[1] += int(ans == t["answer"]); c[2] += int(finished)
    out = {"render": args.render, "think": args.think, "n_ent": args.n_ent, "guess_base": round(1/args.n_ent, 3),
           "by_depth": {d: {"n": c[0], "acc": round(c[1]/max(1, c[0]), 3), "finished": round(c[2]/max(1, c[0]), 3)} for d, c in by.items()}}
    (EXP / "runs").mkdir(exist_ok=True)
    tag = f"{args.render}_{'think' if args.think else 'nothink'}"
    json.dump(out, open(EXP / "runs" / f"prop_{tag}.json", "w"), indent=1)
    print(f"[prop] {tag} (guess {out['guess_base']}): " + " ".join(f"d{d}={out['by_depth'][d]['acc']:.2f}" for d in args.depths), flush=True)
    print(f"[prop] wrote runs/prop_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
