#!/usr/bin/env python3
"""Evaluate frozen vs LoRA-trained 4B on held-out FRESH tasks (Milestone 3).

greedy@1 (deployable single-shot) + pass@k (coverage / diversity-collapse check) by depth. Pass
--adapter to eval the trained model; omit for frozen. Tasks unseen during training (fresh seed +
optionally held-out depths) test whether banking generalizes rather than memorizes.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import gen_tasks as G  # noqa: E402
import code_env as E  # noqa: E402


def extract(p, prompt, seq_ids):
    text = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    c, _ = E.extract_candidate_code(text, "transform")
    return c or ""


def grade(task, code):
    return bool(code) and bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-tasks", type=Path, default=EXP / "data" / "eval_tasks.jsonl")
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--think", action="store_true", help="thinking on (default off for single-shot banking eval)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.eval_tasks.read_text().splitlines() if l.strip()]
    import gen_lib as GL
    p = GL.Probe()
    tag = "frozen"
    if args.adapter:
        from peft import PeftModel
        p.model = PeftModel.from_pretrained(p.model, args.adapter)
        p.model.eval()
        tag = f"trained({args.adapter})"
    print(f"eval {tag}: {len(tasks)} tasks, k={args.k}, think={args.think}", flush=True)

    prompts = [G.prompt_for(t) for t in tasks]

    def greedy(think):
        g = p.gen_sequences(prompts, think=think, budget=args.budget, greedy=True, batch_size=32)
        return [grade(t, extract(p, pr, gg["seq_ids"])) for t, pr, gg in zip(tasks, prompts, g)]

    def passk(think):
        rep = [pr for pr in prompts for _ in range(args.k)]
        s = p.gen_sequences(rep, think=think, budget=args.budget, greedy=False, batch_size=48)
        s_ok = [grade(tasks[i // args.k], extract(p, rep[i], s[i]["seq_ids"])) for i in range(len(rep))]
        return [any(s_ok[i * args.k:(i + 1) * args.k]) for i in range(len(tasks))]

    nothink_g = greedy(False)   # the banking target: correct code in ONE fast forward, no thinking
    think_g = greedy(True)      # deployable single-shot reference (comparable to M1/M2)
    think_pk = passk(True)      # coverage / diversity-collapse check

    def agg(ok):
        by = defaultdict(lambda: [0, 0])
        for t, o in zip(tasks, ok):
            by[t["depth"]][0] += 1; by[t["depth"]][1] += int(o)
        return {"overall": round(sum(ok) / len(tasks), 3),
                "by_depth": {d: round(v[1] / v[0], 3) for d, v in sorted(by.items())}}

    out = {"tag": tag, "n_tasks": len(tasks), "k": args.k,
           "nothink_greedy@1": agg(nothink_g), "think_greedy@1": agg(think_g), f"think_pass@{args.k}": agg(think_pk)}
    args.out.write_text(json.dumps(out, indent=2))
    print(f"\n{tag}  (n={len(tasks)})")
    for m in ["nothink_greedy@1", "think_greedy@1", f"think_pass@{args.k}"]:
        print(f"  {m:18s} {out[m]['overall']:.3f}   " + "  ".join(f"d{d}:{a:.2f}" for d, a in out[m]["by_depth"].items()))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
