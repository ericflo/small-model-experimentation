#!/usr/bin/env python3
"""Is ICL retrieval or induction? Eval the base model on few-shot cipher application across a familiarity gradient
(caesar/atbash/affine/random, matched complexity). Key metric: per-letter accuracy split by whether the query
letter was SEEN in the examples (lookup) vs UNSEEN (generalization -- only a retrieved/induced RULE handles it).
If unseen-letter accuracy tracks familiarity, ICL = retrieval of familiar rules, not induction of novel ones.
Application-only control (cipher table given) = the execution ceiling."""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts")); sys.path.insert(0, str(EXP / "src"))
import ic_family as IC  # noqa: E402
ANS = re.compile(r"Answer:\s*([a-zA-Z]+)")


def parse(text):
    text = text.split("</think>")[-1] if "</think>" in text else text
    ms = ANS.findall(text)
    return ms[-1].strip().lower() if ms else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True, choices=["caesar", "atbash", "affine", "random"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--application", action="store_true")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--seed", type=int, default=300)
    args = ap.parse_args()

    tasks = [IC.gen_task(args.kind, args.seed * 100000 + i) for i in range(args.n)]
    print(f"[icl] {len(tasks)} tasks | kind={args.kind} application={args.application} think={args.think}", flush=True)
    import gen_lib as GL
    p = GL.Probe()
    prompts = [p.prompt(IC.render(t, application_only=args.application), enable_thinking=args.think) for t in tasks]
    budget = 1024 if args.think else None
    gg = p.gen_sequences(prompts, think=args.think, budget=budget, greedy=True, answer_max=64, batch_size=48)
    seen_c = seen_n = uns_c = uns_n = exact = 0
    for t, pr, g in zip(tasks, prompts, gg):
        out = parse(p.tok.decode(g["seq_ids"][len(p._ids(pr)):], skip_special_tokens=False))
        ans = t["answer"]
        exact += int(out == ans)
        for i, (true_ch, seen) in enumerate(zip(ans, t["query_seen_mask"])):
            got = out[i] if i < len(out) else ""
            correct = int(got == true_ch)
            if seen: seen_c += correct; seen_n += 1
            else: uns_c += correct; uns_n += 1
    out = {"kind": args.kind, "application": args.application, "think": args.think, "n": len(tasks),
           "exact_string": round(exact / len(tasks), 3),
           "seen_letter_acc": round(seen_c / max(1, seen_n), 3),
           "unseen_letter_acc": round(uns_c / max(1, uns_n), 3), "n_seen": seen_n, "n_unseen": uns_n}
    (EXP / "runs").mkdir(exist_ok=True)
    tag = f"{args.kind}{'_app' if args.application else ''}{'_think' if args.think else ''}"
    json.dump(out, open(EXP / "runs" / f"icl_{tag}.json", "w"), indent=1)
    print(f"[icl] {tag}: exact-string {out['exact_string']:.2f} | SEEN-letter {out['seen_letter_acc']:.2f} | "
          f"UNSEEN-letter {out['unseen_letter_acc']:.2f} (generalization)", flush=True)
    print(f"[icl] wrote runs/icl_{tag}.json", flush=True)


if __name__ == "__main__":
    main()
