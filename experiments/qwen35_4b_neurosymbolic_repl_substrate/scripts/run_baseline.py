#!/usr/bin/env python3
"""Frozen Qwen3.5-4B failure profile on the procedural substrate (single-shot, thinking on).

Reports greedy@1 (deployable single shot) and pass@k (coverage ceiling) by composition depth, against
the reference-oracle 100% solvability ceiling. This is Milestone 1: prove the substrate is solvable,
hard-and-graded, and shows a coverage->deployment gap the neurosymbolic REPL loop can then target.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import gen_tasks as G  # noqa: E402
import code_env as E  # noqa: E402


def extract(p, prompt, seq_ids):
    plen = len(p._ids(prompt))
    text = p.tok.decode(seq_ids[plen:], skip_special_tokens=False)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    code, _ = E.extract_candidate_code(text, "transform")
    return code or ""


def grade(task, code):
    if not code:
        return False
    r = E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))
    return bool(r["full_pass"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-depth", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    if args.smoke:
        args.per_depth, args.depths, args.k = 2, [1, 3, 5], 2

    (EXP / "data").mkdir(exist_ok=True)
    tasks = G.build_dataset(args.depths, args.per_depth, seed=args.seed)
    (EXP / "data" / "tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks over depths {args.depths} (n={args.per_depth}/depth)", flush=True)

    # reference-oracle solvability (sanity ceiling)
    orc = sum(grade(t, G.reference_code(t)) for t in tasks)
    print(f"reference oracle solvable: {orc}/{len(tasks)}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    print(f"model loaded {p.load_secs:.0f}s", flush=True)
    prompts = [G.prompt_for(t) for t in tasks]
    t0 = time.time()

    # greedy@1, thinking on
    greedy = p.gen_sequences(prompts, think=True, budget=args.budget, greedy=True, batch_size=24)
    g_ok = [grade(t, extract(p, pr, gr["seq_ids"])) for t, pr, gr in zip(tasks, prompts, greedy)]
    print(f"greedy@1 done [{time.time()-t0:.0f}s]", flush=True)

    # pass@k, thinking on (sampled)
    rep = [pr for pr in prompts for _ in range(args.k)]
    t1 = time.time()
    samp = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, batch_size=48)
    samp_ok = [grade(tasks[i // args.k], extract(p, rep[i], samp[i]["seq_ids"])) for i in range(len(rep))]
    passk = [any(samp_ok[i * args.k:(i + 1) * args.k]) for i in range(len(tasks))]
    print(f"pass@{args.k} done [{time.time()-t1:.0f}s]", flush=True)

    # aggregate by depth
    by = defaultdict(lambda: {"n": 0, "greedy": 0, "passk": 0, "oracle": 0})
    for t, go, pk in zip(tasks, g_ok, passk):
        d = by[t["depth"]]
        d["n"] += 1; d["greedy"] += int(go); d["passk"] += int(pk)
        d["oracle"] += int(grade(t, G.reference_code(t)))
    rows = []
    for depth in sorted(by):
        d = by[depth]
        rows.append({"depth": depth, "n": d["n"],
                     "greedy@1": round(d["greedy"] / d["n"], 3),
                     f"pass@{args.k}": round(d["passk"] / d["n"], 3),
                     "oracle": round(d["oracle"] / d["n"], 3)})
    overall = {"greedy@1": round(sum(g_ok) / len(tasks), 3),
               f"pass@{args.k}": round(sum(passk) / len(tasks), 3),
               "coverage_deployment_gap": round((sum(passk) - sum(g_ok)) / len(tasks), 3)}
    out = {"n_tasks": len(tasks), "depths": args.depths, "k": args.k, "by_depth": rows, "overall": overall}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "baseline.json").write_text(json.dumps(out, indent=2))
    with (EXP / "data" / "baseline_records.jsonl").open("w") as f:
        for t, go, pk in zip(tasks, g_ok, passk):
            f.write(json.dumps({"task_id": t["task_id"], "depth": t["depth"], "greedy": go, "passk": pk}) + "\n")

    print("\n=== FROZEN 4B FAILURE PROFILE (thinking on) ===")
    print(f"{'depth':>5} {'n':>4} {'greedy@1':>9} {'pass@'+str(args.k):>8} {'oracle':>7}")
    for r in rows:
        print(f"{r['depth']:>5} {r['n']:>4} {r['greedy@1']:>9.3f} {r['pass@'+str(args.k)]:>8.3f} {r['oracle']:>7.3f}")
    print(f"OVERALL greedy@1 {overall['greedy@1']:.3f}  pass@{args.k} {overall[f'pass@{args.k}']:.3f}  "
          f"coverage->deployment gap {overall['coverage_deployment_gap']:+.3f}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [r["depth"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(ds, [r["oracle"] for r in rows], "o--", color="#8d99ae", label="reference oracle (solvable)")
    ax.plot(ds, [r[f"pass@{args.k}"] for r in rows], "s-", color="#2a9d8f", label=f"pass@{args.k} (coverage)")
    ax.plot(ds, [r["greedy@1"] for r in rows], "^-", color="#e76f51", label="greedy@1 (deployable)")
    ax.set_xlabel("composition depth (difficulty)"); ax.set_ylabel("full-test accuracy")
    ax.set_ylim(0, 1.02); ax.set_title("Frozen Qwen3.5-4B on the procedural substrate (thinking on)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "failure_profile.png", dpi=130)
    print("wrote runs/baseline.json, analysis/failure_profile.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
