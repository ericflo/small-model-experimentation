#!/usr/bin/env python3
"""Phase 2 discriminator: on the SAME verified grid tasks, three prompting conditions isolate the wall.

  bare          : Phase-1 protocol (I/O examples only) — the full problem (identify + execute).
  plan_given    : prompt states the exact op pipeline; model only translates to code (execution only).
  intermediates : visible examples show full state chains (observability restored; ops still unnamed).

Pre-registered: P7 plan-given erases depth+destruction effects; P8 intermediates rescues the k effect.
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
import gen_factorial as F  # noqa: E402
import code_env as E  # noqa: E402
import decompose_lib as D  # noqa: E402


def chain_states(task):
    """Recompute intermediate states of the TRUE pipeline for each visible example (interpreter replay)."""
    steps = []
    for s in task["target_ops"]:
        op = s.split("(")[0]
        k = int(s[s.index("(") + 1:-1]) if "(" in s else None
        steps.append((op, k))
    chains = []
    for ex in task["visible"]:
        st = tuple(ex["input"])
        chain = [list(st)]
        for op, k in steps:
            st = D.apply_prim(op, k, (st,))[0]
            chain.append(list(st))
        chains.append(chain)
    return chains


def prompt_bare(t):
    return G.prompt_for(t)


def prompt_plan(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    plan = " then ".join(t["target_ops"])
    return ("You are given input/output examples of a Python function `transform` on lists of integers.\n\n"
            f"Examples:\n{lines}\n\n"
            f"The transformation is exactly this pipeline of steps, in order: {plan}.\n"
            "Step meanings: " + ", ".join(f"{n}: {D.DESC[n]}" for n in sorted({o.split('(')[0] for o in t['target_ops']})) + ".\n\n"
            "Implement `def transform(xs):` applying these steps in order. Respond with only the function "
            "in one ```python code block.")


def prompt_inter(t):
    chains = chain_states(t)
    lines = []
    for ch in chains[:6]:
        lines.append(" -> ".join(repr(x) for x in ch))
    return ("A Python function `transform` maps a list of integers to a list of integers via a fixed "
            "sequence of simple steps. Below, each line shows one input passing through EVERY intermediate "
            "state to the final output:\n\n" + "\n".join(lines) + "\n\n"
            "Infer the step sequence and implement `def transform(xs):` reproducing the overall mapping. "
            "Respond with only the function in one ```python code block.")


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    c, _ = E.extract_candidate_code(txt, "transform")
    return c or ""


def grade(task, code):
    return bool(code) and bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-cell", type=int, default=20, help="tasks per (d,k) cell, drawn from grid_tasks")
    ap.add_argument("--k-samples", type=int, default=4)
    ap.add_argument("--budget", type=int, default=512)
    args = ap.parse_args()
    CELLS = [(2, 0), (2, 2), (3, 0), (3, 2), (4, 0), (4, 2)]
    if args.smoke:
        CELLS, args.per_cell, args.k_samples = [(2, 0), (2, 2)], 2, 2

    allt = [json.loads(l) for l in (EXP / "data" / "grid_tasks.jsonl").read_text().splitlines()]
    tasks = []
    for (d, k) in CELLS:
        cell = [t for t in allt if t["depth"] == d and t["n_destr"] == k][:args.per_cell]
        tasks += cell
    print(f"{len(tasks)} tasks over cells {CELLS}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    print(f"model loaded {p.load_secs:.0f}s", flush=True)
    conds = {"bare": prompt_bare, "plan_given": prompt_plan, "intermediates": prompt_inter}
    out = defaultdict(dict)
    t0 = time.time()
    for cname, pf in conds.items():
        prompts = [p.prompt(pf(t), enable_thinking=True) for t in tasks]
        rep = [pr for pr in prompts for _ in range(args.k_samples)]
        gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, batch_size=48)
        for ti, t in enumerate(tasks):
            oks = [grade(t, extract(p, rep[ti * args.k_samples + j], gens[ti * args.k_samples + j]["seq_ids"]))
                   for j in range(args.k_samples)]
            out[t["task_id"]][cname] = {"passk": any(oks), "n_ok": sum(oks)}
        print(f"  {cname} done [{time.time()-t0:.0f}s]", flush=True)

    with (EXP / "data" / "discriminator_records.jsonl").open("w") as f:
        for t in tasks:
            f.write(json.dumps({"task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                                **{c: out[t["task_id"]][c] for c in conds}}) + "\n")
    print("\n=== DISCRIMINATOR (pass@%d by cell) ===" % args.k_samples)
    print(f"{'cell':>8} {'bare':>6} {'plan_given':>11} {'intermediates':>14}")
    for (d, k) in CELLS:
        cell = [t for t in tasks if t["depth"] == d and t["n_destr"] == k]
        if not cell:
            continue
        row = [sum(out[t['task_id']][c]["passk"] for t in cell) / len(cell) for c in conds]
        print(f"  d{d}k{k}  {row[0]:>6.2f} {row[1]:>11.2f} {row[2]:>14.2f}   (n={len(cell)})")
    print("wrote data/discriminator_records.jsonl")


if __name__ == "__main__":
    main()
