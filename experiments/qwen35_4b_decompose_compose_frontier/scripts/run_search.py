#!/usr/bin/env python3
"""Frontier search: does model-guided decompose-and-compose crack depth-3 (that monolithic sampling can't),
and does the model's guidance beat matched-budget brute-force enumeration?

For each held-out task, run guided + brute search to a high call budget and record calls-to-solution +
hidden-generalization; the solve-rate-vs-call-budget curve then compares them at any matched budget.
Monolithic sampling (the M2 baseline the decompose loop must beat to be a frontier win) is graded too.
Hidden-generalizing depth-3 solutions are saved for banking (M-next).
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
import decompose_lib as D  # noqa: E402


def build_tasks(depths, per_depth, seed, k_visible=10, m_hidden=8):
    import random
    rng = random.Random(seed)
    tasks, tid = [], 0
    for d in depths:
        made = 0
        while made < per_depth:
            t = G.make_task(tid, d, rng, k_visible=k_visible, m_hidden=m_hidden)
            if t is not None:
                tasks.append(t); tid += 1; made += 1
    return tasks


def monolithic(p, tasks, k, budget):
    """M2-style: sample k whole transforms (thinking), select by visible pass, grade hidden."""
    import gen_lib  # noqa
    prompts = [p.prompt(_mono_prompt(t), enable_thinking=True) for t in tasks]
    rep = [pr for pr in prompts for _ in range(k)]
    gens = p.gen_sequences(rep, think=True, budget=budget, greedy=False, batch_size=48)
    out = []
    for ti, t in enumerate(tasks):
        best, bs = "", -1
        for j in range(k):
            code = _extract(p, rep[ti * k + j], gens[ti * k + j]["seq_ids"])
            r = E.execute_public_and_asserts(code, G.to_public_cases(t), [])
            if sum(r["public_passed"]) > bs:
                bs, best = sum(r["public_passed"]), code
        full = E.execute_public_and_asserts(best, G.to_public_cases(t), G.to_hidden_asserts(t))["full_pass"] if best else False
        out.append(bool(full))
    return out


def _mono_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return ("Infer the rule mapping input list to output list and implement it.\n\nExamples:\n" + lines +
            "\n\nWrite `def transform(xs):` reproducing this for all such inputs. Only a ```python code block.")


def _extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    c, _ = E.extract_candidate_code(txt, "transform")
    return c or ""


def curve(records, depth, budgets):
    """solve-rate (hidden-generalizing) at each call budget, for depth-`depth` tasks."""
    rs = [r for r in records if r["depth"] == depth]
    return {B: round(sum(1 for r in rs if r["hidden"] and r["calls"] <= B) / len(rs), 3) for B in budgets}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-depth", type=int, default=40)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--top-p", type=int, default=6)
    ap.add_argument("--max-budget", type=int, default=4000)
    ap.add_argument("--k-mono", type=int, default=8)
    ap.add_argument("--seed", type=int, default=777)
    args = ap.parse_args()
    if args.smoke:
        args.per_depth, args.depths, args.max_budget = 4, [2, 3], 1500

    tasks = build_tasks(args.depths, args.per_depth, args.seed)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "search_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} held-out tasks (depths {args.depths}, {args.per_depth}/depth, k_visible=10)", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    dz = D.Decomposer(p)
    print(f"model loaded {p.load_secs:.0f}s", flush=True)

    rec = {"guided": [], "brute": []}
    for mode in ("guided", "brute"):
        t0 = time.time()
        for t in tasks:
            r = dz.search(t, mode, beam_width=args.beam, top_p=args.top_p, max_depth=t["depth"], call_budget=args.max_budget)
            hid = bool(r["solved"] and D.verify_hidden(t, r["prims"]))
            rec[mode].append({"task_id": t["task_id"], "depth": t["depth"], "solved": r["solved"],
                              "hidden": hid, "calls": r["calls"], "prims": r["prims"] if hid else None})
        print(f"  {mode} done [{time.time()-t0:.0f}s]", flush=True)

    t0 = time.time()
    mono = monolithic(p, tasks, args.k_mono, budget=512)
    print(f"  monolithic done [{time.time()-t0:.0f}s]", flush=True)

    budgets = [50, 100, 200, 400, 800, 1500, 3000, args.max_budget]
    summary = {"n_per_depth": args.per_depth, "depths": args.depths, "budgets": budgets, "beam": args.beam, "top_p": args.top_p}
    for depth in args.depths:
        n = sum(1 for t in tasks if t["depth"] == depth)
        mono_rate = sum(1 for t, m in zip(tasks, mono) if t["depth"] == depth and m) / n
        summary[f"depth{depth}"] = {
            "n": n, "monolithic_hidden": round(mono_rate, 3),
            "guided_curve": curve(rec["guided"], depth, budgets),
            "brute_curve": curve(rec["brute"], depth, budgets),
            "guided_final": round(sum(1 for r in rec["guided"] if r["depth"] == depth and r["hidden"]) / n, 3),
            "brute_final": round(sum(1 for r in rec["brute"] if r["depth"] == depth and r["hidden"]) / n, 3),
            "guided_mean_calls": round(sum(r["calls"] for r in rec["guided"] if r["depth"] == depth and r["hidden"]) / max(1, sum(1 for r in rec["guided"] if r["depth"] == depth and r["hidden"])), 0),
            "brute_mean_calls": round(sum(r["calls"] for r in rec["brute"] if r["depth"] == depth and r["hidden"]) / max(1, sum(1 for r in rec["brute"] if r["depth"] == depth and r["hidden"])), 0),
        }
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "search_summary.json").write_text(json.dumps(summary, indent=2))
    # save hidden-generalizing solutions (union) for banking
    found = {}
    for mode in ("guided", "brute"):
        for r in rec[mode]:
            if r["hidden"] and r["task_id"] not in found:
                t = next(x for x in tasks if x["task_id"] == r["task_id"])
                found[r["task_id"]] = {"task_id": r["task_id"], "depth": r["depth"],
                                       "prompt": _mono_prompt(t), "code": D.prims_to_code([tuple(x) for x in r["prims"]])}
    (EXP / "data" / "found_solutions.jsonl").write_text("\n".join(json.dumps(v) for v in found.values()) + "\n")

    print("\n=== FRONTIER SEARCH (hidden-generalizing solve rate) ===")
    for depth in args.depths:
        s = summary[f"depth{depth}"]
        print(f"depth {depth} (n={s['n']}): monolithic {s['monolithic_hidden']:.3f} | "
              f"guided {s['guided_final']:.3f} ({s['guided_mean_calls']:.0f} calls) | brute {s['brute_final']:.3f} ({s['brute_mean_calls']:.0f} calls)")
        print(f"   guided curve: {s['guided_curve']}")
        print(f"   brute  curve: {s['brute_curve']}")
    print(f"\nsaved {len(found)} hidden-generalizing solutions for banking -> data/found_solutions.jsonl")


if __name__ == "__main__":
    main()
