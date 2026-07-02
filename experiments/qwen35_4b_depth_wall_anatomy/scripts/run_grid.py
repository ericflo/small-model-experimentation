#!/usr/bin/env python3
"""Factorial battery: what determines the compositional frontier — serial depth (d) or information
destruction (k)? Pre-registered predictions in reports/prereg.md (committed before this ran).

Per task: monolithic thinking greedy@1 + pass@K (hidden-graded), per-candidate visible/hidden pass
(false-pass slice, P5), thinking length (P6), and the first-op letter-logit ranking (planner slice, P4).
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


def extract(p, prompt, seq_ids):
    txt = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in txt:
        txt = txt.split("</think>")[-1]
    c, _ = E.extract_candidate_code(txt, "transform")
    return c or ""


def run_case(task, code):
    if not code:
        return False, False
    r = E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))
    return bool(r["visible_all_pass"]), bool(r["full_pass"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-per-cell", type=int, default=25)
    ap.add_argument("--k-samples", type=int, default=6)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    cells = [(d, k) for d in (1, 2, 3, 4, 5) for k in range(0, min(d, 3) + 1)]
    if args.smoke:
        cells, args.n_per_cell, args.k_samples = [(2, 0), (2, 2), (4, 0)], 3, 2

    tasks = F.build_grid(cells, args.n_per_cell, seed=args.seed)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "grid_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks over {len(cells)} cells (n={args.n_per_cell}/cell)", flush=True)
    orc = sum(run_case(t, G.reference_code(t))[1] for t in tasks)
    print(f"oracle solvable: {orc}/{len(tasks)}", flush=True)

    import gen_lib as GL
    import decompose_lib as D
    p = GL.Probe()
    dz = D.Decomposer(p)
    print(f"model loaded {p.load_secs:.0f}s", flush=True)
    t0 = time.time()

    prompts = [G.prompt_for(t) for t in tasks]
    # greedy@1 (thinking)
    g = p.gen_sequences(prompts, think=True, budget=args.budget, greedy=True, batch_size=32)
    greedy = []
    for t, pr, gg in zip(tasks, prompts, g):
        vis, full = run_case(t, extract(p, pr, gg["seq_ids"]))
        greedy.append({"vis": vis, "full": full, "n_think": gg.get("n_think", 0)})
    print(f"greedy done [{time.time()-t0:.0f}s]", flush=True)

    # pass@K sampled (thinking) + per-candidate false-pass records
    rep = [pr for pr in prompts for _ in range(args.k_samples)]
    s = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False, batch_size=48)
    samp = []
    for i in range(len(rep)):
        t = tasks[i // args.k_samples]
        vis, full = run_case(t, extract(p, rep[i], s[i]["seq_ids"]))
        samp.append({"vis": vis, "full": full, "n_think": s[i].get("n_think", 0)})
    print(f"sampled done [{time.time()-t0:.0f}s]", flush=True)

    # planner slice: first-op letter-logit ranking on every task (one forward each)
    inp_states = [[tuple(ex["input"]) for ex in t["visible"]] for t in tasks]
    tgt_states = [[tuple(ex["output"]) for ex in t["visible"]] for t in tasks]
    ranked = []
    for i in range(0, len(tasks), 64):
        ranked += [dz._rank([st], tg)[0] for st, tg in zip(inp_states[i:i+64], tgt_states[i:i+64])]
    print(f"planner slice done [{time.time()-t0:.0f}s]", flush=True)

    with (EXP / "data" / "grid_records.jsonl").open("w") as f:
        for ti, t in enumerate(tasks):
            true_first = t["target_ops"][0].split("(")[0]
            rk = ranked[ti]
            cands = samp[ti * args.k_samples:(ti + 1) * args.k_samples]
            f.write(json.dumps({
                "task_id": t["task_id"], "depth": t["depth"], "n_destr": t["n_destr"],
                "destr_pos": t["destr_pos"], "target_ops": t["target_ops"],
                "greedy_full": greedy[ti]["full"], "greedy_vis": greedy[ti]["vis"], "greedy_nthink": greedy[ti]["n_think"],
                "passk_full": any(c["full"] for c in cands),
                "cand_vis": [c["vis"] for c in cands], "cand_full": [c["full"] for c in cands],
                "mean_nthink": sum(c["n_think"] for c in cands) / len(cands),
                "first_op_rank": (rk.index(true_first) + 1) if true_first in rk else None,
            }) + "\n")

    # cell table
    by = defaultdict(lambda: {"n": 0, "g": 0, "pk": 0})
    for ti, t in enumerate(tasks):
        c = by[(t["depth"], t["n_destr"])]
        c["n"] += 1; c["g"] += int(greedy[ti]["full"])
        c["pk"] += int(any(samp[ti * args.k_samples + j]["full"] for j in range(args.k_samples)))
    print("\n=== GRID (greedy / pass@%d) ===" % args.k_samples)
    print("d\\k " + "  ".join(f"{k:>10d}" for k in range(4)))
    for d in (1, 2, 3, 4, 5):
        row = []
        for k in range(4):
            c = by.get((d, k))
            row.append(f"{c['g']/c['n']:.2f}/{c['pk']/c['n']:.2f}" if c else "    -    ")
        print(f" {d}  " + "  ".join(f"{x:>10s}" for x in row))
    print(f"\nwrote data/grid_tasks.jsonl, data/grid_records.jsonl [{time.time()-t0:.0f}s total]")


if __name__ == "__main__":
    main()
