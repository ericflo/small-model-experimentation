#!/usr/bin/env python3
"""Milestone 2: neurosymbolic REPL loop vs matched-compute sample-more.

Arms (all thinking on, graded on HIDDEN held-out examples; the loop only ever sees VISIBLE):
  greedy1     : single greedy draft (= turn 0).
  repl_real   : <=T turns, each conditioned on prior code + real execution feedback (actual vs expected).
  repl_nofb   : <=T turns, control -- told only "that was wrong, try a different approach" (no exec content).
  sample_more : T independent sampled drafts, select the one passing the most VISIBLE examples (the bar).

Central question: does execution-grounded self-correction beat independent sampling at equal generation
budget, and is it the FEEDBACK CONTENT (repl_real vs repl_nofb) that drives it?
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
    text = p.tok.decode(seq_ids[len(p._ids(prompt)):], skip_special_tokens=False)
    if "</think>" in text:
        text = text.split("</think>")[-1]
    code, _ = E.extract_candidate_code(text, "transform")
    return code or ""


def run_visible(task, code):
    if not code:
        return {"visible_all_pass": False, "public_passed": [False] * len(task["visible"]),
                "public_outputs": [""] * len(task["visible"])}
    r = E.execute_public_and_asserts(code, G.to_public_cases(task), [])
    if not r["public_outputs"]:
        r["public_outputs"] = [""] * len(task["visible"])
    return r


def grade_hidden(task, code):
    if not code:
        return False
    return bool(E.execute_public_and_asserts(code, G.to_public_cases(task), G.to_hidden_asserts(task))["full_pass"])


def refine_prompt(task, prior_code, res, mode):
    base = G.prompt_for(task)
    lines = [base, "", "Your previous attempt was:", "```python", prior_code, "```", ""]
    if mode == "real":
        fails, oks = [], []
        for ex, ok, out in zip(task["visible"], res["public_passed"], res["public_outputs"]):
            if ok:
                oks.append(f"transform({ex['input']!r}) == {ex['output']!r}")
            else:
                got = out if out else "an error"
                fails.append(f"transform({ex['input']!r}) returned {got}, but should return {ex['output']!r}")
        lines.append("Running it on the examples gives:")
        lines += fails
        if oks:
            lines.append("(already correct on " + str(len(oks)) + " example(s))")
        lines.append("")
        lines.append("Use these results to fix the function so EVERY example is correct. "
                     "Respond with only the corrected function in one ```python block.")
    else:
        lines.append("That attempt was incorrect. Try a different approach. "
                     "Respond with only the corrected function in one ```python block.")
    return "\n".join(lines)


def repl(p, tasks, max_turns, mode, budget, *, greedy=False, init=None, batch_size=32):
    """Turn-by-turn batched sampled loop; tasks passing all visible commit and drop out.
    If `init` (per-task {code,res}) is given, reuse it as turn 0 and refine from turn 1 (paired control)."""
    st = [{"code": "", "turn0_code": "", "turn0_res": None, "turns": 0, "committed": False,
           "res": None, "history": []} for _ in tasks]
    if init is not None:
        for i, t in enumerate(tasks):
            c, r = init[i]["code"], init[i]["res"]
            st[i].update(code=c, turn0_code=c, turn0_res=r, res=r, turns=1,
                         history=[{"prompt": G.prompt_for(t), "code": c}],
                         committed=r["visible_all_pass"])
        active = [i for i in range(len(tasks)) if not st[i]["committed"]]
        start = 1
    else:
        active = list(range(len(tasks)))
        start = 0
    for turn in range(start, max_turns):
        if not active:
            break
        prompts = [G.prompt_for(tasks[i]) if turn == 0 else refine_prompt(tasks[i], st[i]["code"], st[i]["res"], mode)
                   for i in active]
        gens = p.gen_sequences(prompts, think=True, budget=budget, greedy=greedy, batch_size=batch_size)
        nxt = []
        for i, pr, g in zip(active, prompts, gens):
            code = extract(p, pr, g["seq_ids"])
            res = run_visible(tasks[i], code)
            st[i].update(code=code, turns=turn + 1, res=res)
            st[i]["history"].append({"prompt": pr, "code": code})
            if turn == 0:
                st[i]["turn0_code"], st[i]["turn0_res"] = code, res
            if res["visible_all_pass"]:
                st[i]["committed"] = True
            else:
                nxt.append(i)
        active = nxt
        print(f"    {mode} turn {turn+1}: {len(active)} active", flush=True)
    return st


def sample_more(p, tasks, n, budget, batch_size=48):
    """Return per-task list of n candidate dicts {code, vscore (visible pass count), hidden}."""
    rep = [G.prompt_for(t) for t in tasks for _ in range(n)]
    gens = p.gen_sequences(rep, think=True, budget=budget, greedy=False, batch_size=batch_size)
    per_task = []
    for ti, t in enumerate(tasks):
        cands = []
        for j in range(n):
            c = extract(p, rep[ti * n + j], gens[ti * n + j]["seq_ids"])
            cands.append({"code": c, "vscore": sum(run_visible(t, c)["public_passed"]), "hidden": grade_hidden(t, c)})
        per_task.append(cands)
    return per_task


def select_at(per_task, n):
    """Visible-test-selected hidden pass using the first n candidates (ties -> first by max)."""
    return [max(cands[:n], key=lambda c: c["vscore"])["hidden"] for cands in per_task]


def oracle_at(per_task, n):
    return [any(c["hidden"] for c in cands[:n]) for cands in per_task]


def by_depth(tasks, ok, gens):
    d = defaultdict(lambda: {"n": 0, "ok": 0, "gens": 0})
    for t, o, g in zip(tasks, ok, gens):
        r = d[t["depth"]]; r["n"] += 1; r["ok"] += int(o); r["gens"] += g
    return {dep: {"n": r["n"], "acc": round(r["ok"] / r["n"], 3), "mean_gens": round(r["gens"] / r["n"], 2)}
            for dep, r in sorted(d.items())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--per-depth", type=int, default=30)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=101)  # different seed from baseline -> fresh eval tasks
    args = ap.parse_args()
    if args.smoke:
        args.per_depth, args.depths, args.turns = 3, [2, 4], 3

    tasks = G.build_dataset(args.depths, args.per_depth, seed=args.seed)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "repl_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    print(f"{len(tasks)} tasks over depths {args.depths}; turns/samples={args.turns}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    print(f"model loaded {p.load_secs:.0f}s", flush=True)
    arms = {}
    t0 = time.time()

    # deterministic single-shot reference (comparable to the M1 baseline)
    g1 = p.gen_sequences([G.prompt_for(t) for t in tasks], think=True, budget=args.budget, greedy=True, batch_size=32)
    arms["greedy1"] = ([grade_hidden(t, extract(p, G.prompt_for(t), gg["seq_ids"])) for t, gg in zip(tasks, g1)],
                       [1 for _ in tasks])
    print(f"  greedy1 done [{time.time()-t0:.0f}s]", flush=True)

    print("  [repl_real] (sampled turns)", flush=True)
    real = repl(p, tasks, args.turns, "real", args.budget)
    arms["repl_real"] = ([grade_hidden(t, s["code"]) for t, s in zip(tasks, real)], [s["turns"] for s in real])
    print(f"  repl_real done [{time.time()-t0:.0f}s]", flush=True)

    print("  [repl_nofb] (paired: reuses real's turn-0)", flush=True)
    init = [{"code": s["turn0_code"], "res": s["turn0_res"]} for s in real]
    nofb = repl(p, tasks, args.turns, "nofb", args.budget, init=init)
    arms["repl_nofb"] = ([grade_hidden(t, s["code"]) for t, s in zip(tasks, nofb)], [s["turns"] for s in nofb])

    print("  [sample_more]", flush=True)
    smp = sample_more(p, tasks, args.turns, args.budget)  # per-task list of n candidate dicts
    arms["sample_more"] = (select_at(smp, args.turns), [args.turns for _ in tasks])
    sm_curve = {n: round(sum(select_at(smp, n)) / len(tasks), 3) for n in range(1, args.turns + 1)}
    sm_oracle_curve = {n: round(sum(oracle_at(smp, n)) / len(tasks), 3) for n in range(1, args.turns + 1)}
    sm_oracle = sm_oracle_curve[args.turns]
    print(f"  all arms done [{time.time()-t0:.0f}s]", flush=True)

    summary = {"n_tasks": len(tasks), "depths": args.depths, "turns": args.turns,
               "sample_more_oracle_passk": sm_oracle,
               "sample_more_selected_curve": sm_curve, "sample_more_oracle_curve": sm_oracle_curve, "arms": {}}
    for name, (ok, gens) in arms.items():
        summary["arms"][name] = {
            "acc": round(sum(ok) / len(ok), 3),
            "mean_gens": round(sum(gens) / len(gens), 2),
            "by_depth": by_depth(tasks, ok, gens)}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "repl_summary.json").write_text(json.dumps(summary, indent=2))
    # per-task records + full repl_real trajectories (for Milestone 3 self-training)
    with (EXP / "data" / "repl_records.jsonl").open("w") as f:
        for t, s, o in zip(tasks, real, arms["repl_real"][0]):
            f.write(json.dumps({"task_id": t["task_id"], "depth": t["depth"], "final_code": s["code"],
                                "turn0_code": s["turn0_code"], "turns": s["turns"],
                                "committed": s["committed"], "hidden_pass": o,
                                "history": s["history"]}) + "\n")

    print("\n=== M2: REPL vs SAMPLE-MORE (hidden acc, thinking on) ===")
    for name in ["greedy1", "sample_more", "repl_nofb", "repl_real"]:
        a = summary["arms"][name]
        print(f"  {name:12s} acc {a['acc']:.3f}  mean_gens {a['mean_gens']:.2f}   " +
              "  ".join(f"d{dp}:{v['acc']:.2f}" for dp, v in a["by_depth"].items()))
    print(f"  sample_more selected curve (by #samples): {sm_curve}")
    print(f"  sample_more pass@{args.turns} oracle (coverage): {sm_oracle:.3f}")
    rr, rn = summary["arms"]["repl_real"], summary["arms"]["repl_nofb"]
    print(f"\nKEY: repl_real ({rr['acc']:.3f} @ {rr['mean_gens']:.1f} gens) vs sample_more curve -> "
          f"does feedback beat independent sampling at MATCHED compute?")
    print(f"     repl_real {rr['acc']:.3f} vs repl_nofb {rn['acc']:.3f} -> is it the execution-feedback CONTENT?")

    # matched-compute figure: sample_more accuracy-vs-#samples, with the REPL points overlaid
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ns = list(range(1, args.turns + 1))
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(ns, [sm_curve[n] for n in ns], "s-", color="#2a9d8f", label="sample-more + visible-select")
    ax.plot(ns, [sm_oracle_curve[n] for n in ns], "o--", color="#8d99ae", label=f"sample-more oracle (pass@n)")
    ax.scatter([1], [summary["arms"]["greedy1"]["acc"]], color="#264653", zorder=5, label="greedy@1")
    ax.scatter([rn["mean_gens"]], [rn["acc"]], color="#e9c46a", s=90, zorder=5, marker="^", label="REPL no-feedback")
    ax.scatter([rr["mean_gens"]], [rr["acc"]], color="#e76f51", s=120, zorder=6, marker="*", label="REPL real feedback")
    ax.set_xlabel("generations (compute budget)"); ax.set_ylabel("hidden-test accuracy (deployable)")
    ax.set_ylim(0, max(sm_oracle_curve[args.turns], rr["acc"]) + 0.08)
    ax.set_title("Does execution-grounded self-correction beat independent sampling at matched compute?")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
    fig.savefig(EXP / "analysis" / "repl_vs_samplemore.png", dpi=130)
    print("wrote runs/repl_summary.json, analysis/repl_vs_samplemore.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
