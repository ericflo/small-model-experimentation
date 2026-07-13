#!/usr/bin/env python3
"""Escalation vs breadth: is compute better spent on more breadth (low budget)
or more serial depth (high think budget) on the same 4B?

Two think-mode MBPP pools over the SAME 120 tasks: budget 256 (tight -> hard
tasks forced to commit early) and budget 2048 (generous). Each candidate has
full_pass, p_true, n_think, code_len. Per-candidate compute is measured in
generated tokens (n_think + answer tokens). We compare, at MATCHED token
compute and with verifier-free confidence-select (argmax p_true) as the deploy
rule:
  * pure-256   : spend all compute on breadth at budget 256
  * pure-2048  : spend all compute on depth at budget 2048
  * escalate   : k0 breadth at 256; ABSTAINED tasks (low max p_true) re-solved
                 at 2048; the C40/C41 confidence signal aims the C44/C55 lever.
Headline: does escalate (or pure-2048) beat pure-256 on accuracy-vs-tokens,
especially on the hard/abstained subset? CPU-only, deterministic.
"""
from __future__ import annotations
import json, random, statistics
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
R = 200
SEED = 31337
CHARS_PER_TOK = 3.5


def pt(c):
    v = c.get("p_true")
    return float(v) if v is not None else 0.0


def cost(c):  # generated tokens = thinking + answer (answer est from code chars)
    return c.get("n_think", 0) + c.get("code_len", 0) / CHARS_PER_TOK


def load(b):
    return json.loads((EXP / "runs" / f"pool_think_b{b}.json").read_text())


def conf_select(cands):
    return max(cands, key=pt)


def curve(pool, rng, kmax=7):
    """accuracy & mean token-cost of conf-select vs k (subsample-averaged)."""
    out = []
    for k in range(1, kmax + 1):
        accs, toks = [], []
        for t in pool:
            cands = t["cands"]
            subs = [cands] if k >= len(cands) else [rng.sample(cands, k) for _ in range(R)]
            accs.append(statistics.mean(conf_select(s)["full_pass"] for s in subs))
            toks.append(statistics.mean(sum(cost(c) for c in s) for s in subs))
        out.append({"k": k, "acc": round(statistics.mean(accs), 4),
                    "tokens": round(statistics.mean(toks), 1)})
    return out


def main():
    rng = random.Random(SEED)
    p256, p2048 = load(256), load(2048)
    by_id = {t["task_id"]: t for t in p2048}
    tasks = [t for t in p256 if t["task_id"] in by_id]
    print(f"paired tasks: {len(tasks)}")

    # 0) sanity: solvability & per-cand pass at each budget
    for name, pool in (("256", p256), ("2048", p2048)):
        cds = [c for t in pool for c in t["cands"]]
        print(f"  b{name}: task-solvable@k6 "
              f"{statistics.mean(any(c['full_pass'] for c in t['cands']) for t in pool):.3f} | "
              f"per-cand pass {statistics.mean(c['full_pass'] for c in cds):.3f} | "
              f"mean n_think {statistics.mean(c.get('n_think',0) for c in cds):.0f} | "
              f"mean cost {statistics.mean(cost(c) for c in cds):.0f} tok")

    # 1) accuracy-vs-tokens frontier, pure-256 vs pure-2048
    print("\n=== conf-select accuracy vs total tokens ===")
    c256, c2048 = curve(p256, rng), curve(p2048, rng)
    print("  pure-256 :", " ".join(f"k{r['k']}({r['tokens']:.0f}tok->{r['acc']:.3f})" for r in c256))
    print("  pure-2048:", " ".join(f"k{r['k']}({r['tokens']:.0f}tok->{r['acc']:.3f})" for r in c2048))

    # 2) escalation on the abstained subset, matched compute.
    # base = conf-select over k0=2 @256 for all tasks; abstain bottom by max p_true.
    k0 = 2
    base = {}
    for t in tasks:
        sub = rng.sample(t["cands"], k0) if len(t["cands"]) > k0 else t["cands"]
        sel = conf_select(sub)
        base[t["task_id"]] = {"maxpt": max(pt(c) for c in sub), "pass": sel["full_pass"],
                              "tok": sum(cost(c) for c in sub)}
    print("\n=== escalation vs breadth on the abstained subset (matched compute) ===")
    print(f"{'abst_frac':>9} {'n_abst':>7} {'base_acc':>8} {'esc_acc':>8} {'brd_acc':>8} "
          f"{'esc_tok':>8} {'brd_tok':>8}")
    abstained_rows = []
    for frac in (0.2, 0.3, 0.4, 0.5):
        ranked = sorted(tasks, key=lambda t: base[t["task_id"]]["maxpt"])
        n_abst = max(1, int(len(tasks) * frac))
        abst = ranked[:n_abst]
        base_acc = statistics.mean(base[t["task_id"]]["pass"] for t in abst)
        # ESCALATE: re-solve each abstained task with the 2048 pool (conf-select, k1=2)
        esc_acc, esc_extra = [], []
        for t in abst:
            hi = by_id[t["task_id"]]["cands"]
            sub = rng.sample(hi, 2) if len(hi) > 2 else hi
            esc_acc.append(conf_select(sub)["full_pass"]); esc_extra.append(sum(cost(c) for c in sub))
        esc_tok = statistics.mean(esc_extra)
        # BREADTH control at matched compute: give abstained tasks MORE 256 samples,
        # as many as the escalation tokens buy (cap at the pool size).
        c256_mean = statistics.mean(cost(c) for t in p256 for c in t["cands"])
        k_extra = max(1, round(esc_tok / c256_mean))
        brd_acc, brd_tok = [], []
        for t in abst:
            lo = t["cands"]
            kk = min(k_extra, len(lo))
            subs = [rng.sample(lo, kk) for _ in range(R)]
            brd_acc.append(statistics.mean(conf_select(s)["full_pass"] for s in subs))
            brd_tok.append(kk * c256_mean)
        print(f"{frac:>9.2f} {n_abst:>7} {base_acc:>8.3f} "
              f"{statistics.mean(esc_acc):>8.3f} {statistics.mean(brd_acc):>8.3f} "
              f"{esc_tok:>8.0f} {statistics.mean(brd_tok):>8.0f}  (breadth k={k_extra})")
        abstained_rows.append({
            "abstain_frac": frac, "n_abstained": n_abst,
            "base_acc": round(base_acc, 3),
            "escalate_acc": round(statistics.mean(esc_acc), 3),
            "breadth_acc": round(statistics.mean(brd_acc), 3),
            "breadth_k": k_extra})

    out = {"paired_tasks": len(tasks), "pure256_curve": c256, "pure2048_curve": c2048,
           "abstained_escalation": abstained_rows}
    (EXP / "runs" / "escalation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {EXP/'runs'/'escalation.json'}")


if __name__ == "__main__":
    main()
