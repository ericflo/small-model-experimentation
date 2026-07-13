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
    """Merge the base pool with any extension shards (pool_think_b{b}*.json)."""
    rows, seen = [], set()
    for f in sorted((EXP / "runs").glob(f"pool_think_b{b}*.json")):
        for t in json.loads(f.read_text()):
            if t["task_id"] not in seen:
                seen.add(t["task_id"]); rows.append(t)
    return rows


def bootstrap_ci(deltas, rng, B=2000, lo=2.5, hi=97.5):
    """Paired bootstrap over per-task (esc-brd) deltas -> (mean, ci_lo, ci_hi)."""
    n = len(deltas)
    means = []
    for _ in range(B):
        s = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    return (statistics.mean(deltas), means[int(lo / 100 * B)], means[int(hi / 100 * B)])


def conf_select(cands):
    return max(cands, key=pt)


def auroc_solvable(pool):
    """AUROC of max-p_true predicting whether the task is solvable within the pool."""
    pairs = [(max(pt(c) for c in t["cands"]), any(c["full_pass"] for c in t["cands"])) for t in pool]
    pos = [m for m, s in pairs if s]; neg = [m for m, s in pairs if not s]
    if not pos or not neg:
        return float("nan")
    return (sum(1 for a in pos for b in neg if a > b)
            + 0.5 * sum(1 for a in pos for b in neg if a == b)) / (len(pos) * len(neg))


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
    print(f"{'abst_frac':>9} {'n_abst':>7} {'base':>6} {'escal':>6} {'breadth':>7} "
          f"{'esc-brd':>8} {'95% CI':>18}")
    abstained_rows = []
    c256_mean = statistics.mean(cost(c) for t in p256 for c in t["cands"])
    for frac in (0.2, 0.3, 0.4, 0.5):
        ranked = sorted(tasks, key=lambda t: base[t["task_id"]]["maxpt"])
        n_abst = max(1, int(len(tasks) * frac))
        abst = ranked[:n_abst]
        base_acc = statistics.mean(base[t["task_id"]]["pass"] for t in abst)
        # per-task R-averaged conf-select accuracy for ESCALATE (2048, k1=2) and
        # BREADTH (256, k_extra chosen so the two arms match token compute).
        esc_means, esc_tok_l = [], []
        for t in abst:
            hi = by_id[t["task_id"]]["cands"]
            subs = [rng.sample(hi, 2) if len(hi) > 2 else hi for _ in range(R)]
            esc_means.append(statistics.mean(conf_select(s)["full_pass"] for s in subs))
            esc_tok_l.append(statistics.mean(sum(cost(c) for c in s) for s in subs))
        esc_tok = statistics.mean(esc_tok_l)
        k_extra = max(1, round(esc_tok / c256_mean))
        brd_means = []
        for t in abst:
            lo = t["cands"]; kk = min(k_extra, len(lo))
            subs = [rng.sample(lo, kk) for _ in range(R)]
            brd_means.append(statistics.mean(conf_select(s)["full_pass"] for s in subs))
        deltas = [e - b for e, b in zip(esc_means, brd_means)]
        dmean, ci_lo, ci_hi = bootstrap_ci(deltas, rng)
        print(f"{frac:>9.2f} {n_abst:>7} {base_acc:>6.3f} "
              f"{statistics.mean(esc_means):>6.3f} {statistics.mean(brd_means):>7.3f} "
              f"{dmean:>+8.3f} [{ci_lo:+.3f},{ci_hi:+.3f}]  (brd k={k_extra})")
        abstained_rows.append({
            "abstain_frac": frac, "n_abstained": n_abst,
            "base_acc": round(base_acc, 3),
            "escalate_acc": round(statistics.mean(esc_means), 3),
            "breadth_acc": round(statistics.mean(brd_means), 3),
            "esc_minus_brd": round(dmean, 3),
            "ci95": [round(ci_lo, 3), round(ci_hi, 3)], "breadth_k": k_extra})

    out = {"paired_tasks": len(tasks), "pure256_curve": c256, "pure2048_curve": c2048,
           "abstained_escalation": abstained_rows,
           "solvability_auroc": {"b256": round(auroc_solvable(p256), 3),
                                 "b2048": round(auroc_solvable(p2048), 3)}}
    (EXP / "runs" / "escalation.json").write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {EXP/'runs'/'escalation.json'}")


if __name__ == "__main__":
    main()
