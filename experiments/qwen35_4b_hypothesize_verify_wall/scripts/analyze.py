#!/usr/bin/env python3
"""Analysis: pre-registered primary contrast + characterization + verdict.

PRIMARY (pre-registered): per arm vs base, pooled list+string at DEPTH 3, probe-robust
skeleton-coverage@K, one-sided paired bootstrap, Holm-corrected across the 3 arm-vs-base
contrasts. C36's "un-installable/un-elicitable" reading falsified iff a corrected contrast has
one-sided 95% CI lower bound > 0 AND point diff >= falsify diff (0.10), with the dsl_sft claim
additionally required to hold in the zero-or-one-trained-window stratum.

Everything else (per-cell tables, depth-2, greedy@1, legacy metric, full-solve, mimicry rates,
parse rates, compute parity, K'-matched base) is characterization and can NEVER flip the verdict.
"""
from __future__ import annotations

import argparse
import json
import sys
from math import comb
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hv_common as C  # noqa: E402

EXP = C.EXP
ARMS = ["scaffold", "c45_zero", "dsl_sft"]
N_BOOT = 10_000


def load_eval(tag, sfx):
    p = EXP / "runs" / f"eval_{tag}{sfx}.json"
    return json.load(open(p)) if p.exists() else None


def task_flag(row, flag):
    return int(any(s[flag] for s in row["samples"]))


def cov_at_k(c, K, k):
    """Unbiased coverage@k estimate from c successes in K samples."""
    k = min(k, K)
    if c == 0:
        return 0.0
    if K - c < k:
        return 1.0
    return 1.0 - comb(K - c, k) / comb(K, k)


def paired_boot(a, b, seed=7):
    """One-sided paired bootstrap for mean(a-b) > 0. Returns diff, ci_lo (one-sided 95%), p."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(N_BOOT, len(d)))
    boots = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(boots, 5)), float((boots <= 0).mean())


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, mx = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        mx = max(mx, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, mx)
    return adj


def rows_by_id(ev):
    return {(r["family"], r["depth"], r["task_id"]): r for r in ev["rows"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    sfx = C.sfx(args.smoke)
    cfg = C.load_cfg()
    evs = {tag: load_eval(tag, sfx) for tag in ["base", "base_nothink"] + ARMS}
    assert evs["base"], "missing base eval -- run scripts/run.py first"
    base = evs["base"]
    K = base["K"]
    keys_d3 = [(r["family"], r["depth"], r["task_id"]) for r in base["rows"] if r["depth"] == 3]
    base_by = rows_by_id(base)

    verdict = {"smoke": args.smoke, "K": K, "n_depth3": len(keys_d3), "primary": {},
               "cells": {}, "compute_parity": {}, "mimicry": {}, "flags": {}}

    # ---- primary contrasts: pooled depth-3 probe-robust coverage@K, Holm across 3 -----------
    base_d3 = [task_flag(base_by[k], "probe_robust") for k in keys_d3]
    verdict["base_d3_probe_robust_cov"] = round(float(np.mean(base_d3)), 4)
    pvals, present = [], []
    for arm in ARMS:
        ev = evs[arm]
        if not ev:
            continue
        arm_by = rows_by_id(ev)
        arm_d3 = [task_flag(arm_by[k], "probe_robust") for k in keys_d3]
        diff, ci_lo, p = paired_boot(arm_d3, base_d3)
        verdict["primary"][arm] = {"cov": round(float(np.mean(arm_d3)), 4),
                                   "diff_vs_base": round(diff, 4),
                                   "ci_lo_one_sided95": round(ci_lo, 4), "p_raw": round(p, 5)}
        pvals.append(p)
        present.append(arm)
    for arm, p_adj in zip(present, holm(pvals)):
        e = verdict["primary"][arm]
        e["p_holm"] = round(p_adj, 5)
        e["significant"] = bool(p_adj < 0.05 and e["ci_lo_one_sided95"] > 0)
        e["clears_falsify_diff"] = bool(e["diff_vs_base"] >= 0.10)
    print("[analyze] PRIMARY (pooled depth-3 probe-robust coverage@K, arm - base):", flush=True)
    for arm in present:
        e = verdict["primary"][arm]
        print(f"[analyze]   {arm:10s} cov {e['cov']:.3f} diff {e['diff_vs_base']:+.3f} "
              f"ci_lo {e['ci_lo_one_sided95']:+.3f} p_holm {e['p_holm']:.4f} "
              f"{'SIG' if e['significant'] else 'ns'}", flush=True)

    # ---- window-overlap stratification for dsl_sft depth-3 ----------------------------------
    traces_p = EXP / "data" / f"traces{sfx}.jsonl"
    if evs["dsl_sft"] and traces_p.exists():
        trained = set()
        for r in C.load_jsonl(traces_p):
            types = tuple(op for op, _ in r["ops"])
            if len(types) == 2:
                trained.add((r["family"], types))
        eval_tasks = {(t["family"], t["depth"], t["task_id"]): t
                      for t in C.load_jsonl(EXP / "data" / f"eval_tasks{sfx}.jsonl")}
        dsl_by = rows_by_id(evs["dsl_sft"])
        strata = {0: [], 1: [], 2: []}
        for k in keys_d3:
            tt = C.true_types(eval_tasks[k])
            n_win = sum(((k[0], (tt[i], tt[i + 1])) in trained) for i in range(2))
            strata[n_win].append((task_flag(dsl_by[k], "probe_robust"),
                                  task_flag(base_by[k], "probe_robust")))
        strat = {}
        for n_win, pairs in strata.items():
            label = {0: "zero", 1: "one", 2: "both"}[n_win]
            if pairs:
                a, b = zip(*pairs)
                strat[label] = {"n": len(pairs), "dsl_cov": round(float(np.mean(a)), 4),
                                "base_cov": round(float(np.mean(b)), 4),
                                "diff": round(float(np.mean(a) - np.mean(b)), 4)}
            else:
                strat[label] = {"n": 0}
            print(f"[analyze] dsl_sft d3 stratum {label}-windows-trained: {strat[label]}", flush=True)
        zo = [p for n_win in (0, 1) for p in strata[n_win]]
        if zo:
            a, b = zip(*zo)
            diff, ci_lo, p = paired_boot(list(a), list(b))
            strat["zero_or_one"] = {"n": len(zo), "diff": round(diff, 4),
                                    "ci_lo_one_sided95": round(ci_lo, 4), "p_raw": round(p, 5)}
        verdict["dsl_window_strata"] = strat

    # ---- per-cell characterization + mimicry + parse rates ----------------------------------
    def cell_stats(ev):
        out = {}
        for r in ev["rows"]:
            key = f"{r['family']}_d{r['depth']}"
            c = out.setdefault(key, {"n": 0, "probe": 0, "legacy": 0, "full": 0, "greedy_full": 0,
                                     "greedy_probe": 0, "samp": 0, "parsed": 0, "mim": 0})
            c["n"] += 1
            c["probe"] += task_flag(r, "probe_robust")
            c["legacy"] += task_flag(r, "legacy")
            c["full"] += task_flag(r, "full")
            c["greedy_full"] += int(r["greedy"]["full"])
            c["greedy_probe"] += int(r["greedy"]["probe_robust"])
            for s in r["samples"]:
                c["samp"] += 1
                c["parsed"] += int(s["parsed"])
                c["mim"] += int(s["mimicry"])
        for c in out.values():
            n, ns = c.pop("n"), c.pop("samp")
            c.update({k: round(c[k] / n, 4) for k in ("probe", "legacy", "full", "greedy_full", "greedy_probe")})
            c["parse_rate"] = round(c.pop("parsed") / max(1, ns), 4)
            c["mimicry_rate"] = round(c.pop("mim") / max(1, ns), 4)
            c["n"] = n
        return out

    for tag in ["base", "base_nothink"] + ARMS:
        if evs[tag]:
            verdict["cells"][tag] = cell_stats(evs[tag])
    print("[analyze] per-cell probe-robust cov@K | legacy | full | parse | mimicry:", flush=True)
    for tag, cells in verdict["cells"].items():
        for key in sorted(cells):
            c = cells[key]
            print(f"[analyze]   {tag:13s} {key:9s} {c['probe']:.2f} | {c['legacy']:.2f} | "
                  f"{c['full']:.2f} | {c['parse_rate']:.2f} | {c['mimicry_rate']:.2f}", flush=True)

    # ---- compute-parity table + K'-matched base presentation --------------------------------
    for tag in ["base", "base_nothink"] + ARMS:
        ev = evs[tag]
        if not ev:
            continue
        by_depth = {}
        for r in ev["rows"]:
            d = by_depth.setdefault(r["depth"], {"pt": [], "gt": [], "forced": 0, "ns": 0})
            d["pt"].append(r["prompt_tokens"])
            for s in r["samples"]:
                d["gt"].append(s["gen_tokens"])
                d["forced"] += int(s["forced"])
                d["ns"] += 1
        verdict["compute_parity"][tag] = {
            f"d{d}": {"mean_prompt_tokens": round(float(np.mean(v["pt"])), 1),
                      "mean_gen_tokens_per_sample": round(float(np.mean(v["gt"])), 1),
                      "total_gen_tokens_per_task": round(float(np.sum(v["gt"]) / len(v["pt"])), 1),
                      "forced_close_rate": round(v["forced"] / max(1, v["ns"]), 4)}
            for d, v in sorted(by_depth.items())}
    if evs["scaffold"]:
        cp_b = verdict["compute_parity"]["base"]["d3"]
        cp_s = verdict["compute_parity"]["scaffold"]["d3"]
        k_prime = int(round(K * cp_s["total_gen_tokens_per_task"]
                            / max(1.0, cp_b["total_gen_tokens_per_task"])))
        counts = [sum(s["probe_robust"] for s in base_by[k]["samples"]) for k in keys_d3]
        cov_kp = float(np.mean([cov_at_k(c, K, min(k_prime, K)) for c in counts]))
        verdict["k_prime_matched_base"] = {
            "k_prime": k_prime, "capped_at_K": bool(k_prime > K),
            "base_d3_probe_cov_at_min(k_prime,K)": round(cov_kp, 4),
            "note": "K' = K * scaffold_total_gen_tokens / base_total_gen_tokens (depth-3); "
                    "if capped, base has no extra samples -- matched-token base is bounded below by this"}
        print(f"[analyze] K'-matched base: K'={k_prime} -> base d3 probe-cov "
              f"{cov_kp:.3f} (capped={k_prime > K})", flush=True)

    # ---- pre-registered flags ----------------------------------------------------------------
    verdict["flags"]["base_think_clears_wall"] = bool(verdict["base_d3_probe_robust_cov"] >= 0.25)
    fc = cfg["decision"]["forced_close_contingency"]
    needs_probe = [arm for arm in present
                   if not verdict["primary"][arm]["significant"]
                   and verdict["compute_parity"].get(arm, {}).get("d3", {}).get("forced_close_rate", 0) > fc]
    verdict["flags"]["budget_2048_probe_required_for"] = needs_probe
    fmt_conf = [arm for arm in present
                if not verdict["primary"][arm]["significant"]
                and any(verdict["cells"][arm][cell]["parse_rate"]
                        < 0.5 * max(0.01, verdict["cells"]["base"][cell]["parse_rate"])
                        for cell in verdict["cells"][arm])]
    verdict["flags"]["format_confounded_null_candidates"] = fmt_conf
    trap_p = EXP / "runs" / f"trap_gate{sfx}.json"
    if trap_p.exists():
        verdict["gates_trap"] = json.load(open(trap_p))
    for g in ("gate_c45", "gate_install"):
        gp = EXP / "runs" / f"{g}{sfx}.json"
        if gp.exists():
            verdict[g] = json.load(open(gp))

    sig = [a for a in present if verdict["primary"][a]["significant"]
           and verdict["primary"][a]["clears_falsify_diff"]]
    if "dsl_sft" in sig and "dsl_window_strata" in verdict:
        zo = verdict["dsl_window_strata"].get("zero_or_one", {})
        if not (zo.get("n", 0) and zo.get("ci_lo_one_sided95", -1) > 0):
            sig = [a for a in sig if a != "dsl_sft"]
            verdict["flags"]["dsl_lift_both_windows_only"] = True
    verdict["c36_falsified_for_content_free_installs"] = bool(sig)
    verdict["lifting_arms"] = sig
    print(f"[analyze] VERDICT: c36_falsified_for_content_free_installs={bool(sig)} "
          f"(arms: {sig or 'none'})", flush=True)

    out_p = EXP / "runs" / f"verdict{sfx}.json"
    # --- budget-2048 contingency probes (pre-committed: scaffold/c45_zero; post-hoc control: base) ---
    cont = {}
    b2048 = {}
    for arm in ("scaffold", "c45_zero", "base"):
        f = EXP / "runs" / f"eval_{arm}_b2048{sfx}.json"
        if not f.exists():
            continue
        rows = json.load(open(f))["rows"]
        b2048[arm] = rows
        ns = sum(len(r["samples"]) for r in rows)
        cont[arm] = {"pooled_d3_probe_cov": round(sum(task_flag(r, "probe_robust") for r in rows) / len(rows), 4),
                     "parse_rate": round(sum(x["parsed"] for r in rows for x in r["samples"]) / ns, 3),
                     "forced_close_rate": round(sum(x["forced"] for r in rows for x in r["samples"]) / ns, 3),
                     "n": len(rows), "K": rows[0].get("K", 12) if rows else None, "budget": 2048}
    if "scaffold" in b2048 and "base" in b2048:
        bid = {(r["family"], r["task_id"]): float(task_flag(r, "probe_robust")) for r in b2048["base"]}
        pairs = [(float(task_flag(r, "probe_robust")), bid[(r["family"], r["task_id"])]) for r in b2048["scaffold"]]
        diff, ci_lo, pv = paired_boot([a for a, _ in pairs], [b for _, b in pairs])
        seeds = {sd: paired_boot([a for a, _ in pairs], [b for _, b in pairs], seed=sd)[2] for sd in (0, 7, 42)}
        cont["scaffold_vs_base_matched_2048"] = {
            "diff": round(diff, 4), "ci_lo_one_sided95": round(ci_lo, 4), "p_one_sided": round(pv, 4),
            "p_across_seeds": {str(k): round(v, 4) for k, v in seeds.items()},
            "note": "POST-HOC control contrast (base@2048 not pre-registered); borderline and seed-fragile"}
    if cont:
        verdict["contingency_b2048"] = cont
    json.dump(verdict, open(out_p, "w"), indent=1)
    print(f"[analyze] wrote {out_p.name}", flush=True)

    # ---- figure -------------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    tags = ["base"] + present
    covs = [verdict["base_d3_probe_robust_cov"]] + [verdict["primary"][a]["cov"] for a in present]
    axes[0].bar(range(len(tags)), covs, color=["#888"] + ["#3b6fb6"] * len(present))
    axes[0].set_xticks(range(len(tags)), tags, rotation=15)
    axes[0].set_ylabel(f"probe-robust skeleton-cov@{K}")
    axes[0].set_title("Depth-3 pooled (PRIMARY)")
    axes[0].axhline(0.25, ls="--", c="#c33", lw=0.8)
    axes[0].set_ylim(0, 1)
    cells = sorted(verdict["cells"]["base"])
    width = 0.8 / len(tags)
    for j, tag in enumerate(tags):
        vals = [verdict["cells"][tag][c]["probe"] for c in cells]
        axes[1].bar([i + j * width for i in range(len(cells))], vals, width, label=tag)
    axes[1].set_xticks([i + 0.4 for i in range(len(cells))], cells, rotation=15)
    axes[1].set_title("Per-cell probe-robust cov@K")
    axes[1].legend(fontsize=7)
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    (EXP / "analysis").mkdir(exist_ok=True)
    fig_p = EXP / "analysis" / f"hypothesize_verify_wall{sfx}.png"
    fig.savefig(fig_p, dpi=150)
    print(f"[analyze] wrote {fig_p.relative_to(EXP)}", flush=True)


if __name__ == "__main__":
    main()
