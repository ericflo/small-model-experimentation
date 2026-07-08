#!/usr/bin/env python3
"""Verdict: does confidence-filtered banking recover execution-verified banking's gain, and does calibration
survive? Review-hardened: paired bootstrap over frozen tasks for every delta; JOINT bootstrap for the
recovery ratio (same task resamples in numerator and denominator) with CI; decision = pre-registered
TRICHOTOMY (conf~rand / intermediate / conf~exec), not a point-estimate cutoff; inflation headline =
mean_p_true_incorrect drift; self-distribution calibration reported alongside the fixed judge set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
R = EXP / "runs"
rng = np.random.default_rng(0)

ap = argparse.ArgumentParser()
ap.add_argument("--arms", type=str, default="base,exec,conf_strat,conf_global,rand",
                help="first arm is the baseline")
ap.add_argument("--modes", type=str, default="nothink,think")
ap.add_argument("--summary-file", type=str, default="runs/arms_summary.json")
args = ap.parse_args()
ARMS = args.arms.split(",")
BASE = ARMS[0]
MODES = args.modes.split(",")
COL = {a: c for a, c in zip(ARMS, ["#9ca3af", "#16a34a", "#2563eb", "#60a5fa", "#eab308", "#f97316"])}


def load(tag, mode):
    f = R / f"eval_{tag}_{mode}.json"
    return json.load(open(f))["records"] if f.exists() else None


def vec(recs, key, depth=None):
    sel = [r for r in recs if depth is None or r["depth"] == depth]
    if key == "cov_any":
        return np.array([r["cov_full"] > 0 for r in sel], float)
    if key == "cov_frac":
        return np.array([r["cov_full"] / r["K"] for r in sel], float)
    return np.array([r[key] for r in sel], float)


def pboot(a, b, n=10000):
    idx = [rng.integers(0, len(a), len(a)) for _ in range(n)]
    d = np.array([float(np.mean(a[i] - b[i])) for i in idx])
    return {"diff": round(float(np.mean(a - b)), 3),
            "ci": [round(float(np.percentile(d, 2.5)), 3), round(float(np.percentile(d, 97.5)), 3)],
            "p_one_sided": round(float(np.mean(d <= 0)), 4)}


def joint_ratio(conf_v, exec_v, base_v, n=10000):
    """Bootstrap (conf-base)/(exec-base) with the SAME task resamples top and bottom."""
    ratios = []
    for _ in range(n):
        i = rng.integers(0, len(base_v), len(base_v))
        den = float(np.mean(exec_v[i] - base_v[i]))
        if den > 1e-9:
            ratios.append(float(np.mean(conf_v[i] - base_v[i])) / den)
    if len(ratios) < n * 0.5:
        return None  # exec gain too unstable to define a ratio
    return {"point": round(float(np.mean(conf_v - base_v)) / max(1e-9, float(np.mean(exec_v - base_v))), 2),
            "ci": [round(float(np.percentile(ratios, 2.5)), 2), round(float(np.percentile(ratios, 97.5)), 2)],
            "frac_valid": round(len(ratios) / n, 2)}


out = {"arms_summary": json.load(open(EXP / args.summary_file)) if (EXP / args.summary_file).exists() else None,
       "eval": {}, "paired": {}, "recovery": {}, "trichotomy": {}, "calibration": {}, "calibration_self": {},
       "calibration_think": {}, "calibration_self_think": {},
       "note_posthoc": "think.greedy_full cells are POST-HOC (added after seeing raw evals); pre-registered "
                       "cells are nothink.greedy_full, nothink.cov_frac, think.cov_any, think.cov_frac"}

evals = {}
for tag in ARMS:
    for mode in MODES:
        recs = load(tag, mode)
        if recs:
            evals[(tag, mode)] = recs
depths = sorted({r["depth"] for recs in evals.values() for r in recs})

for (tag, mode), recs in evals.items():
    for d in depths:
        rs = [r for r in recs if r["depth"] == d]
        if not rs:
            continue
        K = rs[0]["K"]
        out["eval"].setdefault(f"{tag}_{mode}", {})[f"d{d}"] = {
            "n": len(rs), "greedy@1": round(float(np.mean([r["greedy_full"] for r in rs])), 3),
            "cov_any@K": round(float(np.mean([r["cov_full"] > 0 for r in rs])), 3),
            "cov_frac": round(float(np.mean([r["cov_full"] / K for r in rs])), 3),
            "uniq": round(float(np.mean([r["n_unique"] for r in rs])), 1)}

PREREG = {("nothink", "greedy_full"), ("nothink", "cov_frac"), ("think", "cov_any"), ("think", "cov_frac")}
CELLS = [("nothink", "greedy_full"), ("nothink", "cov_frac"), ("think", "cov_any"), ("think", "cov_frac"),
         ("think", "greedy_full")]  # last cell is post-hoc -- see note_posthoc
for mode, key in CELLS:
    if (BASE, mode) not in evals:
        continue
    for d in list(depths) + [None]:  # per-depth + pooled
        dname = (f"d{d}" if d else "pooled") + ("" if (mode, key) in PREREG else ".POSTHOC")
        base_v = vec(evals[(BASE, mode)], key, d)
        if len(base_v) == 0:
            continue
        cell = {}
        for arm in ARMS[1:]:
            if (arm, mode) in evals:
                av = vec(evals[(arm, mode)], key, d)
                if len(av) == len(base_v):
                    cell[f"{arm}_vs_{BASE}"] = pboot(av, base_v)
        for a, b in [("conf_strat", "rand"), ("conf_strat", "exec"), ("conf_global", "conf_strat")]:
            if (a, mode) in evals and (b, mode) in evals:
                av, bv = vec(evals[(a, mode)], key, d), vec(evals[(b, mode)], key, d)
                if len(av) == len(bv) > 0:
                    cell[f"{a}_vs_{b}"] = pboot(av, bv)
        if cell:
            out["paired"][f"{mode}.{key}.{dname}"] = cell
        # joint recovery ratio + trichotomy where the exec ceiling moved
        if all((t, mode) in evals for t in ["exec", "conf_strat"]):
            ev, cv = vec(evals[("exec", mode)], key, d), vec(evals[("conf_strat", mode)], key, d)
            if len(ev) == len(base_v) == len(cv) and float(np.mean(ev - base_v)) >= 0.10:
                jr = joint_ratio(cv, ev, base_v)
                if jr:
                    out["recovery"][f"{mode}.{key}.{dname}"] = jr
                cvr = cell.get("conf_strat_vs_rand", {})
                cve = cell.get("conf_strat_vs_exec", {})
                if cvr and cve:
                    beats_rand = cvr["ci"][0] > 0      # conf_strat > rand, CI excludes 0
                    below_exec = cve["ci"][1] < 0      # conf_strat < exec, CI excludes 0
                    out["trichotomy"][f"{mode}.{key}.{dname}"] = (
                        ("conf~exec" if not below_exec else "intermediate") if beats_rand else "conf~rand")

for tag in ARMS:
    for pref, dst in [("calib_", "calibration"), ("calib_self_", "calibration_self"),
                      ("calib_think_", "calibration_think"), ("calib_self_think_", "calibration_self_think")]:
        f = R / f"{pref}{tag}.json"
        if f.exists():
            out[dst][tag] = json.load(open(f))

cal, cals = out["calibration"], out["calibration_self"]
calt, calst = out["calibration_think"], out["calibration_self_think"]
cal_line = ""
if BASE in calt and "conf_strat" in calt:  # THINK judge = the working filter on this substrate (headline)
    b, c = calt[BASE], calt["conf_strat"]
    cal_line = (f" CALIBRATION of the THINK judge (the flywheel's filter; fixed judge set): within-AUROC "
                f"{b['auroc_within_mean']}->{c['auroc_within_mean']}, P(True)-on-INCORRECT "
                f"{b['mean_p_true_incorrect']}->{c['mean_p_true_incorrect']} (inflation headline)"
                + (f"; exec-banked within-AUROC {calt['exec']['auroc_within_mean']}" if "exec" in calt else "")
                + ".")
    if BASE in calst and "conf_strat" in calst:
        cal_line += (f" Self-distribution (think judge, own candidates): AUROC pooled "
                     f"{calst[BASE]['auroc_pooled']}->{calst['conf_strat']['auroc_pooled']}, own pass-rate "
                     f"{calst[BASE].get('self_pass_rate')}->{calst['conf_strat'].get('self_pass_rate')}.")
if BASE in cal and "conf_strat" in cal:
    b, c = cal[BASE], cal["conf_strat"]
    cal_line += (f" [secondary, no-think judge -- within-task chance on this substrate pre-banking: within-AUROC "
                 f"{b['auroc_within_mean']}->{c['auroc_within_mean']}, P(True)-on-INCORRECT "
                 f"{b['mean_p_true_incorrect']}->{c['mean_p_true_incorrect']}]")
    if BASE in cals and "conf_strat" in cals:
        cal_line += (f" Self-distribution (no-think): AUROC pooled {cals[BASE]['auroc_pooled']}->"
                     f"{cals['conf_strat']['auroc_pooled']}, own pass-rate "
                     f"{cals[BASE].get('self_pass_rate')}->{cals['conf_strat'].get('self_pass_rate')}.")
pur = out["arms_summary"]
pur_line = ""
if pur and "exec" in pur:
    pur_line = (f" Purity exec {pur['exec']['purity_full_pass']} / conf_strat "
                f"{pur['conf_strat']['purity_full_pass']} / conf_global {pur['conf_global']['purity_full_pass']}"
                f" / rand {pur['rand']['purity_full_pass']} at matched n={pur['gate']['n_pairs']}"
                f" (pool AUROC {pur['gate']['pool_auroc_pooled']}).")
tri = out["trichotomy"]
verd = ("Verifier-free banking verdict (pre-registered trichotomy per cell where exec gain >= 0.10): "
        + (", ".join(f"{k}: {v}" for k, v in sorted(tri.items())) if tri else "no cell cleared the exec-gain bar")
        + ". Joint-bootstrap recovery " + (", ".join(f"{k}={v['point']} CI{v['ci']}" for k, v in
                                                     sorted(out["recovery"].items())) or "n/a")
        + "." + pur_line + cal_line)
out["verdict"] = verd
(R / "verdict.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out["eval"], indent=1))
print()
print(verd)

# figure
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
width = 0.8 / max(1, len(ARMS))
for ax, (mode, key, ylab, title) in zip(axes[:2], [
        ("nothink", "greedy@1", "greedy@1 (no-think)", "Deployable single-shot"),
        ("think", "cov_any@K", "coverage@16 (think)", "Coverage ceiling (C18's effect locus)")]):
    for i, arm in enumerate(ARMS):
        k = f"{arm}_{mode}"
        if k in out["eval"]:
            ys = [out["eval"][k].get(f"d{d}", {}).get(key, 0) for d in depths]
            ax.bar([x + (i - len(ARMS) / 2 + 0.5) * width for x in range(len(depths))], ys, width,
                   label=arm, color=COL.get(arm))
    ax.set_xticks(range(len(depths))); ax.set_xticklabels([f"depth {d}" for d in depths])
    ax.set_ylabel(ylab); ax.legend(fontsize=7); ax.set_title(title)
ax = axes[2]
csrc, cname = (calt, "think judge") if calt else (cal, "no-think judge")
tags = [t for t in ARMS if t in csrc]
if tags:
    xs = np.arange(len(tags))
    ax.bar(xs - 0.2, [csrc[t].get("auroc_within_mean") or 0 for t in tags], 0.36,
           label="within-AUROC (fixed set)", color="#2563eb")
    ax.bar(xs + 0.2, [csrc[t].get("mean_p_true_incorrect") or 0 for t in tags], 0.36,
           label="P(True) on INCORRECT (inflation)", color="#ef4444")
    ax.axhline(0.5, ls=":", color="#888")
    ax.set_xticks(xs); ax.set_xticklabels(tags, fontsize=8)
    ax.set_title(f"Does the judge survive banking? ({cname})"); ax.legend(fontsize=7)
fig.suptitle("Can P(True) replace the execution verifier in the banking flywheel?", y=1.03)
fig.tight_layout()
(EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "verifier_free_banking.png", dpi=130, bbox_inches="tight")
print("wrote analysis/verifier_free_banking.png")
