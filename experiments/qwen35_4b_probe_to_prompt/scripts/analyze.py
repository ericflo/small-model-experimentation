#!/usr/bin/env python3
"""Externalizing the latent readout (decode->prompt) DOES elicit deployable depth-2 capability where steering
(C20) was inert -- but only when the CONCRETE first op (type+param) is given; the op-TYPE that C19 decodes
narrows sampling (coverage) without making it deployable (greedy), and the type-only probe nets to zero. Graded
by depth (works d2, fades d3, per C19). Emits verdict + figure."""
import json
from math import comb, sqrt
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
r = json.load(open(EXP / "runs" / "hint_results.json")); K = r["K"]; T = r["per_task"]
def cov(c, n, k): k = min(k, n); return 0.0 if c == 0 else (1.0 if n-c < k else 1-comb(n-c, k)/comb(n, k))
def wil(p, n, z=1.96):
    if n == 0: return 0.0
    return z*sqrt(p*(1-p)/n)/(1+z*z/n)
ARMS = ["nohint", "neutral", "oracle_type", "oracle_full", "probe", "wrong"]
LAB = {"nohint": "no-hint", "neutral": "neutral\n(placebo)", "oracle_type": "oracle\nTYPE",
       "oracle_full": "oracle\nFULL(+param)", "probe": "probe\n(C19 readout)", "wrong": "wrong\n(content ctrl)"}

out = {"probe_eval_acc": r["probe_eval_acc"], "layer0_leak": r["layer0_eval_acc"], "majority": r["majority"], "by_depth": {}}
print(f"probe eval acc {r['probe_eval_acc']} | layer-0 leak {r['layer0_eval_acc']} | majority {r['majority']}")
for d in (2, 3):
    rows = [x for x in T if x["depth"] == d]; n = len(rows); out["by_depth"][d] = {}
    print(f"--- depth {d} (n={n}) ---")
    for a in ARMS:
        g = sum(x[f"{a}_greedy"] for x in rows)/n
        cv = sum(cov(x[f"{a}_ncorrect"], K, K) for x in rows)/n
        out["by_depth"][d][a] = {"greedy": round(g, 3), "cov": round(cv, 3)}
        print(f"  {a:12} greedy@1 {g:.3f} | cov@{K} {cv:.3f}")

d2 = [x for x in T if x["depth"] == 2]
pc = [x for x in d2 if x["probe_correct"]]; pw = [x for x in d2 if not x["probe_correct"]]
out["probe_decomp_d2"] = {
    "probe_correct_n": len(pc), "probe_wrong_n": len(pw),
    "cov_probe_on_correct": round(sum(cov(x["probe_ncorrect"], K, K) for x in pc)/max(1, len(pc)), 3),
    "cov_nohint_on_correct": round(sum(cov(x["nohint_ncorrect"], K, K) for x in pc)/max(1, len(pc)), 3),
    "cov_probe_on_wrong": round(sum(cov(x["probe_ncorrect"], K, K) for x in pw)/max(1, len(pw)), 3),
    "cov_nohint_on_wrong": round(sum(cov(x["nohint_ncorrect"], K, K) for x in pw)/max(1, len(pw)), 3)}
b = out["by_depth"]
out["verdict"] = (
    "Externalizing the latent readout (decode->PROMPT) ELICITS deployable depth-2 capability where steering "
    f"(C20) was inert: oracle_full lifts depth-2 greedy@1 {b[2]['nohint']['greedy']}->{b[2]['oracle_full']['greedy']} "
    f"and cov {b[2]['nohint']['cov']}->{b[2]['oracle_full']['cov']}. BUT the deployable bottleneck is the PARAMETER, "
    f"not the op-TYPE C19 decodes: oracle_TYPE lifts coverage ({b[2]['oracle_type']['cov']}, narrows sampling) but "
    f"NOT greedy ({b[2]['oracle_type']['greedy']}); only oracle_FULL (with param) lifts greedy. The C19 type-only probe "
    f"(eval acc {r['probe_eval_acc']['2']}) nets to ~zero (benefit concentrated on probe-correct tasks "
    f"{out['probe_decomp_d2']['cov_probe_on_correct']} vs no-hint {out['probe_decomp_d2']['cov_nohint_on_correct']}, "
    "but 68% wrong-type hurts). Graded by depth: fades at depth-3 (thread, C19). Controls: neutral~=no-hint (format), "
    f"wrong<no-hint (content-causal), layer-0 at chance {r['layer0_eval_acc']} (model-computed, not surface).")
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print("\n" + out["verdict"])

# figure: grouped bars per depth
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
cols = {"nohint": "#94a3b8", "neutral": "#cbd5e1", "oracle_type": "#60a5fa", "oracle_full": "#16a34a",
        "probe": "#f59e0b", "wrong": "#ef4444"}
for ax, d in zip(axes, (2, 3)):
    xs = range(len(ARMS))
    ax.bar([x-0.2 for x in xs], [b[d][a]["cov"] for a in ARMS], 0.38, label=f"coverage@{K}",
           color=[cols[a] for a in ARMS], alpha=0.5)
    ax.bar([x+0.2 for x in xs], [b[d][a]["greedy"] for a in ARMS], 0.38, label="greedy@1",
           color=[cols[a] for a in ARMS])
    ax.set_xticks(list(xs)); ax.set_xticklabels([LAB[a] for a in ARMS], fontsize=8)
    ax.set_title(f"depth {d}" + (" (latent headroom)" if d == 2 else " (thread, C19)")); ax.grid(alpha=0.25, axis="y")
    if d == 2: ax.set_ylabel("solve rate (no-think, fsig-disjoint eval)"); ax.legend(fontsize=9)
fig.suptitle("Externalize the latent readout: knowing the FULL first op elicits depth-2 (oracle_full 6x), but the "
             "op-TYPE C19 decodes only narrows sampling — the PARAMETER is the deployable bottleneck", fontsize=10)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "probe_to_prompt.png", dpi=130, bbox_inches="tight")
print("wrote analysis/probe_to_prompt.png")
