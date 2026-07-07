#!/usr/bin/env python3
"""End-to-end bank+value-fill deploy. Confirms C33's ~0.51 (bank-fill deploy 0.463) BUT the decisive brute-force
control reveals the real story: brute-force structure enumeration + value-fill + execution-consensus deploys at
0.975 (near-solving depth-3) WITHOUT the model. Using the model's structure CAPS deploy at its structure-coverage
(0.46) -- worse than ignoring the model. With the interpreter available, free structure-search dominates;
banking's value is forward-pass-only. Emits verdict + figure."""
import json
from math import sqrt
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
d = json.load(open(EXP / "runs" / "bankfill_results.json")); rows = d["rows"]; n = d["n"]; K = d["k"]
def wil(p): z = 1.96; return z*sqrt(p*(1-p)/n)/(1+z*z/n)
def R(key): return round(sum(bool(r[key]) for r in rows)/n, 3)
from math import comb
def covk(c, kk=8): kk = min(kk, K); return 0.0 if c == 0 else (1.0 if K-c < kk else 1-comb(K-c, kk)/comb(K, kk))

m = {
 "banked_greedy": round(sum(r["b_greedy"] for r in rows)/n, 3),
 "banked_cov8": round(sum(covk(r["b_cov"]) for r in rows)/n, 3),
 "struct_cov": round(sum(r["struct"] for r in rows)/n, 3),
 "bank_deploy": R("bank_deploy"), "bank_cov": R("bank_cov"),
 "brute_deploy": R("brute_deploy"), "brute_cov": R("brute_cov"),
 "brute_overfit": round(sum(b for b, _ in [r["brute_overfit"] for r in rows])/max(1, sum(v for _, v in [r["brute_overfit"] for r in rows])), 3),
 "mean_brute_skels": round(sum(r["brute_skels"] for r in rows)/n, 1),
 "mean_bank_inferred": round(sum(r["bank_skels"] for r in rows)/n, 1),
 "infer_yield": round(sum(r["infer_yield"] for r in rows)/n, 3),
}
m["bank_deploy_ci"] = round(wil(m["bank_deploy"]), 3); m["brute_deploy_ci"] = round(wil(m["brute_deploy"]), 3)
print(json.dumps(m, indent=1))
m["verdict"] = (
 f"End-to-end bank+value-fill CONFIRMS C33's ~0.51 (bank-fill deploy {m['bank_deploy']} +/-{m['bank_deploy_ci']}, "
 f"~= struct-cov {m['struct_cov']}) BUT the decisive brute-force control reveals the real story: brute-force "
 f"structure enumeration (all 4096 depth-3 skeletons) + value-fill + execution-consensus deploys at "
 f"{m['brute_deploy']} +/-{m['brute_deploy_ci']} (near-solving depth-3) using NO model structure -- because after "
 f"the 8-visible filter only ~{m['mean_brute_skels']} skeletons survive and consensus picks the right one "
 f"(visible-overfit {m['brute_overfit']}). Using the MODEL's structure (bank-fill) CAPS deploy at the model's "
 f"structure-coverage {m['struct_cov']} -- worse than ignoring the model ({m['brute_deploy']}), because the model "
 f"proposes the right structure only ~48% of the time. So with the interpreter available (mission-legal, free "
 f"selection per C17), free structure-SEARCH dominates the model at deploy; banking's structure is a forward-PASS "
 f"asset (0.20->0.475, C33), NOT a deploy-with-interpreter asset. Scope: brute-force wins because the depth-3 "
 f"structure space (4096) is enumerable; the model's structure-pruning would only matter when the space is too "
 f"large to brute-force.")
(EXP / "runs" / "verdict_bankfill.json").write_text(json.dumps(m, indent=1))
print("\n" + m["verdict"])

fig, ax = plt.subplots(figsize=(10, 5.4))
labels = ["banked\ngreedy@1\n(forward pass)", "bank-fill\n(model structure\n+fill+select)", "brute-fill\n(SEARCH structure\n+fill+select)"]
vals = [m["banked_greedy"], m["bank_deploy"], m["brute_deploy"]]
cis = [0, m["bank_deploy_ci"], m["brute_deploy_ci"]]
cols = ["#94a3b8", "#a855f7", "#16a34a"]
ax.bar(range(3), vals, 0.55, color=cols, yerr=cis, capsize=5)
for i, v in enumerate(vals): ax.text(i, v+0.02, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
ax.axhline(m["struct_cov"], ls="--", color="#a855f7", alpha=0.6, label=f"banked structure-cov {m['struct_cov']} (bank-fill ceiling)")
ax.axhline(m["brute_cov"], ls=":", color="#16a34a", alpha=0.5, label=f"brute coverage {m['brute_cov']}")
ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0, 1.05)
ax.set_ylabel("deploy rate (held-out depth-3, n=80, execution-consensus select)")
ax.legend(fontsize=8, loc="center left"); ax.grid(alpha=0.25, axis="y")
ax.set_title("Bank+value-fill deploy: the model's structure (0.46) is DOMINATED by brute-force structure search (0.975).\n"
             "With the interpreter, free structure-search near-solves depth-3; banking's structure is forward-pass-only.", fontsize=9.5)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "bank_fill_deploy.png", dpi=130, bbox_inches="tight")
print("wrote analysis/bank_fill_deploy.png")
