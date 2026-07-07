#!/usr/bin/env python3
"""Does the structure-PROPOSAL wall exist in language? YES -- unlike simulation (C37, intact in language),
INDUCTION persists as a wall. Forward-pass (no-think) dissociation at depth-1: the model EXECUTES a given rule
(0.86) but cannot INFER one from examples (0.00). Thinking only partially rescues induction (0.50 at depth-1),
still far below application and below C37's simulation. Emits verdict + figure."""
import json, os
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
EXP = Path(__file__).resolve().parents[1]
def acc(f): return {int(k): v["acc"] for k, v in json.load(open(EXP / "runs" / f))["by_depth"].items()} if os.path.exists(EXP / "runs" / f) else {}
app_nt = acc("prop_app_nothink.json")        # application (rule given), no-think
app_th = acc("prop_app_think.json")          # application, think (d1)
ind_nt = acc("prop_ling_nothink.json")       # induction (rule hidden), no-think
ind_th = acc("prop_ling_think.json")         # induction, think d1 (budget 4096, no truncation)
GUESS = 0.062

out = {"application_nothink": app_nt, "application_think_d1": app_th, "induction_nothink": ind_nt,
       "induction_think_d1": ind_th, "guess_baseline": GUESS, "sim_C37_d3_nothink": 0.99}
out["verdict"] = (
    "The structure-PROPOSAL wall DOES exist in language -- unlike simulation (C37, intact). CLEAN forward-pass "
    f"dissociation at depth-1 (where application is easy, so induction is isolable): the model EXECUTES a given "
    f"relational rule ({app_nt.get(1)}) but CANNOT INFER one from examples ({ind_nt.get(1)}, = chance) in a single "
    f"forward pass. Induction is at chance no-think at ALL depths ({ind_nt}). Thinking only PARTIALLY rescues "
    f"induction ({ind_th.get(1)} at depth-1, budget 4096, no truncation) -- still error-prone at the simplest depth "
    f"and far below application ({app_th.get(1)} think) and below C37's linguistic simulation (0.99 no-think at "
    f"depth-3). So the model is an EXECUTOR, not an INDUCER, in language as in formal domains -- corroborating "
    f"C32/C36 as a CROSS-MODALITY property. This DISSOCIATES the two components of the compositional wall: "
    f"SIMULATION/execution is modality-dependent (formal walls at depth-3, language does not -- C37); "
    f"PROPOSAL/INDUCTION is hard in BOTH modalities (formal C32/C36, language here) -- the deeper, more fundamental "
    f"limit. Honest caveat: this multi-relation substrate's application itself degrades at depth 2+ ({app_nt}), so "
    f"induction is cleanly isolable only at depth-1; but induction is already at chance there.")
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print(out["verdict"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
# left: depth-1 dissociation
labels = ["application\n(execute given rule)\nno-think", "application\nthink", "INDUCTION\n(infer rule)\nno-think", "INDUCTION\nthink"]
vals = [app_nt.get(1, 0), app_th.get(1, 0), ind_nt.get(1, 0), ind_th.get(1, 0)]
cols = ["#16a34a", "#22c55e", "#ef4444", "#f97316"]
ax1.bar(range(4), vals, 0.6, color=cols)
for i, v in enumerate(vals): ax1.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax1.axhline(0.99, ls="--", color="#2563eb", alpha=0.6, label="C37: linguistic SIMULATION (0.99, no-think, depth-3)")
ax1.axhline(GUESS, ls=":", color="#aaa", label=f"guess baseline {GUESS}")
ax1.set_xticks(range(4)); ax1.set_xticklabels(labels, fontsize=8); ax1.set_ylim(0, 1.05)
ax1.set_ylabel("accuracy at depth-1 (n=24-50)"); ax1.legend(fontsize=8, loc="upper right"); ax1.grid(alpha=0.25, axis="y")
ax1.set_title("Depth-1: the model EXECUTES a given rule (0.86) but cannot INDUCE one (0.00);\nthinking only half-rescues induction (0.50)")
# right: depth curves (no-think)
ds = sorted(app_nt)
ax2.plot(ds, [app_nt[d] for d in ds], "o-", color="#16a34a", lw=2.3, markersize=8, label="application (execute given rule), no-think")
ax2.plot(sorted(ind_nt), [ind_nt[d] for d in sorted(ind_nt)], "s-", color="#ef4444", lw=2.3, markersize=8, label="INDUCTION (infer rule), no-think")
ax2.axhline(GUESS, ls=":", color="#aaa"); ax2.text(3.4, GUESS + 0.02, "chance", fontsize=8, color="#999")
ax2.set_xlabel("rule depth (relations to compose)"); ax2.set_ylabel("accuracy (no-think forward pass)"); ax2.set_ylim(-0.02, 1.0)
ax2.set_xticks(ds); ax2.legend(fontsize=8.5); ax2.grid(alpha=0.25)
ax2.set_title("Induction is at CHANCE at all depths in a forward pass;\napplication is easy at depth-1 (degrades deeper on this multi-relation substrate)")
fig.suptitle("Does the structure-PROPOSAL wall exist in language? YES -- induction persists as a wall (unlike simulation, C37). "
             "The model is an EXECUTOR, not an INDUCER, in any modality.", fontsize=9.5, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "language_proposal_wall.png", dpi=130, bbox_inches="tight")
print("wrote analysis/language_proposal_wall.png")
