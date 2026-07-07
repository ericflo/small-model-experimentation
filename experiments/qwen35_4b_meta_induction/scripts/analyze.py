#!/usr/bin/env python3
"""Can SFT install the SKILL of induction? Consolidate base vs answer-only SFT (4k, 8k shift episodes) on
in-family (held-out shift) and out-of-family (affine) induction, against the base EXECUTE ceilings. Verdict:
SFT PARTIALLY lifts the induction wall (data-limited) but does not cleanly install the skill -- procedure-specific
(weak OOF transfer) + catastrophic forgetting of execution."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
R = {  # consolidated measured accuracies
    "shift_induce": {"base": 0.087, "sft4k": 0.350, "sft8k": 0.400}, "shift_execute_ceiling": 0.72,
    "affine_induce": {"base": 0.213, "sft4k": 0.267, "sft8k": 0.297}, "affine_execute_ceiling": 0.457,
    "shift_execute_after_sft4k": 0.093, "chance": 0.10,
    "sft8k_shift_top_answer_frac": round(59 / 300, 2), "sft4k_shift_top_answer_frac": round(111 / 300, 2),
}
R["verdict"] = (
    "SFT PARTIALLY lifts the induction wall but does NOT cleanly install the skill. In-family (held-out shift): "
    f"answer-only SFT moves induction from chance ({R['shift_induce']['base']}) to {R['shift_induce']['sft4k']} "
    f"(4k) to {R['shift_induce']['sft8k']} (8k) -- DATA-LIMITED and improving, but plateauing well BELOW the base "
    f"EXECUTE ceiling ({R['shift_execute_ceiling']}), so the skill is only partially installed. OUT-OF-FAMILY "
    f"(affine, a different structure): induction barely moves ({R['affine_induce']['base']} -> "
    f"{R['affine_induce']['sft8k']}), far below in-family -- the model learned a SHIFT-SPECIFIC procedure, not "
    f"general induction. Two costs: (1) CATASTROPHIC FORGETTING -- answer-only SFT crashed the model's EXECUTE "
    f"ability from {R['shift_execute_ceiling']} to {R['shift_execute_after_sft4k']}; (2) a default-fallback BIAS "
    f"(over-outputs one digit) that shrinks with data ({R['sft4k_shift_top_answer_frac']} at 4k -> "
    f"{R['sft8k_shift_top_answer_frac']} at 8k). CONCLUSION: the induction wall is neither a hard architectural "
    f"bound (SFT lifts it ~4.6x above chance, scaling with data) nor cleanly liftable (partial, procedure-specific, "
    f"forgets execution) -- consistent with the arc's 'executor, not inducer': trained to induce, the model learns "
    f"a specific procedure, not the general skill, and trades away its executor competence.")
(EXP / "runs" / "verdict.json").write_text(json.dumps(R, indent=1))
print(R["verdict"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
stages = ["base", "sft4k", "sft8k"]
x = range(len(stages))
ax1.plot(x, [R["shift_induce"][s] for s in stages], "o-", color="#2563eb", lw=2.4, markersize=10, label="IN-FAMILY: shift induction")
ax1.plot(x, [R["affine_induce"][s] for s in stages], "s-", color="#f97316", lw=2.4, markersize=10, label="OUT-OF-FAMILY: affine induction")
ax1.axhline(R["shift_execute_ceiling"], ls="--", color="#2563eb", alpha=0.5, label="shift EXECUTE ceiling (0.72)")
ax1.axhline(R["affine_execute_ceiling"], ls="--", color="#f97316", alpha=0.5, label="affine EXECUTE ceiling (0.46)")
ax1.axhline(R["chance"], ls=":", color="#888", label="chance (0.1)")
for s in stages:
    ax1.text(stages.index(s), R["shift_induce"][s] + 0.02, f"{R['shift_induce'][s]:.2f}", ha="center", fontsize=9, fontweight="bold", color="#2563eb")
ax1.set_xticks(list(x)); ax1.set_xticklabels(["base", "SFT 4k", "SFT 8k"]); ax1.set_ylim(0, 0.8)
ax1.set_ylabel("induction accuracy (held-out rules)"); ax1.legend(fontsize=8, loc="upper left")
ax1.set_title("SFT PARTIALLY lifts induction (data-limited) but plateaus below\nthe EXECUTE ceiling; OOF (affine) barely moves = shift-specific")
# forgetting + the cost
labels = ["shift EXECUTE\n(base)", "shift EXECUTE\n(after answer-only SFT)", "shift INDUCE\n(base)", "shift INDUCE\n(SFT 8k)"]
vals = [R["shift_execute_ceiling"], R["shift_execute_after_sft4k"], R["shift_induce"]["base"], R["shift_induce"]["sft8k"]]
cols = ["#16a34a", "#dc2626", "#94a3b8", "#2563eb"]
ax2.bar(range(4), vals, color=cols)
for i, v in enumerate(vals): ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8); ax2.set_ylim(0, 0.85)
ax2.set_ylabel("accuracy")
ax2.set_title("The cost: answer-only SFT installs partial induction (+0.31)\nbut CATASTROPHICALLY FORGETS execution (0.72 -> 0.09)")
fig.suptitle("Can SFT install the skill of induction? Partially -- the wall is neither a hard bound nor cleanly liftable (procedure-specific, forgets execution)", fontsize=10, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "meta_induction.png", dpi=130, bbox_inches="tight")
print("wrote analysis/meta_induction.png")
