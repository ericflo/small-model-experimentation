#!/usr/bin/env python3
"""C45: is GENERAL induction-via-reasoning installable? Train a general enumerate-and-verify CoT on families
{a=1,3,9}; test held-out family a=7. Held-out a=7 (0.905) is AS HIGH as in-family (0.875-0.955) -> the general
hypothesize-verify procedure TRANSFERS to a novel rule family. Resolves C44's shift-specificity: with a general
procedure + multi-family training, induction generalizes."""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
M = {"a1_infam": 0.955, "a3_infam": 0.930, "a9_infam": 0.875, "a7_heldout": 0.905,
     "c44_shift_specific_OOF_affine": 0.13, "chance": 0.10}
M["verdict"] = (
    "GENERAL induction-via-reasoning IS installable. Training a GENERAL enumerate-and-verify CoT procedure "
    "(try each candidate multiplier, keep the one that fits the examples, apply) on families {a=1,3,9} and testing "
    f"the HELD-OUT family a=7 (never seen as the answer): held-out a=7 induction = {M['a7_heldout']}, AS HIGH as "
    f"in-family (a=1 {M['a1_infam']}, a=3 {M['a3_infam']}, a=9 {M['a9_infam']}). The model GENERALIZES to a novel "
    "rule family -- it learned the general hypothesize-verify-apply PROCEDURE, not just the trained families. This "
    f"RESOLVES C44's shift-specificity (shift-only CoT transferred to OOF affine at only {M['c44_shift_specific_OOF_affine']}): "
    "with a GENERAL procedure and MULTI-family training, induction generalizes to unseen rules. Combined with C44 "
    "(induction is a serial-compute limit: reasoning 1.00 vs forward-pass 0.01), the full picture is that the fixed "
    "4B CAN be taught GENERAL induction -- but only as a SERIAL reasoning procedure (it lives in the chain-of-thought "
    "tokens, not the weights). Give it a general hypothesize-and-verify strategy via SFT and it induces novel rules. "
    "CAVEAT: the held-out multiplier a=7's arithmetic (7*p) was seen in training as a REJECTED candidate, so the "
    "arithmetic is in-distribution; what generalizes is ACCEPTING a=7 as the answer via verify -- i.e. the "
    "general induction LOGIC, not novel arithmetic. Single seed.")
(EXP / "runs" / "verdict_general.json").write_text(json.dumps(M, indent=1))
print(M["verdict"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
labs = ["a=1\n(trained)", "a=3\n(trained)", "a=9\n(trained)", "a=7\nHELD-OUT"]
vals = [M["a1_infam"], M["a3_infam"], M["a9_infam"], M["a7_heldout"]]
cols = ["#94a3b8", "#94a3b8", "#94a3b8", "#16a34a"]
ax1.bar(range(4), vals, color=cols)
for i, v in enumerate(vals): ax1.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax1.axhline(M["chance"], ls=":", color="#888", label="chance (0.1)")
ax1.set_xticks(range(4)); ax1.set_xticklabels(labs, fontsize=9); ax1.set_ylim(0, 1.05)
ax1.set_ylabel("induction accuracy (via reasoning)"); ax1.legend(fontsize=9)
ax1.set_title("General induction TRANSFERS to a held-out rule family:\na=7 (never trained) is as accurate as the trained families")
# contrast with C44 shift-specific
ax2.bar([0, 1], [M["c44_shift_specific_OOF_affine"], M["a7_heldout"]], color=["#ef4444", "#16a34a"])
ax2.text(0, M["c44_shift_specific_OOF_affine"] + 0.02, f"{M['c44_shift_specific_OOF_affine']:.2f}", ha="center", fontsize=11, fontweight="bold")
ax2.text(1, M["a7_heldout"] + 0.02, f"{M['a7_heldout']:.2f}", ha="center", fontsize=11, fontweight="bold")
ax2.axhline(M["chance"], ls=":", color="#888")
ax2.set_xticks([0, 1]); ax2.set_xticklabels(["C44: shift-SPECIFIC CoT\n-> out-of-family (affine)", "C45: GENERAL verify CoT\n+ multi-family -> held-out a=7"], fontsize=8.5)
ax2.set_ylim(0, 1.05); ax2.set_ylabel("out-of-family / held-out induction accuracy")
ax2.set_title("What makes induction GENERALIZE: a general hypothesize-verify\nprocedure + multi-family training (not a single hand-coded rule)")
fig.suptitle("Is GENERAL induction-via-reasoning installable? YES -- a general hypothesize-and-verify procedure, trained multi-family, transfers to novel rule families", fontsize=9.5, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "general_induction.png", dpi=130, bbox_inches="tight")
print("wrote analysis/general_induction.png")
