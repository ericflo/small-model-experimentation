#!/usr/bin/env python3
"""Generator-verifier gap metrics: generation vs verification skill, the selection gap, thinking asymmetry."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

EXP = Path(__file__).resolve().parents[1]


def main():
    recs = [json.loads(l) for l in (EXP / "data" / "records.jsonl").read_text().splitlines() if l.strip()]
    labs = {(r["task_id"], r["sample"]): r["full_pass"]
            for r in (json.loads(l) for l in (EXP / "data" / "labels.jsonl").read_text().splitlines() if l.strip())}
    for r in recs:
        r["full_pass"] = bool(labs[(r["task_id"], r["sample"])])

    y = np.array([int(r["full_pass"]) for r in recs])
    pa_nt = np.array([r["pa_nothink"] for r in recs])
    pa_th = np.array([r["pa_think"] for r in recs])

    # generation
    by_task = defaultdict(list)
    for r in recs:
        by_task[r["task_id"]].append(r)
    pass1 = float(y.mean())
    passk = float(np.mean([1.0 if any(x["full_pass"] for x in v) else 0.0 for v in by_task.values()]))

    def vstats(pa):
        auroc = roc_auc_score(y, pa) if len(set(y)) == 2 else float("nan")
        bacc = balanced_accuracy_score(y, (pa > 0.5).astype(int))
        acc = float(((pa > 0.5).astype(int) == y).mean())
        yes = float((pa > 0.5).mean())
        return {"auroc": round(auroc, 3), "balanced_acc": round(bacc, 3), "raw_acc": round(acc, 3), "say_A_rate": round(yes, 3)}

    verify_nt, verify_th = vstats(pa_nt), vstats(pa_th)

    # verifier-selected: per task pick own candidate with max P(A) -> its pass
    def selected(pa_key):
        sel = []
        for v in by_task.values():
            best = max(v, key=lambda r: r[pa_key])
            sel.append(1 if best["full_pass"] else 0)
        return float(np.mean(sel))
    sel_nt, sel_th = selected("pa_nothink"), selected("pa_think")
    # random-selection baseline = pass@1; oracle = passk
    # fraction of the pass@1->oracle gap the verifier closes:
    def closed(sel):
        return (sel - pass1) / (passk - pass1) if passk > pass1 else float("nan")

    # foreign control (should be judged incorrect -> low P(A))
    fpa_nt = np.array([r["foreign_pa_nothink"] for r in recs])
    fpa_th = np.array([r["foreign_pa_think"] for r in recs])
    foreign = {"mean_pa_nothink": round(float(fpa_nt.mean()), 3), "reject_rate_nothink": round(float((fpa_nt < 0.5).mean()), 3),
               "mean_pa_think": round(float(fpa_th.mean()), 3), "reject_rate_think": round(float((fpa_th < 0.5).mean()), 3)}

    out = {
        "n_tasks": len(by_task), "n_candidates": len(recs),
        "generation": {"pass@1": round(pass1, 3), f"pass@{max(len(v) for v in by_task.values())}_oracle": round(passk, 3)},
        "verification_intrinsic": {"no_think": verify_nt, "think": verify_th},
        "verifier_selected": {
            "pass@1_random": round(pass1, 3), "no_think_selected": round(sel_nt, 3),
            "think_selected": round(sel_th, 3), "oracle": round(passk, 3),
            "gap_closed_no_think": round(closed(sel_nt), 3), "gap_closed_think": round(closed(sel_th), 3)},
        "foreign_control": foreign,
        "forced_think_frac": round(float(np.mean([r["forced_think"] for r in recs])), 3),
    }
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "summary.json").write_text(json.dumps(out, indent=2))

    print("\n=== GENERATOR-VERIFIER GAP ===")
    print(f"GENERATION:   pass@1 {pass1:.3f}   oracle pass@k {passk:.3f}")
    print(f"VERIFICATION (intrinsic, discriminate own good vs bad candidates):")
    print(f"  no_think: balanced_acc {verify_nt['balanced_acc']:.3f}  AUROC {verify_nt['auroc']:.3f}  (say-A {verify_nt['say_A_rate']:.2f})")
    print(f"  think:    balanced_acc {verify_th['balanced_acc']:.3f}  AUROC {verify_th['auroc']:.3f}  (say-A {verify_th['say_A_rate']:.2f})")
    print(f"VERIFIER-SELECTED best-of-{max(len(v) for v in by_task.values())}:")
    print(f"  pass@1(random) {pass1:.3f} -> no_think-selected {sel_nt:.3f} -> think-selected {sel_th:.3f} -> oracle {passk:.3f}")
    print(f"  gap closed: no_think {closed(sel_nt):+.2f}  think {closed(sel_th):+.2f}")
    print(f"FOREIGN control (should reject): reject_rate no_think {foreign['reject_rate_nothink']:.2f}  think {foreign['reject_rate_think']:.2f}")

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    ax1.bar(["pass@1\n(generate)", "verify-sel\nno_think", "verify-sel\nthink", "oracle\npass@k"],
            [pass1, sel_nt, sel_th, passk], color=["#264653", "#e9c46a", "#2a9d8f", "#8d99ae"], edgecolor="k", lw=0.5)
    for i, v in enumerate([pass1, sel_nt, sel_th, passk]):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax1.set_ylim(0, 1); ax1.set_ylabel("MBPP accuracy"); ax1.set_title("Can the model's own verifier close pass@1 -> oracle?")
    ax2.bar(["no_think", "think"], [verify_nt["balanced_acc"], verify_th["balanced_acc"]], color=["#e9c46a", "#2a9d8f"], edgecolor="k", lw=0.5, label="balanced acc")
    ax2.axhline(0.5, color="gray", ls=":", lw=0.8)
    for i, v in enumerate([verify_nt["balanced_acc"], verify_th["balanced_acc"]]):
        ax2.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, 1); ax2.set_ylabel("verification balanced accuracy"); ax2.set_title("Intrinsic verification skill (own good vs bad)")
    fig.tight_layout(); fig.savefig(EXP / "analysis" / "gen_verify.png", dpi=130)
    print("\nwrote runs/summary.json, analysis/gen_verify.png")


if __name__ == "__main__":
    main()
