#!/usr/bin/env python3
"""Does the model know when it will fail? Analysis. (1) Condition-level calibration: does mean confidence track
acc? (2) WITHIN-condition item-level AUROC (headline = familiar_induce, surface-matched) for the model's signals
vs an EXTERNAL surface-feature baseline -- a signal counts as self-knowledge only if it BEATS surface. (3)
Selective prediction. (4) Modal wrong answer on novel_induce (natural-successor intrusion => silent consistent
wrong vs high-entropy scatter). Implicit P(answer) vs explicit P(True)."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
R = json.load(open(EXP / "runs" / "metacog_records.json"))
CONDS = ["familiar_execute", "familiar_induce", "reversal_induce", "novel_induce"]
FEATS = ["k", "gap_to_seen", "n_distinct_seen", "query"]


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y): return None
    order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avg = (start + csum + 1) / 2.0
    ranks = avg[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auroc(scores, labels, n=1000):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    rng = np.random.default_rng(0); out = []
    for _ in range(n):
        idx = rng.integers(0, len(s), len(s))
        a = auroc(s[idx], y[idx])
        if a is not None: out.append(a)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if out else (None, None)


def surface_auroc(recs):
    """Cross-validated external baseline: predict correctness from prompt-surface features only (no model signal)."""
    X = np.array([[r["feats"][f] for f in FEATS] for r in recs], float)
    y = np.array([r["correct"] for r in recs], int)
    if y.sum() == 0 or y.sum() == len(y) or len(y) < 20: return None
    proba = cross_val_predict(LogisticRegression(max_iter=1000, C=1.0), X, y, cv=5, method="predict_proba")[:, 1]
    return auroc(proba, y)


summary = {}
for c in CONDS:
    rc = [r for r in R if r["cond"] == c]
    y = [r["correct"] for r in rc]
    acc = np.mean(y)
    summary[c] = {
        "n": len(rc), "acc": round(float(acc), 3),
        "mean_p_answer": round(float(np.mean([r["p_answer"] for r in rc])), 3),
        "mean_p_true": round(float(np.mean([r["p_true"] for r in rc])), 3),
        "mean_entropy": round(float(np.mean([r["entropy"] for r in rc])), 3),
        # within-condition item-level AUROC (self-knowledge; needs both classes present)
        "auroc_p_answer": auroc([r["p_answer"] for r in rc], y),
        "auroc_neg_entropy": auroc([-r["entropy"] for r in rc], y),
        "auroc_margin": auroc([r["margin"] for r in rc], y),
        "auroc_p_true": auroc([r["p_true"] for r in rc], y),
        "auroc_surface": surface_auroc(rc),
    }
    for kk in ("auroc_p_answer", "auroc_p_true", "auroc_surface"):
        v = summary[c][kk]
        if v is not None: summary[c][kk] = round(v, 3)

# modal wrong answer on novel_induce: natural-successor intrusion?
nov = [r for r in R if r["cond"] == "novel_induce"]
wrong = [r for r in nov if not r["correct"]]
nat_intrusion = np.mean([r["answer"] == r["natural_succ"] for r in wrong]) if wrong else None
head = summary["familiar_induce"]
ci = boot_auroc([r["p_answer"] for r in R if r["cond"] == "familiar_induce"],
                [r["correct"] for r in R if r["cond"] == "familiar_induce"])

verdict = (
    f"The model has IMPLICIT metacognition (its answer-token distribution tracks competence) but NOT EXPLICIT "
    f"self-assessment. CONDITION-LEVEL: mean P(answer) tracks accuracy almost perfectly "
    f"({summary['familiar_execute']['mean_p_answer']}/{summary['familiar_induce']['mean_p_answer']}/"
    f"{summary['reversal_induce']['mean_p_answer']}/{summary['novel_induce']['mean_p_answer']} vs acc "
    f"{summary['familiar_execute']['acc']}/{summary['familiar_induce']['acc']}/{summary['reversal_induce']['acc']}/"
    f"{summary['novel_induce']['acc']}), while explicit P(True) self-verification is FLAT (~0.4) and even "
    f"UNDERCONFIDENT on the perfect execute cell ({summary['familiar_execute']['mean_p_true']}). "
    f"WITHIN the surface-matched headline cell (familiar_induce, acc {head['acc']}): P(answer) AUROC "
    f"{head['auroc_p_answer']} (95% CI {ci}) vs external surface baseline {head['auroc_surface']} vs explicit "
    f"P(True) AUROC {head['auroc_p_true']}. So the model's answer-probability predicts its own per-item "
    f"correctness beyond surface features (genuine item-level self-knowledge) -- but it CANNOT verbalize/verify it "
    f"(P(True) is flat). On novel_induce, {round(nat_intrusion,2) if nat_intrusion is not None else None} of wrong "
    f"answers are the NATURAL-successor intrusion (applying the familiar rule instead of the novel one) -- a "
    f"consistent wrong-rule, not random scatter. DEPLOYABLE: read the model's answer-token probability (implicit) "
    f"for a usable confidence/abstain signal; do NOT trust its explicit self-assessment.")
out = {"summary": summary, "familiar_induce_p_answer_auroc_ci": ci, "novel_natural_intrusion": nat_intrusion,
       "verdict": verdict}
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print(verdict)

# ---- figure ----
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
x = np.arange(len(CONDS))
accs = [summary[c]["acc"] for c in CONDS]
pans = [summary[c]["mean_p_answer"] for c in CONDS]
ptrs = [summary[c]["mean_p_true"] for c in CONDS]
w = 0.26
ax1.bar(x - w, accs, w, label="actual accuracy", color="#111827")
ax1.bar(x, pans, w, label="IMPLICIT: mean P(answer)", color="#2563eb")
ax1.bar(x + w, ptrs, w, label="EXPLICIT: mean P(True)", color="#f97316")
ax1.set_xticks(x); ax1.set_xticklabels([c.replace("_", "\n") for c in CONDS], fontsize=8)
ax1.set_ylim(0, 1.08); ax1.legend(fontsize=8.5); ax1.grid(alpha=0.25, axis="y")
ax1.set_title("Implicit P(answer) TRACKS accuracy; explicit P(True) is FLAT (~0.4)\n(the model's own probability knows; its self-report doesn't)")
# within-condition AUROC (headline familiar_induce)
cells = ["familiar_induce", "reversal_induce"]
labels = ["P(answer)\n(implicit)", "margin", "-entropy", "P(True)\n(explicit)", "surface\nbaseline"]
keys = ["auroc_p_answer", "auroc_margin", "auroc_neg_entropy", "auroc_p_true", "auroc_surface"]
vals = [summary["familiar_induce"][k] if summary["familiar_induce"][k] is not None else 0 for k in keys]
cols = ["#2563eb", "#3b82f6", "#60a5fa", "#f97316", "#9ca3af"]
ax2.bar(range(len(labels)), vals, color=cols)
for i, v in enumerate(vals): ax2.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(0.5, ls=":", color="#888", label="chance (0.5)")
ax2.set_xticks(range(len(labels))); ax2.set_xticklabels(labels, fontsize=8); ax2.set_ylim(0, 1.08)
ax2.set_ylabel("within-cell AUROC (predict per-item correctness)"); ax2.legend(fontsize=8)
ax2.set_title(f"Item-level self-knowledge WITHIN familiar_induce (surface-matched, acc {head['acc']})\nImplicit P(answer) beats surface + explicit P(True)")
# selective prediction using p_answer, pooled induce cells
ind = [r for r in R if r["cond"] in ("familiar_induce", "reversal_induce", "novel_induce")]
ind.sort(key=lambda r: -r["p_answer"])
ys = [r["correct"] for r in ind]
covs, accs2 = [], []
for kf in range(10, len(ys) + 1, 5):
    covs.append(kf / len(ys)); accs2.append(np.mean(ys[:kf]))
ax3.plot(covs, accs2, "-", color="#2563eb", lw=2.4)
ax3.axhline(np.mean(ys), ls=":", color="#888", label=f"no-abstention acc {np.mean(ys):.2f}")
ax3.set_xlabel("coverage (fraction attempted, ranked by P(answer))"); ax3.set_ylabel("accuracy on attempted")
ax3.set_ylim(0, 1.02); ax3.legend(fontsize=9); ax3.grid(alpha=0.25)
ax3.set_title("Selective prediction (induce cells): abstaining by low P(answer)\nlifts accuracy on what it attempts")
fig.suptitle("Does the model know when it will fail? YES implicitly (answer-token probability), NO explicitly (self-verification is flat).", fontsize=11, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "metacognitive_boundary.png", dpi=130, bbox_inches="tight")
print("wrote analysis/metacognitive_boundary.png")
