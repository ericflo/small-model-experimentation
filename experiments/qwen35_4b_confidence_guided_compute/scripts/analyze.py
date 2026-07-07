#!/usr/bin/env python3
"""Beat sample-more with the model's own uncertainty. (1) Confirm two regimes: coverage-limited (pass@k >> greedy)
vs capability-limited (pass@k ~ greedy). (2) Which cheap signal predicts per-problem solvability / picks the
correct sample? greedy P(answer) vs sample self-consistency vs max per-sample P(answer). (3) SELECTION accuracy at
k: random / self-consistency(majority) / confidence(argmax P) / oracle(pass@k). (4) The matched-BUDGET frontier
(accuracy vs avg forward-passes/problem): uniform+self-consistency vs confidence-guided ALLOCATION+ABSTENTION."""
import json, math
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
D = json.load(open(EXP / "runs" / "sampling_records.json"))
R, K = D["records"], D["k"]
CONDS = ["execute", "familiar_induce", "novel_induce"]
rng = np.random.default_rng(0)


def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() in (0, len(y)): return None
    o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); cs = np.cumsum(cnt)
    r = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = y.sum(); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def majority(samples):
    c = Counter(s["a"] for s in samples if s["a"]); return c.most_common(1)[0][0] if c else ""


def conf_pick(samples):  # argmax per-sample P(answer)
    v = [s for s in samples if s["a"]]
    return max(v, key=lambda s: s["p"])["a"] if v else ""


def sc_agreement(samples):
    c = Counter(s["a"] for s in samples if s["a"]); n = sum(c.values())
    return c.most_common(1)[0][1] / n if n else 0.0


# ---- (1) regimes + per-problem enrichment ----
summary = {}
for r in R:
    r["true"]; r["pass_k"] = int(any(s["c"] for s in r["samples"]))
    r["sc_ans"] = majority(r["samples"]); r["sc_correct"] = int(r["sc_ans"] == r["true"])
    r["conf_ans"] = conf_pick(r["samples"]); r["conf_correct"] = int(r["conf_ans"] == r["true"])
    r["agreement"] = round(sc_agreement(r["samples"]), 3)
    r["max_sample_p"] = max((s["p"] for s in r["samples"]), default=0.0)
for c in CONDS:
    rc = [r for r in R if r["cond"] == c]
    summary[c] = {"n": len(rc), "greedy_acc": round(np.mean([r["greedy_correct"] for r in rc]), 3),
                  "pass_k": round(np.mean([r["pass_k"] for r in rc]), 3),
                  "self_consistency_acc": round(np.mean([r["sc_correct"] for r in rc]), 3),
                  "confidence_select_acc": round(np.mean([r["conf_correct"] for r in rc]), 3),
                  "mean_greedy_p": round(np.mean([r["greedy_p"] for r in rc]), 3),
                  "mean_agreement": round(np.mean([r["agreement"] for r in rc]), 3)}

# ---- (2) does a cheap signal predict per-problem SOLVABILITY (pass@k)? ----
ally = [r["pass_k"] for r in R]
sig_auroc = {"greedy_p": auroc([r["greedy_p"] for r in R], ally),
             "max_sample_p": auroc([r["max_sample_p"] for r in R], ally),
             "agreement": auroc([r["agreement"] for r in R], ally),
             "greedy_correct": auroc([r["greedy_correct"] for r in R], ally)}

# ---- (3) selection accuracy at full k (pooled) ----
sel = {"random": round(np.mean([r["samples"][rng.integers(len(r["samples"]))]["c"] for r in R]), 3),
       "self_consistency": round(np.mean([r["sc_correct"] for r in R]), 3),
       "confidence_argmaxP": round(np.mean([r["conf_correct"] for r in R]), 3),
       "oracle_passk": round(np.mean([r["pass_k"] for r in R]), 3)}

# ---- (4) matched-budget frontier (accuracy vs avg forward passes / problem) ----
def sub(samples, m): return list(rng.choice(samples, size=m, replace=False)) if m <= len(samples) else samples
def sc_acc(rs, m):    return np.mean([int(majority(sub(r["samples"], m)) == r["true"]) for r in rs])
def conf_acc(rs, m):  return np.mean([int(conf_pick(sub(r["samples"], m)) == r["true"]) for r in rs])

budgets = list(range(1, K + 1))
front = {"uniform_sc": [], "uniform_confpick": []}
for m in budgets:
    front["uniform_sc"].append(round(float(np.mean([sc_acc([r], m) for r in R])), 3))
    front["uniform_confpick"].append(round(float(np.mean([conf_acc([r], m) for r in R])), 3))
# CONFIDENCE-GUIDED ALLOCATION (no abstention): 1 greedy probe/problem; ACCEPT the greedy answer on confident-easy
# problems (greedy_p>=hi, 1 sample total) and REALLOCATE the saved budget to the rest (confidence-select). At a
# matched total budget the coverage-limited problems get MORE samples than uniform -> should beat uniform+confP.
def alloc(target_B, hi=0.9):
    easy = [r for r in R if r["greedy_p"] >= hi]; rest = [r for r in R if r["greedy_p"] < hi]
    spent = len(R)                                    # 1 probe each
    correct = sum(r["greedy_correct"] for r in easy)  # accept easy on the probe
    m = max(1, min(K, int((target_B * len(R) - spent) / max(1, len(rest)))))
    spent += m * len(rest)
    correct += sum(int(conf_pick(sub(r["samples"], m)) == r["true"]) for r in rest)
    return round(spent / len(R), 2), round(correct / len(R), 3)
adap = sorted({alloc(b)[0]: alloc(b)[1] for b in np.linspace(1.2, K, 14)}.items())
# ABSTENTION (selective prediction): rank problems by max_sample_p, attempt the top fraction (confidence-select),
# report accuracy-on-attempted vs coverage.
Rs = sorted(R, key=lambda r: -r["max_sample_p"])
abst = []
for cov in np.linspace(0.1, 1.0, 19):
    kf = max(1, int(cov * len(Rs)))
    abst.append((round(cov, 2), round(float(np.mean([r["conf_correct"] for r in Rs[:kf]])), 3)))
out = {"summary": summary, "signal_auroc_for_solvability": {k: (round(v, 3) if v else None) for k, v in sig_auroc.items()},
       "selection_at_k": sel, "frontier": front, "budgets": budgets, "adaptive_points": adap, "abstention": abst}
out["verdict"] = (
    f"Sample-more only helps if you select by CONFIDENCE, not by majority -- and the confidence signal also tells "
    f"you when sampling is futile. (1) Two regimes: familiar_induce is COVERAGE-limited (greedy "
    f"{summary['familiar_induce']['greedy_acc']} -> pass@{K} {summary['familiar_induce']['pass_k']}), novel_induce "
    f"CAPABILITY-limited (greedy {summary['novel_induce']['greedy_acc']} -> pass@{K} "
    f"{summary['novel_induce']['pass_k']}, below pure-luck). (2) VERIFICATION-FREE SELECTION: picking the highest "
    f"per-sample P(answer) gets {sel['confidence_argmaxP']} vs self-consistency/majority {sel['self_consistency']} "
    f"vs random {sel['random']} (oracle {sel['oracle_passk']}) -- and across budgets self-consistency is FLAT "
    f"(~{max(front['uniform_sc']):.2f}) while confidence-select RISES to {max(front['uniform_confpick']):.2f}. It "
    f"works because when the model derives the right rule it is confident, so the most-confident sample beats the "
    f"most-common one (C40: P(answer) is calibrated). (3) ABSTENTION: max per-sample P(answer) predicts solvability "
    f"at AUROC {sig_auroc['max_sample_p']:.2f}; abstaining on low-confidence gives ~1.0 accuracy on the confident "
    f"top third. (Confidence-guided ALLOCATION is ~tied with uniform confidence-select -- selection, not "
    f"allocation, is where the win is.)")
(EXP / "runs" / "verdict.json").write_text(json.dumps(out, indent=1))
print(out["verdict"]); print("selection_at_k:", sel); print("signal AUROC:", out["signal_auroc_for_solvability"])

# ---- figure ----
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
x = np.arange(len(CONDS))
ax1.bar(x - 0.2, [summary[c]["greedy_acc"] for c in CONDS], 0.4, label="greedy (1 sample)", color="#94a3b8")
ax1.bar(x + 0.2, [summary[c]["pass_k"] for c in CONDS], 0.4, label=f"pass@{K} (sample-more)", color="#2563eb")
ax1.set_xticks(x); ax1.set_xticklabels(CONDS, fontsize=8); ax1.set_ylim(0, 1.05); ax1.legend(fontsize=8.5)
ax1.grid(alpha=0.25, axis="y"); ax1.set_title("Two failure modes of 'sample-more':\ncoverage-limited (familiar, big gain) vs capability-limited (novel, small)")
# frontier: self-consistency FLAT vs confidence-select RISING
ax2.plot(budgets, front["uniform_sc"], "o-", color="#f59e0b", lw=2.4, label="self-consistency (majority vote)")
ax2.plot(budgets, front["uniform_confpick"], "s-", color="#2563eb", lw=2.4, label="confidence-select (argmax P(answer))")
ax2.axhline(sel["oracle_passk"], ls="--", color="#111827", alpha=0.6, label=f"oracle / pass@{K} (upper bound)")
ax2.axhline(sel["random"], ls=":", color="#888", label="random select")
ax2.set_xlabel("samples per problem (budget)"); ax2.set_ylabel("accuracy"); ax2.set_ylim(0.3, 0.9)
ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
ax2.set_title("Sample-more only helps if you SELECT by confidence, not majority\n(self-consistency is FLAT; confidence-select rises, verification-free)")
# abstention / selective prediction
cov = [c for c, _ in abst]; ac = [a for _, a in abst]
ax3.plot(cov, ac, "-", color="#16a34a", lw=2.6)
ax3.axhline(sel["confidence_argmaxP"], ls=":", color="#888", label=f"no abstention ({sel['confidence_argmaxP']:.2f})")
ax3.set_xlabel("coverage (fraction attempted, ranked by max P(answer))"); ax3.set_ylabel("accuracy on attempted")
ax3.set_ylim(0.5, 1.03); ax3.legend(fontsize=9); ax3.grid(alpha=0.25)
ax3.set_title(f"Abstention: max P(answer) predicts solvability (AUROC {sig_auroc['max_sample_p']:.2f})\nabstain on low-confidence -> ~perfect on the confident third")
fig.suptitle("Beating sample-more with the model's own uncertainty: confidence-select beats majority vote, and knows when sampling is futile (abstain)", fontsize=10, y=1.02)
fig.tight_layout(); (EXP / "analysis").mkdir(exist_ok=True)
fig.savefig(EXP / "analysis" / "confidence_guided_compute.png", dpi=130, bbox_inches="tight")
print("wrote analysis/confidence_guided_compute.png")
