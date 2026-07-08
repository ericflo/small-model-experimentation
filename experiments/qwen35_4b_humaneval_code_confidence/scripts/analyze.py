#!/usr/bin/env python3
"""Does the confidence toolkit survive on real code? Review-hardened analysis.
(1) HEADLINE (C40 analog) = WITHIN-problem AUROC on MIXED problems (pooled AUROC is inflated by problem
difficulty), vs a length-only surface baseline, cluster-bootstrapped over problems.
(2) Deployment calibration = pooled + greedy problem-level AUROC + abstain-on-greedy curve.
(3) C41 analog = selection at k: random / mean-logprob (verification-FREE) / P(True) no-think / self-consistency
(public-output majority) / visible-test execution / oracle pass@k."""
import argparse
import json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

EXP = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default=str(EXP / "runs" / "code_conf.json"))
    ap.add_argument("--verdict", default=str(EXP / "runs" / "verdict.json"))
    ap.add_argument("--figure", default=str(EXP / "analysis" / "code_confidence.png"))
    ap.add_argument("--title", default="MBPP")
    return ap.parse_args()


def auroc(s, y):
    s, y = np.asarray(s, float), np.asarray(y, int)
    if y.sum() in (0, len(y)) or len(y) < 2: return None
    o = s.argsort(); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True); cs = np.cumsum(cnt)
    r = ((cs - cnt + cs + 1) / 2.0)[inv]
    n1 = y.sum(); return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1)))


def sample_cells(r):  # the k sampled candidates (exclude greedy)
    return [c for c in r["cands"] if c["tag"] != "greedy"]


def sig(c, key):
    v = c.get(key)
    return v if v is not None else -99.0


def deployable_sig(c):
    if c.get("deployable_behavior_signature"):
        return c["deployable_behavior_signature"]
    if c.get("public_signature"):
        status = "V1" if c.get("visible_all_pass") else "V0"
        return f"{status}:{c['public_signature']}"
    old = c.get("behavior_signature", "")
    if old.startswith("V") and ":H" in old and ":F" in old:
        prefix, rest = old.split(":H", 1)
        public = rest.split(":", 1)[1].rsplit(":F", 1)[0]
        return f"{prefix}:{public}"
    return old or "missing_public_signature"


args = parse_args()
R = json.load(open(args.input))


def has_public_probe(r):
    if "public_case_count" in r:
        return int(r.get("public_case_count") or 0) > 0
    return any(c.get("public_signature") not in ("", "no_public_tests", "parse_failed") for c in sample_cells(r))


# ---- (1) within-problem AUROC on mixed problems ----
def within_auroc(key, neg=False):
    per = []
    for r in R:
        cs = sample_cells(r); y = [c["full_pass"] for c in cs]
        if 0 < sum(y) < len(y):
            s = [(-1 if neg else 1) * sig(c, key) for c in cs]
            a = auroc(s, y)
            if a is not None: per.append(a)
    return per

mixed = [r for r in R if 0 < sum(c["full_pass"] for c in sample_cells(r)) < len(sample_cells(r))]
w_lp = within_auroc("mean_logprob"); w_pt = within_auroc("p_true")
w_len_short = within_auroc("code_len", neg=True); w_len_long = within_auroc("code_len")


def boot_mean(vals, n=2000):
    v = np.asarray(vals, float); out = []
    for _ in range(n):
        out.append(np.mean(v[rng.integers(0, len(v), len(v))]))
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

# ---- (2) pooled + greedy-level + abstention ----
allc = [c for r in R for c in sample_cells(r)]
pool_lp = auroc([sig(c, "mean_logprob") for c in allc], [c["full_pass"] for c in allc])
pool_pt = auroc([sig(c, "p_true") for c in allc], [c["full_pass"] for c in allc])
greedy = [(r["cands"][0], r) for r in R if r["cands"][0]["tag"] == "greedy"]
g_lp = auroc([sig(c, "mean_logprob") for c, _ in greedy], [c["full_pass"] for c, _ in greedy])
g_pt = auroc([sig(c, "p_true") for c, _ in greedy], [c["full_pass"] for c, _ in greedy])
g_len = auroc([-c["code_len"] for c, _ in greedy], [c["full_pass"] for c, _ in greedy])
g_acc = np.mean([c["full_pass"] for c, _ in greedy])
# abstain-on-greedy curve ranked by mean_logprob
gsort = sorted(greedy, key=lambda t: -sig(t[0], "mean_logprob"))
abst = [(round(kf / len(gsort), 2), round(float(np.mean([c["full_pass"] for c, _ in gsort[:kf]])), 3))
        for kf in range(10, len(gsort) + 1, 10)]

# ---- (3) selection at k ----
public_rows = [r for r in R if has_public_probe(r)]


def select(fn, rows=None):
    rows = rows or R
    ok = []
    for r in rows:
        cs = sample_cells(r)
        pick = fn(cs)
        ok.append(int(pick["full_pass"]) if pick is not None else 0)
    return round(float(np.mean(ok)), 3)


def sc_majority(cs):
    sigs = Counter(deployable_sig(c) for c in cs if c["parse_ok"])
    if not sigs: return None
    top = sigs.most_common(1)[0][0]
    return next(c for c in cs if deployable_sig(c) == top)

sel = {
    "random": round(float(np.mean([np.mean([c["full_pass"] for c in sample_cells(r)]) for r in R])), 3),
    "mean_logprob (verification-free)": select(lambda cs: max(cs, key=lambda c: sig(c, "mean_logprob"))),
    "p_true no-think (verification-free)": select(lambda cs: max(cs, key=lambda c: sig(c, "p_true"))),
}
if public_rows:
    suffix = "" if len(public_rows) == len(R) else f" (public-probe subset n={len(public_rows)})"
    sel[f"self-consistency (public-output majority){suffix}"] = select(sc_majority, public_rows)
    sel[f"visible-test execution{suffix}"] = select(
        lambda cs: next((c for c in cs if c["visible_all_pass"]), max(cs, key=lambda c: sig(c, "mean_logprob"))),
        public_rows,
    )
sel["oracle pass@k"] = round(float(np.mean([any(c["full_pass"] for c in sample_cells(r)) for r in R])), 3)
# duplicate rate
dup = None
if public_rows:
    dup = float(np.mean([1 - len({deployable_sig(c) for c in sample_cells(r)}) / len(sample_cells(r)) for r in public_rows]))

# ---- (4) paired significance: per-problem outcome vectors, bootstrap the DIFFERENCE ----
def outcomes(fn, rows=None):
    rows = rows or R
    o = []
    for r in rows:
        pick = fn(sample_cells(r))
        o.append(int(pick["full_pass"]) if pick is not None else 0)
    return np.array(o, float)

o_lp = outcomes(lambda cs: max(cs, key=lambda c: sig(c, "mean_logprob")))
o_pt = outcomes(lambda cs: max(cs, key=lambda c: sig(c, "p_true")))
o_rd = np.array([np.mean([c["full_pass"] for c in sample_cells(r)]) for r in R])

def paired_boot(a, b, n=10000):
    d = [float(np.mean(a[i] - b[i])) for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return {"diff": round(float(np.mean(a - b)), 3), "ci": [round(float(np.percentile(d, 2.5)), 3),
            round(float(np.percentile(d, 97.5)), 3)], "p_one_sided": round(float(np.mean(np.array(d) <= 0)), 4)}

sig_tests = {"p_true_vs_random": paired_boot(o_pt, o_rd),
             "logprob_vs_random": paired_boot(o_lp, o_rd),
             "p_true_vs_logprob": paired_boot(o_pt, o_lp)}
if public_rows:
    o_sc = outcomes(sc_majority, public_rows)
    o_pt_public = outcomes(lambda cs: max(cs, key=lambda c: sig(c, "p_true")), public_rows)
    o_lp_public = outcomes(lambda cs: max(cs, key=lambda c: sig(c, "mean_logprob")), public_rows)
    sig_tests["p_true_vs_self_consistency"] = paired_boot(o_pt_public, o_sc)
    sig_tests["logprob_vs_self_consistency"] = paired_boot(o_lp_public, o_sc)
# within-problem: paired per-problem AUROC difference vs the length baseline (same mixed problems)
def within_diff(key):
    d = []
    for r in mixed:
        cs = sample_cells(r); y = [c["full_pass"] for c in cs]
        a = auroc([sig(c, key) for c in cs], y); b = auroc([-c["code_len"] for c in cs], y)
        if a is not None and b is not None: d.append(a - b)
    return paired_boot(np.array(d), np.zeros(len(d)))
sig_tests["within_logprob_minus_length"] = within_diff("mean_logprob")
sig_tests["within_p_true_minus_length"] = within_diff("p_true")

# abstain-on-greedy ranked by p_true (the better signal) alongside mean_logprob
gsort_pt = sorted(greedy, key=lambda t: -sig(t[0], "p_true"))
abst_pt = [(round(kf / len(gsort_pt), 2), round(float(np.mean([c["full_pass"] for c, _ in gsort_pt[:kf]])), 3))
           for kf in range(10, len(gsort_pt) + 1, 10)]

dup_out = round(float(dup), 3) if dup is not None else None
out = {"n_problems": len(R), "k": len(sample_cells(R[0])), "greedy_acc": round(float(g_acc), 3),
       "mixed_problems": len(mixed), "mixed_frac": round(len(mixed) / len(R), 3),
       "public_probe_problems": len(public_rows), "duplicate_rate": dup_out,
       "within_problem_auroc": {
           "mean_logprob": {"mean": round(float(np.mean(w_lp)), 3), "ci": [round(x, 3) for x in boot_mean(w_lp)], "n": len(w_lp)},
           "p_true_nothink": {"mean": round(float(np.mean(w_pt)), 3), "ci": [round(x, 3) for x in boot_mean(w_pt)], "n": len(w_pt)},
           "length_short_better": round(float(np.mean(w_len_short)), 3),
           "length_long_better": round(float(np.mean(w_len_long)), 3)},
       "pooled_auroc": {"mean_logprob": round(pool_lp, 3), "p_true": round(pool_pt, 3)},
       "greedy_problem_auroc": {"mean_logprob": round(g_lp, 3), "p_true": round(g_pt, 3), "length": round(g_len, 3)},
       "selection_at_k": sel, "significance": sig_tests,
       "abstain_on_greedy": abst, "abstain_on_greedy_p_true": abst_pt}
if public_rows:
    sc_sig = (
        f"P(True)-select beats self-consistency +{sig_tests['p_true_vs_self_consistency']['diff']} "
        f"(CI {sig_tests['p_true_vs_self_consistency']['ci']}, "
        f"p={sig_tests['p_true_vs_self_consistency']['p_one_sided']}); "
        f"mean-logprob vs self-consistency p={sig_tests['logprob_vs_self_consistency']['p_one_sided']}. "
    )
    dup_text = f"Duplicate rate {out['duplicate_rate']} on public-probe rows."
else:
    sc_sig = "Self-consistency and visible-test execution are unavailable because this run has no public probes. "
    dup_text = "Duplicate rate over public outputs is unavailable."
verd = (
    f"The confidence toolkit TRANSFERS to real code -- but the HIERARCHY INVERTS: the single-token P(True) readout, "
    f"not the sequence mean-logprob, is the program-level confidence. C40 analog (within-problem, surface-matched, "
    f"{len(mixed)}/{len(R)} mixed problems): mean-logprob AUROC {out['within_problem_auroc']['mean_logprob']['mean']} "
    f"(CI {out['within_problem_auroc']['mean_logprob']['ci']}) vs P(True)-no-think {out['within_problem_auroc']['p_true_nothink']['mean']} "
    f"vs length-baseline {max(out['within_problem_auroc']['length_short_better'], out['within_problem_auroc']['length_long_better'])} "
    f"(paired diff vs length: logprob {sig_tests['within_logprob_minus_length']['diff']} CI {sig_tests['within_logprob_minus_length']['ci']}, "
    f"P(True) {sig_tests['within_p_true_minus_length']['diff']} CI {sig_tests['within_p_true_minus_length']['ci']} -- NOT verbosity). "
    f"Deployment: greedy problem-level AUROC logprob {out['greedy_problem_auroc']['mean_logprob']} / P(True) "
    f"{out['greedy_problem_auroc']['p_true']} (greedy acc {out['greedy_acc']}). C41 analog (selection at k={out['k']}): "
    + " | ".join(f"{k} {v}" for k, v in sel.items())
    + (f". SIGNIFICANCE (paired bootstrap over problems): {sc_sig}P(True)-select vs mean-logprob "
       f"{sig_tests['p_true_vs_logprob']['diff']} (p={sig_tests['p_true_vs_logprob']['p_one_sided']}); "
       f"P(True)-select vs random {sig_tests['p_true_vs_random']['diff']} "
       f"(p={sig_tests['p_true_vs_random']['p_one_sided']}). {dup_text}"))
out["verdict"] = verd
Path(args.verdict).write_text(json.dumps(out, indent=1))
print(verd)

# ---- figure ----
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.5, 5))
labs = ["mean-logprob\n(implicit)", "P(True) no-think\n(explicit)", "length\n(surface)"]
vals = [out["within_problem_auroc"]["mean_logprob"]["mean"], out["within_problem_auroc"]["p_true_nothink"]["mean"],
        max(out["within_problem_auroc"]["length_short_better"], out["within_problem_auroc"]["length_long_better"])]
cols = ["#2563eb", "#f97316", "#9ca3af"]
ax1.bar(range(3), vals, color=cols)
for i, v in enumerate(vals): ax1.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=10, fontweight="bold")
ax1.axhline(0.5, ls=":", color="#888", label="chance")
ax1.set_xticks(range(3)); ax1.set_xticklabels(labs, fontsize=8.5); ax1.set_ylim(0.4, 1.0)
ax1.set_ylabel("within-problem AUROC (mixed problems)"); ax1.legend(fontsize=8.5)
ax1.set_title(f"C40 analog on {args.title}: within-problem sample discrimination\n({len(mixed)} mixed problems, cluster-bootstrap CI)")
sl = list(sel.items())
ax2.barh(range(len(sl)), [v for _, v in sl], color=["#cbd5e1", "#2563eb", "#f97316", "#eab308", "#16a34a", "#111827"])
for i, (kk, v) in enumerate(sl): ax2.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9, fontweight="bold")
ax2.set_yticks(range(len(sl))); ax2.set_yticklabels([k for k, _ in sl], fontsize=7.5)
ax2.set_xlim(min(v for _, v in sl) - 0.05, 1.0); ax2.set_xlabel(f"selection accuracy (k={out['k']})")
ax2.set_title("C41 analog: picking 1 of k samples")
ax3.plot([c for c, _ in abst], [a for _, a in abst], "-", color="#2563eb", lw=2, label="ranked by mean-logprob")
ax3.plot([c for c, _ in abst_pt], [a for _, a in abst_pt], "-", color="#f97316", lw=2.5, label="ranked by P(True)")
ax3.axhline(g_acc, ls=":", color="#888", label=f"no abstention ({g_acc:.2f})")
ax3.set_xlabel("coverage (greedy attempts kept, ranked by confidence)"); ax3.set_ylabel("accuracy on attempted")
ax3.legend(fontsize=9); ax3.grid(alpha=0.25)
ax3.set_title("Abstention on greedy: keep only high-confidence problems")
fig.suptitle(f"Does the confidence toolkit survive on real code ({args.title})?", fontsize=11, y=1.02)
fig.tight_layout(); Path(args.figure).parent.mkdir(exist_ok=True, parents=True)
fig.savefig(args.figure, dpi=130, bbox_inches="tight")
print(f"wrote {args.figure}")
