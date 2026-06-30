#!/usr/bin/env python3
"""Fit per-layer linear probes on answer-token activations; does thinking make correctness
more linearly decodable? Grouped-by-task CV, bootstrap CIs, shuffled-label control, and a
deployable false-pass test among visible-test passers.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parents[1]
ACTS = EXP.parents[1] / "large_artifacts" / "qwen35_4b_thinking_content_vs_compute"
RNG = np.random.RandomState(0)


def load():
    labels = [json.loads(l) for l in (EXP / "data" / "labels.jsonl").read_text().splitlines() if l.strip()]
    by_cond = defaultdict(list)
    for r in labels:
        by_cond[r["cond"]].append(r)
    return by_cond


def oof_proba(X, y, groups, n_splits=5):
    # C=0.4 + standardized features converge fast on 2560-dim activations; n_jobs parallelizes folds.
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=0.4, solver="lbfgs"))
    gkf = GroupKFold(n_splits=min(n_splits, len(set(groups))))
    return cross_val_predict(clf, X, y, cv=gkf, groups=groups, method="predict_proba", n_jobs=-1)[:, 1]


def boot_auc_ci(y, proba, groups, n=100):
    g = np.array(groups); uniq = np.unique(g); aucs = []
    for _ in range(n):
        samp = RNG.choice(uniq, len(uniq), replace=True)
        mask = np.concatenate([np.where(g == u)[0] for u in samp])
        yy, pp = y[mask], proba[mask]
        if len(set(yy)) == 2:
            aucs.append(roc_auc_score(yy, pp))
    return (np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)) if aucs else (float("nan"), float("nan"))


def main():
    by_cond = load()
    conds = [c for c in ["no_think", "foreign", "shuffle", "real"] if c in by_cond]
    results = {}
    for cond in conds:
        recs = sorted(by_cond[cond], key=lambda r: r["row"])
        acts = np.load(ACTS / f"acts_{cond}.npy")  # [N, L+1, H]
        rows = [r["row"] for r in recs]
        A = acts[rows].astype(np.float32)            # align to labels order
        y = np.array([int(r["full_pass"]) for r in recs])
        vis = np.array([int(r["visible_pass"]) for r in recs])
        groups = np.array([r["task_id"] for r in recs])
        n_layers = A.shape[1]
        base_rate = y.mean()
        if len(set(y)) < 2:
            print(f"  [{cond}] degenerate labels (pass={base_rate:.2f}); skipping"); continue

        layer_auc = []
        for l in range(n_layers):
            proba = oof_proba(A[:, l, :], y, groups)
            layer_auc.append(roc_auc_score(y, proba))
        best_l = int(np.argmax(layer_auc))
        proba_best = oof_proba(A[:, best_l, :], y, groups)
        lo, hi = boot_auc_ci(y, proba_best, groups)
        # shuffled-label control at best layer
        y_shuf = RNG.permutation(y)
        auc_shuf = roc_auc_score(y_shuf, oof_proba(A[:, best_l, :], y_shuf, groups))
        # false-pass detection among visible passers
        vp = vis == 1
        vp_auc = float("nan"); vp_n = int(vp.sum()); vp_fullrate = float("nan")
        if vp.sum() > 5 and len(set(y[vp])) == 2:
            vp_auc = roc_auc_score(y[vp], proba_best[vp]); vp_fullrate = y[vp].mean()

        results[cond] = {
            "n": len(y), "base_pass_rate": round(base_rate, 3),
            "best_layer": best_l, "best_layer_auc": round(layer_auc[best_l], 3),
            "best_layer_auc_ci": [round(lo, 3), round(hi, 3)],
            "shuffled_label_auc": round(auc_shuf, 3),
            "layer_auc": [round(a, 3) for a in layer_auc],
            "visible_passer_n": vp_n, "visible_passer_full_rate": round(vp_fullrate, 3) if vp_fullrate == vp_fullrate else None,
            "visible_passer_auc": round(vp_auc, 3) if vp_auc == vp_auc else None,
        }
        print(f"  [{cond}] best L{best_l} AUC={layer_auc[best_l]:.3f} "
              f"CI[{lo:.3f},{hi:.3f}] shuf={auc_shuf:.3f} pass={base_rate:.2f} "
              f"vis-passer AUC={vp_auc:.3f} (n={vp_n})", flush=True)

    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "probe_results.json").write_text(json.dumps(results, indent=2))

    # table
    lines = ["| condition | base pass | best layer | probe AUC | 95% CI | shuffled-label | visible-passer AUC (n) |",
             "| --- | ---: | ---: | ---: | :---: | ---: | ---: |"]
    for c in conds:
        if c not in results:
            continue
        r = results[c]
        lines.append(f"| {c} | {r['base_pass_rate']:.2f} | {r['best_layer']} | {r['best_layer_auc']:.3f} | "
                     f"[{r['best_layer_auc_ci'][0]:.2f},{r['best_layer_auc_ci'][1]:.2f}] | {r['shuffled_label_auc']:.3f} | "
                     f"{r['visible_passer_auc']} ({r['visible_passer_n']}) |")
    table = "\n".join(lines)
    (EXP / "analysis" / "probe_summary.md").write_text(table + "\n")
    print("\n" + table)

    # figure: AUC vs layer per condition
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"no_think": "#264653", "foreign": "#8d99ae", "shuffle": "#f4a261", "real": "#2a9d8f"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        if c not in results:
            continue
        ax.plot(range(len(results[c]["layer_auc"])), results[c]["layer_auc"], "-",
                color=colors.get(c, None), label=c, linewidth=1.8)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_xlabel("layer (0=embeddings)")
    ax.set_ylabel("probe AUC (predict full-test pass from answer-token activation)")
    ax.set_title("Thinking content vs compute: foreign/shuffle/real separability (MBPP)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(EXP / "analysis" / "auc_vs_layer.png", dpi=130)
    print(f"\nwrote runs/probe_results.json, analysis/probe_summary.md, analysis/auc_vs_layer.png")


if __name__ == "__main__":
    main()
