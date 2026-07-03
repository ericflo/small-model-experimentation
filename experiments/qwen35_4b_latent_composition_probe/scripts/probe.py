#!/usr/bin/env python3
"""Train linear probes per layer/depth to decode the composition from residual-stream activations, with
chance / shuffled-label / layer-0 / behavioral baselines. Emits verdicts on P1-P4 and a chart-ready table."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
RNG = np.random.RandomState(0)


def probe_layer(X, y, n_classes, pca_dim=128, test_frac=0.30):
    """Stratified split, standardize+PCA+L2-logistic, return held-out accuracy."""
    idx = np.arange(len(y))
    # stratified split by class
    test = np.zeros(len(y), bool)
    for c in np.unique(y):
        ci = idx[y == c]
        RNG.shuffle(ci)
        k = max(1, int(round(len(ci) * test_frac)))
        test[ci[:k]] = True
    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    if len(np.unique(ytr)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    d = min(pca_dim, Xtr.shape[1], Xtr.shape[0] - 1)
    if d >= 2:
        pca = PCA(n_components=d, random_state=0).fit(Xtr)
        Xtr, Xte = pca.transform(Xtr), pca.transform(Xte)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced")
    clf.fit(Xtr, ytr)
    return float(clf.score(Xte, yte))


def main():
    A = np.load(EXP / "data" / "acts.npy").astype(np.float32)   # [N, L+1, H]
    present = np.load(EXP / "data" / "present.npy")             # [N, 16]
    lab = json.loads((EXP / "data" / "labels.json").read_text())
    names = lab["names"]
    depth = np.array(lab["depth"])
    first_op = np.array([names.index(o) for o in lab["first_op"]])
    # behavioral entries are None outside the measured subset
    solved = np.array([np.nan if s is None else float(s) for s in lab["solved"]])
    named_ok = np.array([np.nan if s is None else float(s) for s in lab["named_ok"]])
    N, L1, H = A.shape
    depths = sorted(set(depth.tolist()))
    print(f"acts {A.shape} | depths {depths}", flush=True)

    results = {"per_depth": {}, "n_layers": L1}
    for d in depths:
        m = depth == d
        Xd = A[m]                       # [n, L+1, H]
        yd = first_op[m]
        n = int(m.sum())
        chance = Counter(yd.tolist()).most_common(1)[0][1] / n  # majority-class chance
        # first-op probe: sweep layers, record best; also layer-0 and shuffled at best layer
        layer_acc = []
        for L in range(L1):
            layer_acc.append(probe_layer(Xd[:, L, :], yd, len(names)))
        best_L = int(np.nanargmax(layer_acc))
        best = layer_acc[best_L]
        layer0 = layer_acc[0]
        y_shuf = yd.copy(); RNG.shuffle(y_shuf)
        shuffled = probe_layer(Xd[:, best_L, :], y_shuf, len(names))
        # presence macro-F1-ish: mean per-primitive binary accuracy at best layer (only primitives that vary)
        pres_accs = []
        Pd = present[m]
        for j in range(len(names)):
            yj = Pd[:, j]
            if 0.05 < yj.mean() < 0.95:
                pres_accs.append(probe_layer(Xd[:, best_L, :], yj, 2))
        pres = float(np.nanmean(pres_accs)) if pres_accs else float("nan")

        results["per_depth"][int(d)] = {
            "n": n,
            "chance_first_op": round(chance, 3),
            "probe_first_op_best": round(best, 3),
            "best_layer": best_L,
            "probe_first_op_layer0": round(layer0, 3),
            "probe_first_op_shuffled": round(shuffled, 3),
            "probe_presence_meanacc": None if pres != pres else round(pres, 3),
            "behavioral_name_first_op": round(float(np.nanmean(named_ok[m])), 3),
            "behavioral_ident_pass1": round(float(np.nanmean(solved[m])), 3),
            "layer_profile": [round(x, 3) for x in layer_acc],
        }
        print(f"depth {d}: probe first-op best={best:.2f}@L{best_L} chance={chance:.2f} "
              f"layer0={layer0:.2f} shuffled={shuffled:.2f} | behav name={np.nanmean(named_ok[m]):.2f} "
              f"ident={np.nanmean(solved[m]):.2f} | presence={pres:.2f}", flush=True)

    # ---- verdicts ----
    pd = results["per_depth"]
    v = {}
    d1 = pd.get(1, {})
    v["P1_methodology"] = {
        "d1_probe_first_op": d1.get("probe_first_op_best"),
        "d1_ge_0.80": bool(d1.get("probe_first_op_best", 0) >= 0.80),
        "shuffled_all_near_chance": all(
            abs(pd[d]["probe_first_op_shuffled"] - pd[d]["chance_first_op"]) <= 0.08 for d in pd)}
    d3 = pd.get(3, {})
    if d3:
        v["P2_test"] = {
            "d3_probe": d3["probe_first_op_best"], "chance": d3["chance_first_op"],
            "layer0": d3["probe_first_op_layer0"],
            "ge_3x_chance": bool(d3["probe_first_op_best"] >= 3 * d3["chance_first_op"]),
            "ge_layer0_plus_0.15": bool(d3["probe_first_op_best"] >= d3["probe_first_op_layer0"] + 0.15),
            "refuted_absent": bool(d3["probe_first_op_best"] <= d3["chance_first_op"] + 0.05)}
        v["P3_latent"] = {
            "d3_probe": d3["probe_first_op_best"], "d3_behavioral_name": d3["behavioral_name_first_op"],
            "d3_ident_pass1": d3["behavioral_ident_pass1"],
            "probe_ge_behavioral_plus_0.15": bool(
                d3["probe_first_op_best"] >= d3["behavioral_name_first_op"] + 0.15)}
        best_L = d3["best_layer"]
        v["P3_latent"]["best_layer_frac"] = round(best_L / (results["n_layers"] - 1), 2)
    v["P4_gradient"] = {int(d): pd[d]["probe_first_op_best"] for d in sorted(pd)}
    # overall verdict
    latent = bool(d3 and not v["P2_test"]["refuted_absent"]
                  and v["P2_test"]["ge_3x_chance"] and v["P3_latent"]["probe_ge_behavioral_plus_0.15"])
    absent = bool(d3 and v["P2_test"]["refuted_absent"])
    v["VERDICT"] = "LATENT" if latent else ("ABSENT" if absent else "PARTIAL")
    results["verdicts"] = v
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "probe_results.json").write_text(json.dumps(results, indent=1))
    print("\n=== VERDICT:", v["VERDICT"], "===")
    print(json.dumps(v, indent=1))
    print("wrote runs/probe_results.json")


if __name__ == "__main__":
    main()
