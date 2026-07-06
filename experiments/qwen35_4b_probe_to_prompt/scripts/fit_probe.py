#!/usr/bin/env python3
"""Fit C19's linear first-op probe (standardize+PCA128+L2-logistic) per depth at the best layer, on training
tasks, and SAVE the fitted probe (scaler+pca+clf+layer) so run_hints.py can decode the first-op on fresh
held-out tasks and externalize it as a prompt hint. Reports internal held-out accuracy (should match C19:
depth-1 ~0.99, depth-2 ~0.42, depth-3 ~0.27)."""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
NAMES = list(FAM_L["prims"])
sys.path.insert(0, str(EXP / "scripts"))
from capture import ident_prompt, ops_of  # noqa: E402
RNG = np.random.RandomState(0)

# probe fsig for train/eval disjointness (list-family, matches other experiments)
PROBEVEC = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]
def fsig(target_ops):
    out = []
    for x in PROBEVEC:
        st = list(x)
        for s in target_ops:
            op = s.split("(")[0]; k = int(s[s.index("(")+1:-1]) if "(" in s else None
            st = FAM.apply_op(FAM_L, op, k, st)
            if st is None: break
        out.append(repr(st))
    return tuple(out)


def fit_at(X, y, dim=128):
    sc = StandardScaler().fit(X); Xs = sc.transform(X)
    d = min(dim, Xs.shape[1], Xs.shape[0]-1)
    pca = PCA(n_components=d, random_state=0).fit(Xs)
    clf = LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced").fit(pca.transform(Xs), y)
    return sc, pca, clf

def heldout_acc(X, y, dim=128, test_frac=0.30):
    idx = np.arange(len(y)); test = np.zeros(len(y), bool)
    for c in np.unique(y):
        ci = idx[y == c]; RNG.shuffle(ci); k = max(1, int(round(len(ci)*test_frac))); test[ci[:k]] = True
    sc, pca, clf = fit_at(X[~test], y[~test], dim)
    return clf.score(pca.transform(sc.transform(X[test])), y[test])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=500)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=2024)  # C19's training seed
    args = ap.parse_args()

    rng = random.Random(args.seed); tasks = []
    for d in args.depths:
        made = 0
        while made < args.n_per_depth:
            t = FAM.make_task(FAM_L, len(tasks), d, rng, k_visible=8, m_hidden=6)
            if t: tasks.append(t); made += 1
    train_fsigs = {fsig(t["target_ops"]) for t in tasks}
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "train_fsigs.json").write_text(json.dumps([list(s) for s in train_fsigs]))
    print(f"[fit] {len(tasks)} training tasks (seed {args.seed})", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in tasks]
    A = p.activations(seqs, batch_size=16)  # [N, L+1, H]
    print(f"[fit] activations {A.shape}", flush=True)
    depth = np.array([t["depth"] for t in tasks])
    first_op = np.array([NAMES.index(ops_of(t)[0]) for t in tasks])

    probes = {}
    for d in args.depths:
        m = depth == d; Xd = A[m]; yd = first_op[m]; L1 = Xd.shape[1]
        # sweep layers on internal held-out, pick best layer (frozen; the HONEST accuracy is the fsig-disjoint
        # eval-set accuracy reported by run_hints, not this internal number)
        accs = [heldout_acc(Xd[:, L, :], yd) for L in range(L1)]
        bestL = int(np.nanargmax(accs)); best = accs[bestL]
        sc, pca, clf = fit_at(Xd[:, bestL, :], yd)          # refit on ALL training at best layer
        sc0, pca0, clf0 = fit_at(Xd[:, 0, :], yd)           # layer-0 (embedding) probe = leak/surface control
        probes[d] = {"layer": bestL, "scaler": sc, "pca": pca, "clf": clf, "heldout_acc": round(best, 3),
                     "scaler0": sc0, "pca0": pca0, "clf0": clf0}
        print(f"[fit] depth {d}: best layer L{bestL} internal-heldout {best:.3f} "
              f"(C19 ref d1~0.99 d2~0.42 d3~0.27; honest number is the fsig-disjoint eval acc)", flush=True)
    pickle.dump(probes, open(EXP / "data" / "probes.pkl", "wb"))
    print("[fit] saved data/probes.pkl (mid-layer + layer-0 probes per depth)", flush=True)


if __name__ == "__main__":
    main()
