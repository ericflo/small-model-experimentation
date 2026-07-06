#!/usr/bin/env python3
"""Is the PARAMETER latent? C30 found externalizing the op-TYPE only narrows sampling; the concrete (op,param)
is the deployable bottleneck. Fit two probes per depth on training activations: a 16-way op-TYPE probe (C19) and
a 32-way CONCRETE-op probe (op+param). Save both (mid-layer + layer-0). run_hints_full.py then decodes the full
first op and injects it as a hint. Layer-0 probe = surface control: the PARAM may be surface-readable from I/O
magnitudes (unlike the type, which C19 showed is computed)."""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
from capture import ident_prompt, ops_of  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
TYPES = list(FAM_L["prims"])
RNG = np.random.RandomState(0)

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
    if len(np.unique(y[~test])) < 2: return float("nan")
    sc, pca, clf = fit_at(X[~test], y[~test], dim)
    return clf.score(pca.transform(sc.transform(X[test])), y[test])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=600)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    rng = random.Random(args.seed); tasks = []
    for d in args.depths:
        made = 0
        while made < args.n_per_depth:
            t = FAM.make_task(FAM_L, len(tasks), d, rng, k_visible=8, m_hidden=6)
            if t: tasks.append(t); made += 1
    CONCRETE = sorted({t["target_ops"][0] for t in tasks})  # ~32 concrete first ops
    cidx = {c: i for i, c in enumerate(CONCRETE)}
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "train_fsigs.json").write_text(json.dumps([list(fsig(t["target_ops"])) for t in tasks]))
    (EXP / "data" / "concrete_vocab.json").write_text(json.dumps(CONCRETE))
    print(f"[fit] {len(tasks)} training tasks | {len(CONCRETE)} concrete first-ops | {len(TYPES)} types", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in tasks]
    A = p.activations(seqs, batch_size=16)
    print(f"[fit] activations {A.shape}", flush=True)
    depth = np.array([t["depth"] for t in tasks])
    y_type = np.array([TYPES.index(ops_of(t)[0]) for t in tasks])
    y_conc = np.array([cidx[t["target_ops"][0]] for t in tasks])

    probes = {}
    for d in args.depths:
        m = depth == d; Xd = A[m]; L1 = Xd.shape[1]
        # type probe (16-way): best layer
        at = [heldout_acc(Xd[:, L, :], y_type[m]) for L in range(L1)]
        Lt = int(np.nanargmax(at))
        sct, pcat, clft = fit_at(Xd[:, Lt, :], y_type[m])
        # concrete probe (32-way): best layer
        ac = [heldout_acc(Xd[:, L, :], y_conc[m]) for L in range(L1)]
        Lc = int(np.nanargmax(ac))
        scc, pcac, clfc = fit_at(Xd[:, Lc, :], y_conc[m])
        # layer-0 concrete (surface control: is the param surface-readable?)
        sc0, pca0, clf0 = fit_at(Xd[:, 0, :], y_conc[m])
        probes[d] = {"type": {"layer": Lt, "scaler": sct, "pca": pcat, "clf": clft, "heldout": round(at[Lt], 3)},
                     "conc": {"layer": Lc, "scaler": scc, "pca": pcac, "clf": clfc, "heldout": round(ac[Lc], 3)},
                     "conc0": {"scaler": sc0, "pca": pca0, "clf": clf0}}
        print(f"[fit] depth {d}: TYPE(16) best L{Lt} heldout {at[Lt]:.3f} | CONCRETE(32) best L{Lc} heldout {ac[Lc]:.3f}", flush=True)
    pickle.dump(probes, open(EXP / "data" / "probes_full.pkl", "wb"))
    print("[fit] saved data/probes_full.pkl", flush=True)


if __name__ == "__main__":
    main()
