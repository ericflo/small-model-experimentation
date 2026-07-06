#!/usr/bin/env python3
"""Is the PARAMETER model-latent or surface-readable? On a LARGE fsig-disjoint eval set (activation-only, cheap),
decode the concrete first op two ways and compare: (A) the mid-layer 32-way probe on the model's residual, vs
(B) an EXTERNAL classifier on raw numeric I/O features (list lengths, sums, min/max, elementwise diffs, sorted-
ness) with NO 4B forward pass. If the external-I/O baseline matches the probe (esp. on param|type), the param is
surface-trivial (externalized feature-engineering, not model-latent elicitation); if the probe beats it, the
model genuinely distilled the param. Replaces the degenerate last-token layer-0 control (RoPE => constant)."""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
from capture import ident_prompt, ops_of  # noqa: E402
from fit_probe_full import fsig  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
TYPES = list(FAM_L["prims"])


def io_features(t):
    F = []
    for e in t["visible"]:
        a, b = e["input"], e["output"]; la, lb = len(a), len(b); sa, sb = sum(a), sum(b)
        ov = list(zip(a, b))
        F.append([la, lb, lb-la, sa, sb, sb-sa, (sb-sa)/max(1, la), (sb/sa if sa else 0),
                  min(a) if a else 0, max(a) if a else 0, min(b) if b else 0, max(b) if b else 0,
                  int(b == sorted(b)), int(b == sorted(b, reverse=True)),
                  len(set(b))-len(set(a)), (b[0]-a[0]) if a and b else 0, (b[-1]-a[-1]) if a and b else 0,
                  sum(1 for x, y in ov if x == y), sum(abs(y-x) for x, y in ov)/max(1, len(ov)),
                  (b[0]/a[0] if (a and b and a[0]) else 0)])
    return np.array(F).mean(axis=0)


def gen_tasks(seed, npd, depths, exclude=None):
    rng = random.Random(seed); tasks = []; per = {d: 0 for d in depths}; tid = seed*1000
    while any(per[d] < npd for d in depths) and tid < seed*1000 + 1_200_000:
        tid += 1; d = depths[tid % len(depths)]
        if per[d] >= npd: continue
        t = FAM.make_task(FAM_L, tid, d, rng, k_visible=8, m_hidden=6)
        if not t: continue
        if exclude and fsig(t["target_ops"]) in exclude: continue
        tasks.append(t); per[d] += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=600)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    args = ap.parse_args()

    probes = pickle.load(open(EXP / "data" / "probes_full.pkl", "rb"))
    CONCRETE = json.loads((EXP / "data" / "concrete_vocab.json").read_text())
    cidx = {c: i for i, c in enumerate(CONCRETE)}
    train_fsigs = {tuple(s) for s in json.loads((EXP / "data" / "train_fsigs.json").read_text())}

    # training tasks (seed 2024) -> external-I/O baseline fit
    train = gen_tasks(2024, 600, args.depths)  # regenerates the SAME training pool used to fit the probes
    Xio_tr = np.array([io_features(t) for t in train])
    yc_tr = np.array([cidx[t["target_ops"][0]] for t in train])
    yt_tr = np.array([TYPES.index(ops_of(t)[0]) for t in train])
    scio = StandardScaler().fit(Xio_tr)
    ext_c = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(scio.transform(Xio_tr), yc_tr)
    ext_t = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(scio.transform(Xio_tr), yt_tr)

    eval_t = gen_tasks(9999, args.n_per_depth, args.depths, exclude=train_fsigs)
    print(f"[dec] {len(eval_t)} fsig-disjoint eval tasks", flush=True)
    import gen_lib as GL
    p = GL.Probe()
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in eval_t]
    A = p.activations(seqs, batch_size=16)

    nparams = {}
    for c in CONCRETE:
        op = c.split("(")[0]
        if "(" in c and c[c.index("(")+1:-1] != "": nparams[op] = nparams.get(op, 0) + 1

    out = {}
    for d in args.depths:
        idx = [i for i, t in enumerate(eval_t) if t["depth"] == d]
        pr = probes[d]; c = pr["conc"]; ty = pr["type"]
        Xa = A[[i for i in idx]]
        pc = c["clf"].predict(c["pca"].transform(c["scaler"].transform(Xa[:, c["layer"], :])))
        pt = ty["clf"].predict(ty["pca"].transform(ty["scaler"].transform(Xa[:, ty["layer"], :])))
        Xio = scio.transform(np.array([io_features(eval_t[i]) for i in idx]))
        ec = ext_c.predict(Xio); et = ext_t.predict(Xio)
        tc = [eval_t[i]["target_ops"][0] for i in idx]; tt = [t.split("(")[0] for t in tc]
        isp = ["(" in x and x[x.index("(")+1:-1] != "" for x in tc]
        def acc(pred, true): return round(float(np.mean([CONCRETE[pred[j]] == true[j] for j in range(len(idx))])), 3)
        def tacc(pred, true): return round(float(np.mean([TYPES[pred[j]] == true[j] for j in range(len(idx))])), 3)
        # param|type: among param tasks whose TYPE is right, is the concrete (param) right?
        def paramgiventype(pred_c):
            pj = [j for j in range(len(idx)) if isp[j] and CONCRETE[pred_c[j]].split("(")[0] == tt[j]]
            if not pj: return (float("nan"), 0, float("nan"))
            a = float(np.mean([CONCRETE[pred_c[j]] == tc[j] for j in pj]))
            ch = float(np.mean([1.0/nparams[tt[j]] for j in pj]))
            return (round(a, 3), len(pj), round(ch, 3))
        pgt_probe = paramgiventype(pc); pgt_ext = paramgiventype(ec)
        out[d] = {
            "n": len(idx), "n_param": int(sum(isp)),
            "probe_concrete": acc(pc, tc), "ext_io_concrete": acc(ec, tc),
            "probe_type": tacc(pt, tt), "ext_io_type": tacc(et, tt),
            "probe_param|type": pgt_probe[0], "ext_param|type": pgt_ext[0],
            "param|type_chance": pgt_probe[2], "n_paramtype_correct": pgt_probe[1]}
        print(f"[dec] depth {d} (n={len(idx)}, {int(sum(isp))} param): "
              f"CONCRETE probe {out[d]['probe_concrete']} vs ext-I/O {out[d]['ext_io_concrete']} | "
              f"TYPE probe {out[d]['probe_type']} vs ext {out[d]['ext_io_type']} | "
              f"PARAM|type probe {pgt_probe[0]} vs ext {pgt_ext[0]} (chance {pgt_probe[2]}, n={pgt_probe[1]})", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "decode_results.json").write_text(json.dumps(out, indent=1))
    print("[dec] wrote runs/decode_results.json", flush=True)


if __name__ == "__main__":
    main()
