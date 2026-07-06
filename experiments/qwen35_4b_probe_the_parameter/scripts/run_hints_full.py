#!/usr/bin/env python3
"""Deployability of the full-op hint, split by param vs non-param first ops. decode_eval showed the op-TYPE is
model-latent (probe > surface I/O) but the PARAMETER is surface-readable (external I/O matches the probe). Here:
does externalizing the full op deploy, is the PARAM the deployable bottleneck (oracle_full - oracle_type on param
tasks), and does the model probe beat the cheap SURFACE pipeline (probe_full vs surface_full)? Arms: no-hint;
oracle_type (true type, no param); oracle_full (true op+param); probe_full (32-way probe); surface_full (external
I/O classifier); wrong_param (true type, WRONG param). Reports greedy@1 + coverage, SPLIT by is_param."""
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
import code_env as E  # noqa: E402
from capture import ident_prompt, ops_of, to_public, to_hidden  # noqa: E402
from fit_probe_full import fsig  # noqa: E402
from decode_eval import io_features, gen_tasks  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
TYPES = list(FAM_L["prims"])


def full_op_repr(s):
    op = s.split("(")[0]; arg = s[s.index("(")+1:-1] if "(" in s else ""
    return f"{op} with parameter {arg}" if arg else op


def hinted(t, phrase):
    return ident_prompt(t).replace("\n\nWrite `def transform", f"\n\nHint: {phrase}\n\nWrite `def transform")


def grade(code, t):
    if not code: return False
    try:
        return bool(E.execute_public_and_asserts(code, to_public(t), to_hidden(t))["full_pass"])
    except Exception:
        return False


def wrong_param_of(concrete, rng):
    """Same op TYPE, a different valid parameter (else fall back to the true op)."""
    op = concrete.split("(")[0]
    if "(" not in concrete or concrete[concrete.index("(")+1:-1] == "": return concrete
    alts = [c for c in ALLC if c.split("(")[0] == op and c != concrete]
    return rng.choice(alts) if alts else concrete


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=130)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--seed", type=int, default=7777)
    args = ap.parse_args()

    global ALLC
    probes = pickle.load(open(EXP / "data" / "probes_full.pkl", "rb"))
    ALLC = json.loads((EXP / "data" / "concrete_vocab.json").read_text())
    cidx = {c: i for i, c in enumerate(ALLC)}
    train_fsigs = {tuple(s) for s in json.loads((EXP / "data" / "train_fsigs.json").read_text())}

    # external-I/O surface classifier (fit on training pool, same as decode_eval)
    train = gen_tasks(2024, 600, args.depths)
    Xtr = StandardScaler().fit(np.array([io_features(t) for t in train]))
    ext = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced").fit(
        Xtr.transform(np.array([io_features(t) for t in train])), np.array([cidx[t["target_ops"][0]] for t in train]))

    rng = random.Random(args.seed); wr = random.Random(4242)
    eval_t = gen_tasks(args.seed, args.n_per_depth, args.depths, exclude=train_fsigs)
    print(f"[full] {len(eval_t)} fresh eval tasks", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in eval_t]
    A = p.activations(seqs, batch_size=16)
    dec_conc, dec_surf, true_conc, is_param = [], [], [], []
    for i, t in enumerate(eval_t):
        d = t["depth"]; c = probes[d]["conc"]
        ci = int(c["clf"].predict(c["pca"].transform(c["scaler"].transform(A[i:i+1, c["layer"], :])))[0])
        si = int(ext.predict(Xtr.transform(io_features(t).reshape(1, -1)))[0])
        dec_conc.append(ALLC[ci]); dec_surf.append(ALLC[si])
        tc = t["target_ops"][0]; true_conc.append(tc)
        is_param.append("(" in tc and tc[tc.index("(")+1:-1] != "")

    arms = {
        "nohint":      [ident_prompt(t) for t in eval_t],
        "oracle_type": [hinted(t, f"the first operation applied to the input is `{true_conc[i].split('(')[0]}`.") for i, t in enumerate(eval_t)],
        "oracle_full": [hinted(t, f"the first operation applied to the input is {full_op_repr(true_conc[i])}.") for i, t in enumerate(eval_t)],
        "probe_full":  [hinted(t, f"the first operation applied to the input is {full_op_repr(dec_conc[i])}.") for i, t in enumerate(eval_t)],
        "surface_full":[hinted(t, f"the first operation applied to the input is {full_op_repr(dec_surf[i])}.") for i, t in enumerate(eval_t)],
        "wrong_param": [hinted(t, f"the first operation applied to the input is {full_op_repr(wrong_param_of(true_conc[i], wr))}.") for i, t in enumerate(eval_t)],
    }
    rec = {i: {"depth": eval_t[i]["depth"], "is_param": is_param[i], "conc_correct": dec_conc[i] == true_conc[i],
               "surf_correct": dec_surf[i] == true_conc[i]} for i in range(len(eval_t))}
    for arm, prompts in arms.items():
        pr_nt = [p.prompt(x, enable_thinking=False) for x in prompts]
        gg = p.gen_sequences(pr_nt, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
        for i, prm, g in zip(range(len(eval_t)), pr_nt, gg):
            code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(prm)):]).strip(), "transform")
            rec[i][f"{arm}_greedy"] = grade(code, eval_t[i])
        flat, fidx = [], []
        for i, prm in enumerate(pr_nt):
            for _ in range(args.k): flat.append(prm); fidx.append(i)
        gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
        nc = {i: 0 for i in range(len(eval_t))}
        for i, prm, g in zip(fidx, flat, gs):
            code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(prm)):]).strip(), "transform")
            nc[i] += int(grade(code, eval_t[i]))
        for i in range(len(eval_t)): rec[i][f"{arm}_ncorrect"] = nc[i]
        pg = sum(rec[i][f"{arm}_greedy"] for i in range(len(eval_t)) if rec[i]["is_param"]) / max(1, sum(is_param))
        print(f"[full] arm {arm:12} greedy@1(param-tasks) {pg:.3f}", flush=True)
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "full_results.json").write_text(json.dumps(
        {"per_task": [rec[i] for i in range(len(eval_t))], "K": args.k,
         "probe_conc_acc": round(sum(r["conc_correct"] for r in rec.values())/len(rec), 3),
         "surf_conc_acc": round(sum(r["surf_correct"] for r in rec.values())/len(rec), 3)}, indent=1))
    print("[full] wrote runs/full_results.json", flush=True)


if __name__ == "__main__":
    main()
