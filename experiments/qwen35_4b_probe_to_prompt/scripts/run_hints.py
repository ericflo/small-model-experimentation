#!/usr/bin/env python3
"""Externalize the latent readout. On FRESH held-out tasks (disjoint from probe-training by fsig), decode the
first-op from the base model's OWN activations (C19 probe) and inject it as a prompt hint, then generate.
Arms per task: no-hint; neutral (placebo, format control); oracle-type (TRUE first-op type = probe's ceiling);
oracle-full (TRUE op WITH parameter = ceiling if param-binding is the bottleneck); probe (decoded type);
wrong (RANDOM wrong type = content-causality control). Also reports fsig-disjoint eval probe accuracy and the
LAYER-0 (embedding) probe accuracy (leak control: is the first op model-COMPUTED or surface-readable?).
Tests whether externalizing the readout elicits latent capability where steering (C20) was inert."""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
import numpy as np

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402
from capture import ident_prompt, ops_of, to_public, to_hidden  # noqa: E402
from fit_probe import fsig  # noqa: E402
FAM_L = FAM.FAMILIES["list"]
NAMES = list(FAM_L["prims"])


def full_op_repr(s):  # "add_k(3)" -> "add_k with parameter 3"; "reverse()" -> "reverse"
    op = s.split("(")[0]
    arg = s[s.index("(")+1:-1] if "(" in s else ""
    return f"{op} with parameter {arg}" if arg else op


def hinted_prompt(t, phrase):
    base = ident_prompt(t)
    ins = f"\n\nHint: {phrase}"
    return base.replace("\n\nWrite `def transform", ins + "\n\nWrite `def transform")


def grade(code, t):
    if not code: return False
    try:
        return bool(E.execute_public_and_asserts(code, to_public(t), to_hidden(t))["full_pass"])
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-depth", type=int, default=80)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n1", type=int, default=None, help="override n for depth 1 (sanity)")
    ap.add_argument("--seed", type=int, default=9999)
    args = ap.parse_args()

    probes = pickle.load(open(EXP / "data" / "probes.pkl", "rb"))
    train_fsigs = {tuple(s) for s in json.loads((EXP / "data" / "train_fsigs.json").read_text())}

    rng = random.Random(args.seed); wr = random.Random(4242); tasks = []
    target = {d: (args.n1 if (d == 1 and args.n1) else args.n_per_depth) for d in args.depths}
    per = {d: 0 for d in args.depths}; tid = 800_000
    while any(per[d] < target[d] for d in args.depths) and tid < 800_000 + 800_000:
        tid += 1; d = args.depths[tid % len(args.depths)]
        if per[d] >= target[d]: continue
        t = FAM.make_task(FAM_L, tid, d, rng, k_visible=8, m_hidden=6)
        if not t or fsig(t["target_ops"]) in train_fsigs: continue
        tasks.append(t); per[d] += 1
    assert all(fsig(t["target_ops"]) not in train_fsigs for t in tasks), "train/eval fsig overlap!"
    print(f"[hints] {len(tasks)} fresh eval tasks (fsig-disjoint from train), per-depth {per}", flush=True)

    import gen_lib as GL
    p = GL.Probe()
    seqs = [p._ids(p.prompt(ident_prompt(t), enable_thinking=False)) for t in tasks]
    A = p.activations(seqs, batch_size=16)
    pred_op, pred0_op, true_op, true_full, wrong_op = [], [], [], [], []
    for i, t in enumerate(tasks):
        d = t["depth"]; pr = probes[d]; L = pr["layer"]
        xi = pr["clf"].predict(pr["pca"].transform(pr["scaler"].transform(A[i:i+1, L, :])))[0]
        x0 = pr["clf0"].predict(pr["pca0"].transform(pr["scaler0"].transform(A[i:i+1, 0, :])))[0]
        pred_op.append(NAMES[int(xi)]); pred0_op.append(NAMES[int(x0)])
        ti = NAMES.index(ops_of(t)[0]); true_op.append(NAMES[ti]); true_full.append(full_op_repr(t["target_ops"][0]))
        w = wr.choice([n for n in NAMES if n != NAMES[ti]]); wrong_op.append(w)
    # probe accuracy on the fsig-disjoint eval set (the HONEST gating number), by depth, vs majority baseline
    acc, acc0, majority = {}, {}, {}
    for d in args.depths:
        idx = [i for i, t in enumerate(tasks) if t["depth"] == d]
        if not idx: continue
        acc[d] = round(sum(pred_op[i] == true_op[i] for i in idx) / max(1, len(idx)), 3)
        acc0[d] = round(sum(pred0_op[i] == true_op[i] for i in idx) / max(1, len(idx)), 3)
        from collections import Counter
        majority[d] = round(Counter(true_op[i] for i in idx).most_common(1)[0][1] / max(1, len(idx)), 3)
    print(f"[hints] probe EVAL acc (mid-layer) {acc} | layer-0 (leak ctrl) {acc0} | majority {majority}", flush=True)

    arms = {
        "nohint":  [ident_prompt(t) for t in tasks],
        "neutral": [hinted_prompt(t, "the first operation applied to the input is one of the standard list primitives.") for t in tasks],
        "oracle_type": [hinted_prompt(t, f"the first operation applied to the input (before any others) is `{true_op[i]}`.") for i, t in enumerate(tasks)],
        "oracle_full": [hinted_prompt(t, f"the first operation applied to the input (before any others) is {true_full[i]}.") for i, t in enumerate(tasks)],
        "probe":   [hinted_prompt(t, f"the first operation applied to the input (before any others) is `{pred_op[i]}`.") for i, t in enumerate(tasks)],
        "wrong":   [hinted_prompt(t, f"the first operation applied to the input (before any others) is `{wrong_op[i]}`.") for i, t in enumerate(tasks)],
    }

    rec = {i: {"depth": tasks[i]["depth"], "probe_correct": pred_op[i] == true_op[i]} for i in range(len(tasks))}
    for arm, prompts in arms.items():
        pr_nt = [p.prompt(x, enable_thinking=False) for x in prompts]
        gg = p.gen_sequences(pr_nt, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
        for i, prm, g in zip(range(len(tasks)), pr_nt, gg):
            ans = p.tok.decode(g["seq_ids"][len(p._ids(prm)):]).strip()
            code, _ = E.extract_candidate_code(ans, "transform")
            rec[i][f"{arm}_greedy"] = grade(code, tasks[i])
        flat, fidx = [], []
        for i, prm in enumerate(pr_nt):
            for _ in range(args.k): flat.append(prm); fidx.append(i)
        gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
        nc = {i: 0 for i in range(len(tasks))}
        for i, prm, g in zip(fidx, flat, gs):
            ans = p.tok.decode(g["seq_ids"][len(p._ids(prm)):]).strip()
            code, _ = E.extract_candidate_code(ans, "transform")
            nc[i] += int(grade(code, tasks[i]))
        for i in range(len(tasks)): rec[i][f"{arm}_ncorrect"] = nc[i]
        gr = sum(rec[i][f"{arm}_greedy"] for i in range(len(tasks))) / len(tasks)
        print(f"[hints] arm {arm:12} greedy@1 {gr:.3f}", flush=True)

    results = {"per_task": [rec[i] for i in range(len(tasks))], "probe_eval_acc": acc,
               "layer0_eval_acc": acc0, "majority": majority, "per_depth": per, "K": args.k}
    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "hint_results.json").write_text(json.dumps(results, indent=1))
    print("[hints] wrote runs/hint_results.json", flush=True)


if __name__ == "__main__":
    main()
