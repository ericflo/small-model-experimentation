#!/usr/bin/env python3
"""End-to-end bank + value-fill DEPLOY, with the decisive brute-force control (per review). Coverage is
tautological (>=struct-cov by construction; brute-force-all-4096 coverage ~1.0), so the headline is single-pick
DEPLOY under a leakage-free selector (output-consensus on fresh probe inputs). The crux: does the banked model's
behavioral STRUCTURE-filter (infer op-type skeletons from each sample's behavior, no oracle) narrow the 4096
skeletons to a CLEAN set that deploys better than brute-force search (all 4096) + execution-selection (free per
C17)? If bank-deploy >> brute-deploy, banking's structure is a deployable lever; if ~=, free search suffices."""
from __future__ import annotations
import argparse, json, random, sys
from collections import Counter
from itertools import product
from math import comb
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402
from skeleton_fill import ident_prompt, py_solves, model_outputs, all_fills, exec_seq, solves, model_structure_correct  # noqa: E402
from gen_skeletons import TYPES  # noqa: E402
FAM_L = FAM.FAMILIES["list"]


def brute_candidates(task, depth):
    """All (skeleton, fill) over the 4096 depth-3 skeletons passing ALL true visible (first-input pruned)."""
    vis = task["visible"]; x1 = vis[0]["input"]; o1 = vis[0]["output"]; out = []
    for skel in product(TYPES, repeat=depth):
        sk = list(skel)
        for fill in all_fills(sk):
            if exec_seq(fill, x1) != o1:
                continue
            if all(exec_seq(fill, e["input"]) == e["output"] for e in vis):
                out.append((tuple(sk), fill))
    return out


def infer_skels(B, vis_inputs, depth):
    """Op-type skeletons whose SOME param-fill reproduces the model behavior B on the visible inputs (no oracle)."""
    if B is None or "__ERR__" in B:
        return set()
    out = set()
    for skel in product(TYPES, repeat=depth):
        sk = list(skel)
        for fill in all_fills(sk):
            if exec_seq(fill, vis_inputs[0]) != B[0]:
                continue
            if all(exec_seq(fill, x) == b for x, b in zip(vis_inputs, B)):
                out.add(tuple(sk)); break
    return out


def consensus_deploy(cands, probes, task):
    """Leakage-free single-pick: run every visible-passer on fresh probe inputs, deploy the plurality output-vector."""
    if not cands:
        return None  # abstain
    sig2fill = {}
    for _, fill in cands:
        sig = tuple(tuple(exec_seq(fill, x)) if exec_seq(fill, x) is not None else None for x in probes)
        sig2fill.setdefault(sig, []).append(fill)
    best = max(sig2fill, key=lambda s: len(sig2fill[s]))
    return solves(sig2fill[best][0], task["hidden"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--eval-file", type=Path, default=EXP / "data" / "eval_frozen_d3.jsonl")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--probes", type=int, default=16)
    args = ap.parse_args()

    tasks = [json.loads(l) for l in args.eval_file.read_text().splitlines() if l.strip()]
    tasks = [t for t in tasks if not FAM.min_depth_leq(FAM_L,
             [e["input"] for e in t["visible"] + t["hidden"]], [e["output"] for e in t["visible"] + t["hidden"]], t["depth"] - 1)]
    print(f"[bf] {len(tasks)} min-depth-verified held-out tasks", flush=True)

    import gen_lib as GL
    from peft import PeftModel
    p = GL.Probe(); p.model = PeftModel.from_pretrained(p.model, args.adapter).eval()
    prompts = [p.prompt(ident_prompt(t), enable_thinking=False) for t in tasks]
    gg = p.gen_sequences(prompts, think=False, budget=None, greedy=True, answer_max=256, batch_size=64)
    b_greedy = [py_solves(E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")[0], t)
                for t, pr, g in zip(tasks, prompts, gg)]
    flat, fidx = [], []
    for i, pr in enumerate(prompts):
        for _ in range(args.k): flat.append(pr); fidx.append(i)
    gs = p.gen_sequences(flat, think=False, budget=None, greedy=False, answer_max=256, batch_size=64)
    codes = {i: [] for i in range(len(tasks))}; b_cov = [0] * len(tasks)
    for i, pr, g in zip(fidx, flat, gs):
        code, _ = E.extract_candidate_code(p.tok.decode(g["seq_ids"][len(p._ids(pr)):]).strip(), "transform")
        codes[i].append(code); b_cov[i] += int(py_solves(code, tasks[i]))
    print("[bf] banked generation done; running search + skeleton-inference + consensus deploy", flush=True)

    rng = random.Random(2024)
    probes = [[rng.randint(-9, 9) for _ in range(rng.randint(4, 8))] for _ in range(args.probes)]
    rows = []
    for i, t in enumerate(tasks):
        depth = t["depth"]; vis_in = [e["input"] for e in t["visible"]]
        struct = any(model_structure_correct(c, t) for c in codes[i])          # C33 signal, SAME samples
        inferred = set().union(*[infer_skels(model_outputs(c, vis_in), vis_in, depth) for c in codes[i]]) if codes[i] else set()
        infer_yield = int(len(inferred) > 0)
        brute = brute_candidates(t, depth)                                       # all-4096 fill (structure NOT from model)
        bank = [c for c in brute if c[0] in inferred]                            # bank-filtered subset
        def cov(cands): return any(solves(f, t["hidden"]) for _, f in cands)
        def overfit(cands):
            vp = len(cands); bad = sum(1 for _, f in cands if not solves(f, t["hidden"])); return (bad, vp)
        rows.append({
            "b_greedy": b_greedy[i], "b_cov": b_cov[i], "struct": struct, "infer_yield": infer_yield,
            "brute_skels": len({c[0] for c in brute}), "bank_skels": len(inferred),
            "brute_cands": len(brute), "bank_cands": len(bank),
            "brute_cov": cov(brute), "bank_cov": cov(bank),
            "brute_deploy": consensus_deploy(brute, probes, t), "bank_deploy": consensus_deploy(bank, probes, t),
            "brute_overfit": overfit(brute), "bank_overfit": overfit(bank)})
    (EXP / "runs").mkdir(exist_ok=True)
    json.dump({"k": args.k, "n": len(tasks), "probes": args.probes, "rows": rows},
              open(EXP / "runs" / "bankfill_results.json", "w"), indent=1)
    n = len(tasks)
    def covk(c, kk): kk = min(kk, args.k); return 0.0 if c == 0 else (1.0 if args.k-c < kk else 1-comb(args.k-c, kk)/comb(args.k, kk))
    def rate(key, sel=lambda r: True):
        rs = [r for r in rows if sel(r)]; return sum(bool(r[key]) for r in rs)/max(1, len(rs))
    print(f"[bf] banked alone: greedy@1 {sum(b_greedy)/n:.3f} | cov@{args.k} {sum(covk(c,args.k) for c in b_cov)/n:.3f} "
          f"| struct-cov(same samples) {sum(r['struct'] for r in rows)/n:.3f} | infer-yield {sum(r['infer_yield'] for r in rows)/n:.3f}", flush=True)
    print(f"[bf] BRUTE-fill (all-4096): deploy(consensus) {rate('brute_deploy'):.3f} | coverage {rate('brute_cov'):.3f} "
          f"| mean skels/task {sum(r['brute_skels'] for r in rows)/n:.0f}", flush=True)
    print(f"[bf] BANK-fill  (model-filtered): deploy(consensus) {rate('bank_deploy'):.3f} | coverage {rate('bank_cov'):.3f} "
          f"| mean skels/task {sum(r['bank_skels'] for r in rows)/n:.1f}", flush=True)
    bo = [r["brute_overfit"] for r in rows]; ba = [r["bank_overfit"] for r in rows]
    print(f"[bf] visible-overfit rate: brute {sum(b for b,_ in bo)/max(1,sum(v for _,v in bo)):.3f} | "
          f"bank {sum(b for b,_ in ba)/max(1,sum(v for _,v in ba)):.3f}", flush=True)
    print("[bf] wrote runs/bankfill_results.json", flush=True)


if __name__ == "__main__":
    main()
