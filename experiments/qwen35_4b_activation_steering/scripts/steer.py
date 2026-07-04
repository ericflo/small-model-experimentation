#!/usr/bin/env python3
"""Activation steering: add C19's mean-difference 'first-op' direction to the residual stream during
generation, and test whether it moves the model's behavior (naming, then identification). Oracle upper-bound
usability test for the latent signal. See reports/prereg.md."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402

PROBE = EXP.parent / "qwen35_4b_latent_composition_probe"
SCRATCH = Path("/tmp/claude-1000/-home-ericflo-Development-small-model-experimentation/"
               "023b1a84-6a82-4a18-85df-570f97a29549/scratchpad/probe_artifacts")
FAM_L = FAM.FAMILIES["list"]
NAMES = list(FAM_L["prims"])
BEST_LAYER = {1: 15, 2: 22}  # C19 probe-index best layers (hidden_states index; decoder layer = L-1)
FIRST_RE = re.compile(r"First:\s*([a-z_]+)")


def ident_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `def transform(xs):` reproducing this for all such inputs. Only a ```python code block.")


def name_prompt(t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"These examples come from applying a pipeline of operations to the input list, in order:\n{lines}\n\n"
            f"Available operations: {', '.join(NAMES)}.\n\n"
            f"Which operation is applied FIRST (directly to the input, before any other)? "
            f"Answer with exactly one operation name on the last line as `First: <name>`.")


def ops_of(t):
    return [s.split("(")[0] for s in t["target_ops"]]


def build_directions(depth, layer):
    """mean(class c) - mean(all) at hidden_states[layer], from C19 cached acts (matched depth)."""
    A = np.load(SCRATCH / "acts.npy")                       # [N, 33, H]
    lab = json.loads((PROBE / "data" / "labels.json").read_text())
    dep = np.array(lab["depth"]); fop = np.array(lab["first_op"])
    m = dep == depth
    X = A[m, layer, :].astype(np.float32)                   # [n, H]
    fo = fop[m]
    mu_all = X.mean(0)
    dirs = {}
    for c in NAMES:
        sel = fo == c
        if sel.sum() >= 3:
            dirs[c] = X[sel].mean(0) - mu_all
    resid_norm = float(np.linalg.norm(X, axis=1).mean())
    return dirs, resid_norm


class Steerer:
    """Forward hook on model.model.layers[dec] adding a per-example [B,H] vector to the residual."""
    def __init__(self, model, dec_layer):
        self.layer = model.model.layers[dec_layer]
        self.vec = None
        self.handle = None

    def set(self, vecs):  # vecs: torch [B, H] or None
        self.vec = vecs

    def _hook(self, module, inp, out):
        if self.vec is None:
            return out
        if isinstance(out, tuple):
            h = out[0]
            h = h + self.vec[:, None, :].to(h.dtype)
            return (h,) + tuple(out[1:])
        return out + self.vec[:, None, :].to(out.dtype)

    def __enter__(self):
        self.handle = self.layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        if self.handle:
            self.handle.remove()


def steered_generate(p, steerer, prompts, vecs, max_new, greedy=True, think=False, force_suffix=None):
    """vecs: np array [B, H] or None. force_suffix: token string appended after the prompt (e.g. 'First:')
    to force a direct answer without thinking. Returns decoded answer text per prompt."""
    rendered = [p.prompt(pr, enable_thinking=think) for pr in prompts]
    ids = [p._ids(r) for r in rendered]
    if force_suffix is not None:
        suf = p.tok(force_suffix, add_special_tokens=False).input_ids
        ids = [x + suf for x in ids]
    pad = p.tok.pad_token_id
    maxlen = max(len(x) for x in ids)
    inp = torch.tensor([[pad] * (maxlen - len(x)) + x for x in ids], device=p.device)
    attn = (inp != pad).long()
    steerer.set(None if vecs is None else torch.tensor(vecs, device=p.device))
    sampling = dict(do_sample=False) if greedy else dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)
    with torch.no_grad():
        out = p.model.generate(input_ids=inp, attention_mask=attn, max_new_tokens=max_new,
                               pad_token_id=pad, **sampling)
    steerer.set(None)
    texts = []
    for j in range(len(prompts)):
        gen = out[j, maxlen:].tolist()
        txt = p.tok.decode(gen, skip_special_tokens=False)
        txt = txt.split("</think>")[-1] if "</think>" in txt else txt
        texts.append(txt)
    return texts


def parse_first(txt):
    m = list(FIRST_RE.finditer(txt))
    if m:
        return m[-1].group(1)
    return next((n for n in NAMES if n in txt), "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--coefs", type=float, nargs="+", default=[0, 2, 4, 8, 16])
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 1])
    ap.add_argument("--seed", type=int, default=555)
    ap.add_argument("--ident-coef", type=float, default=None, help="coef for the identification arm")
    ap.add_argument("--layer", type=int, default=None, help="override steering layer (probe index)")
    ap.add_argument("--no-ident", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n, args.coefs, args.depths = 8, [0, 8], [2]

    import gen_lib as GL
    p = GL.Probe()
    rng = random.Random(args.seed)
    results = {"per_depth": {}}

    for depth in args.depths:
        layer = args.layer if args.layer is not None else BEST_LAYER[depth]
        dirs, resid_norm = build_directions(depth, layer)
        unit = {c: v / (np.linalg.norm(v) + 1e-8) for c, v in dirs.items()}  # for random-norm matching
        # fresh held-out tasks with a known first-op we have a direction for
        tasks = []
        while len(tasks) < args.n:
            t = FAM.make_task(FAM_L, len(tasks), depth, rng, k_visible=8, m_hidden=6)
            if t and ops_of(t)[0] in dirs:
                tasks.append(t)
        steerer = Steerer(p.model, layer - 1)
        t0 = time.time()
        print(f"[depth {depth}] {len(tasks)} tasks, steer layer {layer} (dec {layer-1}), "
              f"resid_norm {resid_norm:.1f}", flush=True)

        def cond_vecs(cond, coef):
            if coef == 0 or cond == "baseline":
                return None
            V = np.zeros((len(tasks), dirs[NAMES[0]].shape[0]), dtype=np.float32)
            for i, t in enumerate(tasks):
                true_c = ops_of(t)[0]
                if cond == "steer_true":
                    V[i] = coef * dirs[true_c]
                elif cond == "steer_wrong":
                    wrong = rng.choice([c for c in dirs if c != true_c])
                    V[i] = coef * dirs[wrong]
                elif cond == "steer_random":
                    r = np.array([rng.gauss(0, 1) for _ in range(V.shape[1])], dtype=np.float32)
                    V[i] = np.linalg.norm(coef * dirs[true_c]) * r / (np.linalg.norm(r) + 1e-8)
            return V

        nprompts = [name_prompt(t) for t in tasks]
        truth = [ops_of(t)[0] for t in tasks]
        by = {}
        with steerer:
            for coef in args.coefs:
                for cond in (["baseline"] if coef == 0 else ["steer_true", "steer_wrong", "steer_random"]):
                    vecs = cond_vecs(cond, coef)
                    # batch the naming generations
                    texts = []
                    B = 32
                    for s in range(0, len(tasks), B):
                        sub = slice(s, s + B)
                        texts += steered_generate(p, steerer, nprompts[sub], None if vecs is None else vecs[sub],
                                                  max_new=8, greedy=True, think=False, force_suffix="First:")
                    picks = [parse_first("First:" + tx) for tx in texts]
                    acc = np.mean([pk == tr for pk, tr in zip(picks, truth)])
                    parse = np.mean([pk in NAMES for pk in picks])
                    by[f"{cond}@{coef:g}"] = {"naming_acc": round(float(acc), 3), "parse": round(float(parse), 3)}
                    print(f"  {cond}@{coef:g}: naming {acc:.2f} (parse {parse:.2f}) [{time.time()-t0:.0f}s]", flush=True)
        results["per_depth"][depth] = {"n": len(tasks), "layer": layer, "resid_norm": round(resid_norm, 1),
                                       "naming": by, "baseline": by["baseline@0"]["naming_acc"]}

        # --- identification arm (secondary): baseline vs steer_true at ident-coef ---
        if depth == 2 and not args.no_ident:
            icoef = args.ident_coef if args.ident_coef is not None else _best_coef(by)
            iprompts = [ident_prompt(t) for t in tasks]
            arm = {}
            with steerer:
                for cond, coef in (("baseline", 0.0), ("steer_true", icoef)):
                    vecs = cond_vecs("steer_true" if cond == "steer_true" else "baseline", coef)
                    solved = []
                    B = 16
                    for s in range(0, len(tasks), B):
                        sub = slice(s, s + B)
                        texts = steered_generate(p, steerer, iprompts[sub],
                                                 None if vecs is None else vecs[sub],
                                                 max_new=512, greedy=True, think=False)
                        for t, tx in zip(tasks[sub], texts):
                            c, _ = E.extract_candidate_code(tx, "transform")
                            pub = [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]]
                            hid = [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]]
                            solved.append(bool(c) and bool(E.execute_public_and_asserts(c, pub, hid)["full_pass"]))
                    arm[cond] = round(float(np.mean(solved)), 3)
                    print(f"  ident {cond}@{coef}: pass@1 {np.mean(solved):.2f} [{time.time()-t0:.0f}s]", flush=True)
            results["per_depth"][depth]["ident_coef"] = icoef
            results["per_depth"][depth]["ident"] = arm

    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "steer_results.json").write_text(json.dumps(results, indent=1))
    print("\nwrote runs/steer_results.json")


def _best_coef(by):
    best, bc = -1, 8.0
    for k, v in by.items():
        if k.startswith("steer_true@"):
            if v["naming_acc"] > best and v["parse"] >= 0.7:
                best = v["naming_acc"]; bc = float(k.split("@")[1])
    return bc


if __name__ == "__main__":
    main()
