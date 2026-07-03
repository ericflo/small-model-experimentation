#!/usr/bin/env python3
"""Anatomy of the generation wall: is it COVERAGE (right program never proposed) or SELECTION (proposed but
not selected)? Per depth/family, draw K identification samples, grade each vs visible+hidden, then compare
first@1, coverage@k (oracle ceiling = sample-more), vfilter (majority-behavior), and mverify (model verifier)
against the coverage ceiling. See reports/prereg.md."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from math import comb
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def ident_prompt(fam, t):
    # No op-menu: matches the canonical C13/C16 bare-identification numbers; the menu sends the model
    # into enumeration rabbit-holes and lowers code-emission rate.
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `{fam['sig']}` reproducing this for all such inputs. Only a ```python code block.")


def judge_task_text(fam, t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the exact rule `transform` (a composition of {', '.join(fam['prims'])}) that maps input "
            f"to output for ALL inputs following the same rule.\nExamples:\n{lines}")


EXEC_TMPL = """import json
{code}
vis_in = {vis_in!r}
vis_exp = {vis_exp!r}
hid_in = {hid_in!r}
hid_exp = {hid_exp!r}
def outs(inputs):
    r = []
    for x in inputs:
        try:
            r.append(repr(transform(x)))
        except BaseException as e:
            r.append("ERR:" + type(e).__name__)
    return r
vo = outs(vis_in); ho = outs(hid_in)
vis_pass = all(a == repr(b) for a, b in zip(vo, vis_exp))
hid_pass = all(a == repr(b) for a, b in zip(ho, hid_exp))
print(json.dumps({{"vis": vis_pass, "hid": hid_pass, "sig": ho}}))
"""


def grade_candidate(code, t):
    """Return (visible_pass, full_pass, behavior_signature_on_hidden_inputs) or (False, False, None)."""
    if not code:
        return (False, False, None)
    safe, _ = E.static_safety_check(code)
    if not safe:
        return (False, False, None)
    script = EXEC_TMPL.format(
        code=code,
        vis_in=[e["input"] for e in t["visible"]], vis_exp=[e["output"] for e in t["visible"]],
        hid_in=[e["input"] for e in t["hidden"]], hid_exp=[e["output"] for e in t["hidden"]])
    r = E.run_python_script(script, timeout_s=5.0)
    if not r["ok"]:
        return (False, False, None)
    try:
        p = json.loads(r["stdout"].strip().splitlines()[-1])
    except Exception:
        return (False, False, None)
    vis, hid = bool(p["vis"]), bool(p["hid"])
    return (vis, vis and hid, tuple(p["sig"]) if vis else None)


def coverage_at_k(full_flags, k):
    """Unbiased pass@k over n samples with c correct."""
    n, c = len(full_flags), sum(full_flags)
    if k > n:
        k = n
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="+", default=["list", "register"])
    ap.add_argument("--n-per-depth", type=int, default=20)
    ap.add_argument("--depths", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--judge-budget", type=int, default=512)
    ap.add_argument("--max-verify", type=int, default=8, help="cap unique visible-passers scored by verifier")
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_per_depth, args.depths, args.K = 4, [1, 2], 12

    import gen_lib as GL
    p = GL.Probe()
    rng = random.Random(args.seed)
    all_recs = []

    for fname in args.families:
        fam = FAM.FAMILIES[fname]
        tasks = []
        for d in args.depths:
            made = 0
            while made < args.n_per_depth:
                t = FAM.make_task(fam, len(tasks), d, rng, k_visible=8, m_hidden=8)
                if t:
                    tasks.append(t); made += 1
        (EXP / "data").mkdir(exist_ok=True)
        (EXP / "data" / f"tasks_{fname}.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
        print(f"[{fname}] {len(tasks)} verified tasks x K={args.K}", flush=True)
        t0 = time.time()

        # --- sample K identification candidates per task ---
        prompts = [p.prompt(ident_prompt(fam, t), enable_thinking=True) for t in tasks]
        rep = [pr for pr in prompts for _ in range(args.K)]
        gens = p.gen_sequences(rep, think=True, budget=args.budget, greedy=False,
                               answer_max=420, batch_size=48)
        print(f"  [{fname}] sampling done [{time.time()-t0:.0f}s]", flush=True)

        # --- extract + grade (dedup by code; thread-pool the sandbox) ---
        per_task_codes = []
        cache = {}
        for ti, t in enumerate(tasks):
            codes = []
            for j in range(args.K):
                seq = gens[ti * args.K + j]["seq_ids"]
                txt = p.tok.decode(seq[len(p._ids(rep[ti * args.K + j])):], skip_special_tokens=False)
                txt = txt.split("</think>")[-1] if "</think>" in txt else txt
                c, _ = E.extract_candidate_code(txt, "transform")
                codes.append(c or "")
            per_task_codes.append(codes)

        uniq = {}
        for ti, t in enumerate(tasks):
            for c in per_task_codes[ti]:
                uniq.setdefault((ti, c), None)
        keys = list(uniq)
        with ThreadPoolExecutor(max_workers=16) as ex:
            results = list(ex.map(lambda k: grade_candidate(k[1], tasks[k[0]]), keys))
        for k, r in zip(keys, results):
            cache[k] = r
        print(f"  [{fname}] graded {len(keys)} unique [{time.time()-t0:.0f}s]", flush=True)

        # --- verifier scoring over unique visible-passers (capped) ---
        judge_items = []  # (ti, code)
        vpass_by_task = []
        for ti, t in enumerate(tasks):
            seen, vps = set(), []
            for c in per_task_codes[ti]:
                if c in seen:
                    continue
                seen.add(c)
                vis, full, sig = cache[(ti, c)]
                if vis:
                    vps.append((c, full))
            # cap by frequency (most-sampled visible-passers first)
            freq = Counter(per_task_codes[ti])
            vps.sort(key=lambda cf: -freq[cf[0]])
            vpass_by_task.append(vps)
            for c, _ in vps[:args.max_verify]:
                judge_items.append((ti, c))
        jscore = {}
        if judge_items:
            jprompts = [p.judge_prompt(judge_task_text(fam, tasks[ti]), c, enable_thinking=True)
                        for ti, c in judge_items]
            pa, _ = p.judge_think(jprompts, budget=args.judge_budget, gen_batch=32, logit_batch=16)
            for (ti, c), s in zip(judge_items, pa):
                jscore[(ti, c)] = s
        print(f"  [{fname}] judged {len(judge_items)} [{time.time()-t0:.0f}s]", flush=True)

        # --- per-task metrics ---
        for ti, t in enumerate(tasks):
            full = [cache[(ti, c)][1] for c in per_task_codes[ti]]
            vis = [cache[(ti, c)][0] for c in per_task_codes[ti]]
            sigs = [cache[(ti, c)][2] for c in per_task_codes[ti]]
            vps = vpass_by_task[ti]
            n_vp = len(vps)
            n_vp_full = sum(1 for _, f in vps if f)
            # vfilter: majority behavior-signature among ALL visible-passing samples
            vf_full = False
            vp_sigs = [(sigs[j], full[j]) for j in range(args.K) if vis[j] and sigs[j] is not None]
            if vp_sigs:
                cnt = Counter(s for s, _ in vp_sigs)
                best_sig = cnt.most_common(1)[0][0]
                vf_full = next(f for s, f in vp_sigs if s == best_sig)
            # mverify: highest verifier score among (capped) unique visible-passers
            mv_full = False
            scored = [(jscore.get((ti, c)), c, f) for c, f in vps[:args.max_verify] if (ti, c) in jscore]
            if scored:
                scored.sort(key=lambda x: -(x[0] if x[0] is not None else -1))
                mv_full = scored[0][2]
            all_recs.append({
                "family": fname, "task_id": t["task_id"], "depth": t["depth"],
                "first1": bool(full[0]), "cov_full": sum(full), "K": args.K,
                "n_vpass": n_vp, "n_vpass_full": n_vp_full,
                "vfilter_full": bool(vf_full), "mverify_full": bool(mv_full),
                "any_full": any(full),
            })

    (EXP / "runs").mkdir(exist_ok=True)
    (EXP / "runs" / "anatomy.json").write_text(json.dumps({"K": args.K, "records": all_recs}, indent=1))

    # --- console summary ---
    print("\n=== wall anatomy (per family x depth) ===")
    by = defaultdict(list)
    for r in all_recs:
        by[(r["family"], r["depth"])].append(r)
    print(f"{'fam':>9} {'d':>2} {'n':>3} {'first@1':>8} {'vfilter':>8} {'mverify':>8} "
          f"{'cov@K':>7} {'rndVP':>7}")
    for (fam, d) in sorted(by):
        rs = by[(fam, d)]
        n = len(rs)
        f1 = sum(r["first1"] for r in rs) / n
        vf = sum(r["vfilter_full"] for r in rs) / n
        mv = sum(r["mverify_full"] for r in rs) / n
        cov = sum(coverage_at_k([True] * r["cov_full"] + [False] * (r["K"] - r["cov_full"]), r["K"]) for r in rs) / n
        rnd = sum((r["n_vpass_full"] / r["n_vpass"]) if r["n_vpass"] else 0.0 for r in rs) / n
        print(f"{fam:>9} {d:>2} {n:>3} {f1:>8.2f} {vf:>8.2f} {mv:>8.2f} {cov:>7.2f} {rnd:>7.2f}")
    print("wrote runs/anatomy.json")


if __name__ == "__main__":
    main()
