#!/usr/bin/env python3
"""Truth-BLIND factorized hypothesize-and-verify CoT traces for the dsl_sft arm (CPU only).

The generator implements the SAME procedure as configs/scaffold_prompt.txt, programmatically:
  1. EVIDENCE   -- length/sign/order/duplicate deltas between input and output of example 1.
  2. SHORTLIST  -- per-stage candidate ops from a FIXED feature->candidate rulebook that is a pure
                   function of visible I/O (never reads the true pipeline).
  3. COMPOSE+CHECK -- candidates composed from the shortlists in fixed rulebook order, each
                   executed on example 1 via the families.py interpreter with intermediate states
                   shown and a MATCH/no verdict; first ex1-pass is verified on examples 2-3.
  <= max_checks_per_trace candidate checks; tasks the blind procedure cannot solve are dropped
  (C45 cot->None pattern, rate reported). Kept traces must also solve the FULL task (visible +
  hidden) -- the train-data purity filter.

BLINDNESS: generate_trace() takes ONLY (family name, visible examples). Verified at build time by
regenerating traces from task copies with the true-ops fields DELETED (access would raise) and
byte-comparing.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hv_common as C  # noqa: E402
from hv_common import FAM  # noqa: E402

EXP = C.EXP
MAX_CHECKS_DEFAULT = 6
VOWELS = set("aeiou")


# ---- feature helpers (pure functions of visible I/O) ---------------------------------------
def _uniq(seq):
    return list(dict.fromkeys(seq))


def _dedup_adj(seq):
    return [v for i, v in enumerate(seq) if i == 0 or v != seq[i - 1]]


def op_str(op, k):
    return f"{op}({k})" if k is not None else op


def fmt(v):
    return repr(v)


# ---- fixed rulebook: single-stage shortlist (x -> y), exact-feature entries first, coarse
# fallbacks after; order is the fixed code order below (truth-independent) --------------------
def list_d1_shortlist(x, y):
    c = []
    n, m = len(x), len(y)
    if m == n:
        if sorted(x) == sorted(y) and x != y:
            if y == sorted(x):
                c.append(("sort_asc", None))
            if y == sorted(x, reverse=True):
                c.append(("sort_desc", None))
            if y == x[::-1]:
                c.append(("reverse", None))
            for r in (1, 2, 3):
                if n and y == x[r % n:] + x[:r % n]:
                    c.append(("rotate_k", r))
        if any(x) and y == [-v for v in x]:
            c.append(("negate", None))
        if any(v < 0 for v in x) and y == [abs(v) for v in x]:
            c.append(("abs_all", None))
        if y == [v * v for v in x]:
            c.append(("square", None))
        d = {b - a for a, b in zip(x, y)}
        if len(d) == 1:
            dv = d.pop()
            if dv in (-3, -2, -1, 1, 2, 3):
                c.append(("add_k", dv))
        for k in (-2, 2, 3):
            if y == [v * k for v in x]:
                c.append(("mul_k", k))
        for k in (2, 3, 4):
            if y == [v % k for v in x]:
                c.append(("mod_k", k))
        if n and y == [sum(x[:i + 1]) for i in range(n)]:
            c.append(("running_sum", None))
        # coarse rearrangement fallbacks (may fail the check -- that is the point)
        if sorted(x) == sorted(y) and x != y:
            c.append(("sort_asc", None))
            c.append(("reverse", None))
    elif m < n:
        if _uniq(x) != x and y == _uniq(x):
            c.append(("unique_stable", None))
        if _dedup_adj(x) != x and y == _dedup_adj(x):
            c.append(("dedup_adjacent", None))
        if 1 <= m <= 4 and y == x[:m]:
            c.append(("take_k", m))
        if 1 <= n - m <= 3 and y == x[n - m:]:
            c.append(("drop_k", n - m))
        if m == n - 1 and y == [x[i + 1] - x[i] for i in range(n - 1)]:
            c.append(("adjacent_diff", None))
        # coarse length-arithmetic fallbacks
        if _uniq(x) != x:
            c.append(("unique_stable", None))
        if _dedup_adj(x) != x:
            c.append(("dedup_adjacent", None))
        if 1 <= m <= 4:
            c.append(("take_k", m))
        if 1 <= n - m <= 3:
            c.append(("drop_k", n - m))
        if m == n - 1:
            c.append(("adjacent_diff", None))
    out = []
    for cand in c:
        if cand not in out:
            out.append(cand)
    return out[:4]


def string_d1_shortlist(x, y):
    c = []
    n, m = len(x), len(y)
    if m == n:
        if sorted(x) == sorted(y) and x != y:
            if y == "".join(sorted(x)):
                c.append(("sort_chars", None))
            if y == x[::-1]:
                c.append(("reverse", None))
            if y == "".join(x[i:i + 2][::-1] for i in range(0, n, 2)):
                c.append(("swap_pairs", None))
            for r in (1, 2, 3):
                if n and y == x[r % n:] + x[:r % n]:
                    c.append(("rotate_k", r))
        for k in (1, 2, 3, 13):
            if y == "".join(chr((ord(ch) - 97 + k) % 26 + 97) for ch in x):
                c.append(("shift_k", k))
        if sorted(x) == sorted(y) and x != y:
            c.append(("reverse", None))
            c.append(("sort_chars", None))
    elif m < n:
        if y == "".join(_dedup_adj(list(x))) and _dedup_adj(list(x)) != list(x):
            c.append(("dedup_adjacent", None))
        if y == "".join(dict.fromkeys(x)) and len(set(x)) < n:
            c.append(("dedup_all", None))
        if any(ch in VOWELS for ch in x) and y == "".join(ch for ch in x if ch not in VOWELS):
            c.append(("remove_vowels", None))
        if 1 <= m <= 4 and y == x[:m]:
            c.append(("take_k", m))
        if 1 <= n - m <= 3 and y == x[n - m:]:
            c.append(("drop_k", n - m))
        for k in (2, 3):
            if y == x[::k]:
                c.append(("keep_every_k", k))
        # coarse fallbacks
        if any(ch in VOWELS for ch in x):
            c.append(("remove_vowels", None))
        if len(set(x)) < n:
            c.append(("dedup_all", None))
        if _dedup_adj(list(x)) != list(x):
            c.append(("dedup_adjacent", None))
        if 1 <= m <= 4:
            c.append(("take_k", m))
        if 1 <= n - m <= 3:
            c.append(("drop_k", n - m))
    else:  # grew
        if y == "".join(ch * 2 for ch in x):
            c.append(("double", None))
        for k in (2, 3):
            if y == x * k:
                c.append(("repeat_k", k))
        if m == 2 * n:
            c.append(("double", None))
            c.append(("repeat_k", 2))
        if n and m == 3 * n:
            c.append(("repeat_k", 3))
    out = []
    for cand in c:
        if cand not in out:
            out.append(cand)
    return out[:4]


# ---- fixed rulebook: FIRST-op proposals for two-stage pipelines (composite evidence) --------
def list_stage1(x, y):
    c = []
    n, m = len(x), len(y)
    if _dedup_adj(x) != x:
        c.append(("dedup_adjacent", None))
    if _uniq(x) != x:
        c.append(("unique_stable", None))
    if any(v < 0 for v in x) and all(v >= 0 for v in y):
        c.append(("abs_all", None))
        c.append(("square", None))
    if m > 1 and y == sorted(y):
        c.append(("sort_asc", None))
    if m > 1 and y == sorted(y, reverse=True):
        c.append(("sort_desc", None))
    if m < n and 1 <= m <= 4:
        c.append(("take_k", m))
    if m < n and 1 <= n - m <= 3:
        c.append(("drop_k", n - m))
    if m == n - 1:
        c.append(("adjacent_diff", None))
    c.append(("reverse", None))
    if m == n:
        c.append(("negate", None))
        c.append(("running_sum", None))
    out = []
    for cand in c:
        if cand not in out:
            out.append(cand)
    return out[:4]


def string_stage1(x, y):
    c = []
    n, m = len(x), len(y)
    if _dedup_adj(list(x)) != list(x):
        c.append(("dedup_adjacent", None))
    if len(set(x)) < n:
        c.append(("dedup_all", None))
    if any(ch in VOWELS for ch in x) and not any(ch in VOWELS for ch in y):
        c.append(("remove_vowels", None))
    if m > 1 and y == "".join(sorted(y)):
        c.append(("sort_chars", None))
    if m > n:
        c.append(("double", None))
        if n and m % n == 0 and m // n in (2, 3):
            c.append(("repeat_k", m // n))
    if m < n and 1 <= m <= 4:
        c.append(("take_k", m))
    if m < n and 1 <= n - m <= 3:
        c.append(("drop_k", n - m))
    if m < n and n and m == len(x[::2]):
        c.append(("keep_every_k", 2))
    c.append(("reverse", None))
    out = []
    for cand in c:
        if cand not in out:
            out.append(cand)
    return out[:4]


D1 = {"list": list_d1_shortlist, "string": string_d1_shortlist}
S1 = {"list": list_stage1, "string": string_stage1}


# ---- evidence text (pure function of example 1) ---------------------------------------------
def evidence_text(famname, x, y):
    n, m = len(x), len(y)
    obs = [f"length {n} -> {m} ({'shrank' if m < n else 'grew' if m > n else 'unchanged'})"]
    if famname == "list":
        if m == n and sorted(x) == sorted(y) and x != y:
            obs.append("output is a rearrangement of the input values")
        if m and y == sorted(y):
            obs.append("output is sorted ascending")
        if any(v < 0 for v in x) and all(v >= 0 for v in y):
            obs.append("negatives in input, none in output")
        if _uniq(x) != x:
            obs.append("input has duplicate values")
        if m == n and sorted(x) != sorted(y):
            obs.append("values changed elementwise")
    else:
        if m == n and sorted(x) == sorted(y) and x != y:
            obs.append("output is a rearrangement of the same characters")
        if m and y == "".join(sorted(y)):
            obs.append("output characters are sorted")
        if any(ch in VOWELS for ch in x) and not any(ch in VOWELS for ch in y):
            obs.append("vowels present in input, absent in output")
        if len(set(x)) < n:
            obs.append("input has repeated characters")
        if m > n:
            obs.append("characters were added or repeated")
    return "; ".join(obs)


# ---- the trace generator (BLIND: only family name + visible examples) -----------------------
def generate_trace(famname, visible, max_checks=MAX_CHECKS_DEFAULT):
    """Return (trace_text, ops) for the first pipeline passing ex1 and verifying on ex2-3,
    or None if the blind procedure cannot solve the task within max_checks candidate checks."""
    fam = C.fam_of(famname)
    x1, y1 = visible[0]["input"], visible[0]["output"]
    lines = [f"EVIDENCE (example 1): {fmt(x1)} -> {fmt(y1)}: {evidence_text(famname, x1, y1)}."]
    checks = 0
    cand_no = 0

    def verify(ops):
        vlines = []
        for j in (1, 2):
            if j >= len(visible):
                break
            ex = visible[j]
            out = C.exec_seq(fam, ops, ex["input"])
            ok = out == ex["output"]
            got = "error" if out is None else fmt(out)
            vlines.append(f"VERIFY example {j + 1}: {fmt(ex['input'])} -> {got} | expected "
                          f"{fmt(ex['output'])} -> {'MATCH' if ok else 'no'}")
            if not ok:
                return vlines, False
        return vlines, True

    # stage A: single-op candidates from the one-step shortlist
    single = D1[famname](x1, y1)
    lines.append("SHORTLIST (one step): " + (", ".join(op_str(o, k) for o, k in single) if single else "(none)"))
    for op, k in single[:2]:
        if checks >= max_checks:
            return None
        checks += 1
        cand_no += 1
        out = FAM.apply_op(fam, op, k, x1)
        got = "error" if out is None else fmt(out)
        ok = out == y1
        lines.append(f"CANDIDATE {cand_no}: {op_str(op, k)} | {fmt(x1)} -> {got} | expected {fmt(y1)} "
                     f"-> {'MATCH' if ok else 'no'}")
        if ok:
            vlines, vok = verify([(op, k)])
            lines += vlines
            if vok:
                lines.append(f"PIPELINE: {op_str(op, k)}")
                return "\n".join(lines), [(op, k)]
            lines.append("Discard this candidate; the pipeline must explain every example.")

    # stage B: two-stage pipelines -- first-op proposals, then a shortlist on the residual
    firsts = S1[famname](x1, y1)
    lines.append("SHORTLIST (first op of two): " + (", ".join(op_str(o, k) for o, k in firsts) if firsts else "(none)"))
    for fop, fk in firsts:
        if checks >= max_checks:
            return None
        mid = FAM.apply_op(fam, fop, fk, x1)
        if mid is None or mid == x1:
            continue
        seconds = D1[famname](mid, y1)
        for sop, sk in seconds[:2]:
            if checks >= max_checks:
                return None
            checks += 1
            cand_no += 1
            out = FAM.apply_op(fam, sop, sk, mid)
            got = "error" if out is None else fmt(out)
            ok = out == y1
            lines.append(f"CANDIDATE {cand_no}: {op_str(fop, fk)} -> {op_str(sop, sk)} | {fmt(x1)} -> "
                         f"{fmt(mid)} -> {got} | expected {fmt(y1)} -> {'MATCH' if ok else 'no'}")
            if ok:
                ops = [(fop, fk), (sop, sk)]
                vlines, vok = verify(ops)
                lines += vlines
                if vok:
                    lines.append(f"PIPELINE: {op_str(fop, fk)} -> {op_str(sop, sk)}")
                    return "\n".join(lines), ops
                lines.append("Discard this candidate; the pipeline must explain every example.")
    return None


def trace_format_ok(text):
    """Programmatic checker for the taught trace format (also the install-gate checker)."""
    return ("EVIDENCE" in text and "SHORTLIST" in text and "CANDIDATE" in text
            and ("MATCH" in text or "-> no" in text))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--target", type=int, default=None)
    args = ap.parse_args()
    cfg = C.load_cfg()
    sfx = C.sfx(args.smoke)
    out_path = EXP / "data" / f"traces{sfx}.jsonl"
    if out_path.exists():
        print(f"[gen_traces] {out_path.name} exists; skip", flush=True)
        return
    target = args.target or (12 if args.smoke else cfg["sft"]["trace_target_n"])
    max_checks = cfg["sft"]["max_checks_per_trace"]

    tasks = C.load_jsonl(EXP / "data" / f"sft_tasks{sfx}.jsonl")
    order = list(range(len(tasks)))
    random.Random(4242).shuffle(order)  # balanced family/depth mixture, fixed seed

    kept, n_unsolved, n_impure = [], 0, 0
    cell = {}
    for i in order:
        if len(kept) >= target:
            break
        t = tasks[i]
        fam = C.fam_of(t["family"])
        res = generate_trace(t["family"], t["visible"], max_checks)
        if res is None:
            n_unsolved += 1
            continue
        trace, ops = res
        # purity filter: the final candidate must solve the FULL task (visible + hidden)
        if not (C.solves(fam, ops, t["visible"]) and C.solves(fam, ops, t["hidden"])):
            n_impure += 1
            continue
        code = FAM.reference_code(fam, ops)
        kept.append({"prompt": C.ident_prompt(fam, t), "think": trace,
                     "answer": f"```python\n{code}\n```",
                     "family": t["family"], "depth": t["depth"],
                     "ops": [[op, k] for op, k in ops],
                     "true_match": [list(p) for p in ops] == t["ops"]})
        key = (t["family"], t["depth"])
        cell[key] = cell.get(key, 0) + 1

    n_seen = n_unsolved + n_impure + len(kept)
    print(f"[gen_traces] tasks seen {n_seen} | kept {len(kept)} | unsolved(dropped) {n_unsolved} "
          f"({n_unsolved / max(1, n_seen):.2f}) | impure(dropped) {n_impure}", flush=True)
    for key in sorted(cell):
        print(f"[gen_traces]   kept {key[0]} d{key[1]}: {cell[key]}", flush=True)
    tm = sum(r["true_match"] for r in kept)
    lens = sorted(len(r["think"]) for r in kept)
    if kept:
        print(f"[gen_traces] final-candidate == true pipeline on {tm}/{len(kept)} kept traces "
              f"(others = behavioral equivalents that also pass hidden)", flush=True)
        print(f"[gen_traces] trace chars: min {lens[0]} med {lens[len(lens) // 2]} max {lens[-1]} "
              f"(~{lens[len(lens) // 2] // 4} tokens median vs think budget {cfg['eval']['budget']})", flush=True)

    # ---- BLINDNESS VERIFICATION: regenerate through oracle-deleted task copies, byte-compare --
    idx_by_prompt = {r["prompt"]: r for r in kept[:20]}
    checked = 0
    for t in tasks:
        pr = C.ident_prompt(C.fam_of(t["family"]), t)
        if pr not in idx_by_prompt:
            continue
        blind = {k: v for k, v in t.items() if k not in ("ops", "target_ops")}  # access would raise
        res = generate_trace(blind["family"], blind["visible"], max_checks)
        assert res is not None, "blind regen failed to solve a previously-solved task"
        assert res[0] == idx_by_prompt[pr]["think"], "trace differs with oracle blinded -- NOT truth-blind"
        checked += 1
    print(f"[gen_traces] blindness check: {checked} traces regenerated with true-ops deleted; byte-identical", flush=True)

    C.dump_jsonl(out_path, kept)
    print(f"[gen_traces] wrote {out_path.name} ({len(kept)} traces)", flush=True)


if __name__ == "__main__":
    main()
