#!/usr/bin/env python3
"""TOOL-SEARCH explorer (CPU-only, no external model): brute-enumerate the interpreter over the fixed 4B's
OWN primitive vocabulary to DISCOVER a verified depth-3 solution for each task — the depth-3 solutions that
monolithic sampling (C17/C21) harvests ~0 of. Render to {prompt, code} SFT pairs to seed banking.

Combines the discovered depth-3 pairs with C21's depth-1+2 self-harvested pairs so the ONLY variable vs C21
is the tool-seeded depth-3 pairs."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))
sys.path.insert(0, str(EXP / "src"))
import common as C  # noqa: E402
import families as FAM  # noqa: E402

FAM_L = FAM.FAMILIES["list"]
C21_TRAIN = EXP.parent / "qwen35_4b_wall_climbing" / "data" / "train.jsonl"  # depth-1+2 self-harvest


def tool_search(fam, task, max_depth=3, seen_cap=400_000):
    """Interpreter BFS over the op-space; return an op-sequence correct on ALL task examples (visible+hidden)
    within max_depth, or None. State = tuple of current values for every example input; target = all outputs.
    (The tool may use all task I/O to find a verified-correct program; held-out generalization is tested on
    DISJOINT tasks, never these.)"""
    inputs = [e["input"] for e in task["visible"]] + [e["input"] for e in task["hidden"]]
    outputs = [e["output"] for e in task["visible"]] + [e["output"] for e in task["hidden"]]
    start = tuple(FAM._key(x) for x in inputs)
    target = tuple(FAM._key(o) for o in outputs)
    if start == target:
        return []
    ops = FAM.all_ops(fam)
    frontier = {start: []}
    seen = {start}
    for _ in range(max_depth):
        nxt = {}
        for st, path in frontier.items():
            states = [list(s) if isinstance(s, tuple) else s for s in st]
            for op, k in ops:
                new = tuple(FAM._key(FAM.apply_op(fam, op, k, s)) for s in states)
                if any(x is None for x in new):
                    continue
                if new == target:
                    return path + [(op, k)]
                if new not in seen:
                    seen.add(new)
                    nxt[new] = path + [(op, k)]
        frontier = nxt
        if not frontier or len(seen) > seen_cap:
            break
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-depth3", type=int, default=120, help="depth-3 TRAIN tasks to solve via tool-search")
    ap.add_argument("--seed", type=int, default=202)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        args.n_depth3 = 8

    rng = random.Random(args.seed)
    t0 = time.time()
    tasks, pairs, solved_tasks, depths_found = [], [], [], []
    tries = 0
    while len(tasks) < args.n_depth3 and tries < args.n_depth3 * 20:
        tries += 1
        t = FAM.make_task(FAM_L, len(tasks), 3, rng, k_visible=8, m_hidden=8)
        if not t:
            continue
        tasks.append(t)
        found = tool_search(FAM_L, t, max_depth=3)
        if found is not None:
            depths_found.append(len(found))
            code = FAM.reference_code(FAM_L, found)
            pairs.append({"prompt": C.ident_prompt(FAM_L, t), "code": code, "depth": 3,
                          "found_ops": [FAM.op_repr(o, k) for o, k in found]})
            solved_tasks.append(t)
    (EXP / "data").mkdir(exist_ok=True)
    (EXP / "data" / "tool_tasks.jsonl").write_text("\n".join(json.dumps(t) for t in tasks) + "\n")
    (EXP / "data" / "tool_depth3.jsonl").write_text("\n".join(json.dumps(x) for x in pairs) + "\n")
    print(f"[tool-search] {len(pairs)}/{len(tasks)} depth-3 tasks solved by interpreter search "
          f"(mean found-depth {sum(depths_found)/max(1,len(depths_found)):.2f}) [{time.time()-t0:.1f}s]", flush=True)

    # sanity: reference code passes the task (visible+hidden) through the code sandbox, for all pairs
    import code_env as E
    bad = 0
    for x, t in zip(pairs, solved_tasks):
        r = E.execute_public_and_asserts(x["code"], C.to_public(t) if hasattr(C, "to_public") else
                                         [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]],
                                         [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]])
        if not r["full_pass"]:
            bad += 1
    print(f"[tool-search] sandbox oracle: {len(pairs) - bad}/{len(pairs)} pairs pass visible+hidden", flush=True)

    # combine with C21's depth-1+2 self-harvest
    base_pairs = []
    if C21_TRAIN.exists():
        base_pairs = [json.loads(l) for l in C21_TRAIN.read_text().splitlines() if l.strip()]
        print(f"[combine] + {len(base_pairs)} depth-1+2 self-harvest pairs from C21", flush=True)
    else:
        print("[combine] WARNING: C21 train.jsonl not found; banking depth-3 only", flush=True)
    combined = base_pairs + [{"prompt": x["prompt"], "code": x["code"], "depth": 3} for x in pairs]
    rng.shuffle(combined)
    (EXP / "data" / "train.jsonl").write_text("\n".join(json.dumps(x) for x in combined) + "\n")
    # train_tasks for eval-exclusion = tool depth-3 tasks (depth-1+2 tasks were a different exp's; disjoint by
    # construction, but include them if available)
    tt = tasks[:]
    c21_tasks = C21_TRAIN.parent / "train_tasks.jsonl"
    if c21_tasks.exists():
        tt += [json.loads(l) for l in c21_tasks.read_text().splitlines() if l.strip()]
    (EXP / "data" / "train_tasks.jsonl").write_text("\n".join(json.dumps(x) for x in tt) + "\n")
    from collections import Counter
    bd = Counter(x["depth"] for x in combined)
    print(f"[combine] {len(combined)} total SFT pairs by depth {dict(sorted(bd.items()))}", flush=True)
    print("wrote data/train.jsonl, data/tool_depth3.jsonl, data/train_tasks.jsonl")


if __name__ == "__main__":
    main()
