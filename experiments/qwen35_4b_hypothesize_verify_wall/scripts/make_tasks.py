#!/usr/bin/env python3
"""Frozen eval + SFT-train + install-gate task builder (CPU only, fully deterministic).

Eval: per family in {list,string} x depth in {2,3}, n_per_cell min-depth-verified tasks (seed 71),
each with 8 visible + 6 hidden examples + 6 fresh PROBE INPUTS (drawn from the same input
distribution, deduped against visible+hidden inputs, executable under the true pipeline; INPUTS
only are stored -- probe labels are computed at eval time and never enter any prompt).

SFT tasks: depths {1,2} (seed 303), op-TYPE-SEQUENCE deduped against ALL eval tasks (0 exact-
skeleton overlap, asserted). Gate tasks: 20 held-out depth-1 tasks (seed 777) for the install gate.

Pipelines are serialized as [op-name, param] lists so families.py can rebuild them exactly.
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


def build_task(fam, tid, depth, rng, n_probe, need_probes=True):
    """One min-depth-verified task with serialized ops (+ fresh probe inputs if requested)."""
    t = FAM.make_task(fam, tid, depth, rng, k_visible=8, m_hidden=6)
    if not t:
        return None
    inp = [e["input"] for e in t["visible"] + t["hidden"]]
    out = [e["output"] for e in t["visible"] + t["hidden"]]
    if depth > 1 and FAM.min_depth_leq(fam, inp, out, depth - 1):
        return None  # C13 discipline (same re-check as cross_substrate.py)
    ops = [C.parse_op_repr(s) for s in t["target_ops"]]
    t["ops"] = [[op, k] for op, k in ops]
    if need_probes:
        seen = {FAM._key(e["input"]) for e in t["visible"] + t["hidden"]}
        probes = []
        for _ in range(600):
            if len(probes) >= n_probe:
                break
            x = fam["mk_input"](rng)
            key = FAM._key(x)
            if key in seen:
                continue
            if C.exec_seq(fam, ops, x) is None:
                continue  # probe labels must be computable via the true pipeline
            seen.add(key)
            probes.append(x)
        if len(probes) < n_probe:
            return None
        t["probe_inputs"] = probes
    return t


def gen_cell(family, depth, n, seed, n_probe, need_probes=True, skip_skels=None, max_tries=300_000):
    fam = C.fam_of(family)
    rng = random.Random(seed)
    tasks, tid = [], 0
    while len(tasks) < n and tid < max_tries:
        tid += 1
        t = build_task(fam, tid, depth, rng, n_probe, need_probes)
        if not t:
            continue
        if skip_skels is not None and (family, tuple(C.true_types(t))) in skip_skels:
            continue
        tasks.append(t)
    if len(tasks) < n:
        raise RuntimeError(f"could not generate {n} tasks for {family} d{depth} (got {len(tasks)})")
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = C.load_cfg()
    sfx = C.sfx(args.smoke)
    eval_path = EXP / "data" / f"eval_tasks{sfx}.jsonl"
    sft_path = EXP / "data" / f"sft_tasks{sfx}.jsonl"
    gate_path = EXP / "data" / f"gate_d1{sfx}.jsonl"
    if eval_path.exists() and sft_path.exists() and gate_path.exists():
        print(f"[make_tasks] all outputs exist ({eval_path.name}, {sft_path.name}, {gate_path.name}); skip", flush=True)
        return

    n_cell = 4 if args.smoke else cfg["eval"]["n_per_cell"]
    n_probe = cfg["eval"]["n_probe_inputs"]
    seed0 = cfg["eval"]["seed"]
    families = cfg["families"]
    depths = cfg["eval"]["depths"]

    # ---- frozen eval -----------------------------------------------------------------------
    eval_rows = []
    for fi, family in enumerate(families):
        for depth in depths:
            cell_seed = seed0 + fi * 10 + depth  # deterministic, cell-independent
            tasks = gen_cell(family, depth, n_cell, cell_seed, n_probe)
            eval_rows += tasks
            print(f"[make_tasks] eval {family} d{depth}: {len(tasks)} tasks (seed {cell_seed})", flush=True)
    eval_skels = {(t["family"], tuple(C.true_types(t))) for t in eval_rows}

    # ---- SFT train tasks (depth 1/2, skeleton-deduped vs ALL eval tasks) --------------------
    sft_seed = cfg["sft"]["trace_seed"]
    sft_counts = {1: 8, 2: 8} if args.smoke else {1: 350, 2: 900}
    sft_rows = []
    for fi, family in enumerate(families):
        for depth in cfg["sft"]["trace_depths"]:
            cell_seed = sft_seed + fi * 10 + depth
            tasks = gen_cell(family, depth, sft_counts[depth], cell_seed, n_probe,
                             need_probes=False, skip_skels=eval_skels)
            sft_rows += tasks
            print(f"[make_tasks] sft {family} d{depth}: {len(tasks)} tasks (seed {cell_seed})", flush=True)

    # leakage check: op-TYPE-SEQUENCE dedup vs ALL eval tasks (must be 0)
    overlap = sum((t["family"], tuple(C.true_types(t))) in eval_skels for t in sft_rows)
    print(f"[make_tasks] skeleton-leak check: {overlap} exact op-type-sequence overlaps "
          f"between {len(sft_rows)} SFT tasks and {len(eval_rows)} eval tasks", flush=True)
    assert overlap == 0, "SFT/eval skeleton leakage"

    # ---- install-gate tasks: held-out depth-1 (fresh instances, deploy-channel gate) --------
    n_gate = 2 if args.smoke else 10
    gate_rows = []
    for fi, family in enumerate(families):
        tasks = gen_cell(family, 1, n_gate, 777 + fi, n_probe, need_probes=False)
        gate_rows += tasks
    print(f"[make_tasks] gate d1: {len(gate_rows)} held-out depth-1 tasks (seed 777+)", flush=True)

    C.dump_jsonl(eval_path, eval_rows)
    C.dump_jsonl(sft_path, sft_rows)
    C.dump_jsonl(gate_path, gate_rows)
    print(f"[make_tasks] wrote {eval_path.name} ({len(eval_rows)}), {sft_path.name} ({len(sft_rows)}), "
          f"{gate_path.name} ({len(gate_rows)})", flush=True)


if __name__ == "__main__":
    main()
