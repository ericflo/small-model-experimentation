#!/usr/bin/env python3
"""Build data for the 3 arms extending C23. Explorer = CPU interpreter brute-search (no external model).
Arm1: extend depth-3 to 2560 (nest C23's 640, EXCLUDE the C23 depth-3 held-out functions -> 0 leakage).
Arm2: upsampled-40 control (40 distinct depth-3 pairs x16 = 640 examples, matched size/mixture to train_640).
Arm3: harvest depth-4 (exclude a fresh depth-4 held-out set) for the next-rung test.
Reuses C23's depth-3 frozen held-out + banked_640; produces all train files + the depth-4 held-out."""
from __future__ import annotations

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
from tool_harvest import tool_search  # noqa: E402

fam = FAM.FAMILIES["list"]
DOSE = EXP.parent / "qwen35_4b_depth3_dose_response"
WALL = EXP.parent / "qwen35_4b_wall_climbing"
PROBE = [[((i * 7 + j * 5) % 19) - 9 for j in range(4 + (i % 5))] for i in range(24)]


def parse(tops):
    return [(s.split("(")[0], int(s[s.index("(") + 1:-1]) if "(" in s else None) for s in tops]


def fsig(tops):
    out = []
    for x in PROBE:
        st = list(x)
        for op, k in parse(tops):
            st = FAM.apply_op(fam, op, k, st)
            if st is None:
                break
        out.append(repr(st))
    return tuple(out)


def main():
    t0 = time.time()
    (EXP / "data").mkdir(exist_ok=True)
    d12 = [json.loads(l) for l in (WALL / "data" / "train.jsonl").read_text().splitlines() if l.strip()]

    # --- reuse C23 depth-3 held-out; build its exclusion signatures ---
    heldout3 = [json.loads(l) for l in (DOSE / "data" / "eval_frozen.jsonl").read_text().splitlines() if l.strip()]
    (EXP / "data" / "eval_frozen_d3.jsonl").write_text("\n".join(json.dumps(t) for t in heldout3) + "\n")
    excl3_f = {fsig(t["target_ops"]) for t in heldout3}
    excl3_o = {tuple(t["target_ops"]) for t in heldout3}

    # --- Arm 1: extend depth-3 to 2560, nesting C23's 640, excluding held-out ---
    c23_d3 = [json.loads(l) for l in (DOSE / "data" / "tool_depth3.jsonl").read_text().splitlines() if l.strip()]
    d3 = [{"prompt": x["prompt"], "code": x["code"], "depth": 3} for x in c23_d3]  # first 640 (C23's, already 0-leak)
    seen_ops = {tuple(x["found_ops"]) for x in c23_d3}
    rng = random.Random(555); tid = 100_000
    while len(d3) < 2560 and tid < 100_000 + 200_000:
        tid += 1
        t = FAM.make_task(fam, tid, 3, rng, k_visible=8, m_hidden=8)
        if not t or fsig(t["target_ops"]) in excl3_f or tuple(t["target_ops"]) in excl3_o:
            continue
        found = tool_search(fam, t, 3)
        if found is None:
            continue
        fo = tuple(FAM.op_repr(o, k) for o, k in found)
        if fo in seen_ops:
            continue
        seen_ops.add(fo)
        d3.append({"prompt": C.ident_prompt(fam, t), "code": FAM.reference_code(fam, found), "depth": 3})
    (EXP / "data" / "tool_depth3_2560.jsonl").write_text("\n".join(json.dumps(x) for x in d3) + "\n")
    print(f"[arm1] depth-3 pairs: {len(d3)} (first 640 = C23's, extended to 2560, 0-leak vs held-out) [{time.time()-t0:.0f}s]", flush=True)

    def write_train(name, extra):
        rng2 = random.Random(7); comb = d12 + extra; rng2.shuffle(comb)
        (EXP / "data" / name).write_text("\n".join(json.dumps(x) for x in comb) + "\n")
        return len(comb)

    for N in (1280, 2560):
        n = write_train(f"train_{N}.jsonl", d3[:N])
        print(f"[arm1] train_{N}.jsonl: {n} pairs (d3={N})", flush=True)

    # --- Arm 2: upsampled-40 (40 distinct x16 = 640 examples), matched to train_640 ---
    up = d3[:40] * 16  # 640 depth-3 examples from 40 distinct
    n = write_train("train_up40.jsonl", up)
    print(f"[arm2] train_up40.jsonl: {n} pairs (40 distinct d3 x16 = {len(up)}; matched size/mixture to train_640)", flush=True)

    # --- Arm 3: depth-4 held-out + training ---
    # held-out first (fresh), then harvest training excluding it
    rng4 = random.Random(909); tid4 = 300_000; heldout4 = []
    while len(heldout4) < 60 and tid4 < 300_000 + 300_000:
        tid4 += 1
        t = FAM.make_task(fam, tid4, 4, rng4, k_visible=8, m_hidden=8)
        if t:
            heldout4.append(t)
    excl4_f = {fsig(t["target_ops"]) for t in heldout4}
    excl4_o = {tuple(t["target_ops"]) for t in heldout4}
    (EXP / "data" / "eval_frozen_d4.jsonl").write_text("\n".join(json.dumps(t) for t in heldout4) + "\n")
    print(f"[arm3] depth-4 held-out: {len(heldout4)} tasks [{time.time()-t0:.0f}s]", flush=True)

    rng4b = random.Random(444); tid4b = 400_000; d4 = []; seen4 = set()
    while len(d4) < 320 and tid4b < 400_000 + 400_000:
        tid4b += 1
        t = FAM.make_task(fam, tid4b, 4, rng4b, k_visible=8, m_hidden=8)
        if not t or fsig(t["target_ops"]) in excl4_f or tuple(t["target_ops"]) in excl4_o:
            continue
        found = tool_search(fam, t, 4)
        if found is None:
            continue
        fo = tuple(FAM.op_repr(o, k) for o, k in found)
        if fo in seen4:
            continue
        seen4.add(fo)
        d4.append({"prompt": C.ident_prompt(fam, t), "code": FAM.reference_code(fam, found), "depth": 4})
    (EXP / "data" / "tool_depth4.jsonl").write_text("\n".join(json.dumps(x) for x in d4) + "\n")
    # banked_d4 = depth-1+2 + 640 depth-3 (scaffold) + depth-4
    n = write_train("train_d4.jsonl", d3[:640] + d4)
    print(f"[arm3] depth-4 tool pairs: {len(d4)}; train_d4.jsonl: {n} pairs (d12 + 640 d3 + {len(d4)} d4) [{time.time()-t0:.0f}s]", flush=True)

    # train_tasks superset (for eval dedup): all d3-2560 tasks + d4 tasks + d12 tasks.
    # (eval_ladder dedups vs data/train_tasks_{d3,d4}.jsonl; write both here as target_ops lists.)
    # For depth-3 eval we reuse C23 held-out (already 0-leak). For depth-4 eval we generated held-out disjoint by construction.
    print(f"done [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
