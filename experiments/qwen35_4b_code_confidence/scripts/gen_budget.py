#!/usr/bin/env python3
"""Escalation-arm generation: THINK-MODE MBPP candidate pools at several think
budgets, for the compute-optimal confidence policy (confidence_policy experiment).

The select+abstain half is settled (post-hoc, conf > majority at matched k). The
open question: for a low-confidence (abstained) task, is compute better spent on
MORE breadth at a low budget, or on MORE serial depth (a higher think budget) on
the SAME 4B? To answer it we need the same tasks solved at >=2 think budgets with
p_true. This generates, per budget, greedy + k think-mode samples per task, each
scored full_pass (hidden-assert execution), p_true (no-think C10 judge), and with
n_think recorded for token-matched compute accounting. Reuses the code_confidence
infra verbatim. Run under .venv (HF backend).
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP))
from src.coverage_utils import sampling_prompt, candidate_from_completion, mbpp_record  # noqa: E402
sys.path.insert(0, str(EXP / "src"))
import gen_lib as GL  # noqa: E402

OUT = EXP.parents[1] / "experiments" / "qwen35_4b_confidence_policy" / "runs"


def load_records(count, offset, visible_tests=1, timeout_s=5.0):
    """MBPP test records, robust to the default ('text') vs sanitized ('prompt')
    field name and string-or-list test_list."""
    import ast as _ast
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp")["test"]
    records = []
    for raw in list(ds)[offset: offset + count * 2]:
        tl = raw.get("test_list")
        if isinstance(tl, str):
            try: tl = _ast.literal_eval(tl)
            except Exception: continue
        imports = raw.get("test_imports")
        if isinstance(imports, str):
            try: imports = _ast.literal_eval(imports)
            except Exception: imports = []
        text = raw.get("prompt") or raw.get("text")
        if not text or not tl:
            continue
        adapted = {"task_id": raw["task_id"], "text": text, "test_list": tl,
                   "test_setup_code": "\n".join(imports or []), "code": raw.get("code", "")}
        rec = mbpp_record(adapted, "heldout", visible_tests, timeout_s)
        if rec is not None:
            records.append(rec)
        if len(records) >= count:
            break
    return records


def build_pool(p, records, prompts, budget, k, answer_max):
    greedy = p.gen_sequences(prompts, think=True, budget=budget, greedy=True,
                             answer_max=answer_max, batch_size=24)
    samp = p.gen_sequences([pr for pr in prompts for _ in range(k)], think=True,
                           budget=budget, greedy=False, answer_max=answer_max, batch_size=24)
    rows = []
    for ri, (rec, pr) in enumerate(zip(records, prompts)):
        plen = len(p._ids(pr))
        entries = [("greedy", greedy[ri])] + [(f"s{j}", samp[ri * k + j]) for j in range(k)]
        row = {"task_id": rec["task_id"], "cands": []}
        for tag, g in entries:
            text = p.tok.decode(g["seq_ids"][plen:], skip_special_tokens=True)
            cand = candidate_from_completion(text, rec, source=tag, order=len(row["cands"]))
            row["cands"].append({
                "tag": tag, "full_pass": bool(cand["full_pass"]),
                "visible_all_pass": bool(cand.get("visible_all_pass", False)),
                "behavior_signature": cand.get("behavior_signature", ""),
                "parse_ok": cand["parse_status"] == "parsed",
                "code": cand["code"], "code_len": len(cand["code"]),
                "n_think": int(g.get("n_think", 0)), "forced": bool(g.get("forced", False)),
            })
        rows.append(row)
    # explicit P(True) judge (no-think, same as C46) for every parsed candidate
    jp, jidx = [], []
    for ri, row in enumerate(rows):
        for ci, c in enumerate(row["cands"]):
            if c["parse_ok"] and c["code"]:
                jp.append(p.judge_prompt(records[ri]["task_text"], c["code"], enable_thinking=False))
                jidx.append((ri, ci))
    ptrue = p.judge_nothink(jp, batch_size=16)
    for (ri, ci), v in zip(jidx, ptrue):
        rows[ri]["cands"][ci]["p_true"] = round(float(v), 4)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--answer-max", type=int, default=420)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 4096])
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_records(a.n, a.offset)
    print(f"[gb] {len(records)} MBPP records; budgets {a.budgets}", flush=True)
    p = GL.Probe()
    prompts = [sampling_prompt(r, p.tok) for r in records]
    for b in a.budgets:
        print(f"[gb] === budget {b} ===", flush=True)
        rows = build_pool(p, records, prompts, b, a.k, a.answer_max)
        outp = OUT / f"pool_think_b{b}.json"
        json.dump(rows, open(outp, "w"))
        npass = sum(c["full_pass"] for r in rows for c in r["cands"])
        nall = sum(len(r["cands"]) for r in rows)
        mth = sum(c["n_think"] for r in rows for c in r["cands"]) / max(1, nall)
        print(f"[gb] wrote {outp} | pass-rate {npass/nall:.3f} | mean n_think {mth:.0f}", flush=True)


if __name__ == "__main__":
    main()
