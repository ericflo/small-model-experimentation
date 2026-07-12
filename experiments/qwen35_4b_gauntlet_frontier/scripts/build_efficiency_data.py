#!/usr/bin/env python3
"""Compression-advantage SFT rows from a K-sample harvest (novel mechanism).

Root-cause diagnosis (C53): the medium/hard menagerie residual is a
serial-compute ceiling — hard items (search/induction) need long derivations
that do not fit the deployed think budget, so the emission policy teaches
early commitment to WRONG answers. This builder attacks that directly by
amortizing serial test-time compute into the weights: it rewards the model's
own SHORTEST correct trace per item so the same problem is solved in fewer
tokens next time.

Per item's group of K on-policy samples:
  * correct = score>=1.0, wrong = score<1.0
  * COMPRESSION target: the shortest correct trace -> weight +1.0 (bank the
    efficient solution). This is the primary "think-faster" signal.
  * BREVITY-ADVANTAGE rows: other correct traces weighted by
    z = (mean_correct_len - len)/std_correct_len clamped to [0, pos_clamp]
    -> a positive-only push toward brevity, active ONLY where the model
    already succeeds (never teaches it to rush an unsolved item).
  * CONTRAST rows: wrong traces (only if some correct exists), weight
    -neg_clamp on the ANSWER span only (C29 collapse guard: never negative
    gradient on thinking, and abs-normalized in the loss).
Items with zero correct samples are skipped and COUNTED (unreachable-at-K:
the genuinely-too-hard tail that needs a different mechanism than this).
CPU-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest-dir", type=Path, default=EXP / "runs" / "harvest_grpo1")
    ap.add_argument("--pos-clamp", type=float, default=0.7)
    ap.add_argument("--neg-clamp", type=float, default=0.30)
    ap.add_argument("--brevity-lambda", type=float, default=0.7)
    ap.add_argument("--max-extra-correct", type=int, default=1)
    ap.add_argument("--max-neg", type=int, default=1)
    ap.add_argument("--max-think", type=int, default=1400)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_efficiency.jsonl")
    args = ap.parse_args()

    rows: list[dict] = []
    stat = Counter()
    len_solved, len_shortest = [], []

    for shard in sorted(args.harvest_dir.glob("atoms_rows*.jsonl.gz")):
        for line in gzip.open(shard, "rt", encoding="utf-8"):
            item = json.loads(line)
            samples = []
            for out in item["outputs"]:
                think, _ = base.split_think(out["text"])
                value = base.extract_answer(out["text"])
                think = think.strip()
                if not think or value is None or out["truncated"]:
                    continue
                samples.append({
                    "score": float(out["score"]),
                    "value": value,
                    "think": think,
                    "len": out["n_thinking_tokens"],
                })
            if not samples:
                stat["item_no_parseable"] += 1
                continue
            correct = [s for s in samples if s["score"] >= 1.0 and s["len"] <= args.max_think]
            wrong = [s for s in samples if s["score"] < 1.0]
            if not correct:
                stat["item_unreachable_at_k"] += 1
                continue
            stat["item_solved"] += 1
            correct.sort(key=lambda s: s["len"])
            base_row = {
                "family": item["family"], "level": item["level"],
                "messages": [{"role": "user", "content": item["prompt"]}],
            }

            # (1) compression target: shortest correct trace, full weight.
            shortest = correct[0]
            rows.append({**base_row, "kind": "eff_compress",
                         "think": shortest["think"],
                         "answer": f"ANSWER: {shortest['value']}",
                         "n_think_tokens": shortest["len"], "row_weight": 1.0})
            len_shortest.append(shortest["len"])
            len_solved.extend(s["len"] for s in correct)

            # (2) brevity-advantage rows among the remaining correct samples.
            if len(correct) >= 2:
                lens = [s["len"] for s in correct]
                mean_l, std_l = statistics.mean(lens), statistics.pstdev(lens) or 1.0
                for s in correct[1:1 + args.max_extra_correct]:
                    z = (mean_l - s["len"]) / std_l
                    weight = max(0.0, min(args.brevity_lambda * z, args.pos_clamp))
                    if weight < 0.05:
                        continue
                    rows.append({**base_row, "kind": "eff_brevity",
                                 "think": s["think"],
                                 "answer": f"ANSWER: {s['value']}",
                                 "n_think_tokens": s["len"],
                                 "row_weight": round(weight, 4)})

            # (3) contrast rows: wrong answers pushed down (answer span only).
            seen = set()
            for s in wrong[: args.max_neg]:
                key = s["value"].strip().lower()
                if key == shortest["value"].strip().lower() or key in seen:
                    continue
                seen.add(key)
                rows.append({**base_row, "kind": "eff_contrast",
                             "think": s["think"],
                             "answer": f"ANSWER: {s['value']}",
                             "n_think_tokens": s["len"],
                             "row_weight": -args.neg_clamp})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    kinds = Counter(r["kind"] for r in rows)
    fams = Counter(r["family"] for r in rows)
    print("efficiency rows:", len(rows), dict(kinds))
    print("items:", dict(stat))
    if len_solved:
        print(f"compression: mean-correct-len {statistics.mean(len_solved):.0f} "
              f"-> shortest-correct-len {statistics.mean(len_shortest):.0f} "
              f"({100*(1-statistics.mean(len_shortest)/statistics.mean(len_solved)):.0f}% shorter)")
    print("by family:", dict(fams))
    return 0


if __name__ == "__main__":
    sys.exit(main())
