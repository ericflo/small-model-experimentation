#!/usr/bin/env python3
"""Matched-context margin pairs at the commit point.

From a K-sample harvest: for items where the model produced BOTH a correct
and >=1 wrong parseable answer, emit pairs sharing one correct sample's
think — (think, ANSWER correct, weight +1.0) and (think, ANSWER wrong_i,
weight -0.30) — so the gradient is a margin exactly at the answer tokens
with reasoning held fixed. CPU-only.
"""
from __future__ import annotations
import argparse, gzip, json, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
from gym import base  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest-dir", type=Path, default=EXP / "runs" / "harvest_grpo1")
    ap.add_argument("--neg-weight", type=float, default=-0.30)
    ap.add_argument("--max-negs", type=int, default=2)
    ap.add_argument("--out", type=Path, default=EXP / "data" / "margin_pairs.jsonl")
    args = ap.parse_args()

    rows = []
    for shard in sorted(args.harvest_dir.glob("atoms_rows*.jsonl.gz")):
        for line in gzip.open(shard, "rt", encoding="utf-8"):
            item = json.loads(line)
            corrects, wrongs = [], []
            for o in item["outputs"]:
                value = base.extract_answer(o["text"])
                if value is None:
                    continue
                (corrects if o["score"] >= 1.0 else wrongs).append((o, value))
            if not corrects or not wrongs:
                continue
            anchor, correct_value = min(corrects, key=lambda p: p[0]["n_thinking_tokens"])
            think, _ = base.split_think(anchor["text"])
            think = think.strip()
            if not think or anchor["n_thinking_tokens"] > 1400:
                continue
            seen = set()
            base_row = {"family": item["family"], "level": item["level"],
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "think": think, "n_think_tokens": anchor["n_thinking_tokens"]}
            rows.append({**base_row, "kind": "margin_pos",
                         "answer": f"ANSWER: {correct_value}", "row_weight": 1.0})
            for _, wrong_value in wrongs:
                key = wrong_value.strip().lower()
                if key in seen or key == correct_value.strip().lower():
                    continue
                seen.add(key)
                if len(seen) > args.max_negs:
                    break
                rows.append({**base_row, "kind": "margin_neg",
                             "answer": f"ANSWER: {wrong_value}",
                             "row_weight": args.neg_weight})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in rows:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print("margin rows:", len(rows), dict(Counter(r["kind"] for r in rows)))
    return 0

if __name__ == "__main__":
    sys.exit(main())
