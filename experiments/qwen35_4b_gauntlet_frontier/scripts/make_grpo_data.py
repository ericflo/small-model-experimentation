#!/usr/bin/env python3
"""Build group-relative advantage-weighted rows from a K-sample harvest.

For each item (group of K on-policy samples): advantage_i = (r_i - mean) / std.
Positive-advantage samples train normally (weight clamped to pos_clamp);
negative-advantage samples get a small negative row weight (C29 guard:
clamped to -neg_clamp) — a mild push-down rather than DPO-style preference
pressure. Groups with no spread (all-correct or all-wrong) are skipped:
they carry no relative signal. CPU-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "grpo.yaml")
    parser.add_argument("--harvest-dir", type=Path, default=EXP / "runs" / "harvest_grpo1")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    grpo = config["grpo"]
    out_path = EXP / grpo["data_out"]

    rows_out, groups, skipped_flat = [], 0, 0
    for shard in sorted(args.harvest_dir.glob("atoms_rows*.jsonl.gz")):
        for line in gzip.open(shard, "rt", encoding="utf-8"):
            item = json.loads(line)
            scores = [float(o["score"]) for o in item["outputs"]]
            if len(scores) < 2 or statistics.pstdev(scores) < float(grpo["min_group_spread"]):
                skipped_flat += 1
                continue
            mean = statistics.mean(scores)
            std = statistics.pstdev(scores)
            groups += 1
            for output in item["outputs"]:
                adv = (float(output["score"]) - mean) / std
                if adv >= 0:
                    weight = min(adv, float(grpo["pos_clamp"]))
                else:
                    weight = max(adv * float(grpo["neg_clamp"]), -float(grpo["neg_clamp"]))
                if abs(weight) < 1e-4:
                    continue
                think, answer = base.split_think(output["text"])
                think, answer = think.strip(), answer.strip()
                if not think or not answer:
                    continue
                # Canonicalize positive answers to the terse shape; keep
                # negatives verbatim (we push down what was actually emitted).
                if weight > 0:
                    value = base.extract_answer(output["text"])
                    if value is None:
                        continue
                    answer = f"ANSWER: {value}"
                rows_out.append({
                    "family": item["family"],
                    "level": item["level"],
                    "kind": "grpo_pos" if weight > 0 else "grpo_neg",
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "think": think,
                    "answer": answer,
                    "n_think_tokens": output["n_thinking_tokens"],
                    "row_weight": round(weight, 4),
                })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"groups with signal: {groups} (flat skipped: {skipped_flat}); rows: {len(rows_out)}",
          dict(Counter(r["kind"] for r in rows_out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
