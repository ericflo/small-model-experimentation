#!/usr/bin/env python3
"""Episode-mastery SFT data: teach multi-turn horizon persistence.

The medium tier's differentiated substance is multi-turn episodes, and the
base model quits at 33-37% of its turn budget. This builder extracts FULLY
successful multi-turn trajectories (verifier score 1.0) and emits every
assistant turn as a training example, so the model learns the full
act-observe-act loop rather than an early terminal action. Novel weighting:
turns from LONGER successful trajectories (more horizon use) get a mild
up-weight (row_weight) to reinforce persistence; the decisive final action
that resolves the goal gets full weight. Turns must be verifier-accepted
(action_ok) and naturally-closed. CPU-only.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from gym import base  # noqa: E402


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as h:
        return [json.loads(l) for l in h if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harvest-dir", type=Path, default=EXP / "runs" / "harvest_epmastery")
    ap.add_argument("--min-turns", type=int, default=2, help="require genuine multi-turn success")
    ap.add_argument("--max-per-instance", type=int, default=2, help="cap successful rollouts per episode instance")
    ap.add_argument("--persist-weight", type=float, default=0.15,
                    help="extra weight per turn of horizon use (capped)")
    ap.add_argument("--out", type=Path, default=EXP / "data" / "sft_epmastery.jsonl")
    args = ap.parse_args()

    # group successful rollouts by instance so K near-identical wins don't over-weight
    by_instance: dict[tuple, list[dict]] = {}
    for shard in sorted(args.harvest_dir.glob("episodes_rows*.jsonl.gz")):
        for row in read_gz(shard):
            if row["score"] < 1.0 or row["n_turns"] < args.min_turns:
                continue
            key = (row["family"], row["level"], row["ep_seed"])
            by_instance.setdefault(key, []).append(row)

    rows_out: list[dict] = []
    for instance_rows in by_instance.values():
        instance_rows.sort(key=lambda r: -r["n_turns"])  # prefer fuller horizon use
        for row in instance_rows[: args.max_per_instance]:
            messages = [
                {"role": "system", "content": row["system_prompt"]},
                {"role": "user", "content": row["initial_observation"]},
            ]
            n_turns = len(row["turns"])
            # up-weight the whole successful trajectory by how much horizon it used
            persist = min(1.0 + args.persist_weight * n_turns, 1.7)
            for i, turn in enumerate(row["turns"]):
                usable = (
                    not turn["forced_close"]
                    and not turn["truncated"]
                    and turn.get("action_ok", True)
                    and turn["action"]
                    and "text" in turn
                )
                if usable:
                    think, _ = base.split_think(turn["text"])
                    think = think.strip()
                    is_final = i == n_turns - 1
                    if think:
                        rows_out.append({
                            "family": row["family"], "level": row["level"],
                            "kind": "epmastery",
                            "messages": [dict(m) for m in messages],
                            "think": think, "answer": turn["action"],
                            "n_think_tokens": turn["n_thinking_tokens"],
                            # decisive final action gets full weight; earlier
                            # act-observe turns get the persistence up-weight
                            "row_weight": round(persist if is_final else min(persist, 1.3), 3),
                        })
                messages.append({"role": "assistant", "content": turn["action"]})
                messages.append({"role": "user", "content": turn["observation"]})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as h:
        for r in rows_out:
            h.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("episode-mastery turns:", len(rows_out),
          "| successful instances:", len(by_instance))
    print("by family:", dict(Counter(r["family"] for r in rows_out)))
    print("by level:", dict(sorted(Counter(r["level"] for r in rows_out).items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
