#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.code_env import stop_prompt  # noqa: E402
from src.eval_utils import model_state_for_prefix  # noqa: E402
from src.jsonl import load_jsonl, write_jsonl  # noqa: E402


def parse_budgets(text: str, max_count: int) -> list[int]:
    rows: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        rows.append(max_count if item == "max" else int(item))
    return sorted(set(item for item in rows if item > 0))


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, dict[str, float]] = {}
    for row in payload.get("candidate_scores", []):
        scores.setdefault(row["record_id"], {})[row["candidate_id"]] = float(row["score"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "mbpp_train_records.jsonl")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "train_stop_examples.jsonl")
    parser.add_argument("--budgets", type=str, default="1,2,4,8,16,max")
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    score_map = load_scores(args.scores)
    max_count = max(len(record["candidates"]) for record in records) if records else 0
    budgets = parse_budgets(args.budgets, max_count)
    rng = random.Random(args.seed)
    examples: list[dict[str, Any]] = []
    for record in records:
        scores = score_map.get(record["record_id"], {})
        for budget in budgets:
            state = model_state_for_prefix(record, budget, scores, rng)
            stop = bool(state["selected_hidden_all"] or budget == budgets[-1])
            examples.append(
                {
                    "dataset": record["dataset"],
                    "record_id": record["record_id"],
                    "task_id": record["task_id"],
                    "budget": budget,
                    "label": "A" if stop else "B",
                    "label_index": 0 if stop else 1,
                    "prompt": stop_prompt(record, state),
                    "selected_hidden_all": state["selected_hidden_all"],
                }
            )
    rng.shuffle(examples)
    write_jsonl(args.out, examples)
    print(
        {
            "examples": len(examples),
            "stop": sum(row["label"] == "A" for row in examples),
            "more": sum(row["label"] == "B" for row in examples),
            "out": str(args.out),
        }
    )


if __name__ == "__main__":
    main()

