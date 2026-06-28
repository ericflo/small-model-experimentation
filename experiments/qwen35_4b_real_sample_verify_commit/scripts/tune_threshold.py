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

from src.eval_utils import model_state_for_prefix, prefix_candidates, visible_coverage  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402


def parse_budgets(text: str, max_count: int) -> list[int]:
    rows: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            rows.append(max_count if item == "max" else int(item))
    return sorted(set(item for item in rows if item > 0))


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, dict[str, float]] = {}
    for row in payload.get("candidate_scores", []):
        scores.setdefault(row["record_id"], {})[row["candidate_id"]] = float(row["score"])
    return scores


def run_threshold(records: list[dict[str, Any]], scores_by_record: dict[str, dict[str, float]], budgets: list[int], threshold: float) -> dict[str, float]:
    rng = random.Random(20260625)
    selected = 0
    samples = 0
    visible_cov = 0
    for record in records:
        scores = scores_by_record.get(record["record_id"], {})
        chosen = None
        for budget in budgets:
            state = model_state_for_prefix(record, budget, scores, rng)
            if (state["visible_count"] > 0 and state["top_score"] >= threshold) or budget == budgets[-1]:
                chosen = state
                break
        assert chosen is not None
        selected += int(chosen["selected_hidden_all"])
        samples += int(chosen["budget"])
        visible_cov += int(visible_coverage(prefix_candidates(record, int(chosen["budget"]))))
    n = len(records)
    return {
        "threshold": threshold,
        "selected_hidden_all": selected / n if n else 0.0,
        "samples_used_mean": samples / n if n else 0.0,
        "visible_coverage": visible_cov / n if n else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "mbpp_train_records.jsonl")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "threshold_tuning.json")
    parser.add_argument("--budgets", type=str, default="1,2,4,8,16,max")
    parser.add_argument("--sample-penalty", type=float, default=0.01)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    score_map = load_scores(args.scores)
    max_count = max(len(record["candidates"]) for record in records) if records else 0
    budgets = parse_budgets(args.budgets, max_count)
    raw_scores = [score for per_record in score_map.values() for score in per_record.values()]
    if not raw_scores:
        thresholds = [0.0]
    else:
        lo, hi = min(raw_scores), max(raw_scores)
        thresholds = [lo - 1.0] + [lo + (hi - lo) * i / 40 for i in range(41)] + [hi + 1.0]
    rows = [run_threshold(records, score_map, budgets, threshold) for threshold in thresholds]
    max_budget = max(budgets) if budgets else 1
    for row in rows:
        row["objective"] = row["selected_hidden_all"] - args.sample_penalty * (row["samples_used_mean"] / max_budget)
    best = sorted(rows, key=lambda row: (-row["objective"], -row["selected_hidden_all"], row["samples_used_mean"]))[0]
    payload = {"best": best, "rows": rows, "budgets": budgets, "sample_penalty": args.sample_penalty}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["best"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

