#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


def code_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def candidate_rank(candidate: dict[str, Any]) -> tuple[int, int, int]:
    if candidate.get("visible_all_pass") and not candidate.get("full_pass"):
        tier = 0
    elif candidate.get("parse_status") == "parsed" and not candidate.get("full_pass"):
        tier = 1
    else:
        tier = 2
    return (tier, int(candidate.get("order", 999999)), len(candidate.get("code", "")))


def build_rows(records: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(args.seed)
    pairs: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    per_task: dict[str, int] = defaultdict(int)
    pair_id = 0
    for record in records:
        candidates = list(record.get("candidates", []))
        positives = [c for c in candidates if c.get("full_pass") and c.get("code")]
        negatives = [c for c in candidates if c.get("code") and not c.get("full_pass") and c.get("parse_status") == "parsed"]
        visible_wrong = [c for c in negatives if c.get("visible_all_pass")]
        if not positives:
            stats["tasks_without_positive"] += 1
            continue
        if not negatives:
            stats["tasks_without_negative"] += 1
            continue
        stats["tasks_with_pair_candidates"] += 1
        positives = sorted({code_hash(c["code"]): c for c in positives}.values(), key=lambda c: int(c.get("order", 999999)))
        hard = sorted(visible_wrong or negatives, key=candidate_rank)
        rng.shuffle(positives)
        selected_pos = positives[: args.max_positives_per_task]
        selected_neg = hard[: args.max_negatives_per_positive]
        for pos in selected_pos:
            for neg in selected_neg:
                chosen = pos["code"]
                rejected = neg["code"]
                if args.shuffle_labels and rng.random() < 0.5:
                    chosen, rejected = rejected, chosen
                public_asserts = [case["assert_src"] for case in record.get("public_cases", [])]
                pair = {
                    "pair_id": f"pair_{pair_id:06d}",
                    "record_id": record["record_id"],
                    "task_id": record.get("task_id"),
                    "task_text": record["task_text"],
                    "entry_point": record["entry_point"],
                    "public_tests": public_asserts,
                    "prompt": record["prompt"],
                    "chosen": chosen,
                    "rejected": rejected,
                    "chosen_source": pos.get("source"),
                    "rejected_source": neg.get("source"),
                    "chosen_order": pos.get("order"),
                    "rejected_order": neg.get("order"),
                    "rejected_visible_all_pass": bool(neg.get("visible_all_pass")),
                    "rejected_functional_signature": neg.get("functional_signature") or neg.get("failure_bits"),
                    "shuffled_label": bool(args.shuffle_labels),
                }
                pairs.append(pair)
                per_task[record["record_id"]] += 1
                pair_id += 1
        if visible_wrong:
            stats["tasks_with_visible_wrong"] += 1
    rng.shuffle(pairs)
    if args.max_pairs:
        pairs = pairs[: args.max_pairs]
    summary = {
        "experiment": EXPERIMENT,
        "source": str(args.records),
        "shuffle_labels": args.shuffle_labels,
        "seed": args.seed,
        "records": len(records),
        "pairs": len(pairs),
        "tasks_with_pairs": len({row["record_id"] for row in pairs}),
        "mean_pairs_per_task": (len(pairs) / len({row["record_id"] for row in pairs})) if pairs else 0.0,
        "visible_wrong_pair_rate": sum(1 for row in pairs if row["rejected_visible_all_pass"]) / len(pairs) if pairs else 0.0,
        "stats": dict(stats),
        "top_pair_tasks": Counter(row["record_id"] for row in pairs).most_common(10),
    }
    return pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-positives-per-task", type=int, default=2)
    parser.add_argument("--max-negatives-per-positive", type=int, default=2)
    parser.add_argument("--max-pairs", type=int, default=512)
    parser.add_argument("--shuffle-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    pairs, summary = build_rows(load_jsonl(args.records), args)
    write_jsonl(args.out, pairs)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

