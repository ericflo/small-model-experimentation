#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import (  # noqa: E402
    BASE_FAMILIES,
    CEILING_FAMILIES,
    FRONTIER_FAMILIES,
    STATIC_BRIDGE_80_ALLOCATION,
    STATIC_BRIDGE_ALLOCATION,
    base_records,
    ceiling_records,
    challenge_records,
    static_bridge_records,
    write_jsonl,
)
from src.dsl import program_case_passes  # noqa: E402


def validate(rows: list[dict]) -> None:
    for record in rows:
        for case in record["visible"] + record["hidden"]:
            if not program_case_passes(record["target_program"], case):
                raise RuntimeError(f"oracle failed on {record['id']}")


def allocation_counts(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    return dict(sorted(counts.items()))


def selector_summary(rows: list[dict]) -> dict[str, float]:
    stats = [row.get("selection_stats", {}) for row in rows if row.get("selection_stats")]
    if not stats:
        return {}
    return {
        "records": len(stats),
        "avg_eliminated_wrong_programs": round(
            sum(item.get("eliminated_wrong_programs", 0) for item in stats) / len(stats),
            3,
        ),
        "avg_remaining_wrong_programs": round(
            sum(item.get("remaining_wrong_programs", 0) for item in stats) / len(stats),
            3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--seed-train-records", type=int, default=240)
    parser.add_argument("--static60-base-records", type=int, default=180)
    parser.add_argument("--static80-base-records", type=int, default=160)
    parser.add_argument("--iid-eval-records", type=int, default=60)
    parser.add_argument("--support-eval-per-family", type=int, default=12)
    parser.add_argument("--ceiling-eval-per-family", type=int, default=12)
    parser.add_argument("--visible-cases", type=int, default=6)
    parser.add_argument("--hidden-cases", type=int, default=18)
    parser.add_argument("--selector-pool-size", type=int, default=224)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    seed_train = base_records(
        rng=rng,
        count=args.seed_train_records,
        split="train_seed",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    static60_base = base_records(
        rng=rng,
        count=args.static60_base_records,
        split="train_static60_base",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    static80_base = base_records(
        rng=rng,
        count=args.static80_base_records,
        split="train_static80_base",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    static60_bridge = static_bridge_records(
        rng=rng,
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        allocation=STATIC_BRIDGE_ALLOCATION,
        case_mode="normal",
    )
    static80_bridge = static_bridge_records(
        rng=rng,
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        allocation=STATIC_BRIDGE_80_ALLOCATION,
        case_mode="normal",
    )
    static60_train = static60_base + static60_bridge
    static80_train = static80_base + static80_bridge
    iid_eval = base_records(
        rng=rng,
        count=args.iid_eval_records,
        split="eval_iid",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    support_eval = challenge_records(
        rng=rng,
        split="eval_support",
        records_per_family=args.support_eval_per_family,
        trace_strategy="eval_random",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        case_mode="hard",
    )
    ceiling_eval = ceiling_records(
        rng=rng,
        split="eval_ceiling",
        records_per_family=args.ceiling_eval_per_family,
        trace_strategy="eval_random",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        case_mode="normal",
    )

    validate(seed_train + static60_train + static80_train + iid_eval + support_eval + ceiling_eval)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "seed" / "dsl_train.jsonl", seed_train)
    write_jsonl(args.out_dir / "static_bridge_60" / "dsl_train.jsonl", static60_train)
    write_jsonl(args.out_dir / "static_bridge_80" / "dsl_train.jsonl", static80_train)
    write_jsonl(args.out_dir / "base_anchor" / "dsl_train_static60_base.jsonl", static60_base)
    write_jsonl(args.out_dir / "base_anchor" / "dsl_train_static80_base.jsonl", static80_base)
    write_jsonl(args.out_dir / "bridge" / "static60_bridge_records.jsonl", static60_bridge)
    write_jsonl(args.out_dir / "bridge" / "static80_bridge_records.jsonl", static80_bridge)
    write_jsonl(args.out_dir / "eval" / "dsl_eval_iid.jsonl", iid_eval)
    write_jsonl(args.out_dir / "eval" / "dsl_eval_support.jsonl", support_eval)
    write_jsonl(args.out_dir / "eval" / "dsl_eval_ceiling.jsonl", ceiling_eval)

    manifest = {
        "seed": args.seed,
        "model_id": "Qwen/Qwen3.5-4B",
        "experiment": "static bridge ceiling breaker",
        "seed_train_records": len(seed_train),
        "static60_base_records": len(static60_base),
        "static60_bridge_records": len(static60_bridge),
        "static60_train_records": len(static60_train),
        "static80_base_records": len(static80_base),
        "static80_bridge_records": len(static80_bridge),
        "static80_train_records": len(static80_train),
        "iid_eval_records": len(iid_eval),
        "support_eval_records": len(support_eval),
        "ceiling_eval_records": len(ceiling_eval),
        "visible_cases_per_record": args.visible_cases,
        "hidden_cases_per_record": args.hidden_cases,
        "selector_pool_size": args.selector_pool_size,
        "base_families": BASE_FAMILIES,
        "support_bridge_families": FRONTIER_FAMILIES,
        "ceiling_families": CEILING_FAMILIES,
        "static60_bridge_allocation": STATIC_BRIDGE_ALLOCATION,
        "static80_bridge_allocation": STATIC_BRIDGE_80_ALLOCATION,
        "static60_bridge_allocation_counts": allocation_counts(static60_bridge),
        "static80_bridge_allocation_counts": allocation_counts(static80_bridge),
        "static60_selector_summary": selector_summary(static60_bridge),
        "static80_selector_summary": selector_summary(static80_bridge),
        "conditions": {
            "seed": "240 base-family random-trace records",
            "static_bridge_60": "180 base-family records plus 60 equal support bridge records",
            "static_bridge_80": "160 base-family records plus 80 equal support bridge records",
        },
        "evaluation_splits": {
            "iid": "base-family retention",
            "support": "hard input cases for support bridge families",
            "ceiling": "deeper held-out composition families absent from bridge training",
        },
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
