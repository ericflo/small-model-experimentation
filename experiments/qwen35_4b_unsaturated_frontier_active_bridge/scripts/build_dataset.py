#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import (  # noqa: E402
    BASE_FAMILIES,
    BRIDGE_TOTAL,
    BRIDGE_ALLOCATION,
    CHALLENGE_FAMILIES,
    base_records,
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
        for case in record.get("case_pool", []):
            if not program_case_passes(record["target_program"], case):
                raise RuntimeError(f"oracle failed on case pool for {record['id']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--seed-train-records", type=int, default=240)
    parser.add_argument("--bridge-base-records", type=int, default=180)
    parser.add_argument("--iid-eval-records", type=int, default=60)
    parser.add_argument("--challenge-eval-per-family", type=int, default=12)
    parser.add_argument("--mining-records-per-family", type=int, default=24)
    parser.add_argument("--visible-cases", type=int, default=6)
    parser.add_argument("--hidden-cases", type=int, default=18)
    parser.add_argument("--mine-case-pool", type=int, default=96)
    parser.add_argument("--selector-pool-size", type=int, default=192)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    import random

    rng = random.Random(args.seed)
    seed_train = base_records(
        rng=rng,
        count=args.seed_train_records,
        split="train_seed",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    bridge_base = base_records(
        rng=rng,
        count=args.bridge_base_records,
        split="train_base_anchor",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    static_bridge = static_bridge_records(
        rng=rng,
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    iid_eval = base_records(
        rng=rng,
        count=args.iid_eval_records,
        split="eval_iid",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    challenge_eval = challenge_records(
        rng=rng,
        split="eval_challenge",
        records_per_family=args.challenge_eval_per_family,
        trace_strategy="eval_random",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    mining_pool = challenge_records(
        rng=rng,
        split="mine",
        records_per_family=args.mining_records_per_family,
        trace_strategy="mining_prompt",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        case_pool_count=args.mine_case_pool,
    )

    static_train = bridge_base + static_bridge
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "seed" / "dsl_train.jsonl", seed_train)
    write_jsonl(args.out_dir / "static_bridge" / "dsl_train.jsonl", static_train)
    write_jsonl(args.out_dir / "base_anchor" / "dsl_train_base_anchor.jsonl", bridge_base)
    write_jsonl(args.out_dir / "bridge" / "static_bridge_records.jsonl", static_bridge)
    write_jsonl(args.out_dir / "eval" / "dsl_eval_iid.jsonl", iid_eval)
    write_jsonl(args.out_dir / "eval" / "dsl_eval_challenge.jsonl", challenge_eval)
    write_jsonl(args.out_dir / "mining" / "dsl_mining_pool.jsonl", mining_pool)

    validate(seed_train + static_train + iid_eval + challenge_eval + mining_pool)
    manifest = {
        "seed": args.seed,
        "model_id": "Qwen/Qwen3.5-4B",
        "frontier_goal": "unsaturated active bridge allocation",
        "seed_train_records": len(seed_train),
        "bridge_base_records": len(bridge_base),
        "static_bridge_records": len(static_bridge),
        "bridge_total": BRIDGE_TOTAL,
        "static_bridge_train_records": len(static_train),
        "iid_eval_records": len(iid_eval),
        "challenge_eval_records": len(challenge_eval),
        "mining_pool_records": len(mining_pool),
        "visible_cases_per_record": args.visible_cases,
        "hidden_cases_per_record": args.hidden_cases,
        "mine_case_pool_per_record": args.mine_case_pool,
        "selector_pool_size": args.selector_pool_size,
        "base_families": BASE_FAMILIES,
        "challenge_families": CHALLENGE_FAMILIES,
        "bridge_allocation": BRIDGE_ALLOCATION,
        "conditions": {
            "seed": "240 base-family random-trace records",
            "static_bridge": "180 base-family records plus 60 frontier-family records with static counterexample-selected traces",
            "seed_mined_bridge": "180 base-family records plus 60 frontier-family records selected against seed-adapter wrong programs",
            "adaptive_bridge": "180 base-family records plus 60 frontier-family records allocated toward wrong programs still produced after static bridge training",
        },
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
