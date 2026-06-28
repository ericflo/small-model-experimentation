#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import (  # noqa: E402
    BASE_FAMILIES,
    CEILING_FAMILIES,
    FRONTIER_FAMILIES,
    STATIC_BRIDGE_ALLOCATION,
    base_records,
    ceiling_records,
    challenge_records,
    static_bridge_records,
    write_jsonl,
)
from src.dsl import program_case_passes  # noqa: E402
from src.graphir import dsl_to_graph, graph_case_passes, safe_execute_graph  # noqa: E402


def _with_graph(record: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(record)
    row["target_graph"] = dsl_to_graph(row["target_program"])
    return row


def _candidate_program(record: dict[str, Any], rng: random.Random) -> str:
    candidates = [record["wrong_program"], *record.get("static_distractors", [])]
    candidates = [item for item in candidates if item and item.strip() != record["target_program"].strip()]
    return rng.choice(candidates)


def as_dsl_record(record: dict[str, Any]) -> dict[str, Any]:
    row = _with_graph(record)
    row["task"] = "dsl"
    row["target_output"] = row["target_program"]
    return row


def as_graph_construct_record(record: dict[str, Any]) -> dict[str, Any]:
    row = _with_graph(record)
    row["task"] = "graph_construct"
    row["target_output"] = row["target_graph"]
    return row


def as_graph_repair_record(record: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    row = _with_graph(record)
    candidate_program = _candidate_program(row, rng)
    candidate_graph = dsl_to_graph(candidate_program)
    row["task"] = "graph_repair"
    row["candidate_program"] = candidate_program
    row["candidate_graph"] = candidate_graph
    row["target_output"] = row["target_graph"]
    visible = []
    for case in row["visible"]:
        visible.append({**case, "candidate_got": safe_execute_graph(candidate_graph, case["input"])})
    row["visible"] = visible
    return row


def validate(rows: list[dict[str, Any]]) -> None:
    for record in rows:
        graph = dsl_to_graph(record["target_program"])
        for case in record["visible"] + record["hidden"]:
            if not program_case_passes(record["target_program"], case):
                raise RuntimeError(f"DSL oracle failed on {record['id']}")
            if not graph_case_passes(graph, case):
                raise RuntimeError(f"GraphIR oracle failed on {record['id']}")


def allocation_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["family"]] = counts.get(row["family"], 0) + 1
    return dict(sorted(counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--base-train-records", type=int, default=180)
    parser.add_argument("--iid-eval-records", type=int, default=60)
    parser.add_argument("--support-eval-per-family", type=int, default=12)
    parser.add_argument("--ceiling-eval-per-family", type=int, default=12)
    parser.add_argument("--visible-cases", type=int, default=6)
    parser.add_argument("--hidden-cases", type=int, default=18)
    parser.add_argument("--selector-pool-size", type=int, default=224)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    base_train = base_records(
        rng=rng,
        count=args.base_train_records,
        split="train_base",
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
    )
    bridge_train = static_bridge_records(
        rng=rng,
        visible_count=args.visible_cases,
        hidden_count=args.hidden_cases,
        pool_size=args.selector_pool_size,
        allocation=STATIC_BRIDGE_ALLOCATION,
        case_mode="normal",
    )
    source_train = base_train + bridge_train
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

    validate(source_train + iid_eval + support_eval + ceiling_eval)

    repair_rng = random.Random(args.seed + 17)
    dsl_train = [as_dsl_record(row) for row in source_train]
    construct_train = [as_graph_construct_record(row) for row in source_train]
    repair_train = [as_graph_repair_record(row, repair_rng) for row in source_train]
    construct_iid = [as_graph_construct_record(row) for row in iid_eval]
    construct_support = [as_graph_construct_record(row) for row in support_eval]
    construct_ceiling = [as_graph_construct_record(row) for row in ceiling_eval]
    dsl_iid = [as_dsl_record(row) for row in iid_eval]
    dsl_support = [as_dsl_record(row) for row in support_eval]
    dsl_ceiling = [as_dsl_record(row) for row in ceiling_eval]
    repair_ceiling = [as_graph_repair_record(row, repair_rng) for row in ceiling_eval]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "source" / "train_source.jsonl", [_with_graph(row) for row in source_train])
    write_jsonl(args.out_dir / "dsl" / "train.jsonl", dsl_train)
    write_jsonl(args.out_dir / "graph_construct" / "train.jsonl", construct_train)
    write_jsonl(args.out_dir / "graph_repair" / "train.jsonl", repair_train)
    write_jsonl(args.out_dir / "eval" / "dsl_iid.jsonl", dsl_iid)
    write_jsonl(args.out_dir / "eval" / "dsl_support.jsonl", dsl_support)
    write_jsonl(args.out_dir / "eval" / "dsl_ceiling.jsonl", dsl_ceiling)
    write_jsonl(args.out_dir / "eval" / "graph_iid.jsonl", construct_iid)
    write_jsonl(args.out_dir / "eval" / "graph_support.jsonl", construct_support)
    write_jsonl(args.out_dir / "eval" / "graph_ceiling.jsonl", construct_ceiling)
    write_jsonl(args.out_dir / "eval" / "graph_repair_ceiling_corrupt.jsonl", repair_ceiling)

    manifest = {
        "seed": args.seed,
        "model_id": "Qwen/Qwen3.5-4B",
        "experiment": "GraphIR self repair",
        "base_train_records": len(base_train),
        "bridge_train_records": len(bridge_train),
        "train_records_per_adapter": len(source_train),
        "iid_eval_records": len(iid_eval),
        "support_eval_records": len(support_eval),
        "ceiling_eval_records": len(ceiling_eval),
        "visible_cases_per_record": args.visible_cases,
        "hidden_cases_per_record": args.hidden_cases,
        "selector_pool_size": args.selector_pool_size,
        "base_families": BASE_FAMILIES,
        "support_bridge_families": FRONTIER_FAMILIES,
        "ceiling_families": CEILING_FAMILIES,
        "bridge_allocation": STATIC_BRIDGE_ALLOCATION,
        "bridge_allocation_counts": allocation_counts(bridge_train),
        "conditions": {
            "dsl_static60_lora": "180 base records plus 60 support bridge records; target is DSL",
            "graphir_construct_lora": "same source records; target is typed register GraphIR",
            "graphir_repair_lora": "same source records with corrupted candidate GraphIR; target is corrected GraphIR",
        },
        "evaluation_splits": {
            "iid": "base-family retention",
            "support": "hard input cases for support bridge families",
            "ceiling": "deeper held-out composition families absent from training",
            "repair_ceiling_corrupt": "oracle corrupted GraphIR repair on ceiling records",
        },
    }
    (args.out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
