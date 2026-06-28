#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.bucket_belief import make_bucket_example, top_split_queries  # noqa: E402
from src.operator_env import LETTERS, build_operator_library, load_jsonl, write_jsonl  # noqa: E402


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_cell = Counter((row["split"], row["library_size"], row["template"]) for row in rows)
    by_label = Counter(row["label"] for row in rows)
    by_state = defaultdict(int)
    for row in rows:
        by_state[(row["record_id"], tuple(row["used_query_indices"]))] += 1
    return {
        "examples": len(rows),
        "states": len(by_state),
        "labels": {letter: by_label.get(letter, 0) for letter in LETTERS},
        "by_cell": {f"{split}/n{n}/{template}": count for (split, n, template), count in sorted(by_cell.items())},
        "candidate_count_mean": sum(int(row["candidate_count"]) for row in rows) / max(len(rows), 1),
        "survivors_if_taken_mean": sum(int(row["survivors_if_taken"]) for row in rows) / max(len(rows), 1),
        "reward_mean": sum(float(row["reward"]) for row in rows) / max(len(rows), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "train_records.jsonl")
    parser.add_argument("--states", type=Path, default=ROOT / "data" / "process_train_states.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "bucket_train_examples.jsonl")
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--candidates-per-state", type=int, default=8)
    parser.add_argument("--max-options", type=int, default=8)
    args = parser.parse_args()

    records = {row["record_id"]: row for row in load_jsonl(args.records)}
    states = load_jsonl(args.states)
    operator_cache: dict[int, Any] = {}
    examples: list[dict[str, Any]] = []
    for state in tqdm(states, desc=f"bucket-examples-{args.out.name}"):
        record = records[state["record_id"]]
        operators = operator_cache.setdefault(record["library_size"], build_operator_library(record["library_size"]))
        used = list(state["used_query_indices"])
        queries = top_split_queries(record, operators, used, args.candidates_per_state)
        for rank, qidx in enumerate(queries):
            example = make_bucket_example(record, operators, used, qidx, max_options=args.max_options)
            example["state_id"] = state["state_id"]
            example["probe_rank_by_split"] = rank + 1
            examples.append(example)

    write_jsonl(args.out, examples)
    manifest = {
        "records": str(args.records),
        "states": str(args.states),
        "out": str(args.out),
        "candidates_per_state": args.candidates_per_state,
        "max_options": args.max_options,
        "summary": summarize(examples),
    }
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
