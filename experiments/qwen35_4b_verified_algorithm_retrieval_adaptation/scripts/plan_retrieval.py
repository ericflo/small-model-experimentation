#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, load_mbpp_records  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\\d+", text.lower())


def record_tokens(record: dict[str, Any]) -> list[str]:
    return tokenize(
        " ".join(
            [
                record["task_text"],
                record["entry_point"],
                " ".join(case["assert_src"] for case in record.get("public_cases", [])),
            ]
        )
    )


def vector(tokens: list[str], idf: dict[str, float]) -> dict[str, float]:
    counts = Counter(tokens)
    return {term: (1.0 + math.log(count)) * idf.get(term, 0.0) for term, count in counts.items()}


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    dot = sum(value * b.get(term, 0.0) for term, value in a.items())
    an = math.sqrt(sum(value * value for value in a.values()))
    bn = math.sqrt(sum(value * value for value in b.values()))
    if an == 0.0 or bn == 0.0:
        return 0.0
    return dot / (an * bn)


def base_miss_tasks(path: Path | None) -> set[int]:
    if path is None:
        return set()
    misses: set[int] = set()
    for row in load_jsonl(path):
        if not any(candidate.get("full_pass") for candidate in row.get("candidates", [])):
            misses.add(int(row["task_id"]))
    return misses


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--eval-split", choices=["mbpp_test", "mbpp_train"], default="mbpp_test")
    parser.add_argument("--count", type=int, default=24)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--baseline-records", type=Path)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260626)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    library = load_jsonl(args.library)
    docs = [set(row.get("retrieval_tokens", [])) for row in library]
    n_docs = max(len(docs), 1)
    df = Counter(term for doc in docs for term in doc)
    idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}
    library_vectors = [vector(row.get("retrieval_tokens", []), idf) for row in library]
    eval_records = load_mbpp_records(args.eval_split, args.count, args.offset, args.visible_tests, timeout_s=5.0)
    misses = base_miss_tasks(args.baseline_records)
    if misses:
        eval_records = [record for record in eval_records if int(record["task_id"]) in misses]

    rows = []
    for index, record in enumerate(eval_records):
        qv = vector(record_tokens(record), idf)
        scored = [(cosine(qv, lv), lib) for lv, lib in zip(library_vectors, library)]
        scored.sort(key=lambda item: (-item[0], item[1]["library_id"]))
        semantic = [{"score": score, "algorithm": lib} for score, lib in scored[: args.top_k]]
        random_items = rng.sample(library, k=min(args.top_k, len(library)))
        shuffled_source = eval_records[(index + 1) % len(eval_records)] if eval_records else record
        shuffled_qv = vector(record_tokens(shuffled_source), idf)
        shuffled_scored = [(cosine(shuffled_qv, lv), lib) for lv, lib in zip(library_vectors, library)]
        shuffled_scored.sort(key=lambda item: (-item[0], item[1]["library_id"]))
        rows.append(
            {
                "record": record,
                "semantic": semantic,
                "random": [{"score": 0.0, "algorithm": lib} for lib in random_items],
                "shuffled": [{"score": score, "algorithm": lib} for score, lib in shuffled_scored[: args.top_k]],
            }
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    summary = {
        "experiment": EXPERIMENT,
        "library_entries": len(library),
        "eval_records": len(rows),
        "base_miss_filter": bool(misses),
        "base_miss_tasks": sorted(misses),
        "top_k": args.top_k,
        "path": str(args.out),
    }
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
