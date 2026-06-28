#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*|\d+", text.lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def mean_pairwise_distance(items: list[dict[str, Any]], field: str) -> float:
    if len(items) < 2:
        return 0.0
    distances = []
    for left, right in combinations(items, 2):
        distances.append(1.0 - jaccard(set(left[field]), set(right[field])))
    return sum(distances) / len(distances)


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


def base_miss_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key not in {"prompt", "candidates"}}
        for row in rows
        if not any(candidate.get("full_pass") for candidate in row.get("candidates", []))
    ]


def annotate_algorithm(algorithm: dict[str, Any]) -> dict[str, Any]:
    row = dict(algorithm)
    row["code_tokens"] = sorted(token_set(row.get("code", "")))
    row["task_tokens"] = sorted(set(row.get("retrieval_tokens", [])))
    return row


def mmr_select(scored: list[dict[str, Any]], top_k: int, lam: float) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    remaining = list(scored)
    if not remaining:
        return selected
    max_score = max(item["score"] for item in remaining) or 1.0
    while remaining and len(selected) < top_k:
        best_index = 0
        best_value = -1e9
        for index, item in enumerate(remaining):
            normalized_score = item["score"] / max_score
            max_similarity = 0.0
            if selected:
                max_similarity = max(
                    jaccard(set(item["algorithm"]["code_tokens"]), set(prev["algorithm"]["code_tokens"]))
                    for prev in selected
                )
            value = lam * normalized_score - (1.0 - lam) * max_similarity
            if value > best_value:
                best_value = value
                best_index = index
        chosen = remaining.pop(best_index)
        chosen["mmr_score"] = best_value
        selected.append(chosen)
    return selected


def retrieval_metrics(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    code_distances = []
    task_distances = []
    scores = []
    for row in rows:
        items = [item["algorithm"] for item in row[key]]
        code_distances.append(mean_pairwise_distance(items, "code_tokens"))
        task_distances.append(mean_pairwise_distance(items, "task_tokens"))
        scores.extend(item["score"] for item in row[key])
    return {
        "mean_pairwise_code_distance": sum(code_distances) / len(code_distances) if code_distances else 0.0,
        "mean_pairwise_task_distance": sum(task_distances) / len(task_distances) if task_distances else 0.0,
        "mean_retrieval_score": sum(scores) / len(scores) if scores else 0.0,
        "min_retrieval_score": min(scores) if scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-records", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--candidate-pool", type=int, default=80)
    parser.add_argument("--mmr-lambda", type=float, default=0.62)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    baseline = load_jsonl(args.baseline_records)
    residual_records = base_miss_rows(baseline)
    library = [annotate_algorithm(row) for row in load_jsonl(args.library)]
    docs = [set(row.get("retrieval_tokens", [])) for row in library]
    n_docs = max(len(docs), 1)
    df = Counter(term for doc in docs for term in doc)
    idf = {term: math.log((1 + n_docs) / (1 + freq)) + 1.0 for term, freq in df.items()}
    library_vectors = [vector(row.get("retrieval_tokens", []), idf) for row in library]

    rows = []
    for record in residual_records:
        qv = vector(record_tokens(record), idf)
        scored = []
        for lib_vector, algorithm in zip(library_vectors, library):
            score = cosine(qv, lib_vector)
            scored.append({"score": score, "algorithm": algorithm})
        scored.sort(key=lambda item: (-item["score"], item["algorithm"]["library_id"]))
        pool = scored[: args.candidate_pool]
        same = [dict(item, retrieval_rank=index) for index, item in enumerate(pool[: args.top_k])]
        independent = mmr_select([dict(item, retrieval_rank=index) for index, item in enumerate(pool)], args.top_k, args.mmr_lambda)
        rows.append(
            {
                "record": record,
                "same_neighborhood": same,
                "independent": independent,
            }
        )

    same_metrics = retrieval_metrics(rows, "same_neighborhood")
    independent_metrics = retrieval_metrics(rows, "independent")
    gate = {
        "passed": independent_metrics["mean_pairwise_code_distance"] >= same_metrics["mean_pairwise_code_distance"] + 0.08
        and independent_metrics["mean_pairwise_code_distance"] >= 0.45,
        "same_mean_pairwise_code_distance": same_metrics["mean_pairwise_code_distance"],
        "independent_mean_pairwise_code_distance": independent_metrics["mean_pairwise_code_distance"],
        "same_mean_pairwise_task_distance": same_metrics["mean_pairwise_task_distance"],
        "independent_mean_pairwise_task_distance": independent_metrics["mean_pairwise_task_distance"],
    }
    summary = {
        "experiment": EXPERIMENT,
        "baseline_records": str(args.baseline_records),
        "library": str(args.library),
        "library_entries": len(library),
        "residual_records": len(rows),
        "residual_task_ids": [row["record"]["task_id"] for row in rows],
        "top_k": args.top_k,
        "candidate_pool": args.candidate_pool,
        "mmr_lambda": args.mmr_lambda,
        "same_neighborhood": same_metrics,
        "independent": independent_metrics,
        "independence_gate": gate,
        "path": str(args.out),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
