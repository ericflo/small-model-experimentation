#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


SCHEDULER_BLOCKS = ["stop", "low4", "mid4", "high4"]


def hidden(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("full_pass"))


def visible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("visible_all_pass"))


def parsed(candidate: dict[str, Any]) -> bool:
    return candidate.get("parse_status") == "parsed"


def tokens(candidate: dict[str, Any]) -> int:
    return int(candidate.get("forward_tokens") or 0)


def temp_of(candidate: dict[str, Any]) -> float | None:
    match = re.search(r"_t([0-9.]+)_", str(candidate.get("source", "")))
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def source_prefix(candidate: dict[str, Any]) -> str:
    source = str(candidate.get("source", ""))
    return source.split("_t")[0] if "_t" in source else source


def by_task(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in load_jsonl(path)}


def select(record: dict[str, Any], selector: Callable[[dict[str, Any]], bool], limit: int) -> list[dict[str, Any]]:
    return [candidate for candidate in record.get("candidates", []) if selector(candidate)][:limit]


def low_block(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    return select(record, lambda c: (temp_of(c) is not None and temp_of(c) <= 0.4) and source_prefix(c) != "base_default", limit)


def mid_block(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    return select(record, lambda c: temp_of(c) is not None and 0.6 <= temp_of(c) <= 0.9 and source_prefix(c) != "base_default", limit)


def high_block(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    return select(record, lambda c: temp_of(c) is not None and temp_of(c) >= 1.0 and source_prefix(c) != "base_default", limit)


def diverse_block(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    return select(record, lambda c: source_prefix(c) == "diverse_extra", limit)


def hot_source_block(record: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    return select(record, lambda c: source_prefix(c) == "hot_extra", limit)


def default_source_block(record: dict[str, Any], limit: int = 8) -> list[dict[str, Any]]:
    return select(record, lambda c: source_prefix(c) == "default_extra", limit)


def base_prefix(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    rows = select(record, lambda c: source_prefix(c) == "base_default", limit)
    return rows if rows else record.get("candidates", [])[:limit]


def base_prefix_train(record: dict[str, Any], limit: int = 4) -> list[dict[str, Any]]:
    # Train pools have a single source prefix, so the first low-temperature samples
    # play the same role as the base/default prefix.
    return record.get("candidates", [])[:limit]


def scheduler_block(record: dict[str, Any], name: str, train_mode: bool) -> list[dict[str, Any]]:
    if name == "stop":
        return []
    if name == "low4":
        rows = low_block(record)
    elif name == "mid4":
        rows = mid_block(record)
    elif name == "high4":
        rows = high_block(record)
    else:
        raise KeyError(name)
    if rows:
        return rows
    # For train pools, source prefixes are not separated. Fall back to temperature
    # bands including the original source prefix.
    if train_mode:
        if name == "low4":
            return select(record, lambda c: temp_of(c) is not None and temp_of(c) <= 0.4, 4)
        if name == "mid4":
            return select(record, lambda c: temp_of(c) is not None and 0.6 <= temp_of(c) <= 0.9, 4)
        if name == "high4":
            return select(record, lambda c: temp_of(c) is not None and temp_of(c) >= 1.0, 4)
    return []


def metrics_for(record: dict[str, Any], candidates: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    denom = max(len(candidates), 1)
    return {
        "record_id": record["record_id"],
        "task_id": int(record["task_id"]),
        "arm_name": arm,
        "candidate_count": len(candidates),
        "coverage": any(hidden(c) for c in candidates),
        "pass1_proxy": bool(candidates and hidden(candidates[0])),
        "visible_coverage": any(visible(c) for c in candidates),
        "parse_success_count": sum(1 for c in candidates if parsed(c)),
        "visible_candidate_count": sum(1 for c in candidates if visible(c)),
        "hidden_pass_candidate_count": sum(1 for c in candidates if hidden(c)),
        "distinct_functional_rate": len({str(c.get("functional_signature")) for c in candidates}) / denom,
        "distinct_behavior_rate": len({str(c.get("behavior_signature")) for c in candidates}) / denom,
        "forward_tokens": sum(tokens(c) for c in candidates),
        "sources": [c.get("source") for c in candidates],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {}
    return {
        "records": n,
        "coverage": sum(r["coverage"] for r in rows) / n,
        "pass1_proxy": sum(r["pass1_proxy"] for r in rows) / n,
        "visible_coverage": sum(r["visible_coverage"] for r in rows) / n,
        "candidate_count_mean": sum(r["candidate_count"] for r in rows) / n,
        "parse_success_mean": sum(r["parse_success_count"] for r in rows) / n,
        "visible_candidates_mean": sum(r["visible_candidate_count"] for r in rows) / n,
        "hidden_pass_candidates_mean": sum(r["hidden_pass_candidate_count"] for r in rows) / n,
        "distinct_functional_rate_mean": sum(r["distinct_functional_rate"] for r in rows) / n,
        "distinct_behavior_rate_mean": sum(r["distinct_behavior_rate"] for r in rows) / n,
        "forward_tokens": sum(r["forward_tokens"] for r in rows),
        "covered_tasks": sorted(r["task_id"] for r in rows if r["coverage"]),
        "pass1_tasks": sorted(r["task_id"] for r in rows if r["pass1_proxy"]),
    }


def features(record: dict[str, Any], prefix: list[dict[str, Any]]) -> list[float]:
    denom = max(len(prefix), 1)
    public_sigs = {str(c.get("public_signature")) for c in prefix}
    code_lens = [len(str(c.get("code", ""))) for c in prefix]
    completion_tokens = [int(c.get("completion_tokens") or 0) for c in prefix]
    return [
        1.0,
        min(len(record.get("task_text", "")) / 240.0, 3.0),
        min(len(record.get("entry_point", "")) / 32.0, 2.0),
        sum(parsed(c) for c in prefix) / denom,
        sum(visible(c) for c in prefix) / denom,
        float(any(visible(c) for c in prefix)),
        len(public_sigs) / denom,
        (sum(code_lens) / denom) / 800.0,
        (sum(completion_tokens) / denom) / 220.0,
    ]


def label_for(record: dict[str, Any]) -> str:
    prefix = base_prefix_train(record, 2)
    if any(hidden(c) for c in prefix):
        return "stop"
    candidates: list[tuple[str, bool, int, int]] = []
    for block in SCHEDULER_BLOCKS[1:]:
        rows = scheduler_block(record, block, train_mode=True)
        combined = prefix + rows
        candidates.append((block, any(hidden(c) for c in combined), sum(tokens(c) for c in rows), sum(visible(c) for c in rows)))
    successful = [row for row in candidates if row[1]]
    if successful:
        successful.sort(key=lambda row: (row[2], -row[3], SCHEDULER_BLOCKS.index(row[0])))
        return successful[0][0]
    candidates.sort(key=lambda row: (-row[3], row[2], SCHEDULER_BLOCKS.index(row[0])))
    return candidates[0][0]


def train_scheduler(train_rows: list[dict[str, Any]], out: Path, seed: int) -> dict[str, Any]:
    torch.manual_seed(seed)
    xs = []
    ys = []
    labels = []
    for record in train_rows:
        label = label_for(record)
        xs.append(features(record, base_prefix_train(record, 2)))
        ys.append(SCHEDULER_BLOCKS.index(label))
        labels.append(label)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x = (x - mean) / std
    model = torch.nn.Linear(x.shape[1], len(SCHEDULER_BLOCKS))
    counts = Counter(ys)
    weights = torch.tensor([1.0 / math.sqrt(max(counts.get(i, 0), 1)) for i in range(len(SCHEDULER_BLOCKS))], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    losses = []
    for epoch in range(220):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 219:
            losses.append(
                {
                    "epoch": epoch,
                    "loss": float(loss.detach()),
                    "train_acc": float((logits.argmax(dim=1) == y).float().mean()),
                }
            )
    payload = {
        "blocks": SCHEDULER_BLOCKS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weight": model.weight.detach().tolist(),
        "bias": model.bias.detach().tolist(),
        "label_counts": dict(Counter(labels)),
        "losses": losses,
    }
    write_json(out, payload)
    return payload


def predict(payload: dict[str, Any], row: list[float]) -> str:
    x = torch.tensor(row, dtype=torch.float32)
    mean = torch.tensor(payload["feature_mean"], dtype=torch.float32)
    std = torch.tensor(payload["feature_std"], dtype=torch.float32)
    weight = torch.tensor(payload["weight"], dtype=torch.float32)
    bias = torch.tensor(payload["bias"], dtype=torch.float32)
    logits = torch.mv(weight, (x - mean) / std) + bias
    return SCHEDULER_BLOCKS[int(logits.argmax().item())]


def evaluate(name: str, eval_rows: list[dict[str, Any]], chooser: Callable[[dict[str, Any]], list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_task = [metrics_for(record, chooser(record), name) for record in eval_rows]
    return {"arm_name": name, "records": summarize(per_task)}, per_task


def oracle_after_prefix(record: dict[str, Any]) -> list[dict[str, Any]]:
    prefix = base_prefix(record, 2)
    best_name = "stop"
    best_score = (any(hidden(c) for c in prefix), -sum(tokens(c) for c in prefix), 0)
    for block in ["low4", "mid4", "high4", "diverse4"]:
        if block == "diverse4":
            rows = diverse_block(record, 4)
        else:
            rows = scheduler_block(record, block, train_mode=False)
        combined = prefix + rows
        score = (any(hidden(c) for c in combined), -sum(tokens(c) for c in combined), -len(rows))
        if score > best_score:
            best_score = score
            best_name = block
    if best_name == "diverse4":
        return prefix + diverse_block(record, 4)
    return prefix + scheduler_block(record, best_name, train_mode=False)


def round_robin(record: dict[str, Any]) -> list[dict[str, Any]]:
    blocks = [
        base_prefix(record, 2),
        default_source_block(record, 2),
        hot_source_block(record, 2),
        diverse_block(record, 2),
    ]
    rows: list[dict[str, Any]] = []
    for idx in range(2):
        for block in blocks:
            if idx < len(block):
                rows.append(block[idx])
    return rows[:8]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pool", type=Path, required=True)
    parser.add_argument("--eval-pool", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task-out", type=Path, required=True)
    parser.add_argument("--scheduler-out", type=Path, required=True)
    parser.add_argument("--eval-limit", type=int)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    train_rows = load_jsonl(args.train_pool)
    eval_rows = load_jsonl(args.eval_pool)
    if args.eval_limit:
        eval_rows = eval_rows[: args.eval_limit]
    scheduler = train_scheduler(train_rows, args.scheduler_out, args.seed)

    learned_counts: Counter[str] = Counter()

    def learned(record: dict[str, Any]) -> list[dict[str, Any]]:
        prefix = base_prefix(record, 2)
        block = predict(scheduler, features(record, prefix))
        learned_counts[block] += 1
        return prefix + scheduler_block(record, block, train_mode=False)

    arms: list[tuple[str, Callable[[dict[str, Any]], list[dict[str, Any]]]]] = [
        ("base_prefix_k4", lambda r: base_prefix(r, 4)),
        ("default_extra_k8", lambda r: default_source_block(r, 8)),
        ("hot_extra_k8", lambda r: hot_source_block(r, 8)),
        ("diverse_extra_k8", lambda r: diverse_block(r, 8)),
        ("low_mid_high_static_k8", lambda r: base_prefix(r, 2) + low_block(r, 2) + mid_block(r, 2) + high_block(r, 2)),
        ("source_round_robin_k8", round_robin),
        ("prefix2_low4", lambda r: base_prefix(r, 2) + low_block(r, 4)),
        ("prefix2_mid4", lambda r: base_prefix(r, 2) + mid_block(r, 4)),
        ("prefix2_high4", lambda r: base_prefix(r, 2) + high_block(r, 4)),
        ("learned_scheduler_after_prefix2", learned),
        ("oracle_best_block_after_prefix2", oracle_after_prefix),
        ("full_union_all_candidates", lambda r: r.get("candidates", [])),
    ]

    results = []
    per_task_rows = []
    for name, chooser in arms:
        summary, per_task = evaluate(name, eval_rows, chooser)
        results.append(summary)
        per_task_rows.extend(per_task)
    payload = {
        "train_records": len(train_rows),
        "eval_records": len(eval_rows),
        "scheduler": scheduler,
        "learned_action_counts": dict(learned_counts),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    write_jsonl(args.per_task_out, per_task_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
