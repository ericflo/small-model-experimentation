#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


BLOCKS = ["stop", "hot_next4", "mixed_low4", "mixed_high4", "constrained4"]


def cand_tokens(candidate: dict[str, Any]) -> int:
    return int(candidate.get("forward_tokens") or 0)


def parsed(candidate: dict[str, Any]) -> bool:
    return candidate.get("parse_status") == "parsed"


def visible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("visible_all_pass"))


def hidden(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("full_pass"))


def load_by_task(path: Path) -> dict[int, dict[str, Any]]:
    rows = load_jsonl(path)
    return {int(row["task_id"]): row for row in rows}


def select_by_source(record: dict[str, Any], needles: tuple[str, ...]) -> list[dict[str, Any]]:
    out = []
    for candidate in record.get("candidates", []):
        source = str(candidate.get("source", ""))
        if any(needle in source for needle in needles):
            out.append(candidate)
    return out


class Pools:
    def __init__(self, hot: dict[int, dict[str, Any]], mixed: dict[int, dict[str, Any]], constrained: dict[int, dict[str, Any]]) -> None:
        self.hot = hot
        self.mixed = mixed
        self.constrained = constrained
        self.task_ids = sorted(set(hot) & set(mixed) & set(constrained))

    def prefix(self, task_id: int) -> list[dict[str, Any]]:
        return self.hot[task_id].get("candidates", [])[:2]

    def block(self, task_id: int, block_name: str) -> list[dict[str, Any]]:
        if block_name == "stop":
            return []
        if block_name == "hot_next4":
            return self.hot[task_id].get("candidates", [])[2:6]
        if block_name == "mixed_low4":
            rows = select_by_source(self.mixed[task_id], ("_t0.2_", "_t0.7_"))
            return rows[:4] if rows else self.mixed[task_id].get("candidates", [])[:4]
        if block_name == "mixed_high4":
            rows = select_by_source(self.mixed[task_id], ("_t1_", "_t1.2_"))
            return rows[:4] if rows else self.mixed[task_id].get("candidates", [])[4:8]
        if block_name == "constrained4":
            return self.constrained[task_id].get("candidates", [])[:4]
        raise KeyError(block_name)

    def task_record(self, task_id: int) -> dict[str, Any]:
        return self.hot[task_id]


def schedule_record(record: dict[str, Any], candidates: list[dict[str, Any]], arm_name: str) -> dict[str, Any]:
    out = {
        "record_id": record["record_id"],
        "task_id": record["task_id"],
        "task_text": record["task_text"],
        "entry_point": record["entry_point"],
        "arm_name": arm_name,
        "candidate_count": len(candidates),
        "coverage": any(hidden(c) for c in candidates),
        "pass1_proxy": bool(candidates and hidden(candidates[0])),
        "visible_coverage": any(visible(c) for c in candidates),
        "parse_success_count": sum(1 for c in candidates if parsed(c)),
        "visible_candidate_count": sum(1 for c in candidates if visible(c)),
        "hidden_pass_candidate_count": sum(1 for c in candidates if hidden(c)),
        "forward_tokens": sum(cand_tokens(c) for c in candidates),
        "distinct_functional_count": len({str(c.get("functional_signature")) for c in candidates}),
        "distinct_behavior_count": len({str(c.get("behavior_signature")) for c in candidates}),
        "chosen_sources": [c.get("source") for c in candidates],
    }
    denom = max(len(candidates), 1)
    out["distinct_functional_rate"] = out["distinct_functional_count"] / denom
    out["distinct_behavior_rate"] = out["distinct_behavior_count"] / denom
    return out


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {}
    return {
        "records": n,
        "coverage": sum(1 for r in rows if r["coverage"]) / n,
        "pass1_proxy": sum(1 for r in rows if r["pass1_proxy"]) / n,
        "visible_coverage": sum(1 for r in rows if r["visible_coverage"]) / n,
        "candidate_count_mean": sum(r["candidate_count"] for r in rows) / n,
        "parse_success_mean": sum(r["parse_success_count"] for r in rows) / n,
        "visible_candidates_mean": sum(r["visible_candidate_count"] for r in rows) / n,
        "hidden_pass_candidates_mean": sum(r["hidden_pass_candidate_count"] for r in rows) / n,
        "distinct_functional_rate_mean": sum(r["distinct_functional_rate"] for r in rows) / n,
        "distinct_behavior_rate_mean": sum(r["distinct_behavior_rate"] for r in rows) / n,
        "forward_tokens": sum(r["forward_tokens"] for r in rows),
        "covered_tasks": sorted(int(r["task_id"]) for r in rows if r["coverage"]),
        "pass1_tasks": sorted(int(r["task_id"]) for r in rows if r["pass1_proxy"]),
    }


def evaluate_schedule(pools: Pools, name: str, chooser: Callable[[Pools, int], list[dict[str, Any]]]) -> dict[str, Any]:
    rows = [schedule_record(pools.task_record(task_id), chooser(pools, task_id), name) for task_id in pools.task_ids]
    return {"arm_name": name, "records": summarize(rows), "per_task": rows}


def prefix_features(pools: Pools, task_id: int) -> list[float]:
    record = pools.task_record(task_id)
    prefix = pools.prefix(task_id)
    denom = max(len(prefix), 1)
    public_sigs = {str(c.get("public_signature")) for c in prefix}
    code_lens = [len(str(c.get("code", ""))) for c in prefix]
    completion_tokens = [int(c.get("completion_tokens") or 0) for c in prefix]
    return [
        1.0,
        min(len(record.get("task_text", "")) / 240.0, 3.0),
        min(len(record.get("entry_point", "")) / 32.0, 2.0),
        sum(1 for c in prefix if parsed(c)) / denom,
        sum(1 for c in prefix if visible(c)) / denom,
        float(any(visible(c) for c in prefix)),
        len(public_sigs) / denom,
        (sum(code_lens) / denom) / 800.0,
        (sum(completion_tokens) / denom) / 220.0,
    ]


def label_for_task(pools: Pools, task_id: int) -> str | None:
    prefix = pools.prefix(task_id)
    if any(hidden(c) for c in prefix):
        return "stop"
    candidates: list[tuple[str, bool, int, int]] = []
    for block_name in BLOCKS[1:]:
        block = pools.block(task_id, block_name)
        combined = prefix + block
        candidates.append(
            (
                block_name,
                any(hidden(c) for c in combined),
                sum(cand_tokens(c) for c in block),
                sum(1 for c in block if visible(c)),
            )
        )
    successful = [row for row in candidates if row[1]]
    if successful:
        successful.sort(key=lambda row: (row[2], -row[3], BLOCKS.index(row[0])))
        return successful[0][0]
    # No block reaches hidden correctness. Keep a label so the model learns a fallback,
    # but choose by deployable visible-pass evidence rather than hidden target knowledge.
    candidates.sort(key=lambda row: (-row[3], row[2], BLOCKS.index(row[0])))
    return candidates[0][0] if candidates else None


def train_scheduler(train_pools: Pools, seed: int, out: Path) -> dict[str, Any]:
    random.seed(seed)
    torch.manual_seed(seed)
    xs: list[list[float]] = []
    ys: list[int] = []
    labels: list[str] = []
    for task_id in train_pools.task_ids:
        label = label_for_task(train_pools, task_id)
        if label is None:
            continue
        xs.append(prefix_features(train_pools, task_id))
        ys.append(BLOCKS.index(label))
        labels.append(label)
    if len(xs) < 8:
        raise RuntimeError("not enough scheduler training labels")
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x_norm = (x - mean) / std
    model = torch.nn.Linear(x.shape[1], len(BLOCKS))
    counts = Counter(ys)
    weights = torch.tensor([1.0 / math.sqrt(max(counts.get(i, 0), 1)) for i in range(len(BLOCKS))], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    losses: list[dict[str, float]] = []
    for epoch in range(220):
        opt.zero_grad(set_to_none=True)
        logits = model(x_norm)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == 219:
            pred = logits.argmax(dim=1)
            losses.append({"epoch": epoch, "loss": float(loss.detach()), "train_acc": float((pred == y).float().mean())})
    payload = {
        "blocks": BLOCKS,
        "feature_mean": mean.tolist(),
        "feature_std": std.tolist(),
        "weight": model.weight.detach().tolist(),
        "bias": model.bias.detach().tolist(),
        "label_counts": Counter(labels),
        "losses": losses,
    }
    serializable = dict(payload)
    serializable["label_counts"] = dict(payload["label_counts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, serializable)
    return serializable


def predict_block(model_payload: dict[str, Any], features: list[float]) -> str:
    x = torch.tensor(features, dtype=torch.float32)
    mean = torch.tensor(model_payload["feature_mean"], dtype=torch.float32)
    std = torch.tensor(model_payload["feature_std"], dtype=torch.float32)
    weight = torch.tensor(model_payload["weight"], dtype=torch.float32)
    bias = torch.tensor(model_payload["bias"], dtype=torch.float32)
    logits = torch.mv(weight, (x - mean) / std) + bias
    return BLOCKS[int(logits.argmax().item())]


def candidate_choosers(model_payload: dict[str, Any] | None, static_block: str) -> dict[str, Callable[[Pools, int], list[dict[str, Any]]]]:
    def hot_k4(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        return pools.hot[task_id].get("candidates", [])[:4]

    def hot_k8(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        return pools.hot[task_id].get("candidates", [])[:8]

    def mixed_k8(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        return pools.mixed[task_id].get("candidates", [])[:8]

    def constrained_k4(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        return pools.constrained[task_id].get("candidates", [])[:4]

    def prefix_plus(block_name: str) -> Callable[[Pools, int], list[dict[str, Any]]]:
        return lambda pools, task_id: pools.prefix(task_id) + pools.block(task_id, block_name)

    def round_robin(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        hot = pools.hot[task_id].get("candidates", [])
        mixed = pools.mixed[task_id].get("candidates", [])
        constrained = pools.constrained[task_id].get("candidates", [])
        rows: list[dict[str, Any]] = []
        for idx in range(4):
            if idx < len(hot):
                rows.append(hot[idx])
            if idx < len(constrained):
                rows.append(constrained[idx])
            if idx < len(mixed):
                rows.append(mixed[idx])
            if len(rows) >= 8:
                break
        return rows[:8]

    def oracle_after_prefix(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        prefix = pools.prefix(task_id)
        best_block = "stop"
        best_score = (any(hidden(c) for c in prefix), -sum(cand_tokens(c) for c in prefix), 0)
        for block_name in BLOCKS[1:]:
            block = pools.block(task_id, block_name)
            combined = prefix + block
            score = (any(hidden(c) for c in combined), -sum(cand_tokens(c) for c in combined), -BLOCKS.index(block_name))
            if score > best_score:
                best_score = score
                best_block = block_name
        return prefix + pools.block(task_id, best_block)

    def learned_after_prefix(pools: Pools, task_id: int) -> list[dict[str, Any]]:
        if model_payload is None:
            return prefix_plus(static_block)(pools, task_id)
        block_name = predict_block(model_payload, prefix_features(pools, task_id))
        return pools.prefix(task_id) + pools.block(task_id, block_name)

    return {
        "base_hot_k4": hot_k4,
        "base_hot_k8": hot_k8,
        "base_mixed_k8": mixed_k8,
        "constrained_k4": constrained_k4,
        "hot2_plus_hot_next4": prefix_plus("hot_next4"),
        "hot2_plus_mixed_low4": prefix_plus("mixed_low4"),
        "hot2_plus_mixed_high4": prefix_plus("mixed_high4"),
        "hot2_plus_constrained4": prefix_plus("constrained4"),
        "round_robin_policy_k8": round_robin,
        f"train_selected_{static_block}": prefix_plus(static_block),
        "learned_scheduler_after_hot2": learned_after_prefix,
        "oracle_best_block_after_hot2": oracle_after_prefix,
    }


def best_static_block(train_pools: Pools) -> str:
    best_name = "hot_next4"
    best_score = (-1.0, 0.0)
    for block_name in BLOCKS[1:]:
        rows = [
            schedule_record(
                train_pools.task_record(task_id),
                train_pools.prefix(task_id) + train_pools.block(task_id, block_name),
                block_name,
            )
            for task_id in train_pools.task_ids
        ]
        summary = summarize(rows)
        score = (float(summary["coverage"]), -float(summary["forward_tokens"]))
        if score > best_score:
            best_score = score
            best_name = block_name
    return best_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-hot", type=Path, required=True)
    parser.add_argument("--train-mixed", type=Path, required=True)
    parser.add_argument("--train-constrained", type=Path, required=True)
    parser.add_argument("--eval-hot", type=Path, required=True)
    parser.add_argument("--eval-mixed", type=Path, required=True)
    parser.add_argument("--eval-constrained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task-out", type=Path, required=True)
    parser.add_argument("--scheduler-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    train_pools = Pools(load_by_task(args.train_hot), load_by_task(args.train_mixed), load_by_task(args.train_constrained))
    eval_pools = Pools(load_by_task(args.eval_hot), load_by_task(args.eval_mixed), load_by_task(args.eval_constrained))
    scheduler_payload = train_scheduler(train_pools, args.seed, args.scheduler_out)
    static_block = best_static_block(train_pools)
    chooser_map = candidate_choosers(scheduler_payload, static_block)
    results = []
    per_task_rows = []
    for name, chooser in chooser_map.items():
        result = evaluate_schedule(eval_pools, name, chooser)
        results.append({"arm_name": name, "records": result["records"]})
        for row in result["per_task"]:
            item = dict(row)
            item["portfolio_arm"] = name
            per_task_rows.append(item)
    payload = {
        "scheduler": scheduler_payload,
        "train_task_count": len(train_pools.task_ids),
        "eval_task_count": len(eval_pools.task_ids),
        "train_selected_static_block": static_block,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    write_jsonl(args.per_task_out, per_task_rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
