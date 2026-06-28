#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


ACTIONS = ["stop", "hot_next4", "constrained4"]


def hidden(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("full_pass"))


def visible(candidate: dict[str, Any]) -> bool:
    return bool(candidate.get("visible_all_pass"))


def parsed(candidate: dict[str, Any]) -> bool:
    return candidate.get("parse_status") == "parsed"


def tokens(candidate: dict[str, Any]) -> int:
    return int(candidate.get("forward_tokens") or 0)


def by_task(path: Path) -> dict[int, dict[str, Any]]:
    return {int(row["task_id"]): row for row in load_jsonl(path)}


def features(record: dict[str, Any], prefix: list[dict[str, Any]]) -> list[float]:
    denom = max(len(prefix), 1)
    public_sigs = {str(c.get("public_signature")) for c in prefix}
    code_lens = [len(str(c.get("code", ""))) for c in prefix]
    comp = [int(c.get("completion_tokens") or 0) for c in prefix]
    return [
        1.0,
        min(len(record.get("task_text", "")) / 240.0, 3.0),
        min(len(record.get("entry_point", "")) / 32.0, 2.0),
        sum(parsed(c) for c in prefix) / denom,
        sum(visible(c) for c in prefix) / denom,
        float(any(visible(c) for c in prefix)),
        len(public_sigs) / denom,
        (sum(code_lens) / denom) / 800.0,
        (sum(comp) / denom) / 220.0,
    ]


def label_for(hot8: dict[str, Any], constrained: dict[str, Any]) -> str:
    hot4 = hot8.get("candidates", [])[:4]
    if any(hidden(c) for c in hot4):
        return "stop"
    hot_next = hot8.get("candidates", [])[4:8]
    constrained4 = constrained.get("candidates", [])[:4]
    hot_success = any(hidden(c) for c in hot4 + hot_next)
    constrained_success = any(hidden(c) for c in hot4 + constrained4)
    if constrained_success and not hot_success:
        return "constrained4"
    if hot_success and not constrained_success:
        return "hot_next4"
    if constrained_success and hot_success:
        return "constrained4" if sum(tokens(c) for c in constrained4) <= sum(tokens(c) for c in hot_next) else "hot_next4"
    # No route succeeds. Prefer not spending more unless visible evidence says the
    # prefix is poor.
    return "hot_next4" if not any(visible(c) for c in hot4) else "stop"


def train_linear(xs: list[list[float]], ys: list[int], seed: int) -> tuple[dict[str, Any], list[dict[str, float]]]:
    torch.manual_seed(seed)
    x = torch.tensor(xs, dtype=torch.float32)
    y = torch.tensor(ys, dtype=torch.long)
    mean = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-4)
    x = (x - mean) / std
    model = torch.nn.Linear(x.shape[1], len(ACTIONS))
    counts = Counter(ys)
    weights = torch.tensor([1.0 / math.sqrt(max(counts.get(i, 0), 1)) for i in range(len(ACTIONS))], dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=0.05, weight_decay=0.01)
    losses = []
    for epoch in range(160):
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, y, weight=weights)
        loss.backward()
        opt.step()
        if epoch in {0, 40, 80, 120, 159}:
            losses.append({"epoch": epoch, "loss": float(loss.detach()), "train_acc": float((logits.argmax(dim=1) == y).float().mean())})
    return {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weight": model.weight.detach().tolist(),
        "bias": model.bias.detach().tolist(),
    }, losses


def predict(model: dict[str, Any], row: list[float]) -> str:
    x = torch.tensor(row, dtype=torch.float32)
    mean = torch.tensor(model["mean"], dtype=torch.float32)
    std = torch.tensor(model["std"], dtype=torch.float32)
    weight = torch.tensor(model["weight"], dtype=torch.float32)
    bias = torch.tensor(model["bias"], dtype=torch.float32)
    logits = torch.mv(weight, (x - mean) / std) + bias
    return ACTIONS[int(logits.argmax().item())]


def schedule(hot8: dict[str, Any], constrained: dict[str, Any], action: str) -> list[dict[str, Any]]:
    hot4 = hot8.get("candidates", [])[:4]
    if action == "stop":
        return hot4
    if action == "hot_next4":
        return hot4 + hot8.get("candidates", [])[4:8]
    if action == "constrained4":
        return hot4 + constrained.get("candidates", [])[:4]
    raise KeyError(action)


def record_metrics(record: dict[str, Any], candidates: list[dict[str, Any]], action: str) -> dict[str, Any]:
    denom = max(len(candidates), 1)
    return {
        "task_id": int(record["task_id"]),
        "action": action,
        "candidate_count": len(candidates),
        "coverage": any(hidden(c) for c in candidates),
        "pass1_proxy": bool(candidates and hidden(candidates[0])),
        "visible_coverage": any(visible(c) for c in candidates),
        "parse_success_count": sum(parsed(c) for c in candidates),
        "distinct_functional_rate": len({str(c.get("functional_signature")) for c in candidates}) / denom,
        "forward_tokens": sum(tokens(c) for c in candidates),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    return {
        "records": n,
        "coverage": sum(r["coverage"] for r in rows) / n,
        "pass1_proxy": sum(r["pass1_proxy"] for r in rows) / n,
        "visible_coverage": sum(r["visible_coverage"] for r in rows) / n,
        "candidate_count_mean": sum(r["candidate_count"] for r in rows) / n,
        "parse_success_mean": sum(r["parse_success_count"] for r in rows) / n,
        "distinct_functional_rate_mean": sum(r["distinct_functional_rate"] for r in rows) / n,
        "forward_tokens": sum(r["forward_tokens"] for r in rows),
        "covered_tasks": sorted(r["task_id"] for r in rows if r["coverage"]),
        "action_counts": dict(Counter(r["action"] for r in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hot8", type=Path, required=True)
    parser.add_argument("--constrained", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-task-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    hot = by_task(args.hot8)
    constrained = by_task(args.constrained)
    task_ids = sorted(set(hot) & set(constrained))
    labels = {task_id: label_for(hot[task_id], constrained[task_id]) for task_id in task_ids}
    rows = []
    all_losses = []
    for heldout in task_ids:
        train_ids = [task_id for task_id in task_ids if task_id != heldout]
        xs = [features(hot[task_id], hot[task_id].get("candidates", [])[:4]) for task_id in train_ids]
        ys = [ACTIONS.index(labels[task_id]) for task_id in train_ids]
        model, losses = train_linear(xs, ys, args.seed + heldout)
        all_losses.append({"heldout_task_id": heldout, "losses": losses})
        action = predict(model, features(hot[heldout], hot[heldout].get("candidates", [])[:4]))
        rows.append(record_metrics(hot[heldout], schedule(hot[heldout], constrained[heldout], action), action))
    payload = {
        "task_count": len(task_ids),
        "oracle_label_counts": dict(Counter(labels.values())),
        "summary": summarize(rows),
        "losses": all_losses,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    write_jsonl(args.per_task_out, rows)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
