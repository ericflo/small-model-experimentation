#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.budget_policy import (  # noqa: E402
    MORE_LETTER,
    STOP_LETTER,
    current_state_metrics,
    evaluate_commit,
    prompt_for_state,
    rollout_greedy_until,
)
from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402
from src.operator_env import build_operator_library, full_pool_choice, load_jsonl  # noqa: E402


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["library_size"], row["template"])].append(row)
    summary: list[dict[str, Any]] = []
    for (policy, library_size, template), items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "policy": policy,
                "library_size": library_size,
                "template": template,
                "records": n,
                "selected_hidden_all": sum(1 for row in items if row["selected_hidden_all"]) / n,
                "selected_exact_pair": sum(1 for row in items if row.get("selected_pair") == row.get("target_pair")) / n,
                "used_probes_mean": sum(int(row["used_probes"]) for row in items) / n,
                "candidate_count_mean": sum(int(row["candidate_count"]) for row in items) / n,
                "hidden_equivalent_candidates_mean": sum(int(row["hidden_equivalent_candidates"]) for row in items) / n,
            }
        )
    return summary


def run_fixed(record: dict[str, Any], operators: Any, budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    used = rollout_greedy_until(record, operators, budget)
    return used, []


def run_threshold(record: dict[str, Any], operators: Any, threshold: int, max_budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        metrics = current_state_metrics(record, operators, used)
        if int(metrics["candidate_count"]) <= threshold:
            actions.append({"budget": len(used), "action": STOP_LETTER, "candidate_count": metrics["candidate_count"]})
            break
        choice = full_pool_choice(record, operators, used, "fullpool_max_split")
        actions.append({"budget": len(used), "action": MORE_LETTER, "chosen_query_index": int(choice["query_index"])})
        used.append(int(choice["query_index"]))
    return used, actions


def run_oracle_stop(record: dict[str, Any], operators: Any, max_budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        metrics = current_state_metrics(record, operators, used)
        if metrics["selected_hidden_all"]:
            actions.append({"budget": len(used), "action": STOP_LETTER, "candidate_count": metrics["candidate_count"]})
            break
        choice = full_pool_choice(record, operators, used, "fullpool_max_split")
        actions.append({"budget": len(used), "action": MORE_LETTER, "chosen_query_index": int(choice["query_index"])})
        used.append(int(choice["query_index"]))
    return used, actions


@torch.no_grad()
def run_model(
    record: dict[str, Any],
    operators: Any,
    model: Any,
    tokenizer: Any,
    max_budget: int,
    max_length: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        prompt = prompt_for_state(record, operators, used, max_budget=max_budget)
        logits = last_token_action_logits(model, tokenizer, [prompt], max_length=max_length)[0, :2].float()
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        action = STOP_LETTER if int(torch.argmax(logits).detach().cpu()) == 0 else MORE_LETTER
        metrics = current_state_metrics(record, operators, used)
        actions.append(
            {
                "budget": len(used),
                "action": action,
                "prob_stop": float(probs[0]),
                "prob_more": float(probs[1]),
                "candidate_count": metrics["candidate_count"],
                "selected_hidden_all": metrics["selected_hidden_all"],
            }
        )
        if action == STOP_LETTER:
            break
        choice = full_pool_choice(record, operators, used, "fullpool_max_split")
        actions[-1]["chosen_query_index"] = int(choice["query_index"])
        used.append(int(choice["query_index"]))
    return used, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument(
        "--policy",
        choices=["fixed", "threshold", "oracle_stop", "base", "adapter"],
        required=True,
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--fixed-budget", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=100)
    parser.add_argument("--max-budget", type=int, default=10)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.limit:
        records = records[: args.limit]
    out = args.out or (ROOT / "reports" / "eval" / f"{args.name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    model = None
    if args.policy in {"base", "adapter"}:
        tokenizer = load_tokenizer(args.model_path)
        model = load_base_model(args.model_path, for_training=False)
        if args.policy == "adapter":
            if args.adapter_dir is None:
                raise ValueError("--adapter-dir is required for adapter policy")
            model = attach_existing_lora(model, args.adapter_dir, is_trainable=False)
        model.eval()

    operator_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"eval-{args.name}"):
        operators = operator_cache.setdefault(record["library_size"], build_operator_library(record["library_size"]))
        if args.policy == "fixed":
            used, actions = run_fixed(record, operators, min(args.fixed_budget, args.max_budget))
        elif args.policy == "threshold":
            used, actions = run_threshold(record, operators, args.threshold, args.max_budget)
        elif args.policy == "oracle_stop":
            used, actions = run_oracle_stop(record, operators, args.max_budget)
        else:
            assert model is not None and tokenizer is not None
            used, actions = run_model(record, operators, model, tokenizer, args.max_budget, args.max_length)
        metrics = evaluate_commit(record, operators, used)
        rows.append(
            {
                "policy": args.name,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "target_pair": list(record["target_pair"]),
                "used_query_indices": list(used),
                **metrics,
            }
        )
        for action in actions:
            action_rows.append(
                {
                    "policy": args.name,
                    "record_id": record["record_id"],
                    "library_size": record["library_size"],
                    "template": record["template"],
                    **action,
                }
            )

    payload = {
        "name": args.name,
        "policy": args.policy,
        "fixed_budget": args.fixed_budget,
        "threshold": args.threshold,
        "max_budget": args.max_budget,
        "records": rows,
        "actions": action_rows,
        "summary": summarize(rows),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(rows), "actions": len(action_rows)}, indent=2))


if __name__ == "__main__":
    main()
