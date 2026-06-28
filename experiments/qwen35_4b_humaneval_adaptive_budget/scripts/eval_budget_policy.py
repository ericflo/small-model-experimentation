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

from src.budget_policy import MORE_LETTER, STOP_LETTER, evaluate_commit, prompt_for_state, rollout_greedy_until  # noqa: E402
from src.humaneval_env import current_state_metrics, greedy_next_probe, load_jsonl  # noqa: E402
from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["policy"]].append(row)
    summary: list[dict[str, Any]] = []
    for policy, items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "policy": policy,
                "records": n,
                "selected_hidden_correct": sum(1 for row in items if row["selected_hidden_correct"]) / n,
                "target_reachable": sum(1 for row in items if row["target_reachable"]) / n,
                "used_probes_mean": sum(int(row["used_probes"]) for row in items) / n,
                "candidate_count_mean": sum(int(row["candidate_count"]) for row in items) / n,
                "agreement_cluster_count_mean": sum(int(row["agreement_cluster_count"]) for row in items) / n,
                "selected_cluster_fraction_mean": sum(float(row["selected_cluster_fraction"]) for row in items) / n,
                "hidden_correct_survivors_mean": sum(int(row["hidden_correct_survivors"]) for row in items) / n,
            }
        )
    return summary


def run_fixed(record: dict[str, Any], budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    return rollout_greedy_until(record, budget), []


def run_threshold(record: dict[str, Any], threshold: int, max_budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        metrics = current_state_metrics(record, used)
        if 100 * float(metrics["selected_cluster_fraction"]) >= threshold:
            actions.append(
                {
                    "budget": len(used),
                    "action": STOP_LETTER,
                    "candidate_count": metrics["candidate_count"],
                    "selected_cluster_fraction": metrics["selected_cluster_fraction"],
                }
            )
            break
        choice = greedy_next_probe(record, used)
        if choice is None:
            break
        actions.append({"budget": len(used), "action": MORE_LETTER, "chosen_probe_index": int(choice["probe_index"])})
        used.append(int(choice["probe_index"]))
    return used, actions


def run_oracle_stop(record: dict[str, Any], max_budget: int) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        metrics = current_state_metrics(record, used)
        if metrics["selected_hidden_correct"]:
            actions.append({"budget": len(used), "action": STOP_LETTER, "candidate_count": metrics["candidate_count"]})
            break
        choice = greedy_next_probe(record, used)
        if choice is None:
            break
        actions.append({"budget": len(used), "action": MORE_LETTER, "chosen_probe_index": int(choice["probe_index"])})
        used.append(int(choice["probe_index"]))
    return used, actions


@torch.no_grad()
def run_model(
    record: dict[str, Any],
    model: Any,
    tokenizer: Any,
    max_budget: int,
    max_length: int,
) -> tuple[list[int], list[dict[str, Any]]]:
    used: list[int] = []
    actions: list[dict[str, Any]] = []
    while len(used) < max_budget:
        prompt = prompt_for_state(record, used, max_budget=max_budget)
        logits = last_token_action_logits(model, tokenizer, [prompt], max_length=max_length)[0, :2].float()
        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        action = STOP_LETTER if int(torch.argmax(logits).detach().cpu()) == 0 else MORE_LETTER
        metrics = current_state_metrics(record, used)
        actions.append(
            {
                "budget": len(used),
                "action": action,
                "prob_stop": float(probs[0]),
                "prob_more": float(probs[1]),
                "candidate_count": metrics["candidate_count"],
                "agreement_cluster_count": metrics["agreement_cluster_count"],
                "selected_cluster_fraction": metrics["selected_cluster_fraction"],
                "selected_hidden_correct": metrics["selected_hidden_correct"],
            }
        )
        if action == STOP_LETTER:
            break
        choice = greedy_next_probe(record, used)
        if choice is None:
            break
        actions[-1]["chosen_probe_index"] = int(choice["probe_index"])
        used.append(int(choice["probe_index"]))
    return used, actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument("--policy", choices=["fixed", "threshold", "oracle_stop", "base", "adapter"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--fixed-budget", type=int, default=3)
    parser.add_argument("--threshold", type=int, default=1)
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

    rows: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"eval-{args.name}"):
        if args.policy == "fixed":
            used, actions = run_fixed(record, min(args.fixed_budget, args.max_budget))
        elif args.policy == "threshold":
            used, actions = run_threshold(record, args.threshold, args.max_budget)
        elif args.policy == "oracle_stop":
            used, actions = run_oracle_stop(record, args.max_budget)
        else:
            assert model is not None and tokenizer is not None
            used, actions = run_model(record, model, tokenizer, args.max_budget, args.max_length)
        metrics = evaluate_commit(record, used)
        rows.append(
            {
                "policy": args.name,
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "entry_point": record["entry_point"],
                "used_probe_indices": list(used),
                **metrics,
            }
        )
        for action in actions:
            action_rows.append({"policy": args.name, "record_id": record["record_id"], **action})

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
