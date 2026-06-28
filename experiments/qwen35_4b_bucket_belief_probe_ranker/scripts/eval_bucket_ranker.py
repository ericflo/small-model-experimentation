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

from src.bucket_belief import make_bucket_example, predicted_expected_survivors, top_split_queries  # noqa: E402
from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402
from src.operator_env import (  # noqa: E402
    build_operator_library,
    candidate_mask,
    first_prior_pair,
    full_pool_choice,
    hidden_equivalent_count,
    load_jsonl,
    pair_hidden_matches,
)


def evaluate_state(record: dict[str, Any], operators: Any, used: list[int]) -> dict[str, Any]:
    mask = candidate_mask(record, operators, used)
    selected = first_prior_pair(mask)
    target_pair = tuple(record["target_pair"])
    return {
        "candidate_count": int(mask.sum()),
        "target_reachable": bool(mask[target_pair]),
        "selected_pair": list(selected) if selected is not None else None,
        "selected_exact_pair": bool(selected == target_pair),
        "selected_hidden_all": pair_hidden_matches(record, operators, selected),
        "hidden_equivalent_candidates": hidden_equivalent_count(record, operators, mask),
    }


def choose_nonmodel(
    policy: str,
    record: dict[str, Any],
    operators: Any,
    used: list[int],
    candidate_count: int,
    max_options: int,
) -> dict[str, Any]:
    if policy == "fullpool_oracle":
        choice = full_pool_choice(record, operators, used, "fullpool_oracle")
        return {
            "query_index": int(choice["query_index"]),
            "chosen_reward": choice["reward"],
            "predicted_expected_survivors": None,
            "oracle_rank": None,
            "bucket_label": None,
        }

    queries = top_split_queries(record, operators, used, candidate_count)
    if policy == "split_top1":
        chosen = make_bucket_example(record, operators, used, queries[0], max_options=max_options)
        oracle_rank = 1
    elif policy == "oracle_topk":
        examples = [make_bucket_example(record, operators, used, qidx, max_options=max_options) for qidx in queries]
        chosen = max(examples, key=lambda row: (float(row["reward"]), -float(row["expected_remaining"]), -int(row["query_index"])))
        oracle_rank = examples.index(chosen) + 1
    else:
        raise ValueError(policy)
    return {
        "query_index": int(chosen["query_index"]),
        "chosen_reward": float(chosen["reward"]),
        "predicted_expected_survivors": None,
        "oracle_rank": oracle_rank,
        "bucket_label": chosen["label"],
    }


@torch.no_grad()
def choose_bucket_model(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    operators: Any,
    used: list[int],
    candidate_count: int,
    max_options: int,
    max_length: int,
) -> dict[str, Any]:
    queries = top_split_queries(record, operators, used, candidate_count)
    examples = [make_bucket_example(record, operators, used, qidx, max_options=max_options) for qidx in queries]
    logits = last_token_action_logits(model, tokenizer, [row["prompt"] for row in examples], max_length=max_length).float()
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    scored: list[dict[str, Any]] = []
    for rank, (example, prob) in enumerate(zip(examples, probs), start=1):
        scored.append(
            {
                "query_index": int(example["query_index"]),
                "split_rank": rank,
                "bucket_label": example["label"],
                "chosen_reward": float(example["reward"]),
                "true_survivors": int(example["survivors_if_taken"]),
                "expected_remaining": float(example["expected_remaining"]),
                "predicted_expected_survivors": float(predicted_expected_survivors(example, prob)),
                "label_probability": float(prob[int(example["label_index"])]),
                "top_probability": float(max(prob)),
                "top_letter": "ABCDEFGH"[int(torch.tensor(prob).argmax().item())],
            }
        )
    best = min(
        scored,
        key=lambda row: (
            float(row["predicted_expected_survivors"]),
            float(row["expected_remaining"]),
            int(row["split_rank"]),
        ),
    )
    oracle = max(scored, key=lambda row: (float(row["chosen_reward"]), -float(row["expected_remaining"])))
    return {
        "query_index": int(best["query_index"]),
        "chosen_reward": float(best["chosen_reward"]),
        "predicted_expected_survivors": float(best["predicted_expected_survivors"]),
        "oracle_rank": int(oracle["split_rank"]),
        "bucket_label": best["bucket_label"],
        "model_choice_split_rank": int(best["split_rank"]),
        "model_choice_label_probability": float(best["label_probability"]),
        "model_choice_top_probability": float(best["top_probability"]),
        "model_choice_top_letter": best["top_letter"],
        "oracle_query_index": int(oracle["query_index"]),
        "oracle_reward": float(oracle["chosen_reward"]),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["library_size"], row["template"], row["budget"])].append(row)
    summary: list[dict[str, Any]] = []
    for (policy, library_size, template, budget), items in sorted(groups.items()):
        n = len(items)
        reward_items = [row for row in items if row.get("chosen_reward") is not None]
        summary.append(
            {
                "policy": policy,
                "library_size": library_size,
                "template": template,
                "budget": budget,
                "records": n,
                "selected_hidden_all": sum(1 for row in items if row["selected_hidden_all"]) / n,
                "selected_exact_pair": sum(1 for row in items if row["selected_exact_pair"]) / n,
                "target_reachable": sum(1 for row in items if row["target_reachable"]) / n,
                "candidate_count_mean": sum(int(row["candidate_count"]) for row in items) / n,
                "hidden_equivalent_candidates_mean": sum(int(row["hidden_equivalent_candidates"]) for row in items) / n,
                "mean_chosen_reward": sum(float(row["chosen_reward"]) for row in reward_items) / max(len(reward_items), 1),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument(
        "--policy",
        choices=["split_top1", "oracle_topk", "fullpool_oracle", "base_bucket", "adapter_bucket"],
        required=True,
    )
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--max-options", type=int, default=8)
    parser.add_argument("--max-budget", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.limit:
        records = records[: args.limit]
    output_path = args.out or (ROOT / "reports" / "eval" / f"{args.name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    model = None
    if args.policy in {"base_bucket", "adapter_bucket"}:
        tokenizer = load_tokenizer(args.model_path)
        model = load_base_model(args.model_path, for_training=False)
        if args.policy == "adapter_bucket":
            if args.adapter_dir is None:
                raise ValueError("--adapter-dir is required for adapter_bucket")
            model = attach_existing_lora(model, args.adapter_dir, is_trainable=False)
        model.eval()

    operator_cache: dict[int, Any] = {}
    rows: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"eval-{args.name}"):
        operators = operator_cache.setdefault(record["library_size"], build_operator_library(record["library_size"]))
        used: list[int] = []
        rows.append(
            {
                "policy": args.name,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "budget": 0,
                "used_query_indices": list(used),
                **evaluate_state(record, operators, used),
            }
        )
        for step in range(args.max_budget):
            if args.policy in {"base_bucket", "adapter_bucket"}:
                assert model is not None and tokenizer is not None
                choice = choose_bucket_model(
                    model,
                    tokenizer,
                    record,
                    operators,
                    used,
                    args.candidate_count,
                    args.max_options,
                    args.max_length,
                )
            else:
                choice = choose_nonmodel(args.policy, record, operators, used, args.candidate_count, args.max_options)
            used.append(int(choice["query_index"]))
            metrics = evaluate_state(record, operators, used)
            row = {
                "policy": args.name,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "budget": step + 1,
                "used_query_indices": list(used),
                "candidate_probe_count": args.candidate_count if args.policy != "fullpool_oracle" else len(record["query_pool"]) - step,
                **choice,
                **metrics,
            }
            rows.append(row)
            actions.append(
                {
                    "policy": args.name,
                    "record_id": record["record_id"],
                    "library_size": record["library_size"],
                    "template": record["template"],
                    "step": step,
                    **choice,
                }
            )

    payload = {
        "name": args.name,
        "policy": args.policy,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "candidate_count": args.candidate_count,
        "max_options": args.max_options,
        "records": rows,
        "actions": actions,
        "summary": summarize(rows),
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(output_path), "summary_rows": len(payload["summary"]), "records": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
