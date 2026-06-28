#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402
from src.operator_env import (  # noqa: E402
    LETTERS,
    action_diagnostics,
    build_operator_library,
    candidate_mask,
    first_prior_pair,
    hidden_equivalent_count,
    load_jsonl,
    pair_hidden_matches,
)
from src.prompts import process_prompt  # noqa: E402


def available_queries(record: dict[str, Any], used: list[int]) -> list[int]:
    used_set = set(used)
    return [idx for idx in range(len(record["query_pool"])) if idx not in used_set][: len(LETTERS)]


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


def choose_nonmodel_action(policy: str, diag: dict[str, Any], rng: random.Random) -> str:
    if policy == "random":
        return rng.choice(LETTERS)
    if policy == "oracle":
        return diag["oracle_action"]
    if policy == "max_split":
        best = min(
            diag["actions"],
            key=lambda row: (float(row["expected_remaining"]), int(row["largest"]), -float(row["entropy"]), row["letter"]),
        )
        return best["letter"]
    raise ValueError(policy)


@torch.no_grad()
def choose_model_action(
    model: Any,
    tokenizer: Any,
    record: dict[str, Any],
    diag: dict[str, Any],
    max_length: int,
    scramble_features: bool,
    rng: random.Random,
) -> tuple[str, list[float]]:
    permutation = None
    if scramble_features:
        permutation = list(range(len(LETTERS)))
        rng.shuffle(permutation)
    prompt = process_prompt(record, diag, feature_permutation=permutation)
    logits = last_token_action_logits(model, tokenizer, [prompt], max_length=max_length)[0].float()
    probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
    return LETTERS[int(torch.argmax(logits).detach().cpu())], probs


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["policy"], row["library_size"], row["template"], row["budget"])].append(row)
    summary: list[dict[str, Any]] = []
    for (policy, library_size, template, budget), items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "policy": policy,
                "library_size": library_size,
                "template": template,
                "budget": budget,
                "records": n,
                "selected_hidden_all": sum(1 for x in items if x["selected_hidden_all"]) / n,
                "selected_exact_pair": sum(1 for x in items if x["selected_exact_pair"]) / n,
                "target_reachable": sum(1 for x in items if x["target_reachable"]) / n,
                "candidate_count_mean": sum(x["candidate_count"] for x in items) / n,
                "hidden_equivalent_candidates_mean": sum(x["hidden_equivalent_candidates"] for x in items) / n,
                "mean_chosen_reward": sum(x.get("chosen_reward", 0.0) for x in items if x["budget"] > 0) / max(sum(1 for x in items if x["budget"] > 0), 1),
            }
        )
    return {"rows": len(rows), "summary": summary}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=ROOT / "data" / "eval_records.jsonl")
    parser.add_argument("--policy", choices=["random", "max_split", "oracle", "base", "adapter"], required=True)
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-budget", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--scramble-features", action="store_true")
    parser.add_argument("--seed", type=int, default=20260626)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    records = load_jsonl(args.records)
    if args.limit:
        records = records[: args.limit]
    output_path = args.out or (ROOT / "reports" / "eval" / f"{args.name}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

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
        used: list[int] = []
        base_metrics = evaluate_state(record, operators, used)
        rows.append(
            {
                "policy": args.name,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "budget": 0,
                "used_query_indices": list(used),
                **base_metrics,
            }
        )
        for step in range(args.max_budget):
            queries = available_queries(record, used)
            diag = action_diagnostics(record, operators, queries, used)
            if args.policy in {"random", "max_split", "oracle"}:
                letter = choose_nonmodel_action(args.policy, diag, rng)
                probs = None
            else:
                assert model is not None and tokenizer is not None
                letter, probs = choose_model_action(model, tokenizer, record, diag, args.max_length, args.scramble_features, rng)
            action_index = LETTERS.index(letter)
            chosen = diag["actions"][action_index]
            used.append(int(chosen["query_index"]))
            metrics = evaluate_state(record, operators, used)
            row = {
                "policy": args.name,
                "record_id": record["record_id"],
                "library_size": record["library_size"],
                "template": record["template"],
                "budget": step + 1,
                "used_query_indices": list(used),
                "chosen_action": letter,
                "chosen_query_index": int(chosen["query_index"]),
                "chosen_reward": float(chosen["reward"]),
                "oracle_action": diag["oracle_action"],
                "oracle_reward": max(float(a["reward"]) for a in diag["actions"]),
                "action_matches_oracle": letter == diag["oracle_action"],
                **metrics,
            }
            rows.append(row)
            action_rows.append(
                {
                    "policy": args.name,
                    "record_id": record["record_id"],
                    "library_size": record["library_size"],
                    "template": record["template"],
                    "step": step,
                    "chosen_action": letter,
                    "oracle_action": diag["oracle_action"],
                    "chosen_reward": float(chosen["reward"]),
                    "oracle_reward": max(float(a["reward"]) for a in diag["actions"]),
                    "candidate_count_before": int(diag["candidate_count"]),
                    "probs": probs,
                }
            )

    payload = {
        "name": args.name,
        "policy": args.policy,
        "adapter_dir": str(args.adapter_dir) if args.adapter_dir else None,
        "scramble_features": args.scramble_features,
        "records": rows,
        "actions": action_rows,
        "summary": summarize(rows)["summary"],
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(output_path), "summary": payload["summary"][:4], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

