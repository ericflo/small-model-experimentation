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

from src.code_env import verifier_prompt  # noqa: E402
from src.eval_utils import coverage, prefix_candidates, select_candidate, visible_candidates, visible_coverage  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_quant_model, load_tokenizer  # noqa: E402


def parse_budgets(text: str, records: list[dict[str, Any]]) -> list[int]:
    max_count = max(len(record["candidates"]) for record in records) if records else 0
    budgets: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "max":
            budgets.append(max_count)
        else:
            budgets.append(int(item))
    return sorted(set(budget for budget in budgets if budget > 0))


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["policy"], int(row["budget"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, policy, budget), items in sorted(groups.items()):
        n = len(items)
        cov = sum(row["coverage"] for row in items) / n
        visible_cov = sum(row["visible_coverage"] for row in items) / n
        selected = sum(row["selected_hidden_all"] for row in items) / n
        summary.append(
            {
                "dataset": dataset,
                "policy": policy,
                "budget": budget,
                "records": n,
                "coverage": cov,
                "visible_coverage": visible_cov,
                "selected_hidden_all": selected,
                "coverage_captured": selected / visible_cov if visible_cov else 0.0,
                "visible_candidates_mean": sum(row["visible_candidate_count"] for row in items) / n,
                "sampled_candidates_mean": sum(row["prefix_candidate_count"] for row in items) / n,
            }
        )
    return summary


@torch.no_grad()
def score_record(record: dict[str, Any], model: Any, tokenizer: Any, max_length: int) -> dict[str, float]:
    candidates = [candidate for candidate in record["candidates"] if candidate.get("visible_all_pass")]
    if not candidates:
        return {}
    prompts = [verifier_prompt(record, candidate) for candidate in candidates]
    logits = last_token_action_logits(model, tokenizer, prompts, letters=["A", "B"], max_length=max_length).float()
    scores = (logits[:, 0] - logits[:, 1]).detach().cpu().tolist()
    return {candidate["candidate_id"]: float(score) for candidate, score in zip(candidates, scores)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--policy", choices=["first_visible", "shortest_visible", "random_visible", "public_signature_majority", "oracle_coverage", "base_verifier", "sft_verifier"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="1,2,4,8,16,max")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    budgets = parse_budgets(args.budgets, records)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    model = None
    needs_model = args.policy in {"base_verifier", "sft_verifier"}
    if needs_model:
        tokenizer = load_tokenizer(args.model_path)
        model = load_quant_model(args.model_path, for_training=False)
        if args.policy == "sft_verifier":
            if args.adapter_dir is None:
                raise ValueError("--adapter-dir is required for sft_verifier")
            model = attach_existing_lora(model, args.adapter_dir, is_trainable=False)
        model.eval()

    rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"eval-{args.name}"):
        scores: dict[str, float] | None = None
        selection_policy = args.policy
        if needs_model:
            assert model is not None and tokenizer is not None
            scores = score_record(record, model, tokenizer, args.max_length)
            selection_policy = "model"
            for candidate_id, score in scores.items():
                candidate = next(item for item in record["candidates"] if item["candidate_id"] == candidate_id)
                score_rows.append(
                    {
                        "policy": args.name,
                        "dataset": record["dataset"],
                        "record_id": record["record_id"],
                        "candidate_id": candidate_id,
                        "score": score,
                        "full_pass": bool(candidate.get("full_pass")),
                        "visible_all_pass": bool(candidate.get("visible_all_pass")),
                    }
                )
        for budget in budgets:
            candidates = prefix_candidates(record, budget)
            selected = select_candidate(candidates, selection_policy, rng, scores=scores)
            visible = visible_candidates(candidates)
            rows.append(
                {
                    "dataset": record["dataset"],
                    "record_id": record["record_id"],
                    "task_id": record["task_id"],
                    "policy": args.name,
                    "budget": budget,
                    "prefix_candidate_count": len(candidates),
                    "visible_candidate_count": len(visible),
                    "coverage": coverage(candidates),
                    "visible_coverage": visible_coverage(candidates),
                    "selected_candidate_id": selected["candidate_id"] if selected else None,
                    "selected_source": selected["source"] if selected else None,
                    "selected_hidden_all": bool(selected and selected.get("full_pass")),
                }
            )

    payload = {
        "name": args.name,
        "policy": args.policy,
        "records_path": str(args.records),
        "budgets": budgets,
        "records": rows,
        "candidate_scores": score_rows,
        "summary": summarize(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows), "score_rows": len(score_rows)}, indent=2))


if __name__ == "__main__":
    main()

