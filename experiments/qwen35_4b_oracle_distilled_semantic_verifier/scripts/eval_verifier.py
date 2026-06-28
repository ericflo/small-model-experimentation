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

from src.code_env import candidate_prompt, load_jsonl, record_coverage, visible_candidates  # noqa: E402
from src.model_policy import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_base_model, load_tokenizer  # noqa: E402


def select_candidate(record: dict[str, Any], policy: str, rng: random.Random, scores: dict[str, float] | None = None) -> dict[str, Any] | None:
    candidates = visible_candidates(record)
    if not candidates:
        return None
    if policy == "first":
        return candidates[0]
    if policy == "shortest":
        return sorted(candidates, key=lambda c: (len(c["code"]), c["candidate_id"]))[0]
    if policy == "random":
        return rng.choice(candidates)
    if policy == "oracle":
        positives = [candidate for candidate in candidates if candidate["hidden_all_pass"]]
        return positives[0] if positives else candidates[0]
    if policy == "model":
        assert scores is not None
        return sorted(candidates, key=lambda c: (-scores[c["candidate_id"]], c["candidate_id"]))[0]
    raise ValueError(f"unknown policy: {policy}")


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["policy"])].append(row)
    summary: list[dict[str, Any]] = []
    for (dataset, policy), items in sorted(groups.items()):
        n = len(items)
        coverage = sum(1 for row in items if row["candidate_pool_coverage"]) / n
        selected = sum(1 for row in items if row["selected_hidden_all"]) / n
        summary.append(
            {
                "dataset": dataset,
                "policy": policy,
                "records": n,
                "candidate_pool_coverage": coverage,
                "selected_hidden_all": selected,
                "coverage_captured": selected / coverage if coverage else 0.0,
                "visible_candidates_mean": sum(row["visible_candidate_count"] for row in items) / n,
                "hidden_pass_visible_candidates_mean": sum(row["hidden_pass_visible_candidates"] for row in items) / n,
            }
        )
    return summary


@torch.no_grad()
def model_scores(record: dict[str, Any], model: Any, tokenizer: Any, max_length: int) -> dict[str, float]:
    candidates = visible_candidates(record)
    prompts = [candidate_prompt(record, candidate) for candidate in candidates]
    logits = last_token_action_logits(model, tokenizer, prompts, max_length=max_length).float()
    scores = (logits[:, 0] - logits[:, 1]).detach().cpu().tolist()
    return {candidate["candidate_id"]: float(score) for candidate, score in zip(candidates, scores)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--policy", choices=["first", "shortest", "random", "oracle", "base", "adapter"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260625)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    if args.limit:
        records = records[: args.limit]
    rng = random.Random(args.seed)
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
    score_rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"eval-{args.name}"):
        scores = None
        selection_policy = args.policy
        if args.policy in {"base", "adapter"}:
            assert model is not None and tokenizer is not None
            scores = model_scores(record, model, tokenizer, args.max_length)
            selection_policy = "model"
            for candidate in visible_candidates(record):
                score_rows.append(
                    {
                        "policy": args.name,
                        "record_id": record["record_id"],
                        "candidate_id": candidate["candidate_id"],
                        "score": scores[candidate["candidate_id"]],
                        "hidden_all_pass": candidate["hidden_all_pass"],
                    }
                )
        selected = select_candidate(record, selection_policy, rng, scores=scores)
        visible = visible_candidates(record)
        rows.append(
            {
                "policy": args.name,
                "dataset": record["dataset"],
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "candidate_pool_coverage": record_coverage(record),
                "visible_candidate_count": len(visible),
                "hidden_pass_visible_candidates": sum(1 for candidate in visible if candidate["hidden_all_pass"]),
                "selected_candidate_id": selected["candidate_id"] if selected else None,
                "selected_source": selected["source"] if selected else None,
                "selected_hidden_all": bool(selected and selected["hidden_all_pass"]),
            }
        )

    payload = {
        "name": args.name,
        "policy": args.policy,
        "records_path": str(args.records),
        "records": rows,
        "candidate_scores": score_rows,
        "summary": summarize(rows),
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(out), "records": len(rows), "score_rows": len(score_rows)}, indent=2))


if __name__ == "__main__":
    main()

