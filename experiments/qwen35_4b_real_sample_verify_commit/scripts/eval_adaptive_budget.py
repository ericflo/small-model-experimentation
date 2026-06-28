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

from src.code_env import stop_prompt  # noqa: E402
from src.eval_utils import coverage, model_state_for_prefix, prefix_candidates, visible_coverage  # noqa: E402
from src.jsonl import load_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, attach_existing_lora, last_token_action_logits, load_quant_model, load_tokenizer  # noqa: E402


def parse_budgets(text: str, max_count: int) -> list[int]:
    rows: list[int] = []
    for item in text.split(","):
        item = item.strip()
        if item:
            rows.append(max_count if item == "max" else int(item))
    return sorted(set(item for item in rows if item > 0))


def load_scores(path: Path) -> dict[str, dict[str, float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores: dict[str, dict[str, float]] = {}
    for row in payload.get("candidate_scores", []):
        scores.setdefault(row["record_id"], {})[row["candidate_id"]] = float(row["score"])
    return scores


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["policy"])].append(row)
    out: list[dict[str, Any]] = []
    for (dataset, policy), items in sorted(groups.items()):
        n = len(items)
        cov = sum(row["coverage"] for row in items) / n
        visible_cov = sum(row["visible_coverage"] for row in items) / n
        selected = sum(row["selected_hidden_all"] for row in items) / n
        out.append(
            {
                "dataset": dataset,
                "policy": policy,
                "records": n,
                "coverage": cov,
                "visible_coverage": visible_cov,
                "selected_hidden_all": selected,
                "coverage_captured": selected / visible_cov if visible_cov else 0.0,
                "samples_used_mean": sum(row["samples_used"] for row in items) / n,
                "visible_candidates_seen_mean": sum(row["visible_candidates_seen"] for row in items) / n,
            }
        )
    return out


@torch.no_grad()
def stop_decision(model: Any, tokenizer: Any, record: dict[str, Any], state: dict[str, Any], max_length: int) -> str:
    prompt = stop_prompt(record, state)
    logits = last_token_action_logits(model, tokenizer, [prompt], letters=["A", "B"], max_length=max_length).float()[0]
    return "A" if float(logits[0] - logits[1]) >= 0 else "B"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--mode", choices=["oracle_stop", "threshold", "sft_stop"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=str, default="1,2,4,8,16,max")
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--stop-adapter-dir", type=Path)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260625)
    args = parser.parse_args()

    records = load_jsonl(args.records)
    score_map = load_scores(args.scores)
    max_count = max(len(record["candidates"]) for record in records) if records else 0
    budgets = parse_budgets(args.budgets, max_count)
    rng = random.Random(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = None
    model = None
    if args.mode == "sft_stop":
        if args.stop_adapter_dir is None:
            raise ValueError("--stop-adapter-dir is required for sft_stop")
        tokenizer = load_tokenizer(args.model_path)
        model = attach_existing_lora(load_quant_model(args.model_path, for_training=False), args.stop_adapter_dir, is_trainable=False)
        model.eval()

    rows: list[dict[str, Any]] = []
    for record in tqdm(records, desc=f"adaptive-{args.name}"):
        scores = score_map.get(record["record_id"], {})
        chosen_state: dict[str, Any] | None = None
        for budget in budgets:
            state = model_state_for_prefix(record, budget, scores, rng)
            stop = False
            if args.mode == "oracle_stop":
                stop = bool(state["selected_hidden_all"] or budget == budgets[-1])
            elif args.mode == "threshold":
                stop = bool((state["visible_count"] > 0 and state["top_score"] >= args.threshold) or budget == budgets[-1])
            else:
                assert model is not None and tokenizer is not None
                action = stop_decision(model, tokenizer, record, state, args.max_length)
                stop = bool(action == "A" or budget == budgets[-1])
            if stop:
                chosen_state = state
                break
        assert chosen_state is not None
        candidates = prefix_candidates(record, int(chosen_state["budget"]))
        rows.append(
            {
                "dataset": record["dataset"],
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "policy": args.name,
                "samples_used": int(chosen_state["budget"]),
                "visible_candidates_seen": int(chosen_state["visible_count"]),
                "coverage": coverage(candidates),
                "visible_coverage": visible_coverage(candidates),
                "selected_candidate_id": chosen_state["selected_candidate_id"],
                "selected_source": chosen_state["selected_source"],
                "selected_hidden_all": bool(chosen_state["selected_hidden_all"]),
            }
        )

    payload = {
        "name": args.name,
        "mode": args.mode,
        "records_path": str(args.records),
        "scores_path": str(args.scores),
        "budgets": budgets,
        "threshold": args.threshold,
        "records": rows,
        "summary": summarize(rows),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

