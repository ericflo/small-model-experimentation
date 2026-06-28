#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT, estimate_text_tokens  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402
from src.model_utils import DEFAULT_MODEL_PATH, code_chat_prompt, load_quant_model, load_tokenizer  # noqa: E402


def public_tests(record: dict[str, Any]) -> list[str]:
    return [case["assert_src"] for case in record.get("public_cases", [])]


def task_prompt(record: dict[str, Any]) -> str:
    return f"""Return only Python code. Do not use markdown.

Target task:
{record['task_text']}

Define a function named `{record['entry_point']}` and any helpers needed.

Public tests:
{chr(10).join(public_tests(record))}
"""


@torch.no_grad()
def completion_logprob(model: Any, tokenizer: Any, context: str, completion: str, max_length: int) -> dict[str, Any]:
    text = context + completion
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()
    input_ids = encoded["input_ids"][0]
    if input_ids.shape[0] > max_length:
        return {"ok": False, "reason": f"too_long:{int(input_ids.shape[0])}", "token_count": 0, "logprob_sum": -1e9, "logprob_mean": -1e9}
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    out = model(**encoded)
    logprobs = torch.log_softmax(out.logits[0, :-1, :], dim=-1)
    start_char = len(context)
    token_ids = input_ids.tolist()
    lps: list[float] = []
    for idx, (lo, hi) in enumerate(offsets):
        if idx == 0 or hi <= start_char:
            continue
        token_id = token_ids[idx]
        lps.append(float(logprobs[idx - 1, token_id].detach().cpu()))
    return {
        "ok": True,
        "reason": "ok",
        "token_count": len(lps),
        "logprob_sum": sum(lps),
        "logprob_mean": sum(lps) / len(lps) if lps else -1e9,
    }


def summarize(rows: list[dict[str, Any]], selector_name: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("selected")]
    return {
        "selector": selector_name,
        "records": len(rows),
        "commit_count": len(selected),
        "commit_rate": len(selected) / len(rows) if rows else 0.0,
        "selected_hidden_correct": sum(1 for row in selected if row.get("full_pass")),
        "selected_recovery_rate": sum(1 for row in selected if row.get("full_pass")) / len(rows) if rows else 0.0,
        "selected_visible_hidden_wrong": sum(1 for row in selected if row.get("visible_all_pass") and not row.get("full_pass")),
        "selected_false_pass_rate": (
            sum(1 for row in selected if row.get("visible_all_pass") and not row.get("full_pass")) / len(selected) if selected else 0.0
        ),
    }


def first_visible(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [candidate for candidate in candidates if candidate.get("visible_all_pass")]
    if not visible:
        return None
    return sorted(visible, key=lambda item: (item.get("pool_rank", 999), item.get("order", 999)))[0]


def best_map(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    visible = [candidate for candidate in candidates if candidate.get("visible_all_pass")]
    if not visible:
        return None
    return sorted(visible, key=lambda item: (-item.get("map_logprob_mean", -1e9), item.get("pool_rank", 999), item.get("order", 999)))[0]


def oracle(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    correct = [candidate for candidate in candidates if candidate.get("full_pass")]
    if not correct:
        return None
    return sorted(correct, key=lambda item: (item.get("pool_rank", 999), item.get("order", 999)))[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-records", nargs="+", type=Path, required=True)
    parser.add_argument("--pool-names", nargs="+", type=str, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    if len(args.candidate_records) != len(args.pool_names):
        raise ValueError("--candidate-records and --pool-names must align")

    tokenizer = load_tokenizer(args.model_path, padding_side="left")
    model = load_quant_model(args.model_path, for_training=False)
    model.eval()

    by_record: dict[str, dict[str, Any]] = {}
    usage = {"scored_candidates": 0, "prompt_tokens_estimate": 0, "completion_tokens_estimate": 0, "forward_tokens_estimate": 0}
    for pool_rank, (path, pool_name) in enumerate(zip(args.candidate_records, args.pool_names)):
        for record in load_jsonl(path):
            bundle = by_record.setdefault(record["record_id"], {"record": {key: value for key, value in record.items() if key != "candidates"}, "candidates": []})
            prompt = code_chat_prompt(tokenizer, task_prompt(record))
            for candidate in record.get("candidates", []):
                item = dict(candidate)
                item["pool_name"] = pool_name
                item["pool_rank"] = pool_rank
                item["pool_candidate_id"] = f"{pool_name}:{candidate.get('candidate_id')}:{candidate.get('order', 0)}"
                if item.get("parse_status") == "parsed":
                    score = completion_logprob(model, tokenizer, prompt, item.get("code", ""), args.max_length)
                else:
                    score = {"ok": False, "reason": "parse_failed", "token_count": 0, "logprob_sum": -1e9, "logprob_mean": -1e9}
                item["map_score_ok"] = score["ok"]
                item["map_logprob_mean"] = score["logprob_mean"]
                item["map_logprob_sum"] = score["logprob_sum"]
                item["map_token_count"] = score["token_count"]
                usage["scored_candidates"] += 1
                usage["prompt_tokens_estimate"] += estimate_text_tokens(tokenizer, prompt)
                usage["completion_tokens_estimate"] += estimate_text_tokens(tokenizer, item.get("code", ""))
                bundle["candidates"].append(item)
    usage["forward_tokens_estimate"] = usage["prompt_tokens_estimate"] + usage["completion_tokens_estimate"]

    out_rows: list[dict[str, Any]] = []
    selector_rows: dict[str, list[dict[str, Any]]] = {"first_visible": [], "map_mean": [], "oracle_hidden": []}
    for record_id, bundle in sorted(by_record.items(), key=lambda item: int(item[1]["record"]["task_id"])):
        candidates = bundle["candidates"]
        choices = {
            "first_visible": first_visible(candidates),
            "map_mean": best_map(candidates),
            "oracle_hidden": oracle(candidates),
        }
        row = {
            "record_id": record_id,
            "task_id": bundle["record"]["task_id"],
            "task_text": bundle["record"]["task_text"],
            "candidate_count": len(candidates),
            "visible_candidate_count": sum(1 for c in candidates if c.get("visible_all_pass")),
            "hidden_correct_candidate_count": sum(1 for c in candidates if c.get("full_pass")),
            "coverage": any(c.get("full_pass") for c in candidates),
            "selectors": {},
            "candidates": [
                {
                    "pool_candidate_id": c["pool_candidate_id"],
                    "pool_name": c["pool_name"],
                    "source": c.get("source"),
                    "visible_all_pass": c.get("visible_all_pass"),
                    "full_pass": c.get("full_pass"),
                    "map_logprob_mean": c.get("map_logprob_mean"),
                    "map_token_count": c.get("map_token_count"),
                }
                for c in candidates
            ],
        }
        for name, chosen in choices.items():
            selected = {
                "selected": chosen is not None,
                "pool_candidate_id": chosen.get("pool_candidate_id") if chosen else None,
                "pool_name": chosen.get("pool_name") if chosen else None,
                "visible_all_pass": bool(chosen.get("visible_all_pass")) if chosen else False,
                "full_pass": bool(chosen.get("full_pass")) if chosen else False,
                "map_logprob_mean": chosen.get("map_logprob_mean") if chosen else None,
            }
            row["selectors"][name] = selected
            selector_rows[name].append({"selected": selected["selected"], "visible_all_pass": selected["visible_all_pass"], "full_pass": selected["full_pass"]})
        out_rows.append(row)

    summary = {
        "experiment": EXPERIMENT,
        "records": len(out_rows),
        "pool_names": args.pool_names,
        "candidate_records": [str(path) for path in args.candidate_records],
        "pool_coverage": sum(1 for row in out_rows if row["coverage"]) / len(out_rows) if out_rows else 0.0,
        "selectors": {name: summarize(rows, name) for name, rows in selector_rows.items()},
        "usage_estimate": usage,
        "path": str(args.out),
    }
    write_jsonl(args.out, out_rows)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
