#!/usr/bin/env python3
"""Score correct versus reversed teachers on identical student prefixes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from analyze_qualification import stratified_bootstrap_lcb  # noqa: E402
from gym.families import load as load_family  # noqa: E402
from io_utils import load_config, read_jsonl, write_json  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--quick-output", type=Path, required=True)
    parser.add_argument("--deep-output", type=Path, required=True)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "teacher_audit.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    items = {row["audit_id"]: row for row in read_jsonl(args.items)}
    quick = {row["id"]: row for row in read_jsonl(args.quick_output)}
    deep = {row["id"]: row for row in read_jsonl(args.deep_output)}
    if set(items) != set(quick) or set(items) != set(deep):
        raise SystemExit("teacher-audit row mismatch")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"], revision=config["model"]["revision"],
        trust_remote_code=True, use_fast=True,
    )
    paired = {"quick": [], "deep": []}
    scores = []
    expected_continuations = int(config["teacher_audit"]["continuations_per_branch"])
    for key in sorted(items):
        item = items[key]
        family = load_family(item["family"])
        prefix = [int(value) for value in item["prefix_ids"]]
        if (
            len(quick[key]["outputs"]) != expected_continuations
            or len(deep[key]["outputs"]) != expected_continuations
        ):
            raise SystemExit(f"teacher-audit continuation count mismatch for {key}")
        scoring_item = {
            "id": item["source_id"], "family": item["family"],
            "level": item["level"], "prompt": item["prompt"], "gold": item["gold"],
            "answer_domain": item.get("answer_domain"),
        }
        quick_scores = [
            float(
                family.score_atom(
                    scoring_item,
                    tokenizer.decode(
                        prefix + [int(value) for value in output["token_ids"]],
                        skip_special_tokens=False,
                    ),
                )
            )
            for output in quick[key]["outputs"]
        ]
        deep_scores = [
            float(
                family.score_atom(
                    scoring_item,
                    tokenizer.decode(
                        prefix + [int(value) for value in output["token_ids"]],
                        skip_special_tokens=False,
                    ),
                )
            )
            for output in deep[key]["outputs"]
        ]
        quick_score = sum(quick_scores) / len(quick_scores)
        deep_score = sum(deep_scores) / len(deep_scores)
        correct = quick_score if item["stratum"] == "quick" else deep_score
        wrong = deep_score if item["stratum"] == "quick" else quick_score
        row = {
            "family": item["family"], "kind": "atom", "level": int(item["level"]),
        }
        paired[item["stratum"]].append((row, correct - wrong))
        scores.append({
            "id": key, "family": item["family"], "level": item["level"],
            "stratum": item["stratum"], "quick_score": quick_score,
            "deep_score": deep_score, "quick_continuation_scores": quick_scores,
            "deep_continuation_scores": deep_scores,
            "correct_minus_wrong": correct - wrong,
        })
    result_by_stratum = {}
    samples = int(config["evaluation"]["paired_bootstrap_samples"])
    confidence = float(config["evaluation"]["confidence"])
    for index, stratum in enumerate(("quick", "deep")):
        rows = paired[stratum]
        cells = {}
        for row, value in rows:
            cell = (row["family"], row["kind"], row["level"])
            cells.setdefault(cell, []).append(value)
        mean = sum(sum(values) / len(values) for values in cells.values()) / len(cells)
        lcb = stratified_bootstrap_lcb(
            rows, samples=samples, confidence=confidence, seed=7000 + index
        )
        result_by_stratum[stratum] = {
            "n": len(rows), "paired_macro_delta": mean,
            "one_sided_lcb": lcb, "passed": mean > 0.0 and lcb > 0.0,
        }
    passed = all(row["passed"] for row in result_by_stratum.values())
    result = {
        "stage": "same_prefix_teacher_audit",
        "config": str(config_path),
        "prefix_identity": "both teachers continued the exact same token-id prefix",
        "branch_sampling": {
            "top_k": int(config["teacher_audit"]["branch_top_k"]),
            "continuations_per_prefix": expected_continuations,
            "aggregation": "mean verified outcome across sampled top-k branches",
        },
        "by_stratum": result_by_stratum,
        "scores": scores,
        "gate": {"passed": passed},
        "downstream_authorization": "locality_pilot" if passed else "stop_before_locality_pilot",
    }
    write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key != "scores"}, indent=2))
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
