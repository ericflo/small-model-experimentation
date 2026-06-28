#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dsl import program_is_valid, program_pass_count  # noqa: E402
from src.edit_closure import ClosureConfig, closure_programs, select_by_visible  # noqa: E402
from src.modeling import load_jsonl  # noqa: E402


def metric(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    successes = sum(1 for row in rows if row[key])
    return {"successes": successes, "records": len(rows), "rate": successes / len(rows) if rows else 0.0}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)

    keys = [
        "base_rerank_hidden_all",
        "closure_hidden_all",
        "conservative_closure_hidden_all",
        "strict_closure_hidden_all",
        "closure_oracle_hidden_all",
        "base_rerank_visible_all",
        "closure_visible_all",
        "conservative_closure_visible_all",
        "strict_closure_visible_all",
        "closure_improved_hidden",
        "closure_damaged_hidden",
        "conservative_closure_improved_hidden",
        "conservative_closure_damaged_hidden",
        "conservative_closure_accepted",
        "strict_closure_improved_hidden",
        "strict_closure_damaged_hidden",
        "strict_closure_accepted",
    ]
    return {
        "overall": {key: metric(rows, key) for key in keys},
        "by_family": {
            family: {key: metric(subset, key) for key in keys}
            for family, subset in sorted(by_family.items())
        },
    }


def candidate_programs(row: dict[str, Any], mode: str, max_base_candidates: int) -> list[dict[str, str]]:
    if mode == "selected":
        return [{"kind": "base_selected", "program": row["selected_program"]}]
    if mode == "greedy":
        return [{"kind": "base_greedy", "program": row["greedy_program"]}]
    candidates = []
    for candidate in row.get("candidates", []):
        program = candidate.get("program")
        if program:
            candidates.append({"kind": candidate.get("kind", "candidate"), "program": program})
    if not candidates:
        candidates.append({"kind": "base_selected", "program": row["selected_program"]})
    dedup = {}
    for candidate in candidates:
        dedup.setdefault(candidate["program"], candidate)
    return list(dedup.values())[:max_base_candidates]


def top_by(programs: list[str], cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    scored = []
    for program in programs:
        valid = program_is_valid(program)
        scored.append(
            {
                "program": program,
                "valid": valid,
                "passes": program_pass_count(program, cases) if valid else 0,
                "total": len(cases),
                "length": len(program),
            }
        )
    scored.sort(key=lambda item: (item["passes"], int(item["valid"]), -item["length"]), reverse=True)
    return scored[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-mode", choices=["all", "selected", "greedy"], default="all")
    parser.add_argument("--max-base-candidates", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--max-variants-per-seed", type=int, default=900)
    parser.add_argument("--max-total-variants-per-seed", type=int, default=2400)
    parser.add_argument("--store-top-k", type=int, default=3)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    baseline = json.loads(args.baseline_result.read_text(encoding="utf-8"))
    records = load_jsonl(args.data)
    if args.max_records:
        records = records[: args.max_records]
    by_id = {record["id"]: record for record in records}
    config = ClosureConfig(
        max_variants_per_seed=args.max_variants_per_seed,
        max_total_variants=args.max_total_variants_per_seed,
        rounds=args.rounds,
    )

    rows = []
    for base_row in tqdm(baseline["rows"][: len(records)], desc="closure"):
        record = by_id[base_row["id"]]
        seed_candidates = candidate_programs(base_row, args.candidate_mode, args.max_base_candidates)
        generated: dict[str, str] = {}
        seed_errors = []
        for candidate in seed_candidates:
            program = candidate["program"]
            generated.setdefault(program, candidate["kind"])
            try:
                for closure_program in closure_programs(program, record["visible"], config):
                    generated.setdefault(closure_program, candidate["kind"])
            except Exception as exc:
                seed_errors.append({"kind": candidate["kind"], "program": program, "error": str(exc)})

        programs = list(generated.keys())
        selected = select_by_visible(programs, record["visible"])
        selected_hidden = program_pass_count(selected["program"], record["hidden"]) if selected["valid"] else 0
        top_hidden = top_by(programs, record["hidden"], limit=max(1, args.store_top_k))
        oracle = top_hidden[0]
        top_visible = top_by(programs, record["visible"], limit=max(1, args.store_top_k))

        base_hidden = base_row["selected_hidden_passes"]
        base_visible = base_row["selected_visible_passes"]
        hidden_total = len(record["hidden"])
        visible_total = len(record["visible"])
        accept_closure = selected["valid"] and selected["visible_passes"] > base_visible
        if accept_closure:
            conservative_program = selected["program"]
            conservative_visible = selected["visible_passes"]
            conservative_hidden = selected_hidden
            conservative_valid = selected["valid"]
            conservative_source = "closure"
        else:
            conservative_program = base_row["selected_program"]
            conservative_visible = base_visible
            conservative_hidden = base_hidden
            conservative_valid = program_is_valid(conservative_program)
            conservative_source = "base"
        strict_accept = selected["valid"] and selected["visible_passes"] == visible_total and base_visible < visible_total
        if strict_accept:
            strict_program = selected["program"]
            strict_visible = selected["visible_passes"]
            strict_hidden = selected_hidden
            strict_valid = selected["valid"]
            strict_source = "closure"
        else:
            strict_program = base_row["selected_program"]
            strict_visible = base_visible
            strict_hidden = base_hidden
            strict_valid = program_is_valid(strict_program)
            strict_source = "base"
        rows.append(
            {
                "id": record["id"],
                "family": record["family"],
                "target_program": record["target_program"],
                "wrong_program": record["wrong_program"],
                "base_selected_program": base_row["selected_program"],
                "base_selected_kind": base_row["selected_kind"],
                "base_candidate_count": base_row["candidate_count"],
                "base_selected_visible_passes": base_visible,
                "base_selected_hidden_passes": base_hidden,
                "base_rerank_visible_all": base_visible == visible_total,
                "base_rerank_hidden_all": base_hidden == hidden_total,
                "closure_program": selected["program"],
                "closure_valid": selected["valid"],
                "closure_visible_passes": selected["visible_passes"],
                "closure_hidden_passes": selected_hidden,
                "closure_visible_all": selected["visible_passes"] == visible_total,
                "closure_hidden_all": selected_hidden == hidden_total,
                "conservative_closure_program": conservative_program,
                "conservative_closure_valid": conservative_valid,
                "conservative_closure_source": conservative_source,
                "conservative_closure_visible_passes": conservative_visible,
                "conservative_closure_hidden_passes": conservative_hidden,
                "conservative_closure_visible_all": conservative_visible == visible_total,
                "conservative_closure_hidden_all": conservative_hidden == hidden_total,
                "strict_closure_program": strict_program,
                "strict_closure_valid": strict_valid,
                "strict_closure_source": strict_source,
                "strict_closure_visible_passes": strict_visible,
                "strict_closure_hidden_passes": strict_hidden,
                "strict_closure_visible_all": strict_visible == visible_total,
                "strict_closure_hidden_all": strict_hidden == hidden_total,
                "closure_oracle_program": oracle["program"],
                "closure_oracle_hidden_passes": oracle["passes"],
                "closure_oracle_hidden_all": oracle["passes"] == hidden_total,
                "closure_generated_count": len(programs),
                "closure_valid_count": sum(1 for program in programs if program_is_valid(program)),
                "closure_seed_count": len(seed_candidates),
                "closure_seed_errors": seed_errors,
                "closure_improved_hidden": selected_hidden > base_hidden,
                "closure_damaged_hidden": selected_hidden < base_hidden,
                "conservative_closure_improved_hidden": conservative_hidden > base_hidden,
                "conservative_closure_damaged_hidden": conservative_hidden < base_hidden,
                "conservative_closure_accepted": accept_closure,
                "strict_closure_improved_hidden": strict_hidden > base_hidden,
                "strict_closure_damaged_hidden": strict_hidden < base_hidden,
                "strict_closure_accepted": strict_accept,
                "visible_total": visible_total,
                "hidden_total": hidden_total,
                "top_visible": top_visible,
                "top_hidden": top_hidden,
            }
        )

    result = {
        "baseline_result": str(args.baseline_result),
        "data": str(args.data),
        "records": len(rows),
        "candidate_mode": args.candidate_mode,
        "max_base_candidates": args.max_base_candidates,
        "closure_config": {
            "rounds": config.rounds,
            "max_variants_per_seed": config.max_variants_per_seed,
            "max_total_variants_per_seed": config.max_total_variants,
        },
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
