#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.active_core import (  # noqa: E402
    ProgramBank,
    best_oracle_option,
    case_key,
    dedupe_pool,
    query_option_rows,
    render_policy_prompt,
)
from src.modeling import load_jsonl  # noqa: E402
from src.sketch import complete_sketch, sketch_variants  # noqa: E402


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def synthesize_target_programs(record: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[dict[str, Any]]]:
    programs: list[str] = []
    seen: set[str] = set()
    sketch_rows = []
    variants = sketch_variants(record["target_sketch"], max_variants=args.max_sketch_variants)
    for variant_index, sketch in enumerate(variants):
        completed = complete_sketch(
            sketch,
            record["visible"],
            max_programs_per_sketch=args.max_programs_per_sketch,
            max_hole_options=args.max_hole_options,
        )
        added = 0
        for program in completed:
            if program in seen:
                continue
            if len(programs) >= args.max_total_programs_per_record:
                break
            seen.add(program)
            programs.append(program)
            added += 1
        sketch_rows.append(
            {
                "variant_index": variant_index,
                "is_original_variant": variant_index == 0,
                "sketch": sketch,
                "completed_programs": len(completed),
                "added_programs": added,
            }
        )
        if len(programs) >= args.max_total_programs_per_record:
            break
    return programs, sketch_rows


def serializable_options(options: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keep = [
        "option_id",
        "input",
        "expected",
        "viable_candidates",
        "output_buckets",
        "largest_bucket",
        "max_possible_eliminated",
        "actual_eliminated",
        "target_bucket_size",
        "entropy",
        "buckets",
        "case_key",
    ]
    return [{key: option[key] for key in keep} for option in options]


def build_rows(records: list[dict[str, Any]], split: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    examples = []
    diagnostics = []
    for record_index, record in enumerate(tqdm(records, desc=f"policy-data:{split}")):
        programs, sketch_rows = synthesize_target_programs(record, args)
        bank = ProgramBank(programs, record)
        query_pool = dedupe_pool(record)
        if args.max_query_pool_cases and len(query_pool) > args.max_query_pool_cases:
            query_pool = random.Random(args.seed + record_index * 31).sample(query_pool, args.max_query_pool_cases)
        observed = list(record["visible"])
        used = {case_key(case) for case in observed}
        record_examples = 0
        for step in range(1, args.max_steps + 1):
            options = query_option_rows(
                bank=bank,
                observed=observed,
                pool=query_pool,
                used=used,
                max_policy_candidates=args.max_policy_candidates,
                max_options=args.max_options,
                max_buckets=args.max_buckets,
                rng=random.Random(args.seed + record_index * 1009 + step),
            )
            target = best_oracle_option(options)
            if target is None or target["actual_eliminated"] <= 0:
                break
            prompt = render_policy_prompt(record=record, observed=observed, options=options, step=step)
            examples.append(
                {
                    "id": f"{record['id']}::step{step}",
                    "source_id": record["id"],
                    "split": split,
                    "family": record["family"],
                    "step": step,
                    "prompt": prompt,
                    "target_action": target["option_id"],
                    "target_input": target["input"],
                    "target_expected": target["expected"],
                    "target_actual_eliminated": target["actual_eliminated"],
                    "target_entropy": target["entropy"],
                    "target_bucket_size": target["target_bucket_size"],
                    "options": serializable_options(options),
                }
            )
            used.add(case_key(target["case"]))
            observed.append(target["case"])
            record_examples += 1
        diagnostics.append(
            {
                "id": record["id"],
                "family": record["family"],
                "split": split,
                "examples": record_examples,
                "synthesized_program_count": len(programs),
                "target_program_synthesized": record["target_program"] in programs,
                "visible_consistent_candidates": sum(
                    1 for passes in bank.visible_passes if passes == len(record["visible"])
                ),
                "candidate_oracle_hidden_all": max(bank.hidden_passes, default=0) == len(record["hidden"]),
                "query_pool_count": len(query_pool),
                "sketches": sketch_rows,
            }
        )
    return examples, diagnostics


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, int] = {}
    by_step: dict[str, int] = {}
    for row in rows:
        by_family[row["family"]] = by_family.get(row["family"], 0) + 1
        step = str(row["step"])
        by_step[step] = by_step.get(step, 0) + 1
    return {
        "examples": len(rows),
        "by_family": dict(sorted(by_family.items())),
        "by_step": dict(sorted(by_step.items(), key=lambda item: int(item[0]))),
        "avg_target_actual_eliminated": round(
            sum(row["target_actual_eliminated"] for row in rows) / len(rows),
            3,
        )
        if rows
        else 0.0,
        "avg_target_entropy": round(sum(row["target_entropy"] for row in rows) / len(rows), 3) if rows else 0.0,
    }


def load_many(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        records.extend(load_jsonl(path))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=Path, nargs="+", required=True)
    parser.add_argument("--eval-data", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "policy")
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--max-eval-records", type=int, default=80)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--max-options", type=int, default=24)
    parser.add_argument("--max-buckets", type=int, default=6)
    parser.add_argument("--max-policy-candidates", type=int, default=128)
    parser.add_argument("--max-query-pool-cases", type=int, default=48)
    parser.add_argument("--max-programs-per-sketch", type=int, default=5000)
    parser.add_argument("--max-total-programs-per-record", type=int, default=4000)
    parser.add_argument("--max-hole-options", type=int, default=28)
    parser.add_argument("--max-sketch-variants", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260702)
    args = parser.parse_args()

    train_records = load_many(args.train_data)
    eval_records = load_many(args.eval_data)
    if args.max_train_records:
        train_records = train_records[: args.max_train_records]
    if args.max_eval_records:
        eval_records = eval_records[: args.max_eval_records]

    train_rows, train_diag = build_rows(train_records, "train", args)
    eval_rows, eval_diag = build_rows(eval_records, "eval", args)

    write_jsonl(args.out_dir / "policy_train.jsonl", train_rows)
    write_jsonl(args.out_dir / "policy_eval.jsonl", eval_rows)
    write_jsonl(args.out_dir / "policy_train_diagnostics.jsonl", train_diag)
    write_jsonl(args.out_dir / "policy_eval_diagnostics.jsonl", eval_diag)

    manifest = {
        "seed": args.seed,
        "train_data": [str(path) for path in args.train_data],
        "eval_data": [str(path) for path in args.eval_data],
        "train_records": len(train_records),
        "eval_records": len(eval_records),
        "train_summary": summarize(train_rows),
        "eval_summary": summarize(eval_rows),
        "max_steps": args.max_steps,
        "max_options": args.max_options,
        "max_buckets": args.max_buckets,
        "max_policy_candidates": args.max_policy_candidates,
        "max_query_pool_cases": args.max_query_pool_cases,
        "max_total_programs_per_record": args.max_total_programs_per_record,
        "label": "oracle-elimination option among displayed candidate query choices",
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "policy_dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
