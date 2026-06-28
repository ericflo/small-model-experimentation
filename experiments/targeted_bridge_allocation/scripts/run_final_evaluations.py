#!/usr/bin/env python
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(WORKSPACE_DIR / "src"))
sys.path.insert(0, str(EXPERIMENT_DIR / "scripts"))

from eval_allocation import evaluate_record, shuffled_traces, summarize  # noqa: E402
from repair_experiment.modeling import load_jsonl, load_model_for_generation, load_tokenizer  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"


NORMAL_SPLITS = {
    "seen_iid": EXPERIMENT_DIR / "data" / "repair_val_seen_iid.jsonl",
    "format_shift": EXPERIMENT_DIR / "data" / "repair_val_format_shift.jsonl",
    "recombination_holdout": EXPERIMENT_DIR / "data" / "repair_val_recombination_holdout.jsonl",
}

ADAPTERS = {
    "uniform2_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "uniform2_trace_lora",
    "uniform4_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "uniform4_trace_lora",
    "hard_target_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "hard_target_trace_lora",
    "hard_target_seen_preserving_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "hard_target_seen_preserving_trace_lora",
    "easy_target_control_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "easy_target_control_trace_lora",
    "modulo16_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "modulo16_trace_lora",
    "length16_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "length16_trace_lora",
    "tuple16_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "tuple16_trace_lora",
    "hard_target_no_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "hard_target_no_trace_lora",
    "hard_target_shuffled_trace": WORKSPACE_DIR / "large_artifacts" / "targeted_bridge_allocation" / "models" / "hard_target_shuffled_trace_lora",
}


@dataclass(frozen=True)
class EvalJob:
    suite: str
    name: str
    split: str
    condition: str
    adapter: Path | None
    data: Path
    output: Path


def planned_jobs(output_dir: Path) -> list[EvalJob]:
    core_conditions = [
        ("frozen_trace", "trace", None, NORMAL_SPLITS),
        ("uniform2_trace", "trace", ADAPTERS["uniform2_trace"], NORMAL_SPLITS),
        ("uniform4_trace", "trace", ADAPTERS["uniform4_trace"], NORMAL_SPLITS),
        ("hard_target_trace", "trace", ADAPTERS["hard_target_trace"], NORMAL_SPLITS),
        ("hard_target_seen_preserving_trace", "trace", ADAPTERS["hard_target_seen_preserving_trace"], NORMAL_SPLITS),
        ("easy_target_control_trace", "trace", ADAPTERS["easy_target_control_trace"], NORMAL_SPLITS),
        ("modulo16_trace", "trace", ADAPTERS["modulo16_trace"], NORMAL_SPLITS),
        ("length16_trace", "trace", ADAPTERS["length16_trace"], NORMAL_SPLITS),
        ("tuple16_trace", "trace", ADAPTERS["tuple16_trace"], NORMAL_SPLITS),
        ("hard_target_no_trace", "no_trace", ADAPTERS["hard_target_no_trace"], NORMAL_SPLITS),
        ("hard_target_shuffled_trace", "trace", ADAPTERS["hard_target_shuffled_trace"], NORMAL_SPLITS),
    ]
    ablation_conditions = [
        ("hard_target_trace_no_trace_prompt", "no_trace", ADAPTERS["hard_target_trace"], NORMAL_SPLITS),
        ("hard_target_trace_shuffled_trace_prompt", "shuffled_trace", ADAPTERS["hard_target_trace"], NORMAL_SPLITS),
    ]

    jobs: list[EvalJob] = []
    for suite, conditions in [("core", core_conditions), ("ablation", ablation_conditions)]:
        for name, condition, adapter, split_map in conditions:
            for split, data in split_map.items():
                jobs.append(
                    EvalJob(
                        suite=suite,
                        name=name,
                        split=split,
                        condition=condition,
                        adapter=adapter,
                        data=data,
                        output=output_dir / f"{suite}__{name}__{split}.json",
                    )
                )
    return jobs


def select_jobs(jobs: list[EvalJob], suite: str) -> list[EvalJob]:
    if suite == "all":
        return jobs
    return [job for job in jobs if job.suite == suite]


def adapter_arg(adapter: Path | None) -> list[str]:
    return ["--adapter", str(adapter)] if adapter else []


def command_for_job(job: EvalJob, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / "scripts" / "eval_allocation.py"),
        "--data",
        str(job.data),
        "--output",
        str(job.output),
        "--condition",
        job.condition,
        "--model-id",
        args.model_id,
        "--revision",
        args.revision,
        "--max-new-tokens",
        str(args.max_new_tokens),
        *adapter_arg(job.adapter),
    ]
    if args.max_records:
        command.extend(["--max-records", str(args.max_records)])
    return command


def status_for_job(job: EvalJob, args: argparse.Namespace) -> dict[str, object]:
    status = {
        **asdict(job),
        "adapter": str(job.adapter) if job.adapter else None,
        "data": str(job.data),
        "output": str(job.output),
        "command": command_for_job(job, args),
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "runtime_seconds": None,
    }
    return status


def run_job(job: EvalJob, args: argparse.Namespace) -> dict[str, object]:
    command = command_for_job(job, args)
    status = status_for_job(job, args)
    if job.output.exists() and not args.force:
        status["status"] = "skipped_existing"
        return status
    if args.dry_run:
        status["status"] = "dry_run"
        return status

    job.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status["started_at"] = started
    print(f"[run] {job.suite} {job.name} {job.split} ({job.condition})", flush=True)
    subprocess.run(command, cwd=WORKSPACE_DIR, check=True)
    finished = time.time()
    status["finished_at"] = finished
    status["runtime_seconds"] = round(finished - started, 3)
    status["status"] = "completed"
    return status


PROMPT_MODE = {
    "trace": "trace",
    "no_trace": "no_trace",
    "shuffled_trace": "trace",
    "final_patch": "final_patch",
}


def evaluate_in_process(job: EvalJob, args: argparse.Namespace, tokenizer, model, records: list[dict[str, object]]) -> None:
    trace_overrides = shuffled_traces(records, args.shuffle_seed) if job.condition == "shuffled_trace" else {}
    prompt_mode = PROMPT_MODE[job.condition]
    results = []
    for record in tqdm(records, desc=f"eval {job.name} {job.split}"):
        results.append(
            evaluate_record(
                model=model,
                tokenizer=tokenizer,
                record=record,
                prompt_mode=prompt_mode,
                trace_override=trace_overrides.get(record["episode_id"]),
                max_new_tokens=args.max_new_tokens,
            )
        )
    metadata = {
        "data": str(job.data),
        "condition": job.condition,
        "prompt_mode": prompt_mode,
        "model_id": args.model_id,
        "revision": args.revision,
        "adapter": str(job.adapter) if job.adapter else None,
        "max_new_tokens": args.max_new_tokens,
        "shuffle_seed": args.shuffle_seed if job.condition == "shuffled_trace" else None,
    }
    payload = {"summary": summarize(results, metadata), "records": results}
    job.output.parent.mkdir(parents=True, exist_ok=True)
    job.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)


def grouped_jobs(jobs: list[EvalJob]) -> list[tuple[Path | None, list[EvalJob]]]:
    groups: list[tuple[Path | None, list[EvalJob]]] = []
    lookup: dict[str, int] = {}
    for job in jobs:
        key = str(job.adapter) if job.adapter else "<base>"
        if key not in lookup:
            lookup[key] = len(groups)
            groups.append((job.adapter, []))
        groups[lookup[key]][1].append(job)
    return groups


def run_jobs_grouped(jobs: list[EvalJob], args: argparse.Namespace, output_dir: Path) -> list[dict[str, object]]:
    statuses: list[dict[str, object]] = []
    data_cache: dict[Path, list[dict[str, object]]] = {}
    total_jobs = len(jobs)
    completed_or_skipped = 0
    for adapter, group in grouped_jobs(jobs):
        runnable = [job for job in group if args.force or not job.output.exists()]
        for job in group:
            if job not in runnable:
                status = status_for_job(job, args)
                status["status"] = "skipped_existing"
                statuses.append(status)
                completed_or_skipped += 1
                print(f"[job {completed_or_skipped}/{total_jobs}] skipped_existing {job.suite} {job.name} {job.split}", flush=True)
                write_manifest(output_dir, statuses, args)
        if args.dry_run:
            for job in runnable:
                status = status_for_job(job, args)
                status["status"] = "dry_run"
                statuses.append(status)
                completed_or_skipped += 1
                print(f"[job {completed_or_skipped}/{total_jobs}] dry_run {job.suite} {job.name} {job.split}", flush=True)
                write_manifest(output_dir, statuses, args)
            continue
        if not runnable:
            continue

        adapter_label = str(adapter) if adapter else "base model"
        print(f"[load] {adapter_label} for {len(runnable)} evaluation jobs", flush=True)
        tokenizer = load_tokenizer(args.model_id, args.revision)
        model = load_model_for_generation(
            args.model_id,
            args.revision,
            str(adapter) if adapter else None,
            load_in_4bit=True,
        )
        try:
            for job in runnable:
                status = status_for_job(job, args)
                started = time.time()
                status["started_at"] = started
                completed_or_skipped += 1
                print(f"[job {completed_or_skipped}/{total_jobs}] {job.suite} {job.name} {job.split} ({job.condition})", flush=True)
                records = data_cache.get(job.data)
                if records is None:
                    records = load_jsonl(job.data)
                    if args.max_records:
                        records = records[: args.max_records]
                    data_cache[job.data] = records
                evaluate_in_process(job, args, tokenizer, model, records)
                finished = time.time()
                status["finished_at"] = finished
                status["runtime_seconds"] = round(finished - started, 3)
                status["status"] = "completed"
                statuses.append(status)
                write_manifest(output_dir, statuses, args)
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return statuses


def write_manifest(output_dir: Path, statuses: list[dict[str, object]], args: argparse.Namespace) -> None:
    payload = {
        "experiment": "targeted_bridge_allocation",
        "model_id": args.model_id,
        "revision": args.revision,
        "max_new_tokens": args.max_new_tokens,
        "max_records": args.max_records,
        "suite": args.suite,
        "jobs": statuses,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "final_evaluation_jobs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["core", "ablation", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "final")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--shuffle-seed", type=int, default=9173)
    parser.add_argument("--execution", choices=["grouped", "subprocess"], default="grouped")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = select_jobs(planned_jobs(args.output_dir), args.suite)
    if args.execution == "grouped":
        statuses = []
        try:
            statuses = run_jobs_grouped(jobs, args, args.output_dir)
        finally:
            write_manifest(args.output_dir, statuses, args)
    else:
        statuses = []
        try:
            for index, job in enumerate(jobs, start=1):
                print(f"[job {index}/{len(jobs)}]", flush=True)
                statuses.append(run_job(job, args))
                write_manifest(args.output_dir, statuses, args)
        finally:
            write_manifest(args.output_dir, statuses, args)


if __name__ == "__main__":
    main()
