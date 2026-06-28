#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = Path(__file__).resolve().parents[3]

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"


NORMAL_SPLITS = {
    "seen_iid": EXPERIMENT_DIR / "data" / "repair_val_seen_iid.jsonl",
    "format_shift": EXPERIMENT_DIR / "data" / "repair_val_format_shift.jsonl",
    "recombination_holdout": EXPERIMENT_DIR / "data" / "repair_val_recombination_holdout.jsonl",
}

ADAPTERS = {
    "dose0_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose0_trace_lora",
    "dose1_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose1_trace_lora",
    "dose2_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose2_trace_lora",
    "dose4_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose4_trace_lora",
    "dose8_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose8_trace_lora",
    "near_miss_focus_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "near_miss_focus_trace_lora",
    "dose0_no_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose0_no_trace_lora",
    "dose0_shuffled_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose0_shuffled_trace_lora",
    "dose8_no_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose8_no_trace_lora",
    "dose8_shuffled_trace": WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models" / "dose8_shuffled_trace_lora",
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
        ("dose0_trace", "trace", ADAPTERS["dose0_trace"], NORMAL_SPLITS),
        ("dose1_trace", "trace", ADAPTERS["dose1_trace"], NORMAL_SPLITS),
        ("dose2_trace", "trace", ADAPTERS["dose2_trace"], NORMAL_SPLITS),
        ("dose4_trace", "trace", ADAPTERS["dose4_trace"], NORMAL_SPLITS),
        ("dose8_trace", "trace", ADAPTERS["dose8_trace"], NORMAL_SPLITS),
        ("near_miss_focus_trace", "trace", ADAPTERS["near_miss_focus_trace"], NORMAL_SPLITS),
        ("dose0_no_trace", "no_trace", ADAPTERS["dose0_no_trace"], NORMAL_SPLITS),
        ("dose0_shuffled_trace", "trace", ADAPTERS["dose0_shuffled_trace"], NORMAL_SPLITS),
        ("dose8_no_trace", "no_trace", ADAPTERS["dose8_no_trace"], NORMAL_SPLITS),
        ("dose8_shuffled_trace", "trace", ADAPTERS["dose8_shuffled_trace"], NORMAL_SPLITS),
    ]
    ablation_conditions = [
        ("dose8_trace_no_trace_prompt", "no_trace", ADAPTERS["dose8_trace"], NORMAL_SPLITS),
        ("dose8_trace_shuffled_trace_prompt", "shuffled_trace", ADAPTERS["dose8_trace"], NORMAL_SPLITS),
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


def run_job(job: EvalJob, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / "scripts" / "eval_bridge.py"),
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

    status = {
        **asdict(job),
        "adapter": str(job.adapter) if job.adapter else None,
        "data": str(job.data),
        "output": str(job.output),
        "command": command,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "runtime_seconds": None,
    }
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


def write_manifest(output_dir: Path, statuses: list[dict[str, object]], args: argparse.Namespace) -> None:
    payload = {
        "experiment": "bridge_dose_recombination_curriculum",
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = select_jobs(planned_jobs(args.output_dir), args.suite)
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
