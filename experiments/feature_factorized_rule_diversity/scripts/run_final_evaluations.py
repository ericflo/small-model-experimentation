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


SPLITS = {
    "singleton_iid": EXPERIMENT_DIR / "data" / "repair_val_singleton_iid.jsonl",
    "composite_iid": EXPERIMENT_DIR / "data" / "repair_val_composite_iid.jsonl",
    "recombination_holdout": EXPERIMENT_DIR / "data" / "repair_val_recombination_holdout.jsonl",
}


ADAPTERS = {
    "singletons_trace": WORKSPACE_DIR
    / "large_artifacts"
    / "feature_factorized_rule_diversity"
    / "models"
    / "singletons_trace_lora",
    "composites_trace": WORKSPACE_DIR
    / "large_artifacts"
    / "feature_factorized_rule_diversity"
    / "models"
    / "composites_trace_lora",
    "mixed_trace": WORKSPACE_DIR
    / "large_artifacts"
    / "feature_factorized_rule_diversity"
    / "models"
    / "mixed_trace_lora",
    "mixed_no_trace": WORKSPACE_DIR
    / "large_artifacts"
    / "feature_factorized_rule_diversity"
    / "models"
    / "mixed_no_trace_lora",
    "mixed_shuffled_trace": WORKSPACE_DIR
    / "large_artifacts"
    / "feature_factorized_rule_diversity"
    / "models"
    / "mixed_shuffled_trace_lora",
}


@dataclass(frozen=True)
class EvalJob:
    suite: str
    name: str
    split: str
    condition: str
    adapter: Path | None
    output: Path

    @property
    def data(self) -> Path:
        return SPLITS[self.split]


def planned_jobs(output_dir: Path) -> list[EvalJob]:
    core_conditions = [
        ("frozen_trace", "trace", None),
        ("singletons_trace", "trace", ADAPTERS["singletons_trace"]),
        ("composites_trace", "trace", ADAPTERS["composites_trace"]),
        ("mixed_trace", "trace", ADAPTERS["mixed_trace"]),
        ("mixed_no_trace", "no_trace", ADAPTERS["mixed_no_trace"]),
        ("mixed_shuffled_trace", "trace", ADAPTERS["mixed_shuffled_trace"]),
    ]
    ablation_conditions = [
        ("mixed_trace_no_trace_prompt", "no_trace", ADAPTERS["mixed_trace"]),
        ("mixed_trace_shuffled_trace_prompt", "shuffled_trace", ADAPTERS["mixed_trace"]),
    ]

    jobs: list[EvalJob] = []
    for name, condition, adapter in core_conditions:
        for split in SPLITS:
            jobs.append(
                EvalJob(
                    suite="core",
                    name=name,
                    split=split,
                    condition=condition,
                    adapter=adapter,
                    output=output_dir / f"core__{name}__{split}.json",
                )
            )
    for name, condition, adapter in ablation_conditions:
        for split in SPLITS:
            jobs.append(
                EvalJob(
                    suite="ablation",
                    name=name,
                    split=split,
                    condition=condition,
                    adapter=adapter,
                    output=output_dir / f"ablation__{name}__{split}.json",
                )
            )
    return jobs


def select_jobs(jobs: list[EvalJob], suite: str) -> list[EvalJob]:
    if suite == "all":
        return jobs
    return [job for job in jobs if job.suite == suite]


def adapter_arg(adapter: Path | None) -> list[str]:
    if adapter is None:
        return []
    return ["--adapter", str(adapter)]


def run_job(job: EvalJob, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable,
        str(EXPERIMENT_DIR / "scripts" / "eval_factorized.py"),
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
        "experiment": "feature_factorized_rule_diversity",
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
