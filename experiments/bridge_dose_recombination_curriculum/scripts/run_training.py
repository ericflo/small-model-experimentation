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
LARGE_DIR = WORKSPACE_DIR / "large_artifacts" / "bridge_dose_recombination_curriculum" / "models"

MODEL_ID = "Qwen/Qwen2.5-Coder-3B-Instruct"
REVISION = "488639f1ff808d1d3d0ba301aef8c11461451ec5"


@dataclass(frozen=True)
class TrainJob:
    suite: str
    name: str
    train: Path
    eval: Path
    mode: str
    shuffle_traces: bool
    output_dir: Path


def planned_jobs() -> list[TrainJob]:
    data = EXPERIMENT_DIR / "data"
    return [
        TrainJob(
            suite="dose",
            name="dose0_trace",
            train=data / "repair_train_dose0.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose0_trace_lora",
        ),
        TrainJob(
            suite="dose",
            name="dose1_trace",
            train=data / "repair_train_dose1.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose1_trace_lora",
        ),
        TrainJob(
            suite="dose",
            name="dose2_trace",
            train=data / "repair_train_dose2.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose2_trace_lora",
        ),
        TrainJob(
            suite="dose",
            name="dose4_trace",
            train=data / "repair_train_dose4.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose4_trace_lora",
        ),
        TrainJob(
            suite="dose",
            name="dose8_trace",
            train=data / "repair_train_dose8.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose8_trace_lora",
        ),
        TrainJob(
            suite="control",
            name="near_miss_focus_trace",
            train=data / "repair_train_near_miss_focus.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "near_miss_focus_trace_lora",
        ),
        TrainJob(
            suite="control",
            name="dose0_no_trace",
            train=data / "repair_train_dose0.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="no_trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose0_no_trace_lora",
        ),
        TrainJob(
            suite="control",
            name="dose0_shuffled_trace",
            train=data / "repair_train_dose0.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=True,
            output_dir=LARGE_DIR / "dose0_shuffled_trace_lora",
        ),
        TrainJob(
            suite="control",
            name="dose8_no_trace",
            train=data / "repair_train_dose8.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="no_trace",
            shuffle_traces=False,
            output_dir=LARGE_DIR / "dose8_no_trace_lora",
        ),
        TrainJob(
            suite="control",
            name="dose8_shuffled_trace",
            train=data / "repair_train_dose8.jsonl",
            eval=data / "repair_val_recombination_holdout.jsonl",
            mode="trace",
            shuffle_traces=True,
            output_dir=LARGE_DIR / "dose8_shuffled_trace_lora",
        ),
    ]


def select_jobs(jobs: list[TrainJob], suite: str) -> list[TrainJob]:
    if suite == "all":
        return jobs
    return [job for job in jobs if job.suite == suite]


def run_job(job: TrainJob, args: argparse.Namespace) -> dict[str, object]:
    command = [
        sys.executable,
        str(WORKSPACE_DIR / "scripts" / "train_repair_lora.py"),
        "--train",
        str(job.train),
        "--eval",
        str(job.eval),
        "--mode",
        job.mode,
        "--model-id",
        args.model_id,
        "--revision",
        args.revision,
        "--output-dir",
        str(job.output_dir),
        "--max-length",
        str(args.max_length),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--rank",
        str(args.rank),
        "--alpha",
        str(args.alpha),
        "--dropout",
        str(args.dropout),
        "--grad-accum",
        str(args.grad_accum),
        "--save-steps",
        str(args.save_steps),
        "--eval-steps",
        str(args.eval_steps),
    ]
    if job.shuffle_traces:
        command.append("--shuffle-traces")
    if args.max_train_records:
        command.extend(["--max-train-records", str(args.max_train_records)])

    status = {
        **asdict(job),
        "train": str(job.train),
        "eval": str(job.eval),
        "output_dir": str(job.output_dir),
        "command": command,
        "status": "pending",
        "started_at": None,
        "finished_at": None,
        "runtime_seconds": None,
    }
    metadata_path = job.output_dir / "experiment_metadata.json"
    if metadata_path.exists() and not args.force:
        status["status"] = "skipped_existing"
        return status
    if args.dry_run:
        status["status"] = "dry_run"
        return status

    job.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    status["started_at"] = started
    print(f"[train] {job.name} ({job.mode}, shuffle_traces={job.shuffle_traces})", flush=True)
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
        "suite": args.suite,
        "max_length": args.max_length,
        "epochs": args.epochs,
        "lr": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "dropout": args.dropout,
        "grad_accum": args.grad_accum,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "max_train_records": args.max_train_records,
        "jobs": statuses,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_jobs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["dose", "control", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=EXPERIMENT_DIR / "reports" / "training")
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--max-length", type=int, default=3072)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1.5e-4)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=30)
    parser.add_argument("--eval-steps", type=int, default=30)
    parser.add_argument("--max-train-records", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = select_jobs(planned_jobs(), args.suite)
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
