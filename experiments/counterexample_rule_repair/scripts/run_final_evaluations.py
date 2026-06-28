#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[3]
LARGE = ROOT / "large_artifacts" / "counterexample_rule_repair"
EVAL_SCRIPT = EXP / "scripts" / "eval_counterexample_rule.py"
DATA = EXP / "data"
REPORTS = EXP / "reports"

SPLITS = {
    "iid": DATA / "repair_val_iid.jsonl",
    "format_holdout": DATA / "repair_val_format_holdout.jsonl",
    "rule_holdout": DATA / "repair_val_rule_holdout.jsonl",
}


def model_path(name: str) -> str:
    return str(LARGE / "models" / name)


def core_jobs() -> list[dict[str, str | None]]:
    jobs: list[dict[str, str | None]] = []
    for split, data_path in SPLITS.items():
        jobs.extend(
            [
                {
                    "name": f"frozen_trace_{split}",
                    "data": str(data_path),
                    "condition": "trace",
                    "adapter": None,
                },
                {
                    "name": f"final_patch_lora_final_patch_{split}",
                    "data": str(data_path),
                    "condition": "final_patch",
                    "adapter": model_path("final_patch_lora"),
                },
                {
                    "name": f"no_trace_lora_no_trace_{split}",
                    "data": str(data_path),
                    "condition": "no_trace",
                    "adapter": model_path("no_trace_lora"),
                },
                {
                    "name": f"shuffled_trace_lora_trace_{split}",
                    "data": str(data_path),
                    "condition": "trace",
                    "adapter": model_path("shuffled_trace_lora"),
                },
                {
                    "name": f"trace_lora_trace_{split}",
                    "data": str(data_path),
                    "condition": "trace",
                    "adapter": model_path("trace_lora"),
                },
            ]
        )
    return jobs


def ablation_jobs() -> list[dict[str, str | None]]:
    jobs: list[dict[str, str | None]] = []
    for split, data_path in SPLITS.items():
        jobs.extend(
            [
                {
                    "name": f"trace_lora_no_trace_{split}",
                    "data": str(data_path),
                    "condition": "no_trace",
                    "adapter": model_path("trace_lora"),
                },
                {
                    "name": f"trace_lora_shuffled_trace_{split}",
                    "data": str(data_path),
                    "condition": "shuffled_trace",
                    "adapter": model_path("trace_lora"),
                },
            ]
        )
    return jobs


def read_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("summary", {})


def run_job(job: dict[str, str | None], max_new_tokens: int, force: bool, max_records: int | None) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = REPORTS / f"final_{job['name']}.json"
    summary = read_summary(output)
    if summary is not None and not force:
        print(f"SKIP {job['name']}: {json.dumps(summary, sort_keys=True)}", flush=True)
        return

    cmd = [
        sys.executable,
        str(EVAL_SCRIPT),
        "--data",
        str(job["data"]),
        "--output",
        str(output),
        "--condition",
        str(job["condition"]),
        "--max-new-tokens",
        str(max_new_tokens),
    ]
    if job["adapter"]:
        cmd.extend(["--adapter", str(job["adapter"])])
    if max_records:
        cmd.extend(["--max-records", str(max_records)])

    print(f"RUN {job['name']}", flush=True)
    subprocess.run(cmd, check=True)
    completed_summary = read_summary(output) or {}
    print(f"DONE {job['name']}: {json.dumps(completed_summary, sort_keys=True)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["core", "ablation", "all"], default="core")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-records", type=int)
    args = parser.parse_args()

    jobs: list[dict[str, str | None]] = []
    if args.suite in ("core", "all"):
        jobs.extend(core_jobs())
    if args.suite in ("ablation", "all"):
        jobs.extend(ablation_jobs())
    for job in jobs:
        run_job(job, args.max_new_tokens, args.force, args.max_records)


if __name__ == "__main__":
    main()
