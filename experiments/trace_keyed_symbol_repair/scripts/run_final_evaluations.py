#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXP = Path("experiments/trace_keyed_symbol_repair")
LARGE = Path("large_artifacts/trace_keyed_symbol_repair")
EVAL_SCRIPT = EXP / "scripts" / "eval_trace_keyed.py"
IID = EXP / "data" / "repair_val_iid.jsonl"
HOLDOUT = EXP / "data" / "repair_val_format_holdout.jsonl"
REPORTS = EXP / "reports"


def core_jobs() -> list[dict[str, str | None]]:
    return [
        {"name": "frozen_trace_iid", "data": str(IID), "condition": "trace", "adapter": None},
        {"name": "frozen_trace_format_holdout", "data": str(HOLDOUT), "condition": "trace", "adapter": None},
        {
            "name": "final_patch_lora_final_patch_iid",
            "data": str(IID),
            "condition": "final_patch",
            "adapter": str(LARGE / "models" / "final_patch_lora"),
        },
        {
            "name": "final_patch_lora_final_patch_format_holdout",
            "data": str(HOLDOUT),
            "condition": "final_patch",
            "adapter": str(LARGE / "models" / "final_patch_lora"),
        },
        {
            "name": "no_trace_lora_no_trace_iid",
            "data": str(IID),
            "condition": "no_trace",
            "adapter": str(LARGE / "models" / "no_trace_lora"),
        },
        {
            "name": "no_trace_lora_no_trace_format_holdout",
            "data": str(HOLDOUT),
            "condition": "no_trace",
            "adapter": str(LARGE / "models" / "no_trace_lora"),
        },
        {
            "name": "shuffled_trace_lora_trace_iid",
            "data": str(IID),
            "condition": "trace",
            "adapter": str(LARGE / "models" / "shuffled_trace_lora"),
        },
        {
            "name": "shuffled_trace_lora_trace_format_holdout",
            "data": str(HOLDOUT),
            "condition": "trace",
            "adapter": str(LARGE / "models" / "shuffled_trace_lora"),
        },
        {
            "name": "trace_lora_trace_iid",
            "data": str(IID),
            "condition": "trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
        {
            "name": "trace_lora_trace_format_holdout",
            "data": str(HOLDOUT),
            "condition": "trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
    ]


def ablation_jobs() -> list[dict[str, str | None]]:
    return [
        {
            "name": "trace_lora_no_trace_iid",
            "data": str(IID),
            "condition": "no_trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
        {
            "name": "trace_lora_no_trace_format_holdout",
            "data": str(HOLDOUT),
            "condition": "no_trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
        {
            "name": "trace_lora_shuffled_trace_iid",
            "data": str(IID),
            "condition": "shuffled_trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
        {
            "name": "trace_lora_shuffled_trace_format_holdout",
            "data": str(HOLDOUT),
            "condition": "shuffled_trace",
            "adapter": str(LARGE / "models" / "trace_lora"),
        },
    ]


def run_job(job: dict[str, str | None], max_new_tokens: int, force: bool) -> None:
    output = REPORTS / f"final_{job['name']}.json"
    legacy_trace_iid = REPORTS / "final_trace_lora_trace_iid.json"
    if job["name"] == "trace_lora_trace_iid" and legacy_trace_iid.exists() and not output.exists():
        output.write_text(legacy_trace_iid.read_text(encoding="utf-8"), encoding="utf-8")
    if output.exists() and not force:
        payload = json.loads(output.read_text(encoding="utf-8"))
        print(f"SKIP {job['name']}: {json.dumps(payload['summary'], sort_keys=True)}", flush=True)
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
    print(f"RUN {job['name']}", flush=True)
    subprocess.run(cmd, check=True)
    payload = json.loads(output.read_text(encoding="utf-8"))
    print(f"DONE {job['name']}: {json.dumps(payload['summary'], sort_keys=True)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["core", "ablation", "all"], default="core")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    jobs = []
    if args.suite in ("core", "all"):
        jobs.extend(core_jobs())
    if args.suite in ("ablation", "all"):
        jobs.extend(ablation_jobs())
    for job in jobs:
        run_job(job, args.max_new_tokens, args.force)


if __name__ == "__main__":
    main()
