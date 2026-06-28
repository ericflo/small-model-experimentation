#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_eval(args, mode: str, output: Path) -> dict:
    cmd = [
        sys.executable,
        "scripts/eval_repair_synthetic.py",
        "--data",
        str(args.data),
        "--output",
        str(output),
        "--model-id",
        args.model_id,
        "--revision",
        args.revision,
        "--prompt-mode",
        mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    if args.max_records:
        cmd.extend(["--max-records", str(args.max_records)])
    subprocess.run(cmd, check=True)
    return json.loads(output.read_text(encoding="utf-8"))["summary"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", default="cdbee75f17c01a7cc42f958dc650907174af0554")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    scratch = args.output.parent / (args.output.stem + "_parts")
    scratch.mkdir(parents=True, exist_ok=True)
    modes = ["trace", "no_trace", "wrong_patch_only", "trace_only", "gold_file_removed"]
    summaries = {}
    for mode in modes:
        summaries[mode] = run_eval(args, mode, scratch / f"{mode}.json")
    normal = summaries["trace"]["repair_after_first_failure@1"]
    payload = {
        "summaries": summaries,
        "trace_ablation_drop_no_trace": normal - summaries["no_trace"]["repair_after_first_failure@1"],
        "trace_ablation_drop_wrong_patch_only": normal - summaries["wrong_patch_only"]["repair_after_first_failure@1"],
        "trace_ablation_drop_trace_only": normal - summaries["trace_only"]["repair_after_first_failure@1"],
        "trace_ablation_drop_gold_file_removed": normal - summaries["gold_file_removed"]["repair_after_first_failure@1"],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
