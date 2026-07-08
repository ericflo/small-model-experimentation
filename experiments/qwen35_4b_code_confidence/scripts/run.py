#!/usr/bin/env python3
"""Run harness: full pipeline = eval_code_conf.py (generate + execute + confidence signals) then analyze.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run (4 problems, k=2), no analysis")
    parser.add_argument("--dataset", choices=["mbpp", "humaneval"], default="mbpp")
    parser.add_argument("--n", type=int, default=260)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument("--visible-tests", type=int, default=1)
    parser.add_argument("--answer-max", type=int, default=420)
    parser.add_argument("--judge-batch-size", type=int, default=16)
    parser.add_argument("--out-name", default=None)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    default_out_name = "code_conf" if args.dataset == "mbpp" else f"{args.dataset}_code_conf"
    out_name = args.out_name or default_out_name
    eval_cmd = [
        sys.executable,
        str(SCRIPTS / "eval_code_conf.py"),
        "--dataset",
        args.dataset,
        "--visible-tests",
        str(args.visible_tests),
        "--answer-max",
        str(args.answer_max),
        "--judge-batch-size",
        str(args.judge_batch_size),
        "--out-name",
        out_name,
    ]
    if args.smoke:
        return subprocess.call(eval_cmd + ["--n", "4", "--k", "2"])
    rc = subprocess.call(eval_cmd + ["--n", str(args.n), "--k", str(args.k)])
    if rc != 0:
        return rc
    if out_name == default_out_name:
        verdict = "verdict.json" if args.dataset == "mbpp" else f"{args.dataset}_verdict.json"
        figure = "code_confidence.png" if args.dataset == "mbpp" else f"{args.dataset}_code_confidence.png"
    else:
        verdict = f"{out_name}_verdict.json"
        figure = f"{out_name}.png"
    title = args.title or ("MBPP" if args.dataset == "mbpp" else "HumanEval")
    return subprocess.call([
        sys.executable,
        str(SCRIPTS / "analyze.py"),
        "--input",
        str(SCRIPTS.parent / "runs" / f"{out_name}.json"),
        "--verdict",
        str(SCRIPTS.parent / "runs" / verdict),
        "--figure",
        str(SCRIPTS.parent / "analysis" / figure),
        "--title",
        title,
    ])


if __name__ == "__main__":
    sys.exit(main())
