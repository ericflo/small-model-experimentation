#!/usr/bin/env python3
"""Run the frozen branch-authorization recovery or one exact producer stage."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src" / "recovery.py"
SPEC = importlib.util.spec_from_file_location("_state_formation_branch_recovery", SOURCE)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load branch-recovery source")
recovery = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recovery
SPEC.loader.exec_module(recovery)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--archive-failure", action="store_true")
    mode.add_argument("--retire-failure", action="store_true")
    mode.add_argument("--stage", choices=recovery.ALLOWED_STAGES)
    parser.add_argument("--archive-commit")
    parser.add_argument("--capacity", choices=("lora", "fullrank"), default="lora")
    parser.add_argument("--objective", choices=("joint", "state_only"), default="joint")
    parser.add_argument("--eval-set", choices=("trigger", "contrast"), default="trigger")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--initialization-bundle")
    parser.add_argument("--model-smoke-receipt")
    parser.add_argument("--positive-control-receipt")
    parser.add_argument("--authorization-receipt")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.retire_failure and not args.archive_commit:
        parser.error("--retire-failure requires --archive-commit")
    if not args.retire_failure and args.archive_commit:
        parser.error("--archive-commit is only valid with --retire-failure")
    if args.stage and args.seed is None:
        parser.error("recovered producer stages require --seed")
    return args


def _producer_argv(args: argparse.Namespace) -> list[str]:
    values: list[tuple[str, Any]] = [
        ("--stage", args.stage),
        ("--capacity", args.capacity),
        ("--objective", args.objective),
        ("--eval-set", args.eval_set),
        ("--seed", args.seed),
        ("--checkpoint", args.checkpoint),
        ("--initialization-bundle", args.initialization_bundle),
        ("--model-smoke-receipt", args.model_smoke_receipt),
        ("--positive-control-receipt", args.positive_control_receipt),
        ("--authorization-receipt", args.authorization_receipt),
        ("--output", args.output),
    ]
    result: list[str] = []
    for flag, value in values:
        if value is not None:
            result.extend((flag, str(value)))
    return result


def _invocation(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "stage": args.stage,
        "capacity": args.capacity,
        "objective": args.objective,
        "eval_set": args.eval_set,
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "initialization_bundle": args.initialization_bundle,
        "model_smoke_receipt": args.model_smoke_receipt,
        "positive_control_receipt": args.positive_control_receipt,
        "authorization_receipt": args.authorization_receipt,
        "output": args.output,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        result = recovery.run_smoke()
    elif args.archive_failure:
        result = recovery.archive_failure()
    elif args.retire_failure:
        result = recovery.retire_failure(args.archive_commit)
    else:
        result = recovery.invoke_producer(_invocation(args), _producer_argv(args))
    public = {
        key: result[key]
        for key in (
            "experiment_id",
            "status",
            "receipt_identity_sha256",
            "producer_status",
            "producer_output",
            "archive_path",
            "archive_commit",
        )
        if key in result
    }
    print(json.dumps(public, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
