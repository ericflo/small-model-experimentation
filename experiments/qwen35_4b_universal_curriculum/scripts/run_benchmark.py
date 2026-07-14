#!/usr/bin/env python3
"""Run one multi-model aggregate-only Menagerie event through the trusted gateway."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
GATEWAY = ROOT / "scripts" / "run_benchmark_aggregate.py"
PUBLIC_FAMILIES = {
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack",
    "sirens", "stockade", "toolsmith", "warren",
}
GATEWAY_KEYS = {
    "schema_version", "stage", "tier", "think_budget", "seed", "backend", "model",
    "model_merge_receipt_sha256", "benchmark_runner_sha256",
    "benchmark_source_inventory_sha256", "benchmark_source_file_count",
    "aggregate", "per_family", "within_budget", "wall_seconds",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_event(
    path: Path, tier: str, think_budget: int | None, seed: int, model: Path
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(payload) != GATEWAY_KEYS
        or payload.get("schema_version") != 1
        or payload.get("stage") != "menagerie_aggregate_gateway"
        or payload.get("tier") != tier
        or payload.get("think_budget") != think_budget
        or payload.get("seed") != seed
        or payload.get("backend") != "qwen_vllm"
        or Path(payload.get("model", "")).resolve() != model.resolve()
        or payload.get("within_budget") is not True
        or set(payload.get("per_family", {})) != PUBLIC_FAMILIES
        or payload.get("model_merge_receipt_sha256") != sha256_file(model / "merge_receipt.json")
    ):
        raise ValueError(f"aggregate gateway event failed authentication: {path}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument("--candidate", required=True, help="candidate label for promotion gates")
    parser.add_argument("--control", default="blend", help="strong-control label")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.think_budget is not None and args.think_budget <= 0:
        parser.error("--think-budget must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe event name")
    models = {}
    for specification in args.model:
        label, separator, raw_path = specification.partition("=")
        path = Path(raw_path).resolve()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
            parser.error(f"invalid model specification: {specification}")
        if label in models or not (path / "merge_receipt.json").is_file():
            parser.error(f"duplicate label or missing merge receipt: {specification}")
        models[label] = path
    if "base" not in models or args.candidate not in models or args.control not in models:
        parser.error("models must include base, candidate, and control labels")
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")

    ledger = EXP / "runs" / "benchmark_events.jsonl"
    prior = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    if any(row.get("seed") == args.seed for row in prior):
        parser.error("benchmark seed already consumed by a completed event")
    budget_label = str(args.think_budget) if args.think_budget is not None else "native"
    output_dir = (
        EXP / "runs" / "benchmark"
        / f"{args.tier}_tb{budget_label}_seed{args.seed}_{args.name}"
    )
    if output_dir.exists() and not args.resume:
        parser.error("partial event exists; use --resume after auditing it")
    output_dir.mkdir(parents=True, exist_ok=True)

    events = {}
    for label, model in models.items():
        output = output_dir / f"{label}.json"
        failure = output_dir / f"{label}.failure.json"
        if failure.exists():
            parser.error(f"preserved failure exists for {label}; use a new event name")
        if not output.exists():
            command = [
                str(PYTHON), str(GATEWAY), "--tier", args.tier, "--seed", str(args.seed),
                "--model", str(model), "--out", str(output),
            ]
            if args.think_budget is not None:
                command.extend(("--think-budget", str(args.think_budget)))
            completed = subprocess.run(
                command, cwd=ROOT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE, text=True, check=False,
            )
            if completed.returncode != 0:
                diagnostic = "unclassified_gateway_failure"
                safe_stderr = completed.stderr.strip()
                aggregate_match = re.fullmatch(
                    r"aggregate benchmark gateway failed; category=([a-z_]+); "
                    r"private output suppressed",
                    safe_stderr,
                )
                runner_match = re.fullmatch(
                    r"benchmark runner failed with exit code \d+; private aggregate "
                    r"state=([a-z_]+); raw stdout/stderr suppressed",
                    safe_stderr,
                )
                if aggregate_match:
                    diagnostic = aggregate_match.group(1)
                elif runner_match:
                    diagnostic = f"runner_failure_{runner_match.group(1)}"
                failure.write_text(
                    json.dumps({
                        "schema_version": 1,
                        "name": args.name,
                        "tier": args.tier,
                        "think_budget": args.think_budget,
                        "seed": args.seed,
                        "arm": label,
                        "model": str(model),
                        "model_merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
                        "gateway_exit_code": completed.returncode,
                        "safe_diagnostic": diagnostic,
                        "score_emitted": False,
                        "raw_streams_exposed": False,
                        "benchmark_output_exposed": False,
                    }, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                raise SystemExit(
                    f"aggregate gateway failed for {label} with exit {completed.returncode} "
                    f"({diagnostic}); "
                    "private output remained suppressed"
                )
        events[label] = load_event(
            output, args.tier, args.think_budget, args.seed, model
        )

    signatures = {
        (
            event["benchmark_runner_sha256"],
            event["benchmark_source_inventory_sha256"],
            event["benchmark_source_file_count"],
        )
        for event in events.values()
    }
    if len(signatures) != 1:
        raise ValueError("benchmark implementation changed between paired arms")

    base = events["base"]
    comparisons = {}
    for label, event in events.items():
        if label == "base":
            continue
        family_delta = {
            family: event["per_family"][family] - base["per_family"][family]
            for family in sorted(PUBLIC_FAMILIES)
        }
        comparisons[f"{label}_minus_base"] = {
            "aggregate_delta": event["aggregate"] - base["aggregate"],
            "per_family_delta": family_delta,
            "positive_families": sum(value > 0 for value in family_delta.values()),
            "nonnegative_families": sum(value >= 0 for value in family_delta.values()),
            "minimum_family_delta": min(family_delta.values()),
        }
    candidate = events[args.candidate]
    control = events[args.control]
    control_delta = {
        family: candidate["per_family"][family] - control["per_family"][family]
        for family in sorted(PUBLIC_FAMILIES)
    }
    candidate_base = comparisons[f"{args.candidate}_minus_base"]
    promotion = {
        "positive_aggregate": candidate_base["aggregate_delta"] > 0,
        "no_negative_family": candidate_base["minimum_family_delta"] >= 0,
        "strictly_positive_every_family": candidate_base["positive_families"] == len(PUBLIC_FAMILIES),
        "beats_control_aggregate": candidate["aggregate"] > control["aggregate"],
    }
    promotion["passes_pilot_gate"] = (
        promotion["positive_aggregate"] and promotion["no_negative_family"]
    )
    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": args.seed,
        "gateway_sha256": sha256_file(GATEWAY),
        "models": {label: str(path) for label, path in models.items()},
        "scores": {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        },
        "comparisons": comparisons,
        f"{args.candidate}_minus_{args.control}": {
            "aggregate_delta": candidate["aggregate"] - control["aggregate"],
            "per_family_delta": control_delta,
        },
        "promotion": promotion,
    }
    result = output_dir / "summary.json"
    if result.exists():
        parser.error("refusing to overwrite event summary")
    result.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "name": args.name, "tier": args.tier,
            "think_budget": args.think_budget, "seed": args.seed,
            "summary": str(result), "summary_sha256": sha256_file(result),
        }, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
