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
PROMOTION_RECEIPT = EXP / "runs" / "local" / "seed88014_promotion.json"
LOCAL_SEED = 88014
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
FROZEN_NAME = "pilot"
FROZEN_TIER = "quick"
FROZEN_THINK_BUDGET = 1024
FROZEN_SEED = 78144
FROZEN_PARENT = "designed_fresh_parent"
FROZEN_REPLAY_CONTROL = "replay_repeat"
FROZEN_CANDIDATES = ("axis_curriculum",)
FROZEN_MODEL_PATHS = {
    "base": (
        ROOT / "large_artifacts" / "qwen35_4b_universal_curriculum"
        / "merged" / "base_reserialized"
    ),
    FROZEN_PARENT: (
        ROOT / "large_artifacts" / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
        / "merged" / "designed_fresh"
    ),
    FROZEN_REPLAY_CONTROL: (
        ROOT / "large_artifacts" / EXP.name / "merged" / "replay_repeat"
    ),
    "axis_curriculum": ROOT / "large_artifacts" / EXP.name / "merged" / "axis_curriculum",
}
# External arms authenticate against these frozen full-weight hashes; this
# experiment's arms authenticate against their committed merge receipts.
FROZEN_EXTERNAL_WEIGHTS = {
    "base": "b654e033d525d87cbbd746bb681d80813c4b00d8e6202cb3edcfb6dfa3b416db",
    FROZEN_PARENT: "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_clean_pushed_main(paths: list[Path]) -> None:
    if git_output(["git", "status", "--short"]):
        raise ValueError("benchmark event requires a clean worktree")
    if git_output(["git", "branch", "--show-current"]) != "main":
        raise ValueError("benchmark event requires branch main")
    if git_output(["git", "rev-parse", "HEAD"]) != git_output(["git", "rev-parse", "origin/main"]):
        raise ValueError("benchmark event requires HEAD == origin/main")
    for path in paths:
        relative = path.resolve().relative_to(ROOT).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout
        if not path.is_file() or path.read_bytes() != committed:
            raise ValueError(f"benchmark prerequisite differs from HEAD: {relative}")


def authenticate_model_weights(label: str, model: Path) -> str:
    """Bind the arm to real weights, not just a self-consistent receipt."""
    weights = model / "model.safetensors"
    observed = sha256_file(weights)
    if label in FROZEN_EXTERNAL_WEIGHTS:
        expected = FROZEN_EXTERNAL_WEIGHTS[label]
    else:
        merge_receipt = EXP / "runs" / "merges" / f"{label}.json"
        require_clean_pushed_main([merge_receipt])
        payload = json.loads(merge_receipt.read_text(encoding="utf-8"))
        files = payload.get("weight_files", [])
        if (
            Path(payload.get("merged", "")).resolve() != model.resolve()
            or len(files) != 1
            or files[0].get("name") != "model.safetensors"
        ):
            raise ValueError(f"merge receipt does not describe this composite: {label}")
        expected = files[0].get("sha256")
    if observed != expected:
        raise ValueError(f"benchmark arm weights changed for {label}: {observed}")
    return observed


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


def authenticate_local_promotion(candidate: str) -> dict:
    if not PROMOTION_RECEIPT.is_file():
        raise ValueError("local promotion receipt is absent; benchmark stays sealed")
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("seed") != LOCAL_SEED
        or payload.get("aggregate_seed") != FROZEN_SEED
        or payload.get("aggregate_seed_open") is not True
        or payload.get("benchmark_data_read") is not False
        or payload.get("promoted") != candidate
    ):
        raise ValueError(
            "local promotion receipt does not authorize this candidate benchmark"
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--tier", choices=("quick", "medium"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--think-budget", type=int)
    parser.add_argument("--model", action="append", required=True, help="label=/merged/model")
    parser.add_argument(
        "--candidate", choices=FROZEN_CANDIDATES, required=True,
        help="the single locally promoted candidate label",
    )
    parser.add_argument("--parent", default=FROZEN_PARENT, help="immediate parent label")
    parser.add_argument(
        "--replay-control", default=FROZEN_REPLAY_CONTROL,
        help="exact-token-matched replay continuation",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (
        args.name != FROZEN_NAME
        or args.tier != FROZEN_TIER
        or args.seed != FROZEN_SEED
        or args.think_budget != FROZEN_THINK_BUDGET
        or args.parent != FROZEN_PARENT
        or args.replay_control != FROZEN_REPLAY_CONTROL
    ):
        parser.error("benchmark event differs from the preregistered pilot")
    if args.think_budget is not None and args.think_budget <= 0:
        parser.error("--think-budget must be positive")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", args.name):
        parser.error("unsafe event name")
    try:
        promotion = authenticate_local_promotion(args.candidate)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    models = {}
    for specification in args.model:
        label, separator, raw_path = specification.partition("=")
        path = Path(raw_path).resolve()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", label):
            parser.error(f"invalid model specification: {specification}")
        if label in models or not (path / "merge_receipt.json").is_file():
            parser.error(f"duplicate label or missing merge receipt: {specification}")
        if label not in FROZEN_MODEL_PATHS or path != FROZEN_MODEL_PATHS[label].resolve():
            parser.error(f"model path differs from the frozen arm: {specification}")
        models[label] = path
    required_labels = {"base", args.parent, args.replay_control, args.candidate}
    if set(models) != required_labels or len(models) != 4:
        parser.error(
            "models must be exactly base, immediate parent, replay control, "
            "and the promoted candidate"
        )
    if len(set(models.values())) != len(models):
        parser.error("every benchmark arm must name a distinct merged model")
    try:
        require_clean_pushed_main([PROMOTION_RECEIPT])
        weight_hashes = {
            label: authenticate_model_weights(label, model)
            for label, model in sorted(models.items())
        }
    except (OSError, ValueError, json.JSONDecodeError, subprocess.CalledProcessError) as error:
        parser.error(str(error))

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
    base_comparison = comparisons[f"{args.candidate}_minus_base"]
    candidate_comparisons = {args.candidate: {}}
    for control_label in (args.parent, args.replay_control):
        control = events[control_label]
        candidate_comparisons[args.candidate][f"{args.candidate}_minus_{control_label}"] = {
            "aggregate_delta": candidate["aggregate"] - control["aggregate"],
            "per_family_delta": {
                family: candidate["per_family"][family] - control["per_family"][family]
                for family in sorted(PUBLIC_FAMILIES)
            },
        }
    promotion_gate = {
        "strictly_positive_aggregate_vs_base": base_comparison["aggregate_delta"] > 0,
        "beats_replay_control_aggregate": (
            candidate["aggregate"] > events[args.replay_control]["aggregate"]
        ),
        "beats_immediate_parent_aggregate": (
            candidate["aggregate"] > events[args.parent]["aggregate"]
        ),
    }
    promotion_gate["passes_pilot_gate"] = all(promotion_gate.values())
    # The every-family goal gate is recorded and reported from the same event
    # but is NOT part of the pilot pass: the frozen power statement expects it
    # to fail at quick-tier granularity even under the hypothesis.
    goal_gate = {
        "strictly_positive_every_family_vs_base": (
            base_comparison["positive_families"] == len(PUBLIC_FAMILIES)
        ),
        "positive_families_vs_base": base_comparison["positive_families"],
        "family_count": len(PUBLIC_FAMILIES),
        "included_in_pilot_gate": False,
        "power_statement": (
            "quick-tier family scores are ~1/8-step granular with base at zero "
            "on several families; a FAIL here is the expected outcome even "
            "under the hypothesis and reads as 'not confirmed at quick-tier "
            "granularity', never as evidence against the mechanism"
        ),
    }
    promotions = {args.candidate: promotion_gate}
    payload = {
        "schema_version": 1,
        "name": args.name,
        "tier": args.tier,
        "think_budget": args.think_budget,
        "seed": args.seed,
        "gateway_sha256": sha256_file(GATEWAY),
        "candidate": args.candidate,
        "local_promotion_receipt": str(PROMOTION_RECEIPT.resolve()),
        "local_promotion_receipt_sha256": sha256_file(PROMOTION_RECEIPT),
        "local_promotion_promoted": promotion["promoted"],
        "models": {label: str(path) for label, path in models.items()},
        "model_weight_sha256s": weight_hashes,
        "scores": {
            label: {"aggregate": event["aggregate"], "per_family": event["per_family"]}
            for label, event in events.items()
        },
        "comparisons": comparisons,
        "candidate_comparisons": candidate_comparisons,
        "promotions": promotions,
        "goal_gate": {args.candidate: goal_gate},
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
