#!/usr/bin/env python3
"""Stage-gated native-thought seam budget selection and confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    build_splits,
    parse_alias,
    task_fingerprint,
    task_prompt,
    verify_answer,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    if commit.startswith("PENDING_"):
        raise RuntimeError("design boundary has not been anchored to a published commit")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
    paths = {
        "readme": "experiments/qwen35_4b_native_thought_seam_budget_ladder/README.md",
        "preregistration": (
            "experiments/qwen35_4b_native_thought_seam_budget_ladder/"
            "reports/preregistration.md"
        ),
    }
    observed = {
        name: hashlib.sha256(_git(["show", f"{commit}:{path}"]).stdout.encode()).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "readme": str(boundary["readme_sha256"]),
        "preregistration": str(boundary["preregistration_sha256"]),
    }
    result = {
        "schema_version": 1,
        "stage": "design_boundary",
        "passed": bool(ancestor and observed == expected),
        "scientific_result": False,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", result)
    if not result["passed"]:
        raise RuntimeError(f"immutable design boundary failed: {result}")
    return result


def wilson_lower(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total == 0:
        return 0.0
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = rate + z * z / (2.0 * total)
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
    return (center - radius) / denominator


def metrics_at_cap(
    rows: list[dict[str, Any]],
    task_ids: list[str],
    *,
    cap: int,
    minimum_think_tokens: int,
) -> dict[str, Any]:
    classified: list[dict[str, Any]] = []
    for row in rows:
        close_step = row.get("close_step")
        natural = close_step is not None and int(close_step) <= int(cap)
        parseable = natural and row.get("parsed_alias") is not None
        usable = parseable and int(row["think_tokens"]) >= int(minimum_think_tokens)
        classified.append(
            {
                **row,
                "natural_at_cap": natural,
                "parseable_at_cap": parseable,
                "usable_at_cap": usable,
                "correct_at_cap": bool(usable and row["correct"]),
            }
        )
    total = len(classified)
    closed = sum(row["natural_at_cap"] for row in classified)
    parsed = sum(row["parseable_at_cap"] for row in classified)
    usable_rows = [row for row in classified if row["usable_at_cap"]]
    successes = sum(row["correct_at_cap"] for row in usable_rows)
    mixed = 0
    for task_id in task_ids:
        task_rows = [row for row in usable_rows if row["task_id"] == task_id]
        if any(row["correct_at_cap"] for row in task_rows) and any(
            not row["correct_at_cap"] for row in task_rows
        ):
            mixed += 1
    close_steps = [int(row["close_step"]) for row in classified if row["natural_at_cap"]]
    return {
        "cap": int(cap),
        "traces": total,
        "natural_closes": closed,
        "natural_close_rate": closed / total if total else 0.0,
        "natural_close_wilson_lower": wilson_lower(closed, total),
        "parseable": parsed,
        "parse_rate_all": parsed / total if total else 0.0,
        "conditional_parse_rate": parsed / closed if closed else 0.0,
        "usable_traces": len(usable_rows),
        "usable_successes": successes,
        "usable_success_rate": successes / len(usable_rows) if usable_rows else 0.0,
        "mixed_usable_tasks": mixed,
        "cap_contacts": total - closed,
        "max_close_step": max(close_steps) if close_steps else None,
    }


def gate_pass(metrics: dict[str, Any], gates: dict[str, Any], *, confirmation: bool) -> bool:
    passed = bool(
        metrics["natural_close_rate"] >= float(gates["natural_close_rate_min"])
        and metrics["conditional_parse_rate"] >= float(gates["conditional_parse_rate_min"])
        and metrics["usable_traces"] >= int(gates["usable_traces_min"])
        and float(gates["usable_success_rate_min"])
        <= metrics["usable_success_rate"]
        <= float(gates["usable_success_rate_max"])
        and metrics["mixed_usable_tasks"] >= int(gates["mixed_usable_tasks_min"])
    )
    if confirmation:
        passed = passed and metrics["natural_close_wilson_lower"] >= float(
            gates["natural_close_wilson_lower_min"]
        )
    return bool(passed)


def gate_reachability(config: dict[str, Any]) -> dict[str, Any]:
    traces_per_task = int(config["generation"]["traces_per_task"])
    receipts: dict[str, Any] = {}
    for stage, task_key in (
        ("selection", "budget_selection_tasks"),
        ("confirmation", "seam_confirmation_tasks"),
    ):
        tasks = int(config["data"][task_key])
        total = tasks * traces_per_task
        gates = config["gates"][stage]
        minimum_close = math.ceil(float(gates["natural_close_rate_min"]) * total - 1e-12)
        minimum_usable = int(gates["usable_traces_min"])
        minimum_mixed = int(gates["mixed_usable_tasks_min"])
        possible = bool(
            minimum_close <= total
            and minimum_usable <= total
            and 2 * minimum_mixed <= minimum_usable
            and minimum_mixed <= tasks
            and float(gates["usable_success_rate_min"]) <= 0.5
            <= float(gates["usable_success_rate_max"])
        )
        if stage == "confirmation":
            possible = possible and wilson_lower(total, total) >= float(
                gates["natural_close_wilson_lower_min"]
            )
        receipts[stage] = {
            "tasks": tasks,
            "traces": total,
            "minimum_close_count": minimum_close,
            "minimum_usable_count": minimum_usable,
            "minimum_mixed_tasks": minimum_mixed,
            "passed": possible,
        }
    result = {
        "schema_version": 1,
        "stage": "gate_reachability",
        "scientific_result": False,
        "passed": all(value["passed"] for value in receipts.values()),
        "receipts": receipts,
    }
    if not result["passed"]:
        raise RuntimeError(f"a frozen terminal gate is mathematically unreachable: {result}")
    return result


def _parent_fingerprints() -> tuple[set[str], dict[str, int]]:
    parents = (
        "qwen35_4b_jacobian_value_transport",
        "qwen35_4b_native_thought_jacobian_value_transport",
    )
    combined: set[str] = set()
    counts: dict[str, int] = {}
    for parent in parents:
        values: set[str] = set()
        directory = ROOT / "experiments" / parent / "data" / "procedural"
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                    values.add(task_fingerprint(row))
        counts[parent] = len(values)
        combined.update(values)
    return combined, counts


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    generation = config["generation"]
    rungs = [int(value) for value in generation["budget_rungs"]]
    if rungs != sorted(set(rungs)) or rungs != [256, 512, 1024]:
        raise RuntimeError("the frozen budget ladder must be exactly [256, 512, 1024]")
    if (
        int(generation["prompt_max_tokens"])
        + max(rungs)
        + int(generation["answer_max_tokens"])
        > int(generation["total_max_tokens"])
    ):
        raise RuntimeError("frozen context envelope is infeasible")
    aliases = dict(config["data"]["operation_aliases"])
    if len(set(aliases.values())) != len(aliases):
        raise RuntimeError("operation aliases must be one-to-one")
    parent, parent_counts = _parent_fingerprints()
    splits = build_splits(config)
    fingerprints: list[str] = []
    split_receipts: dict[str, Any] = {}
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(path, rows)
        values = [task_fingerprint(row) for row in rows]
        fingerprints.extend(values)
        counts = Counter(str(row["first_op"]) for row in rows)
        split_receipts[split] = {
            "items": len(rows),
            "path": str(path.relative_to(EXP)),
            "sha256": sha256_file(path),
            "unique_fingerprints": len(set(values)),
            "parent_overlap": sum(value in parent for value in values),
            "first_op_counts": {
                name: counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS
            },
        }
    if len(fingerprints) != len(set(fingerprints)):
        raise RuntimeError("new procedural splits overlap each other")
    if any(receipt["parent_overlap"] for receipt in split_receipts.values()):
        raise RuntimeError("new procedural rows overlap a scientific parent")
    reachability = gate_reachability(config)
    write_json(RUNS_DIR / "smoke" / "gate_reachability.json", reachability)
    manifest = {
        "schema_version": 1,
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "budget_rungs": rungs,
        "operation_aliases": aliases,
        "identifiable_first_operations": list(IDENTIFIABLE_FIRST_OPERATIONS),
        "parent_fingerprint_counts": parent_counts,
        "total_unique_fingerprints": len(set(fingerprints)),
        "splits": split_receipts,
        "firewall": {
            "benchmark_content_used": False,
            "fresh_procedural_only": True,
            "visible_first_type_identifiable_by_exhaustive_enumeration": True,
        },
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "items": len(fingerprints),
        "unique_fingerprints": len(set(fingerprints)),
        "parent_overlap": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "gate_reachability_pass": reachability["passed"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_model(config: dict[str, Any]):
    from model_ops import QwenCachedThinkModel

    return QwenCachedThinkModel(config)


def _generation_kwargs(config: dict[str, Any], cap: int) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "max_think_steps": int(cap),
        "answer_max_tokens": int(generation["answer_max_tokens"]),
        "total_max_tokens": int(generation["total_max_tokens"]),
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
    }


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import torch
    import transformers

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = _load_model(config)
    aliases = dict(config["data"]["operation_aliases"])
    alias_ids = {alias: model.leading_space_token_id(alias) for alias in aliases.values()}
    if len(set(alias_ids.values())) != len(alias_ids):
        raise RuntimeError("alias token IDs are not one-to-one")
    task = read_jsonl(DATA_DIR / "budget_selection.jsonl")[0]
    prepared = model.prepare(
        task_prompt(task, aliases),
        prompt_max_tokens=int(config["generation"]["prompt_max_tokens"]),
    )
    generated = model.generate(
        prepared["input_ids"],
        seed=_stable_seed(int(config["seeds"]["budget_selection"]), task["task_id"], "smoke"),
        **_generation_kwargs(config, 8),
    )
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and generated["cache_contract_pass"]
        and len(generated["generated_token_ids"]) >= 1
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "passed": passed,
        "scientific_result": False,
        "outcomes_recorded": False,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "layers": model.n_layers,
            "hidden_size": model.d_model,
            "vocab_size": model.vocab_size,
            "load_seconds": model.load_seconds,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "token_ids": {
            "think_open": model.think_open_id,
            "think_close": model.think_close_id,
            "eos": model.eos_id,
            "aliases": alias_ids,
        },
        "prompt_tokens": prepared["prompt_tokens"],
        "generated_tokens": len(generated["generated_token_ids"]),
        "cache_contract_pass": generated["cache_contract_pass"],
        "forward_input_lengths": generated["forward_input_lengths"],
        "correctness_recorded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUNS_DIR / "model_smoke" / "result.json", result)
    if not passed:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _require_model_smoke() -> None:
    path = RUNS_DIR / "model_smoke" / "result.json"
    if not path.exists() or read_json(path).get("passed") is not True:
        raise RuntimeError("a passing model smoke is required")


def _generate_rows(
    config: dict[str, Any],
    *,
    split: str,
    cap: int,
    seed_key: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    aliases = dict(config["data"]["operation_aliases"])
    tasks = read_jsonl(DATA_DIR / f"{split}.jsonl")
    model = _load_model(config)
    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    total = len(tasks) * int(config["generation"]["traces_per_task"])
    completed = 0
    for task in tasks:
        prepared = model.prepare(
            task_prompt(task, aliases),
            prompt_max_tokens=int(config["generation"]["prompt_max_tokens"]),
        )
        for trace_index in range(int(config["generation"]["traces_per_task"])):
            seed = _stable_seed(
                int(config["seeds"][seed_key]), task["task_id"], str(trace_index)
            )
            generated = model.generate(
                prepared["input_ids"], seed=seed, **_generation_kwargs(config, cap)
            )
            parsed = parse_alias(generated["answer_text"], aliases)
            rows.append(
                {
                    "task_id": task["task_id"],
                    "trace_index": trace_index,
                    "seed": seed,
                    "prompt_tokens": prepared["prompt_tokens"],
                    "max_think_steps": cap,
                    "natural_close": generated["natural_close"],
                    "close_step": generated["close_step"],
                    "think_tokens": generated["think_tokens"],
                    "answer_tokens": generated["answer_tokens"],
                    "stopped_by": generated["stopped_by"],
                    "parsed_alias": parsed,
                    "correct": bool(
                        generated["natural_close"]
                        and parsed is not None
                        and verify_answer(task, generated["answer_text"], aliases)
                    ),
                    "cache_contract_pass": generated["cache_contract_pass"],
                    "forward_calls": generated["forward_calls"],
                    "generated_token_ids": generated["generated_token_ids"],
                    "generated_text": generated["generated_text"],
                }
            )
            completed += 1
            if completed % 4 == 0 or completed == total:
                print(f"{split}: {completed}/{total} traces", flush=True)
    if not all(row["cache_contract_pass"] for row in rows):
        raise RuntimeError("the KV-cache token-length contract failed in a scientific row")
    environment = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "sampled_tokens": sum(len(row["generated_token_ids"]) for row in rows),
        "forward_calls": sum(int(row["forward_calls"]) for row in rows),
    }
    return rows, environment


def run_budget_selection(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    _require_model_smoke()
    rungs = [int(value) for value in config["generation"]["budget_rungs"]]
    rows, environment = _generate_rows(
        config,
        split="budget_selection",
        cap=max(rungs),
        seed_key="budget_selection",
    )
    rows_path = RUNS_DIR / "budget_selection_rows.jsonl"
    write_jsonl(rows_path, rows)
    tasks = read_jsonl(DATA_DIR / "budget_selection.jsonl")
    task_ids = [str(task["task_id"]) for task in tasks]
    metrics = []
    selected_cap = None
    for cap in rungs:
        value = metrics_at_cap(
            rows,
            task_ids,
            cap=cap,
            minimum_think_tokens=int(config["generation"]["minimum_usable_think_tokens"]),
        )
        value["gate_pass"] = gate_pass(value, config["gates"]["selection"], confirmation=False)
        metrics.append(value)
        if selected_cap is None and value["gate_pass"]:
            selected_cap = cap
    result = {
        "schema_version": 1,
        "stage": "budget_selection",
        "passed": selected_cap is not None,
        "decision": "BUDGET_SELECTED" if selected_cap is not None else "NO_BUDGET_SELECTED",
        "scientific_result": True,
        "design_passed": design["passed"],
        "selection_policy": "smallest passing cap",
        "nested_right_censoring": True,
        "selected_cap": selected_cap,
        "items": len(tasks),
        "traces": len(rows),
        "metrics_by_cap": metrics,
        "rows_sha256": sha256_file(rows_path),
        **environment,
    }
    write_json(RUNS_DIR / "budget_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_seam_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    _require_model_smoke()
    selection_path = RUNS_DIR / "budget_selection.json"
    if not selection_path.exists():
        raise RuntimeError("budget selection has not run")
    selection = read_json(selection_path)
    if selection.get("passed") is not True or selection.get("selected_cap") is None:
        raise RuntimeError("no budget was selected; confirmation is ineligible")
    selection_rows = RUNS_DIR / "budget_selection_rows.jsonl"
    if sha256_file(selection_rows) != selection["rows_sha256"]:
        raise RuntimeError("budget-selection rows changed after cap selection")
    selected_cap = int(selection["selected_cap"])
    rows, environment = _generate_rows(
        config,
        split="seam_confirmation",
        cap=selected_cap,
        seed_key="seam_confirmation",
    )
    rows_path = RUNS_DIR / "seam_confirmation_rows.jsonl"
    write_jsonl(rows_path, rows)
    tasks = read_jsonl(DATA_DIR / "seam_confirmation.jsonl")
    metrics = metrics_at_cap(
        rows,
        [str(task["task_id"]) for task in tasks],
        cap=selected_cap,
        minimum_think_tokens=int(config["generation"]["minimum_usable_think_tokens"]),
    )
    passed = bool(
        design["passed"]
        and gate_pass(metrics, config["gates"]["confirmation"], confirmation=True)
    )
    result = {
        "schema_version": 1,
        "stage": "seam_confirmation",
        "passed": passed,
        "decision": "NATURAL_SEAM_REPLICATED" if passed else "SEAM_NOT_REPLICATED",
        "scientific_result": True,
        "selected_cap": selected_cap,
        "selection_summary_sha256": sha256_file(selection_path),
        "items": len(tasks),
        "traces": len(rows),
        "metrics": metrics,
        "rows_sha256": sha256_file(rows_path),
        **environment,
    }
    write_json(RUNS_DIR / "seam_confirmation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "budget-selection", "seam-confirmation"),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "model-smoke":
        run_model_smoke(config)
    elif args.stage == "budget-selection":
        run_budget_selection(config)
    else:
        run_seam_confirmation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
