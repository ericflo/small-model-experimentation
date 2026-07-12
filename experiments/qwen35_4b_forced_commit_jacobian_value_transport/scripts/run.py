#!/usr/bin/env python3
"""Stage-gated forced-commit seam and Jacobian value-transport harness."""

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
    matching_depth_one,
    matching_first_types,
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
        ["git", *args], cwd=ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    if commit.startswith("PENDING_"):
        raise RuntimeError("design boundary is not anchored")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
    paths = {
        "readme": "experiments/qwen35_4b_forced_commit_jacobian_value_transport/README.md",
        "preregistration": (
            "experiments/qwen35_4b_forced_commit_jacobian_value_transport/"
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
    lens_hash = sha256_file(EXP / config["lens"]["path"])
    result = {
        "schema_version": 1,
        "stage": "design_boundary",
        "scientific_result": False,
        "passed": bool(
            ancestor and observed == expected and lens_hash == str(config["lens"]["sha256"])
        ),
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "lens_observed_sha256": lens_hash,
        "lens_expected_sha256": str(config["lens"]["sha256"]),
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", result)
    if not result["passed"]:
        raise RuntimeError(f"immutable design boundary failed: {result}")
    return result


def _parent_fingerprints() -> tuple[set[str], dict[str, int]]:
    parents = (
        "qwen35_4b_jacobian_value_transport",
        "qwen35_4b_native_thought_jacobian_value_transport",
        "qwen35_4b_native_thought_seam_budget_ladder",
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


def gate_reachability(config: dict[str, Any]) -> dict[str, Any]:
    traces_per_task = int(config["generation"]["traces_per_task"])
    receipts: dict[str, Any] = {}
    for stage, task_key in (
        ("seam_selection", "seam_selection_tasks"),
        ("seam_confirmation", "seam_confirmation_tasks"),
    ):
        tasks = int(config["data"][task_key])
        traces = tasks * traces_per_task
        gates = config["gates"][stage]
        min_forced = math.ceil(float(gates["forced_commit_rate_min"]) * traces - 1e-12)
        min_mixed = int(gates["mixed_policy_tasks_min"])
        possible = bool(
            min_forced <= traces
            and min_mixed <= tasks
            and 2 * min_mixed <= traces
            and float(gates["policy_success_rate_min"]) <= 0.5
            <= float(gates["policy_success_rate_max"])
            and float(gates["policy_parse_rate_min"]) <= 1.0
            and float(gates["forced_parse_rate_min"]) <= 1.0
        )
        receipts[stage] = {
            "tasks": tasks,
            "traces": traces,
            "minimum_forced_commits": min_forced,
            "minimum_mixed_tasks": min_mixed,
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
        raise RuntimeError(f"a frozen seam gate is unreachable: {result}")
    return result


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    if sha256_file(lens_path) != str(config["lens"]["sha256"]):
        raise RuntimeError("frozen replicated lens hash mismatch")
    generation = config["generation"]
    rungs = [int(value) for value in generation["cap_rungs"]]
    if rungs != [256, 512, 1024]:
        raise RuntimeError("frozen cap ladder changed")
    if int(generation["prompt_max_tokens"]) + max(rungs) + int(
        generation["answer_max_tokens"]
    ) > int(generation["total_max_tokens"]):
        raise RuntimeError("context envelope is infeasible")
    parent, parent_counts = _parent_fingerprints()
    splits = build_splits(config)
    all_fingerprints: list[str] = []
    receipts: dict[str, Any] = {}
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(path, rows)
        fingerprints = [task_fingerprint(row) for row in rows]
        all_fingerprints.extend(fingerprints)
        counts = Counter(str(row["first_op"]) for row in rows)
        for row in rows:
            inputs = [example["input"] for example in row["visible"]]
            outputs = tuple(tuple(example["output"]) for example in row["visible"])
            if matching_first_types(inputs, outputs) != {row["first_op"]}:
                raise RuntimeError("visible first-operation identifiability changed")
            if matching_depth_one(inputs, outputs):
                raise RuntimeError("an exact-depth-two row has a depth-one fit")
        receipts[split] = {
            "items": len(rows),
            "path": str(path.relative_to(EXP)),
            "sha256": sha256_file(path),
            "unique_fingerprints": len(set(fingerprints)),
            "parent_overlap": sum(value in parent for value in fingerprints),
            "first_op_counts": {
                name: counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS
            },
            "depth_one_fit_count": 0,
        }
    if len(all_fingerprints) != len(set(all_fingerprints)):
        raise RuntimeError("fresh split fingerprints overlap")
    if any(value["parent_overlap"] for value in receipts.values()):
        raise RuntimeError("fresh rows overlap a scientific parent")
    reachability = gate_reachability(config)
    write_json(RUNS_DIR / "smoke" / "gate_reachability.json", reachability)
    manifest = {
        "schema_version": 1,
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "lens_sha256": sha256_file(lens_path),
        "parent_fingerprint_counts": parent_counts,
        "total_unique_fingerprints": len(set(all_fingerprints)),
        "splits": receipts,
        "firewall": {
            "benchmark_content_used": False,
            "fresh_procedural_only": True,
            "visible_first_type_identifiable": True,
            "visible_depth_one_fit_rejected": True,
        },
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "scientific_result": False,
        "passed": True,
        "items": len(all_fingerprints),
        "unique_fingerprints": len(set(all_fingerprints)),
        "parent_overlap": 0,
        "depth_one_fit_count": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "gate_reachability_pass": reachability["passed"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_model(config: dict[str, Any]):
    from model_ops import QwenCommitModel

    return QwenCommitModel(config)


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    value = config["generation"]
    return {
        "answer_cap": int(value["answer_max_tokens"]),
        "total_max_tokens": int(value["total_max_tokens"]),
        "temperature": float(value["temperature"]),
        "top_p": float(value["top_p"]),
        "top_k": int(value["top_k"]),
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
    lens_state = torch.load(EXP / config["lens"]["path"], map_location="cpu", weights_only=True)
    ranks = {
        int(layer): int(torch.linalg.matrix_rank(direction.float()))
        for layer, direction in lens_state["directions"].items()
        if int(layer) in set(config["lens"]["band"])
    }
    task = read_jsonl(DATA_DIR / "seam_selection.jsonl")[0]
    prepared = model.prepare(
        task_prompt(task, aliases), prompt_max_tokens=int(config["generation"]["prompt_max_tokens"])
    )
    trace = model.generate_trace(
        prepared["input_ids"],
        seed=_stable_seed(int(config["seeds"]["seam_selection_trace"]), task["task_id"], "smoke"),
        thought_cap=8,
        **_generation_kwargs(config),
    )
    if trace["natural_close"] or trace["stopped_by"] == "eos_before_close":
        raise RuntimeError("eight-token smoke unexpectedly terminated before forced replay")
    forced = model.force_commit(
        prepared["input_ids"],
        trace["generated_token_ids"][:8],
        seed=_stable_seed(int(config["seeds"]["seam_selection_answer"]), task["task_id"], "smoke"),
        answer_cap=2,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        top_k=int(config["generation"]["top_k"]),
    )
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and set(ranks.values()) == {24}
        and trace["cache_contract_pass"]
        and forced["cache_contract_pass"]
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "scientific_result": False,
        "outcomes_recorded": False,
        "passed": passed,
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
        "lens_ranks": {str(layer): rank for layer, rank in ranks.items()},
        "trace_forward_input_lengths": trace["forward_input_lengths"],
        "forced_forward_input_lengths": forced["forward_input_lengths"],
        "trace_cache_contract_pass": trace["cache_contract_pass"],
        "forced_cache_contract_pass": forced["cache_contract_pass"],
        "forced_close_injected": True,
        "counterfactual_policy_acknowledged": True,
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
        raise RuntimeError("a passing outcome-blind model smoke is required")


def policy_metrics(rows: list[dict[str, Any]], task_ids: list[str], cap: int) -> dict[str, Any]:
    cap_rows = [row for row in rows if int(row["cap"]) == int(cap)]
    forced = [row for row in cap_rows if row["commit_mode"] == "forced"]
    natural = [row for row in cap_rows if row["commit_mode"] == "natural"]
    malformed = [row for row in cap_rows if row["commit_mode"] == "malformed_pre_cap"]
    parsed = sum(row["parseable"] for row in cap_rows)
    forced_parsed = sum(row["parseable"] for row in forced)
    success = sum(row["correct"] for row in cap_rows)
    mixed = 0
    for task_id in task_ids:
        values = [row for row in cap_rows if row["task_id"] == task_id]
        if any(row["correct"] for row in values) and any(not row["correct"] for row in values):
            mixed += 1
    return {
        "cap": int(cap),
        "traces": len(cap_rows),
        "natural_commits": len(natural),
        "forced_commits": len(forced),
        "malformed_pre_cap": len(malformed),
        "forced_commit_rate": len(forced) / len(cap_rows) if cap_rows else 0.0,
        "policy_parse_rate": parsed / len(cap_rows) if cap_rows else 0.0,
        "forced_parse_rate": forced_parsed / len(forced) if forced else 0.0,
        "policy_success_rate": success / len(cap_rows) if cap_rows else 0.0,
        "policy_successes": success,
        "mixed_policy_tasks": mixed,
        "answer_cap_rate": (
            sum(row["answer_stopped_by"] == "answer_cap" for row in cap_rows) / len(cap_rows)
            if cap_rows else 0.0
        ),
    }


def seam_gate(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        metrics["policy_parse_rate"] >= float(gates["policy_parse_rate_min"])
        and metrics["forced_parse_rate"] >= float(gates["forced_parse_rate_min"])
        and metrics["forced_commit_rate"] >= float(gates["forced_commit_rate_min"])
        and float(gates["policy_success_rate_min"])
        <= metrics["policy_success_rate"]
        <= float(gates["policy_success_rate_max"])
        and metrics["mixed_policy_tasks"] >= int(gates["mixed_policy_tasks_min"])
        and metrics["answer_cap_rate"] <= float(gates["answer_cap_rate_max"])
    )


def _policy_row(
    model: Any,
    config: dict[str, Any],
    task: dict[str, Any],
    prepared: dict[str, Any],
    trace: dict[str, Any],
    *,
    trace_index: int,
    cap: int,
    answer_seed: int,
) -> dict[str, Any]:
    aliases = dict(config["data"]["operation_aliases"])
    close_step = trace["close_step"]
    if close_step is not None and int(close_step) <= int(cap):
        commit_mode = "natural"
        answer_text = trace["answer_text"]
        answer_tokens = trace["answer_tokens"]
        answer_stopped_by = trace["stopped_by"]
        answer_cache_pass = trace["cache_contract_pass"]
        forced_counterfactual = False
    else:
        token_ids = trace["generated_token_ids"]
        if len(token_ids) < cap or model.eos_id in token_ids[:cap] or model.think_close_id in token_ids[:cap]:
            return {
                "task_id": task["task_id"],
                "trace_index": trace_index,
                "cap": cap,
                "commit_mode": "malformed_pre_cap",
                "answer_seed": answer_seed,
                "parseable": False,
                "parsed_alias": None,
                "correct": False,
                "answer_tokens": 0,
                "answer_stopped_by": "unavailable",
                "answer_cache_contract_pass": trace["cache_contract_pass"],
                "counterfactual_to_natural_close": False,
            }
        forced = model.force_commit(
            prepared["input_ids"], token_ids[:cap], seed=answer_seed, **_generation_kwargs(config)
        )
        commit_mode = "forced"
        answer_text = forced["answer_text"]
        answer_tokens = forced["answer_tokens"]
        answer_stopped_by = forced["stopped_by"]
        answer_cache_pass = forced["cache_contract_pass"]
        forced_counterfactual = True
    parsed = parse_alias(answer_text, aliases)
    return {
        "task_id": task["task_id"],
        "trace_index": trace_index,
        "cap": cap,
        "commit_mode": commit_mode,
        "answer_seed": answer_seed,
        "parseable": parsed is not None,
        "parsed_alias": parsed,
        "correct": bool(parsed is not None and verify_answer(task, answer_text, aliases)),
        "answer_tokens": int(answer_tokens),
        "answer_stopped_by": answer_stopped_by,
        "answer_cache_contract_pass": bool(answer_cache_pass),
        "counterfactual_to_natural_close": forced_counterfactual,
        "answer_text": answer_text,
    }


def _generate_seam(
    config: dict[str, Any], *, split: str, caps: list[int], trace_seed_key: str, answer_seed_key: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import torch

    model = _load_model(config)
    aliases = dict(config["data"]["operation_aliases"])
    tasks = read_jsonl(DATA_DIR / f"{split}.jsonl")
    trace_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    max_cap = max(caps)
    total = len(tasks) * int(config["generation"]["traces_per_task"])
    completed = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        prepared = model.prepare(
            task_prompt(task, aliases), prompt_max_tokens=int(config["generation"]["prompt_max_tokens"])
        )
        for trace_index in range(int(config["generation"]["traces_per_task"])):
            trace_seed = _stable_seed(
                int(config["seeds"][trace_seed_key]), task["task_id"], str(trace_index)
            )
            trace = model.generate_trace(
                prepared["input_ids"],
                seed=trace_seed,
                thought_cap=max_cap,
                **_generation_kwargs(config),
            )
            trace_rows.append(
                {
                    "task_id": task["task_id"],
                    "trace_index": trace_index,
                    "trace_seed": trace_seed,
                    "prompt_tokens": prepared["prompt_tokens"],
                    **trace,
                }
            )
            for cap in caps:
                answer_seed = _stable_seed(
                    int(config["seeds"][answer_seed_key]),
                    task["task_id"],
                    str(trace_index),
                    str(cap),
                )
                policy_rows.append(
                    _policy_row(
                        model,
                        config,
                        task,
                        prepared,
                        trace,
                        trace_index=trace_index,
                        cap=cap,
                        answer_seed=answer_seed,
                    )
                )
            completed += 1
            if completed % 4 == 0 or completed == total:
                print(f"{split}: {completed}/{total} traces", flush=True)
    if not all(row["cache_contract_pass"] for row in trace_rows):
        raise RuntimeError("a trace row failed the cache audit")
    if not all(row["answer_cache_contract_pass"] for row in policy_rows):
        raise RuntimeError("a policy answer row failed the cache audit")
    environment = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "trace_sampled_tokens": sum(len(row["generated_token_ids"]) for row in trace_rows),
        "trace_forward_calls": sum(int(row["forward_calls"]) for row in trace_rows),
        "policy_answer_tokens": sum(int(row["answer_tokens"]) for row in policy_rows),
    }
    return trace_rows, policy_rows, environment


def run_seam_selection(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    _require_model_smoke()
    caps = [int(value) for value in config["generation"]["cap_rungs"]]
    traces, policy, environment = _generate_seam(
        config,
        split="seam_selection",
        caps=caps,
        trace_seed_key="seam_selection_trace",
        answer_seed_key="seam_selection_answer",
    )
    trace_path = RUNS_DIR / "seam_selection_traces.jsonl"
    policy_path = RUNS_DIR / "seam_selection_policy_rows.jsonl"
    write_jsonl(trace_path, traces)
    write_jsonl(policy_path, policy)
    task_ids = [row["task_id"] for row in read_jsonl(DATA_DIR / "seam_selection.jsonl")]
    metrics = []
    selected = None
    for cap in caps:
        value = policy_metrics(policy, task_ids, cap)
        value["gate_pass"] = seam_gate(value, config["gates"]["seam_selection"])
        metrics.append(value)
        if selected is None and value["gate_pass"]:
            selected = cap
    result = {
        "schema_version": 1,
        "stage": "seam_selection",
        "scientific_result": True,
        "passed": selected is not None,
        "decision": "FORCED_COMMIT_CAP_SELECTED" if selected is not None else "FORCED_COMMIT_SEAM_FAIL",
        "counterfactual_policy": True,
        "design_passed": design["passed"],
        "selected_cap": selected,
        "selection_policy": "smallest passing cap",
        "items": len(task_ids),
        "traces": len(traces),
        "metrics_by_cap": metrics,
        "trace_rows_sha256": sha256_file(trace_path),
        "policy_rows_sha256": sha256_file(policy_path),
        **environment,
    }
    write_json(RUNS_DIR / "seam_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_seam_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    _require_model_smoke()
    selection_path = RUNS_DIR / "seam_selection.json"
    if not selection_path.exists():
        raise RuntimeError("forced-commit cap selection has not run")
    selection = read_json(selection_path)
    if selection.get("passed") is not True or selection.get("selected_cap") is None:
        raise RuntimeError("no forced-commit cap was selected")
    if sha256_file(RUNS_DIR / "seam_selection_traces.jsonl") != selection["trace_rows_sha256"]:
        raise RuntimeError("selection traces changed after cap freeze")
    if sha256_file(RUNS_DIR / "seam_selection_policy_rows.jsonl") != selection["policy_rows_sha256"]:
        raise RuntimeError("selection policy rows changed after cap freeze")
    cap = int(selection["selected_cap"])
    traces, policy, environment = _generate_seam(
        config,
        split="seam_confirmation",
        caps=[cap],
        trace_seed_key="seam_confirmation_trace",
        answer_seed_key="seam_confirmation_answer",
    )
    trace_path = RUNS_DIR / "seam_confirmation_traces.jsonl"
    policy_path = RUNS_DIR / "seam_confirmation_policy_rows.jsonl"
    write_jsonl(trace_path, traces)
    write_jsonl(policy_path, policy)
    task_ids = [row["task_id"] for row in read_jsonl(DATA_DIR / "seam_confirmation.jsonl")]
    metrics = policy_metrics(policy, task_ids, cap)
    passed = bool(design["passed"] and seam_gate(metrics, config["gates"]["seam_confirmation"]))
    result = {
        "schema_version": 1,
        "stage": "seam_confirmation",
        "scientific_result": True,
        "passed": passed,
        "decision": "FORCED_COMMIT_SEAM_REPLICATED" if passed else "FORCED_COMMIT_SEAM_NOT_REPLICATED",
        "counterfactual_policy": True,
        "selected_cap": cap,
        "items": len(task_ids),
        "traces": len(traces),
        "metrics": metrics,
        "selection_summary_sha256": sha256_file(selection_path),
        "trace_rows_sha256": sha256_file(trace_path),
        "policy_rows_sha256": sha256_file(policy_path),
        **environment,
    }
    write_json(RUNS_DIR / "seam_confirmation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "model-smoke",
            "seam-selection",
            "seam-confirmation",
            "prefix-value",
            "control-calibration",
            "causal-confirmation",
        ),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "model-smoke":
        run_model_smoke(config)
    elif args.stage == "seam-selection":
        run_seam_selection(config)
    elif args.stage == "seam-confirmation":
        run_seam_confirmation(config)
    else:
        unavailable(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
