#!/usr/bin/env python3
"""Run gated depth-5 search controls and the matched-compute direct-sampling baseline."""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import math
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))
import experiment_common as C  # noqa: E402
import families as F  # noqa: E402
import model_scoring as M  # noqa: E402
import search_core as S  # noqa: E402
from vllm_runner import EngineConfig, MODEL_ID, MODEL_REVISION, SamplingConfig, VLLMRunner  # noqa: E402


RESULT_METHODS = (
    "thinking",
    "thinking_shuffled",
    "nothink",
    "nextop",
    "uniform_seeded",
    "surface",
    "budget_truncated_brute",
    "oracle_live",
    "direct_sample_more",
    "direct_sample_more_total",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    temporary.replace(path)


def _save_result(method: str, suffix: str, result: dict[str, Any]) -> None:
    path = EXP / "runs" / f"search_{method}{suffix}.json"
    _atomic_json(path, result)


def _input_paths(suffix: str) -> dict[str, Path]:
    paths = {
        "config": EXP / "configs" / "default.yaml",
        "tasks": EXP / "data" / f"primary_tasks{suffix}.jsonl",
        "oracles": EXP / "data" / f"primary_oracle{suffix}.jsonl",
        "calibration_candidates": EXP / "data" / f"calibration_candidates{suffix}.jsonl",
        "calibration_verdict": EXP / "runs" / f"calibration_verdict{suffix}.json",
        "oracle_gate": EXP / "runs" / f"oracle_gate{suffix}.json",
        "data_audit": EXP / "runs" / f"data_audit{suffix}.json",
        "run_search": Path(__file__).resolve(),
        "search_core": EXP / "scripts" / "search_core.py",
        "experiment_common": EXP / "scripts" / "experiment_common.py",
        "model_scoring": EXP / "scripts" / "model_scoring.py",
        "families": EXP / "src" / "families.py",
        "vllm_runner": EXP / "src" / "vllm_runner.py",
        "full_brute": EXP / "runs" / f"full_brute{suffix}.json",
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise RuntimeError("missing mandatory search inputs: " + ", ".join(missing))
    return paths


def _fingerprints(suffix: str) -> dict[str, str]:
    return {name: _sha256(path) for name, path in _input_paths(suffix).items()}


def _receipt_path(suffix: str) -> Path:
    return EXP / "runs" / f"search_model_receipt{suffix}.json"


def _result_paths(suffix: str) -> dict[str, Path]:
    return {method: EXP / "runs" / f"search_{method}{suffix}.json" for method in RESULT_METHODS}


def _auxiliary_paths(suffix: str) -> dict[str, Path]:
    paths = {
        f"scores_{method}": _score_path(method, suffix)
        for method in ("thinking", "thinking_shuffled", "nothink", "nextop")
    }
    for shard in (0, 1):
        paths[f"direct_pool_shard{shard}"] = (
            EXP / "runs" / f"direct_sample_pool_shard{shard}{suffix}.jsonl"
        )
        paths[f"direct_pool_shard{shard}_receipt"] = (
            EXP / "runs" / f"direct_sample_pool_shard{shard}{suffix}.receipt.json"
        )
    return paths


def _valid_complete_cache(suffix: str, inputs: dict[str, str]) -> bool:
    path = _receipt_path(suffix)
    if not path.exists():
        return False
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 2 or receipt.get("input_sha256") != inputs:
        raise RuntimeError("search receipt is stale or incompatible; rerun with --recompute")
    expected = _result_paths(suffix)
    recorded = receipt.get("result_sha256", {})
    for method, result_path in expected.items():
        if not result_path.exists() or recorded.get(method) != _sha256(result_path):
            raise RuntimeError("search cache is incomplete or corrupted; rerun with --recompute")
    for name, auxiliary_path in _auxiliary_paths(suffix).items():
        if (
            not auxiliary_path.exists()
            or receipt.get("auxiliary_sha256", {}).get(name) != _sha256(auxiliary_path)
        ):
            raise RuntimeError(
                "search auxiliary cache is incomplete or corrupted; rerun with --recompute"
            )
    return True


def _clear_search_cache(suffix: str) -> None:
    paths = list(_result_paths(suffix).values()) + list(_auxiliary_paths(suffix).values())
    paths.extend(
        [_receipt_path(suffix), EXP / "runs" / f"search_verdict{suffix}.json"]
    )
    for path in paths:
        path.unlink(missing_ok=True)


def _validate_source_artifacts(payload: dict[str, Any], receipt_name: str) -> None:
    sources = payload.get("source_artifacts")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError(f"{receipt_name} has no relational source fingerprints")
    for relative, fingerprint in sources.items():
        path = EXP / str(relative)
        expected = fingerprint.get("sha256") if isinstance(fingerprint, dict) else None
        if not path.is_file() or expected != _sha256(path):
            raise RuntimeError(f"{receipt_name} is stale for source artifact {relative}")


def _validate_launch_inputs(
    cfg: dict[str, Any], suffix: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    for name in ("data_audit", "oracle_gate"):
        path = EXP / "runs" / f"{name}{suffix}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not payload.get("passed"):
            raise RuntimeError(f"mandatory {name} did not pass")
        _validate_source_artifacts(payload, name)
    calibration = json.loads(
        (EXP / "runs" / f"calibration_verdict{suffix}.json").read_text(encoding="utf-8")
    )
    _validate_source_artifacts(calibration, "calibration_verdict")
    tasks = C.load_jsonl(EXP / "data" / f"primary_tasks{suffix}.jsonl")
    oracle_rows = C.load_jsonl(EXP / "data" / f"primary_oracle{suffix}.jsonl")
    expected_n = int(cfg["task"]["smoke_n"] if suffix else cfg["task"]["primary_n"])
    expected_depth = int(cfg["task"]["primary_depth"])
    task_ids = [str(task.get("task_id")) for task in tasks]
    oracle_ids = [str(row.get("task_id")) for row in oracle_rows]
    if len(tasks) != expected_n or len(set(task_ids)) != expected_n:
        raise RuntimeError("primary task count or uniqueness does not match frozen config")
    if set(task_ids) != set(oracle_ids) or len(set(oracle_ids)) != len(oracle_ids):
        raise RuntimeError("primary task/oracle IDs are not an exact one-to-one match")
    for task in tasks:
        if int(task.get("depth", -1)) != expected_depth:
            raise RuntimeError("primary task depth does not match frozen config")
        audit = task.get("min_depth_audit", {})
        required_levels = list(range(1, expected_depth))
        if not (
            audit.get("algorithm") == "uncapped_behavioral_bfs"
            and audit.get("seen_cap") is None
            and audit.get("exhaustive_decision") is True
            and audit.get("within_limit") is False
            and audit.get("found_depth") is None
            and int(audit.get("max_depth", -1)) == expected_depth - 1
            and audit.get("levels_fully_exhausted") == required_levels
        ):
            raise RuntimeError(f"invalid exact-depth receipt for {task['task_id']}")
    brute = json.loads(
        (EXP / "runs" / f"full_brute{suffix}.json").read_text(encoding="utf-8")
    )
    if not (
        brute.get("exact") is True
        and int(brute.get("task_count", -1)) == expected_n
        and brute.get("task_source_sha256")
        == _sha256(EXP / "data" / f"primary_tasks{suffix}.jsonl")
    ):
        raise RuntimeError("full-brute receipt is missing, stale, or non-exact")
    return tasks, {str(row["task_id"]): row for row in oracle_rows}


def _score_path(method: str, suffix: str) -> Path:
    return EXP / "runs" / f"search_scores_{method}{suffix}.jsonl"


def _append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _candidate_record(
    task: dict[str, Any], prefix: tuple[str, ...], method: str, layer: int, index: int
) -> dict[str, Any]:
    return C.calibration_record(
        task,
        prefix,
        record_id=f"{method}:{task['task_id']}:l{layer}:c{index:04d}",
        method=method,
        layer=layer,
        parent_prefix=list(prefix[:-1]),
        child_operation=prefix[-1],
    )


def _accumulate(counter: Counter[str], accounting: dict[str, Any]) -> None:
    for key, value in accounting.items():
        if isinstance(value, int) and not isinstance(value, bool):
            counter[key] += value


def _run_viability_arm(
    scorer: M.ModelScorer,
    tasks: list[dict[str, Any]],
    *,
    method: str,
    beam: int,
    thinking_budget: int,
    fill_cap: int,
    run_seed: int,
    suffix: str,
) -> dict[str, Any]:
    thinking = method in {"thinking", "thinking_shuffled"}
    frontiers = {str(task["task_id"]): [()] for task in tasks}
    model_work = {str(task["task_id"]): Counter() for task in tasks}
    layer_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_file = _score_path(method, suffix)
    score_file.unlink(missing_ok=True)
    started = time.perf_counter()
    for layer in range(1, int(tasks[0]["depth"]) + 1):
        records = []
        candidate_by_id = {}
        grouped: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        for task in tasks:
            task_id = str(task["task_id"])
            candidates = S.expand(frontiers[task_id])
            for index, prefix in enumerate(candidates):
                record = _candidate_record(task, prefix, method, layer, index)
                records.append(record)
                candidate_by_id[str(record["id"])] = (task_id, prefix, tuple(prefix[:-1]))
                grouped[task_id].append(prefix)
        if thinking:
            scored, summary = scorer.score_thinking_viability(
                records, thinking_budget=thinking_budget, run_seed=run_seed + layer
            )
        else:
            scored, summary = scorer.score_no_think_viability(records, run_seed=run_seed + layer)
        scores_by_task: dict[str, dict[tuple[str, ...], float]] = defaultdict(dict)
        enriched = []
        for record, row in zip(records, scored):
            task_id, prefix, parent = candidate_by_id[str(row["id"])]
            scores_by_task[task_id][prefix] = float(row["p_viable"])
            _accumulate(model_work[task_id], row["accounting"])
            row.update(
                {
                    "task_id": task_id,
                    "layer": layer,
                    "candidate_prefix": list(prefix),
                    "parent_prefix": list(parent),
                    "raw_score": float(row["p_viable"]),
                }
            )
            enriched.append(row)
        if method == "thinking_shuffled":
            for task in tasks:
                task_id = str(task["task_id"])
                by_parent: dict[tuple[str, ...], list[tuple[str, ...]]] = defaultdict(list)
                for prefix in grouped[task_id]:
                    by_parent[prefix[:-1]].append(prefix)
                for parent, prefixes in by_parent.items():
                    ordered = sorted(prefixes)
                    values = [scores_by_task[task_id][prefix] for prefix in ordered]
                    random.Random(f"{run_seed}:{task_id}:{layer}:{parent}").shuffle(values)
                    for prefix, value in zip(ordered, values):
                        scores_by_task[task_id][prefix] = value
            for row in enriched:
                prefix = tuple(row["candidate_prefix"])
                row["assigned_score"] = scores_by_task[row["task_id"]][prefix]
        else:
            for row in enriched:
                row["assigned_score"] = row["raw_score"]
        _append_rows(score_file, enriched)
        for task in tasks:
            task_id = str(task["task_id"])
            candidates = grouped[task_id]
            frontiers[task_id] = S.select_beam(candidates, scores_by_task[task_id], beam)
            layer_logs[task_id].append(
                {
                    "layer": layer,
                    "expanded": len(candidates),
                    "retained": len(frontiers[task_id]),
                }
            )
        print(
            f"[search:{method}] layer {layer}/{tasks[0]['depth']} scored={len(records)} "
            f"sampled={summary['accounting']['sampled_tokens']}",
            flush=True,
        )
    task_rows = []
    for task_index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        evaluated = S.evaluate_leaves(task, frontiers[task_id], fill_cap=fill_cap)
        task_rows.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "shard": task_index % 2,
                "method": method,
                "layers": layer_logs[task_id],
                "model_accounting": dict(model_work[task_id]),
                **evaluated,
            }
        )
    return {
        "schema_version": 1,
        "method": method,
        "beam_width": beam,
        "thinking_budget": thinking_budget if thinking else 0,
        "wall_seconds": time.perf_counter() - started,
        "score_artifact": score_file.name,
        "rows": task_rows,
    }


def _run_nextop_arm(
    scorer: M.ModelScorer,
    tasks: list[dict[str, Any]],
    *,
    beam: int,
    fill_cap: int,
    run_seed: int,
    suffix: str,
) -> dict[str, Any]:
    method = "nextop"
    frontiers = {str(task["task_id"]): [()] for task in tasks}
    cumulative = {str(task["task_id"]): {(): 0.0} for task in tasks}
    model_work = {str(task["task_id"]): Counter() for task in tasks}
    layer_logs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_file = _score_path(method, suffix)
    score_file.unlink(missing_ok=True)
    started = time.perf_counter()
    for layer in range(1, int(tasks[0]["depth"]) + 1):
        records = []
        parent_map = {}
        for task in tasks:
            task_id = str(task["task_id"])
            for index, parent in enumerate(frontiers[task_id]):
                record = C.calibration_record(
                    task,
                    parent,
                    record_id=f"nextop:{task_id}:l{layer}:p{index:02d}",
                    method=method,
                    layer=layer,
                    choices=list(F.TYPES),
                )
                records.append(record)
                parent_map[str(record["id"])] = (task_id, parent)
        scored, summary = scorer.score_next_operation_likelihood(records, run_seed=run_seed + layer)
        candidates_by_task: dict[str, list[tuple[str, ...]]] = defaultdict(list)
        scores_by_task: dict[str, dict[tuple[str, ...], float]] = defaultdict(dict)
        enriched = []
        for row in scored:
            task_id, parent = parent_map[str(row["id"])]
            _accumulate(model_work[task_id], row["accounting"])
            for operation, probability in row["choice_probabilities"].items():
                child = parent + (operation,)
                candidates_by_task[task_id].append(child)
                scores_by_task[task_id][child] = cumulative[task_id][parent] + math.log(
                    max(float(probability), 1e-12)
                )
            row.update({"task_id": task_id, "layer": layer, "parent_prefix": list(parent)})
            enriched.append(row)
        _append_rows(score_file, enriched)
        for task in tasks:
            task_id = str(task["task_id"])
            candidates = candidates_by_task[task_id]
            frontiers[task_id] = S.select_beam(candidates, scores_by_task[task_id], beam)
            cumulative[task_id] = {
                prefix: scores_by_task[task_id][prefix] for prefix in frontiers[task_id]
            }
            layer_logs[task_id].append(
                {
                    "layer": layer,
                    "expanded": len(candidates),
                    "retained": len(frontiers[task_id]),
                }
            )
        print(f"[search:nextop] layer {layer} parents={len(records)}", flush=True)
    rows = []
    for task_index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        rows.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "shard": task_index % 2,
                "method": method,
                "layers": layer_logs[task_id],
                "model_accounting": dict(model_work[task_id]),
                **S.evaluate_leaves(task, frontiers[task_id], fill_cap=fill_cap),
            }
        )
    return {
        "schema_version": 1,
        "method": method,
        "beam_width": beam,
        "wall_seconds": time.perf_counter() - started,
        "score_artifact": score_file.name,
        "rows": rows,
    }


def _direct_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    user = (
        C.DSL_TASK_TEXT
        + "\n\nVISIBLE EXAMPLES\n"
        + json.dumps(task["visible"], sort_keys=True, separators=(",", ":"))
        + f"\n\nInfer exactly {task['depth']} operation TYPES. Do not give parameter values. "
        + "Return only one JSON list on the final answer line, for example "
        + '["reverse","add_k","take_k","sort_asc","negate"].'
    )
    return [
        {
            "role": "system",
            "content": "Infer operation-type pipelines from visible examples. Follow the exact JSON format.",
        },
        {"role": "user", "content": user},
    ]


def _parse_skeleton(text: str, depth: int) -> tuple[str, ...] | None:
    answer = text.split("</think>")[-1]
    chunks = re.findall(r"\[[^\[\]]*\]", answer, flags=re.S)
    for chunk in reversed(chunks):
        try:
            value = json.loads(chunk)
        except json.JSONDecodeError:
            try:
                value = ast.literal_eval(chunk)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(value, list) or len(value) != depth:
            continue
        skeleton = tuple(str(item).split("(", 1)[0].strip() for item in value)
        if all(name in F.TYPES for name in skeleton):
            return skeleton
    return None


def _direct_cost(output: dict[str, Any], basis: str) -> int:
    sampled = int(output["n_sampled_tokens"])
    if basis == "sampled_tokens":
        return sampled
    if basis == "total_model_tokens":
        return (
            int(output["n_stage1_prompt_tokens"])
            + int(output["n_stage2_prompt_tokens"])
            + sampled
        )
    raise ValueError(f"unknown direct matching basis: {basis}")


def _prefix_under_cap(
    outputs: list[dict[str, Any]], cap: int, basis: str
) -> tuple[list[dict[str, Any]], int, int | None, bool]:
    used: list[dict[str, Any]] = []
    spent = 0
    next_cost = None
    ordered = sorted(outputs, key=lambda row: int(row["sample_index"]))
    for output in ordered:
        cost = _direct_cost(output, basis)
        if spent + cost > cap:
            next_cost = cost
            break
        used.append(output)
        spent += cost
    exhausted = len(used) == len(ordered) and spent < cap
    return used, spent, next_cost, exhausted


def _direct_accounting(outputs: list[dict[str, Any]]) -> dict[str, int]:
    sampled = sum(int(row["n_sampled_tokens"]) for row in outputs)
    prefill = sum(
        int(row["n_stage1_prompt_tokens"]) + int(row["n_stage2_prompt_tokens"])
        for row in outputs
    )
    request_count = sum(
        1 + int(int(row["n_stage2_prompt_tokens"]) > 0) for row in outputs
    )
    return {
        "requests": request_count,
        "completions": request_count,
        "prefill_tokens": prefill,
        "sampled_tokens": sampled,
        "total_model_tokens": prefill + sampled,
        "retained_thinking_tokens": sum(int(row["n_thinking_tokens"]) for row in outputs),
        "injected_prompt_tokens": sum(int(row["n_injected_tokens"]) for row in outputs),
    }


def _generate_direct_pool(
    runner: VLLMRunner,
    tasks: list[dict[str, Any]],
    *,
    pool_k: int,
    thinking_budget: int,
    answer_max: int,
    run_seed: int,
    suffix: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    sampling = SamplingConfig(
        thinking="budget",
        thinking_budget=thinking_budget,
        answer_max_tokens=answer_max,
        n=pool_k,
        run_seed=run_seed,
    )
    generated_by_task: dict[str, dict[str, Any]] = {}
    receipts: dict[str, Any] = {}
    shard_count = 2
    for shard in range(shard_count):
        shard_tasks = [task for index, task in enumerate(tasks) if index % shard_count == shard]
        records = [
            {
                "id": str(task["task_id"]),
                "messages": _direct_messages(task),
                "meta": {"task_id": task["task_id"], "shard": shard},
            }
            for task in shard_tasks
        ]
        rows, receipt = runner.generate(records, sampling)
        expected_ids = [str(task["task_id"]) for task in shard_tasks]
        if [str(row.get("id")) for row in rows] != expected_ids:
            raise RuntimeError(f"direct shard {shard} changed frozen task order")
        if any(len(row.get("outputs", [])) != pool_k for row in rows):
            raise RuntimeError(f"direct shard {shard} did not return exactly pool_k outputs per task")
        output_path = EXP / "runs" / f"direct_sample_pool_shard{shard}{suffix}.jsonl"
        _atomic_jsonl(output_path, rows)
        sidecar = {
            "schema_version": 1,
            "shard": shard,
            "task_ids": expected_ids,
            "output_sha256": _sha256(output_path),
            "runner_receipt": receipt,
        }
        _atomic_json(
            EXP / "runs" / f"direct_sample_pool_shard{shard}{suffix}.receipt.json",
            sidecar,
        )
        receipts[str(shard)] = sidecar
        generated_by_task.update({str(row["id"]): row for row in rows})
    if set(generated_by_task) != {str(task["task_id"]) for task in tasks}:
        raise RuntimeError("direct sample shards did not cover the frozen task set exactly")
    return generated_by_task, receipts


def _derive_direct_arm(
    tasks: list[dict[str, Any]],
    generated_by_task: dict[str, dict[str, Any]],
    thinking_result: dict[str, Any],
    *,
    method: str,
    basis: str,
    pool_k: int,
    thinking_budget: int,
    fill_cap: int,
) -> dict[str, Any]:
    think_by_task = {str(row["task_id"]): row for row in thinking_result["rows"]}
    result_rows = []
    for task_index, task in enumerate(tasks):
        task_id = str(task["task_id"])
        generated = generated_by_task[task_id]
        all_outputs = list(generated["outputs"])
        cap = int(think_by_task[task_id]["model_accounting"][basis])
        used_outputs, spent, next_cost, exhausted = _prefix_under_cap(all_outputs, cap, basis)
        parsed = []
        for output in used_outputs:
            skeleton = _parse_skeleton(output["text"], int(task["depth"]))
            if skeleton is not None:
                parsed.append(skeleton)
        parsed = list(dict.fromkeys(parsed))
        evaluated = S.evaluate_leaves(task, parsed, fill_cap=fill_cap)
        retained = _direct_accounting(used_outputs)
        gross = _direct_accounting(all_outputs)
        result_rows.append(
            {
                "task_id": task_id,
                "task_index": task_index,
                "shard": task_index % 2,
                "method": method,
                "match_basis": basis,
                "sample_pool_k": pool_k,
                "token_cap_from_thinking_search": cap,
                "matched_tokens": spent,
                "unspent_token_budget": cap - spent,
                "next_excluded_sample_cost": next_cost,
                "pool_exhausted_before_cap": exhausted,
                "pool_capacity_tokens": gross[basis],
                "matched_completions": len(used_outputs),
                "pool_completions": len(all_outputs),
                "retained_model_accounting": retained,
                "gross_pool_model_accounting": gross,
                "parse_rate": sum(
                    _parse_skeleton(output["text"], int(task["depth"])) is not None
                    for output in used_outputs
                )
                / max(1, len(used_outputs)),
                "unique_parsed_skeletons": len(parsed),
                **evaluated,
            }
        )
    return {
        "schema_version": 2,
        "method": method,
        "match_basis": basis,
        "thinking_budget": thinking_budget,
        "pool_k": pool_k,
        "matching_rule": (
            "per-task deterministic sample-index prefix under the corresponding "
            "thinking-search logical-token cap"
        ),
        "rows": result_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="discard this mode's search cache and recompute; never overrides a scientific gate",
    )
    args = parser.parse_args()
    suffix = "_smoke" if args.smoke else ""
    cfg = C.load_config()
    inputs = _fingerprints(suffix)
    if args.recompute:
        _clear_search_cache(suffix)
    elif _valid_complete_cache(suffix, inputs):
        print("[search] validated cached run", flush=True)
        return 0
    else:
        partial = [str(path) for path in _result_paths(suffix).values() if path.exists()]
        if partial:
            raise RuntimeError(
                "partial search artifacts exist without a valid completion receipt; "
                "inspect them, then use --recompute"
            )

    tasks, oracles = _validate_launch_inputs(cfg, suffix)
    verdict = json.loads(
        (EXP / "runs" / f"calibration_verdict{suffix}.json").read_text(encoding="utf-8")
    )
    if not verdict.get("passed") and not args.smoke:
        print("[search] recognition gate failed; primary search is not authorized", flush=True)
        return 1
    beam = 2 if args.smoke else int(cfg["search"]["beam_width"])
    think_budget = 32 if args.smoke else int(cfg["judge"]["thinking_budget"])
    fill_cap = 256 if args.smoke else int(cfg["search"]["parameter_fill_cap_per_task"])
    run_seed = int(cfg["judge"]["run_seed"]) + 100
    task_ids = [str(task["task_id"]) for task in tasks]

    def seal(result: dict[str, Any]) -> dict[str, Any]:
        result["input_sha256"] = inputs
        result["task_ids"] = task_ids
        result["settings"] = {
            "beam_width": beam,
            "thinking_budget": think_budget,
            "parameter_fill_cap_per_task": fill_cap,
            "run_seed": run_seed,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "backend": "vllm",
        }
        return result

    # Model-free controls first.
    surface = S.fit_surface_prior(C.load_jsonl(EXP / "data" / f"calibration_candidates{suffix}.jsonl"))
    for method in ("uniform_seeded", "surface", "oracle_live"):
        result = S.run_static_arm(
            tasks,
            oracles,
            method=method,
            beam=beam,
            fill_cap=fill_cap,
            surface_score=surface if method == "surface" else None,
        )
        _save_result(method, suffix, seal(result))
    budget_result = S.run_budget_truncated_brute(tasks, fill_cap=fill_cap)
    _save_result("budget_truncated_brute", suffix, seal(budget_result))

    engine_cfg = EngineConfig(
        max_model_len=int(cfg["judge"]["max_model_len"]),
        gpu_memory_utilization=float(cfg["judge"]["gpu_memory_utilization"]),
        max_num_seqs=int(cfg["judge"]["max_num_seqs"]),
        max_num_batched_tokens=8192,
    )
    direct_receipts: dict[str, Any] = {}
    with VLLMRunner(engine_cfg) as runner:
        scorer = M.ModelScorer(runner)
        model_results: dict[str, dict[str, Any]] = {}
        for offset, method in enumerate(("thinking", "thinking_shuffled", "nothink")):
            result = _run_viability_arm(
                scorer,
                tasks,
                method=method,
                beam=beam,
                thinking_budget=think_budget,
                fill_cap=fill_cap,
                run_seed=run_seed + offset * 20,
                suffix=suffix,
            )
            result = seal(result)
            _save_result(method, suffix, result)
            model_results[method] = result
        result = _run_nextop_arm(
            scorer,
            tasks,
            beam=beam,
            fill_cap=fill_cap,
            run_seed=run_seed + 60,
            suffix=suffix,
        )
        result = seal(result)
        _save_result("nextop", suffix, result)
        model_results["nextop"] = result
        pool_k = 256 if args.smoke else int(cfg["search"]["direct_sample_pool_k"])
        generated_by_task, direct_receipts = _generate_direct_pool(
            runner,
            tasks,
            pool_k=pool_k,
            thinking_budget=think_budget,
            answer_max=int(cfg["search"]["direct_sample_max_tokens"]),
            run_seed=run_seed + 80,
            suffix=suffix,
        )
        for method, basis in (
            ("direct_sample_more", "sampled_tokens"),
            ("direct_sample_more_total", "total_model_tokens"),
        ):
            direct = _derive_direct_arm(
                tasks,
                generated_by_task,
                model_results["thinking"],
                method=method,
                basis=basis,
                pool_k=pool_k,
                thinking_budget=think_budget,
                fill_cap=fill_cap,
            )
            _save_result(method, suffix, seal(direct))
        runtime = runner.runtime_metadata()
        receipt = {
            "schema_version": 2,
            "model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "backend": "vllm",
            "input_sha256": inputs,
            "task_ids": task_ids,
            "settings": seal({})["settings"],
            "engine_config": dataclasses.asdict(engine_cfg),
            "engine_args": runner.engine_args,
            "resolved_cudagraph": runner.resolved_cudagraph,
            "runtime": runtime,
            "direct_generation_receipts": direct_receipts,
        }
    receipt["result_sha256"] = {
        method: _sha256(path) for method, path in _result_paths(suffix).items()
    }
    receipt["auxiliary_sha256"] = {
        name: _sha256(path) for name, path in _auxiliary_paths(suffix).items()
    }
    _atomic_json(_receipt_path(suffix), receipt)
    print("[search] all arms complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
