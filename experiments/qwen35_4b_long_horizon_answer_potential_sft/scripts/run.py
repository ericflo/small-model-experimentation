#!/usr/bin/env python3
"""Restartable long-horizon answer-potential experiment orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from hf_scorer import HFAnswerPotentialScorer  # noqa: E402
from io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from model_ops import AnswerPotentialModel  # noqa: E402
from pivot import choose_pivot, natural_checkpoint_indices  # noqa: E402
from selector import (  # noqa: E402
    deranged_sources,
    oversample_to,
    select_task,
    sft_record,
)
from shards import read_jsonl_gz, valid_receipt, write_jsonl_gz  # noqa: E402
from stats import kendall_tau_b, mean, paired_bootstrap, roc_auc  # noqa: E402
from task_data import build_all  # noqa: E402
from vllm_runner import (  # noqa: E402
    EngineConfig,
    MODEL_ID,
    MODEL_REVISION,
    SamplingConfig,
    VLLMRunner,
)

CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("default config must be a mapping")
    if value["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("one-model invariant mismatch")
    return value


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    ancestor = (
        _run(["git", "merge-base", "--is-ancestor", commit, head], check=False).returncode
        == 0
    )
    paths = {
        "readme": "experiments/qwen35_4b_long_horizon_answer_potential_sft/README.md",
        "preregistration": "experiments/qwen35_4b_long_horizon_answer_potential_sft/reports/preregistration.md",
        "design_review": "experiments/qwen35_4b_long_horizon_answer_potential_sft/reports/design_review.md",
    }
    observed = {}
    for name, path in paths.items():
        payload = _run(["git", "show", f"{commit}:{path}"]).stdout.encode("utf-8")
        observed[name] = hashlib.sha256(payload).hexdigest()
    expected = {name: str(boundary[f"{name}_sha256"]) for name in paths}
    passed = ancestor and observed == expected
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "design_commit": commit,
        "current_head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "scientific_gpu_work_preceded_by_design_commit": True,
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def engine_config(
    config: dict[str, Any], *, model_override: Path | None = None
) -> EngineConfig:
    value = config["engine"]
    return EngineConfig(
        max_model_len=int(value["max_model_len"]),
        gpu_memory_utilization=float(value["gpu_memory_utilization"]),
        max_num_seqs=int(value["max_num_seqs"]),
        max_num_batched_tokens=int(value["max_num_batched_tokens"]),
        enable_prefix_caching=bool(value["prefix_caching"]),
        cudagraph_capture_sizes=tuple(
            int(item) for item in value["cudagraph_capture_sizes"]
        ),
        model_override=model_override,
    )


def external_root(config: dict[str, Any]) -> Path:
    return Path(str(config["artifacts"]["external_root"]))


def build_data() -> dict[str, Any]:
    manifest = build_all(DATA_DIR)
    print(f"[data] {manifest['audit']['split_counts']}", flush=True)
    return manifest


def load_split(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        build_data()
    return read_jsonl(path)


def _stage_index(path: Path, *, stage: str, split: str) -> dict[str, Any]:
    if path.is_file():
        value = read_json(path)
        if value.get("stage") != stage or value.get("split") != split:
            raise RuntimeError(f"stage-index identity mismatch: {path}")
        return value
    return {
        "schema_version": 1,
        "stage": stage,
        "split": split,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "shards": {},
    }


def _summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(row["n_tokens"]) for row in rows)
    prior_available = sum(row.get("prior_logprob_mean") is not None for row in rows)
    return {
        "rows": len(rows),
        "natural_close": sum(bool(row["natural_close"]) for row in rows),
        "loop": sum(bool(row["loop_flag"]) for row in rows),
        "continued": sum(bool(row.get("continued")) for row in rows),
        "prior_available": prior_available,
        "sampled_tokens": sum(int(row["n_sampled_tokens"]) for row in rows),
        "min_tokens": lengths[0] if lengths else None,
        "median_tokens": lengths[len(lengths) // 2] if lengths else None,
        "max_tokens": lengths[-1] if lengths else None,
    }


def generate_pool(
    config: dict[str, Any], *, split: str, n: int, stage: str
) -> dict[str, Any]:
    design_boundary_receipt(config)
    items = load_split(split)
    sampling = config["sampling"]
    root = external_root(config) / "pools" / stage
    shard_dir = root / "traces"
    index_path = root / "index.json"
    index = _stage_index(index_path, stage=stage, split=split)
    started = time.perf_counter()
    with AnswerPotentialModel(engine_config(config)) as model:
        for item_number, item in enumerate(items, 1):
            task_id = str(item["id"])
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            print(f"[{stage}] {item_number}/{len(items)} {task_id}", flush=True)
            rows, generation_meta = model.generate_thoughts(
                [item],
                n=n,
                max_tokens=int(sampling["natural_close_allowance"]),
                run_seed=int(sampling["run_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
                logprobs=int(sampling["logprobs"]),
                stage=stage,
                chunk_size=1,
            )
            rows, continuation_meta = model.continue_unclosed_thoughts(
                [item],
                rows,
                max_tokens=int(sampling["nonloop_continuation_tokens"]),
                run_seed=int(sampling["continuation_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            if any(
                row["n_tokens"] and row.get("prior_logprob_mean") is None
                for row in rows
            ):
                raise RuntimeError("sampled trace prior logprob missing")
            artifact = write_jsonl_gz(shard_dir / f"{task_id}.jsonl.gz", rows)
            index["shards"][task_id] = {
                "artifact": artifact,
                "summary": _summarize_traces(rows),
                "generation_elapsed_seconds": generation_meta["elapsed_seconds"],
                "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
            }
            index["logical_counts"] = continuation_meta["logical_counts"]
            index["runtime"] = generation_meta["runtime"]
            index["engine"] = generation_meta["engine"]
            index["resolved_cudagraph"] = generation_meta["resolved_cudagraph"]
            write_json(index_path, index)
    summaries = [entry["summary"] for entry in index["shards"].values()]
    total_rows = sum(int(value["rows"]) for value in summaries)
    summary = {
        "schema_version": 1,
        "stage": stage,
        "split": split,
        "tasks": len(index["shards"]),
        "rows": total_rows,
        "natural_close": sum(int(value["natural_close"]) for value in summaries),
        "loops": sum(int(value["loop"]) for value in summaries),
        "prior_available": sum(int(value["prior_available"]) for value in summaries),
        "sampled_tokens": sum(int(value["sampled_tokens"]) for value in summaries),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / f"{stage}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def ensure_minimum_natural_train_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen mechanical N=16 top-up rule only where needed."""
    root = external_root(config) / "pools" / "train_independent"
    index_path = root / "index.json"
    index = read_json(index_path)
    items = {str(item["id"]): item for item in load_split("train")}
    minimum = int(config["selector"]["minimum_natural_per_task"])

    def eligible_count(task_id: str) -> int:
        rows = read_jsonl_gz(Path(index["shards"][task_id]["artifact"]["path"]))
        return sum(row["natural_close"] and not row["loop_flag"] for row in rows)

    counts = {task_id: eligible_count(task_id) for task_id in sorted(index["shards"])}
    deficient = [task_id for task_id, count in counts.items() if count < minimum]
    topup_rows = 0
    if deficient:
        sampling = config["sampling"]
        with AnswerPotentialModel(engine_config(config)) as model:
            for batch in range(1, 5):
                active = [task_id for task_id in deficient if counts[task_id] < minimum]
                if not active:
                    break
                for task_id in active:
                    prior_rows = read_jsonl_gz(
                        Path(index["shards"][task_id]["artifact"]["path"])
                    )
                    new_rows, generation_meta = model.generate_thoughts(
                        [items[task_id]],
                        n=16,
                        max_tokens=int(sampling["natural_close_allowance"]),
                        run_seed=int(sampling["run_seed"]) + 10_000 * batch,
                        temperature=float(sampling["temperature"]),
                        top_p=float(sampling["top_p"]),
                        top_k=int(sampling["top_k"]),
                        logprobs=int(sampling["logprobs"]),
                        stage=f"train_topup_b{batch}",
                        chunk_size=1,
                    )
                    new_rows, continuation_meta = model.continue_unclosed_thoughts(
                        [items[task_id]],
                        new_rows,
                        max_tokens=int(sampling["nonloop_continuation_tokens"]),
                        run_seed=int(sampling["continuation_seed"]) + 10_000 * batch,
                        temperature=float(sampling["temperature"]),
                        top_p=float(sampling["top_p"]),
                        top_k=int(sampling["top_k"]),
                    )
                    combined = [*prior_rows, *new_rows]
                    artifact = write_jsonl_gz(
                        root / "traces" / f"{task_id}.jsonl.gz", combined
                    )
                    index["shards"][task_id] = {
                        "artifact": artifact,
                        "summary": _summarize_traces(combined),
                        "topup_batches": batch,
                        "generation_elapsed_seconds": generation_meta["elapsed_seconds"],
                        "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
                    }
                    counts[task_id] = sum(
                        row["natural_close"] and not row["loop_flag"] for row in combined
                    )
                    topup_rows += len(new_rows)
                    index["logical_counts"] = continuation_meta["logical_counts"]
                    write_json(index_path, index)
    remaining = sorted(task_id for task_id, count in counts.items() if count < minimum)
    result = {
        "schema_version": 1,
        "minimum": minimum,
        "initially_deficient": deficient,
        "topup_rows": topup_rows,
        "remaining_deficient": remaining,
        "counts": counts,
    }
    write_json(RUNS_DIR / "train_natural_minimum.json", result)
    print(
        f"[topup] initially_deficient={len(deficient)} rows={topup_rows} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    return result


def analyze_termination_pilot(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize termination mechanics without reading correctness."""
    index = read_json(
        external_root(config) / "pools" / "termination_pilot" / "index.json"
    )
    rows = [
        row
        for task_id in sorted(index["shards"])
        for row in read_jsonl_gz(
            Path(index["shards"][task_id]["artifact"]["path"])
        )
    ]
    lengths = sorted(int(row["n_tokens"]) for row in rows)
    result = {
        "schema_version": 1,
        "tasks": len(index["shards"]),
        "traces": len(rows),
        "natural_close": sum(bool(row["natural_close"]) for row in rows),
        "natural_close_rate": mean([bool(row["natural_close"]) for row in rows]),
        "exact_periodic_loops": sum(bool(row["loop_flag"]) for row in rows),
        "initial_allowance_contacts": sum(bool(row.get("continued")) for row in rows),
        "still_unclosed_after_continuation": sum(
            bool(row.get("continued")) and not bool(row["natural_close"]) for row in rows
        ),
        "greater_than_512_tokens": sum(int(row["n_tokens"]) > 512 for row in rows),
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "max_tokens": lengths[-1],
        "sampled_tokens": sum(int(row["n_sampled_tokens"]) for row in rows),
        "correctness_inspected": False,
    }
    write_json(RUNS_DIR / "termination_pilot_analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def score_pool(
    config: dict[str, Any], *, source_stage: str, output_stage: str, split: str
) -> dict[str, Any]:
    design_boundary_receipt(config)
    parity_path = RUNS_DIR / "scorer_parity_32.json"
    if not parity_path.is_file() or not bool(read_json(parity_path).get("passed")):
        raise RuntimeError("bulk HF scoring requires the passed 32-row parity gate")
    items = {str(item["id"]): item for item in load_split(split)}
    source_root = external_root(config) / "pools" / source_stage
    source_index = read_json(source_root / "index.json")
    output_root = external_root(config) / "pools" / output_stage
    output_index_path = output_root / "index.json"
    output_index = _stage_index(
        output_index_path, stage=output_stage, split=split
    )
    started = time.perf_counter()
    scorer = HFAnswerPotentialScorer()
    try:
        for task_number, task_id in enumerate(sorted(source_index["shards"]), 1):
            previous = output_index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            rows = read_jsonl_gz(
                Path(source_index["shards"][task_id]["artifact"]["path"])
            )
            eligible = [
                row
                for row in rows
                if row["natural_close"] and not row["loop_flag"]
            ]
            print(
                f"[{output_stage}] {task_number}/{len(source_index['shards'])} "
                f"{task_id}: {len(eligible)} eligible",
                flush=True,
            )
            scored = [scorer.score_trace(items[task_id], row) for row in eligible]
            artifact = write_jsonl_gz(
                output_root / "scores" / f"{task_id}.jsonl.gz", scored
            )
            output_index["shards"][task_id] = {
                "artifact": artifact,
                "source_artifact_sha256": source_index["shards"][task_id][
                    "artifact"
                ]["sha256"],
                "eligible": len(eligible),
            }
            write_json(output_index_path, output_index)
    finally:
        scorer.close()
    summary = {
        "schema_version": 1,
        "stage": output_stage,
        "split": split,
        "tasks": len(output_index["shards"]),
        "rows": sum(int(entry["artifact"]["rows"]) for entry in output_index["shards"].values()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(output_index_path),
    }
    write_json(RUNS_DIR / f"{output_stage}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def analyze_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Report selector informativeness without gating the SFT matrix."""
    raw_index = read_json(
        external_root(config) / "pools" / "calibration_independent" / "index.json"
    )
    score_index = read_json(
        external_root(config) / "pools" / "calibration_scores" / "index.json"
    )
    rollout_index = read_json(
        external_root(config) / "pools" / "calibration_rollouts_r4" / "index.json"
    )
    items = {str(item["id"]): item for item in load_split("calibration")}
    per_task = []
    top_ks = (1, 2, 4, 8)
    metric_names = {
        "answer_gain": "answer_gain_per_answer_token",
        "joint_gain": "joint_gain_per_answer_token",
        "negative_length": "negative_length",
        "trace_prior": "prior_logprob_mean",
    }
    top_curves: dict[str, dict[int, list[float]]] = {
        name: {k: [] for k in top_ks} for name in (*metric_names, "seeded_random")
    }
    for task_id in sorted(items):
        raw = read_jsonl_gz(Path(raw_index["shards"][task_id]["artifact"]["path"]))
        scores = read_jsonl_gz(Path(score_index["shards"][task_id]["artifact"]["path"]))
        rollouts = read_jsonl_gz(Path(rollout_index["shards"][task_id]["artifact"]["path"]))
        raw_by_id = {str(row["trace_id"]): row for row in raw}
        rollout_by_id = {str(row["trace_id"]): row for row in rollouts}
        joined = []
        for score in scores:
            trace_id = str(score["trace_id"])
            if trace_id not in rollout_by_id:
                continue
            raw_row = raw_by_id[trace_id]
            rollout = rollout_by_id[trace_id]
            joined.append(
                {
                    **score,
                    "negative_length": -float(raw_row["n_tokens"]),
                    "success_fraction": float(rollout["success_fraction"]),
                    "any_success": bool(rollout["any_success"]),
                }
            )
        labels = [row["any_success"] for row in joined]
        aucs = {}
        for name, field in metric_names.items():
            finite = [
                row for row in joined if row.get(field) is not None and math.isfinite(float(row[field]))
            ]
            aucs[name] = roc_auc(
                [row["any_success"] for row in finite],
                [float(row[field]) for row in finite],
            )
            ranked = sorted(
                finite,
                key=lambda row: (-float(row[field]), str(row["trace_id"])),
            )
            for k in top_ks:
                if ranked:
                    top_curves[name][k].append(
                        mean([row["success_fraction"] for row in ranked[:k]])
                    )
        seeded = list(joined)
        random_seed = int.from_bytes(
            hashlib.blake2b(task_id.encode("utf-8"), digest_size=8).digest(), "big"
        ) + int(config["sampling"]["control_seed"])
        import random

        random.Random(random_seed).shuffle(seeded)
        for k in top_ks:
            if seeded:
                top_curves["seeded_random"][k].append(
                    mean([row["success_fraction"] for row in seeded[:k]])
                )
        tau = kendall_tau_b(
            [float(row["answer_gain_per_answer_token"]) for row in joined],
            [float(row["joint_gain_per_answer_token"]) for row in joined],
        )
        per_task.append(
            {
                "task_id": task_id,
                "family": items[task_id]["family"],
                "level": items[task_id]["level"],
                "raw_traces": len(raw),
                "eligible_scored": len(scores),
                "rollout_traces": len(joined),
                "success_rate": mean([row["success_fraction"] for row in joined]),
                "auroc": aucs,
                "answer_joint_kendall_tau_b": tau,
            }
        )
    result = {
        "schema_version": 1,
        "tasks": len(per_task),
        "raw_candidates": sum(row["raw_traces"] for row in per_task),
        "eligible_candidates": sum(row["eligible_scored"] for row in per_task),
        "task_macro_auroc": {
            name: mean(
                [row["auroc"][name] for row in per_task if row["auroc"][name] is not None]
            )
            for name in metric_names
        },
        "top_k_mean_rollout_success": {
            name: {str(k): mean(values) for k, values in curves.items()}
            for name, curves in top_curves.items()
        },
        "task_macro_answer_joint_kendall_tau_b": mean(
            [
                row["answer_joint_kendall_tau_b"]
                for row in per_task
                if row["answer_joint_kendall_tau_b"] is not None
            ]
        ),
        "family": {
            family: {
                "tasks": sum(row["family"] == family for row in per_task),
                "mean_success": mean(
                    [row["success_rate"] for row in per_task if row["family"] == family]
                ),
                "answer_auroc": mean(
                    [
                        row["auroc"]["answer_gain"]
                        for row in per_task
                        if row["family"] == family and row["auroc"]["answer_gain"] is not None
                    ]
                ),
            }
            for family in sorted({row["family"] for row in per_task})
        },
        "per_task": per_task,
        "effectiveness_gate_used": False,
    }
    write_json(RUNS_DIR / "calibration_analysis.json", result)
    print(json.dumps({key: result[key] for key in ("raw_candidates", "eligible_candidates", "task_macro_auroc", "top_k_mean_rollout_success")}, indent=2), flush=True)
    return result


def run_scorer_parity(
    config: dict[str, Any], *, source_stage: str, split: str
) -> dict[str, Any]:
    """Run the preregistered 32-row exact vLLM/HF likelihood gate."""
    design_boundary_receipt(config)
    source_index = read_json(
        external_root(config) / "pools" / source_stage / "index.json"
    )
    items = {str(item["id"]): item for item in load_split(split)}
    traces: list[dict[str, Any]] = []
    for task_id in sorted(source_index["shards"]):
        rows = read_jsonl_gz(
            Path(source_index["shards"][task_id]["artifact"]["path"])
        )
        traces.extend(
            row for row in rows if row["natural_close"] and not row["loop_flag"]
        )
        if len(traces) >= int(config["scoring"]["vllm_parity_rows"]):
            break
    count = int(config["scoring"]["vllm_parity_rows"])
    traces = traces[:count]
    if len(traces) != count:
        raise RuntimeError(f"parity requires {count} eligible rows; found {len(traces)}")
    parity_items = [items[str(trace["task_id"])] for trace in traces]
    unique_items = {str(item["id"]): item for item in parity_items}
    with AnswerPotentialModel(engine_config(config)) as model:
        vllm_rows, vllm_meta = model.score_answer_potential(
            list(unique_items.values()), traces, include_decoy=False
        )
    scorer = HFAnswerPotentialScorer()
    try:
        hf_rows = [scorer.score_trace(items[str(row["task_id"])], row) for row in traces]
    finally:
        scorer.close()
    vllm_by_id = {str(row["trace_id"]): row for row in vllm_rows}
    comparisons = []
    for row in hf_rows:
        vllm_row = vllm_by_id[str(row["trace_id"])]
        delta = (
            float(row["answer_ll_sum"]) - float(vllm_row["canonical_ll_sum"])
        ) / len(row["answer_token_ids"])
        comparisons.append(
            {
                "trace_id": row["trace_id"],
                "task_id": row["task_id"],
                "hf_answer_ll_sum": row["answer_ll_sum"],
                "vllm_answer_ll_sum": vllm_row["canonical_ll_sum"],
                "signed_mean_token_delta": delta,
                "abs_mean_token_delta": abs(delta),
            }
        )
    maximum = max(row["abs_mean_token_delta"] for row in comparisons)
    threshold = float(config["scoring"]["max_abs_mean_token_delta"])
    result = {
        "schema_version": 1,
        "source_stage": source_stage,
        "rows": len(comparisons),
        "max_abs_mean_token_delta": maximum,
        "threshold": threshold,
        "passed": maximum <= threshold,
        "vllm_meta": vllm_meta,
        "comparisons": comparisons,
    }
    write_json(RUNS_DIR / "scorer_parity_32.json", result)
    if not result["passed"]:
        raise RuntimeError(f"32-row scorer parity failed: {maximum} > {threshold}")
    print(f"[parity] rows={count} max_delta={maximum:.6f} passed", flush=True)
    return result


def plan_pivots(config: dict[str, Any]) -> dict[str, Any]:
    """Score natural root checkpoints and freeze one pivot per train task."""
    design_boundary_receipt(config)
    raw_index = read_json(
        external_root(config) / "pools" / "train_independent" / "index.json"
    )
    score_index = read_json(
        external_root(config) / "pools" / "train_independent_scores" / "index.json"
    )
    items = {str(item["id"]): item for item in load_split("train")}
    output_root = external_root(config) / "pools" / "train_pivots"
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage="train_pivots", split="train")
    branch = config["branch"]
    started = time.perf_counter()
    scorer = HFAnswerPotentialScorer()
    try:
        for task_number, task_id in enumerate(sorted(raw_index["shards"]), 1):
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            traces = read_jsonl_gz(
                Path(raw_index["shards"][task_id]["artifact"]["path"])
            )
            scores = read_jsonl_gz(
                Path(score_index["shards"][task_id]["artifact"]["path"])
            )
            if not scores:
                raise RuntimeError(f"no eligible independent root for {task_id}")
            root_score = max(
                scores,
                key=lambda row: (
                    float(row["joint_gain_per_answer_token"]),
                    str(row["trace_id"]),
                ),
            )
            trace_by_id = {str(row["trace_id"]): row for row in traces}
            root = trace_by_id[str(root_score["trace_id"])]
            indices = natural_checkpoint_indices(
                scorer.tokenizer,
                root["token_ids"],
                max_checkpoints=int(branch["max_checkpoints"]),
            )
            checkpoints = []
            for token_index in indices:
                if token_index == int(root["n_tokens"]):
                    checkpoint_score = root_score
                else:
                    partial = {
                        **root,
                        "trace_id": f"{root['trace_id']}::checkpoint::{token_index}",
                        "token_ids": root["token_ids"][:token_index],
                        "n_tokens": token_index,
                    }
                    checkpoint_score = scorer.score_trace(items[task_id], partial)
                checkpoints.append(
                    {
                        "token_index": token_index,
                        "answer_gain_per_answer_token": checkpoint_score[
                            "answer_gain_per_answer_token"
                        ],
                        "joint_gain_per_answer_token": checkpoint_score[
                            "joint_gain_per_answer_token"
                        ],
                    }
                )
            decision = choose_pivot(
                checkpoints,
                minimum_positive_jump=float(
                    branch["minimum_positive_jump_per_answer_token"]
                ),
                fallback_fraction=float(branch["fallback_fraction"]),
                full_length=int(root["n_tokens"]),
            )
            pivot_index = int(decision["pivot_token_index"])
            plan = {
                "schema_version": 1,
                "task_id": task_id,
                "family": root["family"],
                "level": root["level"],
                "root_trace_id": root["trace_id"],
                "root_tokens": root["n_tokens"],
                "root_joint_gain_per_answer_token": root_score[
                    "joint_gain_per_answer_token"
                ],
                "prefix_token_ids": root["token_ids"][:pivot_index],
                "checkpoints": checkpoints,
                **decision,
            }
            artifact = write_jsonl_gz(
                output_root / "plans" / f"{task_id}.jsonl.gz", [plan]
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "root_source_sha256": raw_index["shards"][task_id]["artifact"][
                    "sha256"
                ],
                "score_source_sha256": score_index["shards"][task_id]["artifact"][
                    "sha256"
                ],
                "pivot_token_index": pivot_index,
            }
            write_json(index_path, index)
            print(
                f"[train_pivots] {task_number}/{len(raw_index['shards'])} "
                f"{task_id} root={root['n_tokens']} pivot={pivot_index}",
                flush=True,
            )
    finally:
        scorer.close()
    summary = {
        "schema_version": 1,
        "stage": "train_pivots",
        "tasks": len(index["shards"]),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / "train_pivots_summary.json", summary)
    return summary


def generate_branch_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Generate the registered sixteen pivot suffixes for every train task."""
    design_boundary_receipt(config)
    items = load_split("train")
    item_by_id = {str(item["id"]): item for item in items}
    plan_index = read_json(
        external_root(config) / "pools" / "train_pivots" / "index.json"
    )
    output_root = external_root(config) / "pools" / "train_branches"
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage="train_branches", split="train")
    sampling = config["sampling"]
    started = time.perf_counter()
    with AnswerPotentialModel(engine_config(config)) as model:
        for task_number, task_id in enumerate(sorted(plan_index["shards"]), 1):
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            plan = read_jsonl_gz(
                Path(plan_index["shards"][task_id]["artifact"]["path"])
            )[0]
            rows, branch_meta = model.generate_pivot_branches(
                [item_by_id[task_id]],
                [plan],
                n=int(sampling["train_branch_n"]),
                total_allowance=int(sampling["natural_close_allowance"]),
                run_seed=int(sampling["branch_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            rows, continuation_meta = model.continue_unclosed_thoughts(
                [item_by_id[task_id]],
                rows,
                max_tokens=int(sampling["nonloop_continuation_tokens"]),
                run_seed=int(sampling["continuation_seed"]) + 1,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            artifact = write_jsonl_gz(
                output_root / "traces" / f"{task_id}.jsonl.gz", rows
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "summary": _summarize_traces(rows),
                "plan_sha256": plan_index["shards"][task_id]["artifact"]["sha256"],
                "branch_elapsed_seconds": branch_meta["elapsed_seconds"],
                "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
            }
            index["logical_counts"] = continuation_meta["logical_counts"]
            index["runtime"] = branch_meta["runtime"]
            index["engine"] = branch_meta["engine"]
            write_json(index_path, index)
            print(
                f"[train_branches] {task_number}/{len(plan_index['shards'])} {task_id}",
                flush=True,
            )
    summaries = [row["summary"] for row in index["shards"].values()]
    summary = {
        "schema_version": 1,
        "stage": "train_branches",
        "tasks": len(index["shards"]),
        "rows": sum(int(row["rows"]) for row in summaries),
        "natural_close": sum(int(row["natural_close"]) for row in summaries),
        "loops": sum(int(row["loop"]) for row in summaries),
        "sampled_tokens": sum(int(row["sampled_tokens"]) for row in summaries),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / "train_branches_summary.json", summary)
    return summary


def rollout_pool(
    config: dict[str, Any],
    *,
    split: str,
    source_stages: list[str],
    output_stage: str,
    r: int,
) -> dict[str, Any]:
    """Generate restartable per-task answer rollouts over one or more pools."""
    design_boundary_receipt(config)
    items = {str(item["id"]): item for item in load_split(split)}
    source_indices = [
        read_json(external_root(config) / "pools" / stage / "index.json")
        for stage in source_stages
    ]
    task_ids = sorted(set.intersection(*(set(index["shards"]) for index in source_indices)))
    output_root = external_root(config) / "pools" / output_stage
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage=output_stage, split=split)
    sampling = config["sampling"]
    started = time.perf_counter()
    with AnswerPotentialModel(engine_config(config)) as model:
        for task_number, task_id in enumerate(task_ids, 1):
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            traces: list[dict[str, Any]] = []
            source_sha256 = []
            for source_index in source_indices:
                entry = source_index["shards"][task_id]
                source_sha256.append(entry["artifact"]["sha256"])
                traces.extend(read_jsonl_gz(Path(entry["artifact"]["path"])))
            traces = [
                row for row in traces if row["natural_close"] and not row["loop_flag"]
            ]
            rows, metadata = model.generate_answer_rollouts(
                [items[task_id]],
                traces,
                r=r,
                max_tokens=int(sampling["answer_max_tokens"]),
                run_seed=int(sampling["continuation_seed"]) + 101 * r,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            artifact = write_jsonl_gz(
                output_root / "rollouts" / f"{task_id}.jsonl.gz", rows
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "source_sha256": source_sha256,
                "eligible": len(traces),
                "elapsed_seconds": metadata["elapsed_seconds"],
            }
            index["logical_counts"] = metadata["logical_counts"]
            index["runtime"] = metadata["runtime"]
            write_json(index_path, index)
            print(
                f"[{output_stage}] {task_number}/{len(task_ids)} {task_id}: {len(traces)} traces",
                flush=True,
            )
    summary = {
        "schema_version": 1,
        "stage": output_stage,
        "tasks": len(index["shards"]),
        "rows": sum(int(row["artifact"]["rows"]) for row in index["shards"].values()),
        "rollouts": sum(int(row["artifact"]["rows"]) * r for row in index["shards"].values()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / f"{output_stage}_summary.json", summary)
    return summary


def build_sft_datasets(config: dict[str, Any]) -> dict[str, Any]:
    """Join candidate evidence and materialize all six frozen SFT arms."""
    from transformers import AutoTokenizer

    design_boundary_receipt(config)
    root = external_root(config)
    source_names = {
        "independent": "train_independent",
        "branch": "train_branches",
        "independent_scores": "train_independent_scores",
        "branch_scores": "train_branch_scores",
        "rollouts": "train_rollouts_r1",
    }
    indices = {
        key: read_json(root / "pools" / value / "index.json")
        for key, value in source_names.items()
    }
    items = {str(item["id"]): item for item in load_split("train")}
    selector_config = config["selector"]
    seed = int(config["sampling"]["control_seed"])
    selections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    details_root = root / "selection" / "tasks"
    for task_number, task_id in enumerate(sorted(items), 1):
        traces: list[dict[str, Any]] = []
        scores: list[dict[str, Any]] = []
        for key in ("independent", "branch"):
            traces.extend(
                read_jsonl_gz(Path(indices[key]["shards"][task_id]["artifact"]["path"]))
            )
        for key in ("independent_scores", "branch_scores"):
            scores.extend(
                read_jsonl_gz(Path(indices[key]["shards"][task_id]["artifact"]["path"]))
            )
        rollouts = read_jsonl_gz(
            Path(indices["rollouts"]["shards"][task_id]["artifact"]["path"])
        )
        selected = select_task(
            traces,
            scores,
            rollouts,
            selector_config=selector_config,
            seed=seed,
        )
        selections[task_id] = selected
        write_jsonl_gz(
            details_root / f"{task_id}.jsonl.gz",
            [
                {
                    "task_id": task_id,
                    "arm": arm,
                    "trace_ids": [row["trace_id"] for row in rows],
                    "trace_tokens": [row["n_tokens"] for row in rows],
                    "source_kinds": [row.get("source_kind") for row in rows],
                }
                for arm, rows in selected.items()
                if arm != "eligible"
            ],
        )
        if task_number % 50 == 0:
            print(f"[selection] {task_number}/{len(items)}", flush=True)

    required = int(selector_config["minimum_natural_per_task"])
    core_tasks = [
        task_id
        for task_id, selected in selections.items()
        if all(
            len(selected[arm]) >= required
            for arm in ("answer_potential", "joint_potential", "random_natural")
        )
    ]
    deficient = sorted(set(items) - set(core_tasks))
    # The preregistered terminal top-up rule excludes any still-deficient task
    # symmetrically; it never weakens the two-natural-trace requirement.

    chosen: dict[str, list[dict[str, Any] | None]] = {
        "answer_potential": [],
        "joint_potential": [],
        "random_natural": [],
        "success_rft": [],
        "empty": [],
    }
    row_tasks: dict[str, list[str]] = {key: [] for key in chosen}
    for task_id in core_tasks:
        selected = selections[task_id]
        for arm in ("answer_potential", "joint_potential", "random_natural"):
            rows = selected[arm][:required]
            chosen[arm].extend(rows)
            row_tasks[arm].extend([task_id] * len(rows))
        success = selected["success_rft"][:required]
        chosen["success_rft"].extend(success)
        row_tasks["success_rft"].extend([task_id] * len(success))
        chosen["empty"].extend([None] * required)
        row_tasks["empty"].extend([task_id] * required)

    answer_rows = [row for row in chosen["answer_potential"] if row is not None]
    shuffle_rows = deranged_sources(answer_rows)
    chosen["potential_shuffle"] = shuffle_rows
    row_tasks["potential_shuffle"] = [str(row["task_id"]) for row in shuffle_rows]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    max_length = int(config["sft"]["max_length"])
    encoded: dict[str, list[dict[str, Any]]] = {}
    for arm in config["sft"]["arms"]:
        rows = []
        for ordinal, (task_id, trace) in enumerate(
            zip(row_tasks[arm], chosen[arm]), 1
        ):
            rows.append(
                sft_record(
                    arm=arm,
                    item=items[task_id],
                    trace=trace,
                    tokenizer=tokenizer,
                    ordinal=ordinal,
                    max_length=max_length,
                )
            )
        encoded[arm] = rows
    target_rows = len(encoded["answer_potential"])
    success_unique_rows = len(encoded["success_rft"])
    success_unique_tasks = len({row["task_id"] for row in encoded["success_rft"]})
    encoded["success_rft"] = oversample_to(
        encoded["success_rft"], target_rows, seed=seed
    )
    for arm, rows in encoded.items():
        if len(rows) != target_rows:
            raise RuntimeError(f"row-matching failure for {arm}: {len(rows)} != {target_rows}")

    dataset_root = root / "sft"
    receipts = {
        arm: write_jsonl_gz(dataset_root / f"{arm}.jsonl.gz", rows)
        for arm, rows in encoded.items()
    }
    arm_summary = {}
    for arm, rows in encoded.items():
        arm_summary[arm] = {
            "rows": len(rows),
            "unique_record_ids": len({row["record_id"].split("::repeat", 1)[0] for row in rows}),
            "unique_tasks": len({row["task_id"] for row in rows}),
            "trace_tokens": sum(int(row["trace_tokens"]) for row in rows),
            "forward_tokens": sum(int(row["total_tokens"]) for row in rows),
            "branch_rows": sum(row.get("source_kind") == "pivot_branch" for row in rows),
            "artifact": receipts[arm],
        }
    summary = {
        "schema_version": 1,
        "core_tasks": len(core_tasks),
        "deficient_tasks": deficient,
        "target_rows_per_arm": target_rows,
        "success_unique_rows_before_oversampling": success_unique_rows,
        "success_unique_tasks": success_unique_tasks,
        "arms": arm_summary,
    }
    write_json(EXP / "data" / "sft_manifest.json", summary)
    write_json(RUNS_DIR / "selection_summary.json", summary)
    print(
        f"[selection] core_tasks={len(core_tasks)} rows/arm={target_rows} "
        f"success_tasks={success_unique_tasks}",
        flush=True,
    )
    return summary


def _score_evaluation_rows(
    items: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from collections import Counter

    from gym import base
    from gym.families import load as load_train_family
    from gym.heldout_families import load as load_held_family

    scored_rows = []
    all_thought_lengths: list[int] = []
    all_completion_lengths: list[int] = []
    for row in rows:
        task_id = str(row["id"])
        item = items[task_id]
        module = (
            load_held_family(str(item["family"]))
            if item["family"] in {"brinework", "spindle"}
            else load_train_family(str(item["family"]))
        )
        outcomes = []
        for output in row["outputs"]:
            answer_value = base.extract_answer(str(output["text"]))
            score = float(module.score_atom(item, str(output["text"])))
            outcomes.append(
                {
                    **output,
                    "answer_value": answer_value,
                    "score": score,
                    "correct": score == 1.0,
                    "parsed": answer_value is not None,
                }
            )
            all_thought_lengths.append(int(output["n_thinking_tokens"]))
            all_completion_lengths.append(int(output["n_sampled_tokens"]))
        parsed = [outcome["answer_value"] for outcome in outcomes if outcome["answer_value"] is not None]
        majority_value = Counter(parsed).most_common(1)[0][0] if parsed else None
        majority_correct = any(
            outcome["answer_value"] == majority_value and outcome["correct"]
            for outcome in outcomes
        ) if majority_value is not None else False
        scored_rows.append(
            {
                **{key: value for key, value in row.items() if key != "outputs"},
                "family": item["family"],
                "level": item["level"],
                "outputs": outcomes,
                "any_correct": any(outcome["correct"] for outcome in outcomes),
                "majority_answer": majority_value,
                "majority_correct": majority_correct,
                "unique_answers": len(set(parsed)),
            }
        )
    tasks = len(scored_rows)
    outputs = [outcome for row in scored_rows for outcome in row["outputs"]]
    lengths = sorted(all_thought_lengths)
    summary = {
        "tasks": tasks,
        "samples": len(outputs),
        "sample_accuracy": sum(outcome["correct"] for outcome in outputs) / len(outputs),
        "parse_rate": sum(outcome["parsed"] for outcome in outputs) / len(outputs),
        "natural_close_rate": sum(outcome["thinking_closed"] for outcome in outputs) / len(outputs),
        "pass_at_n": sum(row["any_correct"] for row in scored_rows) / tasks,
        "majority_at_n": sum(row["majority_correct"] for row in scored_rows) / tasks,
        "mean_unique_answers": sum(row["unique_answers"] for row in scored_rows) / tasks,
        "mean_thinking_tokens": sum(all_thought_lengths) / len(all_thought_lengths),
        "median_thinking_tokens": lengths[len(lengths) // 2],
        "p95_thinking_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "sampled_tokens": sum(all_completion_lengths),
        "logical_prompt_tokens": sum(int(row["n_prompt_tokens"]) * len(row["outputs"]) for row in scored_rows),
    }
    return scored_rows, summary


def evaluate_matrix(config: dict[str, Any], *, mode: str) -> dict[str, Any]:
    """Evaluate base and every seed-42 merged arm on fresh local splits."""
    if mode not in {"greedy", "sample8"}:
        raise ValueError("mode must be greedy or sample8")
    design_boundary_receipt(config)
    artifact_root = external_root(config)
    arms = ["base", *config["sft"]["arms"]]
    splits = ("iid_eval", "hard_eval", "held_family_eval", "rendering_eval")
    eval_root = artifact_root / "evaluation" / "seed42" / mode
    index_path = eval_root / "index.json"
    index = read_json(index_path) if index_path.is_file() else {
        "schema_version": 1,
        "mode": mode,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "arms": {},
    }
    evaluation = config["evaluation"]
    for arm in arms:
        pending = [
            split
            for split in splits
            if not (
                split in index["arms"].get(arm, {})
                and valid_receipt(index["arms"][arm][split]["artifact"])
            )
        ]
        if not pending:
            continue
        model_override = (
            None
            if arm == "base"
            else artifact_root / "merged" / "seed42" / arm
        )
        if model_override is not None and not (model_override / "config.json").is_file():
            raise RuntimeError(f"missing merged checkpoint for {arm}: {model_override}")
        print(f"[eval:{mode}] loading {arm}", flush=True)
        with VLLMRunner(
            engine_config(config, model_override=model_override)
        ) as runner:
            for split in pending:
                split_items = load_split(split)
                item_by_id = {str(item["id"]): item for item in split_items}
                records = [
                    {
                        "id": item["id"],
                        "messages": [{"role": "user", "content": item["prompt"]}],
                        "meta": {
                            "family": item["family"],
                            "level": item["level"],
                            "split": split,
                        },
                    }
                    for item in split_items
                ]
                sampling = SamplingConfig(
                    thinking="natural",
                    n=1 if mode == "greedy" else int(evaluation["sampled_k"]),
                    max_tokens=int(evaluation["natural_max_tokens"]),
                    greedy=mode == "greedy",
                    temperature=(
                        None if mode == "greedy" else float(evaluation["sampling_temperature"])
                    ),
                    top_p=None if mode == "greedy" else float(evaluation["sampling_top_p"]),
                    top_k=None if mode == "greedy" else int(evaluation["sampling_top_k"]),
                    run_seed=int(evaluation["seeds"][0 if mode == "greedy" else 1]),
                )
                raw_rows, metadata = runner.generate(records, sampling)
                scored, summary = _score_evaluation_rows(item_by_id, raw_rows)
                artifact = write_jsonl_gz(
                    eval_root / arm / f"{split}.jsonl.gz", scored
                )
                index.setdefault("arms", {}).setdefault(arm, {})[split] = {
                    "artifact": artifact,
                    "summary": summary,
                    "generation_metadata": metadata,
                    "model_override": None if model_override is None else str(model_override),
                }
                write_json(index_path, index)
                print(
                    f"[eval:{mode}] {arm}/{split} "
                    f"accuracy={summary['sample_accuracy']:.4f} pass={summary['pass_at_n']:.4f}",
                    flush=True,
                )
    compact = {
        arm: {
            split: entry["summary"]
            for split, entry in split_entries.items()
        }
        for arm, split_entries in index["arms"].items()
    }
    write_json(RUNS_DIR / f"evaluation_{mode}_summary.json", compact)
    return compact


def train_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Train every mandatory seed-42 arm, restarting at completed receipts."""
    design_boundary_receipt(config)
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            "missing separate Transformers training environment at .venv; "
            "create it per docs/compute_environment.md"
        )
    artifact_root = external_root(config)
    results = {}
    for arm in config["sft"]["arms"]:
        dataset = artifact_root / "sft" / f"{arm}.jsonl.gz"
        output = artifact_root / "adapters" / "seed42" / arm
        receipt_path = output / "training_receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if (
                receipt.get("arm") == arm
                and int(receipt.get("seed", -1)) == int(config["sft"]["screen_seed"])
                and receipt.get("dataset_sha256") == sha256_file(dataset)
                and int(receipt.get("skipped_rows", -1)) == 0
            ):
                results[arm] = receipt
                continue
            raise RuntimeError(f"stale or mismatched training receipt: {receipt_path}")
        print(f"[train-matrix] starting {arm}", flush=True)
        subprocess.run(
            [
                str(python),
                str(EXP / "scripts" / "train_think.py"),
                "--arm",
                arm,
                "--dataset",
                str(dataset),
                "--out",
                str(output),
                "--seed",
                str(config["sft"]["screen_seed"]),
            ],
            cwd=ROOT,
            check=True,
        )
        results[arm] = read_json(receipt_path)
    write_json(RUNS_DIR / "training_matrix_summary.json", results)
    return results


def merge_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Merge all adapters into deployable composite checkpoints."""
    design_boundary_receipt(config)
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("missing .venv for adapter merge")
    artifact_root = external_root(config)
    receipts = {}
    for arm in config["sft"]["arms"]:
        adapter = artifact_root / "adapters" / "seed42" / arm
        output = artifact_root / "merged" / "seed42" / arm
        receipt_path = output / "merge_receipt.json"
        adapter_receipt = read_json(adapter / "training_receipt.json")
        adapter_hash = adapter_receipt["artifacts"]["adapter_model.safetensors"]["sha256"]
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt.get("adapter_sha256") == adapter_hash:
                receipts[arm] = receipt
                continue
            raise RuntimeError(f"stale merged checkpoint receipt: {receipt_path}")
        print(f"[merge-matrix] starting {arm}", flush=True)
        subprocess.run(
            [
                str(python),
                str(EXP / "scripts" / "merge_adapter.py"),
                "--adapter",
                str(adapter),
                "--out",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        index_files = sorted(output.glob("*.index.json"))
        weight_files = sorted(output.glob("*.safetensors"))
        receipt = {
            "schema_version": 1,
            "arm": arm,
            "adapter": str(adapter),
            "adapter_sha256": adapter_hash,
            "output": str(output),
            "config_sha256": sha256_file(output / "config.json"),
            "weight_files": [
                {"name": path.name, "bytes": path.stat().st_size}
                for path in weight_files
            ],
            "weight_index_sha256": (
                sha256_file(index_files[0]) if index_files else None
            ),
        }
        write_json(receipt_path, receipt)
        receipts[arm] = receipt
    write_json(RUNS_DIR / "merge_matrix_summary.json", receipts)
    return receipts


def behavioral_difference_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Reject no-op merged deployments before result-bearing evaluation."""
    design_boundary_receipt(config)
    artifact_root = external_root(config)
    items = load_split("termination_pilot")[:8]
    records = [
        {
            "id": item["id"],
            "messages": [{"role": "user", "content": item["prompt"]}],
            "meta": {"family": item["family"], "level": item["level"]},
        }
        for item in items
    ]
    sampling = SamplingConfig(
        thinking="natural",
        n=1,
        max_tokens=4096,
        greedy=True,
        run_seed=int(config["evaluation"]["seeds"][0]) + 901,
    )

    def generate(label: str, model_override: Path | None) -> list[dict[str, Any]]:
        path = artifact_root / "deployment_probe" / f"{label}.jsonl.gz"
        receipt_path = artifact_root / "deployment_probe" / f"{label}.receipt.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if valid_receipt(receipt):
                return read_jsonl_gz(path)
        with VLLMRunner(
            engine_config(config, model_override=model_override)
        ) as runner:
            rows, metadata = runner.generate(records, sampling)
        receipt = write_jsonl_gz(path, rows)
        write_json(
            receipt_path,
            {**receipt, "generation_metadata": metadata, "label": label},
        )
        return rows

    base0 = generate("base0", None)
    base1 = generate("base1", None)
    base_tokens = {
        str(row["id"]): row["outputs"][0]["token_ids"] for row in base0
    }
    base_repeat_tokens = {
        str(row["id"]): row["outputs"][0]["token_ids"] for row in base1
    }
    null_differences = sum(
        base_tokens[task_id] != base_repeat_tokens[task_id]
        for task_id in base_tokens
    )
    if null_differences:
        raise RuntimeError(
            f"greedy base/base deployment probe is nondeterministic on {null_differences} tasks"
        )
    arms = {}
    for arm in config["sft"]["arms"]:
        rows = generate(
            arm, artifact_root / "merged" / "seed42" / arm
        )
        installed = {
            str(row["id"]): row["outputs"][0]["token_ids"] for row in rows
        }
        differences = sum(
            base_tokens[task_id] != installed[task_id] for task_id in base_tokens
        )
        arms[arm] = {"tasks": len(items), "token_sequence_differences": differences}
        if differences == 0:
            raise RuntimeError(f"merged deployment for {arm} is a behavioral no-op")
    result = {
        "schema_version": 1,
        "passed": True,
        "base_base_differences": null_differences,
        "arms": arms,
    }
    write_json(RUNS_DIR / "behavioral_difference_probe.json", result)
    return result


def analyze_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen paired decision rules to completed evaluations."""
    artifact_root = external_root(config)
    greedy_index = read_json(
        artifact_root / "evaluation" / "seed42" / "greedy" / "index.json"
    )
    sample_index = read_json(
        artifact_root / "evaluation" / "seed42" / "sample8" / "index.json"
    )
    arms = ["base", *config["sft"]["arms"]]
    splits = ("iid_eval", "hard_eval", "held_family_eval", "rendering_eval")
    task_scores: dict[str, dict[str, dict[str, float]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for arm in arms:
        task_scores[arm] = {}
        metrics[arm] = {}
        for split in splits:
            rows = read_jsonl_gz(
                Path(greedy_index["arms"][arm][split]["artifact"]["path"])
            )
            scores = {
                str(row["id"]): float(row["outputs"][0]["score"])
                for row in rows
            }
            parsed = [bool(row["outputs"][0]["parsed"]) for row in rows]
            closed = [bool(row["outputs"][0]["thinking_closed"]) for row in rows]
            by_family: dict[str, list[float]] = {}
            for row in rows:
                by_family.setdefault(str(row["family"]), []).append(
                    float(row["outputs"][0]["score"])
                )
            task_scores[arm][split] = scores
            metrics[arm][split] = {
                "tasks": len(rows),
                "accuracy": mean(list(scores.values())),
                "parse_rate": mean(parsed),
                "natural_close_rate": mean(closed),
                "family_macro": mean([mean(values) for values in by_family.values()]),
                "family_accuracy": {
                    family: mean(values) for family, values in sorted(by_family.items())
                },
                "mean_thinking_tokens": mean(
                    [float(row["outputs"][0]["n_thinking_tokens"]) for row in rows]
                ),
                "actual_forward_tokens": sum(
                    int(row["n_prompt_tokens"])
                    + int(row["outputs"][0]["n_sampled_tokens"])
                    for row in rows
                ),
            }

    comparisons = {}
    contrast_pairs = (
        ("answer_potential", "random_natural"),
        ("answer_potential", "success_rft"),
        ("answer_potential", "potential_shuffle"),
        ("joint_potential", "random_natural"),
        ("joint_potential", "success_rft"),
        ("joint_potential", "potential_shuffle"),
        ("answer_potential", "base"),
        ("joint_potential", "base"),
    )
    for treatment, baseline in contrast_pairs:
        key = f"{treatment}_minus_{baseline}"
        comparisons[key] = paired_bootstrap(
            {
                task_id: (
                    task_scores[treatment]["iid_eval"][task_id],
                    task_scores[baseline]["iid_eval"][task_id],
                )
                for task_id in task_scores[treatment]["iid_eval"]
            },
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"])
            + _stable_analysis_offset(key),
        )

    treatment_verdicts = {}
    for treatment in ("answer_potential", "joint_potential"):
        baselines = ("random_natural", "success_rft")
        baseline = max(
            baselines,
            key=lambda arm: metrics[arm]["iid_eval"]["accuracy"],
        )
        primary = comparisons[f"{treatment}_minus_{baseline}"]
        shuffled = comparisons[f"{treatment}_minus_potential_shuffle"]
        parse_delta = (
            metrics[treatment]["iid_eval"]["parse_rate"]
            - metrics[baseline]["iid_eval"]["parse_rate"]
        )
        family_delta = (
            metrics[treatment]["iid_eval"]["family_macro"]
            - metrics[baseline]["iid_eval"]["family_macro"]
        )
        positive = (
            float(primary["mean_delta"]) >= float(config["evaluation"]["positive_delta"])
            and float(primary["ci95_low"]) > 0
            and float(shuffled["mean_delta"]) > 0
            and parse_delta >= float(config["evaluation"]["noninferiority_delta"])
            and family_delta >= float(config["evaluation"]["noninferiority_delta"])
        )
        treatment_verdicts[treatment] = {
            "strongest_trace_baseline": baseline,
            "primary_contrast": primary,
            "shuffle_contrast": shuffled,
            "parse_delta": parse_delta,
            "family_macro_delta": family_delta,
            "full_trace_banking_positive": positive,
            "seed43_triggered": positive,
        }

    base_sample_rows = read_jsonl_gz(
        Path(sample_index["arms"]["base"]["iid_eval"]["artifact"]["path"])
    )
    sample_more = {}
    for k in (1, 2, 4, 8):
        sample_more[str(k)] = {
            "pass_at_k": mean(
                [
                    any(output["correct"] for output in row["outputs"][:k])
                    for row in base_sample_rows
                ]
            ),
            "actual_forward_tokens": sum(
                int(row["n_prompt_tokens"]) * k
                + sum(int(output["n_sampled_tokens"]) for output in row["outputs"][:k])
                for row in base_sample_rows
            ),
        }
    result = {
        "schema_version": 1,
        "metrics": metrics,
        "paired_iid_comparisons": comparisons,
        "treatment_verdicts": treatment_verdicts,
        "base_sample_more_curve": sample_more,
        "sample8_summaries": {
            arm: {
                split: sample_index["arms"][arm][split]["summary"]
                for split in splits
            }
            for arm in arms
        },
        "complete_seed42_matrix": all(
            arm in metrics and all(split in metrics[arm] for split in splits)
            for arm in arms
        ),
        "any_seed43_trigger": any(
            row["seed43_triggered"] for row in treatment_verdicts.values()
        ),
    }
    write_json(RUNS_DIR / "final_analysis.json", result)
    print(json.dumps({"iid": {arm: metrics[arm]["iid_eval"]["accuracy"] for arm in arms}, "verdicts": treatment_verdicts}, indent=2), flush=True)
    return result


def _stable_analysis_offset(text_value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text_value.encode("utf-8"), digest_size=4).digest(), "big"
    ) % 100_000


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design_boundary_receipt(config)
    build_data()
    items = load_split("termination_pilot")[:2]
    sampling = config["sampling"]
    smoke_dir = RUNS_DIR / "smoke"
    with AnswerPotentialModel(engine_config(config)) as model:
        traces, generation_meta = model.generate_thoughts(
            items,
            n=2,
            max_tokens=4096,
            run_seed=int(sampling["run_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
            logprobs=0,
            stage="smoke",
            chunk_size=1,
        )
        traces, continuation_meta = model.continue_unclosed_thoughts(
            items,
            traces,
            max_tokens=1024,
            run_seed=int(sampling["continuation_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
        vllm_scores, score_meta = model.score_answer_potential(
            items, traces, include_decoy=False
        )
        rollouts, rollout_meta = model.generate_answer_rollouts(
            items,
            traces,
            r=1,
            max_tokens=int(sampling["answer_max_tokens"]),
            run_seed=int(sampling["continuation_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
    write_jsonl(smoke_dir / "traces.jsonl", traces)
    write_jsonl(smoke_dir / "vllm_scores.jsonl", vllm_scores)
    write_jsonl(smoke_dir / "rollouts.jsonl", rollouts)
    scorer = HFAnswerPotentialScorer()
    try:
        item_by_id = {str(item["id"]): item for item in items}
        hf_scores = [
            scorer.score_trace(item_by_id[str(trace["task_id"])], trace)
            for trace in traces
        ]
    finally:
        scorer.close()
    write_jsonl(smoke_dir / "hf_scores.jsonl", hf_scores)
    vllm_by_trace = {row["trace_id"]: row for row in vllm_scores}
    deltas = [
        abs(row["answer_ll_sum"] - vllm_by_trace[row["trace_id"]]["canonical_ll_sum"])
        / len(row["answer_token_ids"])
        for row in hf_scores
    ]
    result = {
        "schema_version": 1,
        "passed": (
            len(traces) == 4
            and all(row.get("prior_logprob_mean") is not None for row in traces)
            and all(math.isfinite(value) for value in deltas)
            and max(deltas) <= float(config["scoring"]["max_abs_mean_token_delta"])
        ),
        "traces": _summarize_traces(traces),
        "max_abs_mean_token_delta_hf_vllm": max(deltas),
        "parity_threshold": config["scoring"]["max_abs_mean_token_delta"],
        "generation_meta": generation_meta,
        "continuation_meta": continuation_meta,
        "vllm_score_meta": score_meta,
        "rollout_meta": rollout_meta,
    }
    write_json(smoke_dir / "result.json", result)
    print(json.dumps(result["traces"], indent=2), flush=True)
    print(
        f"[smoke] parity={result['max_abs_mean_token_delta_hf_vllm']:.6f} "
        f"passed={result['passed']}",
        flush=True,
    )
    if not result["passed"]:
        raise RuntimeError("long-horizon smoke failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--stage",
        choices=(
            "data",
            "smoke",
            "pilot",
            "calibration-generate",
            "scorer-parity",
            "calibration-score",
            "calibration-rollouts",
            "calibration-analyze",
            "harvest-generate",
            "harvest-score",
            "pivot-plan",
            "branch-generate",
            "branch-score",
            "train-rollouts",
            "select",
            "train",
            "merge",
            "deployment-probe",
            "evaluate-greedy",
            "evaluate-sample8",
            "analyze",
            "full",
        ),
        default="smoke",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.stage = "smoke"
    config = load_config()
    if args.stage == "data":
        design_boundary_receipt(config)
        build_data()
    elif args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "pilot":
        generate_pool(
            config,
            split="termination_pilot",
            n=int(config["sampling"]["pilot_n"]),
            stage="termination_pilot",
        )
        analyze_termination_pilot(config)
    elif args.stage == "calibration-generate":
        generate_pool(
            config,
            split="calibration",
            n=int(config["sampling"]["calibration_n"]),
            stage="calibration_independent",
        )
    elif args.stage == "scorer-parity":
        run_scorer_parity(
            config,
            source_stage="calibration_independent",
            split="calibration",
        )
    elif args.stage == "calibration-score":
        score_pool(
            config,
            source_stage="calibration_independent",
            output_stage="calibration_scores",
            split="calibration",
        )
    elif args.stage == "calibration-rollouts":
        rollout_pool(
            config,
            split="calibration",
            source_stages=["calibration_independent"],
            output_stage="calibration_rollouts_r4",
            r=int(config["sampling"]["calibration_rollouts_per_trace"]),
        )
    elif args.stage == "calibration-analyze":
        analyze_calibration(config)
    elif args.stage == "harvest-generate":
        generate_pool(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        )
        ensure_minimum_natural_train_pool(config)
    elif args.stage == "harvest-score":
        score_pool(
            config,
            source_stage="train_independent",
            output_stage="train_independent_scores",
            split="train",
        )
    elif args.stage == "pivot-plan":
        plan_pivots(config)
    elif args.stage == "branch-generate":
        generate_branch_pool(config)
    elif args.stage == "branch-score":
        score_pool(
            config,
            source_stage="train_branches",
            output_stage="train_branch_scores",
            split="train",
        )
    elif args.stage == "train-rollouts":
        rollout_pool(
            config,
            split="train",
            source_stages=["train_independent", "train_branches"],
            output_stage="train_rollouts_r1",
            r=int(config["sampling"]["train_rollouts_per_trace"]),
        )
    elif args.stage == "select":
        build_sft_datasets(config)
    elif args.stage == "train":
        train_matrix(config)
    elif args.stage == "merge":
        merge_matrix(config)
    elif args.stage == "deployment-probe":
        behavioral_difference_probe(config)
    elif args.stage == "evaluate-greedy":
        evaluate_matrix(config, mode="greedy")
    elif args.stage == "evaluate-sample8":
        evaluate_matrix(config, mode="sample8")
    elif args.stage == "analyze":
        analyze_evaluation(config)
    else:
        build_data()
        generate_pool(
            config,
            split="termination_pilot",
            n=int(config["sampling"]["pilot_n"]),
            stage="termination_pilot",
        )
        analyze_termination_pilot(config)
        generate_pool(
            config,
            split="calibration",
            n=int(config["sampling"]["calibration_n"]),
            stage="calibration_independent",
        )
        run_scorer_parity(config, source_stage="calibration_independent", split="calibration")
        score_pool(
            config,
            source_stage="calibration_independent",
            output_stage="calibration_scores",
            split="calibration",
        )
        rollout_pool(
            config,
            split="calibration",
            source_stages=["calibration_independent"],
            output_stage="calibration_rollouts_r4",
            r=int(config["sampling"]["calibration_rollouts_per_trace"]),
        )
        analyze_calibration(config)
        generate_pool(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        )
        ensure_minimum_natural_train_pool(config)
        score_pool(
            config,
            source_stage="train_independent",
            output_stage="train_independent_scores",
            split="train",
        )
        plan_pivots(config)
        generate_branch_pool(config)
        score_pool(
            config,
            source_stage="train_branches",
            output_stage="train_branch_scores",
            split="train",
        )
        rollout_pool(
            config,
            split="train",
            source_stages=["train_independent", "train_branches"],
            output_stage="train_rollouts_r1",
            r=int(config["sampling"]["train_rollouts_per_trace"]),
        )
        build_sft_datasets(config)
        train_matrix(config)
        merge_matrix(config)
        behavioral_difference_probe(config)
        evaluate_matrix(config, mode="greedy")
        evaluate_matrix(config, mode="sample8")
        analyze_evaluation(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
