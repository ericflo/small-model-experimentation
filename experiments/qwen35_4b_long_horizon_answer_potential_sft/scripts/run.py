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
from io_utils import read_json, read_jsonl, write_json, write_jsonl  # noqa: E402
from model_ops import AnswerPotentialModel  # noqa: E402
from shards import read_jsonl_gz, valid_receipt, write_jsonl_gz  # noqa: E402
from task_data import build_all  # noqa: E402
from vllm_runner import EngineConfig, MODEL_ID, MODEL_REVISION  # noqa: E402

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


def engine_config(config: dict[str, Any]) -> EngineConfig:
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


def score_pool(
    config: dict[str, Any], *, source_stage: str, output_stage: str, split: str
) -> dict[str, Any]:
    design_boundary_receipt(config)
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
            "calibration-score",
            "harvest-generate",
            "harvest-score",
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
    elif args.stage == "calibration-generate":
        generate_pool(
            config,
            split="calibration",
            n=int(config["sampling"]["calibration_n"]),
            stage="calibration_independent",
        )
    elif args.stage == "calibration-score":
        score_pool(
            config,
            source_stage="calibration_independent",
            output_stage="calibration_scores",
            split="calibration",
        )
    elif args.stage == "harvest-generate":
        generate_pool(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        )
    elif args.stage == "harvest-score":
        score_pool(
            config,
            source_stage="train_independent",
            output_stage="train_independent_scores",
            split="train",
        )
    else:
        raise RuntimeError("full pipeline is not yet implemented; refusing partial execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
