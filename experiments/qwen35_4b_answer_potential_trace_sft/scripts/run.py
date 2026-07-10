#!/usr/bin/env python3
"""Gated orchestrator for answer-potential trace SFT.

Stages are restartable and write atomic compact artifacts.  ``calibrate`` is
the first scientific GPU run.  ``full`` is hard-gated on both the immutable
design commit and a passing G0 receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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

from g0 import (  # noqa: E402
    attach_scores_and_rollouts,
    evaluate_g0,
    make_premember_checkpoints,
    selected_top_by_gain,
)
from io_utils import (  # noqa: E402
    artifact_receipt,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)
from model_ops import (  # noqa: E402
    AnswerPotentialModel,
    make_foreign_controls,
    make_token_shuffled_controls,
)
from task_data import build_all  # noqa: E402
from vllm_runner import EngineConfig  # noqa: E402

CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"
CAL_DIR = RUNS_DIR / "calibration"


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
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("default config must be a mapping")
    return value


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    expected = config["design_boundary"]
    commit = str(expected["commit"])
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    ancestor = _run(
        ["git", "merge-base", "--is-ancestor", commit, head], check=False
    ).returncode == 0
    paths = {
        "preregistration": "experiments/qwen35_4b_answer_potential_trace_sft/reports/preregistration.md",
        "readme": "experiments/qwen35_4b_answer_potential_trace_sft/README.md",
    }
    observed: dict[str, str] = {}
    for name, path in paths.items():
        payload = _run(["git", "show", f"{commit}:{path}"]).stdout.encode("utf-8")
        observed[name] = hashlib.sha256(payload).hexdigest()
    passed = (
        ancestor
        and observed["preregistration"] == expected["preregistration_sha256"]
        and observed["readme"] == expected["readme_sha256"]
    )
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "design_commit": commit,
        "current_head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": {
            "preregistration": expected["preregistration_sha256"],
            "readme": expected["readme_sha256"],
        },
        "gpu_scale_work_preceded_by_design_commit": True,
    }
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design-boundary verification failed: {receipt}")
    return receipt


def build_data() -> dict[str, Any]:
    print("[data] regenerating frozen procedural splits and firewall audit", flush=True)
    manifest = build_all(DATA_DIR)
    receipts = {
        path.stem: artifact_receipt(path, rows=len(read_jsonl(path)))
        for path in sorted(DATA_DIR.glob("*.jsonl"))
    }
    manifest["artifact_receipts"] = receipts
    write_json(DATA_DIR / "manifest.json", manifest)
    print(f"[data] split counts: {manifest['audit']['split_counts']}", flush=True)
    return manifest


def engine_config(config: dict[str, Any]) -> EngineConfig:
    engine = config["engine"]
    return EngineConfig(
        max_model_len=int(engine["max_model_len"]),
        gpu_memory_utilization=float(engine["gpu_memory_utilization"]),
        max_num_seqs=int(engine["max_num_seqs"]),
        max_num_batched_tokens=int(engine["max_num_batched_tokens"]),
        enable_prefix_caching=bool(engine["prefix_caching"]),
        cudagraph_capture_sizes=tuple(int(value) for value in engine["cudagraph_capture_sizes"]),
    )


def _save_rows(name: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    path = CAL_DIR / f"{name}.jsonl"
    write_jsonl(path, rows)
    write_json(CAL_DIR / f"{name}.meta.json", {**metadata, "artifact": artifact_receipt(path, rows=len(rows))})
    print(f"[calibrate] wrote {name}: {len(rows)} rows", flush=True)


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design_boundary_receipt(config)
    build_data()
    items = [row for row in read_jsonl(DATA_DIR / "calibration.jsonl") if row["potential_scorable"]][:2]
    sampling = config["sampling"]
    started = time.perf_counter()
    with AnswerPotentialModel(engine_config(config)) as model:
        print("[smoke] sampling four thought-only traces", flush=True)
        traces, trace_meta = model.generate_thoughts(
            items,
            n=2,
            max_tokens=64,
            run_seed=int(sampling["run_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
        print("[smoke] scoring exact teacher-forced answer spans", flush=True)
        scores, score_meta = model.score_answer_potential(items, traces, include_decoy=True)
        print("[smoke] sampling fresh trace-conditioned answers", flush=True)
        rollouts, rollout_meta = model.generate_answer_rollouts(
            items,
            traces,
            r=2,
            max_tokens=int(sampling["answer_max_tokens"]),
            run_seed=int(sampling["continuation_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
    smoke_dir = RUNS_DIR / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(smoke_dir / "traces.jsonl", traces)
    write_jsonl(smoke_dir / "scores.jsonl", scores)
    write_jsonl(smoke_dir / "rollouts.jsonl", rollouts)
    result = {
        "schema_version": 1,
        "passed": len(traces) == 4 and len(scores) == 4 and len(rollouts) == 4,
        "elapsed_seconds": time.perf_counter() - started,
        "counts": {"items": len(items), "traces": len(traces), "scores": len(scores), "rollouts": len(rollouts)},
        "natural_close_rate": sum(row["natural_close"] for row in traces) / len(traces),
        "finite_scores": all(
            isinstance(row["gain_sum"], (int, float))
            and row["gain_sum"] == row["gain_sum"]
            for row in scores
        ),
        "metadata": {"traces": trace_meta, "scores": score_meta, "rollouts": rollout_meta},
    }
    result["passed"] = bool(result["passed"] and result["finite_scores"])
    write_json(smoke_dir / "result.json", result)
    if not result["passed"]:
        raise RuntimeError(f"GPU smoke failed: {result}")
    print(f"[smoke] passed in {result['elapsed_seconds']:.1f}s", flush=True)
    return result


def run_calibration(config: dict[str, Any]) -> dict[str, Any]:
    design_boundary_receipt(config)
    build_data()
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    all_items = read_jsonl(DATA_DIR / "calibration.jsonl")
    scorable_items = [row for row in all_items if row["potential_scorable"]]
    item_by_id = {str(row["id"]): row for row in scorable_items}
    sampling = config["sampling"]
    gate_config = config["scorer_gate"]

    print(
        f"[calibrate] loading one engine for {len(all_items)} prompts x "
        f"{sampling['calibration_n']} thoughts; {len(scorable_items)} prompts enter G0",
        flush=True,
    )
    with AnswerPotentialModel(engine_config(config)) as model:
        traces, trace_meta = model.generate_thoughts(
            all_items,
            n=int(sampling["calibration_n"]),
            max_tokens=int(sampling["max_think_tokens"]),
            run_seed=int(sampling["run_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
        _save_rows("thoughts", traces, trace_meta)
        scorable_traces = [row for row in traces if str(row["task_id"]) in item_by_id]

        canonical, canonical_meta = model.score_answer_potential(
            scorable_items, scorable_traces, boundary="canonical", include_decoy=True
        )
        _save_rows("potential", canonical, canonical_meta)

        format_scores, format_meta = model.score_answer_potential(
            scorable_items, scorable_traces, boundary="format_variant", include_decoy=False
        )
        _save_rows("potential_format_variant", format_scores, format_meta)

        shuffled_controls = make_token_shuffled_controls(
            scorable_traces, seed=int(sampling["control_seed"])
        )
        shuffled_scores, shuffled_meta = model.score_answer_potential(
            scorable_items, shuffled_controls, include_decoy=False
        )
        _save_rows("potential_token_shuffled", shuffled_scores, shuffled_meta)

        foreign_controls = make_foreign_controls(scorable_traces)
        foreign_scores, foreign_meta = model.score_answer_potential(
            scorable_items, foreign_controls, include_decoy=False
        )
        _save_rows("potential_foreign", foreign_scores, foreign_meta)

        rollout_rows, rollout_meta = model.generate_answer_rollouts(
            scorable_items,
            scorable_traces,
            r=int(sampling["calibration_rollouts_per_trace"]),
            max_tokens=int(sampling["answer_max_tokens"]),
            run_seed=int(sampling["continuation_seed"]),
            temperature=float(sampling["temperature"]),
            top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]),
        )
        _save_rows("rollouts", rollout_rows, rollout_meta)

        joined = attach_scores_and_rollouts(scorable_traces, canonical, rollout_rows)
        selected = selected_top_by_gain(joined)
        checkpoints, checkpoint_diagnostics = make_premember_checkpoints(
            selected, item_by_id, model.runner.tokenizer
        )
        checkpoint_scores, checkpoint_meta = model.score_answer_potential(
            scorable_items, checkpoints, include_decoy=False
        )
        _save_rows("potential_premember", checkpoint_scores, checkpoint_meta)
        write_json(CAL_DIR / "premember_diagnostics.json", checkpoint_diagnostics)

    gate = evaluate_g0(
        traces=scorable_traces,
        canonical_scores=canonical,
        format_scores=format_scores,
        shuffled_scores=shuffled_scores,
        foreign_scores=foreign_scores,
        rollout_rows=rollout_rows,
        premention_scores=checkpoint_scores,
        premention_diagnostics=checkpoint_diagnostics,
        auroc_min=float(gate_config["within_task_auroc_min"]),
        uplift_min=float(gate_config["top1_uplift_min"]),
        kendall_min=float(gate_config["kendall_tau_min"]),
        premention_fraction_min=float(gate_config["premember_fraction_min"]),
        bootstrap_resamples=int(gate_config["bootstrap_resamples"]),
        bootstrap_seed=int(gate_config["bootstrap_seed"]),
        random_seed=int(sampling["control_seed"]),
    )
    write_json(CAL_DIR / "g0.json", gate)
    selector_freeze = {
        "schema_version": 1,
        "source": "configs/default.yaml preregistered defaults; no post-result tuning",
        **config["selector"],
    }
    write_json(CAL_DIR / "selector_freeze.json", selector_freeze)
    print(f"[calibrate] G0 passed={gate['passed']} criteria={gate['criteria']}", flush=True)
    return gate


def require_passing_g0(config: dict[str, Any]) -> dict[str, Any]:
    design_boundary_receipt(config)
    path = CAL_DIR / "g0.json"
    if not path.is_file():
        raise RuntimeError("full run requires runs/calibration/g0.json; run --stage calibrate first")
    gate = read_json(path)
    if gate.get("gate") != "G0" or gate.get("passed") is not True:
        raise RuntimeError("full run is prohibited because preregistered G0 did not pass")
    return gate


def run_full(config: dict[str, Any]) -> dict[str, Any]:
    require_passing_g0(config)
    raise RuntimeError("full-stage implementation is not complete; refusing an unaudited expensive run")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("data", "smoke", "calibrate", "full"), default="smoke")
    parser.add_argument("--smoke", action="store_true", help="compatibility alias for --stage smoke")
    args = parser.parse_args(argv)
    if args.smoke:
        args.stage = "smoke"
    config = load_config()
    if args.stage == "data":
        design_boundary_receipt(config)
        build_data()
    elif args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "calibrate":
        run_calibration(config)
    else:
        run_full(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
