#!/usr/bin/env python3
"""Collect the frozen failure-selection rollout from the merged replay parent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv-vllm" / "bin" / "python"
RUNNER = EXP / "src" / "vllm_runner.py"
INPUT = EXP / "data" / "parent_rollout_input_seed66114.jsonl"
TASK_MANIFEST = EXP / "data" / "rollout_task_manifest.json"
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
PARENT = ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "replay_after_close"
PARENT_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_after_close.json"
OUTPUT = EXP / "runs" / "parent_rollout" / "seed66114.jsonl"
METADATA = EXP / "runs" / "parent_rollout" / "seed66114.meta.json"
RECEIPT = EXP / "runs" / "parent_rollout" / "seed66114.receipt.json"
LOG = EXP / "runs" / "parent_rollout" / "seed66114.log"
SEED = 66114
ROWS = 624
MAX_TOKENS = 1024
PARENT_TRACKED_RECEIPT_SHA256 = "bc78f33218afb99b4ebd5b173f1f24aa628b20fad82d627b00529cabf911d550"
PARENT_EXTERNAL_RECEIPT_SHA256 = "aa763255cb3b05599e765948d3a3db1787d5813b1cfafbdc7e1c21653ae745a3"
PARENT_WEIGHTS_SHA256 = "7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"not a JSON object: {path}")
    return value


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def normalize_log() -> None:
    lines = LOG.read_text(encoding="utf-8").splitlines()
    LOG.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def authenticate_inputs() -> dict:
    manifest = load_json(TASK_MANIFEST)
    tracked = load_json(PARENT_RECEIPT)
    external = PARENT / "merge_receipt.json"
    weights = PARENT / "model.safetensors"
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != 77114
        or manifest.get("rows") != ROWS
        or manifest.get("runner_input", {}).get("sha256") != sha256_file(INPUT)
        or manifest.get("runner_input", {}).get("excludes_oracle_think_answer_and_audit") is not True
        or sha256_file(PARENT_RECEIPT) != PARENT_TRACKED_RECEIPT_SHA256
        or tracked.get("name") != "replay_after_close"
        or tracked.get("merge_receipt_sha256") != PARENT_EXTERNAL_RECEIPT_SHA256
        or tracked.get("weight_files", [{}])[0].get("sha256") != PARENT_WEIGHTS_SHA256
        or not external.is_file()
        or sha256_file(external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or not weights.is_file()
        or weights.stat().st_size != 9_078_620_536
        or not (PARENT / "config.json").is_file()
    ):
        raise ValueError("merged replay parent or rollout substrate failed authentication")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--recover-completed",
        action="store_true",
        help="authenticate complete output/metadata/log without rerunning generation",
    )
    args = parser.parse_args()
    manifest = authenticate_inputs()
    command = [
        str(PYTHON),
        "-B",
        str(RUNNER),
        "--input", str(INPUT),
        "--output", str(OUTPUT),
        "--metadata", str(METADATA),
        "--model-override", str(PARENT),
        "--thinking", "natural",
        "--n", "1",
        "--max-tokens", str(MAX_TOKENS),
        "--greedy",
        "--seed", str(SEED),
        "--max-model-len", "4096",
        "--gpu-memory-utilization", "0.90",
        "--max-num-seqs", "16",
        "--max-num-batched-tokens", "8192",
        "--cudagraph-capture-size", "1",
        "--cudagraph-capture-size", "2",
        "--cudagraph-capture-size", "4",
        "--cudagraph-capture-size", "8",
        "--cudagraph-capture-size", "16",
    ]
    git_head = run_text(["git", "rev-parse", "HEAD"])
    origin_main = run_text(["git", "rev-parse", "origin/main"])
    branch = run_text(["git", "branch", "--show-current"])
    preflight_status = run_text(["git", "status", "--short"])
    elapsed: float | None = None
    if branch != "main" or git_head != origin_main:
        parser.error("collection requires pushed main at the current HEAD")
    if args.recover_completed:
        if RECEIPT.exists() or not all(path.is_file() for path in (OUTPUT, METADATA, LOG)):
            parser.error("recovery requires complete output/metadata/log and no receipt")
    else:
        if preflight_status:
            parser.error("fresh collection requires a clean worktree")
        if any(path.exists() for path in (OUTPUT, METADATA, RECEIPT, LOG)):
            parser.error("refusing to overwrite a parent rollout artifact")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        with LOG.open("x", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
                },
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            returncode = process.wait()
        elapsed = time.perf_counter() - started
        if returncode != 0:
            raise SystemExit(f"parent rollout failed with exit {returncode}; preserved {LOG}")

    normalize_log()
    metadata = load_json(METADATA)
    rows = [
        json.loads(line)
        for line in OUTPUT.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if (
        len(rows) != ROWS
        or len({row.get("id") for row in rows}) != ROWS
        or any(len(row.get("outputs", [])) != 1 for row in rows)
        or metadata.get("schema_version") != 4
        or Path(metadata.get("model", "")).resolve() != PARENT.resolve()
        or metadata.get("model_revision") is not None
        or metadata.get("adapter") is not None
        or metadata.get("runner_sha256") != sha256_file(RUNNER)
        or metadata.get("counts", {}).get("requests") != ROWS
        or metadata.get("counts", {}).get("completions") != ROWS
        or metadata.get("sampling", {}).get("thinking") != "natural"
        or metadata.get("sampling", {}).get("n") != 1
        or metadata.get("sampling", {}).get("max_tokens") != MAX_TOKENS
        or metadata.get("sampling", {}).get("greedy") is not True
        or metadata.get("sampling", {}).get("run_seed") != SEED
        or metadata.get("input", {}).get("sha256") != manifest["runner_input"]["sha256"]
        or metadata.get("runtime", {}).get("git_commit") != git_head
    ):
        raise SystemExit("vLLM rollout output or metadata failed the frozen contract")
    payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "authenticated_replay_parent_failure_selection_rollout",
        "seed": SEED,
        "rows": ROWS,
        "model": str(PARENT.resolve()),
        "parent_experiment": PARENT_EXP.name,
        "parent_arm": "replay_after_close",
        "parent_tracked_receipt_sha256": PARENT_TRACKED_RECEIPT_SHA256,
        "parent_external_receipt_sha256": PARENT_EXTERNAL_RECEIPT_SHA256,
        "parent_weights_sha256": PARENT_WEIGHTS_SHA256,
        "runner_sha256": sha256_file(RUNNER),
        "input_sha256": sha256_file(INPUT),
        "rollouts": str(OUTPUT.resolve()),
        "rollouts_sha256": sha256_file(OUTPUT),
        "metadata": str(METADATA.resolve()),
        "metadata_sha256": sha256_file(METADATA),
        "log": str(LOG.resolve()),
        "log_sha256": sha256_file(LOG),
        "wrapper_wall_seconds": elapsed,
        "sampled_tokens": metadata["counts"]["sampled_tokens"],
        "sampled_tokens_per_second": metadata["timing"]["sampled_tokens_per_second"],
        "backend": "vllm_merged_composite",
        "sampling": {
            "thinking": "natural",
            "greedy": True,
            "n": 1,
            "max_tokens": MAX_TOKENS,
            "max_model_len": 4096,
        },
        "git": {
            "head": git_head,
            "branch": branch,
            "origin_main": origin_main,
            "preflight_status": preflight_status if not args.recover_completed else "attested by stage preflight",
            "runner_observed_dirty": metadata["runtime"]["git_dirty"],
            "runner_dirty_reason": "collector opens its durable log before runner metadata is sampled",
        },
        "recovery": {"used": args.recover_completed, "generation_rerun": False},
        "benchmark_data_read": False,
        "aggregate_seed_open": False,
        "command": command,
    }
    RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
