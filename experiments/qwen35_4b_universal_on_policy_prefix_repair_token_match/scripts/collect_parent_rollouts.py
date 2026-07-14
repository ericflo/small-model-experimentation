#!/usr/bin/env python3
"""Collect one frozen vLLM rollout from the explicitly merged close_xi parent."""

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
INPUT = EXP / "data" / "parent_rollout_input.jsonl"
TASK_MANIFEST = EXP / "data" / "rollout_task_manifest.json"
MERGED = ROOT / "large_artifacts" / EXP.name / "merged" / "close_xi_parent"
MERGE_RECEIPT = EXP / "runs" / "merges" / "close_xi_parent.json"
OUTPUT = EXP / "runs" / "parent_rollout" / "seed66113.jsonl"
METADATA = EXP / "runs" / "parent_rollout" / "seed66113.meta.json"
RECEIPT = EXP / "runs" / "parent_rollout" / "seed66113.receipt.json"
LOG = EXP / "runs" / "parent_rollout" / "seed66113.log"
SEED = 66113
ROWS = 288
MAX_TOKENS = 1024


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


def authenticate_inputs() -> None:
    manifest = load_json(TASK_MANIFEST)
    if (
        manifest.get("experiment_id") != EXP.name
        or manifest.get("construction_seed") != 77113
        or manifest.get("rows") != ROWS
        or manifest.get("runner_input", {}).get("sha256") != sha256_file(INPUT)
        or manifest.get("runner_input_excludes_hidden_oracle_fields") is not True
    ):
        raise ValueError("parent rollout input failed its frozen task manifest")
    merge = load_json(MERGE_RECEIPT)
    external = MERGED / "merge_receipt.json"
    if (
        merge.get("schema_version") != 1
        or merge.get("experiment_id") != EXP.name
        or merge.get("name") != "close_xi_parent"
        or Path(merge.get("merged", "")).resolve() != MERGED.resolve()
        or merge.get("merge_receipt_sha256") != sha256_file(external)
        or not (MERGED / "config.json").is_file()
    ):
        raise ValueError("merged parent failed authentication")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args()
    authenticate_inputs()
    if any(path.exists() for path in (OUTPUT, METADATA, RECEIPT, LOG)):
        parser.error("refusing to overwrite a parent rollout artifact")
    command = [
        str(PYTHON),
        str(RUNNER),
        "--input", str(INPUT),
        "--output", str(OUTPUT),
        "--metadata", str(METADATA),
        "--model-override", str(MERGED),
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

    metadata = load_json(METADATA)
    rows = [json.loads(line) for line in OUTPUT.read_text(encoding="utf-8").splitlines() if line]
    manifest = load_json(TASK_MANIFEST)
    if (
        len(rows) != ROWS
        or len({row.get("id") for row in rows}) != ROWS
        or any(len(row.get("outputs", [])) != 1 for row in rows)
        or metadata.get("schema_version") != 4
        or Path(metadata.get("model", "")).resolve() != MERGED.resolve()
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
        or metadata.get("input", {}).get("sha256")
        != manifest.get("runner_input", {}).get("sha256")
        or metadata.get("runtime", {}).get("git_dirty") is not False
    ):
        raise SystemExit("vLLM rollout output or metadata failed the frozen collection contract")
    payload = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "authenticated_parent_rollout_collection",
        "seed": SEED,
        "model": str(MERGED.resolve()),
        "merge_receipt": str(MERGE_RECEIPT.resolve()),
        "merge_receipt_sha256": sha256_file(MERGE_RECEIPT),
        "runner": str(RUNNER.resolve()),
        "runner_sha256": sha256_file(RUNNER),
        "input": str(INPUT.resolve()),
        "input_sha256": sha256_file(INPUT),
        "rows": ROWS,
        "rollouts": str(OUTPUT.resolve()),
        "rollouts_sha256": sha256_file(OUTPUT),
        "metadata": str(METADATA.resolve()),
        "metadata_sha256": sha256_file(METADATA),
        "log": str(LOG.resolve()),
        "log_sha256": sha256_file(LOG),
        "wall_seconds": elapsed,
        "sampled_tokens": metadata["counts"]["sampled_tokens"],
        "sampled_tokens_per_second": metadata["timing"]["sampled_tokens_per_second"],
        "backend": "vllm_merged_composite",
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
