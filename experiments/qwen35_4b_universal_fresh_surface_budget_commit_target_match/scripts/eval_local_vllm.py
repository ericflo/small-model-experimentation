#!/usr/bin/env python3
"""Run the frozen fresh local gate on four explicit composites through vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv-vllm" / "bin" / "python"
RUNNER = EXP / "src" / "vllm_runner.py"
SOURCE = EXP / "data" / "local_tasks_seed88013.jsonl"
INPUT = EXP / "data" / "local_input_seed88013.jsonl"
DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
LOCAL_RECEIPT = EXP / "runs" / "local" / "seed88013.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / "seed88013_promotion.json"
FAILURE_RECEIPT = EXP / "runs" / "local" / "seed88013.failure.json"
SEED = 88013
AGGREGATE_SEED = 78143
ROWS = 104
MAX_TOKENS = 1024
LABELS = (
    "replay_after_close_parent",
    "replay_repeat",
    "designed_fresh",
    "budget_commit",
)
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
PARENT_MERGE_RECEIPT = PARENT_EXP / "runs" / "merges" / "replay_after_close.json"
PARENT_MERGE_RECEIPT_SHA256 = (
    "bc78f33218afb99b4ebd5b173f1f24aa628b20fad82d627b00529cabf911d550"
)
PARENT_TREE_SHA256 = (
    "d3493b44e1776024bee1422f9ad27153a3e01c3b2337993499129ca68eab2f7b"
)
PARENT_FILES = [
    {
        "name": "chat_template.jinja",
        "sha256": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "size": 7756,
    },
    {
        "name": "config.json",
        "sha256": "a1c80f0efa6f83f631eaa9c25ffa166e3b1f9db395cc3b14374dfc0962261f60",
        "size": 2829,
    },
    {
        "name": "generation_config.json",
        "sha256": "0c46d8aa4f0ae5e611c961f70b87c83fb696043c1e319337708e96f882180de1",
        "size": 116,
    },
    {
        "name": "merge_receipt.json",
        "sha256": "aa763255cb3b05599e765948d3a3db1787d5813b1cfafbdc7e1c21653ae745a3",
        "size": 895,
    },
    {
        "name": "model.safetensors",
        "sha256": "7ab4c419f70135d3fe058dba6e79e3a9a61c6661d43e6acb9662f331efe36e2e",
        "size": 9078620536,
    },
    {
        "name": "tokenizer.json",
        "sha256": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "size": 19989325,
    },
    {
        "name": "tokenizer_config.json",
        "sha256": "9cf04fffe3d8c3b85e439fb35c7acad0761ab51c422a8c4256d9f887c3a0be7d",
        "size": 1125,
    },
]
PARENT_MERGED = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
    / "merged"
    / "replay_after_close"
)
MERGED = {
    "replay_after_close_parent": PARENT_MERGED,
    "replay_repeat": ROOT / "large_artifacts" / EXP.name / "merged" / "replay_repeat",
    "designed_fresh": ROOT / "large_artifacts" / EXP.name / "merged" / "designed_fresh",
    "budget_commit": ROOT / "large_artifacts" / EXP.name / "merged" / "budget_commit",
}
# TODO-PIN: the orchestrator fills each tree sha256 after merge_trained_arm.py
# publishes the corresponding merged composite; the eval refuses to run while
# any pin is still None.
EXPECTED_TREE_SHA256: dict[str, str | None] = {
    "replay_repeat": None,  # TODO-PIN
    "designed_fresh": None,  # TODO-PIN
    "budget_commit": None,  # TODO-PIN
}
ANSWER_RE = re.compile(r"(?:^|\n)ANSWER:\s*(.*?)(?=\n|<\||</|$)", re.DOTALL)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def committed_at_head(path: Path) -> bool:
    relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
    completed = subprocess.run(
        ["git", "show", f"HEAD:{relative}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0 and completed.stdout == path.read_bytes()


def run_text(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def parse_answer(text: str) -> str | None:
    matches = [match.group(1).strip() for match in ANSWER_RE.finditer(text)]
    return matches[-1] if matches and matches[-1] else None


def normalize_log(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")


def preserve_failure(payload: dict) -> None:
    FAILURE_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    with FAILURE_RECEIPT.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def arm_paths(label: str) -> dict[str, Path]:
    stem = EXP / "runs" / "local" / f"seed{SEED}_{label}"
    return {
        "output": stem.with_suffix(".jsonl"),
        "metadata": Path(str(stem) + ".meta.json"),
        "log": stem.with_suffix(".log"),
    }


def authenticate_design() -> dict:
    sys.path.insert(0, str(EXP / "scripts"))
    import gen_local_gate  # noqa: PLC0415

    expected = gen_local_gate.build_outputs(authenticate_parent=False)
    for path, value in expected.items():
        if not path.is_file() or path.read_bytes() != value or not committed_at_head(path):
            raise ValueError(f"local design is absent, changed, or uncommitted: {path}")
    receipt = load_json(DESIGN_RECEIPT)
    if (
        receipt.get("seed") != SEED
        or receipt.get("aggregate_seed") != AGGREGATE_SEED
        or receipt.get("rows") != ROWS
        or receipt.get("arms") != list(LABELS)
        or receipt.get("runner_input", {}).get("sha256") != sha256_file(INPUT)
        or receipt.get("code_sha256", {}).get("runner") != sha256_file(RUNNER)
        or receipt.get("backend", {}).get("name") != "vllm_merged_composite"
        or receipt.get("firewall", {}).get("benchmark_data_read") is not False
    ):
        raise ValueError("local design receipt violates the frozen protocol")
    return receipt


def authenticate_model(label: str) -> dict:
    sys.path.insert(0, str(EXP / "scripts"))
    from merge_trained_arm import (  # noqa: PLC0415
        merged_tree_manifest,
        tree_manifest_sha256,
        validate_published_merge,
    )

    if label == "replay_after_close_parent":
        parent = load_json(PARENT_MERGE_RECEIPT)
        parent_files = merged_tree_manifest(PARENT_MERGED)
        if (
            not committed_at_head(PARENT_MERGE_RECEIPT)
            or sha256_file(PARENT_MERGE_RECEIPT) != PARENT_MERGE_RECEIPT_SHA256
            or parent.get("name") != "replay_after_close"
            or Path(parent.get("merged", "")).resolve() != PARENT_MERGED.resolve()
            or parent_files != PARENT_FILES
            or tree_manifest_sha256(parent_files) != PARENT_TREE_SHA256
        ):
            raise ValueError("published replay-parent composite changed")
        return parent
    if label not in LABELS:
        raise ValueError(f"unknown local arm: {label}")
    expected_tree = EXPECTED_TREE_SHA256[label]
    if expected_tree is None:
        raise ValueError(
            f"model-tree pin for {label} is unfilled (TODO-PIN); "
            "the orchestrator must pin the published merge tree first"
        )
    receipt = validate_published_merge(label)
    arm_files = merged_tree_manifest(MERGED[label])
    if (
        Path(receipt.get("merged", "")).resolve() != MERGED[label].resolve()
        or receipt.get("output_tree_sha256") != expected_tree
        or tree_manifest_sha256(arm_files) != expected_tree
    ):
        raise ValueError(f"published {label} composite tree changed")
    return receipt


def authenticate_models() -> dict[str, dict]:
    return {label: authenticate_model(label) for label in LABELS}


def authenticate_checkpoint(git_head: str) -> None:
    if (
        run_text(["git", "rev-parse", "HEAD"]) != git_head
        or run_text(["git", "rev-parse", "origin/main"]) != git_head
        or run_text(["git", "branch", "--show-current"]) != "main"
    ):
        raise ValueError("git checkpoint changed during the local event")


def authenticate_boundary(label: str, git_head: str) -> dict:
    """Close design/model/git TOCTOU windows around every model process."""
    authenticate_checkpoint(git_head)
    authenticate_design()
    receipt = authenticate_model(label)
    authenticate_checkpoint(git_head)
    return receipt


def command_for(label: str, paths: dict[str, Path]) -> list[str]:
    return [
        str(PYTHON),
        "-B",
        str(RUNNER),
        "--input",
        str(INPUT),
        "--output",
        str(paths["output"]),
        "--metadata",
        str(paths["metadata"]),
        "--model-override",
        str(MERGED[label]),
        "--thinking",
        "natural",
        "--n",
        "1",
        "--max-tokens",
        str(MAX_TOKENS),
        "--greedy",
        "--seed",
        str(SEED),
        "--max-model-len",
        "4096",
        "--gpu-memory-utilization",
        "0.90",
        "--max-num-seqs",
        "16",
        "--max-num-batched-tokens",
        "8192",
        "--cudagraph-capture-size",
        "1",
        "--cudagraph-capture-size",
        "2",
        "--cudagraph-capture-size",
        "4",
        "--cudagraph-capture-size",
        "8",
        "--cudagraph-capture-size",
        "16",
    ]


def validate_raw_arm(
    label: str, paths: dict[str, Path], *, git_head: str, input_ids: set[str]
) -> tuple[list[dict], dict]:
    rows = load_jsonl(paths["output"])
    metadata = load_json(paths["metadata"])
    engine = metadata.get("engine", {})
    engine_args = metadata.get("engine_args", {})
    sampling = metadata.get("sampling", {})
    resolved_sampling = metadata.get("resolved_sampling", {})
    resolved_cudagraph = metadata.get("resolved_cudagraph", {})
    runtime = metadata.get("runtime", {})
    if (
        len(rows) != ROWS
        or {row.get("id") for row in rows} != input_ids
        or len({row.get("id") for row in rows}) != ROWS
        or any(len(row.get("outputs", [])) != 1 for row in rows)
        or metadata.get("schema_version") != 4
        or Path(metadata.get("model", "")).resolve() != MERGED[label].resolve()
        or metadata.get("model_revision") is not None
        or metadata.get("adapter") is not None
        or metadata.get("runner_sha256") != sha256_file(RUNNER)
        or metadata.get("input", {}).get("sha256") != sha256_file(INPUT)
        or metadata.get("counts", {}).get("requests") != ROWS
        or metadata.get("counts", {}).get("completions") != ROWS
        or metadata.get("runtime", {}).get("git_commit") != git_head
        or runtime.get("git_dirty") is not True
        or runtime.get("vllm_enable_v1_multiprocessing") != "0"
        or sampling.get("thinking") != "natural"
        or sampling.get("n") != 1
        or sampling.get("max_tokens") != MAX_TOKENS
        or sampling.get("greedy") is not True
        or sampling.get("run_seed") != SEED
        or resolved_sampling
        != {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        }
        or engine.get("max_model_len") != 4096
        or engine.get("gpu_memory_utilization") != 0.90
        or engine.get("max_num_seqs") != 16
        or engine.get("max_num_batched_tokens") != 8192
        or engine.get("enable_prefix_caching") is not False
        or engine.get("enforce_eager") is not False
        or engine.get("adapter") is not None
        or Path(engine.get("model_override", "")).resolve()
        != MERGED[label].resolve()
        or engine.get("cudagraph_capture_sizes") != [1, 2, 4, 8, 16]
        or Path(engine_args.get("model", "")).resolve() != MERGED[label].resolve()
        or resolved_cudagraph.get("cudagraph_capture_sizes")
        != [1, 2, 4, 8, 16]
        or resolved_cudagraph.get("max_cudagraph_capture_size") != 16
        or resolved_cudagraph.get("has_full_cudagraphs") is not True
    ):
        raise ValueError(f"raw local output failed frozen vLLM contract for {label}")
    return rows, metadata


def summarize(rows: list[dict]) -> dict:
    by_kind: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_kind[row["kind"]].append(row)
    return {
        "rows": len(rows),
        "parsed": sum(row["parsed"] is not None for row in rows),
        "correct": sum(row["correct"] for row in rows),
        "cap_contacts": sum(row["cap_contact"] for row in rows),
        "mean_sampled_tokens": sum(row["n_sampled_tokens"] for row in rows)
        / len(rows),
        "per_kind": {
            kind: {
                "n": len(kind_rows),
                "parsed": sum(row["parsed"] is not None for row in kind_rows),
                "correct": sum(row["correct"] for row in kind_rows),
            }
            for kind, kind_rows in sorted(by_kind.items())
        },
        "answer_counts": dict(
            sorted(
                Counter(
                    row["parsed"] if row["parsed"] is not None else "<NONE>"
                    for row in rows
                ).items()
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.parse_args()
    all_paths = [path for label in LABELS for path in arm_paths(label).values()]
    if (
        LOCAL_RECEIPT.exists()
        or PROMOTION_RECEIPT.exists()
        or FAILURE_RECEIPT.exists()
        or any(path.exists() for path in all_paths)
    ):
        parser.error("refusing to overwrite a local-evaluation artifact")
    git_head = run_text(["git", "rev-parse", "HEAD"])
    origin = run_text(["git", "rev-parse", "origin/main"])
    branch = run_text(["git", "branch", "--show-current"])
    preflight_status = run_text(["git", "status", "--short"])
    if preflight_status or branch != "main" or git_head != origin:
        parser.error("fresh local event requires a clean pushed main checkpoint")
    try:
        authenticate_design()
        model_receipts = authenticate_models()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))

    source_rows = load_jsonl(SOURCE)
    source_by_id = {row["task_id"]: row for row in source_rows}
    input_ids = {row["id"] for row in load_jsonl(INPUT)}
    if len(source_by_id) != ROWS or set(source_by_id) != input_ids:
        parser.error("local source and model-facing input disagree")

    graded_rows: list[dict] = []
    raw_artifacts = {}
    commands = {}
    for label in LABELS:
        paths = arm_paths(label)
        command = command_for(label, paths)
        commands[label] = command
        try:
            model_receipts[label] = authenticate_boundary(label, git_head)
        except (OSError, ValueError, json.JSONDecodeError) as error:
            preserve_failure(
                {
                    "schema_version": 1,
                    "experiment_id": EXP.name,
                    "seed": SEED,
                    "label": label,
                    "failure_stage": "pre_arm_authentication",
                    "error": str(error),
                    "git_head": git_head,
                }
            )
            raise SystemExit(f"local pre-arm authentication failed: {error}")
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        try:
            with paths["log"].open("x", encoding="utf-8") as log:
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
        except OSError as error:
            if paths["log"].is_file():
                normalize_log(paths["log"])
            preserve_failure(
                {
                    "schema_version": 1,
                    "experiment_id": EXP.name,
                    "seed": SEED,
                    "label": label,
                    "failure_stage": "model_process_launch",
                    "error": str(error),
                    "git_head": git_head,
                    "command": command,
                    "log_sha256": (
                        sha256_file(paths["log"])
                        if paths["log"].is_file()
                        else None
                    ),
                }
            )
            raise SystemExit(f"local vLLM launch failed: {error}")
        normalize_log(paths["log"])
        if returncode != 0:
            preserve_failure(
                {
                    "schema_version": 1,
                    "experiment_id": EXP.name,
                    "seed": SEED,
                    "label": label,
                    "failure_stage": "model_process",
                    "returncode": returncode,
                    "git_head": git_head,
                    "command": command,
                    "log_sha256": sha256_file(paths["log"]),
                }
            )
            raise SystemExit(
                f"local vLLM arm {label} failed with exit {returncode}; "
                "preserved artifacts"
            )
        try:
            model_receipts[label] = authenticate_boundary(label, git_head)
            raw_rows, metadata = validate_raw_arm(
                label, paths, git_head=git_head, input_ids=input_ids
            )
        except (OSError, ValueError, json.JSONDecodeError) as error:
            preserve_failure(
                {
                    "schema_version": 1,
                    "experiment_id": EXP.name,
                    "seed": SEED,
                    "label": label,
                    "failure_stage": "post_arm_authentication_or_validation",
                    "error": str(error),
                    "git_head": git_head,
                    "command": command,
                    "log_sha256": sha256_file(paths["log"]),
                }
            )
            raise SystemExit(f"local post-arm validation failed: {error}")
        for raw in raw_rows:
            task = source_by_id[raw["id"]]
            output = raw["outputs"][0]
            text = output["text"]
            parsed = parse_answer(text)
            expected = task["answer"].removeprefix("ANSWER: ").strip()
            graded_rows.append(
                {
                    "adapter": label,
                    "task_id": raw["id"],
                    "kind": task["kind"],
                    "surface": task["surface"],
                    "expected": expected,
                    "parsed": parsed,
                    "correct": parsed == expected,
                    "n_sampled_tokens": output["n_sampled_tokens"],
                    "n_thinking_tokens": output["n_thinking_tokens"],
                    "n_answer_tokens": output["n_answer_tokens"],
                    "cap_contact": bool(output["truncated"])
                    or output["n_sampled_tokens"] >= MAX_TOKENS,
                    "finish_reason": output["finish_reason"],
                    "completion_sha256": sha256_bytes(text.encode()),
                }
            )
        model_receipt = (
            PARENT_MERGE_RECEIPT
            if label == "replay_after_close_parent"
            else EXP / "runs" / "merges" / f"{label}.json"
        )
        raw_artifacts[label] = {
            "model": str(MERGED[label].resolve()),
            "model_receipt": str(model_receipt.resolve()),
            "model_receipt_sha256": sha256_file(model_receipt),
            "output": str(paths["output"].resolve()),
            "output_sha256": sha256_file(paths["output"]),
            "metadata": str(paths["metadata"].resolve()),
            "metadata_sha256": sha256_file(paths["metadata"]),
            "log": str(paths["log"].resolve()),
            "log_sha256": sha256_file(paths["log"]),
            "sampled_tokens": metadata["counts"]["sampled_tokens"],
            "generation_seconds": metadata["timing"]["generation_seconds"],
            "sampled_tokens_per_second": metadata["timing"][
                "sampled_tokens_per_second"
            ],
            "authenticated_model": model_receipts[label].get("name", label),
        }

    summaries = {
        label: summarize([row for row in graded_rows if row["adapter"] == label])
        for label in LABELS
    }
    local = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "fresh_local_capability_gate",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "seed": SEED,
        "rows_per_arm": ROWS,
        "labels": list(LABELS),
        "backend": "vllm_merged_composite",
        "runner_sha256": sha256_file(RUNNER),
        "evaluator_sha256": sha256_file(Path(__file__)),
        "design_receipt": str(DESIGN_RECEIPT.resolve()),
        "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
        "source_sha256": sha256_file(SOURCE),
        "input_sha256": sha256_file(INPUT),
        "summaries": summaries,
        "rows": graded_rows,
        "raw_artifacts": raw_artifacts,
        "commands": commands,
        "git": {
            "head": git_head,
            "branch": branch,
            "origin_main": origin,
            "preflight_status": preflight_status,
            "runner_observed_dirty_expected": True,
            "reason": "wrapper opens durable local artifacts before runner metadata",
        },
        "benchmark_data_read": False,
    }
    LOCAL_RECEIPT.write_text(
        json.dumps(local, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    sys.path.insert(0, str(EXP / "scripts"))
    from check_local import evaluate_promotion  # noqa: PLC0415

    promotion = evaluate_promotion(local)
    promotion.update(
        {
            "experiment_id": EXP.name,
            "local_receipt": str(LOCAL_RECEIPT.resolve()),
            "local_receipt_sha256": sha256_file(LOCAL_RECEIPT),
            "design_receipt_sha256": sha256_file(DESIGN_RECEIPT),
            "backend": "vllm_merged_composite",
            "aggregate_seed": AGGREGATE_SEED,
            "aggregate_seed_open": promotion["promoted"] is not None,
            "benchmark_data_read": False,
        }
    )
    PROMOTION_RECEIPT.write_text(
        json.dumps(promotion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(promotion, indent=2, sort_keys=True, ensure_ascii=False))
    if promotion["promoted"] is None:
        raise SystemExit(
            "no candidate passed frozen local promotion; aggregate seed stays sealed"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
