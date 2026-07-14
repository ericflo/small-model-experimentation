#!/usr/bin/env python3
"""Run the frozen fresh local gate on three merged composites through vLLM."""

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
SOURCE = EXP / "data" / "local_tasks_seed88009.jsonl"
INPUT = EXP / "data" / "local_input_seed88009.jsonl"
DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
LOCAL_RECEIPT = EXP / "runs" / "local" / "seed88009.json"
PROMOTION_RECEIPT = EXP / "runs" / "local" / "seed88009_promotion.json"
SEED = 88009
ROWS = 26
MAX_TOKENS = 1024
LABELS = ("close_xi_parent", "replay_after_close", "prefix_repair_after_close")
MERGED = {
    label: ROOT / "large_artifacts" / EXP.name / "merged" / label for label in LABELS
}
PARENT_MERGE_RECEIPT = EXP / "runs" / "merges" / "close_xi_parent.json"
PARENT_MERGE_RECEIPT_SHA256 = (
    "10c3870deefb638fcbf2f7980fe39e35be4c08f0ab0cbabceafbf87a5231895b"
)
PARENT_EXTERNAL_RECEIPT_SHA256 = (
    "1fbc84b33dfbcef30ec0c7f5f184b5bebcbeb09f2a42033c5b47bd469b275557"
)
PARENT_WEIGHTS_SHA256 = (
    "4933f2dd2e5140597bf3e1dd4e9fecc700a3e3eb7d422b59b84fd0f6e28eb373"
)
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


def arm_paths(label: str) -> dict[str, Path]:
    stem = EXP / "runs" / "local" / f"seed88009_{label}"
    return {
        "output": stem.with_suffix(".jsonl"),
        "metadata": Path(str(stem) + ".meta.json"),
        "log": stem.with_suffix(".log"),
    }


def authenticate_design() -> dict:
    sys.path.insert(0, str(EXP / "scripts"))
    import gen_local_gate  # noqa: PLC0415

    expected = gen_local_gate.build_outputs()
    for path, value in expected.items():
        if not path.is_file() or path.read_bytes() != value or not committed_at_head(path):
            raise ValueError(f"local design is absent, changed, or uncommitted: {path}")
    receipt = load_json(DESIGN_RECEIPT)
    if (
        receipt.get("seed") != SEED
        or receipt.get("rows") != ROWS
        or receipt.get("runner_input", {}).get("sha256") != sha256_file(INPUT)
        or receipt.get("code_sha256", {}).get("runner") != sha256_file(RUNNER)
        or receipt.get("backend", {}).get("name") != "vllm_merged_composite"
    ):
        raise ValueError("local design receipt violates the frozen protocol")
    return receipt


def authenticate_models() -> dict[str, dict]:
    parent = load_json(PARENT_MERGE_RECEIPT)
    parent_external = MERGED["close_xi_parent"] / "merge_receipt.json"
    parent_weights = MERGED["close_xi_parent"] / "model.safetensors"
    if (
        not committed_at_head(PARENT_MERGE_RECEIPT)
        or sha256_file(PARENT_MERGE_RECEIPT) != PARENT_MERGE_RECEIPT_SHA256
        or parent.get("name") != "close_xi_parent"
        or Path(parent.get("merged", "")).resolve()
        != MERGED["close_xi_parent"].resolve()
        or sha256_file(parent_external) != PARENT_EXTERNAL_RECEIPT_SHA256
        or sha256_file(parent_weights) != PARENT_WEIGHTS_SHA256
    ):
        raise ValueError("published parent composite changed")

    sys.path.insert(0, str(EXP / "scripts"))
    from merge_trained_arm import validate_published_merge  # noqa: PLC0415

    return {
        "close_xi_parent": parent,
        "replay_after_close": validate_published_merge("replay_after_close"),
        "prefix_repair_after_close": validate_published_merge(
            "prefix_repair_after_close"
        ),
    }


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
    sampling = metadata.get("sampling", {})
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
        or sampling.get("thinking") != "natural"
        or sampling.get("n") != 1
        or sampling.get("max_tokens") != MAX_TOKENS
        or sampling.get("greedy") is not True
        or sampling.get("run_seed") != SEED
        or engine.get("max_model_len") != 4096
        or engine.get("max_num_seqs") != 16
        or engine.get("max_num_batched_tokens") != 8192
        or engine.get("cudagraph_capture_sizes") != [1, 2, 4, 8, 16]
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
    try:
        design = authenticate_design()
        model_receipts = authenticate_models()
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    git_head = run_text(["git", "rev-parse", "HEAD"])
    preflight_status = run_text(["git", "status", "--short"])
    if preflight_status:
        parser.error("fresh local event requires a clean committed worktree")
    all_paths = [path for label in LABELS for path in arm_paths(label).values()]
    if LOCAL_RECEIPT.exists() or PROMOTION_RECEIPT.exists() or any(
        path.exists() for path in all_paths
    ):
        parser.error("refusing to overwrite a local-evaluation artifact")

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
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
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
        normalize_log(paths["log"])
        if returncode != 0:
            raise SystemExit(
                f"local vLLM arm {label} failed with exit {returncode}; preserved artifacts"
            )
        raw_rows, metadata = validate_raw_arm(
            label, paths, git_head=git_head, input_ids=input_ids
        )
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
        raw_artifacts[label] = {
            "model": str(MERGED[label].resolve()),
            "model_receipt": str(
                PARENT_MERGE_RECEIPT.resolve()
                if label == "close_xi_parent"
                else (EXP / "runs" / "merges" / f"{label}.json").resolve()
            ),
            "model_receipt_sha256": (
                sha256_file(PARENT_MERGE_RECEIPT)
                if label == "close_xi_parent"
                else sha256_file(EXP / "runs" / "merges" / f"{label}.json")
            ),
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
            "aggregate_seed": 78139,
            "aggregate_seed_open": bool(promotion["eligible"]),
        }
    )
    PROMOTION_RECEIPT.write_text(
        json.dumps(promotion, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(promotion, indent=2, sort_keys=True, ensure_ascii=False))
    if not promotion["eligible"]:
        raise SystemExit("prefix-repair candidate failed frozen local promotion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
