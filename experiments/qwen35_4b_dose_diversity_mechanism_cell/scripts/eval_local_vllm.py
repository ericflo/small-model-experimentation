#!/usr/bin/env python3
"""Run the frozen two-instrument mechanism gate on four explicit composites.

One arm (``axis160_direct``) is this experiment's own trained merge; the other
three are inherited, externally published composites re-judged on the fresh
instrument. Grading applies the frozen answer normalization
(check_local.normalize_answer) to BOTH the parsed and the expected answer,
identically for every arm and both instruments, as documented in the frozen
local design receipt. This is a mechanism cell: a complete event always exits
0 — there is no promotion and no aggregate seed to open.
"""

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
SOURCE = EXP / "data" / "local_tasks_seed88020.jsonl"
INPUT = EXP / "data" / "local_input_seed88020.jsonl"
DESIGN_RECEIPT = EXP / "data" / "local_design_receipt.json"
LOCAL_RECEIPT = EXP / "runs" / "local" / "seed88020.json"
MECHANISM_RECEIPT = EXP / "runs" / "local" / "seed88020_mechanism.json"
FAILURE_RECEIPT = EXP / "runs" / "local" / "seed88020.failure.json"
SEED = 88020
ROWS = 144
MAX_TOKENS = 1024
LABELS = (
    "clean_parent",
    "hygiene_explore_direct",
    "replay_clean",
    "axis160_direct",
)
INHERITED_LABELS = ("clean_parent", "hygiene_explore_direct", "replay_clean")
CANDIDATE = "axis160_direct"
# The three inherited arms are externally published composites; their pins are
# constants filled at design freeze (mirroring the frozen local design
# receipt). Only the candidate's own tree pin is deferred.
PARENT_EXP = ROOT / "experiments" / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
DESTACK_EXP = ROOT / "experiments" / "qwen35_4b_hygiene_explore_destack_medium"
COMPOSITE_RECEIPTS = {
    "clean_parent": PARENT_EXP / "runs" / "merges" / "designed_fresh.json",
    "hygiene_explore_direct": DESTACK_EXP / "runs" / "merges" / "hygiene_explore.json",
    "replay_clean": DESTACK_EXP / "runs" / "merges" / "replay_clean.json",
}
EXPECTED_RECEIPT_SHA256 = {
    "clean_parent": "ab3f20cc93d3fe21ead7a1d573edbca2903d59d6f9fe3d2af0c93e823676acc2",
    "hygiene_explore_direct": "22a22a68234de68314064b809352e7449c59ef821235402b66ecb6e5ebcc486a",
    "replay_clean": "24367084da5415ec8c3f922202a2028cc2930c4b99d87ea5e32b93e5c3b90332",
}
EXPECTED_COMPOSITE_NAMES = {
    "clean_parent": "designed_fresh",
    "hygiene_explore_direct": "hygiene_explore",
    "replay_clean": "replay_clean",
}
EXPECTED_INHERITED_TREE_SHA256 = {
    "clean_parent": "93433aa2d5f3f0d6d4540126579c09feee1d8502df702c1563bae28eb7f60255",
    "hygiene_explore_direct": "9eb653d78f05546ca594a831c989fa906d12f3eb7a5a8550d1afcd6bfccc4971",
    "replay_clean": "19759e12b1301a15a2f9b2db311ffff7c08e3d8b0d3237c9e7cd718fa8dc7f67",
}
EXPECTED_WEIGHTS_SHA256 = {
    "clean_parent": "0a3b89cdf57ed8a73590580489d744319c12b44b60991db55b5baba6f7c27979",
    "hygiene_explore_direct": "e21123443a230ada2c73ded411e0b5b7c2b1459856b2c38e4f1beea8958dc02f",
    "replay_clean": "2cef3e5e7ddfbee5d2d2c3a878d64f360dc6994764b9856be7e319ce5187d0b4",
}
MERGED = {
    "clean_parent": (
        ROOT / "large_artifacts" / PARENT_EXP.name / "merged" / "designed_fresh"
    ),
    "hygiene_explore_direct": (
        ROOT / "large_artifacts" / DESTACK_EXP.name / "merged" / "hygiene_explore"
    ),
    "replay_clean": (
        ROOT / "large_artifacts" / DESTACK_EXP.name / "merged" / "replay_clean"
    ),
    "axis160_direct": ROOT / "large_artifacts" / EXP.name / "merged" / "axis160_direct",
}
# TODO-PIN: the orchestrator fills the candidate tree sha256 after
# merge_trained_arm.py publishes the merged composite; the eval refuses to run
# while the pin is still None.
EXPECTED_CANDIDATE_TREE_SHA256: dict[str, str | None] = {
    "axis160_direct": None,
}
ANSWER_RE = re.compile(r"(?:^|\n)ANSWER:\s*(.*?)(?=\n|<\||</|$)", re.DOTALL)

sys.path.insert(0, str(EXP / "scripts"))
from check_local import (  # noqa: E402
    ANSWER_NORMALIZATION,
    evaluate_mechanism,
    finalize_mechanism,
    normalize_answer,
)


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


def grade(parsed: str | None, expected: str) -> bool:
    """Frozen normalized comparison; identical for every arm and instrument."""
    return parsed is not None and normalize_answer(parsed) == normalize_answer(expected)


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
    import gen_local_gate  # noqa: PLC0415

    expected = gen_local_gate.build_outputs(authenticate_composites=False)
    for path, value in expected.items():
        if not path.is_file() or path.read_bytes() != value or not committed_at_head(path):
            raise ValueError(f"local design is absent, changed, or uncommitted: {path}")
    receipt = load_json(DESIGN_RECEIPT)
    if (
        receipt.get("seed") != SEED
        or receipt.get("rows") != ROWS
        or receipt.get("mechanism_cell") is not True
        or receipt.get("arms") != list(LABELS)
        or receipt.get("answer_normalization") != ANSWER_NORMALIZATION
        or receipt.get("runner_input", {}).get("sha256") != sha256_file(INPUT)
        or receipt.get("code_sha256", {}).get("runner") != sha256_file(RUNNER)
        or receipt.get("backend", {}).get("name") != "vllm_merged_composite"
        or receipt.get("firewall", {}).get("benchmark_data_read") is not False
        or receipt.get("firewall", {}).get("no_aggregate_seed_exists") is not True
    ):
        raise ValueError("local design receipt violates the frozen protocol")
    return receipt


def authenticate_model(label: str) -> dict:
    from gen_local_gate import (  # noqa: PLC0415
        merged_tree_manifest,
        tree_manifest_sha256,
    )

    if label in INHERITED_LABELS:
        receipt_path = COMPOSITE_RECEIPTS[label]
        if (
            not committed_at_head(receipt_path)
            or sha256_file(receipt_path) != EXPECTED_RECEIPT_SHA256[label]
        ):
            raise ValueError(f"published composite receipt changed: {label}")
        receipt = load_json(receipt_path)
        arm_files = merged_tree_manifest(MERGED[label])
        if (
            receipt.get("name") != EXPECTED_COMPOSITE_NAMES[label]
            or Path(receipt.get("merged", "")).resolve() != MERGED[label].resolve()
            or receipt.get("output_tree_sha256") != EXPECTED_INHERITED_TREE_SHA256[label]
            or receipt.get("weight_files")
            != [{"name": "model.safetensors", "sha256": EXPECTED_WEIGHTS_SHA256[label]}]
            or tree_manifest_sha256(arm_files) != EXPECTED_INHERITED_TREE_SHA256[label]
        ):
            raise ValueError(f"published {label} composite tree changed")
        return receipt
    if label != CANDIDATE:
        raise ValueError(f"unknown local arm: {label}")
    from merge_trained_arm import validate_published_merge  # noqa: PLC0415

    expected_tree = EXPECTED_CANDIDATE_TREE_SHA256[label]
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
        "correct_before_normalization": sum(
            row["correct_before_normalization"] for row in rows
        ),
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
        or MECHANISM_RECEIPT.exists()
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
                    "correct": grade(parsed, expected),
                    "correct_before_normalization": parsed == expected,
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
            COMPOSITE_RECEIPTS[label]
            if label in INHERITED_LABELS
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
        "stage": "dose_diversity_mechanism_two_instrument_local_gate",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "seed": SEED,
        "rows_per_arm": ROWS,
        "labels": list(LABELS),
        "backend": "vllm_merged_composite",
        "answer_normalization": ANSWER_NORMALIZATION,
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

    # finalize_mechanism is the single shared writer for both this receipt and
    # the check_local --out recovery path, so the two schemas cannot diverge.
    mechanism = finalize_mechanism(
        evaluate_mechanism(local), LOCAL_RECEIPT, LOCAL_RECEIPT.read_bytes()
    )
    MECHANISM_RECEIPT.write_text(
        json.dumps(mechanism, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(mechanism, indent=2, sort_keys=True, ensure_ascii=False))
    # Mechanism cell: any complete event exits 0 — the verdict lives in the
    # receipt and there is no aggregate seed to open.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
