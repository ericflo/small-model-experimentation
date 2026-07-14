#!/usr/bin/env python3
"""Export hash-only parent collision domains without reading sampled outputs."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PARENT = ROOT / "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial"
OUTPUT = EXP / "runs/parent_lineage/collision_manifest.json"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"

ADMINISTRATIVE_SOURCES = {
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/configs/default.yaml": "6f3b065e471ed0f9b602816882993e98294d97527a539589a6f1bda08217a4d3",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/runs/construction/summary.json": "d507e9812af95256b7fe4c436d1fe851a3087360494a3b5333bde0eca187cb78",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/runs/prepared/preoutcome_receipt.json": "180058c6beba980c2c4b3a9dbe3f583231163474d393526773aecca84e23a263",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/construct.py": "79a7c2a87889255622b87ca06ef68603af22e69d739e1a78e7dac211a319cfbe",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/src/identity.py": "ed6a03b939e4f5b1512a29d4844840520422447da3134bafb0be047b572ba551",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/src/mechanics_protocol.py": "b2802a1278d660ba21d89c127429cf991c97a5415c7110da1d46861a260aeaa9",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/src/protocol.py": "628b1235bfa84e476b5bad62c899e8f2279c6a6edb0c5617f98d59af6ad297ec",
    "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/src/task_data.py": "f3a45c095dd0cb4e013402a89986240bc7877b1d3a9d91a1f7bc4b821f75c134",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_digest(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def verified_sources() -> dict[str, str]:
    observed: dict[str, str] = {}
    for relative, expected in sorted(ADMINISTRATIVE_SOURCES.items()):
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"parent administrative source is unsafe: {relative}")
        digest = sha256_file(path)
        if digest != expected:
            raise RuntimeError(f"parent administrative source changed: {relative}")
        observed[relative] = digest
    return observed


def load_parent_constructor() -> Any:
    path = PARENT / "scripts/construct.py"
    spec = importlib.util.spec_from_file_location("frozen_parent_construct", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen parent constructor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_repository_read_firewall() -> None:
    """Deny every repository read outside the exact administrative allowlist."""

    root = ROOT.resolve()
    environment_root = (ROOT / ".venv-vllm").resolve()
    allowed = {
        (ROOT / relative).resolve() for relative in ADMINISTRATIVE_SOURCES
    }
    allowed.add(OUTPUT.resolve())

    def audit(event: str, arguments: tuple[Any, ...]) -> None:
        if event != "open" or not arguments:
            return
        raw = arguments[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(raw).resolve()
        except (OSError, TypeError, ValueError):
            return
        if (
            path.is_relative_to(root)
            and not path.is_relative_to(environment_root)
            and path not in allowed
        ):
            raise PermissionError(
                f"parent collision export forbids undeclared repository read: {path}"
            )

    sys.addaudithook(audit)


def rendered_token_digest(tokenizer: Any, prompt: str, thinking: bool) -> str:
    token_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )
    if isinstance(token_ids, Mapping):
        token_ids = token_ids.get("input_ids")
    if not isinstance(token_ids, list) or any(type(value) is not int for value in token_ids):
        raise RuntimeError("parent rendered token sequence changed type")
    return canonical_digest(token_ids)


def build_manifest() -> dict[str, Any]:
    sources = verified_sources()
    install_repository_read_firewall()
    parent = load_parent_constructor()
    config = yaml.safe_load((PARENT / "configs/default.yaml").read_text())
    tasks, construction = parent.build_tasks(
        config, excluded_public_fingerprints=set()
    )
    data_rows, requests = parent.build_requests(tasks)

    recorded_construction = json.loads(
        (PARENT / "runs/construction/summary.json").read_text()
    )
    public_fingerprints = sorted(
        parent.public_instance_fingerprint(parent.public_task(task))
        for split in ("calibration", "mechanics")
        for task in tasks[split]
    )
    if public_fingerprints != recorded_construction["public_instance_fingerprints"]:
        raise RuntimeError("regenerated parent public fingerprints changed")
    if construction["common_panel_sha256"] != recorded_construction["common_panel_sha256"]:
        raise RuntimeError("regenerated parent common panel changed")

    recorded_preoutcome = json.loads(
        (PARENT / "runs/prepared/preoutcome_receipt.json").read_text()
    )
    request_name_to_relative = {
        name: (
            "experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/"
            f"runs/prepared/{name}_requests.jsonl"
        )
        for name in requests
    }
    for name, rows in requests.items():
        relative = request_name_to_relative[name]
        expected = recorded_preoutcome["request_files"][relative]
        payload = parent.jsonl_bytes(rows)
        if len(rows) != expected["rows"] or sha256_bytes(payload) != expected["sha256"]:
            raise RuntimeError(f"regenerated parent request inventory changed: {name}")

    all_requests = [row for rows in requests.values() for row in rows]
    unique_requests = {row["id"]: row for row in all_requests}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    # The three suffix representations deliberately share causal request IDs.
    # Prompt freshness must therefore range over every row, not ID representatives.
    prompt_values = sorted(
        {row["messages"][0]["content"] for row in all_requests}
    )
    prompt_token_digests = sorted(
        {
            rendered_token_digest(tokenizer, prompt, thinking)
            for prompt in prompt_values
            for thinking in (False, True)
        }
    )
    function_fingerprints = sorted(
        {
            task["common_fingerprint"]
            for split in ("calibration", "mechanics")
            for task in tasks[split]
        }
    )
    if len(function_fingerprints) != 72:
        raise RuntimeError("parent function fingerprint inventory changed")
    representative_rows = list(unique_requests.values())
    derived_seeds = sorted(parent.seed_inventory(representative_rows, config, parent=False))
    return {
        "administrative_sources": sources,
        "benchmark_files_read": [],
        "common_function_fingerprints": function_fingerprints,
        "derived_runner_seeds": derived_seeds,
        "hidden_files_read": [],
        "model": MODEL_ID,
        "model_calls": 0,
        "model_loaded": False,
        "parent_experiment": "qwen35_4b_tokenizer_eos_answer_commit_factorial",
        "parent_raw_sampled_bundles_read": [],
        "prompt_sha256s": sorted(sha256_bytes(value.encode("utf-8")) for value in prompt_values),
        "prompt_token_sequence_sha256s": prompt_token_digests,
        "public_instance_fingerprints": public_fingerprints,
        "request_ids": sorted(unique_requests),
        "revision": MODEL_REVISION,
        "sampled_model_outputs_read": 0,
        "schema_version": 1,
        "seed_key_sha256s": sorted(
            canonical_digest(row["meta"]["seed_key"])
            for row in unique_requests.values()
        ),
        "stage": "hash_only_parent_collision_export",
        "task_ids": sorted(
            task["task_id"]
            for split in ("calibration", "mechanics")
            for task in tasks[split]
        ),
        "tokenizer_loaded": True,
    }


def main() -> int:
    value = build_manifest()
    payload = canonical_bytes(value)
    if OUTPUT.is_symlink():
        raise RuntimeError("parent collision manifest is a symlink")
    if OUTPUT.exists():
        if not OUTPUT.is_file() or OUTPUT.read_bytes() != payload:
            raise RuntimeError("parent collision manifest changed")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_bytes(payload)
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
