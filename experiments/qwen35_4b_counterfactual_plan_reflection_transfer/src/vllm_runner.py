#!/usr/bin/env python3
"""Single-file, experiment-local vLLM runner for Qwen/Qwen3.5-4B.

Copy this file into an experiment's ``src/`` directory and either import
``VLLMRunner`` or execute it as a JSONL command-line tool.  It deliberately
has no repository-local imports.

Input JSONL rows have a unique ``id`` and exactly one of:

    {"id": "p1", "prompt": "an already rendered prompt"}
    {"id": "p2", "messages": [{"role": "user", "content": "..."}]}

An optional JSON-valued ``meta`` field is copied to the output.  The CLI emits
one row per input with an ``outputs`` list and writes a ``.meta.json`` sidecar.

Scientific invariants:

* The only accepted model is Qwen/Qwen3.5-4B at the pinned repository revision.
* Chat messages are rendered here, then exact prompt token IDs go to vLLM.
* Sampling parameters are always explicit; Hugging Face generation defaults
  are never inherited.
* Budgeted thinking defaults to the repository's historical two-stage
  force-close protocol, not vLLM's semantically different native budget.
* ``answer_max_tokens`` directly caps forced continuations; consumers that need
  a stage-independent allowance must gate natural ``n_answer_tokens`` too.
* Seed derivation is stable under input reordering and recorded per sample/stage;
  pre-Hopper sampled tokens are not batch-order invariant.
* vLLM asynchronous scheduling is disabled so the scheduler mode is explicit.
  This does not make pre-Hopper execution batch-invariant.
* Explicit CUDA-graph capture sizes are checked against vLLM's resolved list,
  endpoint, and full-decode mode before generation.
* vLLM and Transformers samples are not RNG-identical.  Never mix backends
  between experimental arms or matched-compute baselines.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

if __name__ == "__main__":
    if sys.flags.no_site != 1:
        raise SystemExit("vLLM generation must start with the pinned interpreter and -I -B -S")
    _EXP_FOR_BOOTSTRAP = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_EXP_FOR_BOOTSTRAP / "src"))
    from runtime_contract import bootstrap_runtime_environment

    bootstrap_runtime_environment(_EXP_FOR_BOOTSTRAP.parents[1], "vllm")


# vLLM's reproducibility guide requires the in-process V1 engine for offline
# deterministic scheduling.  This must be set before importing vllm.
_V1_MULTIPROCESSING = os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING")
if _V1_MULTIPROCESSING not in (None, "0"):
    raise RuntimeError(
        "VLLM_ENABLE_V1_MULTIPROCESSING must be unset or 0 for reproducible offline runs; "
        f"got {_V1_MULTIPROCESSING!r}"
    )
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Calling ``.venv-vllm/bin/python`` directly does not activate the venv, so
# console tools installed beside it (notably ninja, used by FlashInfer JIT)
# are otherwise invisible.  Keep direct invocation as reliable as activation.
_PYTHON_BIN = str(Path(sys.executable).parent)
if _PYTHON_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _PYTHON_BIN + os.pathsep + os.environ.get("PATH", "")

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
RUNNER_SCHEMA_VERSION = 6

_MAMBA_CACHE_BLOCKS_RE = re.compile(
    r"max_num_seqs\b.*?(?:exceeds?|greater\s+than|larger\s+than|more\s+than)"
    r".*?available.*?mamba.*?cache.*?blocks?\D+(\d+)",
    re.IGNORECASE | re.DOTALL,
)
_MAMBA_CACHE_REEXEC_HINT = (
    "lower --max-num-seqs or raise --gpu-memory-utilization; "
    "hybrid Qwen3.5 Mamba cache scales with GPU memory"
)

THINK_TEMPERATURE = 0.6
THINK_TOP_P = 0.95
THINK_TOP_K = 20
NO_THINK_TEMPERATURE = 0.7
NO_THINK_TOP_P = 0.8
NO_THINK_TOP_K = 20
MIN_NONZERO_TEMPERATURE = 0.01
MAX_TEMPERATURE = 2.0
MAX_N = 16_384
HF_MODEL_EOS_TOKEN_ID = 248044
TOKENIZER_EOS_TOKEN_ID = 248046
HF_MODEL_EOS_TOKEN = "<|endoftext|>"
TOKENIZER_EOS_TOKEN = "<|im_end|>"


def _validate_termination_ids(hf_eos_id: int, tokenizer_eos_id: int) -> None:
    """Keep Qwen's model stop token distinct from its tokenizer-declared EOS."""
    expected = (HF_MODEL_EOS_TOKEN_ID, TOKENIZER_EOS_TOKEN_ID)
    observed = (hf_eos_id, tokenizer_eos_id)
    if observed != expected:
        raise RuntimeError(
            "Qwen3.5 termination IDs changed; audit stopping and receipts: "
            f"expected model/tokenizer {expected}, observed {observed}"
        )


def _stable_seed(run_seed: int, record_id: str, sample_index: int, stage: str) -> int:
    payload = f"{run_seed}\0{record_id}\0{sample_index}\0{stage}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_tree(path: Path, excluded: set[str] | None = None) -> str:
    excluded = excluded or set()
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        relative = item.relative_to(path).as_posix()
        if relative in excluded:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(item)))
    return digest.hexdigest()


def _validate_model_override(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    receipt_path = path / "merge_receipt.json"
    if not receipt_path.is_file():
        raise ValueError("merged model override lacks merge_receipt.json")
    receipt = json.loads(receipt_path.read_text())
    required = {
        "schema_version", "experiment_id", "config_sha256", "model_id",
        "model_revision", "source_training_receipt_sha256",
        "source_stage_receipt_sha256", "source_tokenizer_receipt_sha256",
        "source_trainer_sha256", "source_trainer_git_commit",
        "source_recipe_sha256", "source_worktree", "source_runtime_sha256",
        "source_base_snapshot", "source_tokenizer_snapshot",
        "source_training_compute", "source_adapter_tree_sha256",
        "source_adapter_sha256", "source_adapter_config_sha256", "source_arm", "source_seed",
        "source_adapter_inventory", "applied_lora_modules",
        "merge_script_sha256", "merge_git_commit", "merge_worktree",
        "merge_runtime_bootstrap",
        "merge_base_snapshot", "merge_tokenizer_snapshot",
        "merged_checkpoint_inventory",
        "merge_contract_sha256", "tensor_merge", "merge_replay", "merged_tree_sha256",
    }
    config_path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    lineage = path / "source_lineage"
    adapter_tree = lineage / "adapter_tree"
    training_lineage_path = adapter_tree / "training_receipt.json"
    stage_lineage_path = adapter_tree / "source_stage_receipt.json"
    tokenizer_lineage_path = adapter_tree / "source_tokenizer_receipt.json"
    adapter_config_lineage_path = adapter_tree / "adapter_config.json"
    adapter_weights_lineage_path = adapter_tree / "adapter_model.safetensors"
    started_lineage_path = adapter_tree / "STARTED.json"
    if not all(
        item.is_file()
        for item in (
            training_lineage_path,
            stage_lineage_path,
            tokenizer_lineage_path,
            adapter_config_lineage_path,
            adapter_weights_lineage_path,
            started_lineage_path,
        )
    ):
        raise ValueError("merged model override lacks embedded source lineage")
    training_lineage = json.loads(training_lineage_path.read_text())
    tokenizer_lineage = json.loads(tokenizer_lineage_path.read_text())
    adapter_lineage = json.loads(adapter_config_lineage_path.read_text())
    import yaml

    config = yaml.safe_load(config_path.read_text())
    recipe = config["training"]["recipe"]
    merge_contract = config["merge"]
    recipe_sha256 = _sha256_bytes(
        json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode()
    )
    required_training_lineage = {
        "schema_version", "experiment_id", "arm", "seed", "model_id",
        "model_revision", "optimizer_steps", "train_loss", "config_sha256",
        "tokenizer_receipt_sha256", "stage_receipt_sha256",
        "copied_tokenizer_receipt_sha256", "copied_stage_receipt_sha256",
        "trainer_git_commit", "trainer_sha256", "recipe_sha256",
        "record_receipt_sha256", "parity_sha256",
        "adapter_tree_excluding_training_receipt_sha256",
        "worktree", "runtime", "base_snapshot", "tokenizer_snapshot", "compute",
        "load_window_guards",
    }
    required_tokenizer_lineage = {
        "schema_version", "experiment_id", "config_sha256", "runner_sha256",
        "model_id", "model_revision", "tokenizer_class", "tokenizer_eos_token_id",
        "trust_remote_code", "tokenizer_snapshot", "worktree", "record_receipt",
        "parity", "rows", "rows_sha256", "model_calls", "gpu_events",
        "benchmark_reads", "load_window_guard", "runtime_bootstrap",
    }
    training_runtime = training_lineage.get("runtime")
    required_training_runtime = {
        "schema_version",
        "bootstrap",
        "worktree",
        "python",
        "python_executable",
        "python_executable_sha256",
        "python_isolated",
        "python_dont_write_bytecode",
        "python_no_site",
        "platform",
        "packages",
        "packages_sha256",
        "environment_lock",
        "gpu",
        "cuda_toolkit",
    }
    if (
        not isinstance(training_runtime, dict)
        or set(training_runtime) != required_training_runtime
        or type(training_runtime.get("schema_version")) is not int
        or training_runtime["schema_version"] != 4
        or training_runtime.get("packages_sha256")
        != _sha256_bytes(
            json.dumps(
                training_runtime.get("packages"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
    ):
        raise ValueError("training receipt lacks exact runtime metadata")
    from provenance import (
        validate_gpu_identity,
        validate_interpreter_runtime,
        validate_runtime_bootstrap,
        validate_runtime_packages,
    )

    validate_runtime_packages(
        training_runtime,
        Path(__file__).resolve().parents[3] / "requirements-training.lock.txt",
        required_backend="training",
    )
    validate_interpreter_runtime(
        training_runtime,
        Path(__file__).resolve().parents[3],
    )
    validate_runtime_bootstrap(
        training_runtime,
        Path(__file__).resolve().parents[3],
        "training",
    )
    validate_gpu_identity(training_runtime.get("gpu"))
    from merge_replay import authenticate_base_snapshot, base_snapshot_commitment
    from tokenizer_lineage import (
        authenticate_tokenizer_snapshot,
        ensure_closed_tokenizer_view,
    )
    from load_window_guard import validate_load_window_receipt

    exact_base_root, _base_index, _base_structure = authenticate_base_snapshot()
    exact_base = base_snapshot_commitment(exact_base_root)
    exact_tokenizer_path, exact_tokenizer = ensure_closed_tokenizer_view()
    if authenticate_tokenizer_snapshot() != exact_tokenizer:
        raise ValueError("closed/source tokenizer commitments differ")
    exact_worktree = {
        "repo_root": _run_text(["git", "rev-parse", "--show-toplevel"]),
        "git_commit": _run_text(["git", "rev-parse", "HEAD"]),
        "head_mode": "detached",
        "cwd": str(Path.cwd().resolve()),
    }
    compute = training_lineage.get("compute")
    tokenizer_parity = tokenizer_lineage.get("parity")
    tokenizer_totals = (
        tokenizer_parity.get("totals") if isinstance(tokenizer_parity, dict) else None
    )
    training_arm_totals = (
        tokenizer_totals.get(training_lineage.get("arm"))
        if isinstance(tokenizer_totals, dict)
        else None
    )
    sealed_forward_tokens = (
        training_arm_totals.get("forward_tokens")
        if isinstance(training_arm_totals, dict)
        else None
    )
    valid_compute = (
        isinstance(compute, dict)
        and set(compute)
        == {
            "schema_version", "amortization_horizon", "forward_tokens",
            "forward_backward_multiplier", "token_forward_equivalents",
            "epochs", "model_load_seconds", "training_seconds", "gpu_phase_wall_seconds",
        }
        and type(compute.get("schema_version")) is int
        and compute.get("schema_version") == 1
        and compute.get("amortization_horizon")
        == "full_training_charged_to_each_confirmation_split"
        and type(compute.get("epochs")) is int
        and compute["epochs"] == 3
        and type(compute.get("forward_tokens")) is int
        and compute["forward_tokens"] > 0
        and type(sealed_forward_tokens) is int
        and compute["forward_tokens"] == sealed_forward_tokens * compute["epochs"]
        and training_lineage.get("parity_sha256")
        == _sha256_bytes(
            json.dumps(
                tokenizer_parity, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        and compute.get("forward_backward_multiplier") == 4
        and compute.get("token_forward_equivalents") == compute["forward_tokens"] * 4
        and type(compute.get("model_load_seconds")) in {int, float}
        and math.isfinite(compute["model_load_seconds"])
        and compute["model_load_seconds"] > 0
        and type(compute.get("training_seconds")) in {int, float}
        and math.isfinite(compute["training_seconds"])
        and compute["training_seconds"] > 0
        and type(compute.get("gpu_phase_wall_seconds")) in {int, float}
        and math.isfinite(compute["gpu_phase_wall_seconds"])
        and compute.get("gpu_phase_wall_seconds")
        == compute["model_load_seconds"] + compute["training_seconds"]
    )
    if (
        set(receipt) != required
        or receipt.get("schema_version") != 8
        or receipt.get("experiment_id")
        != "qwen35_4b_counterfactual_plan_reflection_transfer"
        or receipt.get("config_sha256") != _sha256_file(config_path)
        or receipt.get("model_id") != MODEL_ID
        or receipt.get("model_revision") != MODEL_REVISION
        or int(receipt.get("applied_lora_modules", 0)) < 1
        or receipt.get("source_arm")
        not in {
            "reflection_correct",
            "reflection_shuffled",
            "auxiliary_plan_label_correct",
            "direct_plan_answer_positive_control",
        }
        or receipt.get("source_seed") not in {47, 53}
        or (
            receipt.get("source_arm") == "direct_plan_answer_positive_control"
            and receipt.get("source_seed") != 47
        )
        or receipt.get("source_trainer_sha256")
        != _sha256_file(Path(__file__).resolve().parents[1] / "scripts" / "train.py")
        or receipt.get("source_trainer_git_commit")
        != _run_text(["git", "rev-parse", "HEAD"])
        or receipt.get("merge_git_commit") != _run_text(["git", "rev-parse", "HEAD"])
        or receipt.get("merge_script_sha256")
        != _sha256_file(Path(__file__).resolve().parents[1] / "scripts" / "merge_adapter.py")
        or receipt.get("source_recipe_sha256") != recipe_sha256
        or receipt.get("merge_contract_sha256")
        != _sha256_bytes(
            json.dumps(merge_contract, sort_keys=True, separators=(",", ":")).encode()
        )
        or receipt.get("source_training_receipt_sha256")
        != _sha256_file(training_lineage_path)
        or receipt.get("source_stage_receipt_sha256")
        != _sha256_file(stage_lineage_path)
        or receipt.get("source_tokenizer_receipt_sha256")
        != _sha256_file(tokenizer_lineage_path)
        or receipt.get("source_adapter_config_sha256")
        != _sha256_file(adapter_config_lineage_path)
        or training_lineage.get("experiment_id")
        != "qwen35_4b_counterfactual_plan_reflection_transfer"
        or set(training_lineage) != required_training_lineage
        or training_lineage.get("schema_version") != 6
        or training_lineage.get("config_sha256") != receipt.get("config_sha256")
        or training_lineage.get("model_id") != MODEL_ID
        or training_lineage.get("model_revision") != MODEL_REVISION
        or training_lineage.get("optimizer_steps") != 36
        or training_lineage.get("arm") != receipt.get("source_arm")
        or training_lineage.get("seed") != receipt.get("source_seed")
        or training_lineage.get("trainer_sha256")
        != receipt.get("source_trainer_sha256")
        or training_lineage.get("trainer_git_commit")
        != receipt.get("source_trainer_git_commit")
        or training_lineage.get("recipe_sha256")
        != receipt.get("source_recipe_sha256")
        or training_lineage.get("stage_receipt_sha256")
        != receipt.get("source_stage_receipt_sha256")
        or training_lineage.get("tokenizer_receipt_sha256")
        != receipt.get("source_tokenizer_receipt_sha256")
        or training_lineage.get("copied_stage_receipt_sha256")
        != receipt.get("source_stage_receipt_sha256")
        or training_lineage.get("copied_tokenizer_receipt_sha256")
        != receipt.get("source_tokenizer_receipt_sha256")
        or training_lineage.get("adapter_tree_excluding_training_receipt_sha256")
        != receipt.get("source_adapter_tree_sha256")
        or training_lineage.get("worktree") != exact_worktree
        or training_runtime.get("worktree") != exact_worktree
        or training_lineage.get("base_snapshot") != exact_base
        or training_lineage.get("tokenizer_snapshot") != exact_tokenizer
        or not valid_compute
        or receipt.get("source_worktree") != exact_worktree
        or receipt.get("merge_worktree") != exact_worktree
        or receipt.get("source_runtime_sha256")
        != _sha256_bytes(
            json.dumps(training_runtime, sort_keys=True, separators=(",", ":")).encode()
        )
        or receipt.get("source_base_snapshot") != exact_base
        or receipt.get("merge_base_snapshot") != exact_base
        or receipt.get("source_tokenizer_snapshot") != exact_tokenizer
        or receipt.get("merge_tokenizer_snapshot") != exact_tokenizer
        or receipt.get("source_training_compute") != compute
        or tokenizer_lineage.get("experiment_id")
        != "qwen35_4b_counterfactual_plan_reflection_transfer"
        or set(tokenizer_lineage) != required_tokenizer_lineage
        or tokenizer_lineage.get("schema_version") != 5
        or tokenizer_lineage.get("config_sha256") != receipt.get("config_sha256")
        or tokenizer_lineage.get("runner_sha256")
        != _sha256_file(Path(__file__).resolve().parents[1] / "scripts" / "tokenizer_receipt.py")
        or tokenizer_lineage.get("model_id") != MODEL_ID
        or tokenizer_lineage.get("model_revision") != MODEL_REVISION
        or tokenizer_lineage.get("tokenizer_eos_token_id") != 248046
        or tokenizer_lineage.get("tokenizer_class") != "Qwen2Tokenizer"
        or tokenizer_lineage.get("trust_remote_code") is not False
        or tokenizer_lineage.get("tokenizer_snapshot") != exact_tokenizer
        or tokenizer_lineage.get("worktree") != exact_worktree
        or tokenizer_lineage.get("rows_sha256")
        != _sha256_bytes(
            json.dumps(
                tokenizer_lineage.get("rows"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        or tokenizer_lineage.get("model_calls") != 0
        or tokenizer_lineage.get("gpu_events") != 0
        or tokenizer_lineage.get("benchmark_reads") != 0
        or int(adapter_lineage.get("r", -1)) != int(recipe["lora_rank"])
        or float(adapter_lineage.get("lora_alpha", -1)) != float(recipe["lora_alpha"])
        or float(adapter_lineage.get("lora_dropout", -1))
        != float(recipe["lora_dropout"])
        or str(adapter_lineage.get("bias")) != str(recipe["lora_bias"])
        or set(adapter_lineage.get("target_modules", []))
        != set(recipe["target_modules"])
    ):
        raise ValueError("merged model override receipt has invalid base or delta identity")
    load_guards = training_lineage.get("load_window_guards")
    if not isinstance(load_guards, dict) or set(load_guards) != {"tokenizer", "model"}:
        raise ValueError("training lineage lacks exact tokenizer/model load guards")
    validate_load_window_receipt(
        load_guards["tokenizer"],
        [exact_tokenizer_path],
        expected_content={"tokenizer": exact_tokenizer},
    )
    validate_load_window_receipt(
        load_guards["model"],
        [exact_base_root],
        expected_content={"base": exact_base},
    )
    validate_load_window_receipt(
        tokenizer_lineage["load_window_guard"],
        [exact_tokenizer_path],
        expected_content={"tokenizer": exact_tokenizer},
    )
    validate_runtime_bootstrap(
        {
            "bootstrap": tokenizer_lineage["runtime_bootstrap"],
            "worktree": tokenizer_lineage["worktree"],
        },
        Path(__file__).resolve().parents[3],
        "training",
    )
    validate_runtime_bootstrap(
        {
            "bootstrap": receipt["merge_runtime_bootstrap"],
            "worktree": receipt["merge_worktree"],
        },
        Path(__file__).resolve().parents[3],
        "training",
    )
    from checkpoint_lineage import adapter_tensor_inventory, merged_checkpoint_inventory

    source_inventory = adapter_tensor_inventory(adapter_weights_lineage_path)
    if (
        source_inventory != receipt.get("source_adapter_inventory")
        or source_inventory["sha256"] != receipt.get("source_adapter_sha256")
        or source_inventory["module_count"] != receipt.get("applied_lora_modules")
        or _sha256_tree(adapter_tree, excluded={"training_receipt.json"})
        != receipt.get("source_adapter_tree_sha256")
    ):
        raise ValueError("retained source adapter tensors/tree differ from merge lineage")
    checkpoint_inventory = merged_checkpoint_inventory(path)
    if checkpoint_inventory != receipt.get("merged_checkpoint_inventory"):
        raise ValueError("merged checkpoint inventory differs from its receipt")
    expected_tensor_merge = {
        "schema_version": 1,
        "implementation": merge_contract["implementation"],
        "shard_policy": merge_contract["shard_policy"],
        "shards": merge_contract["expected_shards"],
        "adapted_module_count": receipt["applied_lora_modules"],
        "unchanged_tensor_count": checkpoint_inventory["tensor_count"]
        - receipt["applied_lora_modules"],
        "equation": merge_contract["adapted_tensor_math"],
    }
    if receipt.get("tensor_merge") != expected_tensor_merge:
        raise ValueError("tensor-level merge receipt differs from the frozen merge contract")
    started = json.loads(started_lineage_path.read_text())
    if (
        set(started)
        != {
            "schema_version", "arm", "seed", "config_sha256",
            "tokenizer_receipt_sha256", "stage_receipt_sha256",
            "trainer_git_commit", "trainer_sha256", "worktree", "runtime_pending",
            "base_snapshot", "tokenizer_snapshot",
            "tokenizer_load_window_guard",
        }
        or started.get("schema_version") != 5
        or started.get("arm") != receipt.get("source_arm")
        or started.get("seed") != receipt.get("source_seed")
        or started.get("config_sha256") != receipt.get("config_sha256")
        or started.get("tokenizer_receipt_sha256")
        != receipt.get("source_tokenizer_receipt_sha256")
        or started.get("stage_receipt_sha256") != receipt.get("source_stage_receipt_sha256")
        or started.get("trainer_git_commit") != receipt.get("source_trainer_git_commit")
        or started.get("trainer_sha256") != receipt.get("source_trainer_sha256")
        or started.get("worktree") != exact_worktree
        or started.get("runtime_pending") is not True
        or started.get("base_snapshot") != exact_base
        or started.get("tokenizer_snapshot") != exact_tokenizer
        or started.get("tokenizer_load_window_guard") != load_guards["tokenizer"]
    ):
        raise ValueError("retained source adapter start record differs from merge lineage")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from stages import read_and_validate_stage_receipt

    read_and_validate_stage_receipt(
        stage_lineage_path,
        config=config,
        config_path=config_path,
        expected_stage=("screen_training" if receipt["source_seed"] == 47 else "replication_training"),
    )
    from merge_replay import verify_merge_equation

    observed_replay = verify_merge_equation(
        merged_path=path,
        adapter_tree=adapter_tree,
        recipe=recipe,
    )
    if observed_replay != receipt.get("merge_replay"):
        raise ValueError("merged tensors differ from exact pinned-base plus LoRA replay")
    if observed_replay["adapted_module_count"] != source_inventory["module_count"]:
        raise ValueError("merge replay module count differs from retained source adapter")
    observed = _sha256_tree(path, excluded={"merge_receipt.json"})
    if receipt.get("merged_tree_sha256") != observed:
        raise ValueError("merged model override tree hash differs from its receipt")
    return {
        "path": str(path.resolve()),
        "merge_receipt_sha256": _sha256_file(receipt_path),
        "merged_tree_sha256": observed,
        "source_training_receipt_sha256": receipt.get("source_training_receipt_sha256"),
        "source_stage_receipt_sha256": receipt.get("source_stage_receipt_sha256"),
        "source_tokenizer_receipt_sha256": receipt.get("source_tokenizer_receipt_sha256"),
        "source_trainer_sha256": receipt.get("source_trainer_sha256"),
        "source_trainer_git_commit": receipt.get("source_trainer_git_commit"),
        "source_runtime_sha256": receipt.get("source_runtime_sha256"),
        "source_training_gpu": training_runtime["gpu"],
        "source_recipe_sha256": receipt.get("source_recipe_sha256"),
        "source_adapter_tree_sha256": receipt.get("source_adapter_tree_sha256"),
        "source_adapter_config_sha256": receipt.get("source_adapter_config_sha256"),
        "source_adapter_inventory": source_inventory,
        "merged_checkpoint_inventory": checkpoint_inventory,
        "merge_replay_sha256": _sha256_bytes(
            json.dumps(observed_replay, sort_keys=True, separators=(",", ":")).encode()
        ),
        "source_arm": receipt.get("source_arm"),
        "source_seed": receipt.get("source_seed"),
    }


def _run_text(command: Sequence[str]) -> str:
    if not command or command[0] not in {"git", "uv", "nvcc"}:
        raise ValueError("runner subprocess is not an allowlisted pinned executable")
    from runtime_contract import _run_preauthenticated_git, run_pinned_executable

    try:
        result = (
            _run_preauthenticated_git(list(command[1:]), cwd=Path.cwd())
            if command[0] == "git"
            else run_pinned_executable(command[0], list(command[1:]), cwd=Path.cwd())
        )
        return result.stdout.strip() if result.returncode == 0 else ""
    except OSError:
        return ""


def _available_mamba_cache_blocks(exc: BaseException) -> int | None:
    """Return vLLM's reported Mamba cache block count for the known hybrid-arch cap."""
    pending: list[BaseException | None] = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop(0)
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        texts = [str(current), *(str(arg) for arg in current.args)]
        for text in texts:
            match = _MAMBA_CACHE_BLOCKS_RE.search(text)
            if match:
                return int(match.group(1))
        pending.extend((current.__cause__, current.__context__))
    return None


def _installed_packages() -> dict[str, str]:
    """Return the full distribution inventory needed to reproduce kernel/runtime state."""
    # Enumerate names for a complete inventory, but resolve each version through
    # importlib's normal distribution lookup.  A last-write-wins scan is unsafe:
    # importing setuptools can expose its vendored dist-info directories after
    # the real site-packages entry (for example packaging 26.0 after 26.2).
    distribution_names: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            normalized = name.lower().replace("_", "-")
            distribution_names.setdefault(normalized, name)
    packages = {
        normalized: importlib.metadata.version(name)
        for normalized, name in distribution_names.items()
    }
    return dict(sorted(packages.items()))


# Importing vLLM adds setuptools' vendored distributions to sys.path. Snapshot
# the actual environment first so the sidecar matches the uv lock instead of
# reporting those implementation-detail copies as separately installed wheels.
_INITIAL_PACKAGES = _installed_packages()


def _environment_lock_metadata() -> dict[str, str] | None:
    """Find the repository lock even after this file is copied into an experiment."""
    starts = (Path(__file__).resolve().parent, Path.cwd().resolve())
    visited: set[Path] = set()
    for start in starts:
        for directory in (start, *start.parents):
            if directory in visited:
                continue
            visited.add(directory)
            lock = directory / "requirements-vllm.lock.txt"
            if lock.is_file():
                return {
                    "path": str(lock),
                    "sha256": _sha256_file(lock),
                }
    return None


def _jsonable_logprobs(value: Any) -> Any:
    """Convert vLLM's nested Logprob objects without importing private types."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable_logprobs(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable_logprobs(item) for item in value]
    if hasattr(value, "logprob"):
        return {
            "logprob": float(value.logprob),
            "rank": getattr(value, "rank", None),
            "decoded_token": getattr(value, "decoded_token", None),
        }
    raise TypeError(f"cannot serialize logprob object of type {type(value).__name__}")


def _resolved_cudagraph_metadata(compilation_config: Any) -> dict[str, Any]:
    """Read the effective CUDA-graph geometry from vLLM's resolved config."""
    try:
        sizes = tuple(
            int(size) for size in compilation_config.cudagraph_capture_sizes
        )
        maximum = int(compilation_config.max_cudagraph_capture_size)
        mode = compilation_config.cudagraph_mode
        mode_name = str(mode.name)
        decode_mode = str(mode.decode_mode().name)
        mixed_mode = str(mode.mixed_mode().name)
        has_full = bool(mode.has_full_cudagraphs())
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "vLLM did not expose resolved CUDA-graph geometry"
        ) from exc
    return {
        "source": "llm_engine.vllm_config.compilation_config",
        "cudagraph_capture_sizes": list(sizes),
        "max_cudagraph_capture_size": maximum,
        "mode": mode_name,
        "decode_mode": decode_mode,
        "mixed_mode": mixed_mode,
        "has_full_cudagraphs": has_full,
    }


def _validate_explicit_cudagraph_resolution(
    requested: tuple[int, ...], resolved: dict[str, Any]
) -> None:
    """Fail closed unless vLLM honored an explicit full-decode graph request."""
    resolved_sizes = tuple(int(size) for size in resolved["cudagraph_capture_sizes"])
    resolved_maximum = int(resolved["max_cudagraph_capture_size"])
    mode_name = str(resolved["mode"])
    decode_mode = str(resolved["decode_mode"])
    mixed_mode = str(resolved["mixed_mode"])
    has_full = bool(resolved["has_full_cudagraphs"])
    supported_mode_geometry = {
        "FULL": ("FULL", "FULL"),
        "FULL_DECODE_ONLY": ("FULL", "NONE"),
        "FULL_AND_PIECEWISE": ("FULL", "PIECEWISE"),
    }
    geometry_matches = (
        resolved_sizes == requested and resolved_maximum == requested[-1]
    )
    mode_matches = (
        mode_name in supported_mode_geometry
        and (decode_mode, mixed_mode) == supported_mode_geometry[mode_name]
        and has_full
    )
    if not geometry_matches or not mode_matches:
        raise RuntimeError(
            "vLLM did not honor the explicit full-decode CUDA-graph request: "
            f"requested={requested}, resolved={resolved_sizes}, "
            f"resolved_max={resolved_maximum}, mode={mode_name}, "
            f"decode_mode={decode_mode}, mixed_mode={mixed_mode}, "
            f"has_full_cudagraphs={has_full}"
        )


def _engine_config_metadata(config: "EngineConfig") -> dict[str, Any]:
    return {
        key: str(value.resolve()) if isinstance(value, Path) else value
        for key, value in dataclasses.asdict(config).items()
    }


def _validate_live_cache_geometry(
    cache: dict[str, Any], config: "EngineConfig"
) -> dict[str, int]:
    """Authenticate pinned vLLM/Qwen hybrid-cache counters without float inversion."""
    required = {
        "num_gpu_blocks",
        "block_size",
        "kv_cache_size_tokens",
        "kv_cache_max_concurrency",
        "enable_prefix_caching",
        "mamba_cache_mode",
        "mamba_block_size",
    }
    if not isinstance(cache, dict) or set(cache) != required:
        raise RuntimeError("live preflight cache geometry schema changed")
    num_blocks = cache["num_gpu_blocks"]
    block_size = cache["block_size"]
    capacity = cache["kv_cache_size_tokens"]
    concurrency_value = cache["kv_cache_max_concurrency"]
    mamba_block_size = cache["mamba_block_size"]
    if (
        not isinstance(num_blocks, int)
        or isinstance(num_blocks, bool)
        or num_blocks < config.max_num_seqs
        or not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
        or not isinstance(capacity, int)
        or isinstance(capacity, bool)
        or capacity < 1
        or isinstance(concurrency_value, bool)
        or not isinstance(concurrency_value, (int, float))
        or not math.isfinite(float(concurrency_value))
        or float(concurrency_value) < config.max_num_seqs
        or not isinstance(mamba_block_size, int)
        or isinstance(mamba_block_size, bool)
        or mamba_block_size < 1
        or cache["enable_prefix_caching"] is not config.enable_prefix_caching
        or cache["mamba_cache_mode"]
        != ("align" if config.enable_prefix_caching else "none")
    ):
        raise RuntimeError("live preflight cache geometry changed")
    concurrency = float(concurrency_value)
    mamba_group_count = 3
    attention_blocks_at_max = math.ceil(config.max_model_len / block_size)
    mamba_blocks_at_max = mamba_group_count * math.ceil(
        config.max_model_len / mamba_block_size
    )
    blocks_per_max_request = attention_blocks_at_max + mamba_blocks_at_max
    if (
        int(concurrency * config.max_model_len) != capacity
        or not math.isclose(
            concurrency,
            num_blocks / blocks_per_max_request,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or block_size != 528
        or mamba_block_size != config.max_model_len
        or blocks_per_max_request != 11
    ):
        raise RuntimeError("live preflight cache geometry changed")
    return {
        "blocks_per_max_request": blocks_per_max_request,
        "attention_blocks_at_max": attention_blocks_at_max,
        "mamba_blocks_at_max": mamba_blocks_at_max,
        "mamba_group_count": mamba_group_count,
    }


def _read_live_engine_geometry(llm: Any, config: "EngineConfig") -> dict[str, Any]:
    try:
        vllm_config = llm.llm_engine.vllm_config
        cache = vllm_config.cache_config
        scheduler = vllm_config.scheduler_config
        model = vllm_config.model_config
        parallel = vllm_config.parallel_config
        cache_receipt = {
            key: getattr(cache, key)
            for key in (
                "num_gpu_blocks",
                "block_size",
                "kv_cache_size_tokens",
                "kv_cache_max_concurrency",
                "enable_prefix_caching",
                "mamba_cache_mode",
                "mamba_block_size",
            )
        }
        if (
            int(model.max_model_len) != config.max_model_len
            or str(model.dtype) not in {"bfloat16", "torch.bfloat16"}
            or int(scheduler.max_num_seqs) != config.max_num_seqs
            or int(scheduler.max_num_batched_tokens) != config.max_num_batched_tokens
            or bool(scheduler.async_scheduling)
            or int(parallel.world_size) != 1
            or int(parallel.tensor_parallel_size) != 1
            or int(parallel.data_parallel_size) != 1
        ):
            raise RuntimeError("live engine geometry differs from frozen protocol")
        cache_shape = _validate_live_cache_geometry(cache_receipt, config)
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("live engine did not expose the pinned capacity geometry") from error
    return {
        "live_model": {
            "max_model_len": int(model.max_model_len),
            "dtype": str(model.dtype),
        },
        "live_scheduler": {
            "max_num_seqs": int(scheduler.max_num_seqs),
            "max_num_batched_tokens": int(scheduler.max_num_batched_tokens),
            "async_scheduling": bool(scheduler.async_scheduling),
        },
        "live_parallel": {
            "world_size": int(parallel.world_size),
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "data_parallel_size": int(parallel.data_parallel_size),
        },
        "live_cache": cache_receipt,
        "cache_shape": cache_shape,
    }


def _capacity_preflight(
    *,
    live: dict[str, Any],
    config: "EngineConfig",
    prompt_lengths: Sequence[int],
    sampling: "SamplingConfig",
    close_tokens: int,
) -> dict[str, Any]:
    if not prompt_lengths:
        raise ValueError("capacity preflight requires prompts")
    reserve = (
        int(sampling.thinking_budget) + close_tokens + sampling.answer_max_tokens
        if sampling.thinking == "budget"
        else sampling.max_tokens
    )
    maximum_total = max(prompt_lengths) + reserve
    active = min(len(prompt_lengths) * sampling.n, config.max_num_seqs)
    block_size = int(live["live_cache"]["block_size"])
    rounded_reservation = math.ceil(maximum_total / block_size) * block_size
    required_tokens = active * rounded_reservation
    capacity_tokens = int(live["live_cache"]["kv_cache_size_tokens"])
    blocks_per_request = int(live["cache_shape"]["blocks_per_max_request"])
    required_blocks = active * blocks_per_request
    available_blocks = int(live["live_cache"]["num_gpu_blocks"])
    if (
        maximum_total > config.max_model_len
        or required_tokens > capacity_tokens
        or required_blocks > available_blocks
    ):
        raise RuntimeError(
            "live KV capacity cannot fit the frozen prompt/generation reservation"
        )
    return {
        "schema_version": 1,
        "decision": "LIVE_KV_CAPACITY_PASS",
        "engine": _engine_config_metadata(config),
        **live,
        "invocation": {
            "requests": len(prompt_lengths),
            "logical_sequences": len(prompt_lengths) * sampling.n,
            "active_sequences": active,
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
            "generation_reserve_tokens": reserve,
            "max_prompt_plus_reserve": maximum_total,
            "attention_rounding_block_tokens": block_size,
            "rounded_sequence_reservation_tokens": rounded_reservation,
            "required_cache_tokens": required_tokens,
            "available_cache_tokens": capacity_tokens,
            "remaining_cache_tokens": capacity_tokens - required_tokens,
            "reserved_blocks_per_sequence": blocks_per_request,
            "required_cache_blocks": required_blocks,
            "available_cache_blocks": available_blocks,
            "remaining_cache_blocks": available_blocks - required_blocks,
        },
    }


@dataclasses.dataclass(frozen=True)
class EngineConfig:
    """Engine settings chosen once, before the model is loaded."""

    max_model_len: int = 16_384
    gpu_memory_utilization: float = 0.90
    max_num_seqs: int = 128
    max_num_batched_tokens: int = 32_768
    enable_prefix_caching: bool = False
    enforce_eager: bool = False
    adapter: Path | None = None
    model_override: Path | None = None
    cudagraph_capture_sizes: tuple[int, ...] | None = None

    def validate(self) -> None:
        if self.max_model_len < 256:
            raise ValueError("max_model_len must be at least 256")
        if not 0.1 <= self.gpu_memory_utilization < 1.0:
            raise ValueError("gpu_memory_utilization must be in [0.1, 1.0)")
        if self.max_num_seqs < 1 or self.max_num_batched_tokens < 1:
            raise ValueError("max_num_seqs and max_num_batched_tokens must be positive")
        if self.adapter is not None and self.model_override is not None:
            raise ValueError("adapter and model_override are mutually exclusive")
        if self.adapter is not None:
            raise ValueError(
                "runtime LoRA adapters are forbidden for Qwen3.5; use a receipt-bound merged model"
            )
        if self.model_override is not None and not self.model_override.is_dir():
            raise ValueError("model_override must be an existing merged-checkpoint directory")
        if self.cudagraph_capture_sizes is not None:
            sizes = self.cudagraph_capture_sizes
            if not sizes or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 1
                for size in sizes
            ):
                raise ValueError(
                    "cudagraph_capture_sizes must contain positive integers"
                )
            if tuple(sorted(set(sizes))) != sizes:
                raise ValueError(
                    "cudagraph_capture_sizes must be strictly increasing and unique"
                )
            if sizes[-1] != self.max_num_seqs:
                raise ValueError(
                    "the largest cudagraph capture size must equal max_num_seqs"
                )
            if self.enforce_eager:
                raise ValueError(
                    "explicit cudagraph capture sizes are incompatible with enforce_eager"
                )


@dataclasses.dataclass(frozen=True)
class SamplingConfig:
    """Explicit generation settings shared by a run."""

    thinking: str = "off"  # off | natural | budget
    thinking_budget: int | None = None
    n: int = 1
    max_tokens: int = 512
    answer_max_tokens: int = 512
    greedy: bool = False
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    run_seed: int = 0
    shuffle_thinking: bool = False
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    logprob_token_ids: tuple[int, ...] = ()
    allow_custom_prompts: bool = False

    def validate(self) -> None:
        if self.thinking not in {"off", "natural", "budget"}:
            raise ValueError("thinking must be one of: off, natural, budget")
        if self.thinking == "budget":
            if self.thinking_budget is None or self.thinking_budget < 1:
                raise ValueError("budget thinking requires thinking_budget >= 1")
        elif self.thinking_budget is not None:
            raise ValueError("thinking_budget is only valid with thinking='budget'")
        if self.n < 1:
            raise ValueError("n must be positive")
        if self.n > MAX_N:
            raise ValueError(f"n cannot exceed this runner's vLLM limit of {MAX_N}")
        if self.greedy and self.n != 1:
            raise ValueError("greedy generation requires n=1")
        if self.max_tokens < 1 or self.answer_max_tokens < 1:
            raise ValueError("token caps must be positive")
        if self.shuffle_thinking and self.thinking != "budget":
            raise ValueError("shuffle_thinking is only valid for budget thinking")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be non-negative")
        if self.logprobs is not None and self.logprobs > 20:
            raise ValueError("logprobs cannot exceed this runner's vLLM max_logprobs=20")
        if self.prompt_logprobs is not None and self.prompt_logprobs < 0:
            raise ValueError("prompt_logprobs must be non-negative")
        if self.prompt_logprobs is not None and self.prompt_logprobs > 20:
            raise ValueError(
                "prompt_logprobs cannot exceed this runner's vLLM max_logprobs=20"
            )
        if self.prompt_logprobs is not None and self.thinking == "budget":
            raise ValueError(
                "prompt_logprobs with two-stage budget thinking are not yet supported; "
                "score the completed sequences in a separate pass"
            )
        if self.logprob_token_ids and self.logprobs != len(self.logprob_token_ids):
            raise ValueError(
                "vLLM requires logprobs to equal the number of logprob_token_ids; "
                f"got logprobs={self.logprobs!r} and {len(self.logprob_token_ids)} IDs"
            )
        resolved = self.resolved_sampling()
        finite_values = {
            "temperature": resolved["temperature"],
            "top_p": resolved["top_p"],
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }
        for name, value in finite_values.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if float(resolved["temperature"]) < 0:
            raise ValueError("temperature must be non-negative")
        if 0 < float(resolved["temperature"]) < MIN_NONZERO_TEMPERATURE:
            raise ValueError(
                f"nonzero temperature must be at least {MIN_NONZERO_TEMPERATURE}; "
                "vLLM silently clamps smaller values"
            )
        if float(resolved["temperature"]) > MAX_TEMPERATURE:
            raise ValueError(f"temperature must not exceed {MAX_TEMPERATURE}")
        if not 0 < float(resolved["top_p"]) <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if int(resolved["top_k"]) < -1:
            raise ValueError("top_k must be 0/-1 (disabled) or positive")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be positive")
        if float(resolved["temperature"]) == 0 and self.n != 1:
            raise ValueError("effective greedy generation (temperature=0) requires n=1")

    def resolved_sampling(self) -> dict[str, float | int]:
        if self.greedy or self.temperature == 0:
            return {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 0,
                "min_p": 0.0,
                "presence_penalty": self.presence_penalty,
                "frequency_penalty": self.frequency_penalty,
                "repetition_penalty": self.repetition_penalty,
            }
        thinking = self.thinking != "off"
        return {
            "temperature": self.temperature
            if self.temperature is not None
            else (THINK_TEMPERATURE if thinking else NO_THINK_TEMPERATURE),
            "top_p": self.top_p
            if self.top_p is not None
            else (THINK_TOP_P if thinking else NO_THINK_TOP_P),
            "top_k": self.top_k
            if self.top_k is not None
            else (THINK_TOP_K if thinking else NO_THINK_TOP_K),
            "min_p": self.min_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclasses.dataclass(frozen=True)
class _PreparedRecord:
    record_id: str
    meta: Any
    prompt_text: str
    prompt_token_ids: list[int]
    prompt_channel: str


def _validate_adapter(adapter: Path | None) -> dict[str, Any] | None:
    if adapter is None:
        return None
    adapter = adapter.expanduser().resolve()
    config_path = adapter / "adapter_config.json"
    weights = sorted(adapter.glob("*.safetensors"))
    if not config_path.is_file():
        raise ValueError(f"adapter is missing adapter_config.json: {adapter}")
    if not weights:
        raise ValueError(f"adapter has no .safetensors weights: {adapter}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("peft_type", "")).upper() != "LORA":
        raise ValueError("only PEFT LoRA adapters are supported")
    base = str(config.get("base_model_name_or_path", ""))
    accepted_base = (
        not base
        or base == MODEL_ID
        or "models--Qwen--Qwen3.5-4B" in base
        or MODEL_REVISION in base
    )
    if not accepted_base:
        raise ValueError(f"adapter targets a different base model: {base!r}")
    if config.get("use_dora"):
        raise ValueError("DoRA adapters are not supported by this runner")
    if config.get("modules_to_save"):
        raise ValueError("adapters with modules_to_save are not supported")
    if str(config.get("bias", "none")).lower() != "none":
        raise ValueError("LoRA bias weights are not supported")
    if config.get("rank_pattern"):
        raise ValueError("per-module rank_pattern adapters are not supported")
    if config.get("alpha_pattern"):
        raise ValueError("per-module alpha_pattern adapters are not supported")
    rank = int(config.get("r", 0))
    if rank < 1:
        raise ValueError("adapter rank r must be positive")
    if rank not in {1, 8, 16, 32, 64, 128, 256, 320, 512}:
        raise ValueError(f"adapter rank {rank} is not supported by vLLM 0.24")
    digest = hashlib.sha256()
    for path in weights:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return {
        "path": str(adapter),
        "rank": rank,
        "base_model_name_or_path": base,
        "target_modules": config.get("target_modules"),
        "config_sha256": _sha256_file(config_path),
        "weights_sha256": digest.hexdigest(),
    }


class VLLMRunner:
    """Importable high-throughput runner for text generation and PEFT LoRA."""

    def __init__(self, config: EngineConfig = EngineConfig()):
        config.validate()
        obsolete_reexec = sorted(
            name
            for name in (
                "QWEN_RUNNER_MAMBA_REEXEC",
                "QWEN_RUNNER_MAMBA_REEXEC_CUDAGRAPH",
            )
            if os.environ.get(name) is not None
        )
        if obsolete_reexec:
            raise RuntimeError(
                "adaptive Mamba geometry is forbidden; unset obsolete re-exec state: "
                + ", ".join(obsolete_reexec)
            )
        self.config = config
        from merge_replay import authenticate_base_snapshot, base_snapshot_commitment
        from load_window_guard import LoadWindowGuard
        from tokenizer_lineage import (
            authenticate_closed_tokenizer_view,
            authenticate_tokenizer_snapshot,
            ensure_closed_tokenizer_view,
        )

        (
            self.base_snapshot_path,
            _base_index,
            _base_structure,
        ) = authenticate_base_snapshot()
        self.base_snapshot_commitment = base_snapshot_commitment(
            self.base_snapshot_path
        )
        (
            self.closed_tokenizer_path,
            self.tokenizer_snapshot_commitment,
        ) = ensure_closed_tokenizer_view()
        self.adapter_info = _validate_adapter(config.adapter)
        self.model_override_info = _validate_model_override(config.model_override)
        self._closed = False

        # In-process vLLM intentionally seeds Python, NumPy, and Torch. Capture
        # the caller's streams before any model/tokenizer setup so constructing
        # this importable runner cannot alter later procedural task generation.
        python_rng_state = random.getstate()
        import numpy as np
        import torch

        numpy_rng_state = np.random.get_state()
        torch_rng_state = torch.random.get_rng_state()
        torch_initial_seed = torch.initial_seed()
        cuda_was_initialized = torch.cuda.is_initialized()
        cuda_rng_states = (
            torch.cuda.get_rng_state_all() if cuda_was_initialized else None
        )

        # Tokenize ourselves so the exact input IDs are auditable and do not
        # depend on vLLM's chat endpoint or reasoning parser.
        from transformers import AutoConfig, Qwen2Tokenizer

        tokenizer_config_content = {
            "base": self.base_snapshot_commitment,
            "tokenizer": self.tokenizer_snapshot_commitment,
        }
        with LoadWindowGuard(
            [self.base_snapshot_path, self.closed_tokenizer_path],
            expected_content=tokenizer_config_content,
        ) as tokenizer_config_guard:
            before_tokenizer_config_content = {
                "base": base_snapshot_commitment(self.base_snapshot_path),
                "tokenizer": authenticate_closed_tokenizer_view(
                    self.closed_tokenizer_path
                ),
            }
            self.tokenizer = Qwen2Tokenizer.from_pretrained(
                str(self.closed_tokenizer_path),
                trust_remote_code=False,
                local_files_only=True,
            )
            after_tokenizer_config_content = {
                "base": base_snapshot_commitment(self.base_snapshot_path),
                "tokenizer": authenticate_closed_tokenizer_view(
                    self.closed_tokenizer_path
                ),
            }
            tokenizer_config_guard.bind_authenticated_content(
                before_tokenizer_config_content, after_tokenizer_config_content
            )
            model_config = AutoConfig.from_pretrained(
                str(self.base_snapshot_path),
                trust_remote_code=False,
                local_files_only=True,
            )
        tokenizer_config_guard_receipt = tokenizer_config_guard.receipt
        if tokenizer_config_guard_receipt is None:
            raise RuntimeError("tokenizer/config load-window guard emitted no receipt")
        if (
            authenticate_tokenizer_snapshot()
            != self.tokenizer_snapshot_commitment
            or authenticate_closed_tokenizer_view(self.closed_tokenizer_path)
            != self.tokenizer_snapshot_commitment
        ):
            raise RuntimeError("tokenizer files changed across runner tokenizer initialization")
        self.hf_eos_id = int(model_config.text_config.eos_token_id)
        self.tokenizer_eos_id = int(self.tokenizer.eos_token_id)
        _validate_termination_ids(self.hf_eos_id, self.tokenizer_eos_id)
        if (
            self.tokenizer.eos_token != TOKENIZER_EOS_TOKEN
            or self.tokenizer.encode(HF_MODEL_EOS_TOKEN, add_special_tokens=False)
            != [HF_MODEL_EOS_TOKEN_ID]
            or self.tokenizer.encode(TOKENIZER_EOS_TOKEN, add_special_tokens=False)
            != [TOKENIZER_EOS_TOKEN_ID]
        ):
            raise RuntimeError(
                "Qwen3.5 termination token strings changed; audit stopping and receipts"
            )
        self.think_open_id = self._single_token_id("<think>")
        self.think_close_id = self._single_token_id("</think>")
        self.close_ids = self.tokenizer.encode(
            "</think>\n\n", add_special_tokens=False
        )
        if not self.close_ids or self.close_ids[0] != self.think_close_id:
            raise RuntimeError("unexpected Qwen3.5 </think> tokenization")
        if (self.think_open_id, self.think_close_id) != (248068, 248069):
            raise RuntimeError(
                "Qwen3.5 think-token IDs changed; audit prompts before continuing: "
                f"{self.think_open_id}, {self.think_close_id}"
            )

        self.thinking_prompt_suffix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n", add_special_tokens=False
        )
        self.no_thinking_prompt_suffix_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
            add_special_tokens=False,
        )

        engine_model = str(
            config.model_override.resolve()
            if config.model_override
            else self.base_snapshot_path
        )
        engine_args: dict[str, Any] = {
            "model": engine_model,
            "tokenizer": str(self.closed_tokenizer_path),
            "trust_remote_code": False,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "max_model_len": config.max_model_len,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_num_seqs": config.max_num_seqs,
            "max_num_batched_tokens": config.max_num_batched_tokens,
            "language_model_only": True,
            "enable_prefix_caching": config.enable_prefix_caching,
            "mamba_cache_mode": "align" if config.enable_prefix_caching else "none",
            "enforce_eager": config.enforce_eager,
            "generation_config": "vllm",
            "max_logprobs": 20,
            "seed": 0,
            # vLLM 0.24 auto-enables async scheduling.  Keep the simpler
            # synchronous mode explicit for offline research.  On pre-Hopper
            # GPUs, fixed request seeds still do not imply batch-invariant or
            # cross-budget prefix-identical samples.
            "async_scheduling": False,
        }
        if config.cudagraph_capture_sizes is None:
            engine_args["max_cudagraph_capture_size"] = config.max_num_seqs
        else:
            engine_args.update(
                cudagraph_capture_sizes=list(config.cudagraph_capture_sizes),
                max_cudagraph_capture_size=config.cudagraph_capture_sizes[-1],
            )
        if self.adapter_info is not None:
            engine_args.update(
                enable_lora=True,
                max_loras=1,
                max_cpu_loras=1,
                max_lora_rank=self.adapter_info["rank"],
            )

        try:
            from vllm import LLM

            started = time.perf_counter()
            engine_load_roots = [
                self.closed_tokenizer_path,
                config.model_override.resolve()
                if config.model_override is not None
                else self.base_snapshot_path,
            ]
            engine_expected_content = {
                "tokenizer": self.tokenizer_snapshot_commitment,
                "engine_model": (
                    self.model_override_info
                    if config.model_override is not None
                    else self.base_snapshot_commitment
                ),
            }
            engine_guard = LoadWindowGuard(
                engine_load_roots, expected_content=engine_expected_content
            )
            engine_guard.__enter__()
            try:
                before_engine_content = {
                    "tokenizer": authenticate_closed_tokenizer_view(
                        self.closed_tokenizer_path
                    ),
                    "engine_model": (
                        _validate_model_override(config.model_override)
                        if config.model_override is not None
                        else base_snapshot_commitment(self.base_snapshot_path)
                    ),
                }
                self.llm = LLM(**engine_args)
                after_engine_content = {
                    "tokenizer": authenticate_closed_tokenizer_view(
                        self.closed_tokenizer_path
                    ),
                    "engine_model": (
                        _validate_model_override(config.model_override)
                        if config.model_override is not None
                        else base_snapshot_commitment(self.base_snapshot_path)
                    ),
                }
                engine_guard.bind_authenticated_content(
                    before_engine_content, after_engine_content
                )
            except Exception as exc:
                engine_guard.__exit__(*sys.exc_info())
                loaded_llm = getattr(self, "llm", None)
                if loaded_llm is not None:
                    engine_core = getattr(
                        getattr(loaded_llm, "llm_engine", None),
                        "engine_core",
                        None,
                    )
                    shutdown = getattr(engine_core, "shutdown", None)
                    if callable(shutdown):
                        shutdown()
                available_blocks = _available_mamba_cache_blocks(exc)
                if (
                    available_blocks is None
                    or available_blocks < 1
                    or available_blocks >= config.max_num_seqs
                ):
                    raise
                original_message = str(exc)
                exc.args = (
                    f"{original_message}\n\nFrozen max_num_seqs={config.max_num_seqs} "
                    f"exceeds the reported Mamba cache capacity {available_blocks}; "
                    "adaptive geometry and process re-exec are forbidden. "
                    f"Hint: {_MAMBA_CACHE_REEXEC_HINT}.",
                )
                raise
            engine_guard_receipt = engine_guard.verify()
            post_base = base_snapshot_commitment(self.base_snapshot_path)
            post_tokenizer = authenticate_tokenizer_snapshot()
            post_closed_tokenizer = authenticate_closed_tokenizer_view(
                self.closed_tokenizer_path
            )
            post_override = _validate_model_override(config.model_override)
            if (
                post_base != self.base_snapshot_commitment
                or post_tokenizer != self.tokenizer_snapshot_commitment
                or post_closed_tokenizer != self.tokenizer_snapshot_commitment
                or post_override != self.model_override_info
            ):
                try:
                    self.llm.llm_engine.engine_core.shutdown()
                finally:
                    raise RuntimeError(
                        "model/tokenizer bytes changed between validation and engine load"
                    )
            from runtime_contract import bind_active_cuda_identity

            self.gpu_identity = bind_active_cuda_identity(
                Path(__file__).resolve().parents[3], torch
            )
            self.post_load_integrity = {
                "base_snapshot": post_base,
                "tokenizer_snapshot": post_tokenizer,
                "model_override": post_override,
                "load_window_guards": {
                    "tokenizer_and_config": tokenizer_config_guard_receipt,
                    "engine": engine_guard_receipt,
                },
                "decision": "LOAD_WINDOWS_IMMUTABLE_AND_POSTLOAD_BYTES_MATCH",
            }
            self.load_seconds = time.perf_counter() - started
        finally:
            random.setstate(python_rng_state)
            np.random.set_state(numpy_rng_state)
            torch.random.set_rng_state(torch_rng_state)
            if cuda_rng_states is not None:
                torch.cuda.set_rng_state_all(cuda_rng_states)
            elif torch.cuda.is_initialized():
                torch.cuda.manual_seed_all(torch_initial_seed)
        self.engine_args = engine_args

        try:
            compilation_config = self.llm.llm_engine.vllm_config.compilation_config
            self.resolved_cudagraph = _resolved_cudagraph_metadata(
                compilation_config
            )
            if config.cudagraph_capture_sizes is not None:
                _validate_explicit_cudagraph_resolution(
                    config.cudagraph_capture_sizes, self.resolved_cudagraph
                )
            self.live_engine_geometry = _read_live_engine_geometry(self.llm, config)
        except (AttributeError, RuntimeError):
            self.close()
            raise

        self.lora_request = None
        if self.adapter_info is not None:
            from vllm.lora.request import LoRARequest

            name_hash = hashlib.sha256(
                self.adapter_info["path"].encode("utf-8")
            ).hexdigest()[:12]
            self.lora_request = LoRARequest(
                f"experiment-{name_hash}", 1, self.adapter_info["path"]
            )

    def close(self) -> None:
        """Release vLLM's engine resources and distributed process group."""
        if self._closed:
            return
        self._closed = True
        llm = getattr(self, "llm", None)
        engine = getattr(llm, "llm_engine", None)
        client = getattr(engine, "engine_core", None)
        shutdown = getattr(client, "shutdown", None)
        if callable(shutdown):
            shutdown()

    def __enter__(self) -> "VLLMRunner":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def _single_token_id(self, text: str) -> int:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) != 1:
            raise RuntimeError(f"expected one token for {text!r}, got {ids}")
        return int(ids[0])

    def _render_messages(self, messages: Any, enable_thinking: bool) -> str:
        if not isinstance(messages, list) or not messages:
            raise ValueError("messages must be a non-empty list")
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("each message must be an object")
            if message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValueError(f"unsupported message role: {message.get('role')!r}")
            if not isinstance(message.get("content"), str):
                raise ValueError("this text-only runner requires string message content")
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError as exc:
            raise RuntimeError(
                "the pinned Qwen3.5 chat template rejected enable_thinking; "
                "refusing a silent prompt-format fallback"
            ) from exc
        if not isinstance(rendered, str):
            raise RuntimeError("chat template did not return rendered text")
        return rendered

    def _prompt_channel(self, token_ids: Sequence[int]) -> str:
        """Classify the exact generation suffix produced by Qwen's chat template."""
        if (
            list(token_ids[-len(self.no_thinking_prompt_suffix_ids) :])
            == self.no_thinking_prompt_suffix_ids
        ):
            return "off"
        if (
            list(token_ids[-len(self.thinking_prompt_suffix_ids) :])
            == self.thinking_prompt_suffix_ids
        ):
            return "thinking"
        return "custom"

    def prepare(
        self,
        records: Sequence[dict[str, Any]],
        thinking: str,
        allow_custom_prompts: bool = False,
    ) -> list[_PreparedRecord]:
        seen: set[str] = set()
        prepared: list[_PreparedRecord] = []
        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"input row {index} is not a JSON object")
            record_id = str(record.get("id", ""))
            if not record_id:
                raise ValueError(f"input row {index} has no non-empty id")
            if record_id in seen:
                raise ValueError(f"duplicate input id: {record_id!r}")
            seen.add(record_id)
            has_prompt = "prompt" in record
            has_messages = "messages" in record
            if has_prompt == has_messages:
                raise ValueError(
                    f"input {record_id!r} must contain exactly one of prompt or messages"
                )
            if has_prompt:
                prompt = record["prompt"]
                if not isinstance(prompt, str):
                    raise ValueError(f"input {record_id!r} prompt must be a string")
            else:
                prompt = self._render_messages(record["messages"], thinking != "off")
            token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            if not token_ids:
                raise ValueError(f"input {record_id!r} produced an empty prompt")
            prompt_channel = self._prompt_channel(token_ids)
            expected_channel = "off" if thinking == "off" else "thinking"
            if prompt_channel != expected_channel and not allow_custom_prompts:
                raise ValueError(
                    f"input {record_id!r} has {prompt_channel!r} prompt channel, but "
                    f"thinking={thinking!r} requires {expected_channel!r}; rerender it or use "
                    "allow_custom_prompts=True/--allow-custom-prompts for an audited custom format"
                )
            prepared.append(
                _PreparedRecord(
                    record_id,
                    record.get("meta"),
                    prompt,
                    list(token_ids),
                    prompt_channel,
                )
            )
        return prepared

    def _check_context(self, records: Sequence[_PreparedRecord], sampling: SamplingConfig) -> None:
        if sampling.thinking == "budget":
            reserve = (
                int(sampling.thinking_budget)
                + len(self.close_ids)
                + sampling.answer_max_tokens
            )
        else:
            reserve = sampling.max_tokens
        too_long = [
            (record.record_id, len(record.prompt_token_ids) + reserve)
            for record in records
            if len(record.prompt_token_ids) + reserve > self.config.max_model_len
        ]
        if too_long:
            detail = ", ".join(f"{rid}={length}" for rid, length in too_long[:5])
            raise ValueError(
                f"prompt + generation cap exceeds max_model_len={self.config.max_model_len}: "
                + detail
            )

    def _params(self, sampling: SamplingConfig, *, max_tokens: int, seed: int, n: int):
        from vllm import SamplingParams

        resolved = sampling.resolved_sampling()
        return SamplingParams(
            n=n,
            temperature=float(resolved["temperature"]),
            top_p=float(resolved["top_p"]),
            top_k=int(resolved["top_k"]),
            min_p=float(resolved["min_p"]),
            presence_penalty=float(resolved["presence_penalty"]),
            frequency_penalty=float(resolved["frequency_penalty"]),
            repetition_penalty=float(resolved["repetition_penalty"]),
            seed=seed,
            max_tokens=max_tokens,
            # The tokenizer marks <|im_end|> as EOS, while the pinned HF model
            # generation config stops one token later at <|endoftext|>.  Ignore
            # vLLM's tokenizer EOS and use the model EOS so migrations preserve
            # the established HF answer boundary and token accounting.
            ignore_eos=True,
            stop_token_ids=[self.hf_eos_id],
            logprobs=sampling.logprobs,
            prompt_logprobs=sampling.prompt_logprobs,
            logprob_token_ids=list(sampling.logprob_token_ids) or None,
            detokenize=True,
            skip_special_tokens=False,
        )

    def _decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=False)

    def _trim_hf_eos(self, token_ids: Sequence[int]) -> list[int]:
        ids = list(token_ids)
        return ids[: ids.index(self.hf_eos_id)] if self.hf_eos_id in ids else ids

    def _ordinary_output(
        self,
        record: _PreparedRecord,
        completion: Any,
        sample_index: int,
        seed: int,
        thinking: str,
    ) -> dict[str, Any]:
        sampled_ids = list(completion.token_ids)
        token_ids = self._trim_hf_eos(sampled_ids)
        close_index = (
            token_ids.index(self.think_close_id)
            if self.think_close_id in token_ids
            else None
        )
        if thinking == "off":
            n_thinking = 0
            n_answer = len(token_ids)
        elif close_index is None:
            n_thinking = len(token_ids)
            n_answer = 0
        else:
            n_thinking = close_index
            n_answer = len(token_ids) - close_index - 1
        return {
            "sample_index": sample_index,
            "stage1_parent_seed": seed,
            "seed_stage1": seed + sample_index,
            "seed_stage2": None,
            "text": self._decode(token_ids),
            "token_ids": token_ids,
            "stage1_token_ids": sampled_ids,
            "injected_token_ids": [],
            "stage2_token_ids": [],
            "n_thinking_tokens": n_thinking,
            "n_answer_tokens": n_answer,
            "n_sampled_tokens": len(sampled_ids),
            "n_injected_tokens": 0,
            "n_completion_tokens": len(token_ids),
            "n_terminal_tokens_trimmed": len(sampled_ids) - len(token_ids),
            "n_stage1_prompt_tokens": len(record.prompt_token_ids),
            "n_stage2_prompt_tokens": 0,
            "thinking_closed": close_index is not None,
            "forced_close": False,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "stage1_finish_reason": completion.finish_reason,
            "stage1_stop_reason": completion.stop_reason,
            "truncated": completion.finish_reason == "length",
            "stage1_cumulative_logprob": completion.cumulative_logprob,
            "stage2_cumulative_logprob": None,
            "sampled_cumulative_logprob": completion.cumulative_logprob,
            "stage1_logprobs": _jsonable_logprobs(completion.logprobs),
            "stage2_logprobs": None,
        }

    def generate(
        self,
        records: Sequence[dict[str, Any]],
        sampling: SamplingConfig = SamplingConfig(),
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        sampling.validate()
        prepared = self.prepare(
            records, sampling.thinking, sampling.allow_custom_prompts
        )
        if not prepared:
            raise ValueError("input is empty")
        self._check_context(prepared, sampling)
        capacity_preflight = _capacity_preflight(
            live=self.live_engine_geometry,
            config=self.config,
            prompt_lengths=[len(record.prompt_token_ids) for record in prepared],
            sampling=sampling,
            close_tokens=len(self.close_ids),
        )
        prompts = [{"prompt_token_ids": record.prompt_token_ids} for record in prepared]
        seeds = [
            _stable_seed(sampling.run_seed, record.record_id, -1, "stage1")
            for record in prepared
        ]
        first_cap = (
            int(sampling.thinking_budget)
            if sampling.thinking == "budget"
            else sampling.max_tokens
        )
        params = [
            self._params(sampling, max_tokens=first_cap, seed=seed, n=sampling.n)
            for seed in seeds
        ]

        started = time.perf_counter()
        first_outputs = self.llm.generate(
            prompts,
            params,
            use_tqdm=False,
            lora_request=self.lora_request,
        )

        rows: list[dict[str, Any]] = []
        continuation_prompts: list[dict[str, list[int]]] = []
        continuation_params: list[Any] = []
        continuation_meta: list[tuple[int, int, list[int], list[int], bool, int, Any]] = []
        # (row index, output index, original stage1, retained thinking,
        #  forced_close, stage2 seed, stage1 completion)

        for row_index, (record, request_output, seed) in enumerate(
            zip(prepared, first_outputs, seeds)
        ):
            row = {
                "id": record.record_id,
                "meta": record.meta,
                "prompt_sha256": _sha256_bytes(record.prompt_text.encode("utf-8")),
                "prompt_token_ids": list(record.prompt_token_ids),
                "n_prompt_tokens": len(record.prompt_token_ids),
                "prompt_channel": record.prompt_channel,
                "prompt_logprobs": _jsonable_logprobs(request_output.prompt_logprobs),
                "outputs": [None] * len(request_output.outputs),
            }
            rows.append(row)
            for completion in request_output.outputs:
                sample_index = int(completion.index)
                if sampling.thinking != "budget":
                    row["outputs"][sample_index] = self._ordinary_output(
                        record, completion, sample_index, seed, sampling.thinking
                    )
                    continue

                stage1_sampled = list(completion.token_ids)
                stage1 = self._trim_hf_eos(stage1_sampled)
                close_index = (
                    stage1.index(self.think_close_id)
                    if self.think_close_id in stage1
                    else None
                )
                naturally_finished = close_index is not None and completion.finish_reason == "stop"
                if naturally_finished:
                    row["outputs"][sample_index] = self._ordinary_output(
                        record, completion, sample_index, seed, "budget"
                    )
                    continue

                retained = stage1[:close_index] if close_index is not None else list(stage1)
                forced_close = close_index is None
                stage2_seed = _stable_seed(
                    sampling.run_seed, record.record_id, sample_index, "stage2"
                )
                if sampling.shuffle_thinking and retained:
                    random.Random(
                        _stable_seed(
                            sampling.run_seed, record.record_id, sample_index, "shuffle"
                        )
                    ).shuffle(retained)
                continuation_prompts.append(
                    {"prompt_token_ids": record.prompt_token_ids + retained + self.close_ids}
                )
                continuation_params.append(
                    self._params(
                        sampling,
                        max_tokens=sampling.answer_max_tokens,
                        seed=stage2_seed,
                        n=1,
                    )
                )
                continuation_meta.append(
                    (
                        row_index,
                        sample_index,
                        stage1_sampled,
                        retained,
                        forced_close,
                        stage2_seed,
                        completion,
                    )
                )

        if continuation_prompts:
            second_outputs = self.llm.generate(
                continuation_prompts,
                continuation_params,
                use_tqdm=False,
                lora_request=self.lora_request,
            )
            for meta, request_output in zip(continuation_meta, second_outputs):
                (
                    row_index,
                    sample_index,
                    stage1_sampled,
                    retained,
                    forced_close,
                    stage2_seed,
                    first_completion,
                ) = meta
                completion = request_output.outputs[0]
                stage2_sampled = list(completion.token_ids)
                stage2 = self._trim_hf_eos(stage2_sampled)
                final_ids = retained + self.close_ids + stage2
                stage2_prompt_tokens = (
                    len(prepared[row_index].prompt_token_ids)
                    + len(retained)
                    + len(self.close_ids)
                )
                stage1_cumulative = first_completion.cumulative_logprob
                stage2_cumulative = completion.cumulative_logprob
                sampled_cumulative = (
                    stage1_cumulative + stage2_cumulative
                    if stage1_cumulative is not None and stage2_cumulative is not None
                    else None
                )
                rows[row_index]["outputs"][sample_index] = {
                    "sample_index": sample_index,
                    "stage1_parent_seed": seeds[row_index],
                    "seed_stage1": seeds[row_index] + sample_index,
                    "seed_stage2": stage2_seed,
                    "text": self._decode(final_ids),
                    "token_ids": final_ids,
                    "stage1_token_ids": stage1_sampled,
                    "retained_thinking_token_ids": retained,
                    "injected_token_ids": list(self.close_ids),
                    "stage2_token_ids": stage2_sampled,
                    "n_thinking_tokens": len(retained),
                    "n_answer_tokens": len(stage2),
                    "n_sampled_tokens": len(stage1_sampled) + len(stage2_sampled),
                    "n_injected_tokens": len(self.close_ids),
                    "n_completion_tokens": len(final_ids),
                    "n_terminal_tokens_trimmed": (
                        len(stage1_sampled) - len(self._trim_hf_eos(stage1_sampled))
                    )
                    + (len(stage2_sampled) - len(stage2)),
                    "n_stage1_prompt_tokens": len(
                        prepared[row_index].prompt_token_ids
                    ),
                    "n_stage2_prompt_tokens": stage2_prompt_tokens,
                    "thinking_closed": True,
                    "forced_close": forced_close,
                    "finish_reason": completion.finish_reason,
                    "stop_reason": completion.stop_reason,
                    "stage1_finish_reason": first_completion.finish_reason,
                    "stage1_stop_reason": first_completion.stop_reason,
                    "truncated": completion.finish_reason == "length",
                    "stage1_cumulative_logprob": stage1_cumulative,
                    "stage2_cumulative_logprob": stage2_cumulative,
                    "sampled_cumulative_logprob": sampled_cumulative,
                    "stage1_logprobs": _jsonable_logprobs(first_completion.logprobs),
                    "stage2_logprobs": _jsonable_logprobs(completion.logprobs),
                }

        generation_seconds = time.perf_counter() - started
        for row in rows:
            if any(output is None for output in row["outputs"]):
                raise RuntimeError(f"internal error: missing output for {row['id']!r}")

        unique_input_prompt = sum(row["n_prompt_tokens"] for row in rows)
        stage1_logical_prompt = sum(
            output["n_stage1_prompt_tokens"]
            for row in rows
            for output in row["outputs"]
        )
        stage2_logical_prompt = sum(
            output["n_stage2_prompt_tokens"]
            for row in rows
            for output in row["outputs"]
        )
        total_sampled = sum(
            output["n_sampled_tokens"] for row in rows for output in row["outputs"]
        )
        total_injected = sum(
            output["n_injected_tokens"] for row in rows for output in row["outputs"]
        )
        summary = {
            "schema_version": RUNNER_SCHEMA_VERSION,
            "model": self.engine_args["model"],
            "base_model": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "runner_sha256": _sha256_file(Path(__file__).resolve()),
            "engine": _engine_config_metadata(self.config),
            "engine_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in self.engine_args.items()
            },
            "resolved_cudagraph": self.resolved_cudagraph,
            "capacity_preflight": capacity_preflight,
            "sampling": dataclasses.asdict(sampling),
            "resolved_sampling": sampling.resolved_sampling(),
            "adapter": self.adapter_info,
            "model_override": self.model_override_info,
            "base_snapshot": self.base_snapshot_commitment,
            "tokenizer_snapshot": self.tokenizer_snapshot_commitment,
            "post_load_integrity": self.post_load_integrity,
            "think_token_ids": {
                "open": self.think_open_id,
                "close": self.think_close_id,
                "forced_close_sequence": self.close_ids,
                "thinking_prompt_suffix": self.thinking_prompt_suffix_ids,
                "no_thinking_prompt_suffix": self.no_thinking_prompt_suffix_ids,
            },
            "termination": {
                "hf_model_eos_token_id": self.hf_eos_id,
                "vllm_tokenizer_eos_ignored": self.tokenizer_eos_id,
            },
            "rng_isolation": {
                "engine_seed": self.engine_args["seed"],
                "caller_global_rng_state_restored": True,
            },
            "counts": {
                "requests": len(rows),
                "completions": sum(len(row["outputs"]) for row in rows),
                "unique_input_prompt_tokens": unique_input_prompt,
                "stage1_logical_prompt_tokens": stage1_logical_prompt,
                "stage2_logical_prompt_tokens": stage2_logical_prompt,
                "logical_model_input_tokens": (
                    stage1_logical_prompt + stage2_logical_prompt
                ),
                "sampled_tokens": total_sampled,
                "injected_tokens": total_injected,
            },
            "timing": {
                "model_load_seconds": self.load_seconds,
                "generation_seconds": generation_seconds,
                "sampled_tokens_per_second": total_sampled / generation_seconds,
            },
            "runtime": self.runtime_metadata(),
        }
        return rows, summary

    def runtime_metadata(self) -> dict[str, Any]:
        git_root = _run_text(["git", "rev-parse", "--show-toplevel"])
        git_commit = _run_text(["git", "rev-parse", "HEAD"]) if git_root else ""
        git_status = _run_text(
            [
                "git",
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignored=matching",
            ]
        ) if git_root else ""
        git_branch = _run_text(["git", "symbolic-ref", "-q", "HEAD"]) if git_root else ""
        uv_version = _run_text(["uv", "--version"])
        cuda_toolkit = _run_text(["nvcc", "--version"])
        python_version = platform.python_version()
        platform_value = platform.platform()
        environment_lock = _environment_lock_metadata()
        from runtime_contract import (
            runtime_bootstrap_receipt,
            seal_runtime_environment,
        )

        seal_runtime_environment(Path(git_root), "vllm")
        return {
            "schema_version": 4,
            "bootstrap": runtime_bootstrap_receipt("vllm"),
            "python": python_version,
            "python_executable": str(Path(sys.executable).resolve()),
            "python_executable_sha256": _sha256_file(Path(sys.executable).resolve()),
            "python_isolated": sys.flags.isolated == 1,
            "python_dont_write_bytecode": bool(sys.dont_write_bytecode),
            "python_no_site": sys.flags.no_site == 1,
            "platform": platform_value,
            "packages": dict(_INITIAL_PACKAGES),
            "packages_sha256": _sha256_bytes(
                json.dumps(
                    dict(_INITIAL_PACKAGES), sort_keys=True, separators=(",", ":")
                ).encode()
            ),
            "environment_lock": environment_lock,
            "uv": uv_version,
            "cuda_toolkit": cuda_toolkit,
            "gpu": dict(self.gpu_identity),
            "vllm_enable_v1_multiprocessing": os.environ.get(
                "VLLM_ENABLE_V1_MULTIPROCESSING"
            ),
            "git_commit": git_commit,
            "git_dirty": bool(git_status),
            "git_root": git_root,
            "cwd": str(Path.cwd().resolve()),
            "git_head_mode": "branch" if git_branch else "detached",
        }


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], str]:
    data = path.read_bytes()
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON on {path}:{line_number}: {exc}") from exc
    return records, _sha256_bytes(data)


def _smoke_records(count: int) -> list[dict[str, Any]]:
    tasks = [
        "Reply with exactly VLLM_OK.",
        "Return only the integer equal to 17 + 25.",
        "Write the reverse of the string abcdef, with no explanation.",
        "Return only a JSON array containing the first five positive odd integers.",
    ]
    return [
        {
            "id": f"smoke-{index:04d}",
            "messages": [{"role": "user", "content": tasks[index % len(tasks)]}],
            "meta": {"task_index": index % len(tasks)},
        }
        for index in range(count)
    ]


def _write_json_atomic(path: Path, value: Any, *, jsonl: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        if jsonl:
            for row in value:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        else:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
    digest = _sha256_file(temporary)
    temporary.replace(path)
    return digest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="input JSONL")
    source.add_argument("--smoke", type=int, metavar="N", help="use N built-in smoke prompts")
    parser.add_argument("--output", type=Path, required=True, help="output JSONL")
    parser.add_argument("--metadata", type=Path, help="metadata JSON; default: OUTPUT.meta.json")
    parser.add_argument(
        "--stage-receipt",
        type=Path,
        help="required hash-bound experiment stage receipt for every non-smoke input run",
    )
    parser.add_argument("--thinking", choices=["off", "natural", "budget"], default="off")
    parser.add_argument("--thinking-budget", type=int)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--frequency-penalty", type=float, default=0.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shuffle-thinking", action="store_true")
    parser.add_argument("--logprobs", type=int)
    parser.add_argument("--prompt-logprobs", type=int)
    parser.add_argument("--logprob-token-id", type=int, action="append", default=[])
    parser.add_argument(
        "--allow-custom-prompts",
        action="store_true",
        help="allow raw prompts whose think-channel suffix does not match --thinking",
    )
    parser.add_argument("--adapter", type=Path)
    parser.add_argument(
        "--model-override",
        type=Path,
        help="local merged composite checkpoint; required for trained Qwen3.5 adapters",
    )
    parser.add_argument("--max-model-len", type=int, default=16_384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32_768)
    parser.add_argument(
        "--cudagraph-capture-size",
        type=int,
        action="append",
        default=None,
        help=(
            "explicit CUDA-graph capture size; repeat in strictly increasing order "
            "and end at --max-num-seqs"
        ),
    )
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--include-prompt-token-ids",
        action="store_true",
        help="include exact prompt IDs in output rows (metadata always records counts/hashes)",
    )
    return parser.parse_args(argv)


def _validate_cli_stage_receipt(
    path: Path,
    records: Sequence[dict[str, Any]],
    model_override: Path | None,
) -> dict[str, Any]:
    """Bind every experiment generation to the exact staged evidence ancestry."""
    import yaml

    experiment_root = Path(__file__).resolve().parents[1]
    config_path = experiment_root / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["authorization"]["evaluation"] is not True:
        raise ValueError("evaluation is not authorized by the committed config")
    metas = [record.get("meta") for record in records]
    if any(not isinstance(meta, dict) for meta in metas):
        raise ValueError("staged generation requires sealed metadata on every input")
    splits = {str(meta.get("split", "")) for meta in metas}
    kinds = {str(meta.get("input_kind", "")) for meta in metas}
    if len(splits) != 1 or len(kinds) != 1 or "" in splits or "" in kinds:
        raise ValueError("staged generation input split/kind is ambiguous")
    split = splits.pop()
    input_kind = kinds.pop()
    source_seed = None
    if model_override is not None:
        merge_receipt = json.loads((model_override / "merge_receipt.json").read_text())
        source_seed = int(merge_receipt.get("source_seed", -1))
    screen = int(config["training"]["staged_seeds"]["screen"])
    replication = int(config["training"]["staged_seeds"]["replication"])
    if split == "confirmation":
        expected_stage = "confirmation"
    elif split == "calibration" and model_override is None:
        expected_stage = "calibration_generation"
    elif source_seed == replication:
        expected_stage = "replication_training"
    elif source_seed in {None, screen} and split in {
        "calibration",
        "qualification",
        "retention",
    }:
        expected_stage = "screen_training"
    else:
        raise ValueError("generation model/split does not map to a frozen stage")
    sys.path.insert(0, str(experiment_root / "src"))
    from stages import read_and_validate_stage_receipt

    receipt = read_and_validate_stage_receipt(
        path,
        config=config,
        config_path=config_path,
        expected_stage=expected_stage,
    )
    return {
        "authorized_stage": expected_stage,
        "stage_receipt_path": str(path.resolve()),
        "stage_receipt_sha256": _sha256_file(path),
        "config_sha256": receipt["config_sha256"],
        "issuer_git_commit": receipt["issuer_git_commit"],
        "split": split,
        "input_kind": input_kind,
        "source_seed": source_seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    from runtime_contract import require_detached_execution_worktree

    require_detached_execution_worktree(Path(__file__).resolve().parents[3])
    args = _parse_args(argv)
    metadata_path = args.metadata or args.output.with_name(args.output.name + ".meta.json")
    if args.output.resolve() == metadata_path.resolve():
        raise ValueError("--metadata must not be the same path as --output")
    if args.input is not None and args.input.resolve() in {
        args.output.resolve(),
        metadata_path.resolve(),
    }:
        raise ValueError("--input must not be overwritten by --output or --metadata")
    if args.input is not None:
        records, input_sha256 = _read_jsonl(args.input)
        input_description = str(args.input.resolve())
        if args.stage_receipt is None:
            raise ValueError("--stage-receipt is required for every non-smoke input run")
        generation_stage = _validate_cli_stage_receipt(
            args.stage_receipt, records, args.model_override
        )
    else:
        if args.stage_receipt is not None:
            raise ValueError("built-in smoke must not consume an experiment stage receipt")
        if args.smoke < 1:
            raise ValueError("--smoke N requires N >= 1")
        records = _smoke_records(args.smoke)
        encoded = "".join(json.dumps(row, sort_keys=True) + "\n" for row in records).encode()
        input_sha256 = _sha256_bytes(encoded)
        input_description = f"built-in-smoke:{args.smoke}"
        generation_stage = None

    engine = EngineConfig(
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        cudagraph_capture_sizes=(
            tuple(args.cudagraph_capture_size)
            if args.cudagraph_capture_size is not None
            else None
        ),
        enable_prefix_caching=args.enable_prefix_caching,
        enforce_eager=args.enforce_eager,
        adapter=args.adapter,
        model_override=args.model_override,
    )
    sampling = SamplingConfig(
        thinking=args.thinking,
        thinking_budget=args.thinking_budget,
        n=args.n,
        max_tokens=args.max_tokens,
        answer_max_tokens=args.answer_max_tokens,
        greedy=args.greedy,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
        run_seed=args.seed,
        shuffle_thinking=args.shuffle_thinking,
        logprobs=args.logprobs,
        prompt_logprobs=args.prompt_logprobs,
        logprob_token_ids=tuple(args.logprob_token_id),
        allow_custom_prompts=args.allow_custom_prompts,
    )

    sampling.validate()  # fail before an expensive model load
    runner: VLLMRunner | None = None
    try:
        runner = VLLMRunner(engine)
        rows, summary = runner.generate(records, sampling)
        prepared = runner.prepare(
            records, sampling.thinking, sampling.allow_custom_prompts
        )
        if args.include_prompt_token_ids:
            for row, record in zip(rows, prepared):
                row["prompt_token_ids"] = record.prompt_token_ids
        summary["input"] = {
            "description": input_description,
            "sha256": input_sha256,
        }
        summary["generation_stage"] = generation_stage
        output_sha256 = _write_json_atomic(args.output, rows, jsonl=True)
        summary["output"] = {
            "description": "generated JSONL",
            "sha256": output_sha256,
            "rows": len(rows),
        }
        _write_json_atomic(metadata_path, summary)
        timing = summary["timing"]
        counts = summary["counts"]
        print(
            f"wrote {counts['completions']} completions / {counts['sampled_tokens']} sampled tokens "
            f"at {timing['sampled_tokens_per_second']:.1f} tok/s to {args.output}",
            file=sys.stderr,
        )
    finally:
        if runner is not None:
            runner.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
