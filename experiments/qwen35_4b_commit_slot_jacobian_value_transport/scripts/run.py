#!/usr/bin/env python3
"""Stage-gated commit-slot seam and Jacobian value-transport harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from task_data import (  # noqa: E402
    IDENTIFIABLE_FIRST_OPERATIONS,
    build_splits,
    matching_depth_one,
    matching_first_types,
    parse_alias,
    task_fingerprint,
    task_prompt,
    verify_answer,
)


CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("configuration must be a mapping")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _config_payload_sha256(config: dict[str, Any]) -> str:
    return _canonical_sha256({
        key: value for key, value in config.items() if key != "design_boundary"
    })


def _token_ids_sha256(values: list[int]) -> str:
    return _canonical_sha256([int(value) for value in values])


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    if commit.startswith("PENDING_"):
        raise RuntimeError("design boundary is not anchored")
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode == 0
    paths = {
        "readme": "experiments/qwen35_4b_commit_slot_jacobian_value_transport/README.md",
        "preregistration": (
            "experiments/qwen35_4b_commit_slot_jacobian_value_transport/"
            "reports/preregistration.md"
        ),
        "design_review": (
            "experiments/qwen35_4b_commit_slot_jacobian_value_transport/"
            "reports/design_review.md"
        ),
    }
    observed = {
        name: hashlib.sha256(_git(["show", f"{commit}:{path}"]).stdout.encode()).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "readme": str(boundary["readme_sha256"]),
        "preregistration": str(boundary["preregistration_sha256"]),
        "design_review": str(boundary["design_review_sha256"]),
    }
    config_path = (
        "experiments/qwen35_4b_commit_slot_jacobian_value_transport/"
        "configs/default.yaml"
    )
    committed_config = yaml.safe_load(_git(["show", f"{commit}:{config_path}"]).stdout)
    committed_payload_hash = _config_payload_sha256(committed_config)
    current_payload_hash = _config_payload_sha256(config)
    expected_payload_hash = str(boundary["config_payload_sha256"])
    lens_hash = sha256_file(EXP / config["lens"]["path"])
    result = {
        "schema_version": 1,
        "stage": "design_boundary",
        "scientific_result": False,
        "passed": bool(
            ancestor
            and observed == expected
            and committed_payload_hash == expected_payload_hash
            and current_payload_hash == expected_payload_hash
            and lens_hash == str(config["lens"]["sha256"])
        ),
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "committed_config_payload_sha256": committed_payload_hash,
        "current_config_payload_sha256": current_payload_hash,
        "expected_config_payload_sha256": expected_payload_hash,
        "lens_observed_sha256": lens_hash,
        "lens_expected_sha256": str(config["lens"]["sha256"]),
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", result)
    if not result["passed"]:
        raise RuntimeError(f"immutable design boundary failed: {result}")
    return result


def require_data_receipt(config: dict[str, Any]) -> dict[str, Any]:
    receipt_path = RUNS_DIR / "smoke" / "data_receipt.json"
    manifest_path = DATA_DIR / "manifest.json"
    if not receipt_path.exists() or not manifest_path.exists():
        raise RuntimeError("passing CPU smoke receipts are required")
    receipt = read_json(receipt_path)
    manifest = read_json(manifest_path)
    if receipt.get("passed") is not True or receipt.get("parent_overlap") != 0:
        raise RuntimeError("CPU smoke receipt is not passing and fresh")
    if manifest.get("total_unique_fingerprints") != 96:
        raise RuntimeError("procedural manifest does not contain 96 unique rows")
    observed: dict[str, str] = {}
    for split, record in manifest["splits"].items():
        path = EXP / record["path"]
        observed[split] = sha256_file(path)
        if observed[split] != record["sha256"]:
            raise RuntimeError(f"procedural split changed after CPU smoke: {split}")
    if manifest.get("model_id") != config["model"]["id"]:
        raise RuntimeError("procedural manifest model ID changed")
    if manifest.get("lens_sha256") != config["lens"]["sha256"]:
        raise RuntimeError("procedural manifest lens changed")
    return {
        "receipt_sha256": sha256_file(receipt_path),
        "manifest_sha256": sha256_file(manifest_path),
        "split_sha256": observed,
    }


def _parent_fingerprints() -> tuple[set[str], dict[str, int]]:
    parents = (
        "qwen35_4b_jacobian_value_transport",
        "qwen35_4b_native_thought_jacobian_value_transport",
        "qwen35_4b_native_thought_seam_budget_ladder",
        "qwen35_4b_forced_commit_jacobian_value_transport",
    )
    combined: set[str] = set()
    counts: dict[str, int] = {}
    for parent in parents:
        values: set[str] = set()
        directory = ROOT / "experiments" / parent / "data" / "procedural"
        for path in sorted(directory.glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                    values.add(task_fingerprint(row))
        counts[parent] = len(values)
        combined.update(values)
    return combined, counts


def gate_reachability(config: dict[str, Any]) -> dict[str, Any]:
    traces_per_task = int(config["generation"]["traces_per_task"])
    receipts: dict[str, Any] = {}
    for stage, task_key in (
        ("seam_selection", "seam_selection_tasks"),
        ("seam_confirmation", "seam_confirmation_tasks"),
    ):
        tasks = int(config["data"][task_key])
        traces = tasks * traces_per_task
        gates = config["gates"][stage]
        min_success = math.ceil(float(gates["slot_success_rate_min"]) * traces - 1e-12)
        max_success = math.floor(float(gates["slot_success_rate_max"]) * traces + 1e-12)
        min_mixed = int(gates["mixed_slot_tasks_min"])
        possible = bool(
            min_success <= max_success
            and min_mixed <= tasks
            and 2 * min_mixed <= traces
            and float(gates["real_minus_no_thought_accuracy_min"]) <= 1.0
            and float(gates["real_minus_shuffled_thought_accuracy_min"]) <= 1.0
            and float(gates["finite_slot_rows_rate_min"]) <= 1.0
        )
        receipts[stage] = {
            "tasks": tasks,
            "traces": traces,
            "minimum_successes": min_success,
            "maximum_successes": max_success,
            "minimum_mixed_tasks": min_mixed,
            "passed": possible,
        }
    result = {
        "schema_version": 1,
        "stage": "gate_reachability",
        "scientific_result": False,
        "passed": all(value["passed"] for value in receipts.values()),
        "receipts": receipts,
    }
    if not result["passed"]:
        raise RuntimeError(f"a frozen seam gate is unreachable: {result}")
    return result


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    if sha256_file(lens_path) != str(config["lens"]["sha256"]):
        raise RuntimeError("frozen replicated lens hash mismatch")
    generation = config["generation"]
    rungs = [int(value) for value in generation["cap_rungs"]]
    if rungs != [256, 512, 1024]:
        raise RuntimeError("frozen cap ladder changed")
    if int(generation["prompt_max_tokens"]) + max(rungs) + int(
        generation["answer_max_tokens"]
    ) > int(generation["total_max_tokens"]):
        raise RuntimeError("context envelope is infeasible")
    parent, parent_counts = _parent_fingerprints()
    splits = build_splits(config)
    all_fingerprints: list[str] = []
    receipts: dict[str, Any] = {}
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(path, rows)
        fingerprints = [task_fingerprint(row) for row in rows]
        all_fingerprints.extend(fingerprints)
        counts = Counter(str(row["first_op"]) for row in rows)
        for row in rows:
            inputs = [example["input"] for example in row["visible"]]
            outputs = tuple(tuple(example["output"]) for example in row["visible"])
            if matching_first_types(inputs, outputs) != {row["first_op"]}:
                raise RuntimeError("visible first-operation identifiability changed")
            if matching_depth_one(inputs, outputs):
                raise RuntimeError("an exact-depth-two row has a depth-one fit")
        receipts[split] = {
            "items": len(rows),
            "path": str(path.relative_to(EXP)),
            "sha256": sha256_file(path),
            "unique_fingerprints": len(set(fingerprints)),
            "parent_overlap": sum(value in parent for value in fingerprints),
            "first_op_counts": {
                name: counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS
            },
            "depth_one_fit_count": 0,
        }
    if len(all_fingerprints) != len(set(all_fingerprints)):
        raise RuntimeError("fresh split fingerprints overlap")
    if any(value["parent_overlap"] for value in receipts.values()):
        raise RuntimeError("fresh rows overlap a scientific parent")
    reachability = gate_reachability(config)
    write_json(RUNS_DIR / "smoke" / "gate_reachability.json", reachability)
    manifest = {
        "schema_version": 1,
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "lens_sha256": sha256_file(lens_path),
        "parent_fingerprint_counts": parent_counts,
        "total_unique_fingerprints": len(set(all_fingerprints)),
        "splits": receipts,
        "firewall": {
            "benchmark_content_used": False,
            "fresh_procedural_only": True,
            "visible_first_type_identifiable": True,
            "visible_depth_one_fit_rejected": True,
        },
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "scientific_result": False,
        "passed": True,
        "items": len(all_fingerprints),
        "unique_fingerprints": len(set(all_fingerprints)),
        "parent_overlap": 0,
        "depth_one_fit_count": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "gate_reachability_pass": reachability["passed"],
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _load_model(config: dict[str, Any]):
    from model_ops import QwenCommitModel

    return QwenCommitModel(config)


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    value = config["generation"]
    return {
        "answer_cap": int(value["answer_max_tokens"]),
        "total_max_tokens": int(value["total_max_tokens"]),
        "temperature": float(value["temperature"]),
        "top_p": float(value["top_p"]),
        "top_k": int(value["top_k"]),
    }


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    data = require_data_receipt(config)
    import torch
    import transformers

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model = _load_model(config)
    aliases = dict(config["data"]["operation_aliases"])
    alias_ids = {alias: model.leading_space_token_id(alias) for alias in aliases.values()}
    if len(set(alias_ids.values())) != len(alias_ids):
        raise RuntimeError("alias token IDs are not one-to-one")
    lens_state = torch.load(EXP / config["lens"]["path"], map_location="cpu", weights_only=True)
    ranks = {
        int(layer): int(torch.linalg.matrix_rank(direction.float()))
        for layer, direction in lens_state["directions"].items()
        if int(layer) in set(config["lens"]["band"])
    }
    task = read_jsonl(DATA_DIR / "seam_selection.jsonl")[0]
    prepared = model.prepare(
        task_prompt(task, aliases), prompt_max_tokens=int(config["generation"]["prompt_max_tokens"])
    )
    trace = model.generate_trace(
        prepared["input_ids"],
        seed=_stable_seed(int(config["seeds"]["seam_selection_trace"]), task["task_id"], "smoke"),
        thought_cap=8,
        **_generation_kwargs(config),
    )
    if trace["natural_close"] or trace["stopped_by"] == "eos_before_close":
        raise RuntimeError("eight-token smoke unexpectedly terminated before forced replay")
    forced = model.force_commit(
        prepared["input_ids"],
        trace["generated_token_ids"][:8],
        seed=_stable_seed(int(config["seeds"]["seam_selection_freeform"]), task["task_id"], "smoke"),
        answer_cap=2,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        top_k=int(config["generation"]["top_k"]),
    )
    slot = model.slot_readout(
        prepared["input_ids"],
        trace["generated_token_ids"][:8],
        slot_text=str(config["slot"]["text"]),
        aliases=list(aliases.values()),
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
    )
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and set(ranks.values()) == {24}
        and trace["cache_contract_pass"]
        and forced["cache_contract_pass"]
        and slot["finite"]
        and slot["chosen_alias"] in set(aliases.values())
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "scientific_result": False,
        "outcomes_recorded": False,
        "passed": passed,
        "model": {
            "id": config["model"]["id"],
            "revision": config["model"]["revision"],
            "layers": model.n_layers,
            "hidden_size": model.d_model,
            "vocab_size": model.vocab_size,
            "load_seconds": model.load_seconds,
        },
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        },
        "token_ids": {
            "think_open": model.think_open_id,
            "think_close": model.think_close_id,
            "eos": model.eos_id,
            "aliases": alias_ids,
        },
        "lens_ranks": {str(layer): rank for layer, rank in ranks.items()},
        "trace_forward_input_lengths": trace["forward_input_lengths"],
        "forced_forward_input_lengths": forced["forward_input_lengths"],
        "trace_cache_contract_pass": trace["cache_contract_pass"],
        "forced_cache_contract_pass": forced["cache_contract_pass"],
        "slot_prefill_tokens": slot["prefill_tokens"],
        "slot_token_ids": slot["slot_token_ids"],
        "slot_chosen_from_aliases": slot["chosen_alias"] in set(aliases.values()),
        "slot_finite": slot["finite"],
        "slot_constrained_to_aliases": slot["constrained_to_alias_tokens"],
        "forced_close_injected": True,
        "counterfactual_policy_acknowledged": True,
        "data_receipt": data,
        "correctness_recorded": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUNS_DIR / "model_smoke" / "result.json", result)
    if not passed:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _require_model_smoke() -> str:
    path = RUNS_DIR / "model_smoke" / "result.json"
    if not path.exists() or read_json(path).get("passed") is not True:
        raise RuntimeError("a passing outcome-blind model smoke is required")
    result = read_json(path)
    if result.get("outcomes_recorded") is not False or result.get("correctness_recorded") is not False:
        raise RuntimeError("model smoke outcome-blind contract changed")
    return sha256_file(path)


def slot_metrics(
    slot_rows: list[dict[str, Any]],
    shuffled_slot_rows: list[dict[str, Any]],
    no_thought_rows: list[dict[str, Any]],
    freeform_rows: list[dict[str, Any]],
    task_ids: list[str],
    cap: int,
) -> dict[str, Any]:
    rows = [row for row in slot_rows if int(row["cap"]) == int(cap)]
    shuffled = [row for row in shuffled_slot_rows if int(row["cap"]) == int(cap)]
    controls = [row for row in freeform_rows if int(row["cap"]) == int(cap)]
    successes = sum(bool(row["correct"]) for row in rows)
    mixed = 0
    for task_id in task_ids:
        values = [row for row in rows if row["task_id"] == task_id]
        if any(row["correct"] for row in values) and any(not row["correct"] for row in values):
            mixed += 1
    no_thought_success = sum(bool(row["correct"]) for row in no_thought_rows)
    real_accuracy = successes / len(rows) if rows else 0.0
    no_thought_accuracy = no_thought_success / len(no_thought_rows) if no_thought_rows else 0.0
    shuffled_accuracy = sum(bool(row["correct"]) for row in shuffled) / len(shuffled)
    commit_mode_counts = Counter(str(row["commit_mode"]) for row in rows)
    return {
        "cap": int(cap),
        "slot_rows": len(rows),
        "slot_successes": successes,
        "slot_success_rate": real_accuracy,
        "mixed_slot_tasks": mixed,
        "finite_slot_rows_rate": sum(bool(row["finite"]) for row in rows) / len(rows) if rows else 0.0,
        "finite_shuffled_slot_rows_rate": sum(
            bool(row["finite"]) for row in shuffled
        ) / len(shuffled),
        "finite_no_thought_slot_rows_rate": sum(
            bool(row["finite"]) for row in no_thought_rows
        ) / len(no_thought_rows),
        "commit_mode_counts": dict(sorted(commit_mode_counts.items())),
        "mean_correct_alias_probability": sum(float(row["correct_alias_probability"]) for row in rows) / len(rows),
        "mean_constrained_margin": sum(float(row["constrained_margin"]) for row in rows) / len(rows),
        "mean_full_vocab_alias_probability_mass": sum(
            float(row["full_vocab_alias_probability_mass"]) for row in rows
        ) / len(rows),
        "mean_correct_alias_full_vocab_probability": sum(
            float(row["alias_full_vocab_probabilities"][row["correct_alias"]]) for row in rows
        ) / len(rows),
        "thought_contains_any_alias_rate": sum(
            bool(row.get("thought_contains_any_alias")) for row in rows
        ) / len(rows),
        "thought_contains_correct_alias_rate": sum(
            bool(row.get("thought_contains_correct_alias")) for row in rows
        ) / len(rows),
        "full_vocab_top_is_alias_rate": sum(bool(row["full_vocab_top_is_alias"]) for row in rows) / len(rows),
        "no_thought_slot_successes": no_thought_success,
        "no_thought_slot_accuracy": no_thought_accuracy,
        "real_minus_no_thought_accuracy": real_accuracy - no_thought_accuracy,
        "shuffled_thought_slot_successes": sum(bool(row["correct"]) for row in shuffled),
        "shuffled_thought_slot_accuracy": shuffled_accuracy,
        "shuffled_mean_correct_alias_probability": sum(
            float(row["correct_alias_probability"]) for row in shuffled
        ) / len(shuffled),
        "real_minus_shuffled_thought_accuracy": real_accuracy - shuffled_accuracy,
        "no_thought_mean_correct_alias_probability": sum(
            float(row["correct_alias_probability"]) for row in no_thought_rows
        ) / len(no_thought_rows),
        "close_only_parse_rate": sum(bool(row["parseable"]) for row in controls) / len(controls),
        "close_only_success_rate": sum(bool(row["correct"]) for row in controls) / len(controls),
        "close_only_answer_cap_rate": sum(row["answer_stopped_by"] == "answer_cap" for row in controls) / len(controls),
    }


def seam_gate(metrics: dict[str, Any], gates: dict[str, Any]) -> bool:
    return bool(
        float(gates["slot_success_rate_min"])
        <= metrics["slot_success_rate"]
        <= float(gates["slot_success_rate_max"])
        and metrics["mixed_slot_tasks"] >= int(gates["mixed_slot_tasks_min"])
        and metrics["real_minus_no_thought_accuracy"]
        >= float(gates["real_minus_no_thought_accuracy_min"])
        and metrics["real_minus_shuffled_thought_accuracy"]
        >= float(gates["real_minus_shuffled_thought_accuracy_min"])
        and metrics["finite_slot_rows_rate"] >= float(gates["finite_slot_rows_rate_min"])
    )


def observed_gate_reachability(metrics: dict[str, Any], gates: dict[str, Any]) -> dict[str, bool]:
    maximum = float(gates["slot_success_rate_max"])
    no_thought = bool(
        metrics["no_thought_slot_accuracy"]
        + float(gates["real_minus_no_thought_accuracy_min"])
        <= maximum + 1e-12
    )
    shuffled = bool(
        metrics["shuffled_thought_slot_accuracy"]
        + float(gates["real_minus_shuffled_thought_accuracy_min"])
        <= maximum + 1e-12
    )
    return {
        "no_thought_gain_gate_reachable_under_accuracy_ceiling": no_thought,
        "shuffled_gain_gate_reachable_under_accuracy_ceiling": shuffled,
        "all_observed_gain_gates_reachable": no_thought and shuffled,
    }


def _thought_prefix(model: Any, trace: dict[str, Any], cap: int) -> tuple[list[int] | None, str]:
    close_step = trace["close_step"]
    tokens = list(trace["generated_token_ids"])
    if close_step is not None and int(close_step) <= int(cap):
        return tokens[: int(close_step) - 1], "natural_prefix_replayed"
    if len(tokens) < cap or model.eos_id in tokens[:cap] or model.think_close_id in tokens[:cap]:
        return None, "malformed_pre_cap"
    return tokens[:cap], "forced_at_cap"


def _shuffled_thought(thought: list[int], seed: int) -> tuple[list[int], float]:
    """Return a deterministic token-multiset permutation and moved-position rate."""
    order = sorted(
        range(len(thought)),
        key=lambda index: hashlib.blake2b(
            f"{seed}\0{index}".encode(), digest_size=16
        ).digest(),
    )
    shuffled = [thought[index] for index in order]
    moved = sum(index != source for index, source in enumerate(order))
    return shuffled, moved / len(thought) if thought else 0.0


def _slot_row(
    model: Any,
    config: dict[str, Any],
    task: dict[str, Any],
    prepared: dict[str, Any],
    trace: dict[str, Any],
    *,
    trace_index: int,
    cap: int,
) -> tuple[dict[str, Any], list[int] | None]:
    aliases = dict(config["data"]["operation_aliases"])
    thought, mode = _thought_prefix(model, trace, cap)
    if thought is None:
        empty_alias_probabilities = {alias: 0.0 for alias in aliases.values()}
        return ({
            "task_id": task["task_id"],
            "trace_index": trace_index,
            "cap": cap,
            "commit_mode": mode,
            "finite": False,
            "chosen_alias": None,
            "correct": False,
            "correct_alias": aliases[task["first_op"]],
            "correct_alias_probability": 0.0,
            "constrained_margin": 0.0,
            "constrained_entropy": 0.0,
            "full_vocab_top_is_alias": False,
            "full_vocab_alias_probability_mass": 0.0,
            "alias_full_vocab_probabilities": empty_alias_probabilities,
            "thought_alias_mention_count": 0,
            "thought_contains_any_alias": False,
            "thought_contains_correct_alias": False,
            "thought_last_mentioned_alias": None,
            "thought_tokens": 0,
            "thought_token_ids_sha256": None,
        }, None)
    readout = model.slot_readout(
        prepared["input_ids"],
        thought,
        slot_text=str(config["slot"]["text"]),
        aliases=list(aliases.values()),
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
    )
    correct_alias = aliases[task["first_op"]]
    alias_by_id = {
        model.leading_space_token_id(alias): alias for alias in aliases.values()
    }
    mentioned_aliases = [alias_by_id[token] for token in thought if token in alias_by_id]
    return ({
        "task_id": task["task_id"],
        "trace_index": trace_index,
        "cap": cap,
        "commit_mode": mode,
        "correct_alias": correct_alias,
        "correct": readout["chosen_alias"] == correct_alias,
        "correct_alias_probability": readout["alias_probabilities"][correct_alias],
        "thought_alias_mention_count": len(mentioned_aliases),
        "thought_contains_any_alias": bool(mentioned_aliases),
        "thought_contains_correct_alias": correct_alias in mentioned_aliases,
        "thought_last_mentioned_alias": mentioned_aliases[-1] if mentioned_aliases else None,
        "thought_tokens": len(thought),
        "thought_token_ids_sha256": _token_ids_sha256(thought),
        **readout,
    }, thought)


def _freeform_row(
    model: Any,
    config: dict[str, Any],
    task: dict[str, Any],
    prepared: dict[str, Any],
    thought: list[int] | None,
    *,
    trace_index: int,
    cap: int,
    answer_seed: int,
) -> dict[str, Any]:
    aliases = dict(config["data"]["operation_aliases"])
    if thought is None:
        return {
            "task_id": task["task_id"], "trace_index": trace_index, "cap": cap,
            "parseable": False, "correct": False, "answer_tokens": 0,
            "answer_stopped_by": "unavailable", "answer_cache_contract_pass": True,
        }
    answer = model.force_commit(
        prepared["input_ids"], thought, seed=answer_seed, **_generation_kwargs(config)
    )
    parsed = parse_alias(answer["answer_text"], aliases)
    return {
        "task_id": task["task_id"],
        "trace_index": trace_index,
        "cap": cap,
        "answer_seed": answer_seed,
        "parseable": parsed is not None,
        "parsed_alias": parsed,
        "correct": bool(parsed is not None and verify_answer(task, answer["answer_text"], aliases)),
        "answer_tokens": answer["answer_tokens"],
        "answer_stopped_by": answer["stopped_by"],
        "answer_cache_contract_pass": answer["cache_contract_pass"],
        "answer_text": answer["answer_text"],
    }


def _generate_seam(
    config: dict[str, Any], *, split: str, caps: list[int], trace_seed_key: str,
    freeform_seed_key: str, shuffle_seed_key: str,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]
]:
    import torch

    model = _load_model(config)
    aliases = dict(config["data"]["operation_aliases"])
    tasks = read_jsonl(DATA_DIR / f"{split}.jsonl")
    trace_rows: list[dict[str, Any]] = []
    slot_rows: list[dict[str, Any]] = []
    shuffled_slot_rows: list[dict[str, Any]] = []
    freeform_rows: list[dict[str, Any]] = []
    no_thought_rows: list[dict[str, Any]] = []
    max_cap = max(caps)
    total = len(tasks) * int(config["generation"]["traces_per_task"])
    completed = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for task in tasks:
        prepared = model.prepare(
            task_prompt(task, aliases), prompt_max_tokens=int(config["generation"]["prompt_max_tokens"])
        )
        no_thought = model.slot_readout(
            prepared["input_ids"], [], slot_text=str(config["slot"]["text"]),
            aliases=list(aliases.values()), total_max_tokens=int(config["generation"]["total_max_tokens"]),
        )
        correct_alias = aliases[task["first_op"]]
        no_thought_rows.append({
            "task_id": task["task_id"],
            "correct_alias": correct_alias,
            "correct": no_thought["chosen_alias"] == correct_alias,
            "correct_alias_probability": no_thought["alias_probabilities"][correct_alias],
            **no_thought,
        })
        for trace_index in range(int(config["generation"]["traces_per_task"])):
            trace_seed = _stable_seed(
                int(config["seeds"][trace_seed_key]), task["task_id"], str(trace_index)
            )
            trace = model.generate_trace(
                prepared["input_ids"], seed=trace_seed, thought_cap=max_cap, **_generation_kwargs(config)
            )
            trace_rows.append({
                "task_id": task["task_id"], "trace_index": trace_index,
                "trace_seed": trace_seed, "prompt_tokens": prepared["prompt_tokens"], **trace,
            })
            for cap in caps:
                slot_row, thought = _slot_row(
                    model, config, task, prepared, trace, trace_index=trace_index, cap=cap
                )
                slot_rows.append(slot_row)
                if thought is None:
                    shuffled_slot_rows.append({
                        **slot_row,
                        "control": "shuffled_thought",
                        "shuffle_seed": None,
                        "shuffle_moved_position_rate": 0.0,
                    })
                else:
                    shuffle_seed = _stable_seed(
                        int(config["seeds"][shuffle_seed_key]),
                        task["task_id"], str(trace_index), str(cap),
                    )
                    shuffled_thought, moved_rate = _shuffled_thought(thought, shuffle_seed)
                    token_multiset_match = bool(
                        len(shuffled_thought) == len(thought)
                        and sorted(shuffled_thought) == sorted(thought)
                    )
                    if not token_multiset_match:
                        raise RuntimeError("shuffled thought changed the token multiset")
                    shuffled_readout = model.slot_readout(
                        prepared["input_ids"], shuffled_thought,
                        slot_text=str(config["slot"]["text"]),
                        aliases=list(aliases.values()),
                        total_max_tokens=int(config["generation"]["total_max_tokens"]),
                    )
                    shuffled_slot_rows.append({
                        "task_id": task["task_id"],
                        "trace_index": trace_index,
                        "cap": cap,
                        "commit_mode": slot_row["commit_mode"],
                        "control": "shuffled_thought",
                        "shuffle_seed": shuffle_seed,
                        "shuffle_moved_position_rate": moved_rate,
                        "source_thought_token_ids_sha256": _token_ids_sha256(thought),
                        "shuffled_thought_token_ids_sha256": _token_ids_sha256(
                            shuffled_thought
                        ),
                        "token_multiset_match": token_multiset_match,
                        "thought_tokens": len(shuffled_thought),
                        "correct_alias": correct_alias,
                        "correct": shuffled_readout["chosen_alias"] == correct_alias,
                        "correct_alias_probability": shuffled_readout[
                            "alias_probabilities"
                        ][correct_alias],
                        **shuffled_readout,
                    })
                answer_seed = _stable_seed(
                    int(config["seeds"][freeform_seed_key]), task["task_id"], str(trace_index), str(cap)
                )
                freeform_rows.append(_freeform_row(
                    model, config, task, prepared, thought,
                    trace_index=trace_index, cap=cap, answer_seed=answer_seed,
                ))
            completed += 1
            if completed % 4 == 0 or completed == total:
                print(f"{split}: {completed}/{total} traces", flush=True)
    if not all(row["cache_contract_pass"] for row in trace_rows):
        raise RuntimeError("a trace row failed the cache audit")
    if not all(row["answer_cache_contract_pass"] for row in freeform_rows):
        raise RuntimeError("a close-only control failed the cache audit")
    valid_shuffled_rows = [
        row for row in shuffled_slot_rows if row["commit_mode"] != "malformed_pre_cap"
    ]
    if not all(row["finite"] for row in valid_shuffled_rows):
        raise RuntimeError("a shuffled-thought slot control is nonfinite")
    if not all(row["finite"] for row in no_thought_rows):
        raise RuntimeError("a no-thought slot control is nonfinite")
    expected_slot_rows = total * len(caps)
    observed_counts = {
        "traces": len(trace_rows),
        "slots": len(slot_rows),
        "shuffled_slots": len(shuffled_slot_rows),
        "freeform": len(freeform_rows),
        "no_thought": len(no_thought_rows),
    }
    expected_counts = {
        "traces": total,
        "slots": expected_slot_rows,
        "shuffled_slots": expected_slot_rows,
        "freeform": expected_slot_rows,
        "no_thought": len(tasks),
    }
    if observed_counts != expected_counts:
        raise RuntimeError(
            f"incomplete seam rows: observed={observed_counts}, expected={expected_counts}"
        )
    environment = {
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "trace_sampled_tokens": sum(len(row["generated_token_ids"]) for row in trace_rows),
        "trace_forward_calls": sum(int(row["forward_calls"]) for row in trace_rows),
        "freeform_answer_tokens": sum(int(row["answer_tokens"]) for row in freeform_rows),
        "slot_prefill_tokens": sum(int(row.get("prefill_tokens", 0)) for row in slot_rows),
        "shuffled_slot_prefill_tokens": sum(
            int(row.get("prefill_tokens", 0)) for row in shuffled_slot_rows
        ),
        "row_counts": observed_counts,
    }
    return trace_rows, slot_rows, shuffled_slot_rows, freeform_rows, no_thought_rows, environment


def _write_seam_rows(
    prefix: str, traces: list[dict[str, Any]], slots: list[dict[str, Any]],
    shuffled_slots: list[dict[str, Any]], freeform: list[dict[str, Any]],
    no_thought: list[dict[str, Any]],
) -> dict[str, str]:
    paths = {
        "trace_rows": RUNS_DIR / f"{prefix}_traces.jsonl",
        "slot_rows": RUNS_DIR / f"{prefix}_slot_rows.jsonl",
        "shuffled_slot_rows": RUNS_DIR / f"{prefix}_shuffled_slot_rows.jsonl",
        "freeform_rows": RUNS_DIR / f"{prefix}_freeform_rows.jsonl",
        "no_thought_rows": RUNS_DIR / f"{prefix}_no_thought_rows.jsonl",
    }
    for path, rows in zip(
        paths.values(), (traces, slots, shuffled_slots, freeform, no_thought), strict=True
    ):
        write_jsonl(path, rows)
    return {f"{name}_sha256": sha256_file(path) for name, path in paths.items()}


def run_seam_selection(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    data = require_data_receipt(config)
    model_smoke_sha256 = _require_model_smoke()
    caps = [int(value) for value in config["generation"]["cap_rungs"]]
    traces, slots, shuffled_slots, freeform, no_thought, environment = _generate_seam(
        config, split="seam_selection", caps=caps,
        trace_seed_key="seam_selection_trace", freeform_seed_key="seam_selection_freeform",
        shuffle_seed_key="seam_selection_shuffle",
    )
    hashes = _write_seam_rows(
        "seam_selection", traces, slots, shuffled_slots, freeform, no_thought
    )
    task_ids = [row["task_id"] for row in read_jsonl(DATA_DIR / "seam_selection.jsonl")]
    metrics = []
    selected = None
    for cap in caps:
        value = slot_metrics(slots, shuffled_slots, no_thought, freeform, task_ids, cap)
        value["observed_gate_reachability"] = observed_gate_reachability(
            value, config["gates"]["seam_selection"]
        )
        value["gate_pass"] = seam_gate(value, config["gates"]["seam_selection"])
        metrics.append(value)
        if selected is None and value["gate_pass"]:
            selected = cap
    result = {
        "schema_version": 1,
        "stage": "seam_selection",
        "scientific_result": True,
        "passed": selected is not None,
        "decision": "COMMIT_SLOT_CAP_SELECTED" if selected is not None else "COMMIT_SLOT_SEAM_FAIL",
        "counterfactual_policy": True,
        "syntax_supplied_answer_identity_not_supplied": True,
        "design_passed": design["passed"],
        "model_smoke_sha256": model_smoke_sha256,
        "data_receipt": data,
        "selected_cap": selected,
        "selection_policy": "smallest passing cap",
        "items": len(task_ids),
        "traces": len(traces),
        "metrics_by_cap": metrics,
        **hashes,
        **environment,
    }
    write_json(RUNS_DIR / "seam_selection.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_seam_confirmation(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    data = require_data_receipt(config)
    model_smoke_sha256 = _require_model_smoke()
    selection_path = RUNS_DIR / "seam_selection.json"
    if not selection_path.exists():
        raise RuntimeError("commit-slot cap selection has not run")
    selection = read_json(selection_path)
    if selection.get("passed") is not True or selection.get("selected_cap") is None:
        raise RuntimeError("no commit-slot cap was selected")
    for name in (
        "trace_rows", "slot_rows", "shuffled_slot_rows", "freeform_rows",
        "no_thought_rows",
    ):
        path = {
            "trace_rows": RUNS_DIR / "seam_selection_traces.jsonl",
            "slot_rows": RUNS_DIR / "seam_selection_slot_rows.jsonl",
            "shuffled_slot_rows": RUNS_DIR / "seam_selection_shuffled_slot_rows.jsonl",
            "freeform_rows": RUNS_DIR / "seam_selection_freeform_rows.jsonl",
            "no_thought_rows": RUNS_DIR / "seam_selection_no_thought_rows.jsonl",
        }[name]
        if sha256_file(path) != selection[f"{name}_sha256"]:
            raise RuntimeError(f"selection {name} changed after cap freeze")
    cap = int(selection["selected_cap"])
    traces, slots, shuffled_slots, freeform, no_thought, environment = _generate_seam(
        config, split="seam_confirmation", caps=[cap],
        trace_seed_key="seam_confirmation_trace", freeform_seed_key="seam_confirmation_freeform",
        shuffle_seed_key="seam_confirmation_shuffle",
    )
    hashes = _write_seam_rows(
        "seam_confirmation", traces, slots, shuffled_slots, freeform, no_thought
    )
    task_ids = [row["task_id"] for row in read_jsonl(DATA_DIR / "seam_confirmation.jsonl")]
    metrics = slot_metrics(
        slots, shuffled_slots, no_thought, freeform, task_ids, cap
    )
    metrics["observed_gate_reachability"] = observed_gate_reachability(
        metrics, config["gates"]["seam_confirmation"]
    )
    passed = bool(design["passed"] and seam_gate(metrics, config["gates"]["seam_confirmation"]))
    result = {
        "schema_version": 1,
        "stage": "seam_confirmation",
        "scientific_result": True,
        "passed": passed,
        "decision": "COMMIT_SLOT_SEAM_REPLICATED" if passed else "COMMIT_SLOT_SEAM_NOT_REPLICATED",
        "counterfactual_policy": True,
        "syntax_supplied_answer_identity_not_supplied": True,
        "model_smoke_sha256": model_smoke_sha256,
        "data_receipt": data,
        "selected_cap": cap,
        "items": len(task_ids),
        "traces": len(traces),
        "metrics": metrics,
        "selection_summary_sha256": sha256_file(selection_path),
        **hashes,
        **environment,
    }
    write_json(RUNS_DIR / "seam_confirmation.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def unavailable(stage: str) -> None:
    raise RuntimeError(f"stage {stage!r} is not implemented; refusing a placeholder result")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=(
            "smoke",
            "model-smoke",
            "seam-selection",
            "seam-confirmation",
            "prefix-value",
            "control-calibration",
            "causal-confirmation",
        ),
        default="smoke",
    )
    args = parser.parse_args()
    config = load_config()
    if args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "model-smoke":
        run_model_smoke(config)
    elif args.stage == "seam-selection":
        run_seam_selection(config)
    elif args.stage == "seam-confirmation":
        run_seam_confirmation(config)
    else:
        unavailable(args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
