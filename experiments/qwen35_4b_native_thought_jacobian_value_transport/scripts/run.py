#!/usr/bin/env python3
"""Stage-gated native-thought Jacobian value-transport harness."""

from __future__ import annotations

import argparse
import hashlib
import json
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
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _git(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    ancestor = (
        _git(["merge-base", "--is-ancestor", commit, head], check=False).returncode
        == 0
    )
    paths = {
        "readme": (
            "experiments/qwen35_4b_native_thought_jacobian_value_transport/README.md"
        ),
        "preregistration": (
            "experiments/qwen35_4b_native_thought_jacobian_value_transport/"
            "reports/preregistration.md"
        ),
    }
    observed = {
        name: hashlib.sha256(
            _git(["show", f"{commit}:{path}"]).stdout.encode("utf-8")
        ).hexdigest()
        for name, path in paths.items()
    }
    expected = {
        "readme": str(boundary["readme_sha256"]),
        "preregistration": str(boundary["preregistration_sha256"]),
    }
    lens_hash = sha256_file(EXP / config["lens"]["path"])
    passed = bool(
        ancestor
        and observed == expected
        and lens_hash == str(config["lens"]["sha256"])
    )
    result = {
        "schema_version": 1,
        "stage": "design_boundary",
        "passed": passed,
        "scientific_result": False,
        "design_commit": commit,
        "head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "lens_observed_sha256": lens_hash,
        "lens_expected_sha256": str(config["lens"]["sha256"]),
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", result)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {result}")
    return result


def _stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def _generation_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    generation = config["generation"]
    return {
        "max_think_tokens": int(generation["max_think_tokens"]),
        "max_answer_tokens": int(generation["max_answer_tokens"]),
        "temperature": float(generation["temperature"]),
        "top_p": float(generation["top_p"]),
        "top_k": int(generation["top_k"]),
    }


def _load_model_lens(config: dict[str, Any]):
    from model_ops import ContextLens, QwenThinkModel

    model = QwenThinkModel(config)
    lens = ContextLens.load(str(EXP / config["lens"]["path"]))
    return model, lens


def _parent_fingerprints() -> set[str]:
    directory = ROOT / "experiments" / "qwen35_4b_jacobian_value_transport" / "data" / "procedural"
    values = set()
    for path in sorted(directory.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                values.add(task_fingerprint(row))
    return values


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    observed_lens_hash = sha256_file(lens_path)
    if observed_lens_hash != config["lens"]["sha256"]:
        raise RuntimeError("frozen replicated lens hash mismatch")
    import torch

    lens_state = torch.load(lens_path, map_location="cpu", weights_only=True)
    lens_concepts = tuple(str(value) for value in lens_state["concepts"])
    aliases = dict(config["data"]["operation_aliases"])
    if (
        len(set(aliases.values())) != len(aliases)
        or not set(aliases.values()).issubset(lens_concepts)
    ):
        raise RuntimeError("operation aliases must be unique frozen-lens concepts")
    splits = build_splits(config)
    parent = _parent_fingerprints()
    all_fingerprints: list[str] = []
    split_receipts = {}
    for split, rows in splits.items():
        path = DATA_DIR / f"{split}.jsonl"
        write_jsonl(path, rows)
        fingerprints = [task_fingerprint(row) for row in rows]
        all_fingerprints.extend(fingerprints)
        counts = Counter(str(row["first_op"]) for row in rows)
        split_receipts[split] = {
            "items": len(rows),
            "path": str(path.relative_to(EXP)),
            "sha256": sha256_file(path),
            "unique_fingerprints": len(set(fingerprints)),
            "parent_overlap": sum(value in parent for value in fingerprints),
            "first_op_counts": {
                name: counts[name] for name in IDENTIFIABLE_FIRST_OPERATIONS
            },
        }
    if len(all_fingerprints) != len(set(all_fingerprints)):
        raise RuntimeError("procedural split fingerprints overlap")
    if any(receipt["parent_overlap"] for receipt in split_receipts.values()):
        raise RuntimeError("fresh tasks overlap the direct Jacobian parent")
    manifest = {
        "schema_version": 1,
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "frozen_lens_sha256": observed_lens_hash,
        "frozen_lens_concepts": len(lens_concepts),
        "operation_aliases": aliases,
        "identifiable_first_operations": list(IDENTIFIABLE_FIRST_OPERATIONS),
        "parent_fingerprint_count": len(parent),
        "total_unique_fingerprints": len(set(all_fingerprints)),
        "splits": split_receipts,
        "firewall": {
            "benchmark_content_used": False,
            "fresh_procedural_only": True,
            "visible_first_type_identifiable_by_exhaustive_enumeration": True,
        },
    }
    write_json(DATA_DIR / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "scientific_result": False,
        "lens_sha256": observed_lens_hash,
        "items": len(all_fingerprints),
        "unique_fingerprints": len(set(all_fingerprints)),
        "parent_overlap": 0,
        "split_sizes": {name: len(rows) for name, rows in splits.items()},
        "first_target_types": len(IDENTIFIABLE_FIRST_OPERATIONS),
        "alias_count": len(aliases),
    }
    write_json(RUNS_DIR / "smoke" / "data_receipt.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_model_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    import torch
    import transformers

    from coordinates import dictionary_stats, read_coordinates

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, lens = _load_model_lens(config)
    band = tuple(int(layer) for layer in config["lens"]["band"])
    if lens.concepts != tuple(
        read_json(DATA_DIR / "manifest.json")["operation_aliases"].values()
    ):
        # Aliases intentionally use a subset, while the lens retains all 24.
        aliases = set(config["data"]["operation_aliases"].values())
        if not aliases.issubset(lens.concepts):
            raise RuntimeError("alias concepts are absent from the frozen lens")
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    stats = {
        layer: dictionary_stats(lens.directions[layer], rtol=rtol) for layer in band
    }
    alias_token_ids = {
        alias: model.concept_token_id(alias)
        for alias in config["data"]["operation_aliases"].values()
    }
    if len(set(alias_token_ids.values())) != len(alias_token_ids):
        raise RuntimeError("alias token IDs are not one-to-one")
    tasks = read_jsonl(DATA_DIR / "seam_calibration.jsonl")[:2]
    rows = []
    causal_differences = []
    for task in tasks:
        prepared = model.prepare_thinking(
            task_prompt(task, config["data"]["operation_aliases"]),
            max_sequence_tokens=int(config["generation"]["max_sequence_tokens"]),
        )
        generated = model.generate_full_recompute(
            prepared["input_ids"],
            seed=_stable_seed(
                int(config["seeds"]["seam_generation"]), task["task_id"], "smoke"
            ),
            **_generation_kwargs(config),
        )
        full_ids = generated["input_ids"]
        open_position = int(generated["think_open_position"])
        available = max(0, int(generated["think_tokens"]))
        if available < 2:
            raise RuntimeError("model smoke generated fewer than two thought tokens")
        checkpoint_count = max(1, min(available - 1, available // 2))
        checkpoint_position = open_position + checkpoint_count
        prefix_ids = full_ids[:, : checkpoint_position + 1]
        prefix_activations = model.capture_ids(
            prefix_ids, position=checkpoint_position, layers=band
        )
        extended_activations = model.capture_ids(
            full_ids, position=checkpoint_position, layers=band
        )
        causal_differences.extend(
            float((prefix_activations[layer] - extended_activations[layer]).abs().max())
            for layer in band
        )
        coordinate_finite = all(
            bool(torch.isfinite(read_coordinates(
                prefix_activations[layer].reshape(1, -1),
                lens.directions[layer],
                rtol=rtol,
            )).all())
            for layer in band
        )
        rows.append({
            "task_id": task["task_id"],
            "prompt_tokens": int(prepared["prompt_tokens"]),
            "generated_tokens": len(generated["generated_token_ids"]),
            "think_tokens": available,
            "natural_close": bool(generated["natural_close"]),
            "answer_parseable": parse_alias(
                generated["answer_text"], config["data"]["operation_aliases"]
            ) is not None,
            "checkpoint_position": checkpoint_position,
            "coordinate_finite": coordinate_finite,
            "outcome_correctness_recorded": False,
        })
    causal_max = max(causal_differences, default=float("inf"))
    passed = bool(
        design["passed"]
        and model.n_layers == 32
        and model.d_model == 2560
        and all(stat.effective_rank == 24 for stat in stats.values())
        and all(row["coordinate_finite"] for row in rows)
        and causal_max <= float(config["controls"]["causal_activation_atol"])
    )
    result = {
        "schema_version": 1,
        "stage": "model_smoke",
        "passed": passed,
        "scientific_result": False,
        "outcomes_recorded": False,
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
        "think_token_ids": {
            "open": model.think_open_id,
            "close": model.think_close_id,
            "eos": model.eos_id,
        },
        "alias_token_ids": alias_token_ids,
        "lens_effective_ranks": {
            str(layer): stats[layer].effective_rank for layer in band
        },
        "causal_activation_max_abs": causal_max,
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json(RUNS_DIR / "model_smoke" / "result.json", result)
    if not passed:
        raise RuntimeError(f"model smoke failed: {result}")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def run_seam_calibration(config: dict[str, Any]) -> dict[str, Any]:
    design = design_boundary_receipt(config)
    smoke_path = RUNS_DIR / "model_smoke" / "result.json"
    if not smoke_path.exists() or read_json(smoke_path).get("passed") is not True:
        raise RuntimeError("passing model smoke is required before seam calibration")
    import torch

    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    model, _lens = _load_model_lens(config)
    tasks = read_jsonl(DATA_DIR / "seam_calibration.jsonl")
    aliases = config["data"]["operation_aliases"]
    rows = []
    for task in tasks:
        prepared = model.prepare_thinking(
            task_prompt(task, aliases),
            max_sequence_tokens=int(config["generation"]["max_sequence_tokens"]),
        )
        for trace_index in range(int(config["generation"]["traces_per_task"])):
            seed = _stable_seed(
                int(config["seeds"]["seam_generation"]),
                task["task_id"],
                str(trace_index),
            )
            generated = model.generate_full_recompute(
                prepared["input_ids"], seed=seed, **_generation_kwargs(config)
            )
            parsed = parse_alias(generated["answer_text"], aliases)
            correct = bool(
                generated["natural_close"]
                and parsed is not None
                and verify_answer(task, generated["answer_text"], aliases)
            )
            rows.append({
                "task_id": task["task_id"],
                "trace_index": trace_index,
                "seed": seed,
                "natural_close": bool(generated["natural_close"]),
                "parseable": parsed is not None,
                "parsed_alias": parsed,
                "correct": correct,
                "think_tokens": int(generated["think_tokens"]),
                "answer_tokens": int(generated["answer_tokens"]),
                "stopped_by": generated["stopped_by"],
                "forward_calls": int(generated["forward_calls"]),
                "generated_token_ids": generated["generated_token_ids"],
                "generated_text": generated["generated_text"],
            })
    n = len(rows)
    natural_close_rate = sum(row["natural_close"] for row in rows) / n
    parse_rate = sum(row["parseable"] for row in rows) / n
    success_rate = sum(row["correct"] for row in rows) / n
    by_task = {
        task["task_id"]: [
            row for row in rows if row["task_id"] == task["task_id"]
        ]
        for task in tasks
    }
    mixed_tasks = sum(
        any(row["correct"] for row in task_rows)
        and any(not row["correct"] for row in task_rows)
        for task_rows in by_task.values()
    )
    gates = config["gates"]["seam"]
    passed = bool(
        design["passed"]
        and natural_close_rate >= float(gates["natural_close_rate_min"])
        and parse_rate >= float(gates["parse_rate_min"])
        and float(gates["success_rate_min"])
        <= success_rate
        <= float(gates["success_rate_max"])
        and mixed_tasks >= int(gates["mixed_tasks_min"])
    )
    result = {
        "schema_version": 1,
        "stage": "seam_calibration",
        "passed": passed,
        "decision": "NATURAL_SEAM_PASS" if passed else "NO_NATURAL_SEAM",
        "scientific_result": True,
        "items": len(tasks),
        "traces": n,
        "natural_close_rate": natural_close_rate,
        "parse_rate": parse_rate,
        "success_rate": success_rate,
        "mixed_tasks": mixed_tasks,
        "mean_think_tokens": sum(row["think_tokens"] for row in rows) / n,
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    write_jsonl(RUNS_DIR / "seam_calibration_rows.jsonl", rows)
    write_json(RUNS_DIR / "seam_calibration.json", result)
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
            "seam-calibration",
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
        return 0
    if args.stage == "model-smoke":
        run_model_smoke(config)
        return 0
    if args.stage == "seam-calibration":
        run_seam_calibration(config)
        return 0
    unavailable(args.stage)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
