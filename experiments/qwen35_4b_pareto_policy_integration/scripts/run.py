#!/usr/bin/env python3
"""Resumable orchestration for the corrected Pareto-policy integration study."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path


sys.dont_write_bytecode = True
EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PY = REPO / ".venv" / "bin" / "python"
VLLM_PY = REPO / ".venv-vllm" / "bin" / "python"
sys.path.insert(0, str(EXP / "src"))

from gym.families import ALL_FAMILIES  # noqa: E402
from io_utils import (  # noqa: E402
    canonical_hash,
    load_config,
    resolve_repo_path,
    read_jsonl,
    sha256_file,
    write_json,
    write_jsonl,
)


def _run(command: list[str], *, training: bool = False, allowed: tuple[int, ...] = (0,)) -> int:
    print("[stage] " + " ".join(command), flush=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if training:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(command, cwd=REPO, env=env, check=False)
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _paths(config: dict) -> dict[str, Path]:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    return {
        "root": root,
        "quick_adapter": root / "adapters" / "quick_blend",
        "quick": root / "merged" / "quick_blend",
        "deep_adapter": root / "adapters" / "deep_apex",
        "deep": root / "merged" / "deep_apex",
    }


def _checkpoint_complete(adapter: Path, merged: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
            adapter / "training_receipt.json",
            merged / "config.json",
            merged / "merge_receipt.json",
        )
    )


def _require_gate(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"required gate receipt missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("gate", {}).get("passed"):
        raise SystemExit(f"required gate did not pass: {path}")
    return payload


def _verify_preregistration() -> dict:
    path = EXP / "runs" / "preregistration_receipt.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "locked":
        raise SystemExit("preregistration receipt is not locked")
    for relative, expected in payload.get("frozen_files", {}).items():
        actual = sha256_file(EXP / relative)
        if actual != expected:
            raise SystemExit(
                f"frozen preregistration file changed: {relative} {actual} != {expected}"
            )
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", payload["design_commit"], "HEAD"],
        cwd=REPO,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit("design commit is not an ancestor of HEAD")
    return payload


def _record_checkpoint(tag: str, adapter: Path, merged: Path) -> None:
    receipt = {
        "tag": tag,
        "adapter": str(adapter.resolve()),
        "merged": str(merged.resolve()),
        "training_receipt": json.loads((adapter / "training_receipt.json").read_text()),
        "merge_receipt": json.loads((merged / "merge_receipt.json").read_text()),
    }
    path = EXP / "runs" / "checkpoint_receipts.jsonl"
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    existing = [row for row in existing if row.get("tag") != tag] + [receipt]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in existing),
        encoding="utf-8",
    )


def scientific_smoke(config: dict, config_path: Path) -> dict:
    for path in sorted(list((EXP / "src").rglob("*.py")) + list((EXP / "scripts").glob("*.py"))):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    _run([str(PY), "-m", "unittest", "discover", "-s", str(EXP / "tests"), "-v"])
    completed = subprocess.run(
        [str(PY), str(EXP / "scripts" / "selftest_gym.py")],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    expected = list(config["strata"]["trained_families"]) + list(
        config["strata"]["transfer_families"]
    )
    if expected != list(ALL_FAMILIES):
        raise AssertionError(f"config/registry family mismatch: {expected} != {list(ALL_FAMILIES)}")
    transfer = set(config["strata"]["transfer_families"])
    dataset_receipts = {}
    for key in ("quick_data", "deep_data"):
        path = resolve_repo_path(config["model"][key])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        exposed = sorted({str(row.get("family")) for row in rows} & transfer)
        if exposed:
            raise AssertionError(f"{key} leaks transfer families: {exposed}")
        dataset_receipts[key] = {
            "path": str(path.relative_to(REPO)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "families": sorted({str(row.get("family")) for row in rows}),
        }
    payload = {
        "status": "pass",
        "config": str(config_path.relative_to(EXP)),
        "config_sha256": canonical_hash(config),
        "model": config["model"]["id"],
        "revision": config["model"]["revision"],
        "trained_families": list(config["strata"]["trained_families"]),
        "transfer_families": list(config["strata"]["transfer_families"]),
        "retention_anchor_families": list(config["strata"]["retention_anchor_families"]),
        "specialist_delta_threshold": config["decision"]["specialist_delta_threshold"],
        "datasets": dataset_receipts,
        "selftest_tail": completed.stdout.strip().splitlines()[-3:],
    }
    write_json(EXP / "runs" / "smoke" / "summary.json", payload)
    return payload


def _train_specialist(tag: str, data: Path, adapter: Path, merged: Path, config: dict) -> None:
    if _checkpoint_complete(adapter, merged):
        print(f"[resume] specialist {tag} already complete", flush=True)
        return
    if adapter.exists() or merged.exists():
        raise SystemExit(f"partial checkpoint exists for {tag}: {adapter} / {merged}")
    cfg = config["specialist_train"]
    _run(
        [
            str(PY), str(EXP / "scripts" / "train_specialist.py"),
            "--train", str(data), "--out", str(adapter),
            "--epochs", str(cfg["epochs"]), "--lr", str(cfg["learning_rate"]),
            "--rank", str(cfg["rank"]), "--alpha", str(cfg["alpha"]),
            "--batch-size", str(cfg["batch_size"]), "--grad-accum", str(cfg["grad_accum"]),
            "--max-length", str(cfg["max_length"]), "--w-think", str(cfg["think_loss_weight"]),
            "--seed", str(cfg["seed"]),
        ],
        training=True,
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", config["model"]["id"], "--adapter", str(adapter),
            "--out", str(merged),
        ],
        training=True,
    )
    training = json.loads((adapter / "training_receipt.json").read_text())
    merge = json.loads((merged / "merge_receipt.json").read_text())
    if int(training.get("global_step", 0)) < 1:
        raise SystemExit(f"{tag} training completed zero steps")
    if int(merge.get("nonzero_lora_modules", 0)) != int(merge.get("applied_lora_modules", -1)):
        raise SystemExit(f"{tag} merge contains zero deltas")
    _record_checkpoint(tag, adapter, merged)


def _model_smoke(config: dict) -> None:
    root = resolve_repo_path(config["model"]["artifacts_root"]) / "smoke"
    adapter, merged = root / "adapter", root / "merged"
    base_output = EXP / "runs" / "model_smoke" / "base.jsonl"
    hf_output = EXP / "runs" / "model_smoke" / "hf.json"
    local_output = EXP / "runs" / "model_smoke" / "merged.jsonl"
    local_meta = local_output.with_name(local_output.name + ".meta.json")
    checkpoint_ready = _checkpoint_complete(adapter, merged)
    if checkpoint_ready and hf_output.exists() and local_meta.exists():
        print("[resume] model smoke already complete", flush=True)
        return
    if not checkpoint_ready and (adapter.exists() or merged.exists()):
        raise SystemExit(f"partial model-smoke checkpoint exists: {adapter} / {merged}")
    common = [
        "--smoke", "4", "--thinking", "off", "--greedy", "--max-tokens", "32",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
        "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
        "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
        "--cudagraph-capture-size", "16",
    ]
    if not hf_output.exists():
        _run(
            [str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"), "--output", str(base_output), *common]
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "model_smoke.py"),
                "--vllm-output", str(base_output), "--out", str(hf_output),
            ]
        )
    if not checkpoint_ready:
        cfg = config["specialist_train"]
        _run(
            [
                str(PY), str(EXP / "scripts" / "train_specialist.py"),
                "--train", str(resolve_repo_path(config["model"]["quick_data"])),
                "--out", str(adapter), "--epochs", "1", "--lr", str(cfg["learning_rate"]),
                "--rank", str(cfg["rank"]), "--alpha", str(cfg["alpha"]),
                "--batch-size", "1", "--grad-accum", "1", "--max-length", str(cfg["max_length"]),
                "--w-think", str(cfg["think_loss_weight"]), "--seed", str(cfg["seed"]), "--smoke",
            ],
            training=True,
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_adapter.py"),
                "--base-model", config["model"]["id"], "--adapter", str(adapter), "--out", str(merged),
            ],
            training=True,
        )
    _run(
        [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
            "--output", str(local_output), "--model-override", str(merged), *common,
        ]
    )
    metadata = json.loads(local_meta.read_text())
    merge = json.loads((merged / "merge_receipt.json").read_text())
    if metadata.get("model") != str(merged.resolve()):
        raise SystemExit("model smoke did not load local merged composite")
    if int(merge.get("nonzero_lora_modules", 0)) != int(merge.get("applied_lora_modules", -1)):
        raise SystemExit("model smoke merge was partially or wholly zero")
    _record_checkpoint("model_smoke", adapter, merged)


def _specialists(config: dict) -> None:
    paths = _paths(config)
    _train_specialist(
        "quick_blend", resolve_repo_path(config["model"]["quick_data"]),
        paths["quick_adapter"], paths["quick"], config,
    )
    _train_specialist(
        "deep_apex", resolve_repo_path(config["model"]["deep_data"]),
        paths["deep_adapter"], paths["deep"], config,
    )


def _eval_if_needed(
    *, model: Path, tag: str, scope: str, block_seed: int, config_path: Path,
    decode: str = "greedy",
) -> Path:
    out = EXP / "runs" / "policy_eval" / tag
    scores = out / "scores.json"
    if scores.exists():
        payload = json.loads(scores.read_text())
        if (
            payload.get("scope") == scope
            and int(payload.get("block_seed", -1)) == block_seed
            and payload.get("model") == str(model.resolve())
            and payload.get("decode") == decode
        ):
            print(f"[resume] evaluation {tag} already complete", flush=True)
            return scores
        raise SystemExit(f"stale evaluation exists: {out}")
    _run(
        [
            str(VLLM_PY), str(EXP / "scripts" / "eval_policy.py"),
            "--config", str(config_path), "--model", str(model), "--tag", tag,
            "--scope", scope, "--block-seed", str(block_seed), "--out-dir", str(out),
            "--decode", decode,
        ]
    )
    return scores


def _qualify(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "calibration.json")
    paths = _paths(config)
    if not _checkpoint_complete(paths["quick_adapter"], paths["quick"]):
        raise SystemExit("quick specialist checkpoint incomplete")
    if not _checkpoint_complete(paths["deep_adapter"], paths["deep"]):
        raise SystemExit("deep specialist checkpoint incomplete")
    quick_scores, deep_scores = [], []
    for index, seed in enumerate(config["seeds"]["qualification_blocks"]):
        quick_scores.append(
            _eval_if_needed(
                model=paths["quick"], tag=f"quick_qualification_b{index}",
                scope="qualification", block_seed=int(seed), config_path=config_path,
            )
        )
        deep_scores.append(
            _eval_if_needed(
                model=paths["deep"], tag=f"deep_qualification_b{index}",
                scope="qualification", block_seed=int(seed), config_path=config_path,
            )
        )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_qualification.py"),
            "--config", str(config_path),
            "--quick-policy-scores", *(str(path) for path in quick_scores),
            "--deep-policy-scores", *(str(path) for path in deep_scores),
            "--calibration", str(EXP / "analysis" / "calibration.json"),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "specialist_qualification.json")


def _calibrate(config: dict, config_path: Path) -> None:
    paths = _paths(config)
    if not _checkpoint_complete(paths["quick_adapter"], paths["quick"]):
        raise SystemExit("quick specialist checkpoint incomplete")
    if not _checkpoint_complete(paths["deep_adapter"], paths["deep"]):
        raise SystemExit("deep specialist checkpoint incomplete")
    seed = int(config["seeds"]["calibration"])
    quick = _eval_if_needed(
        model=paths["quick"], tag="quick_calibration", scope="calibration",
        block_seed=seed, config_path=config_path,
    )
    deep = _eval_if_needed(
        model=paths["deep"], tag="deep_calibration", scope="calibration",
        block_seed=seed, config_path=config_path,
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_calibration.py"),
            "--config", str(config_path), "--quick", str(quick), "--deep", str(deep),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "calibration.json")


def _teacher_audit(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "specialist_qualification.json")
    paths = _paths(config)
    root = EXP / "runs" / "teacher_audit"
    input_path = root / "prefixes.jsonl"
    items_path = root / "items.jsonl"
    quick_output = root / "quick.jsonl"
    deep_output = root / "deep.jsonl"
    if not input_path.exists() or not items_path.exists():
        root.mkdir(parents=True, exist_ok=True)
        _run(
            [
                str(PY), str(EXP / "scripts" / "build_teacher_audit.py"),
                "--config", str(config_path),
                "--atom-rows", str(
                    EXP / "runs" / "policy_eval" / "quick_qualification_b0" / "atom_rows.jsonl.gz"
                ),
                "--input-out", str(input_path), "--items-out", str(items_path),
            ]
        )
    remaining_budget = int(config["evaluation"]["thinking_budget"]) - 128
    common = [
        "--input", str(input_path), "--thinking", "budget",
        "--thinking-budget", str(remaining_budget),
        "--answer-max-tokens", str(config["evaluation"]["answer_max_tokens"]),
        "--n", str(config["teacher_audit"]["continuations_per_branch"]),
        "--temperature", str(config["evaluation"]["sample_temperature"]),
        "--top-p", str(config["evaluation"]["sample_top_p"]),
        "--top-k", str(config["teacher_audit"]["branch_top_k"]),
        "--seed", str(int(config["seeds"]["qualification_blocks"][0]) + 777),
        "--allow-custom-prompts", "--include-prompt-token-ids",
        "--max-model-len", str(config["engine"]["max_model_len"]),
        "--gpu-memory-utilization", str(config["engine"]["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["engine"]["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["engine"]["max_num_batched_tokens"]),
    ]
    for size in config["engine"]["cudagraph_capture_sizes"]:
        common.extend(("--cudagraph-capture-size", str(size)))
    for model, output in ((paths["quick"], quick_output), (paths["deep"], deep_output)):
        metadata = output.with_name(output.name + ".meta.json")
        if output.exists() and metadata.exists():
            payload = json.loads(metadata.read_text())
            if payload.get("model") == str(model.resolve()):
                print(f"[resume] teacher audit output {output.name} already complete", flush=True)
                continue
            raise SystemExit(f"stale teacher audit output: {output}")
        _run(
            [
                str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
                "--output", str(output), "--model-override", str(model), *common,
            ]
        )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_teacher_audit.py"),
            "--config", str(config_path), "--items", str(items_path),
            "--quick-output", str(quick_output), "--deep-output", str(deep_output),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "teacher_audit.json")


def _generation_engine_args(config: dict) -> list[str]:
    values = [
        "--max-model-len", str(config["engine"]["max_model_len"]),
        "--gpu-memory-utilization", str(config["engine"]["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["engine"]["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["engine"]["max_num_batched_tokens"]),
    ]
    for size in config["engine"]["cudagraph_capture_sizes"]:
        values.extend(("--cudagraph-capture-size", str(size)))
    return values


def _mopd_round(
    *, config: dict, config_path: Path, paths: dict[str, Path], arm: str,
    seed: int, round_index: int, current: Path, routing: str,
    rollout_source: str = "student", updates_override: int | None = None,
    selection_seed_base: int | None = None,
    target_initial_loss: float | None = None,
) -> Path:
    prompt_path = EXP / "data" / "mopd_prompts" / f"{arm}_s{seed}_r{round_index}.jsonl"
    if not prompt_path.exists():
        _run(
            [
                str(PY), str(EXP / "scripts" / "build_rollout_prompts.py"),
                "--config", str(config_path), "--round", str(round_index),
                "--out", str(prompt_path),
            ]
        )
    round_root = EXP / "runs" / "mopd" / arm / f"seed_{seed}" / f"round_{round_index}"
    rollout_path = round_root / "student_rollouts.jsonl"
    rollout_meta = rollout_path.with_name(rollout_path.name + ".meta.json")
    routed_receipt = rollout_path.with_name(rollout_path.name + ".routed_receipt.json")
    optimizer_updates = int(
        updates_override
        if updates_override is not None
        else config["mopd"]["updates_per_round"]
    )
    selection_seed = int(
        (selection_seed_base if selection_seed_base is not None else seed) + round_index
    )
    if rollout_source == "student" and rollout_path.exists() and rollout_meta.exists():
        meta = json.loads(rollout_meta.read_text())
        if (
            meta.get("model") != str(current.resolve())
            or int(meta.get("sampling", {}).get("n", -1)) != 2
        ):
            raise SystemExit(f"stale rollout policy at {rollout_path}")
        print(f"[resume] MOPD {arm} round {round_index} rollouts", flush=True)
    elif rollout_source == "teacher_routed" and rollout_path.exists() and routed_receipt.exists():
        receipt = json.loads(routed_receipt.read_text())
        if (
            receipt.get("prompt_sha256") != sha256_file(prompt_path)
            or int(receipt.get("samples_per_prompt", -1)) != 2
        ):
            raise SystemExit(f"stale routed rollout prompts at {rollout_path}")
        print(f"[resume] MOPD {arm} round {round_index} routed rollouts", flush=True)
    elif rollout_source == "student":
        _run(
            [
                str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
                "--input", str(prompt_path), "--output", str(rollout_path),
                "--model-override", str(current), "--thinking", "budget",
                "--thinking-budget", str(config["mopd"]["rollout_thinking_budget"]),
                "--answer-max-tokens", str(config["mopd"]["rollout_answer_max_tokens"]),
                "--n", "2", "--temperature", str(config["mopd"]["rollout_temperature"]),
                "--top-p", str(config["mopd"]["rollout_top_p"]),
                "--top-k", str(config["mopd"]["rollout_top_k"]),
                "--seed", str(config["seeds"]["rollout_rounds"][round_index]),
                "--include-prompt-token-ids", *_generation_engine_args(config),
            ]
        )
    elif rollout_source == "teacher_routed":
        prompts = read_jsonl(prompt_path)
        partial_outputs = []
        partial_meta = []
        for stratum, model in (("quick", paths["quick"]), ("deep", paths["deep"])):
            subset = [row for row in prompts if row["meta"]["stratum"] == stratum]
            subset_path = round_root / f"{stratum}_prompts.jsonl"
            output_path = round_root / f"{stratum}_teacher_rollouts.jsonl"
            write_jsonl(subset_path, subset)
            _run(
                [
                    str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
                    "--input", str(subset_path), "--output", str(output_path),
                    "--model-override", str(model), "--thinking", "budget",
                    "--thinking-budget", str(config["mopd"]["rollout_thinking_budget"]),
                    "--answer-max-tokens", str(config["mopd"]["rollout_answer_max_tokens"]),
                    "--n", "2", "--temperature", str(config["mopd"]["rollout_temperature"]),
                    "--top-p", str(config["mopd"]["rollout_top_p"]),
                    "--top-k", str(config["mopd"]["rollout_top_k"]),
                    "--seed", str(config["seeds"]["rollout_rounds"][round_index]),
                    "--include-prompt-token-ids", *_generation_engine_args(config),
                ]
            )
            partial_outputs.extend(read_jsonl(output_path))
            partial_meta.append(json.loads(output_path.with_name(output_path.name + ".meta.json").read_text()))
        partial_outputs.sort(key=lambda row: row["id"])
        write_jsonl(rollout_path, partial_outputs)
        write_json(
            routed_receipt,
            {
                "method": "offpolicy_routed_teacher_rollouts",
                "prompt_sha256": sha256_file(prompt_path),
                "quick_teacher": str(paths["quick"].resolve()),
                "deep_teacher": str(paths["deep"].resolve()),
                "rows": len(partial_outputs),
                "samples_per_prompt": 2,
                "component_runner_metadata": partial_meta,
            },
        )
    else:
        raise ValueError(f"unknown rollout source: {rollout_source}")
    cache = paths["root"] / "teacher_cache" / arm / f"seed_{seed}" / f"round_{round_index}.pt"
    cache_receipt = cache.with_suffix(cache.suffix + ".receipt.json")
    if cache.exists() and cache_receipt.exists():
        receipt = json.loads(cache_receipt.read_text())
        if (
            receipt.get("rollouts_sha256") != sha256_file(rollout_path)
            or receipt.get("routing") != routing
            or int(receipt.get("selection_seed", -1)) != selection_seed
            or int(receipt.get("optimizer_updates", -1)) != optimizer_updates
        ):
            raise SystemExit(f"stale teacher cache: {cache}")
        print(f"[resume] MOPD {arm} round {round_index} teacher cache", flush=True)
    else:
        _run(
            [
                str(PY), str(EXP / "scripts" / "score_routed_teachers.py"),
                "--config", str(config_path), "--rollouts", str(rollout_path),
                "--quick-teacher", str(paths["quick"]), "--deep-teacher", str(paths["deep"]),
                "--routing", routing, "--out", str(cache),
                "--optimizer-updates", str(optimizer_updates),
                "--selection-seed", str(selection_seed),
            ],
            training=True,
        )
    adapter = paths["root"] / "adapters" / arm / f"seed_{seed}" / f"round_{round_index}"
    merged = paths["root"] / "merged" / arm / f"seed_{seed}" / f"round_{round_index}"
    if _checkpoint_complete(adapter, merged):
        training_receipt = json.loads((adapter / "training_receipt.json").read_text())
        if not training_receipt.get("round_loss_gate", {}).get("passed"):
            raise SystemExit(f"preserved MOPD checkpoint failed its loss/safety gate: {adapter}")
        if (
            int(training_receipt.get("optimizer_steps", -1)) != optimizer_updates
            or training_receipt.get("routing") != routing
            or not training_receipt.get("consume_once_verified")
            or training_receipt.get("initial_probe", {}).get(
                "target_mean_corrected_topk_loss"
            ) != target_initial_loss
        ):
            raise SystemExit(f"stale MOPD checkpoint: {adapter}")
        print(f"[resume] MOPD {arm} round {round_index} checkpoint", flush=True)
        return merged
    if adapter.exists() or merged.exists():
        raise SystemExit(f"partial MOPD round checkpoint: {adapter} / {merged}")
    train_command = [
            str(PY), str(EXP / "scripts" / "train_mopd_round.py"),
            "--config", str(config_path), "--base-model", str(current),
            "--teacher-cache", str(cache), "--out", str(adapter),
            "--round", str(round_index), "--seed", str(seed + round_index),
    ]
    if updates_override is not None:
        train_command.extend(("--updates", str(updates_override)))
    if target_initial_loss is not None:
        train_command.extend(("--target-initial-loss", repr(target_initial_loss)))
    return_code = _run(
        train_command,
        training=True,
        allowed=(0, 3),
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(current), "--adapter", str(adapter), "--out", str(merged),
        ],
        training=True,
    )
    merge_receipt = json.loads((merged / "merge_receipt.json").read_text())
    if int(merge_receipt.get("nonzero_lora_modules", 0)) != int(
        merge_receipt.get("applied_lora_modules", -1)
    ):
        raise SystemExit(f"MOPD merge contains zero deltas: {merged}")
    _record_checkpoint(f"{arm}_seed{seed}_round{round_index}", adapter, merged)
    if return_code == 3:
        raise SystemExit(
            f"MOPD {arm} round {round_index} exceeded its frozen loss/KL guard; "
            "stopped checkpoint was preserved"
        )
    return merged


def _locality(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "teacher_audit.json")
    paths = _paths(config)
    seed = int(config["seeds"]["integration_training"][0])
    merged = _mopd_round(
        config=config, config_path=config_path, paths=paths,
        arm="locality", seed=seed, round_index=0, current=paths["quick"],
        routing="correct", updates_override=5,
    )
    cache = paths["root"] / "teacher_cache" / "locality" / f"seed_{seed}" / "round_0.pt"
    adapter = paths["root"] / "adapters" / "locality" / f"seed_{seed}" / "round_0"
    analysis_path = EXP / "analysis" / "locality_pilot.json"
    if analysis_path.exists():
        payload = json.loads(analysis_path.read_text())
        if (
            payload.get("after_model") == str(merged.resolve())
            and payload.get("teacher_cache_sha256") == sha256_file(cache)
            and payload.get("training_receipt_sha256")
            == sha256_file(adapter / "training_receipt.json")
        ):
            print("[resume] locality analysis already complete", flush=True)
            _require_gate(analysis_path)
            return
        raise SystemExit(f"stale locality analysis: {analysis_path}")
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_locality.py"),
            "--config", str(config_path), "--before-model", str(paths["quick"]),
            "--after-model", str(merged), "--teacher-cache", str(cache),
            "--training-receipt", str(adapter / "training_receipt.json"),
        ],
        training=True,
        allowed=(0, 4),
    )
    _require_gate(analysis_path)


def _run_mopd_arm(
    config: dict, config_path: Path, *, arm: str, seed: int, routing: str,
    rollout_source: str = "student", selection_seed_base: int | None = None,
    initial_match_arm: str | None = None, initial_match_seed: int | None = None,
) -> Path:
    paths = _paths(config)
    current = paths["quick"]
    for round_index in range(int(config["mopd"]["rounds"])):
        target_initial_loss = None
        if initial_match_arm is not None:
            if initial_match_seed is None:
                raise ValueError("initial_match_seed is required with initial_match_arm")
            reference = (
                paths["root"] / "adapters" / initial_match_arm
                / f"seed_{initial_match_seed}" / f"round_{round_index}"
                / "training_receipt.json"
            )
            if not reference.is_file():
                raise SystemExit(f"initial-loss reference is missing: {reference}")
            reference_receipt = json.loads(reference.read_text())
            if not reference_receipt.get("round_loss_gate", {}).get("passed"):
                raise SystemExit(f"initial-loss reference failed its gate: {reference}")
            target_initial_loss = float(
                reference_receipt["initial_probe"]["mean_corrected_topk_loss"]
            )
        current = _mopd_round(
            config=config, config_path=config_path, paths=paths, arm=arm,
            seed=seed, round_index=round_index, current=current, routing=routing,
            rollout_source=rollout_source,
            selection_seed_base=selection_seed_base,
            target_initial_loss=target_initial_loss,
        )
    return current


def _integrate(config: dict, config_path: Path, seed: int | None) -> None:
    _require_gate(EXP / "analysis" / "locality_pilot.json")
    selected = int(seed if seed is not None else config["seeds"]["integration_training"][0])
    if selected not in [int(value) for value in config["seeds"]["integration_training"]]:
        raise SystemExit(f"integration seed {selected} was not preregistered")
    result = _run_mopd_arm(
        config, config_path, arm="correct", seed=selected, routing="correct"
    )
    print(f"integrated checkpoint: {result}")


def _controls(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "locality_pilot.json")
    paths = _paths(config)
    primary_seed = int(config["seeds"]["integration_training"][0])
    scheduled_updates = int(config["mopd"]["rounds"]) * int(
        config["mopd"]["updates_per_round"]
    )
    if scheduled_updates != int(config["controls"]["wrong_route_updates"]):
        raise SystemExit("wrong-route update schedule does not match the frozen control")
    if scheduled_updates != int(config["controls"]["offpolicy_updates"]):
        raise SystemExit("off-policy update schedule does not match the frozen control")
    _run_mopd_arm(
        config, config_path, arm="wrong_route",
        seed=int(config["seeds"]["wrong_routing"]), routing="wrong",
        selection_seed_base=primary_seed,
        initial_match_arm="correct", initial_match_seed=primary_seed,
    )
    _run_mopd_arm(
        config, config_path, arm="offpolicy",
        seed=int(config["seeds"]["wrong_routing"]), routing="correct",
        rollout_source="teacher_routed",
        selection_seed_base=primary_seed,
        initial_match_arm="correct", initial_match_seed=primary_seed,
    )
    for weight in config["controls"]["parameter_merge_weights"]:
        tag = f"parameter_merge_deep_{int(round(float(weight) * 100)):02d}"
        out = paths["root"] / "merged" / tag
        if (out / "merge_receipt.json").exists():
            print(f"[resume] {tag}", flush=True)
            continue
        if out.exists():
            raise SystemExit(f"partial parameter merge exists: {out}")
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_weighted_adapters.py"),
                "--quick-adapter", str(paths["quick_adapter"]),
                "--deep-adapter", str(paths["deep_adapter"]),
                "--deep-weight", str(weight), "--out", str(out),
            ],
            training=True,
        )
    union_adapter = paths["root"] / "adapters" / "matched_union_sft"
    union_merged = paths["root"] / "merged" / "matched_union_sft"
    if not _checkpoint_complete(union_adapter, union_merged):
        if union_adapter.exists() or union_merged.exists():
            raise SystemExit("partial matched-union SFT checkpoint exists")
        _run(
            [
                str(PY), str(EXP / "scripts" / "train_specialist.py"),
                "--train", str(resolve_repo_path(config["model"]["deep_data"])),
                "--out", str(union_adapter), "--warm-start", str(paths["quick_adapter"]),
                "--epochs", "1", "--max-steps", str(config["controls"]["matched_sft_steps"]),
                "--lr", str(config["mopd"]["learning_rate"]),
                "--rank", str(config["mopd"]["rank"]), "--alpha", str(config["mopd"]["alpha"]),
                "--batch-size", "1", "--grad-accum", str(config["mopd"]["grad_accum"]),
                "--max-length", str(config["mopd"]["max_length"]),
                "--w-think", str(config["specialist_train"]["think_loss_weight"]),
                "--seed", str(config["specialist_train"]["seed"]),
            ],
            training=True,
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_adapter.py"),
                "--base-model", config["model"]["id"],
                "--adapter", str(union_adapter), "--out", str(union_merged),
            ],
            training=True,
        )
        _record_checkpoint("matched_union_sft", union_adapter, union_merged)


def _confirm(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "locality_pilot.json")
    paths = _paths(config)
    last_round = int(config["mopd"]["rounds"]) - 1
    integration_seeds = [int(value) for value in config["seeds"]["integration_training"]]
    primary_name = f"correct_seed{integration_seeds[0]}"
    models: dict[str, tuple[Path, str]] = {
        "quick": (paths["quick"], "greedy"),
        "deep": (paths["deep"], "greedy"),
    }
    for seed in integration_seeds:
        models[f"correct_seed{seed}"] = (
            paths["root"] / "merged" / "correct" / f"seed_{seed}" / f"round_{last_round}",
            "greedy",
        )
    wrong_seed = int(config["seeds"]["wrong_routing"])
    models["wrong_route"] = (
        paths["root"] / "merged" / "wrong_route" / f"seed_{wrong_seed}" / f"round_{last_round}",
        "greedy",
    )
    models["offpolicy"] = (
        paths["root"] / "merged" / "offpolicy" / f"seed_{wrong_seed}" / f"round_{last_round}",
        "greedy",
    )
    parameter_names = []
    for weight in config["controls"]["parameter_merge_weights"]:
        name = f"parameter_merge_deep_{int(round(float(weight) * 100)):02d}"
        parameter_names.append(name)
        models[name] = (paths["root"] / "merged" / name, "greedy")
    models["matched_union_sft"] = (
        paths["root"] / "merged" / "matched_union_sft", "greedy"
    )
    models["quick_sample8"] = (paths["quick"], "sample8")
    for name, (model, _) in models.items():
        if not (model / "config.json").is_file() or not (model / "merge_receipt.json").is_file():
            raise SystemExit(f"confirmatory arm {name} is incomplete: {model}")

    block_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    arm_scores: dict[str, list[str]] = {}
    for name, (model, decode) in models.items():
        arm_scores[name] = []
        for block_index, block_seed in enumerate(block_seeds):
            scores = _eval_if_needed(
                model=model, tag=f"{name}_confirm_b{block_index}", scope="confirmatory",
                block_seed=block_seed, config_path=config_path, decode=decode,
            )
            arm_scores[name].append(str(scores.resolve()))
    manifest = {
        "stage": "confirmatory_manifest", "config": str(config_path),
        "config_sha256": sha256_file(config_path), "block_seeds": block_seeds,
        "primary_arm": primary_name,
        "replicate_arms": [f"correct_seed{seed}" for seed in integration_seeds[1:]],
        "source_arms": ["quick", "deep"], "quick_arm": "quick", "deep_arm": "deep",
        "control_arms": [
            "wrong_route", "offpolicy", *parameter_names, "matched_union_sft",
        ],
        "sample_more_arm": "quick_sample8", "arms": arm_scores,
    }
    manifest_path = EXP / "runs" / "confirmatory_manifest.json"
    write_json(manifest_path, manifest)
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_confirmation.py"),
            "--config", str(config_path), "--manifest", str(manifest_path),
            "--calibration", str(EXP / "analysis" / "calibration.json"),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "confirmation.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "specialists", "calibrate", "qualify", "teacher-audit", "locality", "integrate", "controls", "confirm"),
    )
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    stage = "smoke" if args.smoke else args.stage
    if stage is None:
        parser.error("pass --smoke or --stage")
    if stage == "smoke":
        print(json.dumps(scientific_smoke(config, config_path), indent=2, sort_keys=True))
        return 0
    _verify_preregistration()
    if stage == "specialists":
        _specialists(config)
        return 0
    if stage == "model-smoke":
        _model_smoke(config)
        return 0
    if stage == "qualify":
        _qualify(config, config_path)
        return 0
    if stage == "calibrate":
        _calibrate(config, config_path)
        return 0
    if stage == "teacher-audit":
        _teacher_audit(config, config_path)
        return 0
    if stage == "locality":
        _locality(config, config_path)
        return 0
    if stage == "integrate":
        _integrate(config, config_path, args.seed)
        return 0
    if stage == "controls":
        _controls(config, config_path)
        return 0
    if stage == "confirm":
        _confirm(config, config_path)
        return 0
    raise SystemExit(f"stage {stage!r} is registered but not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
