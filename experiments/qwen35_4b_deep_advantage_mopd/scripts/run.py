#!/usr/bin/env python3
"""Fail-closed orchestration for deep-only advantage-routed MOPD."""

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
    sha256_file,
    validate_policy_cache_provenance,
    write_json,
)


FROZEN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "reports/literature_review.md",
)


def _run(
    command: list[str],
    *,
    training: bool = False,
    allowed: tuple[int, ...] = (0,),
) -> int:
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
        "quick_adapter": resolve_repo_path(config["model"]["quick_adapter"]),
        "deep_adapter": resolve_repo_path(config["model"]["deep_adapter"]),
        "quick": resolve_repo_path(config["model"]["quick_teacher"]),
        "deep": resolve_repo_path(config["model"]["deep_teacher"]),
        "soup": resolve_repo_path(config["model"]["student_checkpoint"]),
    }


def _merged_complete(path: Path) -> bool:
    return all((path / name).is_file() for name in ("config.json", "model.safetensors", "merge_receipt.json"))


def _source_complete(adapter: Path, merged: Path) -> bool:
    return all(
        path.is_file()
        for path in (
            adapter / "adapter_config.json",
            adapter / "adapter_model.safetensors",
            adapter / "training_receipt.json",
            merged / "config.json",
            merged / "model.safetensors",
            merged / "merge_receipt.json",
        )
    )


def _require_gate(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"required gate receipt missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("gate", {}).get("passed"):
        raise SystemExit(f"required gate did not pass: {path}")
    return payload


def _verify_preregistration() -> dict:
    path = EXP / "runs" / "preregistration_receipt.json"
    if not path.is_file():
        raise SystemExit("preregistration receipt is missing; no model stage is legal")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "locked":
        raise SystemExit("preregistration receipt is not locked")
    if tuple(payload.get("frozen_file_order") or ()) != FROZEN_FILES:
        raise SystemExit("preregistration frozen-file inventory changed")
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


def _write_preregistration_receipt(design_commit: str) -> None:
    if subprocess.run(
        ["git", "cat-file", "-e", f"{design_commit}^{{commit}}"], cwd=REPO, check=False
    ).returncode != 0:
        raise SystemExit(f"unknown design commit: {design_commit}")
    status = subprocess.run(
        ["git", "status", "--short", "--", *[str(EXP / value) for value in FROZEN_FILES]],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise SystemExit(f"frozen design files are not committed:\n{status}")
    receipt = {
        "schema_version": 1,
        "status": "locked",
        "experiment_id": EXP.name,
        "design_commit": design_commit,
        "frozen_file_order": list(FROZEN_FILES),
        "frozen_files": {
            relative: sha256_file(EXP / relative) for relative in FROZEN_FILES
        },
        "model_output_precedes_lock": False,
        "note": "No task-model output existed before this immutable design lock.",
    }
    write_json(EXP / "runs" / "preregistration_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


def scientific_smoke(config: dict, config_path: Path) -> dict:
    for path in sorted(
        list((EXP / "src").rglob("*.py"))
        + list((EXP / "scripts").glob("*.py"))
        + list((EXP / "tests").glob("*.py"))
    ):
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
    seeds = [
        int(config["seeds"]["model_smoke"]),
        int(config["seeds"]["installation_canary"]),
        *[int(value) for value in config["seeds"]["route_qualification_blocks"]],
        *[int(value) for value in config["seeds"]["rollout_rounds"]],
        *[int(value) for value in config["seeds"]["confirmatory_blocks"]],
        *range(
            int(config["benchmark"]["first_seed"]),
            int(config["benchmark"]["first_seed"])
            + int(config["benchmark"]["quick_events"])
            + int(config["benchmark"]["medium_events"]),
        ),
    ]
    if len(seeds) != len(set(seeds)):
        raise AssertionError("generation seed namespaces overlap")
    if int(config["seeds"]["benchmark_first"]) != int(
        config["benchmark"]["first_seed"]
    ):
        raise AssertionError("benchmark seed aliases disagree")
    route_states = int(config["generation"]["qualification_atom_states_per_block"]) + int(
        config["generation"]["qualification_episode_states_per_block"]
    )
    minimum = int(config["route"]["minimum_deep_routed_states_per_block"])
    if route_states < minimum:
        raise AssertionError("route support gate is mathematically unreachable")
    micro_steps = int(config["mopd"]["updates_per_round"]) * int(config["mopd"]["grad_accum"])
    capability = int(config["mopd"]["capability_units_per_round"])
    anchor = int(config["mopd"]["anchor_units_per_round"])
    if capability + anchor != micro_steps or capability < 1 or anchor < 1:
        raise AssertionError("deep-only MOPD/anchor quotas are not integer-reachable")
    if abs(capability / micro_steps - float(config["mopd"]["capability_fraction"])) > 1e-12:
        raise AssertionError("deep-only capability fraction disagrees with exact quotas")
    paths = _paths(config)
    sources = {}
    for name in ("quick", "deep"):
        adapter = paths[f"{name}_adapter"]
        merged = paths[name]
        if not _source_complete(adapter, merged):
            raise AssertionError(f"source {name} checkpoint is incomplete: {adapter} / {merged}")
        sources[name] = {
            "adapter_config_sha256": sha256_file(adapter / "adapter_config.json"),
            "adapter_weights_sha256": sha256_file(adapter / "adapter_model.safetensors"),
            "training_receipt_sha256": sha256_file(adapter / "training_receipt.json"),
            "model_weights_sha256": sha256_file(merged / "model.safetensors"),
            "merge_receipt_sha256": sha256_file(merged / "merge_receipt.json"),
        }
    if not _merged_complete(paths["soup"]):
        raise AssertionError(f"immutable soup checkpoint is incomplete: {paths['soup']}")
    soup_sha256 = sha256_file(paths["soup"] / "model.safetensors")
    if soup_sha256 != str(config["model"]["student_model_sha256"]):
        raise AssertionError("immutable soup checkpoint hash changed")
    payload = {
        "status": "pass",
        "config": str(config_path.relative_to(EXP)),
        "config_sha256": canonical_hash(config),
        "model": config["model"]["id"],
        "revision": config["model"]["revision"],
        "families": expected,
        "transfer_families": list(config["strata"]["transfer_families"]),
        "route_advantage_threshold": config["decision"]["route_advantage_threshold"],
        "final_joint_delta_threshold": config["decision"]["final_joint_delta_threshold"],
        "reachable_counts": {
            "qualification_states_per_block": route_states,
            "qualified_teacher": str(config["route"]["qualified_teacher"]),
            "minimum_deep_per_block": minimum,
            "micro_steps_per_round": micro_steps,
            "capability_units_per_round": capability,
            "anchor_units_per_round": anchor,
            "non_advantage_route_control_units_per_round": capability,
        },
        "source_artifacts": sources,
        "immutable_student": {
            "path": str(paths["soup"].resolve()),
            "model_sha256": soup_sha256,
            "merge_receipt_sha256": sha256_file(
                paths["soup"] / "merge_receipt.json"
            ),
        },
        "selftest_tail": completed.stdout.strip().splitlines()[-3:],
    }
    write_json(EXP / "runs" / "smoke" / "summary.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def _model_smoke(config: dict) -> None:
    root = EXP / "runs" / "model_smoke"
    base_output = root / "base.jsonl"
    hf_output = root / "hf.json"
    if hf_output.is_file():
        payload = json.loads(hf_output.read_text(encoding="utf-8"))
        if payload.get("status") == "pass" and payload.get("model_revision") == config["model"]["revision"]:
            print("[resume] pinned model smoke", flush=True)
            return
        raise SystemExit("stale model smoke receipt")
    root.mkdir(parents=True, exist_ok=True)
    common = [
        "--smoke", "4", "--thinking", "off", "--greedy", "--max-tokens", "32",
        "--seed", str(config["seeds"]["model_smoke"]),
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
        "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
        "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
        "--cudagraph-capture-size", "16",
    ]
    _run([str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"), "--output", str(base_output), *common])
    _run(
        [
            str(PY), str(EXP / "scripts" / "model_smoke.py"),
            "--vllm-output", str(base_output), "--out", str(hf_output),
        ]
    )


def _stamp_local_model_output(output: Path, model: Path) -> None:
    metadata = output.with_name(output.name + ".meta.json")
    merge_receipt = model / "merge_receipt.json"
    if not output.is_file() or not metadata.is_file() or not merge_receipt.is_file():
        raise SystemExit(f"cannot provenance-stamp local output: {output}")
    payload = json.loads(metadata.read_text(encoding="utf-8"))
    if payload.get("model") != str(model.resolve()):
        raise SystemExit(f"runner loaded wrong local model for {output}")
    payload["local_model_provenance"] = {
        "merge_receipt_sha256": sha256_file(merge_receipt),
        "output_sha256": sha256_file(output),
    }
    write_json(metadata, payload)


def _generation_engine_args(config: dict) -> list[str]:
    args = [
        "--max-model-len", str(config["engine"]["max_model_len"]),
        "--gpu-memory-utilization", str(config["engine"]["gpu_memory_utilization"]),
        "--max-num-seqs", str(config["engine"]["max_num_seqs"]),
        "--max-num-batched-tokens", str(config["engine"]["max_num_batched_tokens"]),
    ]
    for size in config["engine"]["cudagraph_capture_sizes"]:
        args.extend(("--cudagraph-capture-size", str(size)))
    return args


def _checkpoint_receipt(config: dict, paths: dict[str, Path]) -> None:
    payload = {
        "schema_version": 1,
        "model": config["model"]["id"],
        "revision": config["model"]["revision"],
        "quick": {
            "path": str(paths["quick"].resolve()),
            "model_sha256": sha256_file(paths["quick"] / "model.safetensors"),
            "merge_receipt_sha256": sha256_file(paths["quick"] / "merge_receipt.json"),
        },
        "deep": {
            "path": str(paths["deep"].resolve()),
            "model_sha256": sha256_file(paths["deep"] / "model.safetensors"),
            "merge_receipt_sha256": sha256_file(paths["deep"] / "merge_receipt.json"),
        },
        "soup": {
            "path": str(paths["soup"].resolve()),
            "model_sha256": sha256_file(paths["soup"] / "model.safetensors"),
            "merge_receipt_sha256": sha256_file(paths["soup"] / "merge_receipt.json"),
        },
    }
    if payload["soup"]["model_sha256"] != str(
        config["model"]["student_model_sha256"]
    ):
        raise SystemExit("checkpoint receipt found the wrong immutable soup")
    write_json(EXP / "runs" / "checkpoint_receipts.json", payload)


def _verify_student(config: dict, config_path: Path) -> None:
    if not (EXP / "runs" / "model_smoke" / "hf.json").is_file():
        raise SystemExit("run model-smoke before verifying the student")
    paths = _paths(config)
    if not _source_complete(paths["quick_adapter"], paths["quick"]):
        raise SystemExit("quick source checkpoint incomplete")
    if not _source_complete(paths["deep_adapter"], paths["deep"]):
        raise SystemExit("deep source checkpoint incomplete")
    soup = paths["soup"]
    if not _merged_complete(soup):
        raise SystemExit(f"immutable soup checkpoint is incomplete: {soup}")
    if sha256_file(soup / "model.safetensors") != str(
        config["model"]["student_model_sha256"]
    ):
        raise SystemExit("immutable soup checkpoint hash mismatch")
    soup_receipt = json.loads((soup / "merge_receipt.json").read_text(encoding="utf-8"))
    if (
        soup_receipt.get("method") != "explicit_convex_lora_delta_merge"
        or abs(
            float(soup_receipt.get("deep_weight", -1.0))
            - float(config["model"]["student_deep_weight"])
        )
        > 1e-12
        or soup_receipt.get("quick_weights_sha256")
        != sha256_file(paths["quick_adapter"] / "adapter_model.safetensors")
        or soup_receipt.get("deep_weights_sha256")
        != sha256_file(paths["deep_adapter"] / "adapter_model.safetensors")
    ):
        raise SystemExit("immutable soup merge provenance mismatch")
    _checkpoint_receipt(config, paths)
    root = EXP / "runs" / "installation_canary"
    prompts = root / "prompts.jsonl"
    if not prompts.is_file():
        _run(
            [
                str(PY), str(EXP / "scripts" / "build_installation_canary.py"),
                "--config", str(config_path), "--out", str(prompts),
            ]
        )
    outputs = {name: root / f"{name}.jsonl" for name in ("base", "quick", "deep", "soup")}
    common = [
        "--input", str(prompts), "--thinking", "budget", "--thinking-budget", "256",
        "--answer-max-tokens", "64", "--greedy",
        "--seed", str(config["seeds"]["installation_canary"]),
        *_generation_engine_args(config),
    ]
    for name, model in (("base", None), ("quick", paths["quick"]), ("deep", paths["deep"]), ("soup", soup)):
        output = outputs[name]
        metadata = output.with_name(output.name + ".meta.json")
        if output.is_file() and metadata.is_file():
            print(f"[resume] installation canary {name}", flush=True)
            continue
        if output.exists() or metadata.exists():
            raise SystemExit(f"partial installation canary output: {output}")
        command = [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
            "--output", str(output), *common,
        ]
        if model is not None:
            command.extend(("--model-override", str(model)))
        _run(command)
        if model is not None:
            _stamp_local_model_output(output, model)
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_installation_canary.py"),
            "--config", str(config_path), "--input", str(prompts),
            "--base", str(outputs["base"]), "--quick", str(outputs["quick"]),
            "--deep", str(outputs["deep"]), "--soup", str(outputs["soup"]),
            "--quick-model", str(paths["quick"]), "--deep-model", str(paths["deep"]),
            "--soup-model", str(soup),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "installation_canary.json")


def _route_qualify(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "installation_canary.json")
    paths = _paths(config)
    blocks = []
    for block, seed_value in enumerate(config["seeds"]["route_qualification_blocks"]):
        seed = int(seed_value)
        root = EXP / "runs" / "route_qualification" / f"block_{block}"
        state_dir = root / "student_states"
        states = state_dir / "states.jsonl"
        if not states.is_file():
            _run(
                [
                    str(VLLM_PY), str(EXP / "scripts" / "generate_student_states.py"),
                    "--config", str(config_path), "--model", str(paths["soup"]),
                    "--block", str(block), "--seed", str(seed), "--out-dir", str(state_dir),
                ]
            )
        branch_files = {}
        for policy, model in (("quick", paths["quick"]), ("deep", paths["deep"]), ("student", paths["soup"])):
            out_dir = root / "branches" / policy
            output = out_dir / "branches.jsonl.gz"
            receipt = out_dir / "receipt.json"
            if output.is_file() and receipt.is_file():
                payload = json.loads(receipt.read_text(encoding="utf-8"))
                if (
                    payload.get("states_sha256") == sha256_file(states)
                    and payload.get("model") == str(model.resolve())
                    and payload.get("policy") == policy
                    and payload.get("branches_sha256") == sha256_file(output)
                ):
                    print(f"[resume] route block {block} {policy} branches", flush=True)
                else:
                    raise SystemExit(f"stale route branch artifact: {out_dir}")
            else:
                if out_dir.exists() and any(out_dir.iterdir()):
                    raise SystemExit(f"partial route branch artifact: {out_dir}")
                _run(
                    [
                        str(VLLM_PY), str(EXP / "scripts" / "branch_states.py"),
                        "--config", str(config_path), "--states", str(states),
                        "--model", str(model), "--policy", policy,
                        "--block-seed", str(seed), "--out-dir", str(out_dir),
                    ]
                )
            branch_files[policy] = output
        assembled = root / "assembled.json"
        if not assembled.is_file():
            _run(
                [
                    str(PY), str(EXP / "scripts" / "assemble_route_block.py"),
                    "--config", str(config_path), "--states", str(states),
                    "--quick", str(branch_files["quick"]), "--deep", str(branch_files["deep"]),
                    "--student", str(branch_files["student"]), "--block", str(block),
                    "--out", str(assembled),
                ]
            )
        blocks.append(assembled)
    code = _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_route_qualification.py"),
            "--config", str(config_path),
            *[value for path in blocks for value in ("--block", str(path))],
        ],
        allowed=(0, 4),
    )
    if code != 0:
        raise SystemExit(code)
    _require_gate(EXP / "analysis" / "route_qualification.json")


def _require_round_gate(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"training receipt missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("round_gate", {}).get("passed"):
        raise SystemExit(f"training safety gate did not pass: {path}")
    return payload


def _adapter_complete(path: Path) -> bool:
    return all(
        (path / name).is_file()
        for name in ("adapter_config.json", "adapter_model.safetensors", "training_receipt.json")
    )


def _validate_state_batch(
    state_dir: Path,
    *,
    config_path: Path,
    model: Path,
    round_index: int,
    batch_index: int,
) -> tuple[Path, Path]:
    states = state_dir / "states.jsonl"
    anchors = state_dir / "anchors.jsonl"
    receipt_path = state_dir / "receipt.json"
    if not all(path.is_file() for path in (states, anchors, receipt_path)):
        raise SystemExit(f"incomplete online student-state batch: {state_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = (
        receipt.get("config_sha256") == sha256_file(config_path),
        receipt.get("model") == str(model.resolve()),
        receipt.get("model_merge_receipt_sha256")
        == sha256_file(model / "merge_receipt.json"),
        receipt.get("mode") == "training",
        int(receipt.get("round", -1)) == round_index,
        int(receipt.get("batch", -1)) == batch_index,
        receipt.get("states_sha256") == sha256_file(states),
        receipt.get("anchors_sha256") == sha256_file(anchors),
        all(receipt.get("engine_protocol", {}).values()),
    )
    if not all(checks):
        raise SystemExit(f"stale online student-state batch: {state_dir}")
    return states, anchors


def _validate_branch_batch(
    out_dir: Path,
    *,
    config_path: Path,
    states: Path,
    model: Path,
    policy: str,
) -> Path:
    output = out_dir / "branches.jsonl.gz"
    receipt_path = out_dir / "receipt.json"
    if not output.is_file() or not receipt_path.is_file():
        raise SystemExit(f"incomplete online branch batch: {out_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    checks = (
        receipt.get("config_sha256") == sha256_file(config_path),
        receipt.get("states_sha256") == sha256_file(states),
        receipt.get("model") == str(model.resolve()),
        receipt.get("model_merge_receipt_sha256")
        == sha256_file(model / "merge_receipt.json"),
        receipt.get("policy") == policy,
        receipt.get("branch_mode") == "selection",
        int(receipt.get("audit_branches", -1)) == 0,
        receipt.get("branches_sha256") == sha256_file(output),
        all(receipt.get("engine_protocol", {}).values()),
    )
    if not all(checks):
        raise SystemExit(f"stale online branch batch: {out_dir}")
    return output


def _prepare_online_round(
    config: dict,
    config_path: Path,
    *,
    current_model: Path,
    training_seed: int,
    round_index: int,
) -> tuple[Path, Path]:
    """Generate/freeze current-student states, routes, and all policy targets."""
    _require_gate(EXP / "analysis" / "route_qualification.json")
    paths = _paths(config)
    if not _merged_complete(current_model):
        raise SystemExit(f"online student checkpoint is incomplete: {current_model}")
    # Every replicate has the identical soup at round zero.  Reuse that exact
    # frozen on-policy draw; later rounds diverge and receive seed-specific data.
    data_seed = int(config["seeds"]["integration_training"][0]) if round_index == 0 else training_seed
    if round_index == 0 and current_model.resolve() != paths["soup"].resolve():
        raise SystemExit("round-zero shared data are legal only from the exact soup")
    root = (
        paths["root"] / "online" / "primary" / f"seed_{data_seed}"
        / f"round_{round_index}"
    )
    manifest = root / "training_round.json"
    target_cache = root / "all_policy_targets.pt"
    maximum = int(config["mopd"]["candidate_batches_per_round_maximum"])
    state_paths: list[Path] = []
    anchor_paths: list[Path] = []
    branch_paths: dict[str, list[Path]] = {"quick": [], "deep": [], "student": []}

    if manifest.is_file():
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        if (
            frozen.get("stage") != "online_advantage_training_round"
            or frozen.get("config_sha256") != sha256_file(config_path)
            or int(frozen.get("round", -1)) != round_index
        ):
            raise SystemExit(f"stale online round manifest: {manifest}")
        candidate_batches = int(frozen["candidate_batches"])
    else:
        candidate_batches = maximum

    for batch_index in range(candidate_batches):
        batch_root = root / "candidates" / f"batch_{batch_index}"
        state_dir = batch_root / "student_states"
        if not state_dir.exists() or not any(state_dir.iterdir()):
            rollout_seed = int(config["seeds"]["rollout_rounds"][round_index]) + batch_index * 10
            block = 100 + round_index * maximum + batch_index
            _run(
                [
                    str(VLLM_PY), str(EXP / "scripts" / "generate_student_states.py"),
                    "--config", str(config_path), "--model", str(current_model),
                    "--mode", "training", "--round", str(round_index),
                    "--batch", str(batch_index), "--block", str(block),
                    "--seed", str(rollout_seed), "--out-dir", str(state_dir),
                ]
            )
        states, anchors = _validate_state_batch(
            state_dir,
            config_path=config_path,
            model=current_model,
            round_index=round_index,
            batch_index=batch_index,
        )
        state_paths.append(states)
        anchor_paths.append(anchors)
        for policy, model in (
            ("quick", paths["quick"]),
            ("deep", paths["deep"]),
            ("student", current_model),
        ):
            out_dir = batch_root / "branches" / policy
            if not out_dir.exists() or not any(out_dir.iterdir()):
                _run(
                    [
                        str(VLLM_PY), str(EXP / "scripts" / "branch_states.py"),
                        "--config", str(config_path), "--states", str(states),
                        "--model", str(model), "--policy", policy,
                        "--branch-mode", "selection",
                        "--block-seed", str(
                            int(config["seeds"]["rollout_rounds"][round_index])
                            + batch_index * 10
                        ),
                        "--out-dir", str(out_dir),
                    ]
                )
            branch_paths[policy].append(
                _validate_branch_batch(
                    out_dir,
                    config_path=config_path,
                    states=states,
                    model=model,
                    policy=policy,
                )
            )
        if not manifest.is_file():
            command = [
                str(PY), str(EXP / "scripts" / "assemble_training_round.py"),
                "--config", str(config_path), "--round", str(round_index),
                "--out", str(manifest),
            ]
            for path in state_paths:
                command.extend(("--states", str(path)))
            for path in anchor_paths:
                command.extend(("--anchors", str(path)))
            for policy in ("quick", "deep", "student"):
                for path in branch_paths[policy]:
                    command.extend((f"--{policy}", str(path)))
            code = _run(command, allowed=(0, 3))
            if code == 0:
                candidate_batches = batch_index + 1
                break
            if batch_index + 1 == maximum:
                terminal = manifest.with_suffix(manifest.suffix + ".supply.json")
                raise SystemExit(
                    f"maximum online batches lack deep/control route supply: {terminal}"
                )
    if not manifest.is_file():
        raise SystemExit(f"online training-round manifest was not created: {manifest}")
    frozen = json.loads(manifest.read_text(encoding="utf-8"))
    expected_units = int(config["mopd"]["updates_per_round"]) * int(config["mopd"]["grad_accum"])
    capability_units = int(config["mopd"]["capability_units_per_round"])
    anchor_units = int(config["mopd"]["anchor_units_per_round"])
    if (
        len(frozen.get("units") or []) != expected_units
        or len({row["state_id"] for row in frozen["units"]}) != expected_units
        or len(frozen.get("control_units") or []) != capability_units
        or len(
            {
                row["state_id"]
                for row in [*frozen["units"], *frozen["control_units"]]
            }
        )
        != expected_units + capability_units
        or frozen.get("unit_counts", {}).get("deep") != capability_units
        or frozen.get("unit_counts", {}).get("soup_anchor") != anchor_units
        or frozen.get("unit_counts", {}).get("non_advantage_route_control")
        != capability_units
    ):
        raise SystemExit(f"online round has invalid consume-once quotas: {manifest}")

    cache_receipt_path = target_cache.with_suffix(target_cache.suffix + ".receipt.json")
    if not target_cache.is_file() and not cache_receipt_path.is_file():
        _run(
            [
                str(PY), str(EXP / "scripts" / "cache_policy_targets.py"),
                "--config", str(config_path), "--round-manifest", str(manifest),
                "--quick", str(paths["quick"]), "--deep", str(paths["deep"]),
                "--soup", str(paths["soup"]), "--out", str(target_cache),
            ],
            training=True,
        )
    if not target_cache.is_file() or not cache_receipt_path.is_file():
        raise SystemExit(f"partial all-policy target cache: {target_cache}")
    cache_receipt = json.loads(cache_receipt_path.read_text(encoding="utf-8"))
    try:
        validate_policy_cache_provenance(cache_receipt, config, config_path)
    except ValueError as error:
        raise SystemExit(f"stale all-policy target cache: {target_cache}: {error}") from error
    if (
        cache_receipt.get("cache_sha256") != sha256_file(target_cache)
        or cache_receipt.get("round_manifest_sha256") != sha256_file(manifest)
        or int(cache_receipt.get("round", -1)) != round_index
        or int(cache_receipt.get("sample_count", -1))
        != expected_units + capability_units
    ):
        raise SystemExit(f"stale all-policy target cache: {target_cache}")
    return manifest, target_cache


def _merge_round_adapter(adapter: Path, base_model: Path, merged: Path) -> None:
    if not _adapter_complete(adapter):
        raise SystemExit(f"cannot merge incomplete adapter: {adapter}")
    if not _merged_complete(merged):
        if merged.exists() and any(merged.iterdir()):
            raise SystemExit(f"partial merged round checkpoint: {merged}")
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_adapter.py"),
                "--adapter", str(adapter), "--base-model", str(base_model),
                "--out", str(merged),
            ],
            training=True,
        )
    receipt = json.loads((merged / "merge_receipt.json").read_text(encoding="utf-8"))
    if (
        Path(receipt.get("base_model", "")).resolve() != base_model.resolve()
        or Path(receipt.get("adapter", "")).resolve() != adapter.resolve()
        or receipt.get("adapter_weights_sha256")
        != sha256_file(adapter / "adapter_model.safetensors")
        or int(receipt.get("applied_lora_modules", 0)) < 1
        or int(receipt.get("nonzero_lora_modules", 0))
        != int(receipt.get("applied_lora_modules", -1))
    ):
        raise SystemExit(f"merged round provenance failed: {merged}")


def _train_mopd_checkpoint(
    config: dict,
    config_path: Path,
    *,
    base_model: Path,
    target_cache: Path,
    adapter: Path,
    merged: Path,
    round_index: int,
    seed: int,
    arm: str,
    updates: int | None = None,
    target_initial_loss: float | None = None,
    merge_even_if_failed: bool = False,
) -> tuple[dict, Path | None]:
    if not _adapter_complete(adapter):
        if adapter.exists() and any(adapter.iterdir()):
            raise SystemExit(f"partial MOPD adapter: {adapter}")
        command = [
            str(PY), str(EXP / "scripts" / "train_mopd_round.py"),
            "--config", str(config_path), "--base-model", str(base_model),
            "--target-cache", str(target_cache), "--out", str(adapter),
            "--round", str(round_index), "--seed", str(seed), "--arm", arm,
        ]
        if updates is not None:
            command.extend(("--updates", str(updates)))
        if target_initial_loss is not None:
            command.extend(("--target-initial-loss", repr(float(target_initial_loss))))
        _run(command, training=True, allowed=(0, 3))
    if not _adapter_complete(adapter):
        raise SystemExit(f"MOPD trainer did not leave a complete adapter receipt: {adapter}")
    receipt_path = adapter / "training_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected_updates = int(
        updates if updates is not None else config["mopd"]["updates_per_round"]
    )
    if (
        receipt.get("arm") != arm
        or int(receipt.get("round", -1)) != round_index
        or int(receipt.get("seed", -1)) != seed
        or int(receipt.get("requested_updates", -1)) != expected_updates
        or Path(receipt.get("base_model", "")).resolve() != base_model.resolve()
        or receipt.get("target_cache_sha256") != sha256_file(target_cache)
    ):
        raise SystemExit(f"stale MOPD training receipt: {receipt_path}")
    passed = bool(receipt.get("round_gate", {}).get("passed"))
    may_merge = passed or (
        merge_even_if_failed
        and int(receipt.get("completed_updates", -1)) == expected_updates
    )
    if may_merge:
        _merge_round_adapter(adapter, base_model, merged)
        return receipt, merged
    return receipt, None


def _locality(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "route_qualification.json")
    paths = _paths(config)
    manifest, target_cache = _prepare_online_round(
        config,
        config_path,
        current_model=paths["soup"],
        training_seed=int(config["seeds"]["integration_training"][0]),
        round_index=0,
    )
    adapter = paths["root"] / "adapters" / "locality_pilot"
    merged = paths["root"] / "merged" / "locality_pilot"
    receipt, merged_result = _train_mopd_checkpoint(
        config,
        config_path,
        base_model=paths["soup"],
        target_cache=target_cache,
        adapter=adapter,
        merged=merged,
        round_index=0,
        seed=int(config["seeds"]["integration_training"][0]),
        arm="primary",
        updates=int(config["locality"]["updates"]),
        merge_even_if_failed=True,
    )
    analysis_path = EXP / "analysis" / "locality_pilot.json"
    if merged_result is None:
        result = {
            "schema_version": 2,
            "stage": "five_update_exact_logit_locality_pilot",
            "config": str(config_path),
            "training_receipt": str((adapter / "training_receipt.json").resolve()),
            "training_receipt_sha256": sha256_file(adapter / "training_receipt.json"),
            "checks": {
                "five_updates": int(receipt.get("completed_updates", -1))
                == int(config["locality"]["updates"]),
                "training_round_safety_gate": False,
            },
            "gate": {"passed": False},
            "downstream_authorization": "stop_before_full_mopd",
            "terminal_reason": receipt.get("round_gate", {}).get("unsafe_reason"),
        }
        write_json(analysis_path, result)
        raise SystemExit("five-update MOPD trainer failed before exact-logit locality")
    code = _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_locality.py"),
            "--config", str(config_path), "--before-model", str(paths["soup"]),
            "--after-model", str(merged_result), "--target-cache", str(target_cache),
            "--training-receipt", str(adapter / "training_receipt.json"),
            "--out", str(analysis_path),
        ],
        training=True,
        allowed=(0, 4),
    )
    if code != 0:
        raise SystemExit(code)
    _require_gate(analysis_path)
    # Bind the pilot to the exact frozen online manifest for auditability.
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload["round_manifest"] = str(manifest.resolve())
    payload["round_manifest_sha256"] = sha256_file(manifest)
    write_json(analysis_path, payload)


def _integration_receipt_path(seed: int) -> Path:
    return EXP / "runs" / "integration" / f"seed_{seed}.json"


def _integration_final_model(config: dict, seed: int) -> Path:
    return (
        _paths(config)["root"] / "merged" / "primary" / f"seed_{seed}"
        / f"round_{int(config['mopd']['rounds']) - 1}"
    )


def _integrate(config: dict, config_path: Path, seed: int | None) -> None:
    _require_gate(EXP / "analysis" / "locality_pilot.json")
    allowed_seeds = [int(value) for value in config["seeds"]["integration_training"]]
    if seed is None or seed not in allowed_seeds:
        raise SystemExit(f"--seed must be one of the frozen integration seeds {allowed_seeds}")
    paths = _paths(config)
    current = paths["soup"]
    rounds = []
    for round_index in range(int(config["mopd"]["rounds"])):
        manifest, target_cache = _prepare_online_round(
            config,
            config_path,
            current_model=current,
            training_seed=seed,
            round_index=round_index,
        )
        adapter = (
            paths["root"] / "adapters" / "primary" / f"seed_{seed}"
            / f"round_{round_index}"
        )
        merged = (
            paths["root"] / "merged" / "primary" / f"seed_{seed}"
            / f"round_{round_index}"
        )
        receipt, merged_result = _train_mopd_checkpoint(
            config,
            config_path,
            base_model=current,
            target_cache=target_cache,
            adapter=adapter,
            merged=merged,
            round_index=round_index,
            seed=seed,
            arm="primary",
        )
        rounds.append(
            {
                "round": round_index,
                "round_manifest": str(manifest.resolve()),
                "round_manifest_sha256": sha256_file(manifest),
                "target_cache": str(target_cache.resolve()),
                "target_cache_sha256": sha256_file(target_cache),
                "training_receipt": str((adapter / "training_receipt.json").resolve()),
                "training_receipt_sha256": sha256_file(adapter / "training_receipt.json"),
                "round_gate": receipt.get("round_gate"),
                "merged": str(merged.resolve()) if merged_result is not None else None,
                "merge_receipt_sha256": (
                    sha256_file(merged / "merge_receipt.json")
                    if merged_result is not None else None
                ),
            }
        )
        stage_receipt = {
            "schema_version": 1,
            "stage": "four_round_deep_advantage_routed_mopd",
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "seed": seed,
            "rounds": rounds,
            "completed_rounds": sum(
                bool(row.get("round_gate", {}).get("passed")) for row in rounds
            ),
            "gate": {
                "passed": len(rounds) == int(config["mopd"]["rounds"])
                and all(row.get("round_gate", {}).get("passed") for row in rounds)
            },
            "final_model": str(merged.resolve()) if merged_result is not None else None,
        }
        write_json(_integration_receipt_path(seed), stage_receipt)
        if merged_result is None or not receipt.get("round_gate", {}).get("passed"):
            raise SystemExit(
                f"primary integration seed {seed} failed safety gate in round {round_index}"
            )
        current = merged_result
    _require_gate(_integration_receipt_path(seed))


def _train_offpolicy_checkpoint(
    config: dict,
    config_path: Path,
    *,
    base_model: Path,
    manifest: Path,
    adapter: Path,
    merged: Path,
    round_index: int,
    seed: int,
    target_initial_loss: float,
) -> tuple[dict, Path | None]:
    if not _adapter_complete(adapter):
        if adapter.exists() and any(adapter.iterdir()):
            raise SystemExit(f"partial off-policy adapter: {adapter}")
        _run(
            [
                str(PY), str(EXP / "scripts" / "train_offpolicy_round.py"),
                "--config", str(config_path), "--base-model", str(base_model),
                "--round-manifest", str(manifest), "--out", str(adapter),
                "--round", str(round_index), "--seed", str(seed),
                "--target-initial-loss", repr(float(target_initial_loss)),
            ],
            training=True,
            allowed=(0, 3),
        )
    if not _adapter_complete(adapter):
        raise SystemExit(f"off-policy trainer left incomplete output: {adapter}")
    receipt = json.loads((adapter / "training_receipt.json").read_text(encoding="utf-8"))
    if (
        receipt.get("method") != "offpolicy_best_selection_continuation_sft"
        or int(receipt.get("round", -1)) != round_index
        or int(receipt.get("seed", -1)) != seed
        or Path(receipt.get("base_model", "")).resolve() != base_model.resolve()
        or receipt.get("round_manifest_sha256") != sha256_file(manifest)
    ):
        raise SystemExit(f"stale off-policy training receipt: {adapter}")
    if not receipt.get("round_gate", {}).get("passed"):
        return receipt, None
    _merge_round_adapter(adapter, base_model, merged)
    return receipt, merged


def _control_final_model(config: dict, name: str) -> Path:
    return (
        _paths(config)["root"] / "merged" / "controls" / name
        / f"round_{int(config['mopd']['rounds']) - 1}"
    )


def _build_parameter_controls(config: dict) -> dict[str, Path]:
    paths = _paths(config)
    result = {}
    for weight in config["controls"]["parameter_merge_deep_weights"]:
        value = float(weight)
        name = f"soup{int(round(value * 100)):02d}"
        out = paths["root"] / "merged" / name
        if not _merged_complete(out):
            if out.exists() and any(out.iterdir()):
                raise SystemExit(f"partial parameter-soup control: {out}")
            _run(
                [
                    str(PY), str(EXP / "scripts" / "merge_weighted_adapters.py"),
                    "--quick-adapter", str(paths["quick_adapter"]),
                    "--deep-adapter", str(paths["deep_adapter"]),
                    "--deep-weight", repr(value), "--out", str(out),
                ],
                training=True,
            )
        receipt = json.loads((out / "merge_receipt.json").read_text(encoding="utf-8"))
        if (
            abs(float(receipt.get("deep_weight", -1.0)) - value) > 1e-12
            or receipt.get("quick_weights_sha256")
            != sha256_file(paths["quick_adapter"] / "adapter_model.safetensors")
            or receipt.get("deep_weights_sha256")
            != sha256_file(paths["deep_adapter"] / "adapter_model.safetensors")
        ):
            raise SystemExit(f"parameter-soup control provenance failed: {out}")
        result[name] = out
    return result


def _controls(config: dict, config_path: Path) -> None:
    primary_seed = int(config["seeds"]["integration_training"][0])
    _require_gate(_integration_receipt_path(primary_seed))
    paths = _paths(config)
    arm_seeds = {
        "non_advantage_route": ("non_advantage_route", int(config["seeds"]["non_advantage_route"])),
        "wrong_teacher": ("wrong_teacher", int(config["seeds"]["wrong_teacher"])),
    }
    control_rows = {}
    for control_name, (arm, seed) in arm_seeds.items():
        current = paths["soup"]
        round_rows = []
        for round_index in range(int(config["mopd"]["rounds"])):
            data_root = (
                paths["root"] / "online" / "primary" / f"seed_{primary_seed}"
                / f"round_{round_index}"
            )
            manifest = data_root / "training_round.json"
            target_cache = data_root / "all_policy_targets.pt"
            if not manifest.is_file() or not target_cache.is_file():
                raise SystemExit(f"primary matched-control data missing: {data_root}")
            primary_adapter = (
                paths["root"] / "adapters" / "primary" / f"seed_{primary_seed}"
                / f"round_{round_index}"
            )
            primary_receipt = _require_round_gate(primary_adapter / "training_receipt.json")
            target_pressure = float(primary_receipt["initial_probe"]["mean_loss"])
            adapter = (
                paths["root"] / "adapters" / "controls" / control_name
                / f"round_{round_index}"
            )
            merged = (
                paths["root"] / "merged" / "controls" / control_name
                / f"round_{round_index}"
            )
            receipt, merged_result = _train_mopd_checkpoint(
                config,
                config_path,
                base_model=current,
                target_cache=target_cache,
                adapter=adapter,
                merged=merged,
                round_index=round_index,
                seed=seed,
                arm=arm,
                target_initial_loss=target_pressure,
            )
            round_rows.append(
                {
                    "round": round_index,
                    "primary_manifest_sha256": sha256_file(manifest),
                    "primary_target_cache_sha256": sha256_file(target_cache),
                    "training_receipt": str((adapter / "training_receipt.json").resolve()),
                    "training_receipt_sha256": sha256_file(adapter / "training_receipt.json"),
                    "round_gate": receipt.get("round_gate"),
                    "merged": str(merged.resolve()) if merged_result is not None else None,
                }
            )
            if merged_result is None:
                control_rows[control_name] = {
                    "seed": seed, "rounds": round_rows, "gate": {"passed": False}
                }
                write_json(EXP / "runs" / "controls.json", {
                    "schema_version": 1, "stage": "matched_controls",
                    "config_sha256": sha256_file(config_path), "controls": control_rows,
                    "gate": {"passed": False},
                })
                raise SystemExit(f"matched control {control_name} failed in round {round_index}")
            current = merged_result
        control_rows[control_name] = {
            "seed": seed,
            "rounds": round_rows,
            "final_model": str(current.resolve()),
            "final_merge_receipt_sha256": sha256_file(current / "merge_receipt.json"),
            "gate": {"passed": True},
        }

    offpolicy_name = "offpolicy_sft"
    offpolicy_seed = int(config["seeds"]["offpolicy_sft"])
    current = paths["soup"]
    offpolicy_rounds = []
    for round_index in range(int(config["mopd"]["rounds"])):
        data_root = (
            paths["root"] / "online" / "primary" / f"seed_{primary_seed}"
            / f"round_{round_index}"
        )
        manifest = data_root / "training_round.json"
        primary_receipt = _require_round_gate(
            paths["root"] / "adapters" / "primary" / f"seed_{primary_seed}"
            / f"round_{round_index}" / "training_receipt.json"
        )
        adapter = (
            paths["root"] / "adapters" / "controls" / offpolicy_name
            / f"round_{round_index}"
        )
        merged = (
            paths["root"] / "merged" / "controls" / offpolicy_name
            / f"round_{round_index}"
        )
        receipt, merged_result = _train_offpolicy_checkpoint(
            config,
            config_path,
            base_model=current,
            manifest=manifest,
            adapter=adapter,
            merged=merged,
            round_index=round_index,
            seed=offpolicy_seed,
            target_initial_loss=float(primary_receipt["initial_probe"]["mean_loss"]),
        )
        offpolicy_rounds.append(
            {
                "round": round_index,
                "primary_manifest_sha256": sha256_file(manifest),
                "training_receipt": str((adapter / "training_receipt.json").resolve()),
                "training_receipt_sha256": sha256_file(adapter / "training_receipt.json"),
                "round_gate": receipt.get("round_gate"),
                "merged": str(merged.resolve()) if merged_result is not None else None,
            }
        )
        if merged_result is None:
            control_rows[offpolicy_name] = {
                "seed": offpolicy_seed, "rounds": offpolicy_rounds,
                "gate": {"passed": False},
            }
            write_json(EXP / "runs" / "controls.json", {
                "schema_version": 1, "stage": "matched_controls",
                "config_sha256": sha256_file(config_path), "controls": control_rows,
                "gate": {"passed": False},
            })
            raise SystemExit(f"off-policy control failed in round {round_index}")
        current = merged_result
    control_rows[offpolicy_name] = {
        "seed": offpolicy_seed,
        "rounds": offpolicy_rounds,
        "final_model": str(current.resolve()),
        "final_merge_receipt_sha256": sha256_file(current / "merge_receipt.json"),
        "gate": {"passed": True},
    }
    parameter_models = _build_parameter_controls(config)
    result = {
        "schema_version": 1,
        "stage": "matched_controls",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "primary_seed": primary_seed,
        "controls": control_rows,
        "parameter_controls": {
            name: {
                "model": str(model.resolve()),
                "merge_receipt_sha256": sha256_file(model / "merge_receipt.json"),
            }
            for name, model in parameter_models.items()
        },
        "gate": {
            "passed": all(row.get("gate", {}).get("passed") for row in control_rows.values())
            and len(parameter_models)
            == len(config["controls"]["parameter_merge_deep_weights"])
        },
    }
    write_json(EXP / "runs" / "controls.json", result)
    _require_gate(EXP / "runs" / "controls.json")


def _validate_confirmation_score(
    path: Path,
    *,
    config_path: Path,
    model: Path,
    seed: int,
    decode: str,
) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_k = 1 if decode == "greedy" else 8
    if (
        payload.get("stage") != "policy_eval"
        or payload.get("scope") != "confirmatory"
        or payload.get("config_sha256") != sha256_file(config_path)
        or payload.get("model") != str(model.resolve())
        or payload.get("model_merge_receipt_sha256")
        != sha256_file(model / "merge_receipt.json")
        or int(payload.get("block_seed", -1)) != seed
        or payload.get("decode") != decode
        or int(payload.get("k", -1)) != expected_k
        or not all(payload.get("engine_protocol", {}).values())
        or len(payload.get("items") or []) == 0
    ):
        raise SystemExit(f"stale or invalid confirmation score artifact: {path}")
    return payload


def _confirm(config: dict, config_path: Path) -> None:
    seeds = [int(value) for value in config["seeds"]["integration_training"]]
    for seed in seeds:
        _require_gate(_integration_receipt_path(seed))
    _require_gate(EXP / "runs" / "controls.json")
    paths = _paths(config)
    parameter_models = _build_parameter_controls(config)
    model_arms: dict[str, Path] = {
        "quick": paths["quick"],
        "deep": paths["deep"],
        "soup": paths["soup"],
        "primary_seed42": _integration_final_model(config, seeds[0]),
        "primary_seed43": _integration_final_model(config, seeds[1]),
        "primary_seed44": _integration_final_model(config, seeds[2]),
        "non_advantage_route": _control_final_model(config, "non_advantage_route"),
        "wrong_teacher": _control_final_model(config, "wrong_teacher"),
        "offpolicy_sft": _control_final_model(config, "offpolicy_sft"),
        **parameter_models,
    }
    for name, model in model_arms.items():
        if not _merged_complete(model):
            raise SystemExit(f"confirmation arm {name} is incomplete: {model}")
    evaluation_arms = {**model_arms, "soup_best8": paths["soup"]}
    block_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    score_paths: dict[str, list[str]] = {name: [] for name in evaluation_arms}
    for block_index, block_seed in enumerate(block_seeds):
        for name, model in evaluation_arms.items():
            decode = "sample8" if name == "soup_best8" else "greedy"
            out_dir = EXP / "runs" / "confirmation" / f"block_{block_index}" / name
            score_path = out_dir / "scores.json"
            if not score_path.is_file():
                if out_dir.exists() and any(out_dir.iterdir()):
                    raise SystemExit(f"partial confirmation arm output: {out_dir}")
                _run(
                    [
                        str(VLLM_PY), str(EXP / "scripts" / "eval_policy.py"),
                        "--config", str(config_path), "--model", str(model),
                        "--tag", f"block_{block_index}_{name}",
                        "--scope", "confirmatory", "--block-seed", str(block_seed),
                        "--decode", decode, "--out-dir", str(out_dir),
                    ]
                )
            _validate_confirmation_score(
                score_path,
                config_path=config_path,
                model=model,
                seed=block_seed,
                decode=decode,
            )
            score_paths[name].append(str(score_path.resolve()))
    strict = [
        "quick", "deep", "soup", "non_advantage_route", "wrong_teacher",
        "offpolicy_sft", "soup25", "soup50", "soup75",
    ]
    manifest = {
        "schema_version": 1,
        "stage": "sealed_confirmation_manifest",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "block_seeds": block_seeds,
        "primary_arm": "primary_seed42",
        "replicate_arms": ["primary_seed43", "primary_seed44"],
        "quick_arm": "quick",
        "deep_arm": "deep",
        "soup_arm": "soup",
        "sample_more_arm": "soup_best8",
        "strict_comparator_arms": strict,
        "arms": score_paths,
        "model_merge_receipts": {
            name: sha256_file(model / "merge_receipt.json")
            for name, model in evaluation_arms.items()
        },
    }
    manifest_path = EXP / "runs" / "confirmation" / "manifest.json"
    write_json(manifest_path, manifest)
    code = _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_confirmation.py"),
            "--config", str(config_path), "--manifest", str(manifest_path),
            "--out", str(EXP / "analysis" / "confirmation.json"),
        ],
        allowed=(0, 4),
    )
    if code != 0:
        raise SystemExit(code)
    _require_gate(EXP / "analysis" / "confirmation.json")


def _benchmark(config: dict, config_path: Path) -> None:
    _require_gate(EXP / "analysis" / "confirmation.json")
    paths = _paths(config)
    primary = _integration_final_model(
        config, int(config["seeds"]["integration_training"][0])
    )
    first = int(config["benchmark"]["first_seed"])
    quick_n = int(config["benchmark"]["quick_events"])
    medium_n = int(config["benchmark"]["medium_events"])
    tier_seeds = {
        "quick": list(range(first, first + quick_n)),
        "medium": list(range(first + quick_n, first + quick_n + medium_n)),
    }
    events = []
    for tier, seeds in tier_seeds.items():
        models = {
            "primary": primary,
            "soup": paths["soup"],
            "visible": paths["quick"] if tier == "quick" else paths["deep"],
        }
        for seed in seeds:
            for label, model in models.items():
                out = (
                    EXP
                    / "runs"
                    / "benchmark"
                    / tier
                    / f"seed_{seed}"
                    / f"{label}.json"
                )
                _run(
                    [
                        str(PY),
                        str(EXP / "scripts" / "bench.py"),
                        "--tier",
                        tier,
                        "--seed",
                        str(seed),
                        "--label",
                        label,
                        "--model",
                        str(model),
                        "--out",
                        str(out),
                    ]
                )
                events.append(out)
    code = _run(
        [
            str(PY),
            str(EXP / "scripts" / "analyze_benchmark.py"),
            *[value for path in events for value in ("--event", str(path))],
            "--out",
            str(EXP / "analysis" / "benchmark.json"),
        ],
        allowed=(0, 4),
    )
    if code != 0:
        raise SystemExit(code)
    _require_gate(EXP / "analysis" / "benchmark.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "model-smoke", "verify-student", "route-qualify", "locality",
            "integrate", "controls", "confirm", "benchmark",
        ),
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lock-design", metavar="COMMIT")
    args = parser.parse_args()
    config, config_path = load_config()
    if args.lock_design:
        if args.smoke or args.stage or args.seed is not None:
            parser.error("--lock-design cannot be combined with a run stage")
        _write_preregistration_receipt(args.lock_design)
        return 0
    if args.smoke:
        scientific_smoke(config, config_path)
        return 0
    if args.stage is None:
        parser.error("choose --smoke, --lock-design, or one explicit --stage")
    if args.seed is not None and args.stage != "integrate":
        parser.error("--seed is valid only with --stage integrate")
    _verify_preregistration()
    if args.stage == "model-smoke":
        _model_smoke(config)
        return 0
    if args.stage == "verify-student":
        _verify_student(config, config_path)
        return 0
    if args.stage == "route-qualify":
        _route_qualify(config, config_path)
        return 0
    if args.stage == "locality":
        _locality(config, config_path)
        return 0
    if args.stage == "integrate":
        _integrate(config, config_path, args.seed)
        return 0
    if args.stage == "controls":
        _controls(config, config_path)
        return 0
    if args.stage == "confirm":
        _confirm(config, config_path)
        return 0
    if args.stage == "benchmark":
        _benchmark(config, config_path)
        return 0
    raise AssertionError(f"unhandled stage {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
