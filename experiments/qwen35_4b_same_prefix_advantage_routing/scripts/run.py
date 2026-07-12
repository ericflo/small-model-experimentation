#!/usr/bin/env python3
"""Fail-closed orchestration for same-prefix advantage routing and MOPD."""

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
        "soup": root / "merged" / "soup40",
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
        int(config["seeds"]["benchmark_first"]),
    ]
    if len(seeds) != len(set(seeds)):
        raise AssertionError("generation seed namespaces overlap")
    route_states = int(config["generation"]["qualification_atom_states_per_block"]) + int(
        config["generation"]["qualification_episode_states_per_block"]
    )
    minimum = int(config["route"]["minimum_routed_states_per_teacher_per_block"])
    if route_states < 2 * minimum:
        raise AssertionError("route support gate is mathematically unreachable")
    micro_steps = int(config["mopd"]["updates_per_round"]) * int(config["mopd"]["grad_accum"])
    capability = round(micro_steps * float(config["mopd"]["capability_fraction"]))
    anchor = micro_steps - capability
    if capability % 2 or capability < 2 or anchor < 1:
        raise AssertionError("MOPD equal-teacher/anchor quotas are not integer-reachable")
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
            "minimum_per_teacher_per_block": minimum,
            "micro_steps_per_round": micro_steps,
            "capability_units_per_round": capability,
            "units_per_teacher_per_round": capability // 2,
            "anchor_units_per_round": anchor,
        },
        "source_artifacts": sources,
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
    write_json(EXP / "runs" / "checkpoint_receipts.json", payload)


def _build_student(config: dict, config_path: Path) -> None:
    if not (EXP / "runs" / "model_smoke" / "hf.json").is_file():
        raise SystemExit("run model-smoke before building the student")
    paths = _paths(config)
    if not _source_complete(paths["quick_adapter"], paths["quick"]):
        raise SystemExit("quick source checkpoint incomplete")
    if not _source_complete(paths["deep_adapter"], paths["deep"]):
        raise SystemExit("deep source checkpoint incomplete")
    soup = paths["soup"]
    if not _merged_complete(soup):
        if soup.exists() and any(soup.iterdir()):
            raise SystemExit(f"partial soup checkpoint exists: {soup}")
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_weighted_adapters.py"),
                "--quick-adapter", str(paths["quick_adapter"]),
                "--deep-adapter", str(paths["deep_adapter"]),
                "--deep-weight", str(config["model"]["student_deep_weight"]),
                "--out", str(soup),
            ],
            training=True,
        )
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "model-smoke", "build-student", "route-qualify", "locality",
            "integrate", "controls", "confirm",
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
    _verify_preregistration()
    if args.stage == "model-smoke":
        _model_smoke(config)
        return 0
    if args.stage == "build-student":
        _build_student(config, config_path)
        return 0
    if args.stage == "route-qualify":
        _route_qualify(config, config_path)
        return 0
    raise SystemExit(
        f"stage {args.stage!r} is frozen but implementation is not yet accepted; "
        "complete and test it before any outcome-bearing run"
    )


if __name__ == "__main__":
    raise SystemExit(main())

