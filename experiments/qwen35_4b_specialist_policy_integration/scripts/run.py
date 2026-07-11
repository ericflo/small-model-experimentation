#!/usr/bin/env python3
"""Resumable gated orchestration for specialist policy integration."""

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
SRC = EXP / "src"
PY = REPO / ".venv" / "bin" / "python"
VLLM_PY = REPO / ".venv-vllm" / "bin" / "python"
sys.path.insert(0, str(SRC))

from curriculum import expert_decision  # noqa: E402
from gym.families import load  # noqa: E402
from io_utils import (  # noqa: E402
    canonical_hash,
    load_config,
    resolve_repo_path,
    sha256_file,
    training_seed,
    write_json,
)


DOMAINS = ("discover", "control", "tools", "compose")
COMPOUND_FAMILIES = ("cipherkiln", "mazeferry", "patchferry", "tripleforge")


def _run(command: list[str], *, training: bool = False, allowed: tuple[int, ...] = (0,)) -> int:
    print("[stage] " + " ".join(command), flush=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if training:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    completed = subprocess.run(command, cwd=REPO, env=env, check=False)
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def _paths(config: dict, domain: str | None = None) -> dict[str, Path]:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    common = {
        "root": root,
        "incumbent_adapter": root / "adapters" / "incumbent_blend",
        "incumbent": root / "merged" / "incumbent_blend",
    }
    if domain is None:
        return common
    common.update(
        {
            "dagger_adapter": root / "adapters" / "dagger" / domain,
            "dagger": root / "merged" / "dagger" / domain,
            "specialist_adapter": root / "adapters" / "specialist" / domain,
            "specialist": root / "merged" / "specialist" / domain,
            "extra_sft_adapter": root / "adapters" / "extra_sft" / domain,
            "extra_sft": root / "merged" / "extra_sft" / domain,
            "shuffled_adapter": root / "adapters" / "shuffled" / domain,
            "shuffled": root / "merged" / "shuffled" / domain,
        }
    )
    return common


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


def _assert_clean_output(adapter: Path, merged: Path) -> None:
    if adapter.exists() or merged.exists():
        raise SystemExit(
            "partial or unreceipted checkpoint exists; inspect and preserve it before retrying: "
            f"adapter={adapter}, merged={merged}"
        )


def _record_checkpoint(tag: str, adapter: Path, merged: Path) -> None:
    receipt = {
        "tag": tag,
        "adapter": str(adapter.resolve()),
        "merged": str(merged.resolve()),
        "training_receipt": json.loads((adapter / "training_receipt.json").read_text(encoding="utf-8")),
        "merge_receipt": json.loads((merged / "merge_receipt.json").read_text(encoding="utf-8")),
    }
    path = EXP / "runs" / "checkpoint_receipts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []
    existing = [row for row in existing if row.get("tag") != tag] + [receipt]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in existing),
        encoding="utf-8",
    )


def _train_and_merge(
    *,
    tag: str,
    model: str | Path,
    train_file: Path,
    adapter: Path,
    merged: Path,
    cfg: dict,
    seed: int,
    max_steps: int = -1,
    smoke: bool = False,
) -> None:
    if _checkpoint_complete(adapter, merged):
        print(f"[resume] checkpoint {tag} already complete", flush=True)
        return
    _assert_clean_output(adapter, merged)
    command = [
            str(PY), str(EXP / "scripts" / "train_dagger.py"),
            "--model", str(model), "--train", str(train_file), "--out", str(adapter),
            "--epochs", str(cfg.get("epochs", 1.0)), "--max-steps", str(max_steps),
            "--lr", str(cfg["learning_rate"]), "--rank", str(cfg["rank"]),
            "--alpha", str(cfg["alpha"]), "--batch-size", str(cfg.get("batch_size", 1)),
            "--grad-accum", str(cfg["grad_accum"]), "--max-length", str(cfg["max_length"]),
            "--w-think", str(cfg["think_loss_weight"]), "--seed", str(seed),
        ]
    if smoke:
        command.append("--smoke")
    _run(command, training=True)
    _run(
        [
            str(PY), str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(model), "--adapter", str(adapter), "--out", str(merged),
        ],
        training=True,
    )
    _record_checkpoint(tag, adapter, merged)


def _train_grpo(
    *,
    tag: str,
    model: Path,
    trajectories: Path,
    anchors: Path,
    adapter: Path,
    merged: Path,
    config_path: Path,
    seed: int,
    shuffled: bool,
) -> None:
    if _checkpoint_complete(adapter, merged):
        print(f"[resume] checkpoint {tag} already complete", flush=True)
        return
    _assert_clean_output(adapter, merged)
    command = [
        str(PY), str(EXP / "scripts" / "train_sequence_grpo.py"),
        "--config", str(config_path), "--model", str(model),
        "--trajectories", str(trajectories), "--anchors", str(anchors),
        "--out", str(adapter), "--seed", str(seed), "--run-tag", f"training_{tag}",
    ]
    if shuffled:
        command.append("--shuffle-advantages")
    return_code = _run(command, training=True, allowed=(0, 3))
    if return_code == 3:
        print(f"[gate] {tag} hit its registered KL/non-finite stop; merging stopped checkpoint", flush=True)
    _run(
        [
            str(PY), str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(model), "--adapter", str(adapter), "--out", str(merged),
        ],
        training=True,
    )
    _record_checkpoint(tag, adapter, merged)


def _expert_score(family_name: str, seed: int, level: int) -> float:
    family = load(family_name)
    episode = family.Episode(seed, level)
    messages = [
        {"role": "system", "content": episode.system_prompt()},
        {"role": "user", "content": episode.initial_observation()},
    ]
    for _ in range(episode.max_turns):
        decision = expert_decision(family_name, episode, messages)
        observation, done = episode.step(decision.action)
        if not episode.last_action_ok:
            raise AssertionError((family_name, level, decision.action, observation))
        messages.extend(
            [{"role": "assistant", "content": decision.action}, {"role": "user", "content": observation}]
        )
        if done:
            break
    return float(episode.score())


def scientific_smoke(config: dict, config_path: Path) -> dict:
    for path in sorted(list((EXP / "src").rglob("*.py")) + list((EXP / "scripts").glob("*.py"))):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    _run([str(PY), str(EXP / "tests" / "test_curriculum.py")])
    _run([str(PY), str(EXP / "tests" / "test_mopd_loss.py")])
    _run([str(VLLM_PY), str(EXP / "tests" / "test_vllm_runner.py")])
    completed = subprocess.run(
        [str(PY), str(EXP / "scripts" / "selftest_gym.py"), "--families", *COMPOUND_FAMILIES],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    expert_scores = {
        family_name: {
            str(level): _expert_score(family_name, 99000 + level, level)
            for level in load(family_name).LEVELS
        }
        for family_name in COMPOUND_FAMILIES
    }
    train = set(config["split"]["train_families"])
    transfer = set(config["split"]["transfer_families"])
    replay_excluded = set(config["split"]["replay_excluded_families"])
    if train & transfer:
        raise AssertionError(f"train/transfer overlap: {sorted(train & transfer)}")
    if transfer != replay_excluded:
        raise AssertionError("every transfer family must be excluded from replay")
    if any(score < 0.999 for row in expert_scores.values() for score in row.values()):
        raise AssertionError("a state-aware compound expert failed")
    payload = {
        "status": "pass",
        "config": str(config_path.relative_to(EXP)),
        "config_sha256": canonical_hash(config),
        "compound_families": list(COMPOUND_FAMILIES),
        "expert_scores": expert_scores,
        "train_families": sorted(train),
        "transfer_families": sorted(transfer),
        "selftest_stdout": completed.stdout.strip().splitlines(),
    }
    write_json(EXP / "runs" / "smoke" / "summary.json", payload)
    return payload


def _require_gate(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"required gate receipt missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("gate", {}).get("passed"):
        raise SystemExit(f"required gate did not pass: {path}")
    return payload


def _model_smoke(config: dict, config_path: Path) -> None:
    out = EXP / "runs" / "model_smoke" / "base.jsonl"
    hf = EXP / "runs" / "model_smoke" / "hf.json"
    smoke_root = resolve_repo_path(config["model"]["artifacts_root"]) / "smoke"
    smoke_adapter = smoke_root / "adapter"
    smoke_merged = smoke_root / "merged"
    merged_output = EXP / "runs" / "model_smoke" / "merged.jsonl"
    merged_meta = merged_output.with_name(merged_output.name + ".meta.json")
    current_training_lock_sha = sha256_file(REPO / "requirements-training.lock.txt")
    hf_payload = json.loads(hf.read_text(encoding="utf-8")) if hf.exists() else {}
    hf_current = (
        hf_payload.get("status") == "pass"
        and hf_payload.get("training_lock", {}).get("sha256") == current_training_lock_sha
    )
    if (
        hf_current
        and _checkpoint_complete(smoke_adapter, smoke_merged)
        and merged_meta.exists()
    ):
        print("[resume] model smoke already passed", flush=True)
        return
    if not hf_current:
        _run(
            [
                str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"), "--smoke", "4",
                "--output", str(out), "--thinking", "off", "--greedy", "--max-tokens", "32",
                "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
                "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
                "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
                "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
                "--cudagraph-capture-size", "16",
            ]
        )
        _run([str(PY), str(EXP / "scripts" / "model_smoke.py"), "--vllm-output", str(out), "--out", str(hf)])
    smoke_cfg = dict(config["incumbent_train"])
    smoke_cfg.update(batch_size=1, grad_accum=1)
    _train_and_merge(
        tag="runtime_train_merge_smoke",
        model=config["model"]["id"],
        train_file=resolve_repo_path(config["model"]["incumbent_data"]),
        adapter=smoke_adapter,
        merged=smoke_merged,
        cfg=smoke_cfg,
        seed=training_seed(config),
        max_steps=2,
        smoke=True,
    )
    _run(
        [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"), "--smoke", "4",
            "--output", str(merged_output), "--model-override", str(smoke_merged),
            "--thinking", "off", "--greedy", "--max-tokens", "32",
            "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
            "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
            "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
            "--cudagraph-capture-size", "16",
        ]
    )
    metadata = json.loads(merged_meta.read_text(encoding="utf-8"))
    if metadata.get("model") != str(smoke_merged.resolve()):
        raise SystemExit("merged-composite smoke did not load the requested local checkpoint")
    if metadata.get("model_revision") is not None or not metadata.get("model_config_sha256"):
        raise SystemExit("merged-composite model fingerprint is incomplete")
    merge = json.loads((smoke_merged / "merge_receipt.json").read_text(encoding="utf-8"))
    if int(merge.get("nonzero_lora_modules", 0)) < 1:
        raise SystemExit("training/merge smoke produced no nonzero LoRA delta")
    receipt = {
        "status": "pass",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "local_model": str(smoke_merged.resolve()),
        "local_model_config_sha256": metadata["model_config_sha256"],
        "merge_receipt_sha256": sha256_file(smoke_merged / "merge_receipt.json"),
        "nonzero_lora_modules": merge["nonzero_lora_modules"],
        "vllm_metadata_sha256": sha256_file(merged_meta),
    }
    write_json(EXP / "runs" / "model_smoke" / "composite.json", receipt)


def _incumbent_canary(config: dict, config_path: Path, merged: Path) -> None:
    canary_dir = EXP / "runs" / "incumbent_merge_canary"
    base_out = canary_dir / "base.jsonl"
    candidate_out = canary_dir / "incumbent.jsonl"
    gate_path = EXP / "analysis" / "incumbent_install_gate.json"
    merge_receipt = merged / "merge_receipt.json"
    current_merge_sha = sha256_file(merge_receipt)
    if gate_path.exists():
        existing = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            existing.get("gate", {}).get("passed")
            and existing.get("merge_receipt_sha256") == current_merge_sha
        ):
            print("[resume] incumbent installation gate already passed", flush=True)
            return
    prompt_path = EXP / "data" / "incumbent_merge_canary.jsonl"
    _run(
        [
            str(PY), str(EXP / "scripts" / "build_merge_canary.py"),
            "--config", str(config_path), "--out", str(prompt_path),
        ]
    )
    common = [
        "--input", str(prompt_path), "--thinking", "budget",
        "--thinking-budget", "256", "--answer-max-tokens", "96", "--greedy",
        "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
        "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
        "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
        "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
        "--cudagraph-capture-size", "16",
    ]
    _run(
        [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
            "--output", str(base_out), *common,
        ]
    )
    _run(
        [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
            "--output", str(candidate_out), "--model-override", str(merged), *common,
        ]
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_merge_canary.py"),
            "--base", str(base_out), "--candidate", str(candidate_out),
            "--merged", str(merged), "--out", str(gate_path),
        ],
        allowed=(0, 4),
    )
    _require_gate(gate_path)


def _checkpoint_canary_output(
    config_path: Path,
    model: Path,
    cache_tag: str,
) -> Path:
    """Generate or verify one reusable local-composite behavioral canary."""
    output_dir = EXP / "runs" / "checkpoint_canaries"
    output = output_dir / f"{cache_tag}.jsonl"
    metadata = output.with_name(output.name + ".meta.json")
    receipt_path = output.with_name(output.name + ".receipt.json")
    merge_path = model / "merge_receipt.json"
    merge_sha = sha256_file(merge_path)
    runner_sha = sha256_file(EXP / "src" / "vllm_runner.py")
    protocol = {
        "thinking": "budget",
        "thinking_budget": 256,
        "answer_max_tokens": 96,
        "greedy": True,
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.85,
        "max_num_seqs": 16,
        "max_num_batched_tokens": 4096,
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16],
    }
    protocol_sha = canonical_hash(protocol)
    if output.exists() or metadata.exists() or receipt_path.exists():
        if not all(path.is_file() for path in (output, metadata, receipt_path)):
            raise SystemExit(f"partial checkpoint canary exists for {cache_tag}")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        if not (
            receipt.get("model") == str(model.resolve())
            and receipt.get("merge_receipt_sha256") == merge_sha
            and receipt.get("runner_sha256") == runner_sha
            and receipt.get("protocol_sha256") == protocol_sha
            and receipt.get("output_sha256") == sha256_file(output)
            and receipt.get("metadata_sha256") == sha256_file(metadata)
            and meta.get("model") == str(model.resolve())
            and meta.get("model_revision") is None
            and meta.get("runner_sha256") == runner_sha
        ):
            raise SystemExit(f"stale checkpoint canary exists for {cache_tag}")
        print(f"[resume] checkpoint canary output {cache_tag} is current", flush=True)
        return output
    prompt_path = EXP / "data" / "incumbent_merge_canary.jsonl"
    _run(
        [
            str(PY), str(EXP / "scripts" / "build_merge_canary.py"),
            "--config", str(config_path), "--out", str(prompt_path),
        ]
    )
    _run(
        [
            str(VLLM_PY), str(EXP / "src" / "vllm_runner.py"),
            "--output", str(output), "--model-override", str(model),
            "--input", str(prompt_path), "--thinking", "budget",
            "--thinking-budget", "256", "--answer-max-tokens", "96", "--greedy",
            "--max-model-len", "4096", "--gpu-memory-utilization", "0.85",
            "--max-num-seqs", "16", "--max-num-batched-tokens", "4096",
            "--cudagraph-capture-size", "1", "--cudagraph-capture-size", "2",
            "--cudagraph-capture-size", "4", "--cudagraph-capture-size", "8",
            "--cudagraph-capture-size", "16",
        ]
    )
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    receipt = {
        "stage": "checkpoint_canary_output",
        "model": str(model.resolve()),
        "merge_receipt_sha256": merge_sha,
        "runner_sha256": runner_sha,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "output_sha256": sha256_file(output),
        "metadata_sha256": sha256_file(metadata),
        "model_config_sha256": meta.get("model_config_sha256"),
    }
    write_json(receipt_path, receipt)
    return output


def _checkpoint_canary(
    config_path: Path,
    *,
    base: Path,
    candidate: Path,
    base_tag: str,
    candidate_tag: str,
    pair_tag: str,
) -> None:
    """Require a structural and greedy behavioral installation proof."""
    gate_path = EXP / "analysis" / "checkpoint_installation" / f"{pair_tag}.json"
    base_merge_sha = sha256_file(base / "merge_receipt.json")
    candidate_merge_sha = sha256_file(candidate / "merge_receipt.json")
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        if (
            gate.get("gate", {}).get("passed")
            and gate.get("base_merge_receipt_sha256") == base_merge_sha
            and gate.get("merge_receipt_sha256") == candidate_merge_sha
        ):
            print(f"[resume] checkpoint installation gate {pair_tag} passed", flush=True)
            return
    base_output = _checkpoint_canary_output(config_path, base, base_tag)
    candidate_output = _checkpoint_canary_output(
        config_path, candidate, candidate_tag
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_merge_canary.py"),
            "--base", str(base_output), "--candidate", str(candidate_output),
            "--base-model", str(base), "--merged", str(candidate),
            "--out", str(gate_path),
        ],
        allowed=(0, 4),
    )
    _require_gate(gate_path)


def _eval(
    config_path: Path,
    config: dict,
    model: Path,
    tag: str,
    decode: str = "greedy",
    *,
    smoke: bool = False,
    families: tuple[str, ...] | None = None,
    scope: str = "calibration",
    episode_seed_base: int | None = None,
    no_atoms: bool = False,
) -> None:
    out_dir = EXP / "runs" / "proxy_eval" / tag
    scores = out_dir / "scores.json"
    merge_receipt = model / "merge_receipt.json"
    current_receipt_sha = sha256_file(merge_receipt) if merge_receipt.exists() else None
    expected_families = (
        list(families)
        if families is not None
        else list(config["split"]["train_families"])
        + list(config["split"]["transfer_families"])
    )
    if smoke:
        expected_families = expected_families[:1]
    expected_levels = [1] if smoke else [int(value) for value in config["proxy_eval"]["levels"]]
    expected_episodes = (
        1
        if smoke
        else int(
            config["proxy_eval"][
                "confirmatory_episodes_per_level"
                if scope == "confirmatory"
                else "calibration_episodes_per_level"
            ]
        )
    )
    expected_seed_base = int(
        episode_seed_base
        if episode_seed_base is not None
        else config["seeds"]["proxy_eval_base"]
    )
    if scores.exists():
        previous = json.loads(scores.read_text(encoding="utf-8"))
        fingerprint = previous.get("model_fingerprint", {})
        if (
            previous.get("model") == str(model.resolve())
            and fingerprint.get("merge_receipt_sha256") == current_receipt_sha
            and previous.get("decode") == decode
            and bool(previous.get("smoke")) == smoke
            and previous.get("scope") == scope
            and previous.get("families") == expected_families
            and previous.get("levels") == expected_levels
            and int(previous.get("episodes_per_level", -1)) == expected_episodes
            and int(previous.get("episode_seed_base", -1)) == expected_seed_base
            and bool(previous.get("atoms_enabled")) == (not no_atoms)
        ):
            print(f"[resume] evaluation {tag} already complete", flush=True)
            return
        raise SystemExit(f"stale evaluation directory exists: {out_dir}")
    command = [
            str(VLLM_PY), str(EXP / "scripts" / "eval_proxy.py"),
            "--config", str(config_path), "--model", str(model), "--tag", tag,
            "--scope", scope, "--decode", decode,
            "--episode-seed-base", str(expected_seed_base),
        ]
    if smoke:
        command.append("--smoke")
    if families is not None:
        command.extend(("--families", *families))
    if no_atoms:
        command.append("--no-atoms")
    _run(command)


def _calibration_gate(config: dict, config_path: Path, paths: dict[str, Path]) -> None:
    if not _checkpoint_complete(paths["incumbent_adapter"], paths["incumbent"]):
        raise SystemExit("incumbent checkpoint is incomplete")
    _incumbent_canary(config, config_path, paths["incumbent"])
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_incumbent.py"),
            "--config", str(config_path),
            "--adapter", str(paths["incumbent_adapter"]),
            "--merged", str(paths["incumbent"]),
            "--encoding-audit", str(EXP / "runs" / "incumbent_encoding_audit.json"),
            "--install-gate", str(EXP / "analysis" / "incumbent_install_gate.json"),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "incumbent_gate.json")
    _eval(
        config_path, config, paths["incumbent"],
        "incumbent_eval_smoke", smoke=True,
    )
    _eval(
        config_path, config, paths["incumbent"],
        "incumbent_compound_calibration", families=COMPOUND_FAMILIES,
        episode_seed_base=int(config["split"]["calibration_seed_base"]),
        no_atoms=True,
    )
    _run(
        [
            str(PY), str(EXP / "scripts" / "analyze_calibration.py"),
            "--config", str(config_path),
            "--scores", str(
                EXP / "runs" / "proxy_eval" / "incumbent_compound_calibration" / "scores.json"
            ),
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "calibration_gate.json")


def _baseline_eval(config: dict, config_path: Path, paths: dict[str, Path]) -> None:
    _require_gate(EXP / "analysis" / "calibration_gate.json")
    _eval(config_path, config, paths["incumbent"], "incumbent_calibration")
    _eval(
        config_path, config, paths["incumbent"],
        "incumbent_best8_calibration", decode="sample8", no_atoms=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "smoke", "model-smoke", "incumbent", "incumbent-canary",
            "calibration-gate", "baseline-eval", "calibrate", "dagger-collect",
            "dagger-train", "rl-collect", "specialist-train", "controls",
            "specialist-eval", "specialist-analyze", "specialist-summary",
        ),
    )
    parser.add_argument("--domain", choices=DOMAINS)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    stage = "smoke" if args.smoke else args.stage
    if stage is None:
        parser.error("pass --smoke or --stage")
    if stage in {"dagger-collect", "dagger-train", "rl-collect", "specialist-train", "controls", "specialist-eval", "specialist-analyze"} and args.domain is None:
        parser.error(f"stage {stage!r} requires --domain")

    if stage == "smoke":
        print(json.dumps(scientific_smoke(config, config_path), indent=2, sort_keys=True))
        return 0
    if stage == "model-smoke":
        _model_smoke(config, config_path)
        return 0

    paths = _paths(config, args.domain)
    seed = training_seed(config)
    if stage == "incumbent":
        _train_and_merge(
            tag="incumbent_blend", model=config["model"]["id"],
            train_file=resolve_repo_path(config["model"]["incumbent_data"]),
            adapter=paths["incumbent_adapter"], merged=paths["incumbent"],
            cfg=config["incumbent_train"], seed=seed,
        )
        audit_path = EXP / "runs" / "incumbent_encoding_audit.json"
        if not audit_path.exists():
            _run(
                [
                    str(PY), str(EXP / "scripts" / "audit_sft_encoding.py"),
                    "--train", str(resolve_repo_path(config["model"]["incumbent_data"])),
                    "--max-length", str(config["incumbent_train"]["max_length"]),
                    "--w-think", str(config["incumbent_train"]["think_loss_weight"]),
                    "--out", str(audit_path),
                ]
            )
    elif stage == "incumbent-canary":
        _incumbent_canary(config, config_path, paths["incumbent"])
    elif stage == "calibration-gate":
        _calibration_gate(config, config_path, paths)
    elif stage == "baseline-eval":
        _baseline_eval(config, config_path, paths)
    elif stage == "calibrate":
        _calibration_gate(config, config_path, paths)
        _baseline_eval(config, config_path, paths)
    elif stage == "dagger-collect":
        _require_gate(EXP / "analysis" / "calibration_gate.json")
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "collect_dagger.py"),
                "--config", str(config_path), "--model", str(paths["incumbent"]),
                "--domain", args.domain,
            ]
        )
    elif stage == "dagger-train":
        _require_gate(EXP / "analysis" / "calibration_gate.json")
        _train_and_merge(
            tag=f"dagger_{args.domain}", model=paths["incumbent"],
            train_file=EXP / "data" / f"dagger_{args.domain}.jsonl",
            adapter=paths["dagger_adapter"], merged=paths["dagger"],
            cfg=config["dagger_train"], seed=seed,
        )
        _checkpoint_canary(
            config_path,
            base=paths["incumbent"], candidate=paths["dagger"],
            base_tag="incumbent_blend", candidate_tag=f"dagger_{args.domain}",
            pair_tag=f"dagger_{args.domain}",
        )
    elif stage == "rl-collect":
        _require_gate(EXP / "analysis" / "calibration_gate.json")
        _require_gate(
            EXP / "analysis" / "checkpoint_installation" / f"dagger_{args.domain}.json"
        )
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "collect_rl.py"),
                "--config", str(config_path), "--model", str(paths["dagger"]),
                "--domain", args.domain,
            ]
        )
    elif stage == "specialist-train":
        smoke_out = paths["root"] / "smoke" / "sequence_grpo" / args.domain
        if not (smoke_out / "training_receipt.json").exists():
            if smoke_out.exists():
                raise SystemExit(f"partial sequence-GRPO smoke exists: {smoke_out}")
            _run(
                [
                    str(PY), str(EXP / "scripts" / "train_sequence_grpo.py"),
                    "--config", str(config_path), "--model", str(paths["dagger"]),
                    "--trajectories", str(
                        EXP / "runs" / "rl_collection" / args.domain / "trajectories.jsonl.gz"
                    ),
                    "--anchors", str(EXP / "data" / f"rl_anchor_{args.domain}.jsonl"),
                    "--out", str(smoke_out), "--seed", str(seed),
                    "--run-tag", f"smoke_sequence_grpo_{args.domain}", "--smoke",
                ],
                training=True,
            )
        _train_grpo(
            tag=f"specialist_{args.domain}", model=paths["dagger"],
            trajectories=EXP / "runs" / "rl_collection" / args.domain / "trajectories.jsonl.gz",
            anchors=EXP / "data" / f"rl_anchor_{args.domain}.jsonl",
            adapter=paths["specialist_adapter"], merged=paths["specialist"],
            config_path=config_path, seed=seed, shuffled=False,
        )
        _checkpoint_canary(
            config_path,
            base=paths["dagger"], candidate=paths["specialist"],
            base_tag=f"dagger_{args.domain}",
            candidate_tag=f"specialist_{args.domain}",
            pair_tag=f"specialist_{args.domain}",
        )
    elif stage == "controls":
        control_cfg = {
            "epochs": 1.0,
            "learning_rate": config["rl_train"]["learning_rate"],
            "rank": config["rl_train"]["rank"],
            "alpha": config["rl_train"]["alpha"],
            "batch_size": 1,
            "grad_accum": config["rl_train"]["grad_accum"],
            "max_length": config["rl_train"]["max_length"],
            "think_loss_weight": config["rl_train"]["think_loss_weight"],
        }
        _train_and_merge(
            tag=f"extra_sft_{args.domain}", model=paths["dagger"],
            train_file=EXP / "data" / f"rl_anchor_{args.domain}.jsonl",
            adapter=paths["extra_sft_adapter"], merged=paths["extra_sft"],
            cfg=control_cfg, seed=seed, max_steps=int(config["controls"]["matched_sft_steps"]),
        )
        _checkpoint_canary(
            config_path,
            base=paths["dagger"], candidate=paths["extra_sft"],
            base_tag=f"dagger_{args.domain}",
            candidate_tag=f"extra_sft_{args.domain}",
            pair_tag=f"extra_sft_{args.domain}",
        )
        _train_grpo(
            tag=f"shuffled_{args.domain}", model=paths["dagger"],
            trajectories=EXP / "runs" / "rl_collection" / args.domain / "trajectories.jsonl.gz",
            anchors=EXP / "data" / f"rl_anchor_{args.domain}.jsonl",
            adapter=paths["shuffled_adapter"], merged=paths["shuffled"],
            config_path=config_path, seed=int(config["seeds"]["shuffled_reward"]), shuffled=True,
        )
        _checkpoint_canary(
            config_path,
            base=paths["dagger"], candidate=paths["shuffled"],
            base_tag=f"dagger_{args.domain}",
            candidate_tag=f"shuffled_{args.domain}",
            pair_tag=f"shuffled_{args.domain}",
        )
    elif stage == "specialist-eval":
        for gate_tag in (
            f"dagger_{args.domain}",
            f"extra_sft_{args.domain}",
            f"shuffled_{args.domain}",
            f"specialist_{args.domain}",
        ):
            _require_gate(
                EXP / "analysis" / "checkpoint_installation" / f"{gate_tag}.json"
            )
        for tag, model, no_atoms in (
            (f"dagger_{args.domain}", paths["dagger"], True),
            (f"extra_sft_{args.domain}", paths["extra_sft"], True),
            (f"shuffled_{args.domain}", paths["shuffled"], True),
            (f"specialist_{args.domain}", paths["specialist"], False),
        ):
            _eval(config_path, config, model, tag, no_atoms=no_atoms)
    elif stage == "specialist-analyze":
        command = [
            str(PY), str(EXP / "scripts" / "analyze_specialist.py"),
            "--config", str(config_path), "--domain", args.domain,
            "--incumbent", str(EXP / "runs" / "proxy_eval" / "incumbent_calibration"),
            "--incumbent-best8", str(EXP / "runs" / "proxy_eval" / "incumbent_best8_calibration"),
            "--dagger", str(EXP / "runs" / "proxy_eval" / f"dagger_{args.domain}"),
            "--extra-sft", str(EXP / "runs" / "proxy_eval" / f"extra_sft_{args.domain}"),
            "--shuffled", str(EXP / "runs" / "proxy_eval" / f"shuffled_{args.domain}"),
            "--specialist", str(EXP / "runs" / "proxy_eval" / f"specialist_{args.domain}"),
        ]
        _run(command, allowed=(0, 4))
    elif stage == "specialist-summary":
        _run(
            [str(PY), str(EXP / "scripts" / "aggregate_specialist_gates.py")],
            allowed=(0, 4),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
