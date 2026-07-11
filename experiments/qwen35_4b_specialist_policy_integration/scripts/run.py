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


def _eval(config_path: Path, config: dict, model: Path, tag: str, decode: str = "greedy") -> None:
    out_dir = EXP / "runs" / "proxy_eval" / tag
    scores = out_dir / "scores.json"
    merge_receipt = model / "merge_receipt.json"
    current_receipt_sha = sha256_file(merge_receipt) if merge_receipt.exists() else None
    if scores.exists():
        previous = json.loads(scores.read_text(encoding="utf-8"))
        fingerprint = previous.get("model_fingerprint", {})
        if (
            previous.get("model") == str(model.resolve())
            and fingerprint.get("merge_receipt_sha256") == current_receipt_sha
            and previous.get("decode") == decode
        ):
            print(f"[resume] evaluation {tag} already complete", flush=True)
            return
        raise SystemExit(f"stale evaluation directory exists: {out_dir}")
    _run(
        [
            str(VLLM_PY), str(EXP / "scripts" / "eval_proxy.py"),
            "--config", str(config_path), "--model", str(model), "--tag", tag,
            "--scope", "calibration", "--decode", decode,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "smoke", "model-smoke", "incumbent", "calibrate", "dagger-collect",
            "dagger-train", "rl-collect", "specialist-train", "controls",
            "specialist-eval", "specialist-analyze",
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
    elif stage == "calibrate":
        if not _checkpoint_complete(paths["incumbent_adapter"], paths["incumbent"]):
            raise SystemExit("incumbent checkpoint is incomplete")
        _eval(config_path, config, paths["incumbent"], "incumbent_calibration")
        _eval(config_path, config, paths["incumbent"], "incumbent_best8_calibration", decode="sample8")
        _run(
            [
                str(PY), str(EXP / "scripts" / "analyze_calibration.py"),
                "--config", str(config_path),
                "--scores", str(EXP / "runs" / "proxy_eval" / "incumbent_calibration" / "scores.json"),
            ],
            allowed=(0, 4),
        )
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
    elif stage == "rl-collect":
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "collect_rl.py"),
                "--config", str(config_path), "--model", str(paths["dagger"]),
                "--domain", args.domain,
            ]
        )
    elif stage == "specialist-train":
        _train_grpo(
            tag=f"specialist_{args.domain}", model=paths["dagger"],
            trajectories=EXP / "runs" / "rl_collection" / args.domain / "trajectories.jsonl.gz",
            anchors=EXP / "data" / f"rl_anchor_{args.domain}.jsonl",
            adapter=paths["specialist_adapter"], merged=paths["specialist"],
            config_path=config_path, seed=seed, shuffled=False,
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
        _train_grpo(
            tag=f"shuffled_{args.domain}", model=paths["dagger"],
            trajectories=EXP / "runs" / "rl_collection" / args.domain / "trajectories.jsonl.gz",
            anchors=EXP / "data" / f"rl_anchor_{args.domain}.jsonl",
            adapter=paths["shuffled_adapter"], merged=paths["shuffled"],
            config_path=config_path, seed=int(config["seeds"]["shuffled_reward"]), shuffled=True,
        )
    elif stage == "specialist-eval":
        for tag, model in (
            (f"dagger_{args.domain}", paths["dagger"]),
            (f"extra_sft_{args.domain}", paths["extra_sft"]),
            (f"shuffled_{args.domain}", paths["shuffled"]),
            (f"specialist_{args.domain}", paths["specialist"]),
        ):
            _eval(config_path, config, model, tag)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
