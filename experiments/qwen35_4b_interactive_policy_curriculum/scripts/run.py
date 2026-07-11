#!/usr/bin/env python3
"""Staged orchestration for the interactive-policy curriculum."""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
PY = REPO / ".venv" / "bin" / "python"
VLLM_PY = REPO / ".venv-vllm" / "bin" / "python"
CONFIG = EXP / "configs" / "curriculum.yaml"


def _run(command: list[str], *, training: bool = False) -> None:
    print("[stage] " + " ".join(command), flush=True)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    if training:
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=REPO, env=env, check=True)


def _config() -> dict:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise SystemExit("one-model invariant failed")
    return config


def _paths(config: dict) -> dict[str, Path]:
    root = REPO / config["model"]["artifacts_root"]
    return {
        "root": root,
        "incumbent_adapter": root / "adapters" / "incumbent_blend",
        "incumbent": root / "merged" / "incumbent_blend",
        "dagger_adapter": root / "adapters" / "dagger",
        "dagger": root / "merged" / "dagger",
        "rl_adapter": root / "adapters" / "rl1",
        "rl": root / "merged" / "rl1",
        "matched_sft_adapter": root / "adapters" / "matched_sft",
        "matched_sft": root / "merged" / "matched_sft",
        "shuffled_adapter": root / "adapters" / "shuffled_rl",
        "shuffled": root / "merged" / "shuffled_rl",
    }


def _record_checkpoint(tag: str, adapter: Path, merged: Path) -> None:
    receipt = {
        "tag": tag,
        "adapter": str(adapter.resolve()),
        "merged": str(merged.resolve()),
        "training_receipt": json.loads((adapter / "training_receipt.json").read_text()),
        "merge_receipt": json.loads((merged / "merge_receipt.json").read_text()),
    }
    path = EXP / "runs" / "checkpoint_receipts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = []
    if path.exists():
        existing = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    existing = [row for row in existing if row.get("tag") != tag] + [receipt]
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in existing),
        encoding="utf-8",
    )


def smoke(config: dict) -> None:
    # Parse every experiment Python source without writing pycache.
    for path in sorted(list((EXP / "src").rglob("*.py")) + list((EXP / "scripts").glob("*.py"))):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    _run([str(PY), str(EXP / "tests" / "test_curriculum.py")])
    _run([str(PY), str(EXP / "tests" / "test_vllm_runner.py")])
    _run([str(PY), str(EXP / "scripts" / "selftest_gym.py")])
    train = set(config["split"]["train_families"])
    transfer = set(config["split"]["transfer_families"])
    if train & transfer:
        raise SystemExit(f"train/transfer family overlap: {sorted(train & transfer)}")
    print(
        json.dumps(
            {
                "smoke": "passed",
                "model": config["model"]["id"],
                "train_families": sorted(train),
                "transfer_families": sorted(transfer),
            },
            indent=2,
            sort_keys=True,
        )
    )


def train_and_merge(
    *,
    tag: str,
    model: str | Path,
    train_file: Path,
    adapter: Path,
    merged: Path,
    epochs: float,
    learning_rate: float,
    rank: int,
    alpha: int,
    batch_size: int,
    grad_accum: int,
    max_length: int,
    think_weight: float,
    seed: int,
    max_steps: int = -1,
) -> None:
    command = [
        str(PY),
        str(EXP / "scripts" / "train_dagger.py"),
        "--model", str(model),
        "--train", str(train_file),
        "--out", str(adapter),
        "--epochs", str(epochs),
        "--max-steps", str(max_steps),
        "--lr", str(learning_rate),
        "--rank", str(rank),
        "--alpha", str(alpha),
        "--batch-size", str(batch_size),
        "--grad-accum", str(grad_accum),
        "--max-length", str(max_length),
        "--w-think", str(think_weight),
        "--seed", str(seed),
    ]
    _run(command, training=True)
    _run(
        [
            str(PY),
            str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(model),
            "--adapter", str(adapter),
            "--out", str(merged),
        ],
        training=True,
    )
    _record_checkpoint(tag, adapter, merged)


def proxy_eval(paths: dict[str, Path]) -> None:
    for tag in ("incumbent", "dagger", "rl", "matched_sft", "shuffled"):
        model = paths.get(tag)
        if model is None or not (model / "config.json").exists():
            continue
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "eval_proxy.py"),
                "--config", str(CONFIG),
                "--model", str(model),
                "--tag", tag,
            ]
        )


def require_gate(name: str) -> dict:
    path = EXP / "analysis" / f"{name}_gate.json"
    if not path.exists():
        raise SystemExit(f"missing {name} gate receipt: {path}")
    payload = json.loads(path.read_text())
    if not payload.get("gate", {}).get("passed"):
        raise SystemExit(f"{name} gate did not pass; refusing downstream stage")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=(
            "incumbent",
            "dagger-collect",
            "dagger-train",
            "proxy-eval",
            "dagger-gate",
            "rl-collect",
            "rl-train",
            "controls",
            "rl-gate",
        ),
    )
    args = parser.parse_args()
    config = _config()
    paths = _paths(config)
    if args.smoke:
        smoke(config)
        return 0
    if args.stage is None:
        parser.error("pass --smoke or one --stage")

    if args.stage == "incumbent":
        cfg = config["incumbent"]
        train_and_merge(
            tag="incumbent_blend",
            model=config["model"]["id"],
            train_file=REPO / config["model"]["incumbent_data"],
            adapter=paths["incumbent_adapter"],
            merged=paths["incumbent"],
            epochs=cfg["epochs"],
            learning_rate=cfg["learning_rate"],
            rank=cfg["rank"], alpha=cfg["alpha"],
            batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
            max_length=cfg["max_length"], think_weight=cfg["think_loss_weight"],
            seed=config["seeds"]["training"],
        )
    elif args.stage == "dagger-collect":
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "collect_dagger.py"),
                "--config", str(CONFIG), "--model", str(paths["incumbent"]),
            ]
        )
    elif args.stage == "dagger-train":
        cfg = config["dagger_train"]
        train_and_merge(
            tag="dagger",
            model=paths["incumbent"],
            train_file=EXP / "data" / "dagger_train.jsonl",
            adapter=paths["dagger_adapter"], merged=paths["dagger"],
            epochs=cfg["epochs"], learning_rate=cfg["learning_rate"],
            rank=cfg["rank"], alpha=cfg["alpha"],
            batch_size=cfg["batch_size"], grad_accum=cfg["grad_accum"],
            max_length=cfg["max_length"], think_weight=cfg["think_loss_weight"],
            seed=config["seeds"]["training"],
        )
    elif args.stage == "proxy-eval":
        proxy_eval(paths)
    elif args.stage == "dagger-gate":
        _run(
            [
                str(PY), str(EXP / "scripts" / "analyze_proxy.py"),
                "--config", str(CONFIG), "--phase", "dagger",
                "--incumbent", str(EXP / "runs" / "proxy_eval" / "incumbent"),
                "--candidate", str(EXP / "runs" / "proxy_eval" / "dagger"),
            ]
        )
    elif args.stage == "rl-collect":
        require_gate("dagger")
        _run(
            [
                str(VLLM_PY), str(EXP / "scripts" / "collect_rl.py"),
                "--config", str(CONFIG), "--model", str(paths["dagger"]),
            ]
        )
    elif args.stage == "rl-train":
        require_gate("dagger")
        _run(
            [
                str(PY), str(EXP / "scripts" / "train_sequence_grpo.py"),
                "--config", str(CONFIG),
                "--model", str(paths["dagger"]),
                "--out", str(paths["rl_adapter"]),
            ],
            training=True,
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_adapter.py"),
                "--base-model", str(paths["dagger"]),
                "--adapter", str(paths["rl_adapter"]),
                "--out", str(paths["rl"]),
            ],
            training=True,
        )
        _record_checkpoint("rl1", paths["rl_adapter"], paths["rl"])
    elif args.stage == "controls":
        require_gate("dagger")
        # Strong static control: 1.5x optimizer steps from the identical
        # DAgger checkpoint on expert labels at the same RL-visited states.
        cfg = config["dagger_train"]
        train_and_merge(
            tag="matched_sft",
            model=paths["dagger"], train_file=EXP / "data" / "rl_anchor.jsonl",
            adapter=paths["matched_sft_adapter"], merged=paths["matched_sft"],
            epochs=1.0, learning_rate=config["rl_train"]["learning_rate"],
            rank=config["rl_train"]["rank"], alpha=config["rl_train"]["alpha"],
            batch_size=1, grad_accum=config["rl_train"]["grad_accum"],
            max_length=config["rl_train"]["max_length"],
            think_weight=config["rl_train"]["think_loss_weight"],
            seed=config["seeds"]["training"],
            max_steps=config["controls"]["matched_sft_steps"],
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "train_sequence_grpo.py"),
                "--config", str(CONFIG), "--model", str(paths["dagger"]),
                "--out", str(paths["shuffled_adapter"]),
                "--max-steps", str(config["controls"]["shuffled_reward_steps"]),
                "--shuffle-advantages",
            ],
            training=True,
        )
        _run(
            [
                str(PY), str(EXP / "scripts" / "merge_adapter.py"),
                "--base-model", str(paths["dagger"]),
                "--adapter", str(paths["shuffled_adapter"]),
                "--out", str(paths["shuffled"]),
            ],
            training=True,
        )
        _record_checkpoint("shuffled_rl", paths["shuffled_adapter"], paths["shuffled"])
    elif args.stage == "rl-gate":
        _run(
            [
                str(PY), str(EXP / "scripts" / "analyze_proxy.py"),
                "--config", str(CONFIG), "--phase", "rl",
                "--incumbent", str(EXP / "runs" / "proxy_eval" / "incumbent"),
                "--candidate", str(EXP / "runs" / "proxy_eval" / "rl"),
                "--dagger", str(EXP / "runs" / "proxy_eval" / "dagger"),
                "--matched-sft", str(EXP / "runs" / "proxy_eval" / "matched_sft"),
                "--shuffled", str(EXP / "runs" / "proxy_eval" / "shuffled"),
            ]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

