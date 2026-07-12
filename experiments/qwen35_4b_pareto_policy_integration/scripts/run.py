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
    sha256_file,
    write_json,
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
    *, model: Path, tag: str, scope: str, block_seed: int, config_path: Path
) -> Path:
    out = EXP / "runs" / "policy_eval" / tag
    scores = out / "scores.json"
    if scores.exists():
        payload = json.loads(scores.read_text())
        if (
            payload.get("scope") == scope
            and int(payload.get("block_seed", -1)) == block_seed
            and payload.get("model") == str(model.resolve())
            and payload.get("decode") == "greedy"
        ):
            print(f"[resume] evaluation {tag} already complete", flush=True)
            return scores
        raise SystemExit(f"stale evaluation exists: {out}")
    _run(
        [
            str(VLLM_PY), str(EXP / "scripts" / "eval_policy.py"),
            "--config", str(config_path), "--model", str(model), "--tag", tag,
            "--scope", scope, "--block-seed", str(block_seed), "--out-dir", str(out),
        ]
    )
    return scores


def _qualify(config: dict, config_path: Path) -> None:
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
        ],
        allowed=(0, 4),
    )
    _require_gate(EXP / "analysis" / "specialist_qualification.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("smoke", "model-smoke", "specialists", "qualify", "teacher-audit", "integrate", "controls", "confirm"),
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
    if stage == "specialists":
        _specialists(config)
        return 0
    if stage == "qualify":
        _qualify(config, config_path)
        return 0
    raise SystemExit(f"stage {stage!r} is registered but not implemented yet")


if __name__ == "__main__":
    raise SystemExit(main())
