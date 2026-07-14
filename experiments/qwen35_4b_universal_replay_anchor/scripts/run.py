#!/usr/bin/env python3
"""Fail-closed staged harness for the replay-anchored curriculum experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"
SCRIPTS = EXP / "scripts"
DATA = EXP / "data"
LARGE = ROOT / "large_artifacts" / EXP.name
ADAPTERS = LARGE / "adapters"
MERGED = LARGE / "merged"
BLEND_ADAPTER = ROOT / "large_artifacts/qwen35_4b_gauntlet_frontier/adapters/blend"
BASE_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized"
BLEND_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/blend"
TOKEN_RECEIPT = DATA / "dose_token_receipt.json"
LOCAL_RECEIPT = EXP / "runs/local/seed88003.json"
LOCAL_GATE = EXP / "runs/local/seed88003_gate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def smoke() -> None:
    run([str(PYTHON), str(SCRIPTS / "materialize_doses.py"), "--check"])
    receipt = json.loads(TOKEN_RECEIPT.read_text(encoding="utf-8"))
    expected = {
        "warm_union.jsonl": "f209c677a734308525a0feb04a14c1e1e3773bea750ef3ee50172687e67a61aa",
        "replay_refresh.jsonl": "5d5d7c4b8a4b0a4f270fe8b2ecaebe356c771948d71b0f7bbeead6bfc04308b6",
    }
    files = {Path(row["path"]).name: row for row in receipt.get("files", [])}
    if (
        receipt.get("model_id") != "Qwen/Qwen3.5-4B"
        or receipt.get("model_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or receipt.get("max_length") != 4096
        or receipt.get("skipped_rows") != 0
        or set(files) != set(expected)
    ):
        raise SystemExit("token receipt failed the frozen smoke contract")
    for name, checksum in expected.items():
        row = files[name]
        path = DATA / name
        if row.get("rows") != 1520 or row.get("sha256") != checksum or sha256_file(path) != checksum:
            raise SystemExit(f"token receipt disagrees with {path}")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("smoke passed: exact nested doses, zero skips, pinned Qwen3.5-4B, scripts compile")


def train_arm(name: str) -> None:
    if name not in {"warm_union", "replay_refresh"}:
        raise ValueError(name)
    run([
        str(PYTHON), str(SCRIPTS / "train_trial.py"),
        "--name", name,
        "--train", str(DATA / f"{name}.jsonl"),
        "--token-receipt", str(TOKEN_RECEIPT),
        "--out", str(ADAPTERS / name),
        "--warm-start", str(BLEND_ADAPTER),
        "--epochs", "1.0",
        "--lr", "1e-5",
        "--rank", "32",
        "--alpha", "64",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--max-length", "4096",
        "--w-think", "0.2",
        "--seed", "42",
    ])


def local_eval() -> None:
    run([
        str(PYTHON), str(SCRIPTS / "eval_curriculum.py"),
        "--adapter", f"warm_union={ADAPTERS / 'warm_union'}",
        "--seed", "88003",
        "--max-new-tokens", "1024",
        "--batch-size", "4",
        "--out", str(LOCAL_RECEIPT),
    ])
    run([
        str(PYTHON), str(SCRIPTS / "check_local.py"), str(LOCAL_RECEIPT),
        "--candidate", "warm_union", "--out", str(LOCAL_GATE),
    ])


def merge() -> None:
    for name in ("replay_refresh", "warm_union"):
        run([
            str(PYTHON), str(SCRIPTS / "merge_trial.py"),
            "--name", name,
            "--adapter", str(ADAPTERS / name),
            "--out", str(MERGED / name),
        ])


def benchmark() -> None:
    gate = json.loads(LOCAL_GATE.read_text(encoding="utf-8"))
    if gate.get("passes") is not True:
        raise SystemExit("candidate did not pass the frozen local gate")
    run([
        str(PYTHON), str(SCRIPTS / "run_benchmark.py"),
        "--name", "pilot1",
        "--tier", "quick",
        "--think-budget", "1024",
        "--seed", "78133",
        "--model", f"base={BASE_MERGED}",
        "--model", f"blend={BLEND_MERGED}",
        "--model", f"replay_refresh={MERGED / 'replay_refresh'}",
        "--model", f"warm_union={MERGED / 'warm_union'}",
        "--candidate", "warm_union",
        "--control", "blend",
        "--mechanism-control", "replay_refresh",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only non-GPU integrity gates")
    parser.add_argument(
        "--stage",
        choices=("train-candidate", "train-control", "train", "local", "merge", "benchmark", "all"),
        help="run one expensive preregistered stage",
    )
    args = parser.parse_args()
    if args.smoke:
        if args.stage:
            parser.error("--smoke and --stage are mutually exclusive")
        smoke()
        return 0
    if not args.stage:
        parser.error("choose --smoke or an explicit --stage")
    smoke()
    if args.stage == "all":
        train_arm("warm_union")
        local_eval()
        train_arm("replay_refresh")
        merge()
        benchmark()
        return 0
    if args.stage in {"train-candidate", "train", "all"}:
        train_arm("warm_union")
    if args.stage in {"train-control", "train", "all"}:
        train_arm("replay_refresh")
    if args.stage in {"local", "all"}:
        local_eval()
    if args.stage in {"merge", "all"}:
        merge()
    if args.stage in {"benchmark", "all"}:
        benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
