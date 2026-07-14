#!/usr/bin/env python3
"""Fail-closed staged harness for the low-density token-matched dose ladder."""

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
ANCHOR_EXP = ROOT / "experiments/qwen35_4b_universal_replay_anchor"
ANCHOR_ADAPTER = ROOT / "large_artifacts/qwen35_4b_universal_replay_anchor/adapters/replay_refresh"
ANCHOR_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_replay_anchor/merged/replay_refresh"
BASE_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized"
BLEND_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/blend"
TOKEN_RECEIPT = DATA / "dose_token_receipt.json"
LOCAL_RECEIPT = EXP / "runs/local/seed88004.json"
PROMOTION_RECEIPT = EXP / "runs/local/seed88004_promotion.json"
ARMS = ("replay_repeat", "designed40", "designed80")
EXPECTED_HASHES = {
    "replay_repeat.jsonl": "479c3d4be38ab9d09880ea7afbea3c98ee15965db442fb61c69860aec220224d",
    "designed40.jsonl": "ccfa328147d31877f99eacd19037a20e9381493aea393e4b84c6e3d63f382099",
    "designed80.jsonl": "c6de63a6d3fae0f7f1ac0ea2fcfca33bde77d26a6c908c113ca1e9180ea5585c",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(command: list[str], *, check: bool = True) -> int:
    print("+ " + " ".join(command), flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        check=False,
    )
    if check and completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def smoke() -> None:
    run([str(PYTHON), str(SCRIPTS / "materialize_doses.py"), "--check"])
    receipt = json.loads(TOKEN_RECEIPT.read_text(encoding="utf-8"))
    files = {Path(row["path"]).name: row for row in receipt.get("files", [])}
    if (
        receipt.get("model_id") != "Qwen/Qwen3.5-4B"
        or receipt.get("model_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or receipt.get("max_length") != 4096
        or receipt.get("skipped_rows") != 0
        or set(files) != set(EXPECTED_HASHES)
    ):
        raise SystemExit("token receipt failed the frozen smoke contract")
    exposures = set()
    for name, checksum in EXPECTED_HASHES.items():
        row = files[name]
        path = DATA / name
        if row.get("rows") != 1520 or row.get("sha256") != checksum or sha256_file(path) != checksum:
            raise SystemExit(f"token receipt disagrees with {path}")
        exposures.add(row.get("total_forward_tokens_per_epoch"))
    if exposures != {1429053}:
        raise SystemExit(f"arms are not exactly forward-token matched: {exposures}")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    print("smoke passed: nested 0/40/80 doses, exact token match, zero skips, scripts compile")


def train_arm(name: str) -> None:
    if name not in ARMS:
        raise ValueError(name)
    run([
        str(PYTHON), str(SCRIPTS / "train_trial.py"),
        "--name", name,
        "--train", str(DATA / f"{name}.jsonl"),
        "--token-receipt", str(TOKEN_RECEIPT),
        "--out", str(ADAPTERS / name),
        "--warm-start", str(ANCHOR_ADAPTER),
        "--epochs", "1.0",
        "--lr", "1e-5",
        "--rank", "32",
        "--alpha", "64",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--max-length", "4096",
        "--w-think", "0.2",
        "--seed", "43",
    ])


def local_eval() -> None:
    run([
        str(PYTHON), str(SCRIPTS / "eval_curriculum.py"),
        "--adapter", f"anchor={ANCHOR_ADAPTER}",
        *[value for name in ARMS for value in ("--adapter", f"{name}={ADAPTERS / name}")],
        "--seed", "88004",
        "--max-new-tokens", "1024",
        "--batch-size", "4",
        "--out", str(LOCAL_RECEIPT),
    ])
    gates = {}
    for name in ARMS:
        path = EXP / "runs/local" / f"seed88004_{name}_gate.json"
        returncode = run([
            str(PYTHON), str(SCRIPTS / "check_local.py"), str(LOCAL_RECEIPT),
            "--candidate", name, "--out", str(path),
        ], check=False)
        gate = json.loads(path.read_text(encoding="utf-8"))
        if (returncode == 0) != bool(gate.get("passes")):
            raise SystemExit(f"local gate exit/receipt mismatch for {name}")
        gates[name] = gate
    payload = {
        "schema_version": 1,
        "seed": 88004,
        "local_receipt": str(LOCAL_RECEIPT),
        "local_receipt_sha256": sha256_file(LOCAL_RECEIPT),
        "eligible": [name for name in ARMS if gates[name]["passes"]],
        "gates": gates,
    }
    PROMOTION_RECEIPT.parent.mkdir(parents=True, exist_ok=True)
    if PROMOTION_RECEIPT.exists():
        raise SystemExit("refusing to overwrite local promotion receipt")
    PROMOTION_RECEIPT.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    if not payload["eligible"]:
        raise SystemExit("no arm passed the frozen local gate; benchmark remains sealed")


def eligible_arms() -> list[str]:
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("seed") != 88004
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
    ):
        raise SystemExit("local promotion receipt failed authentication")
    return list(payload.get("eligible", []))


def merge() -> None:
    names = ["replay_repeat"] + [name for name in eligible_arms() if name != "replay_repeat"]
    for name in names:
        run([
            str(PYTHON), str(SCRIPTS / "merge_trial.py"),
            "--name", name,
            "--adapter", str(ADAPTERS / name),
            "--out", str(MERGED / name),
        ])


def benchmark() -> None:
    eligible = eligible_arms()
    models = [
        f"base={BASE_MERGED}",
        f"blend={BLEND_MERGED}",
        f"replay_refresh={ANCHOR_MERGED}",
        f"replay_repeat={MERGED / 'replay_repeat'}",
    ]
    for name in eligible:
        if name != "replay_repeat":
            models.append(f"{name}={MERGED / name}")
    command = [
        str(PYTHON), str(SCRIPTS / "run_benchmark.py"),
        "--name", "pilot1",
        "--tier", "quick",
        "--think-budget", "1024",
        "--seed", "78134",
        *[value for model in models for value in ("--model", model)],
        *[value for name in eligible for value in ("--candidate", name)],
        "--strong-control", "blend",
        "--anchor", "replay_refresh",
        "--replay-control", "replay_repeat",
    ]
    run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only non-GPU integrity gates")
    parser.add_argument(
        "--stage",
        choices=("train-control", "train-d40", "train-d80", "train", "local", "merge", "benchmark", "all"),
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
    if args.stage in {"train-control", "train", "all"}:
        train_arm("replay_repeat")
    if args.stage in {"train-d40", "train", "all"}:
        train_arm("designed40")
    if args.stage in {"train-d80", "train", "all"}:
        train_arm("designed80")
    if args.stage in {"local", "all"}:
        local_eval()
    if args.stage in {"merge", "all"}:
        merge()
    if args.stage in {"benchmark", "all"}:
        benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
