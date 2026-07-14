#!/usr/bin/env python3
"""Fail-closed staged harness for the close-weighted commit-seam experiment."""

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
REPLAY_REFRESH_ADAPTER = (
    ROOT / "large_artifacts/qwen35_4b_universal_replay_anchor/adapters/replay_refresh"
)
REPLAY_REFRESH_MERGED = (
    ROOT / "large_artifacts/qwen35_4b_universal_replay_anchor/merged/replay_refresh"
)
PARENT_ADAPTER = (
    ROOT / "large_artifacts/qwen35_4b_universal_mid_density_token_match/adapters/designed160"
)
BASE_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized"
BLEND_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/blend"
TOKEN_RECEIPT = DATA / "stream_token_receipt.json"
LOCAL_RECEIPT = EXP / "runs/local/seed88006.json"
PROMOTION_RECEIPT = EXP / "runs/local/seed88006_promotion.json"
PARENT_LABEL = "designed160_parent"
ARMS = ("replay_repeat", "standard_xi", "close_xi")
CANDIDATES = ("standard_xi", "close_xi")
TRAIN_FILES = {
    "replay_repeat": DATA / "replay_repeat.jsonl",
    "standard_xi": DATA / "targeted_standard.jsonl",
    "close_xi": DATA / "targeted_standard.jsonl",
}
EXPECTED_HASHES = {
    "replay_repeat.jsonl": "6ec82e2989eda5f37f51ba0b13e2c8326c8107110c4b193f3ac65621779e81d4",
    "targeted_standard.jsonl": "12fc613bb31a46bcea9acd49b26467656704aa3b3418dab8d920adf057d14f00",
}
EXPECTED_FORWARD_TOKENS = 286814
EXPECTED_ROWS = 320


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
    run([str(PYTHON), str(SCRIPTS / "materialize_streams.py"), "--check"])
    receipt = json.loads(TOKEN_RECEIPT.read_text(encoding="utf-8"))
    files = {Path(row["path"]).name: row for row in receipt.get("files", [])}
    if (
        receipt.get("model_id") != "Qwen/Qwen3.5-4B"
        or receipt.get("model_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or receipt.get("max_length") != 4096
        or receipt.get("skipped_rows") != 0
        or receipt.get("rows_per_arm") != EXPECTED_ROWS
        or receipt.get("forward_tokens_per_arm") != EXPECTED_FORWARD_TOKENS
        or receipt.get("standard_close_byte_identity") is not True
        or set(files) != set(EXPECTED_HASHES)
    ):
        raise SystemExit("token receipt failed the frozen smoke contract")
    for name, checksum in EXPECTED_HASHES.items():
        row = files[name]
        path = DATA / name
        if (
            row.get("rows") != EXPECTED_ROWS
            or row.get("sha256") != checksum
            or sha256_file(path) != checksum
            or row.get("total_forward_tokens_per_epoch") != EXPECTED_FORWARD_TOKENS
        ):
            raise SystemExit(f"token receipt disagrees with {path}")
    target = files["targeted_standard.jsonl"]
    if target.get("kinds", {}).get("u_execute") != 40 or target.get("kinds", {}).get("u_induct") != 40:
        raise SystemExit("targeted stream lost its frozen failure-relevant block")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    run([
        str(PYTHON), "-B", "-m", "unittest", "discover",
        "-s", str(EXP / "tests"), "-q",
    ])
    print(
        "smoke passed: 320 rows/arm, 286814 exact forward tokens, zero skips; "
        "standard/close data bytes identical"
    )


def train_arm(name: str) -> None:
    if name not in ARMS:
        raise ValueError(name)
    command = [
        str(PYTHON), str(SCRIPTS / "train_trial.py"),
        "--name", name,
        "--train", str(TRAIN_FILES[name]),
        "--token-receipt", str(TOKEN_RECEIPT),
        "--out", str(ADAPTERS / name),
        "--warm-start", str(PARENT_ADAPTER),
        "--epochs", "1.0",
        "--lr", "1e-5",
        "--rank", "32",
        "--alpha", "64",
        "--batch-size", "1",
        "--grad-accum", "8",
        "--max-length", "4096",
        "--w-think", "0.2",
        "--w-close", "0.2",
        "--seed", "44",
    ]
    if name == "close_xi":
        command.extend((
            "--target-close-kind", "u_execute",
            "--target-close-kind", "u_induct",
            "--target-w-close", "1.0",
        ))
    run(command)


def local_eval() -> None:
    run([
        str(PYTHON), str(SCRIPTS / "eval_curriculum.py"),
        "--adapter", f"{PARENT_LABEL}={PARENT_ADAPTER}",
        *[value for name in ARMS for value in ("--adapter", f"{name}={ADAPTERS / name}")],
        "--seed", "88006",
        "--max-new-tokens", "1024",
        "--batch-size", "4",
        "--out", str(LOCAL_RECEIPT),
    ])
    gates = {}
    for name in ARMS:
        path = EXP / "runs/local" / f"seed88006_{name}_gate.json"
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
        "seed": 88006,
        "local_receipt": str(LOCAL_RECEIPT),
        "local_receipt_sha256": sha256_file(LOCAL_RECEIPT),
        "eligible": [name for name in CANDIDATES if gates[name]["passes"]],
        "controls": ["replay_repeat", PARENT_LABEL],
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
        raise SystemExit("no treatment arm passed the frozen local gate; benchmark remains sealed")


def eligible_arms() -> list[str]:
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("seed") != 88006
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
        or not set(payload.get("eligible", [])).issubset(CANDIDATES)
    ):
        raise SystemExit("local promotion receipt failed authentication")
    return list(payload.get("eligible", []))


def merge() -> None:
    eligible = eligible_arms()
    if not eligible:
        raise SystemExit("no locally eligible candidate to merge")
    sources = {
        PARENT_LABEL: PARENT_ADAPTER,
        "replay_repeat": ADAPTERS / "replay_repeat",
        **{name: ADAPTERS / name for name in eligible},
    }
    for name, adapter in sources.items():
        run([
            str(PYTHON), str(SCRIPTS / "merge_trial.py"),
            "--name", name,
            "--adapter", str(adapter),
            "--out", str(MERGED / name),
        ])


def benchmark() -> None:
    eligible = eligible_arms()
    if not eligible:
        raise SystemExit("no locally eligible candidate; benchmark remains sealed")
    models = [
        f"base={BASE_MERGED}",
        f"blend={BLEND_MERGED}",
        f"replay_refresh={REPLAY_REFRESH_MERGED}",
        f"{PARENT_LABEL}={MERGED / PARENT_LABEL}",
        f"replay_repeat={MERGED / 'replay_repeat'}",
        *[f"{name}={MERGED / name}" for name in eligible],
    ]
    run([
        str(PYTHON), str(SCRIPTS / "run_benchmark.py"),
        "--name", "pilot1",
        "--tier", "quick",
        "--think-budget", "1024",
        "--seed", "78136",
        *[value for model in models for value in ("--model", model)],
        *[value for name in eligible for value in ("--candidate", name)],
        "--strong-control", "blend",
        "--anchor", "replay_refresh",
        "--parent", PARENT_LABEL,
        "--replay-control", "replay_repeat",
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only non-GPU integrity gates")
    parser.add_argument(
        "--stage",
        choices=(
            "train-control", "train-standard", "train-close", "train",
            "local", "merge", "benchmark", "all",
        ),
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
    if args.stage in {"train-standard", "train", "all"}:
        train_arm("standard_xi")
    if args.stage in {"train-close", "train", "all"}:
        train_arm("close_xi")
    if args.stage in {"local", "all"}:
        local_eval()
    if args.stage in {"merge", "all"}:
        merge()
    if args.stage in {"benchmark", "all"}:
        benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
