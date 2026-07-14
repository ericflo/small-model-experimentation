#!/usr/bin/env python3
"""Fail-closed staged harness for the search-scaffold curriculum experiment."""

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
PARENT_ADAPTER = (
    ROOT / "large_artifacts/qwen35_4b_universal_close_weight_token_match/adapters/close_xi"
)
REPLAY_REFRESH_MERGED = (
    ROOT / "large_artifacts/qwen35_4b_universal_replay_anchor/merged/replay_refresh"
)
BASE_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/base_reserialized"
BLEND_MERGED = ROOT / "large_artifacts/qwen35_4b_universal_curriculum/merged/blend"
TOKEN_RECEIPT = DATA / "stream_token_receipt.json"
LOCAL_RECEIPT = EXP / "runs/local/seed88007.json"
PROMOTION_RECEIPT = EXP / "runs/local/seed88007_promotion.json"
PARENT_LABEL = "close_xi_parent"
CONTROL = "replay_after_close"
CANDIDATE = "scaffold_after_close"
ARMS = (CONTROL, CANDIDATE)
TRAIN_FILES = {
    CONTROL: DATA / "replay_after_close.jsonl",
    CANDIDATE: DATA / "scaffold_after_close.jsonl",
}
EXPECTED_HASHES = {
    "replay_after_close.jsonl": "c157fb135f0934375de3c36d3258b4d2621a09f9831f4eb9f1a8f5bb959c355d",
    "scaffold_after_close.jsonl": "79a8d7c933a220b809447f144f07c2352f89f462198b07b64b30275cf8790b90",
}
EXPECTED_RECEIPT_SHA256 = "eeb12b95c915e9a32755e73db94b5eb69a5aec53788461e1a50aa9b72f1e4a0f"
EXPECTED_FORWARD_TOKENS = 286814
EXPECTED_ROWS = 320
EXPECTED_STAGES = {
    "u_scaffold_apply", "u_scaffold_fit", "u_scaffold_reject",
    "u_scaffold_execute", "u_scaffold_search",
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
    run([str(PYTHON), "-B", str(SCRIPTS / "gen_search_scaffold.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "materialize_streams.py"), "--check"])
    receipt = json.loads(TOKEN_RECEIPT.read_text(encoding="utf-8"))
    files = {Path(row["path"]).name: row for row in receipt.get("files", [])}
    if (
        sha256_file(TOKEN_RECEIPT) != EXPECTED_RECEIPT_SHA256
        or receipt.get("model_id") != "Qwen/Qwen3.5-4B"
        or receipt.get("model_revision") != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
        or receipt.get("max_length") != 4096
        or receipt.get("skipped_rows") != 0
        or receipt.get("rows_per_arm") != EXPECTED_ROWS
        or receipt.get("forward_tokens_per_arm") != EXPECTED_FORWARD_TOKENS
        or receipt.get("position_aligned_streams") is not True
        or set(files) != set(EXPECTED_HASHES)
    ):
        raise SystemExit("token receipt failed the frozen smoke contract")
    for name, checksum in EXPECTED_HASHES.items():
        row = files[name]
        path = DATA / name
        if (
            row.get("rows") != EXPECTED_ROWS
            or row.get("encoded_rows") != EXPECTED_ROWS
            or row.get("skipped_rows") != 0
            or row.get("sha256") != checksum
            or sha256_file(path) != checksum
            or row.get("total_forward_tokens_per_epoch") != EXPECTED_FORWARD_TOKENS
        ):
            raise SystemExit(f"token receipt disagrees with {path}")
    target_kinds = files["scaffold_after_close.jsonl"].get("kinds", {})
    if any(target_kinds.get(kind) != 16 for kind in EXPECTED_STAGES):
        raise SystemExit("candidate stream lost its frozen five-stage scaffold block")
    replay_lines = (DATA / "replay_after_close.jsonl").read_text(encoding="utf-8").splitlines()
    candidate_lines = (DATA / "scaffold_after_close.jsonl").read_text(encoding="utf-8").splitlines()
    if sum(left == right for left, right in zip(replay_lines, candidate_lines, strict=True)) != 200:
        raise SystemExit("streams lost their 200 position-aligned replay slots")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    run([
        str(PYTHON), "-B", "-m", "unittest", "discover",
        "-s", str(EXP / "tests"), "-q",
    ])
    print(
        "smoke passed: 320 rows/arm, 286814 exact forward tokens, zero skips, "
        "200 position-aligned replay slots"
    )


def train_arm(name: str) -> None:
    if name not in ARMS:
        raise ValueError(name)
    run([
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
        "--seed", "45",
    ])


def local_eval() -> None:
    run([
        str(PYTHON), str(SCRIPTS / "eval_curriculum.py"),
        "--adapter", f"{PARENT_LABEL}={PARENT_ADAPTER}",
        *[value for name in ARMS for value in ("--adapter", f"{name}={ADAPTERS / name}")],
        "--seed", "88007",
        "--max-new-tokens", "1024",
        "--batch-size", "4",
        "--out", str(LOCAL_RECEIPT),
    ])
    gates = {}
    for name in (PARENT_LABEL, *ARMS):
        path = EXP / "runs/local" / f"seed88007_{name}_gate.json"
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
        "seed": 88007,
        "local_receipt": str(LOCAL_RECEIPT),
        "local_receipt_sha256": sha256_file(LOCAL_RECEIPT),
        "eligible": [CANDIDATE] if gates[CANDIDATE]["passes"] else [],
        "candidate": CANDIDATE,
        "controls": [CONTROL, PARENT_LABEL],
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
        raise SystemExit("search-scaffold arm failed the frozen local gate; benchmark remains sealed")


def eligible_candidate() -> str:
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("seed") != 88007
        or payload.get("candidate") != CANDIDATE
        or payload.get("eligible") != [CANDIDATE]
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
        or payload.get("gates", {}).get(CANDIDATE, {}).get("passes") is not True
    ):
        raise SystemExit("local promotion receipt failed authentication")
    return CANDIDATE


def merge() -> None:
    eligible_candidate()
    sources = {
        PARENT_LABEL: PARENT_ADAPTER,
        CONTROL: ADAPTERS / CONTROL,
        CANDIDATE: ADAPTERS / CANDIDATE,
    }
    for name, adapter in sources.items():
        run([
            str(PYTHON), str(SCRIPTS / "merge_trial.py"),
            "--name", name,
            "--adapter", str(adapter),
            "--out", str(MERGED / name),
        ])


def benchmark() -> None:
    eligible_candidate()
    models = [
        f"base={BASE_MERGED}",
        f"blend={BLEND_MERGED}",
        f"replay_refresh={REPLAY_REFRESH_MERGED}",
        f"{PARENT_LABEL}={MERGED / PARENT_LABEL}",
        f"{CONTROL}={MERGED / CONTROL}",
        f"{CANDIDATE}={MERGED / CANDIDATE}",
    ]
    run([
        str(PYTHON), str(SCRIPTS / "run_benchmark.py"),
        "--name", "pilot1",
        "--tier", "quick",
        "--think-budget", "1024",
        "--seed", "78137",
        *[value for model in models for value in ("--model", model)],
        "--candidate", CANDIDATE,
        "--strong-control", "blend",
        "--anchor", "replay_refresh",
        "--parent", PARENT_LABEL,
        "--replay-control", CONTROL,
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="run only non-GPU integrity gates")
    parser.add_argument(
        "--stage",
        choices=("train-control", "train-candidate", "train", "local", "merge", "benchmark", "all"),
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
        train_arm(CONTROL)
    if args.stage in {"train-candidate", "train", "all"}:
        train_arm(CANDIDATE)
    if args.stage in {"local", "all"}:
        local_eval()
    if args.stage in {"merge", "all"}:
        merge()
    if args.stage in {"benchmark", "all"}:
        benchmark()
    return 0


if __name__ == "__main__":
    sys.exit(main())
