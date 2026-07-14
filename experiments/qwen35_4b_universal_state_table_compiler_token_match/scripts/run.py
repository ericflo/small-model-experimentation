#!/usr/bin/env python3
"""Fail-closed staged harness for the natural-language state-table curriculum."""

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
DESIGN_RECEIPT = DATA / "design_receipt.json"
LOCAL_RECEIPT = EXP / "runs/local/seed88008.json"
PROMOTION_RECEIPT = EXP / "runs/local/seed88008_promotion.json"
PARENT_LABEL = "close_xi_parent"
CONTROL = "replay_after_close"
CANDIDATE = "state_table_after_close"
ARMS = (CONTROL, CANDIDATE)
TRAIN_FILES = {
    CONTROL: DATA / "replay_after_close.jsonl",
    CANDIDATE: DATA / "state_table_after_close.jsonl",
}
EXPECTED_HASHES = {
    "replay_after_close.jsonl": "2727e29a7c18e551ed9defe21b7f4e4009c7e6399ac1b2376deb7a4c609ba2b5",
    "state_table_after_close.jsonl": "8e1b8fdcc349275ad31b2b1af16fa26384cd433de93cbd7a24d0686e07151355",
}
EXPECTED_RECEIPT_SHA256 = "163e40a61d0b3f4dc541f56ea32510bacb8ce64f658e00f47e5867da4a45f0b8"
EXPECTED_DESIGN_RECEIPT_SHA256 = "0bac3340c1995beb1cff1ea9c3563849ff9f024e3ff6c836894f7f22d50ef837"
EXPECTED_FORWARD_TOKENS = 286814
EXPECTED_ROWS = 320
EXPECTED_STAGES = {
    "u_state_table_execute",
    "u_state_table_score",
    "u_state_table_repair",
    "u_state_table_commit",
}
TARGET_KINDS = {"u_execute", "u_induct", "u_probe"}


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


def require_clean_committed_checkpoint(required_paths: tuple[Path, ...] = ()) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise SystemExit("expensive stage requires a clean incrementally committed worktree")
    for path in required_paths:
        relative = path.resolve().relative_to(ROOT.resolve()).as_posix()
        committed = subprocess.run(
            ["git", "show", f"HEAD:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if (
            committed.returncode != 0
            or not path.is_file()
            or committed.stdout != path.read_bytes()
        ):
            raise SystemExit(f"required checkpoint receipt is not committed at HEAD: {relative}")


def smoke() -> None:
    run([str(PYTHON), "-B", str(SCRIPTS / "gen_state_table_curriculum.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "materialize_streams.py"), "--check"])
    run([str(PYTHON), "-B", str(SCRIPTS / "check_design.py"), "--check"])
    if sha256_file(DESIGN_RECEIPT) != EXPECTED_DESIGN_RECEIPT_SHA256:
        raise SystemExit("design receipt changed")
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
    target_kinds = files["state_table_after_close.jsonl"].get("kinds", {})
    if any(target_kinds.get(kind) != 20 for kind in EXPECTED_STAGES):
        raise SystemExit("candidate stream lost its frozen four-stage curriculum block")
    replay_lines = (DATA / "replay_after_close.jsonl").read_text(encoding="utf-8").splitlines()
    candidate_lines = (DATA / "state_table_after_close.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    if sum(left == right for left, right in zip(replay_lines, candidate_lines, strict=True)) != 200:
        raise SystemExit("streams lost their 200 position-aligned replay slots")
    review = (EXP / "reports" / "design_review.md").read_text(encoding="utf-8")
    if "**Verdict:** `PASS_EXPENSIVE_RUN`." not in review:
        raise SystemExit("adversarial design review has not authorized expensive stages")
    for path in sorted(SCRIPTS.glob("*.py")):
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    run(
        [
            str(PYTHON),
            "-B",
            "-m",
            "unittest",
            "discover",
            "-s",
            str(EXP / "tests"),
            "-q",
        ]
    )
    print(
        "smoke passed: 320 rows/arm, 286814 exact forward tokens, zero skips, "
        "200 position-aligned replay slots"
    )


def train_arm(name: str) -> None:
    if name not in ARMS:
        raise ValueError(name)
    run(
        [
            str(PYTHON),
            str(SCRIPTS / "train_trial.py"),
            "--name",
            name,
            "--train",
            str(TRAIN_FILES[name]),
            "--token-receipt",
            str(TOKEN_RECEIPT),
            "--out",
            str(ADAPTERS / name),
            "--warm-start",
            str(PARENT_ADAPTER),
            "--epochs",
            "1.0",
            "--lr",
            "1e-5",
            "--rank",
            "32",
            "--alpha",
            "64",
            "--batch-size",
            "1",
            "--grad-accum",
            "8",
            "--max-length",
            "4096",
            "--w-think",
            "0.2",
            "--w-close",
            "0.2",
            "--seed",
            "46",
        ]
    )


def correct_count(payload: dict, label: str, kinds: set[str] | None = None) -> int:
    return sum(
        bool(row.get("correct"))
        for row in payload.get("rows", [])
        if row.get("adapter") == label and (kinds is None or row.get("kind") in kinds)
    )


def relative_promotion_checks(local: dict) -> dict[str, bool]:
    candidate_correct = correct_count(local, CANDIDATE)
    candidate_target = correct_count(local, CANDIDATE, TARGET_KINDS)
    return {
        "beats_parent_total_correct": candidate_correct > correct_count(local, PARENT_LABEL),
        "beats_replay_total_correct": candidate_correct > correct_count(local, CONTROL),
        "beats_parent_target_correct": candidate_target
        > correct_count(local, PARENT_LABEL, TARGET_KINDS),
        "beats_replay_target_correct": candidate_target
        > correct_count(local, CONTROL, TARGET_KINDS),
    }


def local_eval() -> None:
    run(
        [
            str(PYTHON),
            str(SCRIPTS / "eval_curriculum.py"),
            "--adapter",
            f"{PARENT_LABEL}={PARENT_ADAPTER}",
            *[
                value
                for name in ARMS
                for value in ("--adapter", f"{name}={ADAPTERS / name}")
            ],
            "--seed",
            "88008",
            "--max-new-tokens",
            "1024",
            "--batch-size",
            "4",
            "--out",
            str(LOCAL_RECEIPT),
        ]
    )
    local = json.loads(LOCAL_RECEIPT.read_text(encoding="utf-8"))
    gates = {}
    for name in (PARENT_LABEL, *ARMS):
        path = EXP / "runs/local" / f"seed88008_{name}_gate.json"
        returncode = run(
            [
                str(PYTHON),
                str(SCRIPTS / "check_local.py"),
                str(LOCAL_RECEIPT),
                "--candidate",
                name,
                "--out",
                str(path),
            ],
            check=False,
        )
        gate = json.loads(path.read_text(encoding="utf-8"))
        if (returncode == 0) != bool(gate.get("passes")):
            raise SystemExit(f"local gate exit/receipt mismatch for {name}")
        gates[name] = gate
    relative_checks = relative_promotion_checks(local)
    passes = gates[CANDIDATE]["passes"] and all(relative_checks.values())
    payload = {
        "schema_version": 1,
        "seed": 88008,
        "local_receipt": str(LOCAL_RECEIPT),
        "local_receipt_sha256": sha256_file(LOCAL_RECEIPT),
        "eligible": [CANDIDATE] if passes else [],
        "candidate": CANDIDATE,
        "controls": [CONTROL, PARENT_LABEL],
        "target_kinds": sorted(TARGET_KINDS),
        "correct_counts": {
            name: {
                "total": correct_count(local, name),
                "target": correct_count(local, name, TARGET_KINDS),
            }
            for name in (PARENT_LABEL, *ARMS)
        },
        "relative_checks": relative_checks,
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
        raise SystemExit("state-table arm failed frozen promotion; benchmark remains sealed")


def eligible_candidate() -> str:
    payload = json.loads(PROMOTION_RECEIPT.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 1
        or payload.get("seed") != 88008
        or payload.get("candidate") != CANDIDATE
        or payload.get("eligible") != [CANDIDATE]
        or payload.get("local_receipt_sha256") != sha256_file(LOCAL_RECEIPT)
        or payload.get("gates", {}).get(CANDIDATE, {}).get("passes") is not True
        or not all(payload.get("relative_checks", {}).values())
        or set(payload.get("relative_checks", {}))
        != {
            "beats_parent_total_correct",
            "beats_replay_total_correct",
            "beats_parent_target_correct",
            "beats_replay_target_correct",
        }
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
        run(
            [
                str(PYTHON),
                str(SCRIPTS / "merge_trial.py"),
                "--name",
                name,
                "--adapter",
                str(adapter),
                "--out",
                str(MERGED / name),
            ]
        )


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
    run(
        [
            str(PYTHON),
            str(SCRIPTS / "run_benchmark.py"),
            "--name",
            "pilot1",
            "--tier",
            "quick",
            "--think-budget",
            "1024",
            "--seed",
            "78138",
            *[value for model in models for value in ("--model", model)],
            "--candidate",
            CANDIDATE,
            "--strong-control",
            "blend",
            "--anchor",
            "replay_refresh",
            "--parent",
            PARENT_LABEL,
            "--replay-control",
            CONTROL,
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--smoke", action="store_true", help="run only non-GPU integrity gates")
    parser.add_argument(
        "--stage",
        choices=("train-control", "train-candidate", "local", "merge", "benchmark"),
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
    if args.stage == "train-control":
        require_clean_committed_checkpoint()
        train_arm(CONTROL)
    elif args.stage == "train-candidate":
        require_clean_committed_checkpoint(
            (EXP / "runs" / "training" / f"{CONTROL}.json",)
        )
        train_arm(CANDIDATE)
    elif args.stage == "local":
        require_clean_committed_checkpoint(
            tuple(EXP / "runs" / "training" / f"{name}.json" for name in ARMS)
        )
        local_eval()
    elif args.stage == "merge":
        require_clean_committed_checkpoint((PROMOTION_RECEIPT,))
        merge()
    elif args.stage == "benchmark":
        require_clean_committed_checkpoint(
            tuple(EXP / "runs" / "merges" / f"{name}.json" for name in (PARENT_LABEL, *ARMS))
        )
        benchmark()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
