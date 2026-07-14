#!/usr/bin/env python3
"""Checkpointed harness for residual-skill successful-sibling distillation."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def output(command: list[str]) -> str:
    return subprocess.run(
        command, cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def require_pushed_checkpoint(relative_path: str) -> None:
    path = ROOT / relative_path
    status = output(["git", "status", "--short"])
    branch = output(["git", "branch", "--show-current"])
    head = output(["git", "rev-parse", "HEAD"])
    origin = output(["git", "rev-parse", "origin/main"])
    if status or branch != "main" or head != origin:
        raise SystemExit("stage requires a clean pushed main checkpoint")
    run(["git", "cat-file", "-e", f"HEAD:{relative_path}"])
    committed = subprocess.run(
        ["git", "show", f"HEAD:{relative_path}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    if not path.is_file() or path.read_bytes() != committed:
        raise SystemExit(f"stage prerequisite differs from HEAD: {relative_path}")


def smoke_policy() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import sibling_policy as policy

    source = {
        "task_id": "fixture",
        "selection_skill": "execute",
        "kind": "u_execute",
        "surface": "fixture",
        "level": 1,
        "messages": [{"role": "user", "content": "fixture"}],
        "answer": "ANSWER: ok",
    }
    outputs = []
    for index in range(policy.SAMPLES_PER_FAILURE):
        correct = index in {2, 8}
        outputs.append({
            "sample_index": index,
            "text": f"sample path {index}\n</think>\n\nANSWER: {'ok' if correct else 'no'}<|im_end|>\n",
            "n_sampled_tokens": 30 if index == 2 else 50 + index,
            "n_thinking_tokens": 20 if index == 2 else 40 + index,
            "thinking_closed": True,
            "finish_reason": "stop",
            "truncated": False,
        })
    best, grades = policy.choose_best_sibling(source, {"outputs": outputs})
    if best is None or best["sample_index"] != 2 or sum(item["qualified"] for item in grades) != 2:
        raise SystemExit("residual successful-sibling policy smoke failed")
    candidates = []
    for skill in policy.EXPECTED_SKILLS:
        for index in range(policy.QUOTA_PER_SKILL):
            candidates.append({
                "task_id": f"{skill}-{index}",
                "skill": skill,
                "kind": f"u_{skill}",
                "best": {**best, "n_sampled_tokens": 30 + index},
            })
    selected, availability = policy.select_balanced(candidates)
    if len(selected) != 40 or set(availability.values()) != {4}:
        raise SystemExit("residual balanced quota smoke failed")


def validate_collection_receipt() -> None:
    directory = EXP / "runs" / "sibling_collection"
    receipt_path = directory / "seed66117.receipt.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw = directory / "seed66117.jsonl"
    meta = directory / "seed66117.meta.json"
    log = directory / "seed66117.log"
    if (
        receipt.get("experiment_id") != EXP.name
        or receipt.get("stage") != "authenticated_replay_parent_successful_sibling_collection"
        or receipt.get("seed") != 66117
        or receipt.get("rows") != 225
        or receipt.get("samples") != 3600
        or receipt.get("rollouts_sha256") != sha256_file(raw)
        or receipt.get("metadata_sha256") != sha256_file(meta)
        or receipt.get("log_sha256") != sha256_file(log)
        or receipt.get("benchmark_data_read") is not False
    ):
        raise SystemExit("published residual sibling collection failed authentication")


def smoke() -> None:
    run([sys.executable, "-B", str(SCRIPTS / "prepare_residual_input.py"), "--check"])
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    smoke_policy()
    validate_collection_receipt()
    selection_receipt = EXP / "data" / "successful_sibling_selection_receipt.json"
    if selection_receipt.exists():
        outcome = json.loads(selection_receipt.read_text(encoding="utf-8")).get("outcome")
        checked = subprocess.run(
            [sys.executable, "-B", str(SCRIPTS / "mine_successful_siblings.py"), "--check"],
            cwd=ROOT,
            check=False,
        )
        expected = 0 if outcome == "PASS_SUCCESSFUL_SIBLING_QUOTAS" else 2
        if checked.returncode != expected:
            raise SystemExit("residual sibling-selection artifacts failed authentication")
    print("PASS: inherited residual input and frozen successful-sibling contracts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("collect-siblings", "select-siblings"))
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "collect-siblings":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_residual_successful_sibling_target_match/data/design_receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "collect_siblings.py")])
        return 0
    if args.stage == "select-siblings":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_residual_successful_sibling_target_match/runs/sibling_collection/seed66117.receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "mine_successful_siblings.py")])
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
