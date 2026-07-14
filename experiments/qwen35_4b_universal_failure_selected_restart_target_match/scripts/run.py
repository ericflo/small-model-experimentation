#!/usr/bin/env python3
"""Checkpointed harness for failure-selected counterfactual restart training."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SCRIPTS = EXP / "scripts"


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


def smoke_miner_contract() -> None:
    sys.path.insert(0, str(SCRIPTS))
    import mine_restarts as miner

    source = {
        "task_id": "fixture",
        "selection_skill": "execute",
        "kind": "u_execute",
        "surface": "fixture",
        "level": 1,
        "answer": "ANSWER: ok",
    }
    output_row = {
        "outputs": [{
            "text": "wrong\n</think>\n\nANSWER: no<|im_end|>\n",
            "n_thinking_tokens": 42,
            "n_sampled_tokens": 50,
            "thinking_closed": True,
            "finish_reason": "stop",
            "truncated": False,
        }]
    }
    classified = miner.classify(source, output_row)
    if not classified["eligible"] or classified["reasons"] != ["wrong_answer"]:
        raise SystemExit("counterfactual-restart miner smoke contract failed")
    inventory = []
    for skill in miner.EXPECTED_SKILLS:
        for index in range(miner.QUOTA_PER_SKILL):
            inventory.append({
                **classified,
                "task_id": f"{skill}-{index}",
                "skill": skill,
                "n_thinking_tokens": 42 + index,
            })
    selected, availability = miner.select_inventory(inventory)
    if len(selected) != 52 or set(availability.values()) != {4}:
        raise SystemExit("balanced restart quota smoke contract failed")


def smoke() -> None:
    run([sys.executable, "-B", str(SCRIPTS / "gen_rollout_tasks.py"), "--check"])
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    smoke_miner_contract()
    rollout_receipt = EXP / "runs" / "parent_rollout" / "seed66114.receipt.json"
    if rollout_receipt.exists():
        receipt = json.loads(rollout_receipt.read_text(encoding="utf-8"))
        if (
            receipt.get("experiment_id") != EXP.name
            or receipt.get("seed") != 66114
            or receipt.get("rows") != 624
            or receipt.get("benchmark_data_read") is not False
        ):
            raise SystemExit("published rollout receipt failed smoke authentication")
    print("PASS: model-free design, merged-parent, freshness, and restart-selection contracts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--stage", choices=("collect-parent", "mine-restarts"))
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "collect-parent":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_failure_selected_restart_target_match/data/design_receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "collect_parent_rollouts.py")])
        return 0
    if args.stage == "mine-restarts":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_failure_selected_restart_target_match/runs/parent_rollout/seed66114.receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "mine_restarts.py")])
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
