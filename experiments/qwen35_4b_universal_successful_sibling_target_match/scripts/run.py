#!/usr/bin/env python3
"""Checkpointed harness for policy-supported successful-sibling distillation."""

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
    greedy = {
        "outputs": [{
            "text": "wrong\n</think>\n\nANSWER: no<|im_end|>\n",
            "n_sampled_tokens": 40,
            "n_thinking_tokens": 32,
            "thinking_closed": True,
            "finish_reason": "stop",
            "truncated": False,
        }]
    }
    graded = policy.grade_greedy(source, greedy)
    if not graded["hard_failure"] or graded["reasons"] != ["wrong_answer"]:
        raise SystemExit("greedy-failure policy smoke contract failed")
    outputs = []
    for index in range(policy.SAMPLES_PER_FAILURE):
        correct = index in {3, 7}
        think = "short correct path" if index == 3 else "longer sampled path"
        outputs.append({
            "sample_index": index,
            "text": f"{think}\n</think>\n\nANSWER: {'ok' if correct else 'no'}<|im_end|>\n",
            "n_sampled_tokens": 20 if index == 3 else 40 + index,
            "n_thinking_tokens": 12 if index == 3 else 30 + index,
            "thinking_closed": True,
            "finish_reason": "stop",
            "truncated": False,
        })
    best, grades = policy.choose_best_sibling(source, {"outputs": outputs})
    if best is None or best["sample_index"] != 3 or sum(item["qualified"] for item in grades) != 2:
        raise SystemExit("successful-sibling qualification smoke contract failed")
    candidates = []
    for skill in policy.EXPECTED_SKILLS:
        for index in range(policy.QUOTA_PER_SKILL):
            candidates.append({
                "task_id": f"{skill}-{index}",
                "skill": skill,
                "kind": f"u_{skill}",
                "best": {**best, "n_sampled_tokens": 20 + index},
            })
    selected, availability = policy.select_balanced(candidates)
    if len(selected) != 52 or set(availability.values()) != {4}:
        raise SystemExit("balanced successful-sibling quota smoke contract failed")


def validate_collection_receipt(kind: str) -> None:
    if kind == "greedy":
        directory = EXP / "runs" / "greedy_collection"
        stem = "seed66115"
        stage = "authenticated_replay_parent_greedy_failure_collection"
        expected_seed = 66115
        expected_n = 1
    else:
        directory = EXP / "runs" / "sibling_collection"
        stem = "seed66116"
        stage = "authenticated_replay_parent_successful_sibling_collection"
        expected_seed = 66116
        expected_n = 16
    receipt_path = directory / f"{stem}.receipt.json"
    if not receipt_path.exists():
        return
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    raw = directory / f"{stem}.jsonl"
    meta = directory / f"{stem}.meta.json"
    log = directory / f"{stem}.log"
    if (
        receipt.get("experiment_id") != EXP.name
        or receipt.get("stage") != stage
        or receipt.get("seed") != expected_seed
        or receipt.get("sampling", {}).get("n") != expected_n
        or receipt.get("rollouts_sha256") != sha256_file(raw)
        or receipt.get("metadata_sha256") != sha256_file(meta)
        or receipt.get("log_sha256") != sha256_file(log)
        or receipt.get("benchmark_data_read") is not False
    ):
        raise SystemExit(f"published {kind} collection failed smoke authentication")


def smoke() -> None:
    run([sys.executable, "-B", str(SCRIPTS / "gen_collection_tasks.py"), "--check"])
    run([sys.executable, "-B", str(SCRIPTS / "check_design.py"), "--check"])
    smoke_policy()
    validate_collection_receipt("greedy")
    failure_receipt = EXP / "data" / "greedy_failure_selection_receipt.json"
    if failure_receipt.exists():
        run([sys.executable, "-B", str(SCRIPTS / "prepare_sibling_sampling.py"), "--check"])
    validate_collection_receipt("sibling")
    selection_receipt = EXP / "data" / "successful_sibling_selection_receipt.json"
    if selection_receipt.exists():
        run([sys.executable, "-B", str(SCRIPTS / "mine_successful_siblings.py"), "--check"])
    print("PASS: frozen two-event successful-sibling collection and selection contracts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument(
        "--stage",
        choices=("collect-greedy", "prepare-siblings", "collect-siblings", "select-siblings"),
    )
    args = parser.parse_args()
    if args.smoke:
        smoke()
        return 0
    if args.stage == "collect-greedy":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_successful_sibling_target_match/data/design_receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "collect_greedy.py")])
        return 0
    if args.stage == "prepare-siblings":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_successful_sibling_target_match/runs/greedy_collection/seed66115.receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "prepare_sibling_sampling.py")])
        return 0
    if args.stage == "collect-siblings":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_successful_sibling_target_match/data/greedy_failure_selection_receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "collect_siblings.py")])
        return 0
    if args.stage == "select-siblings":
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_successful_sibling_target_match/runs/sibling_collection/seed66116.receipt.json"
        )
        run([sys.executable, "-B", str(SCRIPTS / "mine_successful_siblings.py")])
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
