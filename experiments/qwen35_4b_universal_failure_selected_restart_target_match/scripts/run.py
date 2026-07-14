#!/usr/bin/env python3
"""Checkpointed harness for failure-selected counterfactual restart training."""

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
ROLLOUT_DIR = EXP / "runs" / "parent_rollout"
ROLLOUT_HASHES = {
    "seed66114.jsonl": "4bf15134b02ac9f4f4a1b424d9dfc10f280fead4e340a7f77284d0b920c1099f",
    "seed66114.meta.json": "b43b3a0dbf3469a38386bb64007b772622a35a8bf911cd652732855adad1206d",
    "seed66114.log": "668e9b70c04d5714428a546ee28587d74af87ac823f807f4f9200265e4f369ff",
    "seed66114.receipt.json": "1d35c63a70d53d8803666cb8c30f4d0efffd884c7f6ab04adceaf8b05442b381",
}
SELECTION_HASHES = {
    "counterfactual_restart_source.jsonl": "022b1ea4cfe2bb50fca7f5fdc472a0bf228a5d7a7adb637b221b8efe434d951f",
    "failure_inventory_seed66114.json": "c19d3de700c1ccab931298816c259b587ae0476d5105e3a29b75d93007966240",
    "restart_selection_receipt.json": "567d6b020b9120c82bd19fdc7992dc49b927df2b604978ab3d6ae64e2c05b662",
    "selection_summary.json": "2e8a21927fd1e4bb9ad4ca5e26cbf39c6ca97982542ae9d2bb496a3ef6e28ddf",
}
EXPOSURE_HASHES = {
    "sft_blend.jsonl": "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2",
    "predecessor_stream_manifest.json": "abf8b5055e68c0fb2bb6e32a29f7be3b3677a0dd179e77397647777a2aa0966f",
    "source_token_lengths.json": "ac9b9c8a3c9bfc66699781c96792ea72c37701b11719772764e74b35dba10bd6",
    "stream_manifest.json": "7ba55045e72371e3675ba67bcf0bd72f6a0bf645c3ad7d0e92f7282e59d91de1",
    "replay_control.jsonl": "7a8d45666000cbb6bffabf6faab8f9d61006bf3a80275a631238a23cd03b5078",
    "counterfactual_restart_candidate.jsonl": "28deb20e6bfca81f760549b071d0d0df39bfa561c4d09fde0580d81699413190",
    "stream_token_receipt.json": "52a761ef8fd37f3eac88abf8f090013f571a47511daeb26820ca030201b1c170",
}
PARENT_ADAPTER = (
    ROOT
    / "large_artifacts"
    / "qwen35_4b_universal_on_policy_prefix_repair_token_match"
    / "adapters"
    / "replay_after_close"
)
ADAPTER_ROOT = ROOT / "large_artifacts" / EXP.name / "adapters"


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
    rollout_receipt = ROLLOUT_DIR / "seed66114.receipt.json"
    if rollout_receipt.exists():
        for name, expected in ROLLOUT_HASHES.items():
            path = ROLLOUT_DIR / name
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published parent rollout artifact changed: {path}")
        receipt = json.loads(rollout_receipt.read_text(encoding="utf-8"))
        metadata = json.loads((ROLLOUT_DIR / "seed66114.meta.json").read_text(encoding="utf-8"))
        rows = (ROLLOUT_DIR / "seed66114.jsonl").read_text(encoding="utf-8").splitlines()
        if (
            receipt.get("experiment_id") != EXP.name
            or receipt.get("seed") != 66114
            or receipt.get("rows") != 624
            or receipt.get("sampled_tokens") != 304013
            or receipt.get("rollouts_sha256") != ROLLOUT_HASHES["seed66114.jsonl"]
            or receipt.get("metadata_sha256") != ROLLOUT_HASHES["seed66114.meta.json"]
            or receipt.get("log_sha256") != ROLLOUT_HASHES["seed66114.log"]
            or receipt.get("benchmark_data_read") is not False
            or receipt.get("recovery") != {"generation_rerun": False, "used": False}
            or len(rows) != 624
            or metadata.get("counts", {}).get("requests") != 624
            or metadata.get("counts", {}).get("completions") != 624
            or metadata.get("counts", {}).get("sampled_tokens") != 304013
        ):
            raise SystemExit("published rollout receipt failed smoke authentication")
    selection_receipt = EXP / "data" / "restart_selection_receipt.json"
    if selection_receipt.exists():
        for name, expected in SELECTION_HASHES.items():
            path = EXP / "data" / name
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published restart-selection artifact changed: {path}")
        receipt = json.loads(selection_receipt.read_text(encoding="utf-8"))
        summary = json.loads((EXP / "data" / "selection_summary.json").read_text(encoding="utf-8"))
        restarts = (EXP / "data" / "counterfactual_restart_source.jsonl").read_text(encoding="utf-8").splitlines()
        if (
            receipt.get("outcome") != "PASS_RESTART_QUOTAS"
            or receipt.get("selected_rows") != 52
            or receipt.get("training_authorized") is not False
            or len(restarts) != 52
            or summary.get("selected", {}).get("hard_failure_rows") != 40
            or summary.get("selected", {}).get("budget_only_rows") != 12
            or summary.get("selected", {}).get("parent_prefix_rows") != 0
            or summary.get("benchmark_data_read") is not False
            or summary.get("aggregate_seed_open") is not False
        ):
            raise SystemExit("published restart selection failed smoke authentication")
    token_receipt_path = EXP / "data" / "stream_token_receipt.json"
    if token_receipt_path.exists():
        for name, expected in EXPOSURE_HASHES.items():
            path = EXP / "data" / name
            if not path.is_file() or sha256_file(path) != expected:
                raise SystemExit(f"published exposure artifact changed: {path}")
        run([sys.executable, "-B", str(SCRIPTS / "measure_source_tokens.py"), "--check"])
        run([sys.executable, "-B", str(SCRIPTS / "materialize_streams.py"), "--check"])
        run([sys.executable, "-B", str(SCRIPTS / "validate_streams.py"), "--check"])
        receipt = json.loads(token_receipt_path.read_text(encoding="utf-8"))
        if (
            receipt.get("rows_per_arm") != 320
            or receipt.get("forward_tokens_per_arm") != 297731
            or receipt.get("nonzero_target_tokens_per_arm") != 126796
            or receipt.get("absolute_loss_mass_x5_per_arm") != 138164
            or receipt.get("shared_position_aligned_rows") != 200
            or receipt.get("skipped_rows") != 0
            or any(receipt.get("candidate_minus_control_spans", {}).get(axis) != 0 for axis in receipt.get("match_axes", []))
            or receipt.get("training_authorized") is not False
        ):
            raise SystemExit("published exact-exposure receipt failed smoke authentication")
        review = (EXP / "reports" / "compute_review.md").read_text(encoding="utf-8")
        if "**Verdict:** `PASS_CONTROL_TRAINING`." not in review:
            raise SystemExit("second adversarial review has not authorized control training")
        training_dir = EXP / "runs" / "training"
        if (training_dir / "replay_control.json").exists():
            sys.path.insert(0, str(SCRIPTS))
            from train_trial import validate_published_arm

            validate_published_arm("replay_control", require_committed=False)
            if (training_dir / "counterfactual_restart_candidate.json").exists():
                validate_published_arm(
                    "counterfactual_restart_candidate", require_committed=False
                )
    print(
        "PASS: model-free design, merged-parent, freshness, restart selection, "
        "exact exposure, and control-training authorization contracts"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke", action="store_true")
    group.add_argument(
        "--stage",
        choices=("collect-parent", "mine-restarts", "train-control", "train-candidate"),
    )
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
    if args.stage in {"train-control", "train-candidate"}:
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_failure_selected_restart_target_match/data/stream_token_receipt.json"
        )
        require_pushed_checkpoint(
            "experiments/qwen35_4b_universal_failure_selected_restart_target_match/reports/compute_review.md"
        )
        name = (
            "replay_control"
            if args.stage == "train-control"
            else "counterfactual_restart_candidate"
        )
        if args.stage == "train-candidate":
            require_pushed_checkpoint(
                "experiments/qwen35_4b_universal_failure_selected_restart_target_match/runs/training/replay_control.json"
            )
        run([
            sys.executable,
            "-B",
            str(SCRIPTS / "train_trial.py"),
            "--name",
            name,
            "--train",
            str(EXP / "data" / f"{name}.jsonl"),
            "--token-receipt",
            str(EXP / "data" / "stream_token_receipt.json"),
            "--out",
            str(ADAPTER_ROOT / name),
            "--warm-start",
            str(PARENT_ADAPTER),
            "--epochs",
            "1",
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
            "48",
        ])
        return 0
    raise AssertionError(args.stage)


if __name__ == "__main__":
    raise SystemExit(main())
