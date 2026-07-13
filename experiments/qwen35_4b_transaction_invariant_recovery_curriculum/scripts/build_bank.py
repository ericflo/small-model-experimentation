#!/usr/bin/env python3
"""Build matched transaction+recovery and recovery-only transition banks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def group_task_blocks(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["task_id"])].append(row)
    expected = Counter(bank.TRANSITIONS)
    for task_id, block in grouped.items():
        if len(block) != len(bank.TRANSITIONS):
            raise AssertionError(f"{task_id} has {len(block)} rows")
        if Counter(row["transition"] for row in block) != expected:
            raise AssertionError(f"{task_id} is not a complete transition block")
    return grouped


def relabel_block(block: list[dict], arm: str, ordinal: int) -> list[dict]:
    source_id = str(block[0]["task_id"])
    task_id = f"{arm}-replay-{ordinal:03d}-{source_id}"
    result = copy.deepcopy(block)
    for row in result:
        row["task_id"] = task_id
        row["id"] = f"{task_id}-{row['transition']}"
        row["kind"] = f"repo_{arm}"
        row["conditioning"] = arm
        row["think_weight"] = 0.0
    return result


def relabel_transaction(rows: list[dict], arm: str) -> list[dict]:
    result = copy.deepcopy(rows)
    for row in result:
        row["id"] = f"{row['task_id']}-{arm}-{row['transition']}"
        row["kind"] = f"repo_{arm}"
        row["conditioning"] = arm
        row["think_weight"] = 0.0
    return result


def arm_receipt(rows: list[dict]) -> dict:
    by_task = group_task_blocks(rows)
    return {
        "rows": len(rows),
        "tasks": len(by_task),
        "families": dict(Counter(row["family"] for row in rows)),
        "transitions": dict(Counter(row["transition"] for row in rows)),
        "operators": dict(Counter(row["operator"] for row in rows)),
        "all_task_blocks_complete": True,
        "think_weights_zero": all(float(row.get("think_weight", 0.0)) == 0.0 for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bcfg = cfg["bank"]
    source = resolve(bcfg["replay_source"])
    observed_source_hash = sha256_file(source)
    if observed_source_hash != bcfg["replay_source_sha256"]:
        raise SystemExit(f"replay source hash mismatch: {observed_source_hash}")

    train_families = tuple(cfg["families"]["transaction_train"])
    if train_families != repo_tasks.TRANSACTION_TRAIN_FAMILIES:
        raise SystemExit("registered transaction training families do not match source constants")
    tasks_per_family = 1 if args.smoke else int(bcfg["tasks_per_transaction_family"])
    transaction_tasks = repo_tasks.make_tasks(
        train_families, tasks_per_family, int(bcfg["seed"]), str(bcfg["split"])
    )
    built = bank.build_banks(transaction_tasks, trajectories=None)
    if built["uncovered_task_ids"]:
        raise AssertionError(built["uncovered_task_ids"])
    if any(
        not receipt["final_visible_pass"] or not receipt["final_hidden_pass"]
        or receipt["partial_visible_pass"] or receipt["partial_hidden_pass"]
        for receipt in built["replay_receipts"]
    ):
        raise AssertionError("transaction oracle/partial replay invariant failed")

    replay_rows = read_jsonl(source)
    replay_blocks = group_task_blocks(replay_rows)
    rng = random.Random(int(bcfg["seed"]))
    replay_task_ids = sorted(replay_blocks)
    rng.shuffle(replay_task_ids)
    control_tasks = (
        len(transaction_tasks) * 2 if args.smoke else int(bcfg["replay_tasks_in_control"])
    )
    primary_replay_tasks = (
        len(transaction_tasks) if args.smoke else int(bcfg["replay_tasks_in_primary"])
    )
    if control_tasks > len(replay_task_ids) or primary_replay_tasks > control_tasks:
        raise SystemExit("replay source does not contain enough complete task blocks")
    selected = replay_task_ids[:control_tasks]

    transaction_rows = relabel_transaction(
        built["recovery_action_rows"], "transaction_replay"
    )
    primary_replay = [
        row
        for ordinal, task_id in enumerate(selected[:primary_replay_tasks])
        for row in relabel_block(replay_blocks[task_id], "transaction_replay", ordinal)
    ]
    control_replay = [
        row
        for ordinal, task_id in enumerate(selected)
        for row in relabel_block(replay_blocks[task_id], "replay_only", ordinal)
    ]
    primary = transaction_rows + primary_replay
    control = control_replay
    expected_tasks = len(transaction_tasks) + primary_replay_tasks
    if len(group_task_blocks(primary)) != expected_tasks:
        raise AssertionError("primary task count mismatch")
    if len(group_task_blocks(control)) != control_tasks:
        raise AssertionError("control task count mismatch")
    if not args.smoke and expected_tasks != int(bcfg["tasks_per_arm"]):
        raise AssertionError("registered primary task count mismatch")

    start = resolve(cfg["model"]["start_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(
        start, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    target = float(bcfg["target_operator_action_mass"])
    primary_balance = bank.calibrate_transition_loss_mass(
        primary, tokenizer, target_operator_action_mass=target,
        plan_mass_fraction=0.0, max_length=int(cfg["training"]["max_length"]),
    )
    control_balance = bank.calibrate_transition_loss_mass(
        control, tokenizer, target_operator_action_mass=target,
        plan_mass_fraction=0.0, max_length=int(cfg["training"]["max_length"]),
    )
    if primary_balance["weighted_action_mass_by_operator"] != control_balance["weighted_action_mass_by_operator"]:
        raise AssertionError("arms have unequal operator action mass")
    bank.assert_firewall_clean(primary, transaction_tasks)
    bank.assert_firewall_clean(control, transaction_tasks)

    root = resolve(cfg["artifacts"]["root"])
    suffix = "_smoke" if args.smoke else ""
    bank_dir = root / ("bank" + suffix)
    primary_path = bank_dir / "transaction_replay.jsonl"
    control_path = bank_dir / "replay_only.jsonl"
    write_jsonl(primary_path, primary)
    write_jsonl(control_path, control)
    receipt = {
        "schema_version": 1,
        "smoke": bool(args.smoke),
        "seed": int(bcfg["seed"]),
        "replay_source": str(source.resolve()),
        "replay_source_sha256": observed_source_hash,
        "selected_replay_task_ids": selected,
        "transaction_public_manifests": [task.public_manifest() for task in transaction_tasks],
        "transaction_replay_receipts": built["replay_receipts"],
        "arms": {
            "transaction_replay": {
                **arm_receipt(primary),
                "transaction_tasks": len(transaction_tasks),
                "replay_tasks": primary_replay_tasks,
                "balance": primary_balance,
                "path": str(primary_path.resolve()),
            },
            "replay_only": {
                **arm_receipt(control),
                "transaction_tasks": 0,
                "replay_tasks": control_tasks,
                "balance": control_balance,
                "path": str(control_path.resolve()),
            },
        },
        "matched_weighted_operator_action_mass": True,
        "firewall_clean": True,
    }
    bank.assert_firewall_clean(receipt, transaction_tasks)
    receipt_path = bank_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "receipt": str(receipt_path),
        "primary": arm_receipt(primary),
        "control": arm_receipt(control),
        "operator_mass": primary_balance["weighted_action_mass_by_operator"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
