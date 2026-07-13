#!/usr/bin/env python3
"""Build matched policy-counterexample and extra-transaction transition banks."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
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
    task_id = f"{arm}-prior-{ordinal:03d}-{source_id}"
    result = copy.deepcopy(block)
    for row in result:
        row["task_id"] = task_id
        row["id"] = f"{task_id}-{row['transition']}"
        row["kind"] = f"repo_{arm}"
        row["conditioning"] = arm
        row["think_weight"] = 0.0
    return result


def relabel_policy(rows: list[dict], arm: str) -> list[dict]:
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
        "think_weights_zero": all(
            float(row.get("think_weight", 0.0)) == 0.0 for row in rows
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    bcfg = cfg["bank"]
    source = resolve(bcfg["prior_primary_source"])
    observed_source_hash = sha256_file(source)
    if observed_source_hash != bcfg["prior_primary_sha256"]:
        raise SystemExit(f"prior primary source hash mismatch: {observed_source_hash}")
    prior_receipt_path = resolve(bcfg["prior_receipt"])
    observed_receipt_hash = sha256_file(prior_receipt_path)
    if observed_receipt_hash != bcfg["prior_receipt_sha256"]:
        raise SystemExit(f"prior receipt hash mismatch: {observed_receipt_hash}")

    train_families = tuple(cfg["families"]["policy_train"])
    if train_families != repo_tasks.POLICY_TRAIN_FAMILIES:
        raise SystemExit("registered policy training families do not match source constants")
    tasks_per_family = 1 if args.smoke else int(bcfg["tasks_per_policy_family"])
    policy_tasks = repo_tasks.make_tasks(
        train_families, tasks_per_family, int(bcfg["seed"]), str(bcfg["split"])
    )
    policy_digests = [repo_tasks.content_digest(task) for task in policy_tasks]
    if len(set(policy_digests)) != len(policy_digests):
        raise AssertionError("policy bank contains duplicate repository content")
    if not args.smoke:
        calibration_cfg = cfg["evaluation"]["blocks"]["policy_calibration"]
        calibration_tasks = repo_tasks.make_tasks(
            train_families,
            int(calibration_cfg["tasks_per_family"]),
            int(calibration_cfg["seed"]),
            "policy_calibration",
        )
        calibration_digests = {
            repo_tasks.content_digest(task) for task in calibration_tasks
        }
        if set(policy_digests) & calibration_digests:
            raise AssertionError("policy bank overlaps trained-family calibration content")
    built = bank.build_banks(policy_tasks, trajectories=None)
    if built["uncovered_task_ids"]:
        raise AssertionError(built["uncovered_task_ids"])
    if any(
        not receipt["final_visible_pass"]
        or not receipt["final_hidden_pass"]
        or receipt["partial_visible_pass"]
        or receipt["partial_hidden_pass"]
        for receipt in built["replay_receipts"]
    ):
        raise AssertionError("policy oracle/partial replay invariant failed")

    prior_rows = read_jsonl(source)
    if any(
        row.get("kind") != "repo_transaction_replay"
        or float(row.get("think_weight", 0.0)) != 0.0
        for row in prior_rows
    ):
        raise SystemExit("frozen prior source is not a pure zero-think transaction bank")
    prior_blocks = group_task_blocks(prior_rows)
    prior_ids = sorted(prior_blocks)
    replay_ids = [task_id for task_id in prior_ids if "-replay-" in task_id]
    transaction_ids = [task_id for task_id in prior_ids if "-replay-" not in task_id]
    if len(replay_ids) != 24 or len(transaction_ids) != 24:
        raise SystemExit(
            f"expected frozen 24 replay + 24 transaction blocks, got "
            f"{len(replay_ids)} + {len(transaction_ids)}"
        )

    if args.smoke:
        injection_ids = replay_ids[:len(policy_tasks)]
        control_ids = transaction_ids[:len(policy_tasks)] + injection_ids
    else:
        injection_ids = replay_ids[:int(bcfg["injected_policy_tasks"])]
        control_ids = prior_ids[:int(bcfg["prior_tasks_in_control"])]
    if len(injection_ids) != len(policy_tasks):
        raise SystemExit("policy and injection task counts differ")

    injection_transition = str(bcfg["injected_transition"])
    if injection_transition != "diagnosis_to_changed_patch":
        raise SystemExit("only the preregistered near-correct revision seam may be injected")
    policy_blocks = group_task_blocks(built["recovery_action_rows"])
    injection_by_prior = dict(zip(injection_ids, sorted(policy_blocks), strict=True))
    primary = []
    injected_rows = []
    for ordinal, task_id in enumerate(control_ids):
        block = relabel_block(prior_blocks[task_id], "policy_counterexample", ordinal)
        if task_id in injection_by_prior:
            policy_task_id = injection_by_prior[task_id]
            policy_row = next(
                copy.deepcopy(row)
                for row in policy_blocks[policy_task_id]
                if row["transition"] == injection_transition
            )
            replacement_id = block[0]["task_id"]
            policy_row["task_id"] = replacement_id
            policy_row["id"] = f"{replacement_id}-{injection_transition}"
            policy_row["kind"] = "repo_policy_counterexample"
            policy_row["conditioning"] = "policy_counterexample"
            policy_row["think_weight"] = 0.0
            block = [
                policy_row if row["transition"] == injection_transition else row
                for row in block
            ]
            injected_rows.append(policy_row)
        primary.extend(block)
    control_rows = [
        row
        for ordinal, task_id in enumerate(control_ids)
        for row in relabel_block(prior_blocks[task_id], "extra_transaction", ordinal)
    ]
    control = control_rows
    expected_tasks = len(control_ids)
    if len(group_task_blocks(primary)) != expected_tasks:
        raise AssertionError("primary task count mismatch")
    if len(group_task_blocks(control)) != len(control_ids):
        raise AssertionError("control task count mismatch")
    if not args.smoke and expected_tasks != int(bcfg["tasks_per_arm"]):
        raise AssertionError("registered primary task count mismatch")

    start = resolve(cfg["model"]["start_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(
        start, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    target = float(bcfg["target_operator_action_mass"])
    primary_balance = bank.calibrate_transition_loss_mass(
        primary,
        tokenizer,
        target_operator_action_mass=target,
        plan_mass_fraction=0.0,
        max_length=int(cfg["training"]["max_length"]),
    )
    control_balance = bank.calibrate_transition_loss_mass(
        control,
        tokenizer,
        target_operator_action_mass=target,
        plan_mass_fraction=0.0,
        max_length=int(cfg["training"]["max_length"]),
    )
    if (
        primary_balance["weighted_action_mass_by_operator"]
        != control_balance["weighted_action_mass_by_operator"]
    ):
        raise AssertionError("arms have unequal operator action mass")
    bank.assert_firewall_clean(primary, policy_tasks)
    bank.assert_firewall_clean(control, policy_tasks)

    root = resolve(cfg["artifacts"]["root"])
    suffix = "_smoke" if args.smoke else ""
    bank_dir = root / ("bank" + suffix)
    primary_path = bank_dir / "policy_counterexample.jsonl"
    control_path = bank_dir / "extra_transaction.jsonl"
    write_jsonl(primary_path, primary)
    write_jsonl(control_path, control)
    receipt = {
        "schema_version": 1,
        "smoke": bool(args.smoke),
        "seed": int(bcfg["seed"]),
        "prior_primary_source": str(source.resolve()),
        "prior_primary_sha256": observed_source_hash,
        "prior_receipt": str(prior_receipt_path.resolve()),
        "prior_receipt_sha256": observed_receipt_hash,
        "prior_source_composition": {
            "transaction_task_ids": transaction_ids,
            "replay_task_ids": replay_ids,
        },
        "injection_prior_task_ids": injection_ids,
        "injection_transition": injection_transition,
        "injected_row_ids": [row["id"] for row in injected_rows],
        "control_prior_task_ids": control_ids,
        "policy_public_manifests": [task.public_manifest() for task in policy_tasks],
        "policy_content_sha256": policy_digests,
        "policy_replay_receipts": built["replay_receipts"],
        "arms": {
            "policy_counterexample": {
                **arm_receipt(primary),
                "policy_tasks": len(policy_tasks),
                "injected_policy_rows": len(injected_rows),
                "unchanged_prior_rows": len(primary) - len(injected_rows),
                "prior_replay_task_blocks": sum(task_id in replay_ids for task_id in control_ids),
                "prior_transaction_task_blocks": sum(
                    task_id in transaction_ids for task_id in control_ids
                ),
                "balance": primary_balance,
                "path": str(primary_path.resolve()),
            },
            "extra_transaction": {
                **arm_receipt(control),
                "policy_tasks": 0,
                "injected_policy_rows": 0,
                "unchanged_prior_rows": len(control),
                "prior_replay_task_blocks": sum(task_id in replay_ids for task_id in control_ids),
                "prior_transaction_task_blocks": sum(
                    task_id in transaction_ids for task_id in control_ids
                ),
                "balance": control_balance,
                "path": str(control_path.resolve()),
            },
        },
        "matched_weighted_operator_action_mass": True,
        "complete_prior_control": len(control_ids) == len(prior_ids),
        "firewall_clean": True,
    }
    bank.assert_firewall_clean(receipt, policy_tasks)
    receipt_path = bank_dir / "receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "primary": arm_receipt(primary),
                "control": arm_receipt(control),
                "operator_mass": primary_balance["weighted_action_mass_by_operator"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
