#!/usr/bin/env python3
"""Build replay-verified, transition-balanced recovery curriculum arms."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def balanced(receipt: dict, tolerance: float = 1e-7) -> bool:
    operator_masses = list(receipt["weighted_action_mass_by_operator"].values())
    if max(operator_masses) - min(operator_masses) > tolerance:
        return False
    strata = receipt["operator_transition_strata"]
    transition_masses = receipt["weighted_action_mass_by_transition"]
    for transitions in strata.values():
        values = [transition_masses[name] for name in transitions]
        if max(values) - min(values) > tolerance:
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--harvest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--oracle-smoke", action="store_true",
        help="use host-only oracle repairs for a CPU/dev fixture check",
    )
    parser.add_argument("--smoke", action="store_true", help="allow a small real-harvest bank")
    parser.add_argument("--tasks-per-family", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hcfg = cfg["harvest"]
    bcfg = cfg["bank"]
    artifact_root = args.artifact_root or resolve(cfg["artifacts"]["root"])
    n_tasks = args.tasks_per_family or int(hcfg["tasks_per_family"])
    tasks = repo_tasks.make_tasks(
        tuple(cfg["families"]["train"]), n_tasks, int(hcfg["seed"]), str(hcfg["split"])
    )
    if args.oracle_smoke:
        trajectories = None
        source = "HOST_ORACLE_SMOKE_ONLY"
        observed_harvest_gate = {"overall_coverage": True, "per_family_coverage": True}
    else:
        harvest_path = args.harvest or artifact_root / "harvest" / "trajectories.json"
        harvest = json.loads(harvest_path.read_text())
        trajectories = harvest["trajectories"]
        source = str(harvest_path.resolve())
        summary = harvest["summary"]
        observed_harvest_gate = {
            "overall_coverage": (
                float(summary["task_coverage"]) >= float(hcfg["minimum_task_coverage"])
            ),
            "per_family_coverage": all(
                float(row["coverage"]) >= float(hcfg["minimum_per_family_coverage"])
                for row in summary["per_family"].values()
            ),
        }
        expected_ids = {task.task_id for task in tasks}
        observed_ids = {row["task_id"] for row in trajectories}
        if observed_ids != expected_ids:
            raise SystemExit(
                f"harvest/task mismatch: missing={len(expected_ids-observed_ids)} "
                f"extra={len(observed_ids-expected_ids)}"
            )

    built = bank.build_banks(tasks, trajectories)
    start_model = resolve(cfg["model"]["start_checkpoint"])
    tokenizer = AutoTokenizer.from_pretrained(
        start_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = int(cfg["training"]["max_length"])
    arm_plan_fractions = {
        "happy_action": 0.0,
        "recovery_action": 0.0,
        "recovery_reason": float(bcfg["reason_plan_mass_fraction"]),
    }
    rows_by_arm = {
        "happy_action": built.pop("happy_action_rows"),
        "recovery_action": built.pop("recovery_action_rows"),
        "recovery_reason": built.pop("recovery_reason_rows"),
    }
    probe_balance = bank.calibrate_transition_loss_mass(
        rows_by_arm["recovery_action"], tokenizer,
        target_operator_action_mass=1.0, plan_mass_fraction=0.0,
        max_length=max_length,
    )
    raw_action_mass = sum(probe_balance["raw_action_tokens_by_transition"].values())
    target_mass = raw_action_mass * float(bcfg["operator_mass_multiplier"])
    balance = {}
    output_dir = args.output_dir or artifact_root / (
        "bank_smoke" if (args.oracle_smoke or args.smoke) else "bank"
    )
    files = {}
    for arm, rows in rows_by_arm.items():
        balance[arm] = bank.calibrate_transition_loss_mass(
            rows,
            tokenizer,
            target_operator_action_mass=target_mass,
            plan_mass_fraction=arm_plan_fractions[arm],
            max_length=max_length,
        )
        path = output_dir / f"{arm}.jsonl"
        write_jsonl(path, rows)
        files[arm] = {
            "path": str(path.resolve()),
            "sha256": file_digest(path),
            "rows": len(rows),
        }

    serialized = {"rows": rows_by_arm, "receipts": built}
    bank.assert_firewall_clean(serialized, tasks)
    replay_rate = (
        sum(
            row["final_visible_pass"] and row["final_hidden_pass"]
            and not row["partial_visible_pass"] and not row["partial_hidden_pass"]
            for row in built["replay_receipts"]
        ) / max(len(built["replay_receipts"]), 1)
    )
    covered = len(built["replay_receipts"])
    expected_rows = covered * len(bank.TRANSITIONS)
    allowed_small = args.oracle_smoke or args.smoke
    gates = {
        **{
            key: value or allowed_small
            for key, value in observed_harvest_gate.items()
        },
        "minimum_covered_tasks": (
            covered >= int(hcfg["minimum_covered_tasks"]) or allowed_small
        ),
        "exact_rows_per_arm": all(item["rows"] == expected_rows for item in files.values()),
        "replay_pass_rate": replay_rate >= float(bcfg["require_replay_pass_rate"]),
        "all_transitions_present": set(built["transition_counts"]) == set(bank.TRANSITIONS),
        "required_operators_present": set(built["operator_counts"]) == set(bank.OPERATORS),
        "exact_conditional_action_balance": all(balanced(item) for item in balance.values()),
        "no_overlength_rows": all(not item["overlength_row_ids"] for item in balance.values()),
        "matched_arm_row_counts": len({item["rows"] for item in files.values()}) == 1,
    }
    receipt = {
        "schema_version": 1,
        "source": source,
        "oracle_smoke": args.oracle_smoke,
        "real_harvest_smoke": args.smoke,
        "start_checkpoint": str(start_model.resolve()),
        "start_config_sha256": file_digest(start_model / "config.json"),
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "tasks": len(tasks),
        "covered_tasks": covered,
        "uncovered_tasks": len(built["uncovered_task_ids"]),
        "covered_by_family": dict(Counter(row["family"] for row in built["replay_receipts"])),
        "replay_pass_rate": replay_rate,
        "raw_action_mass": raw_action_mass,
        "target_operator_action_mass": target_mass,
        "harvest_gate_observed": observed_harvest_gate,
        "gates": gates,
        "files": files,
        "balance": balance,
        **built,
    }
    bank.assert_firewall_clean(receipt, tasks)
    receipt_path = output_dir / "receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path),
        "tasks": len(tasks),
        "covered_tasks": covered,
        "rows_per_arm": {key: item["rows"] for key, item in files.items()},
        "replay_pass_rate": replay_rate,
        "gates": gates,
    }, indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
