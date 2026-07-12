#!/usr/bin/env python3
"""Build replay-verified compact and action-only banks from a harvest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_agent  # noqa: E402
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


def oracle_trajectories(tasks: list[repo_tasks.RepoTask]) -> list[dict]:
    """Host-only synthetic trajectories for smoke testing; never a full-run input."""
    trajectories = []
    for task in tasks:
        env = repo_tasks.RepoEnv(task)
        steps = []
        try:
            for turn, patch in enumerate(task.oracle_patches, 1):
                action = {"tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new}
                before = env.workspace_digest()
                observation, _done, _success = repo_agent.execute_action(env, action)
                after = env.workspace_digest()
                steps.append({
                    "turn": turn,
                    "action": action,
                    "observation": observation,
                    "before_digest": before,
                    "after_digest": after,
                })
            trajectories.append({
                "task_id": task.task_id,
                "trajectory": 0,
                "workspace_success": env.hidden_pass(),
                "sampled_tokens": 1,
                "turns": len(steps),
                "steps": steps,
            })
        finally:
            env.close()
    return trajectories


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--artifact-root", type=Path, default=None)
    parser.add_argument("--harvest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--oracle-smoke", action="store_true")
    parser.add_argument("--tasks-per-family", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    hcfg = cfg["harvest"]
    artifact_root = args.artifact_root or resolve(cfg["artifacts"]["root"])
    n_tasks = args.tasks_per_family or int(hcfg["tasks_per_family"])
    tasks = repo_tasks.make_tasks(
        tuple(cfg["families"]["train"]), n_tasks, int(hcfg["seed"]), str(hcfg["split"])
    )
    if args.oracle_smoke:
        trajectories = oracle_trajectories(tasks)
        source = "HOST_ORACLE_SMOKE_ONLY"
    else:
        harvest_path = args.harvest or artifact_root / "harvest" / "trajectories.json"
        harvest = json.loads(harvest_path.read_text())
        trajectories = harvest["trajectories"]
        source = str(harvest_path.resolve())
        expected_ids = {task.task_id for task in tasks}
        observed_ids = {row["task_id"] for row in trajectories}
        if observed_ids != expected_ids:
            raise SystemExit(
                f"harvest/task mismatch: missing={len(expected_ids-observed_ids)} "
                f"extra={len(observed_ids-expected_ids)}"
            )

    built = bank.build_banks(tasks, trajectories)
    multiplier = float(cfg["bank"]["repo_loss_multiplier"])
    for key in ("compact_rows", "action_only_rows"):
        for row in built[key]:
            row["row_weight"] *= multiplier
    if built["operator_balance"]:
        built["operator_balance"]["repo_loss_multiplier"] = multiplier
        built["operator_balance"]["scaled_loss_mass"] = {
            key: value * multiplier
            for key, value in built["operator_balance"]["loss_mass"].items()
        }

    bank.assert_firewall_clean(built, tasks)
    output_dir = args.output_dir or artifact_root / ("bank_smoke" if args.oracle_smoke else "bank")
    compact_path = output_dir / "compact.jsonl"
    action_path = output_dir / "action_only.jsonl"
    write_jsonl(compact_path, built.pop("compact_rows"))
    write_jsonl(action_path, built.pop("action_only_rows"))
    replay_rate = (
        sum(row["visible_pass"] and row["hidden_pass"] and row["submitted_success"]
            for row in built["replay_receipts"]) / max(len(built["replay_receipts"]), 1)
    )
    compact_rows = sum(1 for _ in compact_path.open(encoding="utf-8"))
    gate = {
        "minimum_rows": compact_rows >= int(hcfg["minimum_compact_rows"]) or args.oracle_smoke,
        "replay_pass_rate": replay_rate >= float(cfg["bank"]["require_replay_pass_rate"]),
        "operator_presence": set(built["operator_balance"]["counts"]) == set(cfg["bank"]["required_operators"]),
        "operator_mass_equal": len({round(value, 8) for value in built["operator_balance"]["scaled_loss_mass"].values()}) == 1,
    }
    receipt = {
        "schema_version": 1,
        "source": source,
        "oracle_smoke": args.oracle_smoke,
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "tasks": len(tasks),
        "covered_tasks": len(built["replay_receipts"]),
        "uncovered_tasks": len(built["uncovered_task_ids"]),
        "compact_rows": compact_rows,
        "operator_counts": Counter(row["family"] for row in built["replay_receipts"]),
        "replay_pass_rate": replay_rate,
        "compact_path": str(compact_path.resolve()),
        "compact_sha256": file_digest(compact_path),
        "action_only_path": str(action_path.resolve()),
        "action_only_sha256": file_digest(action_path),
        "gates": gate,
        **built,
    }
    bank.assert_firewall_clean(receipt, tasks)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "receipt": str(receipt_path),
        "tasks": len(tasks),
        "covered_tasks": receipt["covered_tasks"],
        "compact_rows": compact_rows,
        "replay_pass_rate": replay_rate,
        "gates": gate,
    }, indent=2))
    if not all(gate.values()):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
