#!/usr/bin/env python3
"""Staged orchestrator and deterministic CPU/GPU smoke entry point."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402
from build_bank import oracle_trajectories  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def cpu_smoke() -> dict:
    cfg = load_config()
    all_families = tuple(cfg["families"]["train"] + cfg["families"]["transfer"])
    tasks = repo_tasks.make_tasks(all_families, 2, seed=73001, split="smoke")
    for task in tasks:
        env = repo_tasks.RepoEnv(task)
        try:
            if env.visible_pass() or env.hidden_pass():
                raise AssertionError(f"{task.task_id} is not initially broken")
            env.apply_oracle()
            if not env.visible_pass() or not env.hidden_pass():
                raise AssertionError(f"{task.task_id} oracle failed")
        finally:
            env.close()
    built = bank.build_banks(tasks, oracle_trajectories(tasks))
    bank.assert_firewall_clean(built, tasks)
    masses = built["operator_balance"]["loss_mass"]
    if len({round(value, 8) for value in masses.values()}) != 1:
        raise AssertionError(f"unbalanced operator mass: {masses}")
    train_dev = repo_tasks.make_tasks(tuple(cfg["families"]["train"]), 1, 73200, "trained_dev")
    transfer_dev = repo_tasks.make_tasks(tuple(cfg["families"]["transfer"]), 1, 73300, "transfer_dev")
    transfer_confirm = repo_tasks.make_tasks(tuple(cfg["families"]["transfer"]), 1, 73400, "transfer_confirm")
    ids = [{task.task_id for task in block} for block in (tasks, train_dev, transfer_dev, transfer_confirm)]
    if any(ids[i] & ids[j] for i in range(len(ids)) for j in range(i + 1, len(ids))):
        raise AssertionError("split task ids overlap")
    gates = cfg["gates"]
    feasibility = {
        "harvest_coverage": float(cfg["harvest"]["minimum_task_coverage"]) <= 1.0,
        "per_family_coverage": float(cfg["harvest"]["minimum_per_family_coverage"]) <= 1.0,
        "transfer_vs_apex": float(gates["transfer_delta_vs_apex_min"]) <= 1.0,
        "transfer_vs_action_only": float(gates["transfer_delta_vs_action_only_min"]) <= 1.0,
        "transfer_vs_sample_more": float(gates["transfer_delta_vs_apex_sample_more_min"]) <= 1.0,
        "verification_absolute": float(gates["verified_given_success_absolute_min"]) <= 1.0,
        "commit_absolute": float(gates["commit_given_verified_absolute_min"]) <= 1.0,
    }
    if not all(feasibility.values()):
        raise AssertionError(f"unreachable configured gate: {feasibility}")
    return {
        "schema_version": 1,
        "families": len(all_families),
        "tasks_selftested": len(tasks),
        "canonical_rows_replayed": len(built["compact_rows"]),
        "canonical_replay_pass_rate": 1.0,
        "operator_counts": built["operator_balance"]["counts"],
        "operator_loss_mass": masses,
        "firewall_clean": True,
        "split_ids_disjoint": True,
        "gate_feasibility": feasibility,
        "status": "PASS",
    }


def run_command(command: list[str]) -> None:
    print("[run] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.gpu_smoke, args.full)) != 1:
        parser.error("choose exactly one of --smoke, --gpu-smoke, --full")
    if args.smoke:
        receipt = cpu_smoke()
        path = EXP / "reports" / "smoke_receipt.json"
        path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0
    cfg = load_config()
    artifact_root = ROOT / cfg["artifacts"]["root"]
    if args.gpu_smoke:
        run_command([
            str(ROOT / ".venv-vllm" / "bin" / "python"), str(EXP / "scripts" / "harvest.py"),
            "--tasks-per-family", "1", "--trajectories", "1", "--max-turns", "4",
            "--output", str(artifact_root / "smoke" / "trajectories.json"),
        ])
        run_command([
            str(ROOT / ".venv" / "bin" / "python"), str(EXP / "scripts" / "build_bank.py"),
            "--tasks-per-family", "1", "--smoke",
            "--harvest", str(artifact_root / "smoke" / "trajectories.json"),
            "--output-dir", str(artifact_root / "smoke" / "bank"),
        ])
        return 0
    # The full path is intentionally explicit and gate-stopping. Later stages
    # are added only after their predecessor has produced a passing receipt.
    run_command([str(ROOT / ".venv-vllm" / "bin" / "python"), str(EXP / "scripts" / "harvest.py")])
    run_command([str(ROOT / ".venv" / "bin" / "python"), str(EXP / "scripts" / "build_bank.py")])
    print("[run] harvest and bank complete; training continuation requires a passing bank receipt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
