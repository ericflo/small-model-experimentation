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


def run_command(
    command: list[str], allowed_returncodes: tuple[int, ...] = (0,)
) -> int:
    print("[run] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def run_if_missing(
    output: Path,
    command: list[str],
    allowed_returncodes: tuple[int, ...] = (0,),
) -> int:
    if output.exists():
        print(f"[resume] {output} exists", flush=True)
        return 0
    return run_command(command, allowed_returncodes=allowed_returncodes)


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
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")
    harvest = artifact_root / "harvest" / "trajectories.json"
    bank_dir = artifact_root / "bank"
    run_if_missing(harvest, [vpy, str(EXP / "scripts" / "harvest.py")])
    run_if_missing(bank_dir / "receipt.json", [py, str(EXP / "scripts" / "build_bank.py")])

    c54 = ROOT / cfg["model"]["c54_apex_data"]
    def train_and_merge(arm: str, repo_file: Path | None) -> None:
        adapter = artifact_root / "adapters" / arm
        train_files = [str(c54)] + ([str(repo_file)] if repo_file else [])
        run_if_missing(adapter / "training_receipt.json", [
            py, str(EXP / "scripts" / "train.py"), "--arm", arm,
            "--train", *train_files, "--out", str(adapter),
            "--max-steps", str(cfg["training"]["max_steps"]),
            "--lr", str(cfg["training"]["learning_rate"]),
            "--rank", str(cfg["training"]["rank"]),
            "--alpha", str(cfg["training"]["alpha"]),
            "--batch-size", str(cfg["training"]["batch_size"]),
            "--grad-accum", str(cfg["training"]["gradient_accumulation_steps"]),
            "--max-length", str(cfg["training"]["max_length"]),
            "--loss-chunk-positions", str(cfg["training"]["loss_chunk_positions"]),
            "--seed", str(cfg["training"]["seed"]),
        ])
        merged = artifact_root / "merged" / arm
        run_if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"),
            "--adapter", str(adapter), "--out", str(merged),
        ])

    train_and_merge("apex_replay", None)
    train_and_merge("compact", bank_dir / "compact.jsonl")

    def evaluate(arm: str, block: str, mode: str = "deep") -> Path:
        output = artifact_root / "eval" / f"{block}_{arm}_{mode}.json"
        run_if_missing(output, [
            vpy, str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm", arm, "--model", str(artifact_root / "merged" / arm),
            "--block", block, "--mode", mode, "--output", str(output),
        ])
        return output

    apex_trained = evaluate("apex_replay", "trained_dev")
    compact_trained = evaluate("compact", "trained_dev")
    apex_transfer = evaluate("apex_replay", "transfer_dev")
    compact_transfer = evaluate("compact", "transfer_dev")
    apex_sample = evaluate("apex_replay", "transfer_dev", "sample_more")
    locality = artifact_root / "eval" / "locality.json"
    run_if_missing(locality, [
        py, str(EXP / "scripts" / "audit_locality.py"),
        "--before-model", str(artifact_root / "merged" / "apex_replay"),
        "--after-model", str(artifact_root / "merged" / "compact"),
        "--contexts", str(EXP / "data" / "locality_contexts.json"),
        "--out", str(locality),
        "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
    ], allowed_returncodes=(0, 4))
    primary_gate = EXP / "analysis" / "repo_primary_gate.json"
    primary_returncode = run_command([
        py, str(EXP / "scripts" / "analyze_primary.py"),
        "--candidate-trained", str(compact_trained), "--apex-trained", str(apex_trained),
        "--candidate-transfer", str(compact_transfer), "--apex-transfer", str(apex_transfer),
        "--sample-more-transfer", str(apex_sample),
        "--locality", str(locality), "--out", str(primary_gate),
    ], allowed_returncodes=(0, 4))
    run_command([
        py, str(EXP / "scripts" / "summarize_primary.py"),
        "--artifact-root", str(artifact_root),
        "--primary-gate", str(primary_gate),
        "--out", str(EXP / "reports" / "result_receipt.json"),
    ])
    if primary_returncode == 4:
        print(
            "[run] necessary compact gates failed; stopping before action-only, "
            "confirmation, and Menagerie",
            flush=True,
        )
        return 4

    # The mechanism control is scientifically necessary only for a compact
    # candidate that survives every gate it can fail without that control.
    train_and_merge("action_only", bank_dir / "action_only.jsonl")
    action_transfer = evaluate("action_only", "transfer_dev")
    dev_gate = EXP / "analysis" / "repo_dev_gate.json"
    run_command([
        py, str(EXP / "scripts" / "analyze_repo.py"),
        "--candidate-trained", str(compact_trained), "--apex-trained", str(apex_trained),
        "--candidate-transfer", str(compact_transfer), "--apex-transfer", str(apex_transfer),
        "--action-transfer", str(action_transfer), "--sample-more-transfer", str(apex_sample),
        "--locality", str(locality), "--out", str(dev_gate),
    ])

    apex_confirm = evaluate("apex_replay", "transfer_confirm")
    action_confirm = evaluate("action_only", "transfer_confirm")
    compact_confirm = evaluate("compact", "transfer_confirm")
    apex_sample_confirm = evaluate("apex_replay", "transfer_confirm", "sample_more")
    run_command([
        py, str(EXP / "scripts" / "analyze_repo.py"),
        "--candidate-trained", str(compact_trained), "--apex-trained", str(apex_trained),
        "--candidate-transfer", str(compact_confirm), "--apex-transfer", str(apex_confirm),
        "--action-transfer", str(action_confirm),
        "--sample-more-transfer", str(apex_sample_confirm),
        "--locality", str(locality), "--out", str(EXP / "analysis" / "repo_confirm_gate.json"),
    ])
    print("[run] whitebox gates passed; assign fresh Menagerie seeds before benchmark use", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
