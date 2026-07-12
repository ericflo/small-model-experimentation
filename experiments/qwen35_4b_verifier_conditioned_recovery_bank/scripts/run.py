#!/usr/bin/env python3
"""Resumable staged orchestrator plus CPU and GPU integration smokes."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml
from transformers import AutoTokenizer

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def cpu_smoke() -> dict:
    cfg = load_config()
    families = tuple(cfg["families"]["train"] + cfg["families"]["transfer"])
    tasks = repo_tasks.make_tasks(families, 1, seed=84501, split="smoke")
    for task in tasks:
        env = repo_tasks.RepoEnv(task)
        try:
            if env.visible_pass() or env.hidden_pass():
                raise AssertionError(f"{task.task_id} is not initially broken")
            # Running tests creates __pycache__; public tools must remain text-only.
            env.run_visible()
            env.search("def ")
            if "__pycache__" in env.tree():
                raise AssertionError("runtime cache leaked into public tree")
        finally:
            env.close()
        env = repo_tasks.RepoEnv(task)
        try:
            env.apply_partial()
            if env.visible_pass() or env.hidden_pass():
                raise AssertionError(f"{task.task_id} partial repair is not unresolved")
        finally:
            env.close()
        env = repo_tasks.RepoEnv(task)
        try:
            env.apply_oracle()
            if not env.visible_pass() or not env.hidden_pass():
                raise AssertionError(f"{task.task_id} oracle failed")
        finally:
            env.close()

    for scenario in ("rejected_patch", "failed_test"):
        episode = repo_agent.Episode(tasks[0], 0, scenario=scenario, scaffold=True)
        try:
            if "RECOVERY RULE:" not in episode.messages[-1]["content"]:
                raise AssertionError(f"scaffold missing for {scenario}")
        finally:
            episode.env.close()

    built = bank.build_banks(tasks, trajectories=None)
    bank.assert_firewall_clean(built, tasks)
    tokenizer = AutoTokenizer.from_pretrained(
        resolve(cfg["model"]["start_checkpoint"]), local_files_only=True,
        trust_remote_code=True, use_fast=True,
    )
    rows_by_arm = {
        "happy_action": built["happy_action_rows"],
        "recovery_action": built["recovery_action_rows"],
        "recovery_reason": built["recovery_reason_rows"],
    }
    probe = bank.calibrate_transition_loss_mass(
        rows_by_arm["recovery_action"], tokenizer,
        target_operator_action_mass=1.0, plan_mass_fraction=0.0,
        max_length=int(cfg["training"]["max_length"]),
    )
    target = sum(probe["raw_action_tokens_by_transition"].values())
    balances = {}
    for arm, rows in rows_by_arm.items():
        balances[arm] = bank.calibrate_transition_loss_mass(
            rows, tokenizer, target_operator_action_mass=target,
            plan_mass_fraction=(
                float(cfg["bank"]["reason_plan_mass_fraction"])
                if arm == "recovery_reason" else 0.0
            ),
            max_length=int(cfg["training"]["max_length"]),
        )
        masses = balances[arm]["weighted_action_mass_by_operator"].values()
        if max(masses) - min(masses) > 1e-7:
            raise AssertionError(f"operator imbalance in {arm}")
        for transitions in balances[arm]["operator_transition_strata"].values():
            values = [
                balances[arm]["weighted_action_mass_by_transition"][transition]
                for transition in transitions
            ]
            if max(values) - min(values) > 1e-7:
                raise AssertionError(f"conditional imbalance in {arm}")

    blocks = [
        repo_tasks.make_tasks(
            tuple(cfg["families"][spec["families"]]), 1, int(spec["seed"]), name
        )
        for name, spec in cfg["evaluation"]["blocks"].items()
    ]
    id_sets = [{task.task_id for task in tasks}, *({task.task_id for task in block} for block in blocks)]
    if any(id_sets[i] & id_sets[j] for i in range(len(id_sets)) for j in range(i + 1, len(id_sets))):
        raise AssertionError("smoke/evaluation split IDs overlap")
    return {
        "schema_version": 1,
        "families": len(families),
        "tasks_selftested": len(tasks),
        "rows_per_arm": {arm: len(rows) for arm, rows in rows_by_arm.items()},
        "transitions": list(bank.TRANSITIONS),
        "operator_action_mass": {
            arm: item["weighted_action_mass_by_operator"] for arm, item in balances.items()
        },
        "reason_plan_mass_fraction": cfg["bank"]["reason_plan_mass_fraction"],
        "partial_states_fail_visible_and_hidden": True,
        "firewall_clean": True,
        "runtime_cache_hidden": True,
        "split_ids_disjoint": True,
        "status": "PASS",
    }


def run_command(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> int:
    print("[run] " + " ".join(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode not in allowed_returncodes:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return completed.returncode


def run_if_missing(
    output: Path, command: list[str], allowed_returncodes: tuple[int, ...] = (0,)
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
        output = EXP / "reports" / "smoke_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(receipt, indent=2))
        return 0

    cfg = load_config()
    artifact_root = resolve(cfg["artifacts"]["root"])
    start_model = resolve(cfg["model"]["start_checkpoint"])
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")
    if args.gpu_smoke:
        smoke = artifact_root / "smoke"
        bank_dir = smoke / "bank"
        run_if_missing(bank_dir / "receipt.json", [
            py, str(EXP / "scripts" / "build_bank.py"), "--oracle-smoke",
            "--tasks-per-family", "1", "--output-dir", str(bank_dir),
        ])
        adapter = smoke / "adapter"
        run_if_missing(adapter / "training_receipt.json", [
            py, str(EXP / "scripts" / "train.py"), "--arm", "recovery_reason",
            "--base-model", str(start_model), "--train", str(bank_dir / "recovery_reason.jsonl"),
            "--out", str(adapter), "--smoke",
        ])
        merged = smoke / "merged"
        run_if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"), "--base-model", str(start_model),
            "--adapter", str(adapter), "--out", str(merged),
        ])
        return run_if_missing(smoke / "eval.json", [
            vpy, str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm", "gpu_smoke", "--model", str(merged), "--block", "calibration",
            "--scenario-set", "recovery", "--mode", "deep", "--tasks-per-family", "1",
            "--output", str(smoke / "eval.json"),
        ])

    harvest = artifact_root / "harvest" / "trajectories.json"
    bank_dir = artifact_root / "bank"
    run_if_missing(harvest, [vpy, str(EXP / "scripts" / "harvest.py")])
    run_if_missing(bank_dir / "receipt.json", [py, str(EXP / "scripts" / "build_bank.py")])
    bank_receipt = json.loads((bank_dir / "receipt.json").read_text())
    if not all(bank_receipt["gates"].values()):
        print("[run] persisted bank receipt fails pre-training gates", flush=True)
        return 4

    def train_and_merge(arm: str) -> Path:
        adapter = artifact_root / "adapters" / arm
        train_cfg = cfg["training"]
        run_if_missing(adapter / "training_receipt.json", [
            py, str(EXP / "scripts" / "train.py"), "--arm", arm,
            "--base-model", str(start_model), "--train", str(bank_dir / f"{arm}.jsonl"),
            "--out", str(adapter), "--epochs", str(train_cfg["epochs"]),
            "--lr", str(train_cfg["learning_rate"]), "--rank", str(train_cfg["rank"]),
            "--alpha", str(train_cfg["alpha"]), "--batch-size", str(train_cfg["batch_size"]),
            "--grad-accum", str(train_cfg["gradient_accumulation_steps"]),
            "--loss-chunk-positions", str(train_cfg["loss_chunk_positions"]),
            "--max-length", str(train_cfg["max_length"]), "--seed", str(train_cfg["seed"]),
        ])
        merged = artifact_root / "merged" / arm
        run_if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"), "--base-model", str(start_model),
            "--adapter", str(adapter), "--out", str(merged),
        ])
        return merged

    models = {arm: train_and_merge(arm) for arm in cfg["training"]["arms"]}

    def evaluate(
        arm: str, model: Path, block: str, scenario_set: str,
        mode: str = "deep", scaffold: bool = False,
    ) -> Path:
        suffix = "_scaffold" if scaffold else ""
        output = artifact_root / "eval" / f"{block}_{scenario_set}_{arm}_{mode}{suffix}.json"
        command = [
            vpy, str(EXP / "scripts" / "eval_repo_agent.py"), "--arm", arm,
            "--model", str(model), "--block", block, "--scenario-set", scenario_set,
            "--mode", mode, "--output", str(output),
        ]
        if scaffold:
            command.append("--scaffold")
        run_if_missing(output, command)
        return output

    calibration = {
        "base": evaluate("base", start_model, "calibration", "recovery"),
        "happy_action": evaluate("happy_action", models["happy_action"], "calibration", "recovery"),
        "recovery_action": evaluate("recovery_action", models["recovery_action"], "calibration", "recovery"),
        "recovery_reason": evaluate("recovery_reason", models["recovery_reason"], "calibration", "recovery"),
    }
    selection = EXP / "analysis" / "candidate_selection.json"
    selection_code = run_command([
        py, str(EXP / "scripts" / "select_candidate.py"),
        "--base", str(calibration["base"]), "--happy", str(calibration["happy_action"]),
        "--action", str(calibration["recovery_action"]),
        "--reason", str(calibration["recovery_reason"]), "--out", str(selection),
    ], allowed_returncodes=(0, 4))
    if selection_code == 4:
        print("[run] calibration mechanism gate failed; stopping before transfer and Menagerie", flush=True)
        return 4
    selected = json.loads(selection.read_text())["selected_arm"]
    candidate_model = models[selected]

    locality_contexts = EXP / "data" / "locality_contexts.json"
    run_if_missing(locality_contexts, [
        py, str(EXP / "scripts" / "build_locality_contexts.py"),
        "--seed", str(cfg["locality"]["seed"]), "--out", str(locality_contexts),
    ])
    locality = artifact_root / "eval" / f"locality_{selected}.json"
    if locality.exists():
        print(f"[resume] {locality} exists", flush=True)
        locality_code = 0 if json.loads(locality.read_text())["gate"]["passed"] else 4
    else:
        locality_code = run_command([
            py, str(EXP / "scripts" / "audit_locality.py"),
            "--before-model", str(start_model), "--after-model", str(candidate_model),
            "--contexts", str(locality_contexts), "--out", str(locality),
            "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
            "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
        ], allowed_returncodes=(0, 4))
    if locality_code == 4:
        print("[run] locality gate failed; stopping before transfer and Menagerie", flush=True)
        return 4

    uncertainty = artifact_root / "eval" / f"transition_uncertainty_{selected}.json"
    run_if_missing(uncertainty, [
        py, str(EXP / "scripts" / "audit_transition_uncertainty.py"),
        "--before-model", str(start_model), "--after-model", str(candidate_model),
        "--bank", str(bank_dir / "recovery_action.jsonl"),
        "--rows-per-transition", "6", "--out", str(uncertainty),
    ])

    def run_transfer_block(block: str) -> int:
        candidate_recovery = evaluate(selected, candidate_model, block, "recovery")
        base_recovery = evaluate("base", start_model, block, "recovery")
        happy_recovery = evaluate("happy_action", models["happy_action"], block, "recovery")
        sample_recovery = evaluate("base", start_model, block, "recovery", mode="sample_more")
        scaffold_recovery = evaluate(
            "base", start_model, block, "recovery", scaffold=True
        )
        candidate_normal = evaluate(selected, candidate_model, block, "normal")
        base_normal = evaluate("base", start_model, block, "normal")
        output = EXP / "analysis" / f"{block}_gate.json"
        return run_command([
            py, str(EXP / "scripts" / "analyze_primary.py"), "--block", block,
            "--candidate-recovery", str(candidate_recovery),
            "--base-recovery", str(base_recovery), "--happy-recovery", str(happy_recovery),
            "--sample-more-recovery", str(sample_recovery),
            "--scaffold-recovery", str(scaffold_recovery),
            "--candidate-normal", str(candidate_normal), "--base-normal", str(base_normal),
            "--locality", str(locality), "--out", str(output),
        ], allowed_returncodes=(0, 4))

    if run_transfer_block("transfer_dev") == 4:
        print("[run] transfer-dev gate failed; stopping before confirmation and Menagerie", flush=True)
        return 4
    if run_transfer_block("transfer_confirm") == 4:
        print("[run] confirmation gate failed; Menagerie remains sealed", flush=True)
        return 4
    print(
        "[run] all white-box gates passed; assign fresh paired Menagerie seeds through "
        "the benchmark CLI before the final evaluation",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
