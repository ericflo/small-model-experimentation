#!/usr/bin/env python3
"""Resumable gated validation-policy counterexample curriculum pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import bank  # noqa: E402
import repo_tasks  # noqa: E402

FROZEN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "src/repo_tasks.py",
    "src/bank.py",
    "src/repo_agent.py",
    "scripts/build_bank.py",
    "scripts/train.py",
    "scripts/eval_repo_agent.py",
    "scripts/analyze_calibration.py",
    "scripts/analyze_policy.py",
    "scripts/analyze_retention.py",
    "scripts/audit_locality.py",
    "scripts/bench.py",
    "scripts/analyze_menagerie.py",
    "scripts/run.py",
)


def config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_design_lock() -> dict:
    path = EXP / "runs" / "preregistration_receipt.json"
    if not path.is_file():
        raise SystemExit("preregistration receipt is missing; no GPU/model stage is legal")
    payload = json.loads(path.read_text())
    if payload.get("status") != "locked" or tuple(payload.get("frozen_file_order", ())) != FROZEN_FILES:
        raise SystemExit("preregistration receipt is not the registered design lock")
    for relative, expected in payload["frozen_files"].items():
        observed = sha256_file(EXP / relative)
        if observed != expected:
            raise SystemExit(f"frozen design changed: {relative} {observed} != {expected}")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", payload["design_commit"], "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestry.returncode:
        raise SystemExit("design commit is not an ancestor of HEAD")
    return payload


def write_design_lock(design_commit: str) -> None:
    if subprocess.run(
        ["git", "cat-file", "-e", f"{design_commit}^{{commit}}"], cwd=ROOT, check=False
    ).returncode:
        raise SystemExit(f"unknown design commit: {design_commit}")
    paths = [str(EXP / relative) for relative in FROZEN_FILES]
    status = subprocess.run(
        ["git", "status", "--short", "--", *paths],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    if status:
        raise SystemExit(f"frozen design files are not committed:\n{status}")
    receipt = {
        "schema_version": 1,
        "status": "locked",
        "experiment_id": EXP.name,
        "design_commit": design_commit,
        "frozen_file_order": list(FROZEN_FILES),
        "frozen_files": {relative: sha256_file(EXP / relative) for relative in FROZEN_FILES},
        "model_output_precedes_lock": False,
        "note": "Only deterministic CPU bank preflight existed before this immutable design lock.",
    }
    output = EXP / "runs" / "preregistration_receipt.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))


def command(argv: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    print("[run] " + " ".join(argv), flush=True)
    child_env = {
        **os.environ,
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(argv, cwd=ROOT, check=False, env=child_env)
    if completed.returncode not in allowed:
        raise subprocess.CalledProcessError(completed.returncode, argv)
    return completed.returncode


def if_missing(output: Path, argv: list[str], allowed: tuple[int, ...] = (0,)) -> int:
    if output.exists():
        print(f"[resume] {output}", flush=True)
        return 0
    return command(argv, allowed)


def gate_if_missing(output: Path, argv: list[str]) -> bool:
    if_missing(output, argv, allowed=(0, 4))
    return bool(json.loads(output.read_text())["gate"]["passed"])


def cpu_smoke() -> dict:
    cfg = config()
    train = tuple(cfg["families"]["policy_train"])
    transfer = tuple(cfg["families"]["policy_transfer"])
    if train != repo_tasks.POLICY_TRAIN_FAMILIES:
        raise AssertionError("training family registration differs")
    if transfer != repo_tasks.POLICY_TRANSFER_FAMILIES:
        raise AssertionError("transfer family registration differs")
    if set(train) & set(transfer):
        raise AssertionError("policy train and transfer families overlap")
    tasks = repo_tasks.make_tasks(train + transfer, 2, 87601, "cpu_smoke")
    for task in tasks:
        for state, expected in (
            ("initial", (False, False)), ("partial", (False, False)),
            ("oracle", (True, True)),
        ):
            env = repo_tasks.RepoEnv(task)
            try:
                if state == "partial":
                    env.apply_partial()
                elif state == "oracle":
                    env.apply_oracle()
                observed = env.visible_pass(), env.hidden_pass()
                if observed != expected:
                    raise AssertionError((task.task_id, state, observed))
            finally:
                env.close()
    built = bank.build_banks(tasks[:4], trajectories=None)
    bank.assert_firewall_clean(built, tasks[:4])
    for key in ("recovery_action_rows", "recovery_reason_rows", "happy_action_rows"):
        rows = built[key]
        if len(rows) != 4 * 7:
            raise AssertionError(f"wrong transition-bank size for {key}")

    bcfg = cfg["bank"]
    ecfg = cfg["evaluation"]["blocks"]
    content_blocks = {
        "bank": repo_tasks.make_tasks(
            train, int(bcfg["tasks_per_policy_family"]), int(bcfg["seed"]), str(bcfg["split"])
        ),
        "calibration": repo_tasks.make_tasks(
            train, int(ecfg["policy_calibration"]["tasks_per_family"]),
            int(ecfg["policy_calibration"]["seed"]), "policy_calibration",
        ),
        "dev": repo_tasks.make_tasks(
            transfer, int(ecfg["policy_dev"]["tasks_per_family"]),
            int(ecfg["policy_dev"]["seed"]), "policy_dev",
        ),
        "confirm": repo_tasks.make_tasks(
            transfer, int(ecfg["policy_confirm"]["tasks_per_family"]),
            int(ecfg["policy_confirm"]["seed"]), "policy_confirm",
        ),
    }
    content_hashes = {
        name: [repo_tasks.content_digest(task) for task in block]
        for name, block in content_blocks.items()
    }
    for name, hashes in content_hashes.items():
        if len(hashes) != len(set(hashes)):
            raise AssertionError(f"{name} has duplicate repository content")
    if set(content_hashes["bank"]) & set(content_hashes["calibration"]):
        raise AssertionError("bank overlaps calibration content")
    if set(content_hashes["dev"]) & set(content_hashes["confirm"]):
        raise AssertionError("dev overlaps confirm content")

    locality = json.loads(resolve(cfg["locality"]["contexts"]).read_text())
    current = {row["content_sha256"] for row in locality["contexts"]}
    if len(current) != int(cfg["locality"]["count"]):
        raise AssertionError("locality context count/hash collision")
    prior_hashes = set()
    for path in (ROOT / "experiments").glob("*/data/*.json"):
        if EXP in path.parents:
            continue
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        rows = payload.get("contexts", []) if isinstance(payload, dict) else []
        prior_hashes.update(
            row.get("content_sha256") for row in rows
            if isinstance(row, dict) and row.get("content_sha256")
        )
    if current & prior_hashes:
        raise AssertionError("fresh locality block overlaps a prior context")
    return {
        "schema_version": 1,
        "status": "PASS",
        "policy_families": len(train) + len(transfer),
        "tasks_selftested": len(tasks),
        "initial_and_partial_fail": True,
        "oracle_passes": True,
        "train_transfer_disjoint": True,
        "conditional_transitions": list(bank.TRANSITIONS),
        "firewall_clean": True,
        "fresh_locality_contexts": len(current),
        "unique_repository_content": {
            name: len(hashes) for name, hashes in content_hashes.items()
        },
        "bank_calibration_disjoint": True,
        "dev_confirm_disjoint": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpu-smoke", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--lock-design", metavar="COMMIT")
    args = parser.parse_args()
    if sum((args.smoke, args.gpu_smoke, args.full, bool(args.lock_design))) != 1:
        parser.error("choose exactly one of --smoke, --gpu-smoke, --full")
    if args.lock_design:
        write_design_lock(args.lock_design)
        return 0
    if args.smoke:
        receipt = cpu_smoke()
        output = EXP / "reports" / "smoke_receipt.json"
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt, indent=2))
        return 0

    verify_design_lock()

    cfg = config()
    artifacts = resolve(cfg["artifacts"]["root"])
    start = resolve(cfg["model"]["start_checkpoint"])
    anchor = resolve(cfg["model"]["locality_anchor"])
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")
    start_hash = cfg["model"]["start_weight_sha256"]
    tcfg = cfg["training"]

    if args.gpu_smoke:
        bank_dir = artifacts / "bank_smoke"
        if_missing(bank_dir / "receipt.json", [
            py, str(EXP / "scripts" / "build_bank.py"), "--smoke",
        ])
        adapter = artifacts / "smoke" / "adapter"
        if_missing(adapter / "training_receipt.json", [
            py, str(EXP / "scripts" / "train.py"),
            "--arm", "policy_counterexample", "--base-model", str(start),
            "--expected-base-weight-sha256", start_hash,
            "--train", str(bank_dir / "policy_counterexample.jsonl"),
            "--out", str(adapter), "--smoke", "--epochs", "1",
            "--lr", str(tcfg["learning_rate"]), "--rank", str(tcfg["rank"]),
            "--alpha", str(tcfg["alpha"]), "--batch-size", str(tcfg["batch_size"]),
            "--grad-accum", str(tcfg["gradient_accumulation_steps"]),
            "--loss-chunk-positions", str(tcfg["loss_chunk_positions"]),
            "--max-length", str(tcfg["max_length"]), "--seed", str(tcfg["seed"]),
        ])
        merged = artifacts / "smoke" / "merged"
        if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(start), "--expected-base-weight-sha256", start_hash,
            "--adapter", str(adapter), "--out", str(merged),
        ])
        output = artifacts / "smoke" / "eval.json"
        if_missing(output, [
            vpy, str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm", "gpu_smoke", "--model", str(merged),
            "--block", "policy_calibration", "--scenario-set", "recovery",
            "--mode", "deep", "--tasks-per-family", "1", "--output", str(output),
        ])
        payload = json.loads(output.read_text())
        if payload["aggregate"]["n_cases"] != 12:
            raise SystemExit("GPU smoke did not cover both recovery states")
        return 0

    bank_dir = artifacts / "bank"
    if_missing(bank_dir / "receipt.json", [
        py, str(EXP / "scripts" / "build_bank.py"),
    ])

    models = {"start": start}
    for arm in tcfg["arms"]:
        adapter = artifacts / "adapters" / arm
        if_missing(adapter / "training_receipt.json", [
            py, str(EXP / "scripts" / "train.py"), "--arm", arm,
            "--base-model", str(start), "--expected-base-weight-sha256", start_hash,
            "--train", str(bank_dir / f"{arm}.jsonl"), "--out", str(adapter),
            "--epochs", str(tcfg["epochs"]), "--lr", str(tcfg["learning_rate"]),
            "--rank", str(tcfg["rank"]), "--alpha", str(tcfg["alpha"]),
            "--batch-size", str(tcfg["batch_size"]),
            "--grad-accum", str(tcfg["gradient_accumulation_steps"]),
            "--loss-chunk-positions", str(tcfg["loss_chunk_positions"]),
            "--max-length", str(tcfg["max_length"]), "--seed", str(tcfg["seed"]),
        ])
        merged = artifacts / "merged" / arm
        if_missing(merged / "merge_receipt.json", [
            py, str(EXP / "scripts" / "merge_adapter.py"),
            "--base-model", str(start), "--expected-base-weight-sha256", start_hash,
            "--adapter", str(adapter), "--out", str(merged),
        ])
        models[arm] = merged
    models["candidate"] = models["policy_counterexample"]
    models["control"] = models["extra_transaction"]

    locality = artifacts / "eval" / "locality_candidate.json"
    locality_code = if_missing(locality, [
        py, str(EXP / "scripts" / "audit_locality.py"),
        "--before-model", str(anchor), "--after-model", str(models["candidate"]),
        "--contexts", str(resolve(cfg["locality"]["contexts"])), "--out", str(locality),
        "--ceiling", str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min", str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens", str(cfg["locality"]["max_context_tokens"]),
    ], allowed=(0, 4))
    if locality_code == 4 or not json.loads(locality.read_text())["gate"]["passed"]:
        print("[run] candidate failed locality; all behavior and Menagerie remain sealed", flush=True)
        return 4

    def evaluate(
        arm: str, model: Path, block: str, scenario_set: str,
        mode: str = "deep",
    ) -> Path:
        output = artifacts / "eval" / f"{block}_{scenario_set}_{arm}_{mode}.json"
        if_missing(output, [
            vpy, str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm", arm, "--model", str(model), "--block", block,
            "--scenario-set", scenario_set, "--mode", mode, "--output", str(output),
        ])
        return output

    calibration = {
        "start": evaluate("start", models["start"], "policy_calibration", "recovery"),
        "control": evaluate("control", models["control"], "policy_calibration", "recovery"),
    }
    feasibility = EXP / "analysis" / "calibration_feasibility.json"
    if not gate_if_missing(feasibility, [
        py, str(EXP / "scripts" / "analyze_calibration.py"),
        "--start", str(calibration["start"]), "--control", str(calibration["control"]),
        "--out", str(feasibility),
    ]):
        print("[run] calibration thresholds are infeasible; candidate remains unexposed", flush=True)
        return 4
    calibration["candidate"] = evaluate(
        "candidate", models["candidate"], "policy_calibration", "recovery"
    )
    calibration_gate = EXP / "analysis" / "calibration_gate.json"
    if not gate_if_missing(calibration_gate, [
        py, str(EXP / "scripts" / "analyze_calibration.py"),
        "--start", str(calibration["start"]), "--control", str(calibration["control"]),
        "--candidate", str(calibration["candidate"]), "--locality", str(locality),
        "--out", str(calibration_gate),
    ]):
        print("[run] policy calibration failed; transfer and Menagerie remain sealed", flush=True)
        return 4

    def policy_block(block: str) -> bool:
        controls = {
            "start": evaluate("start", models["start"], block, "recovery"),
            "control": evaluate("control", models["control"], block, "recovery"),
            "sample_more": evaluate(
                "start", models["start"], block, "recovery", mode="sample_more"
            ),
        }
        feasible = EXP / "analysis" / f"{block}_feasibility.json"
        if not gate_if_missing(feasible, [
            py, str(EXP / "scripts" / "analyze_policy.py"), "--block", block,
            "--start", str(controls["start"]), "--control", str(controls["control"]),
            "--sample-more", str(controls["sample_more"]), "--out", str(feasible),
        ]):
            return False
        candidate = evaluate("candidate", models["candidate"], block, "recovery")
        gate = EXP / "analysis" / f"{block}_gate.json"
        return gate_if_missing(gate, [
            py, str(EXP / "scripts" / "analyze_policy.py"), "--block", block,
            "--start", str(controls["start"]), "--control", str(controls["control"]),
            "--sample-more", str(controls["sample_more"]), "--candidate", str(candidate),
            "--out", str(gate),
        ])

    if not policy_block("policy_dev"):
        print("[run] unseen-policy dev failed; confirm and Menagerie remain sealed", flush=True)
        return 4
    if not policy_block("policy_confirm"):
        print("[run] unseen-policy confirm failed; Menagerie remains sealed", flush=True)
        return 4

    retention = {
        "start_recovery": evaluate("start", models["start"], "broad_recovery", "recovery"),
        "candidate_recovery": evaluate("candidate", models["candidate"], "broad_recovery", "recovery"),
        "start_normal": evaluate("start", models["start"], "broad_recovery", "normal"),
        "candidate_normal": evaluate("candidate", models["candidate"], "broad_recovery", "normal"),
    }
    retention_gate = EXP / "analysis" / "retention_gate.json"
    if not gate_if_missing(retention_gate, [
        py, str(EXP / "scripts" / "analyze_retention.py"),
        "--start-recovery", str(retention["start_recovery"]),
        "--candidate-recovery", str(retention["candidate_recovery"]),
        "--start-normal", str(retention["start_normal"]),
        "--candidate-normal", str(retention["candidate_normal"]),
        "--out", str(retention_gate),
    ]):
        print("[run] broad recovery/normal retention failed; Menagerie remains sealed", flush=True)
        return 4

    print(
        "[run] WHITEBOX_PASS: locality, calibration, two policy-transfer blocks, and "
        "retention all pass; running the frozen paired Menagerie events.",
        flush=True,
    )
    menagerie_log = EXP / "runs" / "menagerie_log.jsonl"
    existing_events = []
    if menagerie_log.exists():
        existing_events = [
            json.loads(line) for line in menagerie_log.read_text().splitlines() if line.strip()
        ]
    for tier, seed in cfg["menagerie"]["paired_seeds"].items():
        if any(
            row.get("tier") == tier and int(row.get("seed", -1)) == int(seed)
            for row in existing_events
        ):
            print(f"[resume] Menagerie {tier} seed {seed}", flush=True)
            continue
        command([
            py, str(EXP / "scripts" / "bench.py"),
            "--tier", str(tier), "--seed", str(seed),
            "--incumbent", str(anchor), "--candidate", str(models["candidate"]),
        ])
    menagerie_gate = EXP / "analysis" / "menagerie_gate.json"
    passed = gate_if_missing(menagerie_gate, [
        py, str(EXP / "scripts" / "analyze_menagerie.py"),
        "--log", str(menagerie_log), "--out", str(menagerie_gate),
    ])
    return 0 if passed else 4


if __name__ == "__main__":
    raise SystemExit(main())
