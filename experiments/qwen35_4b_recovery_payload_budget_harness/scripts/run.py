#!/usr/bin/env python3
"""Resumable payload-capable recovery harness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import repo_agent  # noqa: E402
import repo_tasks  # noqa: E402


def load_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def model_paths(cfg: dict) -> dict[str, Path]:
    return {name: resolve(cfg["model"][name]) for name in ("base", "happy", "action", "candidate")}


def validate_inputs(cfg: dict) -> dict[str, str]:
    paths = model_paths(cfg)
    observed = {}
    for name, path in paths.items():
        config = json.loads((path / "config.json").read_text())
        if config.get("model_type") != "qwen3_5":
            raise SystemExit(f"{name} is not a Qwen/Qwen3.5-4B checkpoint")
        observed[name] = sha256_file(path / "model.safetensors")
        expected = cfg["model"]["expected_weight_sha256"][name]
        if observed[name] != expected:
            raise SystemExit(f"frozen {name} weight hash mismatch")
    locality = resolve(cfg["locality"]["contexts"])
    observed["locality_contexts"] = sha256_file(locality)
    if observed["locality_contexts"] != cfg["locality"]["contexts_sha256"]:
        raise SystemExit("fresh locality context hash mismatch")
    return observed


def fake_output(action: dict, turn: int) -> dict:
    return {
        "text": "</think>\n" + json.dumps(action, separators=(",", ":")),
        "n_thinking_tokens": 1,
        "n_answer_tokens": 16,
        "n_sampled_tokens": 17,
        "thinking_closed": True,
        "forced_close": False,
        "turn": turn,
    }


def cpu_smoke() -> dict:
    cfg = load_config()
    if cfg["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise AssertionError("single-model rule violated")
    if int(cfg["evaluation"]["think_budget"]) != 512 or int(
        cfg["evaluation"]["answer_max_tokens"]
    ) != 512:
        raise AssertionError("payload-harness budget differs from preregistration")
    hashes = validate_inputs(cfg)

    current = json.loads(resolve(cfg["locality"]["contexts"]).read_text())
    prior_paths = (
        ROOT / "experiments/qwen35_4b_recovery_reason_locality_interpolation/data/locality_screen.json",
        ROOT / "experiments/qwen35_4b_recovery_reason_locality_interpolation/data/locality_confirm.json",
    )
    current_hashes = {row["content_sha256"] for row in current["contexts"]}
    if len(current_hashes) != int(cfg["locality"]["count"]):
        raise AssertionError("fresh locality block is duplicated or wrong-sized")
    for path in prior_paths:
        prior = json.loads(path.read_text())
        if current_hashes & {row["content_sha256"] for row in prior["contexts"]}:
            raise AssertionError("fresh locality block overlaps a prior instrument")

    tasks = repo_tasks.make_tasks(
        tuple(cfg["families"]["train"] + cfg["families"]["transfer"]),
        1,
        seed=85310,
        split="smoke",
    )
    episode = repo_agent.Episode(tasks[0], 0, scenario="rejected_patch")
    try:
        episode.consume(fake_output({"tool": "read", "path": tasks[0].oracle_patches[0].path}, 1))
        patch = tasks[0].oracle_patches[0]
        episode.consume(fake_output({
            "tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new,
        }, 2))
        result = episode.finish()
    except Exception:
        episode.env.close()
        raise
    if result["rejected_patch_changed_immediately"]:
        raise AssertionError("inspect-then-patch was misclassified as immediate")
    if not result["rejected_patch_changed_within_two"]:
        raise AssertionError("inspect-then-patch was not retained within two turns")
    if not result["rejected_patch_valid_changed_within_two"]:
        raise AssertionError("valid inspect-then-patch path was rejected")

    per_call = int(cfg["evaluation"]["think_budget"]) + int(
        cfg["evaluation"]["answer_max_tokens"]
    )
    reservations = {}
    for scenario in ("recovery", "normal"):
        budget = cfg["evaluation"][scenario]
        deep = int(budget["deep_turns"]) * per_call
        sample = (
            int(budget["sample_more_trajectories"])
            * int(budget["sample_more_turns_each"])
            * per_call
        )
        if deep != sample:
            raise AssertionError(f"matched-compute reservation differs for {scenario}")
        reservations[scenario] = deep
    blocks = cfg["evaluation"]["blocks"]
    if len({int(spec["seed"]) for spec in blocks.values()}) != len(blocks):
        raise AssertionError("evaluation seeds overlap")
    if set(cfg["families"]["train"]) & set(cfg["families"]["transfer"]):
        raise AssertionError("training and transfer families overlap")
    return {
        "schema_version": 1,
        "status": "PASS",
        "model": cfg["model"]["id"],
        "input_hashes": hashes,
        "think_budget": cfg["evaluation"]["think_budget"],
        "answer_max_tokens": cfg["evaluation"]["answer_max_tokens"],
        "matched_compute_reservations": reservations,
        "fresh_locality_contexts": len(current_hashes),
        "locality_disjoint_from_prior_blocks": True,
        "inspect_then_patch_retained_within_two": True,
        "transfer_families_uninstantiated": True,
        "benchmark_content_accessed": False,
    }


def run_command(command: list[str], allowed_returncodes: tuple[int, ...] = (0,)) -> int:
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
        if output.suffix == ".json":
            payload = json.loads(output.read_text())
            if "gate" in payload and not payload["gate"].get("passed", False):
                return 4
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
    validate_inputs(cfg)
    models = model_paths(cfg)
    artifact_root = resolve(cfg["artifacts"]["root"])
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")

    def evaluate(
        arm: str,
        model: Path,
        block: str,
        scenario_set: str,
        mode: str = "deep",
        scaffold: bool = False,
        tasks_per_family: int | None = None,
        output_override: Path | None = None,
    ) -> Path:
        suffix = "_scaffold" if scaffold else ""
        output = output_override or artifact_root / "eval" / (
            f"{block}_{scenario_set}_{arm}_{mode}{suffix}.json"
        )
        command = [
            vpy,
            str(EXP / "scripts" / "eval_repo_agent.py"),
            "--arm",
            arm,
            "--model",
            str(model),
            "--block",
            block,
            "--scenario-set",
            scenario_set,
            "--mode",
            mode,
            "--output",
            str(output),
        ]
        if scaffold:
            command.append("--scaffold")
        if tasks_per_family is not None:
            command.extend(("--tasks-per-family", str(tasks_per_family)))
        run_if_missing(output, command)
        return output

    if args.gpu_smoke:
        output = artifact_root / "smoke" / "candidate_payload_eval.json"
        evaluate(
            "candidate",
            models["candidate"],
            "calibration",
            "recovery",
            tasks_per_family=1,
            output_override=output,
        )
        payload = json.loads(output.read_text())
        required = (
            "answer_cap_hit_rate_per_turn",
            "invalid_answer_cap_hit_fraction",
        )
        if any(key not in payload["aggregate"] for key in required):
            raise SystemExit("GPU smoke is missing payload telemetry")
        if "valid_changed_patch_within_two" not in payload["aggregate"]["per_scenario"]["rejected_patch"]:
            raise SystemExit("GPU smoke is missing the two-turn transition metric")
        return 0

    locality = artifact_root / "eval" / "locality_candidate.json"
    locality_code = run_if_missing(locality, [
        py,
        str(EXP / "scripts" / "audit_locality.py"),
        "--before-model",
        str(models["base"]),
        "--candidate",
        f"candidate={models['candidate']}",
        "--eligible",
        "candidate",
        "--contexts",
        str(resolve(cfg["locality"]["contexts"])),
        "--out",
        str(locality),
        "--drift-ceiling",
        str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min",
        str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens",
        str(cfg["locality"]["max_context_tokens"]),
        "--expected-contexts",
        str(cfg["locality"]["count"]),
    ], allowed_returncodes=(0, 4))
    if locality_code == 4:
        print("[run] fixed candidate failed fresh locality; stopping before behavior", flush=True)
        return 4

    calibration = {
        name: evaluate(name, models[name], "calibration", "recovery")
        for name in ("base", "happy", "action")
    }
    calibration_feasibility = EXP / "analysis" / "calibration_feasibility.json"
    feasibility_code = run_if_missing(calibration_feasibility, [
        py,
        str(EXP / "scripts" / "check_calibration_feasibility.py"),
        "--base",
        str(calibration["base"]),
        "--happy",
        str(calibration["happy"]),
        "--action",
        str(calibration["action"]),
        "--out",
        str(calibration_feasibility),
    ], allowed_returncodes=(0, 4))
    if feasibility_code == 4:
        print("[run] a calibration gate is unreachable; stopping before candidate", flush=True)
        return 4
    candidate_calibration = evaluate(
        "candidate", models["candidate"], "calibration", "recovery"
    )
    calibration_gate = EXP / "analysis" / "calibration_gate.json"
    calibration_code = run_if_missing(calibration_gate, [
        py,
        str(EXP / "scripts" / "analyze_calibration.py"),
        "--candidate",
        str(candidate_calibration),
        "--base",
        str(calibration["base"]),
        "--happy",
        str(calibration["happy"]),
        "--action",
        str(calibration["action"]),
        "--locality",
        str(locality),
        "--out",
        str(calibration_gate),
    ], allowed_returncodes=(0, 4))
    if calibration_code == 4:
        print("[run] payload-capable calibration gate failed; transfer remains sealed", flush=True)
        return 4

    def run_transfer_block(block: str) -> int:
        base_recovery = evaluate("base", models["base"], block, "recovery")
        happy_recovery = evaluate("happy", models["happy"], block, "recovery")
        action_recovery = evaluate("action", models["action"], block, "recovery")
        sample_recovery = evaluate(
            "base", models["base"], block, "recovery", mode="sample_more"
        )
        scaffold_recovery = evaluate(
            "base", models["base"], block, "recovery", scaffold=True
        )
        base_normal = evaluate("base", models["base"], block, "normal")
        feasibility = EXP / "analysis" / f"{block}_feasibility.json"
        feasibility_code = run_if_missing(feasibility, [
            py,
            str(EXP / "scripts" / "check_gate_feasibility.py"),
            "--block",
            block,
            "--base-recovery",
            str(base_recovery),
            "--happy-recovery",
            str(happy_recovery),
            "--action-recovery",
            str(action_recovery),
            "--sample-more-recovery",
            str(sample_recovery),
            "--scaffold-recovery",
            str(scaffold_recovery),
            "--base-normal",
            str(base_normal),
            "--out",
            str(feasibility),
        ], allowed_returncodes=(0, 4))
        if feasibility_code == 4:
            print(f"[run] {block} contains an unreachable frozen gate", flush=True)
            return 4
        candidate_recovery = evaluate(
            "candidate", models["candidate"], block, "recovery"
        )
        candidate_normal = evaluate("candidate", models["candidate"], block, "normal")
        output = EXP / "analysis" / f"{block}_gate.json"
        return run_command([
            py,
            str(EXP / "scripts" / "analyze_primary.py"),
            "--block",
            block,
            "--candidate-recovery",
            str(candidate_recovery),
            "--base-recovery",
            str(base_recovery),
            "--happy-recovery",
            str(happy_recovery),
            "--action-recovery",
            str(action_recovery),
            "--sample-more-recovery",
            str(sample_recovery),
            "--scaffold-recovery",
            str(scaffold_recovery),
            "--candidate-normal",
            str(candidate_normal),
            "--base-normal",
            str(base_normal),
            "--locality",
            str(locality),
            "--out",
            str(output),
        ], allowed_returncodes=(0, 4))

    if run_transfer_block("transfer_dev") == 4:
        print("[run] transfer-dev gate failed; stopping before confirmation and Menagerie", flush=True)
        return 4
    if run_transfer_block("transfer_confirm") == 4:
        print("[run] transfer confirmation failed; Menagerie remains sealed", flush=True)
        return 4
    print(
        "[run] all white-box gates passed; assign fresh paired Menagerie seeds through "
        "the public benchmark CLI",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
