#!/usr/bin/env python3
"""Resumable locality-first interpolation experiment orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

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


def mix_name(value: float) -> str:
    return f"reason_mix_{round(value * 100):03d}"


def parent_paths(cfg: dict) -> dict[str, Path]:
    parent = cfg["parent"]
    root = resolve(parent["artifact_root"])
    return {
        "action_adapter": resolve(parent["action_adapter"]),
        "reason_adapter": resolve(parent["reason_adapter"]),
        "action_model": resolve(parent["action_model"]),
        "reason_model": resolve(parent["reason_model"]),
        "happy_model": resolve(parent["happy_model"]),
        "calibration_base": root / "eval" / "calibration_recovery_base_deep.json",
        "calibration_happy": root / "eval" / "calibration_recovery_happy_action_deep.json",
        "calibration_action": root / "eval" / "calibration_recovery_recovery_action_deep.json",
    }


def validate_parent(cfg: dict) -> dict[str, str]:
    paths = parent_paths(cfg)
    expected = cfg["parent"]["expected_sha256"]
    files = {
        "action_adapter_weights": paths["action_adapter"] / "adapter_model.safetensors",
        "reason_adapter_weights": paths["reason_adapter"] / "adapter_model.safetensors",
        "action_model_weights": paths["action_model"] / "model.safetensors",
        "reason_model_weights": paths["reason_model"] / "model.safetensors",
        "calibration_base": paths["calibration_base"],
        "calibration_happy": paths["calibration_happy"],
        "calibration_action": paths["calibration_action"],
        "locality_screen": resolve(cfg["locality"]["screen_contexts"]),
        "locality_confirm": resolve(cfg["locality"]["confirm_contexts"]),
    }
    observed = {}
    for name, path in files.items():
        if not path.is_file():
            raise SystemExit(f"missing frozen parent artifact: {path}")
        observed[name] = sha256_file(path)
        if observed[name] != expected[name]:
            raise SystemExit(
                f"frozen parent hash mismatch for {name}: {observed[name]} != {expected[name]}"
            )
    for key in ("action_model", "reason_model", "happy_model"):
        model_cfg = json.loads((paths[key] / "config.json").read_text())
        if model_cfg.get("model_type") != "qwen3_5":
            raise SystemExit(f"{key} is not Qwen3.5")
    return observed


def cpu_smoke() -> dict:
    cfg = load_config()
    if cfg["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise AssertionError("single-model rule violated")
    lambdas = [float(value) for value in cfg["interpolation"]["candidate_lambdas"]]
    if lambdas != [0.10, 0.18, 0.24, 0.30] or len(set(lambdas)) != len(lambdas):
        raise AssertionError("interpolation ladder differs from preregistration")
    observed = validate_parent(cfg)

    locality_payloads = [
        json.loads(resolve(cfg["locality"][key]).read_text())
        for key in ("screen_contexts", "confirm_contexts")
    ]
    expected_count = int(cfg["locality"]["contexts_per_block"])
    if any(len(payload["contexts"]) != expected_count for payload in locality_payloads):
        raise AssertionError("locality block has wrong size")
    content_sets = [
        {row["content_sha256"] for row in payload["contexts"]}
        for payload in locality_payloads
    ]
    if content_sets[0] & content_sets[1]:
        raise AssertionError("locality screen and confirmation overlap")

    generator = torch.Generator().manual_seed(85201)
    action_a = torch.randn(3, 5, generator=generator)
    action_b = torch.randn(7, 3, generator=generator)
    reason_a = torch.randn(3, 5, generator=generator)
    reason_b = torch.randn(7, 3, generator=generator)
    action_delta = action_b @ action_a
    reason_delta = reason_b @ reason_a
    mixed_zero = action_delta + 0.0 * (reason_delta - action_delta)
    mixed_one = action_delta + 1.0 * (reason_delta - action_delta)
    if not torch.equal(mixed_zero, action_delta) or not torch.allclose(mixed_one, reason_delta):
        raise AssertionError("interpolation endpoints are not exact")

    families = tuple(cfg["families"]["train"] + cfg["families"]["transfer"])
    smoke_tasks = repo_tasks.make_tasks(families, 1, seed=85190, split="smoke")
    blocks = cfg["evaluation"]["blocks"]
    if len({int(spec["seed"]) for spec in blocks.values()}) != len(blocks):
        raise AssertionError("evaluation block seeds overlap")
    if set(cfg["families"]["train"]) & set(cfg["families"]["transfer"]):
        raise AssertionError("training and transfer families overlap")
    if any(task.split != "smoke" for task in smoke_tasks):
        raise AssertionError("smoke task split is mislabeled")
    evaluation = cfg["evaluation"]
    per_call = int(evaluation["think_budget"]) + int(evaluation["answer_max_tokens"])
    for scenario in ("recovery", "normal"):
        budget = evaluation[scenario]
        deep = int(budget["deep_turns"]) * per_call
        sampled = (
            int(budget["sample_more_trajectories"])
            * int(budget["sample_more_turns_each"])
            * per_call
        )
        if deep != sampled:
            raise AssertionError(f"matched-compute reservation differs for {scenario}")

    base = json.loads(parent_paths(cfg)["calibration_base"].read_text())["aggregate"]
    happy = json.loads(parent_paths(cfg)["calibration_happy"].read_text())["aggregate"]
    selection = cfg["selection_gates"]
    reachability = {
        "vs_base": 1.0 - float(base["success"]) >= float(
            selection["recovery_delta_vs_base_min"]
        ),
        "vs_happy": 1.0 - float(happy["success"]) >= float(
            selection["recovery_delta_vs_happy_min"]
        ),
        "invalid": -float(base["invalid_action_rate_per_turn"]) <= float(
            selection["invalid_action_delta_vs_base_max"]
        ),
        "rejected_transition": float(
            selection["rejected_patch_transition_absolute_min"]
        ) <= 1.0,
        "failed_transition": float(
            selection["failed_test_changed_patch_within_two_absolute_min"]
        ) <= 1.0,
    }
    if not all(reachability.values()):
        raise AssertionError(f"unreachable calibration gate: {reachability}")
    return {
        "schema_version": 1,
        "status": "PASS",
        "model": cfg["model"]["id"],
        "candidate_lambdas": lambdas,
        "interpolation_formula": (
            "W_apex + (1-lambda)*delta_action + lambda*delta_reason"
        ),
        "parent_hashes": observed,
        "locality_blocks": [len(payload["contexts"]) for payload in locality_payloads],
        "locality_blocks_disjoint": True,
        "interpolation_endpoints_verified": True,
        "split_ids_disjoint_by_family_seed_contract": True,
        "matched_compute_reservations": True,
        "calibration_gates_reachable": reachability,
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
    validate_parent(cfg)
    parents = parent_paths(cfg)
    artifact_root = resolve(cfg["artifacts"]["root"])
    start_model = resolve(cfg["model"]["start_checkpoint"])
    py = str(ROOT / ".venv" / "bin" / "python")
    vpy = str(ROOT / ".venv-vllm" / "bin" / "python")

    def merge_candidate(lam: float) -> Path:
        name = mix_name(lam)
        merged = artifact_root / "merged" / name
        run_if_missing(merged / "interpolation_receipt.json", [
            py,
            str(EXP / "scripts" / "interpolate_adapters.py"),
            "--base-model",
            str(start_model),
            "--action-adapter",
            str(parents["action_adapter"]),
            "--reason-adapter",
            str(parents["reason_adapter"]),
            "--reason-lambda",
            str(lam),
            "--out",
            str(merged),
        ])
        return merged

    lambdas = [float(value) for value in cfg["interpolation"]["candidate_lambdas"]]
    if args.gpu_smoke:
        model = merge_candidate(lambdas[0])
        output = artifact_root / "smoke" / "locality_two_contexts.json"
        return run_if_missing(output, [
            py,
            str(EXP / "scripts" / "audit_locality_ladder.py"),
            "--before-model",
            str(start_model),
            "--candidate",
            f"{mix_name(lambdas[0])}={model}",
            "--eligible",
            mix_name(lambdas[0]),
            "--contexts",
            str(resolve(cfg["locality"]["confirm_contexts"])),
            "--out",
            str(output),
            "--drift-ceiling",
            str(cfg["locality"]["median_non_target_logit_drift_max"]),
            "--entropy-delta-min",
            str(cfg["locality"]["mean_entropy_delta_min"]),
            "--expected-contexts",
            "2",
            "--limit-contexts",
            "2",
        ], allowed_returncodes=(0, 4))

    models = {mix_name(lam): merge_candidate(lam) for lam in lambdas}
    screen = artifact_root / "eval" / "locality_screen.json"
    screen_command = [
        py,
        str(EXP / "scripts" / "audit_locality_ladder.py"),
        "--before-model",
        str(start_model),
        "--candidate",
        f"action_anchor={parents['action_model']}",
    ]
    for name, model in models.items():
        screen_command.extend(("--candidate", f"{name}={model}"))
    screen_command.extend((
        "--candidate",
        f"reason_endpoint={parents['reason_model']}",
        "--eligible",
        "action_anchor",
    ))
    for name in models:
        screen_command.extend(("--eligible", name))
    screen_command.extend((
        "--contexts",
        str(resolve(cfg["locality"]["screen_contexts"])),
        "--out",
        str(screen),
        "--drift-ceiling",
        str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min",
        str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens",
        str(cfg["locality"]["max_context_tokens"]),
        "--expected-contexts",
        str(cfg["locality"]["contexts_per_block"]),
    ))
    screen_code = run_if_missing(screen, screen_command, allowed_returncodes=(0, 4))
    if screen_code == 4:
        print("[run] no locality-screened candidate; stopping before behavior", flush=True)
        return 4
    screen_payload = json.loads(screen.read_text())
    passing = screen_payload["passing_eligible_candidates"]
    if not any(name.startswith("reason_mix_") for name in passing):
        print("[run] no scaled plan contrast passes locality; stopping before behavior", flush=True)
        return 4

    def evaluate(
        arm: str,
        model: Path,
        block: str,
        scenario_set: str,
        mode: str = "deep",
        scaffold: bool = False,
    ) -> Path:
        suffix = "_scaffold" if scaffold else ""
        output = artifact_root / "eval" / f"{block}_{scenario_set}_{arm}_{mode}{suffix}.json"
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
        run_if_missing(output, command)
        return output

    calibration_candidates: dict[str, Path] = {}
    for name in passing:
        if name == "action_anchor":
            calibration_candidates[name] = parents["calibration_action"]
        else:
            calibration_candidates[name] = evaluate(
                name, models[name], "calibration", "recovery"
            )
    selection = EXP / "analysis" / "candidate_selection.json"
    selection_command = [
        py,
        str(EXP / "scripts" / "select_interpolation.py"),
        "--base",
        str(parents["calibration_base"]),
        "--happy",
        str(parents["calibration_happy"]),
        "--locality-screen",
        str(screen),
        "--out",
        str(selection),
    ]
    for name, path in calibration_candidates.items():
        selection_command.extend(("--candidate", f"{name}={path}"))
    selection_code = run_command(selection_command, allowed_returncodes=(0, 4))
    if selection_code == 4:
        print("[run] no locality-safe behavior candidate; stopping before transfer", flush=True)
        return 4
    selected_payload = json.loads(selection.read_text())
    selected = selected_payload["selected_arm"]
    candidate_model = (
        parents["action_model"] if selected == "action_anchor" else models[selected]
    )

    confirm = artifact_root / "eval" / f"locality_confirm_{selected}.json"
    confirm_code = run_if_missing(confirm, [
        py,
        str(EXP / "scripts" / "audit_locality_ladder.py"),
        "--before-model",
        str(start_model),
        "--candidate",
        f"{selected}={candidate_model}",
        "--eligible",
        selected,
        "--contexts",
        str(resolve(cfg["locality"]["confirm_contexts"])),
        "--out",
        str(confirm),
        "--drift-ceiling",
        str(cfg["locality"]["median_non_target_logit_drift_max"]),
        "--entropy-delta-min",
        str(cfg["locality"]["mean_entropy_delta_min"]),
        "--max-context-tokens",
        str(cfg["locality"]["max_context_tokens"]),
        "--expected-contexts",
        str(cfg["locality"]["contexts_per_block"]),
    ], allowed_returncodes=(0, 4))
    if confirm_code == 4:
        print("[run] selected candidate failed independent locality confirmation", flush=True)
        return 4

    def run_transfer_block(block: str) -> int:
        # Frozen controls run before the candidate so every absolute/delta gate
        # is proven attainable on this block before candidate evaluation.
        base_recovery = evaluate("base", start_model, block, "recovery")
        happy_recovery = evaluate("happy_action", parents["happy_model"], block, "recovery")
        action_recovery = evaluate("action_anchor", parents["action_model"], block, "recovery")
        sample_recovery = evaluate("base", start_model, block, "recovery", mode="sample_more")
        scaffold_recovery = evaluate(
            "base", start_model, block, "recovery", scaffold=True
        )
        base_normal = evaluate("base", start_model, block, "normal")
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
            print(f"[run] {block} has an unreachable frozen gate", flush=True)
            return 4
        candidate_recovery = evaluate(selected, candidate_model, block, "recovery")
        candidate_normal = evaluate(selected, candidate_model, block, "normal")
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
            str(confirm),
            "--out",
            str(output),
        ], allowed_returncodes=(0, 4))

    if run_transfer_block("transfer_dev") == 4:
        print("[run] transfer-dev gate failed; stopping before confirmation and Menagerie", flush=True)
        return 4
    if run_transfer_block("transfer_confirm") == 4:
        print("[run] confirmation gate failed; Menagerie remains sealed", flush=True)
        return 4
    print(
        "[run] all white-box gates passed; assign fresh paired Menagerie seeds through "
        "the benchmark CLI before final evaluation",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
