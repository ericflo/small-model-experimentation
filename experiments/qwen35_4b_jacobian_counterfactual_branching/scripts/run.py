#!/usr/bin/env python3
"""Fail-closed staged runner for Jacobian counterfactual branching."""

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
REPO = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from branch_geometry import (  # noqa: E402
    balanced_j_branches,
    geometry_receipt,
    gram_matched_non_j,
    normalized_dictionary,
)
from model_ops import QwenCommitModel  # noqa: E402
from task_data import task_prompt  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate_model_boundary(config: dict) -> dict:
    boundary = config["design_boundary"]
    if boundary.get("status") != "anchored":
        raise RuntimeError("model smoke requires anchored design boundary")
    commit = str(boundary.get("implementation_commit", ""))
    hashes = boundary.get("implementation_hashes", {})
    paths = {
        "runner_sha256": EXP / "scripts" / "run.py",
        "model_ops_sha256": EXP / "src" / "model_ops.py",
        "branch_geometry_sha256": EXP / "src" / "branch_geometry.py",
        "tests_sha256": EXP / "tests" / "test_branch_design.py",
    }
    if not commit or set(hashes) != set(paths):
        raise RuntimeError("model boundary is incomplete")
    subprocess.check_call(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=REPO
    )
    for key, path in paths.items():
        if sha256(path) != hashes[key]:
            raise RuntimeError(f"current implementation hash changed: {key}")
        relative = path.relative_to(REPO).as_posix()
        committed = subprocess.check_output(
            ["git", "show", f"{commit}:{relative}"], cwd=REPO
        )
        if hashlib.sha256(committed).hexdigest() != hashes[key]:
            raise RuntimeError(f"anchored implementation hash changed: {key}")
    return {"implementation_commit": commit, "implementation_hashes": hashes}


def cpu_smoke() -> None:
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    lens_path = EXP / config["lens"]["path"]
    if sha256(lens_path) != config["lens"]["sha256"]:
        raise RuntimeError("frozen lens hash changed")
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    if tuple(lens["concepts"][:12]) != tuple(config["data"]["operation_aliases"].values()):
        raise RuntimeError("public alias/lens concept order changed")
    manifest_path = EXP / "data" / "procedural" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("all_disjoint") is not True:
        raise RuntimeError("fresh data disjointness failed")
    expected = {
        "mechanics": int(config["data"]["mechanics_tasks"]),
        "qualification": int(config["data"]["qualification_tasks"]),
        "confirmation": int(config["data"]["confirmation_tasks"]),
    }
    if {name: int(value["rows"]) for name, value in manifest["splits"].items()} != expected:
        raise RuntimeError("fresh data cardinality changed")
    geometry = {}
    for layer in config["lens"]["band"]:
        directions = lens["directions"][layer]
        if int(torch.linalg.matrix_rank(directions.float()).item()) != 24:
            raise RuntimeError(f"lens layer {layer} lost rank")
        for alpha in config["lens"]["alpha_multipliers"]:
            j = balanced_j_branches(
                directions,
                public_concepts=int(config["lens"]["public_alias_concepts"]),
                target_rms_norm=float(config["lens"]["replicated_median_delta_norms"][layer]) * float(alpha),
            )
            non_j = gram_matched_non_j(
                directions,
                j,
                seed=int(config["seeds"]["non_j_geometry"]) + int(layer),
                rtol=float(config["lens"]["pseudoinverse_rtol"]),
            )
            receipt = geometry_receipt(
                directions, j, non_j, rtol=float(config["lens"]["pseudoinverse_rtol"])
            )
            if receipt["j"]["maximum_sum_abs"] > float(config["controls"]["branch_sum_norm_max"]):
                raise RuntimeError("J branches are not zero sum")
            if receipt["non_j"]["maximum_sum_abs"] > float(config["controls"]["branch_sum_norm_max"]):
                raise RuntimeError("non-J branches are not zero sum")
            if receipt["gram_relative_error"] > float(config["controls"]["gram_relative_error_max"]):
                raise RuntimeError("non-J branch Gram mismatch")
            geometry[f"layer_{layer}_alpha_{alpha}"] = receipt
    result = {
        "schema_version": 1,
        "stage": "cpu-smoke",
        "passed": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "confirmation_opened": False,
        "lens_sha256": sha256(lens_path),
        "config_sha256": sha256(config_path),
        "data_manifest_sha256": sha256(manifest_path),
        "fresh_tasks": expected,
        "ancestor_unique_fingerprints": manifest["ancestor_unique_fingerprints"],
        "geometry": geometry,
        "downstream_available": False,
    }
    write_json(EXP / "runs" / "smoke" / "cpu.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def _branch_bank(config: dict, lens: dict, alpha: float) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    j_by_layer = {}
    non_j_by_layer = {}
    for layer in config["lens"]["band"]:
        directions = lens["directions"][layer]
        j = balanced_j_branches(
            directions,
            public_concepts=int(config["lens"]["public_alias_concepts"]),
            target_rms_norm=float(config["lens"]["replicated_median_delta_norms"][layer]) * float(alpha),
        )
        non_j = gram_matched_non_j(
            directions,
            j,
            seed=int(config["seeds"]["non_j_geometry"]) + int(layer),
            rtol=float(config["lens"]["pseudoinverse_rtol"]),
        )
        j_by_layer[int(layer)] = j
        non_j_by_layer[int(layer)] = non_j
    return j_by_layer, non_j_by_layer


def _live_numeric_receipt(
    config: dict,
    lens: dict,
    j_result: dict,
    non_j_result: dict,
) -> dict[str, object]:
    rows = {}
    maxima = {
        "j_requested_norm_relative_error": 0.0,
        "non_j_paired_norm_relative_error": 0.0,
        "non_j_span_projection_fraction": 0.0,
        "realized_gram_relative_error": 0.0,
        "j_realized_sum_abs": 0.0,
        "non_j_realized_sum_abs": 0.0,
    }
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    for layer in config["lens"]["band"]:
        j_requested = j_result["requested_deltas"][layer].float()
        j_realized = j_result["realized_deltas"][layer].float()
        non_j_realized = non_j_result["realized_deltas"][layer].float()
        j_request_norm = j_requested.norm(dim=-1)
        j_norm = j_realized.norm(dim=-1)
        non_j_norm = non_j_realized.norm(dim=-1)
        j_error = ((j_norm - j_request_norm).abs() / j_request_norm.clamp_min(1e-12)).max()
        non_j_error = ((non_j_norm - j_norm).abs() / j_norm.clamp_min(1e-12)).max()
        dictionary = normalized_dictionary(lens["directions"][layer])
        inverse = torch.linalg.pinv(dictionary, rtol=rtol)
        projection = (non_j_realized @ inverse.T) @ dictionary.T
        projection_fraction = (
            projection.norm(dim=-1) / non_j_norm.clamp_min(1e-12)
        ).max()
        j_gram = j_realized @ j_realized.T
        non_j_gram = non_j_realized @ non_j_realized.T
        gram_error = torch.linalg.norm(j_gram - non_j_gram) / torch.linalg.norm(j_gram).clamp_min(1e-12)
        row = {
            "j_requested_norm_relative_error": float(j_error),
            "non_j_paired_norm_relative_error": float(non_j_error),
            "non_j_span_projection_fraction": float(projection_fraction),
            "realized_gram_relative_error": float(gram_error),
            "j_realized_sum_abs": float(j_realized.sum(dim=0).abs().max()),
            "non_j_realized_sum_abs": float(non_j_realized.sum(dim=0).abs().max()),
        }
        rows[str(layer)] = row
        for key, value in row.items():
            maxima[key] = max(maxima[key], value)
    controls = config["controls"]
    passed = bool(
        maxima["non_j_paired_norm_relative_error"]
        <= float(controls["post_bf16_norm_relative_tolerance"])
        and maxima["non_j_span_projection_fraction"]
        <= float(controls["post_bf16_non_j_span_projection_max"])
    )
    return {
        "by_layer": rows,
        "maxima": maxima,
        "passed": passed,
        "live_frozen_gates": [
            "non_j_paired_norm_relative_error",
            "non_j_span_projection_fraction",
        ],
        "pre_bf16_geometry_gates_verified_in_cpu_smoke": [
            "zero_sum",
            "full_gram",
            "rank",
        ],
    }


def model_smoke() -> None:
    config_path = EXP / "configs" / "default.yaml"
    config = yaml.safe_load(config_path.read_text())
    boundary = _validate_model_boundary(config)
    lens_path = EXP / config["lens"]["path"]
    if sha256(lens_path) != config["lens"]["sha256"]:
        raise RuntimeError("frozen lens hash changed")
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    task = json.loads((EXP / "data" / "procedural" / "mechanics.jsonl").read_text().splitlines()[0])
    public_task = {"visible": task["visible"]}
    aliases_by_operation = config["data"]["operation_aliases"]
    aliases = list(aliases_by_operation.values())
    model = QwenCommitModel(config)
    prepared = model.prepare(
        task_prompt(public_task, aliases_by_operation),
        prompt_max_tokens=int(config["generation"]["prompt_max_tokens"]),
    )
    trace = model.generate_trace(
        prepared["input_ids"],
        seed=int(config["seeds"]["mechanics_prefix"]),
        thought_cap=32,
        answer_cap=1,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        top_k=int(config["generation"]["top_k"]),
    )
    if trace["natural_close"] or trace["think_tokens"] != 32:
        raise RuntimeError("model smoke prefix did not remain in live thought")
    thought_ids = [int(value) for value in trace["generated_token_ids"]]
    j_bank, non_j_bank = _branch_bank(config, lens, alpha=1.0)
    baseline = model.slot_readout(
        prepared["input_ids"],
        thought_ids,
        slot_text=config["generation"]["slot_text"],
        aliases=aliases,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
    )
    j_result = model.branched_slot_readout(
        prepared["input_ids"], thought_ids,
        slot_text=config["generation"]["slot_text"], aliases=aliases,
        branches_by_layer=j_bank,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
    )
    non_j_result = model.branched_slot_readout(
        prepared["input_ids"], thought_ids,
        slot_text=config["generation"]["slot_text"], aliases=aliases,
        branches_by_layer=non_j_bank,
        total_max_tokens=int(config["generation"]["total_max_tokens"]),
        quantization_control={
            "directions_by_layer": {
                int(layer): lens["directions"][layer]
                for layer in config["lens"]["band"]
            },
            "target_norms_by_layer": {
                int(layer): j_result["realized_deltas"][layer].float().norm(dim=-1)
                for layer in config["lens"]["band"]
            },
            "rtol": config["lens"]["pseudoinverse_rtol"],
            "norm_tolerance": config["controls"]["post_bf16_norm_relative_tolerance"],
            "projection_tolerance": config["controls"]["post_bf16_non_j_span_projection_max"],
            "correction_iterations": config["controls"]["live_control_correction_iterations"],
            "correction_damping": config["controls"]["live_control_correction_damping"],
            "lattice_pair_steps": config["controls"]["live_control_lattice_pair_steps"],
            "repair_safety_margin": config["controls"]["live_control_repair_safety_margin"],
        },
    )
    numeric = _live_numeric_receipt(config, lens, j_result, non_j_result)
    result = {
        "schema_version": 1,
        "stage": "model-smoke",
        "passed": bool(
            baseline["finite"] and j_result["finite"] and non_j_result["finite"]
            and all(value == 1 for value in j_result["applications"].values())
            and all(value == 1 for value in non_j_result["applications"].values())
            and numeric["passed"]
        ),
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "model_load_seconds": model.load_seconds,
        "thought_tokens": len(thought_ids),
        "prompt_tokens": prepared["prompt_tokens"],
        "branches": len(aliases),
        "j_finite": j_result["finite"],
        "non_j_finite": non_j_result["finite"],
        "applications_j": j_result["applications"],
        "applications_non_j": non_j_result["applications"],
        "non_j_control_iterations_used": non_j_result["control_iterations_used"],
        "non_j_control_passed_rows": {
            str(layer): int(value.sum().item())
            for layer, value in non_j_result["control_passed_rows"].items()
        },
        "non_j_control_lattice_pair_steps": {
            str(layer): [int(item) for item in value.tolist()]
            for layer, value in non_j_result["control_lattice_pair_steps"].items()
        },
        "numeric": numeric,
        "outcomes_loaded": False,
        "correct_alias_loaded": False,
        "branch_probabilities_recorded": False,
        "confirmation_opened": False,
        "peak_allocated_bytes": (
            int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
        ),
        "config_sha256": sha256(config_path),
        "lens_sha256": sha256(lens_path),
        "design_boundary": boundary,
    }
    write_json(EXP / "runs" / "smoke" / "model_005.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "model-smoke", "mechanics", "qualification", "confirmation"),
    )
    args = parser.parse_args()
    if args.stage == "smoke":
        cpu_smoke()
        return
    if args.stage == "model-smoke":
        model_smoke()
        return
    raise RuntimeError(f"stage {args.stage!r} is unavailable before implementation boundary")


if __name__ == "__main__":
    main()
