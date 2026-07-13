#!/usr/bin/env python3
"""Fail-closed staged harness for semantic-anchor coordinate branching."""

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
CONFIG = EXP / "configs" / "default.yaml"
sys.path.insert(0, str(EXP / "src"))

from task_data import (  # noqa: E402
    behavior_fingerprint,
    build_splits,
    public_mechanics,
    task_fingerprint,
    task_prompt,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def scientific_config_sha256(config: dict) -> str:
    value = json.loads(json.dumps(config))
    value.pop("design_boundary", None)
    value.pop("implementation_boundary", None)
    value.pop("mechanics_boundary", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def ancestor_behavior_fingerprints() -> set[str]:
    """Read experiment procedural artifacts only; benchmark contents stay forbidden."""

    values: set[str] = set()
    for path in sorted((ROOT / "experiments").glob("*/data/procedural/*.jsonl")):
        if EXP in path.parents:
            continue
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if {"depth", "visible", "hidden", "first_op"}.issubset(row):
                values.add(behavior_fingerprint(row))
    return values


def validate_design_boundary(config: dict) -> dict:
    boundary = config["design_boundary"]
    if boundary.get("status") != "anchored":
        raise RuntimeError("scientific design boundary is not anchored")
    commit = str(boundary["commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    paths = {
        "readme_sha256": EXP / "README.md",
        "preregistration_sha256": EXP / "reports" / "preregistration.md",
        "design_review_sha256": EXP / "reports" / "design_review.md",
        "data_manifest_sha256": EXP / "data" / "procedural" / "manifest.json",
        "mechanics_public_sha256": EXP / "data" / "procedural" / "mechanics_public.jsonl",
        "lens_sha256": EXP / "assets" / "context_lens.pt",
    }
    expected = {
        key: str(value) for key, value in boundary["hashes"].items()
        if key != "scientific_config_sha256"
    }
    local = {key: sha256(path) for key, path in paths.items()}
    committed = {}
    for key, path in paths.items():
        relative = path.relative_to(ROOT).as_posix()
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        committed[key] = hashlib.sha256(content).hexdigest()
    config_hash = scientific_config_sha256(config)
    if (
        not ancestor or local != expected or committed != expected
        or config_hash != boundary["hashes"].get("scientific_config_sha256")
    ):
        raise RuntimeError("immutable scientific design boundary changed")
    return {
        "passed": True,
        "commit": commit,
        "design_is_ancestor": ancestor,
        "hashes": expected,
        "scientific_config_sha256": config_hash,
    }


def validate_implementation_boundary(config: dict) -> dict:
    boundary = config["implementation_boundary"]
    if boundary.get("status") != "anchored":
        raise RuntimeError("model stage requires an anchored implementation boundary")
    commit = str(boundary["commit"])
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode == 0
    paths = {
        "runner_sha256": EXP / "scripts" / "run.py",
        "model_ops_sha256": EXP / "src" / "model_ops.py",
        "coordinates_sha256": EXP / "src" / "coordinates.py",
        "branch_geometry_sha256": EXP / "src" / "branch_geometry.py",
        "mechanics_sha256": EXP / "src" / "mechanics.py",
        "task_data_sha256": EXP / "src" / "task_data.py",
        "tests_sha256": EXP / "tests" / "test_design.py",
        "implementation_audit_sha256": (
            EXP / "reports" / "pre_model_implementation_audit.md"
        ),
    }
    expected = {
        key: str(value) for key, value in boundary["hashes"].items()
        if key != "scientific_config_sha256"
    }
    local = {key: sha256(path) for key, path in paths.items()}
    committed = {}
    for key, path in paths.items():
        relative = path.relative_to(ROOT).as_posix()
        content = subprocess.check_output(["git", "show", f"{commit}:{relative}"], cwd=ROOT)
        committed[key] = hashlib.sha256(content).hexdigest()
    config_hash = scientific_config_sha256(config)
    if (
        not ancestor or local != expected or committed != expected
        or config_hash != boundary["hashes"].get("scientific_config_sha256")
    ):
        raise RuntimeError("anchored implementation bytes changed")
    return {
        "passed": True,
        "commit": commit,
        "implementation_is_ancestor": ancestor,
        "hashes": expected,
        "scientific_config_sha256": config_hash,
    }


def validate_mechanics_boundary(config: dict) -> dict:
    boundary = config["mechanics_boundary"]
    if boundary.get("status") != "anchored":
        raise RuntimeError("mechanics requires an anchored full-control boundary")
    commit = str(boundary["commit"])
    if subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
    ).returncode != 0:
        raise RuntimeError("mechanics boundary commit is not an ancestor")
    paths = {
        "model_smoke_summary_sha256": EXP / "runs" / "model_smoke" / "summary.json",
        "model_smoke_prefixes_sha256": EXP / "runs" / "model_smoke" / "prefixes.jsonl",
        "control_summary_sha256": EXP / "runs" / "control_calibration" / "summary.json",
        "control_numeric_rows_sha256": EXP / "runs" / "control_calibration" / "numeric_rows.jsonl",
        "control_positions_sha256": EXP / "runs" / "control_calibration" / "position_contracts.jsonl",
        "control_prefixes_sha256": EXP / "runs" / "control_calibration" / "prefixes.jsonl",
        "control_interventions_sha256": EXP / "runs" / "control_calibration" / "intervention_rows.jsonl",
    }
    expected = {key: str(value) for key, value in boundary["hashes"].items()}
    observed = {key: sha256(path) for key, path in paths.items()}
    committed = {}
    for key, path in paths.items():
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT
        )
        committed[key] = hashlib.sha256(content).hexdigest()
    if observed != expected or committed != expected:
        raise RuntimeError("full-control mechanics boundary bytes changed")
    model_smoke = json.loads(paths["model_smoke_summary_sha256"].read_text())
    control = json.loads(paths["control_summary_sha256"].read_text())
    numeric = read_jsonl(paths["control_numeric_rows_sha256"])
    prefixes = read_jsonl(paths["control_prefixes_sha256"])
    interventions = read_jsonl(paths["control_interventions_sha256"])
    if not (
        model_smoke.get("passed") is True
        and control.get("passed") is True
        and control.get("decision") == "CONTROL_CALIBRATION_PASS"
        and int(control.get("numeric_rows", -1)) == 880
        and len(numeric) == 880
        and all(row.get("passed") is True for row in numeric)
        and len(prefixes) == 4
        and control.get("intervention_rows_pass") is True
        and interventions
        and all(row.get("passed") is True for row in interventions)
        and control.get("outcomes_loaded") is False
        and control.get("correct_alias_loaded") is False
        and control.get("logits_recorded") is False
        and control.get("probabilities_recorded") is False
    ):
        raise RuntimeError("full-control receipt does not authorize mechanics")
    return {
        "passed": True,
        "commit": commit,
        "hashes": expected,
        "control_numeric_rows": len(numeric),
        "prefix_tasks": len(prefixes),
    }


def diagnostic_results(values: list[int], k: int) -> dict[str, list[int]]:
    """Frozen one-step mechanics consequences, independent of task gold."""

    shift = k % len(values)
    return {
        "reverse": values[::-1],
        "sort_asc": sorted(values),
        "sort_desc": sorted(values, reverse=True),
        "abs_all": [abs(value) for value in values],
        "square": [value * value for value in values],
        "negate": [-value for value in values],
        "running_sum": [sum(values[: index + 1]) for index in range(len(values))],
        "adjacent_diff": [
            values[index + 1] - values[index] for index in range(len(values) - 1)
        ],
        "add_k": [value + k for value in values],
        "mul_k": [value * k for value in values],
        "take_k": values[:k],
        "rotate_k": values[shift:] + values[:shift],
    }


def validate_config(config: dict) -> dict:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    aliases = tuple(config["data"]["alias_tokens"])
    operations = tuple(config["data"]["operation_names"])
    labels = tuple(config["anchor"]["result_labels"])
    if len(aliases) != 12 or len(set(aliases)) != 12:
        raise RuntimeError("anchor requires 12 unique aliases")
    if len(operations) != 12 or len(set(operations)) != 12:
        raise RuntimeError("anchor requires 12 unique operations")
    if len(labels) != 12 or len(set(labels)) != 12 or set(labels) & set(aliases):
        raise RuntimeError("result labels must be 12 unique non-alias concepts")
    if tuple(config["lens"]["band"]) != (4, 5, 6, 7, 8):
        raise RuntimeError("frozen intervention band changed")
    results = diagnostic_results(
        [int(value) for value in config["anchor"]["diagnostic_input"]],
        int(config["anchor"]["diagnostic_parameter"]),
    )
    if tuple(results) != operations or len({tuple(value) for value in results.values()}) != 12:
        raise RuntimeError("diagnostic consequences are incomplete or non-unique")
    return {
        "aliases": aliases,
        "operations": operations,
        "result_labels": labels,
        "diagnostic_results": results,
    }


def smoke() -> dict:
    config = yaml.safe_load(CONFIG.read_text())
    validated = validate_config(config)
    design = validate_design_boundary(config)
    lens_path = EXP / config["lens"]["path"]
    observed = sha256(lens_path)
    if observed != config["lens"]["sha256"]:
        raise RuntimeError("frozen context lens hash changed")
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    concepts = tuple(str(value) for value in lens["concepts"])
    if concepts[:12] != validated["aliases"]:
        raise RuntimeError("public alias order differs from frozen lens")
    if concepts[12:] != validated["result_labels"]:
        raise RuntimeError("result-label order differs from frozen lens")
    band = tuple(int(value) for value in config["lens"]["band"])
    if not set(band).issubset(int(value) for value in lens["source_layers"]):
        raise RuntimeError("frozen lens lacks an intervention layer")
    ranks = {
        str(layer): int(torch.linalg.matrix_rank(lens["directions"][layer].float()).item())
        for layer in band
    }
    if any(rank != 24 for rank in ranks.values()):
        raise RuntimeError("frozen context lens lost full concept rank")
    splits = build_splits(config)
    ancestors = ancestor_behavior_fingerprints()
    overlap = {
        split: sorted(
            behavior_fingerprint(task)
            for task in rows
            if behavior_fingerprint(task) in ancestors
        )
        for split, rows in splits.items()
    }
    if any(overlap.values()):
        raise RuntimeError(f"fresh task behavior overlaps ancestor data: {overlap}")
    data_dir = EXP / "data" / "procedural"
    paths = {}
    for split, rows in splits.items():
        path = data_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        paths[split] = path
    public_rows = [public_mechanics(task) for task in splits["mechanics"]]
    public_path = data_dir / "mechanics_public.jsonl"
    write_jsonl(public_path, public_rows)
    allowed_public = {
        "task_id", "visible", "alias_to_operation", "source_alias",
        "result_label_by_operation",
    }
    if any(set(row) != allowed_public for row in public_rows):
        raise RuntimeError("mechanics public schema leaks a sealed task field")
    manifest = {
        "schema_version": 1,
        "seed": int(config["seeds"]["split"]),
        "ancestor_behavior_fingerprints": len(ancestors),
        "ancestor_overlap_count": 0,
        "all_disjoint": True,
        "total_new_unique_behaviors": len({
            behavior_fingerprint(task) for rows in splits.values() for task in rows
        }),
        "total_new_unique_fingerprints": len({
            task_fingerprint(task) for rows in splits.values() for task in rows
        }),
        "splits": {
            split: {
                "rows": len(rows),
                "path": str(paths[split].relative_to(ROOT)),
                "sha256": sha256(paths[split]),
                "unique_behaviors": len({behavior_fingerprint(task) for task in rows}),
                "unique_fingerprints": len({task_fingerprint(task) for task in rows}),
            }
            for split, rows in splits.items()
        },
        "mechanics_public": {
            "rows": len(public_rows),
            "path": str(public_path.relative_to(ROOT)),
            "sha256": sha256(public_path),
            "fields": sorted(allowed_public),
            "sealed_fields_present": False,
        },
        "benchmarks_read": False,
        "scientific_result": False,
    }
    write_json(data_dir / "manifest.json", manifest)
    result = {
        "schema_version": 1,
        "stage": "cpu_smoke",
        "passed": True,
        "model_loaded": False,
        "outcomes_loaded": False,
        "benchmarks_read": False,
        "lens_sha256": observed,
        "lens_rank_by_layer": ranks,
        "aliases": list(validated["aliases"]),
        "result_labels": list(validated["result_labels"]),
        "diagnostic_results": validated["diagnostic_results"],
        "mechanics_rows_planned": int(config["data"]["mechanics_tasks"]) * 11,
        "fresh_task_manifest_sha256": sha256(data_dir / "manifest.json"),
        "ancestor_behavior_fingerprints": len(ancestors),
        "ancestor_overlap_count": 0,
        "fresh_split_rows": {split: len(rows) for split, rows in splits.items()},
        "design_boundary": design,
        "implementation_boundary_status": config["implementation_boundary"]["status"],
        "downstream_available": False,
    }
    path = EXP / "runs" / "smoke" / "cpu.json"
    write_json(path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def stable_seed(base: int, *parts: str) -> int:
    payload = "\0".join((str(base), *parts)).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (2**31)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def consequence_suffix(task: dict, config: dict) -> str:
    results = diagnostic_results(
        [int(value) for value in config["anchor"]["diagnostic_input"]],
        int(config["anchor"]["diagnostic_parameter"]),
    )
    rows = "\n".join(
        f"{result!r} = {task['result_label_by_operation'][operation]}"
        for operation, result in results.items()
    )
    return "\n\nResult label table:\n" + rows + "\n\n" + config["anchor"]["consequence_query_text"]


def probe_suffix(task: dict, config: dict, probe: str) -> str:
    if probe == "direct":
        return str(config["anchor"]["direct_query_text"])
    if probe == "consequence":
        return consequence_suffix(task, config)
    raise ValueError(f"unknown probe {probe!r}")


def load_model_context(config: dict):
    implementation = validate_implementation_boundary(config)
    from model_ops import ContextLens, QwenClampModel

    model = QwenClampModel(config)
    lens = ContextLens.load(str(EXP / config["lens"]["path"]))
    expected = tuple(config["data"]["alias_tokens"] + config["anchor"]["result_labels"])
    if lens.concepts != expected:
        raise RuntimeError("frozen lens concept order differs from configured contracts")
    live_ids = tuple(model.leading_space_token_id(concept) for concept in lens.concepts)
    if live_ids != lens.token_ids or len(set(live_ids)) != 24:
        raise RuntimeError("live one-token concept IDs differ from frozen lens")
    observed_think = (
        int(model.tokenizer.convert_tokens_to_ids("<think>")),
        int(model.tokenizer.convert_tokens_to_ids("</think>")),
    )
    expected_think = (
        int(config["model"]["think_open_id"]),
        int(config["model"]["think_close_id"]),
    )
    if observed_think != expected_think:
        raise RuntimeError("live native-thinking token IDs changed")
    band = tuple(int(value) for value in config["lens"]["band"])
    directions = {layer: lens.directions[layer] for layer in band}
    return model, lens, band, directions, implementation


def generate_public_prefix(model, task: dict, config: dict, *, task_index: int) -> tuple[dict, dict]:
    native = model.prepare_native_prompt(
        task_prompt(task), max_length=int(config["generation"]["prompt_max_tokens"])
    )
    trace = model.generate_native_prefix(
        native,
        seed=int(config["seeds"]["mechanics_prefix"]) + task_index,
        tokens=int(config["generation"]["prefix_tokens"]),
        temperature=float(config["generation"]["temperature"]),
        top_p=float(config["generation"]["top_p"]),
        top_k=int(config["generation"]["top_k"]),
    )
    return native, trace


def prepare_anchor(model, native: dict, trace: dict, task: dict, config: dict, *, alias: str, probe: str):
    return model.prepare_anchor_context(
        native,
        trace["token_ids"],
        prefix_text=str(config["anchor"]["prefix_text"]),
        anchor_alias=alias,
        suffix_text=probe_suffix(task, config, probe),
        max_length=int(config["generation"]["total_max_tokens"]),
    )


def control_bases(reference_delta, directions, *, config: dict, task_id: str, source: str, target: str, probe: str, arm: str, layer: int):
    from coordinates import orthogonal_norm_matched

    values = []
    for draw in range(int(config["controls"]["live_control_candidate_draws"])):
        generator = torch.Generator().manual_seed(
            stable_seed(
                int(config["seeds"][arm]), task_id, probe, source, target,
                arm, str(layer), str(draw),
            )
        )
        values.append(orthogonal_norm_matched(
            reference_delta.float(), directions.float(), generator=generator,
            rtol=float(config["lens"]["pseudoinverse_rtol"]),
        ))
    return torch.cat(values, dim=0)


def numeric_control_patcher(model, source_prepared: dict, reference_deltas: dict, directions: dict, *, config: dict, task_id: str, source: str, target: str, probe: str, arm: str):
    from model_ops import QuantizationAwareOrthogonalPatcher

    bases = {
        layer: control_bases(
            reference_deltas[layer], directions[layer], config=config,
            task_id=task_id, source=source, target=target, probe=probe,
            arm=arm, layer=layer,
        )
        for layer in directions
    }
    controls = config["controls"]
    return QuantizationAwareOrthogonalPatcher(
        model.layers,
        source_prepared["position"],
        bases,
        directions,
        {layer: float(reference_deltas[layer].float().norm()) for layer in directions},
        rtol=float(config["lens"]["pseudoinverse_rtol"]),
        norm_tolerance=float(controls["post_bf16_norm_relative_tolerance"]),
        projection_tolerance=float(controls["post_bf16_non_j_span_projection_max"]),
        correction_iterations=int(controls["live_control_correction_iterations"]),
        correction_damping=float(controls["live_control_correction_damping"]),
        binary_search_steps=int(controls["live_control_binary_search_steps"]),
        lattice_pair_steps=int(controls["live_control_lattice_pair_steps"]),
        repair_safety_margin=float(controls["live_control_repair_safety_margin"]),
    )


def numeric_rows(patcher, *, task_id: str, source: str, target: str, probe: str, arm: str) -> list[dict]:
    return [
        {
            "task_id": task_id,
            "source_alias": source,
            "target_alias": target,
            "probe": probe,
            "arm": arm,
            "layer": int(layer),
            "passed": bool(patcher.passed_by_layer[layer]),
            "j_delta_norm": float(patcher.target_norms_by_layer[layer]),
            "control_delta_norm": float(patcher.deltas[layer].float().norm()),
            "norm_relative_error": float(patcher.norm_errors[layer]),
            "realized_span_projection_fraction": float(patcher.projection_fractions[layer]),
            "chosen_candidate_index": int(patcher.chosen_indices[layer]),
            "correction_iterations": int(patcher.iterations_used[layer]),
            "lattice_pair_steps": int(patcher.lattice_pair_steps[layer]),
        }
        for layer in sorted(patcher.deltas)
    ]


def tensor_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def intervention_rows_for(
    patcher,
    *,
    task_id: str,
    source: str,
    target: str,
    probe: str,
    arm: str,
    allow_zero: bool = False,
) -> list[dict]:
    rows = []
    for layer in sorted(patcher.deltas):
        delta = patcher.deltas[layer].float()
        norm = float(delta.norm())
        finite = bool(torch.isfinite(delta).all())
        exact_once = int(patcher.applications[layer]) == 1
        rows.append({
            "task_id": task_id,
            "source_alias": source,
            "target_alias": target,
            "probe": probe,
            "arm": arm,
            "layer": int(layer),
            "delta_norm": norm,
            "finite": finite,
            "exact_once": exact_once,
            "zero_allowed": allow_zero,
            "passed": bool(finite and exact_once and (allow_zero or norm > 0.0)),
        })
    return rows


def run_numeric_stage(config: dict, *, full: bool) -> dict:
    """Outcome-blind live-bf16 validation; no logits or task labels are written."""

    import time
    import transformers
    from branch_geometry import balanced_j_branches
    from model_ops import AddDeltaPatcher, CoordinateClampPatcher, FullActivationPatcher

    design = validate_design_boundary(config)
    started = time.perf_counter()
    model, lens, band, directions, implementation = load_model_context(config)
    public_path = EXP / "data" / "procedural" / "mechanics_public.jsonl"
    public_rows = read_jsonl(public_path)
    if sha256(public_path) != design["hashes"]["mechanics_public_sha256"]:
        raise RuntimeError("public mechanics bytes changed after design anchor")
    selected_tasks = public_rows if full else public_rows[:1]
    rows: list[dict] = []
    position_contracts = []
    causal_differences = []
    j_nonzero = []
    full_nonzero = []
    intervention_rows: list[dict] = []
    donor_immutable = True
    prefixes = []
    locked_smoke_prefix = None
    model_smoke_sha = None
    if full:
        smoke_summary_path = EXP / "runs" / "model_smoke" / "summary.json"
        smoke_prefix_path = EXP / "runs" / "model_smoke" / "prefixes.jsonl"
        smoke_summary = json.loads(smoke_summary_path.read_text())
        smoke_prefixes = read_jsonl(smoke_prefix_path)
        if (
            smoke_summary.get("passed") is not True
            or smoke_summary.get("decision") != "MODEL_SMOKE_PASS"
            or smoke_summary.get("implementation_boundary", {}).get("commit")
            != implementation["commit"]
            or smoke_summary.get("design_boundary", {}).get("commit") != design["commit"]
            or len(smoke_prefixes) != 1
        ):
            raise RuntimeError("model-smoke receipt does not match current boundaries")
        locked_smoke_prefix = smoke_prefixes[0]
        model_smoke_sha = sha256(smoke_summary_path)
    for task_index, task in enumerate(selected_tasks):
        native, trace = generate_public_prefix(model, task, config, task_index=task_index)
        prefixes.append({
            "task_id": task["task_id"],
            "prompt_tokens": int(native["sequence_tokens"]),
            "thought_tokens": int(trace["tokens"]),
            "thought_sha256": hashlib.sha256(
                json.dumps(trace["token_ids"], separators=(",", ":")).encode()
            ).hexdigest(),
            "thought_token_ids": trace["token_ids"],
        })
        if task_index == 0 and locked_smoke_prefix is not None:
            if prefixes[-1] != locked_smoke_prefix:
                raise RuntimeError("full control prefix differs from locked model-smoke prefix")
        source = str(task["source_alias"])
        aliases = list(config["data"]["alias_tokens"])
        targets = [alias for alias in aliases if alias != source]
        if not full:
            targets = targets[:1]
        captures = {}
        prepared = {}
        for probe in ("direct", "consequence"):
            for alias in aliases:
                value = prepare_anchor(
                    model, native, trace, task, config, alias=alias, probe=probe
                )
                prepared[(probe, alias)] = value
                captures[(probe, alias)] = model.capture(
                    value, layers=band, retain_logits=False
                )
        capture_hashes = {
            (probe, alias, layer): tensor_sha256(
                captures[(probe, alias)]["activations"][layer]
            )
            for probe in ("direct", "consequence")
            for alias in aliases
            for layer in band
        }
        for alias in aliases:
            for layer in band:
                causal_differences.append(float((
                    captures[("direct", alias)]["activations"][layer]
                    - captures[("consequence", alias)]["activations"][layer]
                ).abs().max()))
        additive_banks = {
            layer: balanced_j_branches(
                directions[layer],
                public_concepts=int(config["lens"]["public_alias_concepts"]),
                target_rms_norm=float(config["lens"]["replicated_median_delta_norms"][layer]),
            )
            for layer in band
        }
        logit_dictionary = model.lm_head.weight[list(lens.token_ids)].float().T.detach().cpu()
        logit_directions = {layer: logit_dictionary for layer in band}
        for probe in ("direct", "consequence"):
            all_coordinates = [
                model.donor_coordinates(
                    captures[(probe, alias)]["activations"], directions,
                    rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                for alias in aliases
            ]
            mean_desired = {
                layer: torch.stack([value[layer] for value in all_coordinates]).mean(dim=0)
                for layer in band
            }
            mean_patcher = CoordinateClampPatcher(
                model.layers,
                prepared[(probe, source)]["position"],
                directions,
                mean_desired,
                rtol=float(config["lens"]["pseudoinverse_rtol"]),
            )
            model.apply_without_retaining_logits(prepared[(probe, source)], patcher=mean_patcher)
            intervention_rows.extend(intervention_rows_for(
                mean_patcher, task_id=str(task["task_id"]), source=source,
                target="__mean_all12__", probe=probe, arm="mean_donor_j",
            ))
        full_targets = [alias for alias in aliases if alias != source]
        wrong_by_target = {
            target: full_targets[(index + 1) % len(full_targets)]
            for index, target in enumerate(full_targets)
        }
        for target in targets:
            for probe in ("direct", "consequence"):
                source_prepared = prepared[(probe, source)]
                target_prepared = prepared[(probe, target)]
                source_capture = captures[(probe, source)]
                target_capture = captures[(probe, target)]
                position_contracts.append({
                    "task_id": task["task_id"],
                    "target_alias": target,
                    "probe": probe,
                    "position_equal": source_prepared["position"] == target_prepared["position"],
                    "length_equal": source_prepared["sequence_tokens"] == target_prepared["sequence_tokens"],
                    "source_anchor_id": source_prepared["anchor_id"],
                    "target_anchor_id": target_prepared["anchor_id"],
                    "source_position": source_prepared["position"],
                    "target_position": target_prepared["position"],
                    "source_close_position": source_prepared["think_close_position"],
                    "target_close_position": target_prepared["think_close_position"],
                    "whole_scaffold_tokenization_pass": bool(
                        source_prepared["whole_scaffold_tokenization_pass"]
                        and target_prepared["whole_scaffold_tokenization_pass"]
                    ),
                    "probe_suffix_sha256": hashlib.sha256(
                        probe_suffix(task, config, probe).encode()
                    ).hexdigest(),
                })
                full_patcher = FullActivationPatcher(
                    model.layers,
                    source_prepared["position"],
                    {layer: target_capture["activations"][layer] for layer in band},
                )
                model.apply_without_retaining_logits(source_prepared, patcher=full_patcher)
                full_nonzero.extend(float(value.float().norm()) for value in full_patcher.deltas.values())
                intervention_rows.extend(intervention_rows_for(
                    full_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="full_donor", allow_zero=True,
                ))
                if max(float(value.float().norm()) for value in full_patcher.deltas.values()) <= 0.0:
                    raise RuntimeError("full donor intervention is zero at every layer")
                desired = model.donor_coordinates(
                    target_capture["activations"], directions,
                    rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                j_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], directions, desired,
                    rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                model.apply_without_retaining_logits(source_prepared, patcher=j_patcher)
                reference = dict(j_patcher.deltas)
                j_nonzero.extend(float(value.float().norm()) for value in reference.values())
                intervention_rows.extend(intervention_rows_for(
                    j_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="donor_j",
                ))
                target_index = aliases.index(target)
                additive = {
                    layer: additive_banks[layer][:, target_index] for layer in band
                }
                additive_patcher = AddDeltaPatcher(
                    model.layers, source_prepared["position"], additive
                )
                model.apply_without_retaining_logits(
                    source_prepared, patcher=additive_patcher
                )
                intervention_rows.extend(intervention_rows_for(
                    additive_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="additive_j",
                ))
                wrong = wrong_by_target[target]
                wrong_desired = model.donor_coordinates(
                    captures[(probe, wrong)]["activations"], directions,
                    rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                wrong_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], directions,
                    wrong_desired, rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                model.apply_without_retaining_logits(
                    source_prepared, patcher=wrong_patcher
                )
                intervention_rows.extend(intervention_rows_for(
                    wrong_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm=f"wrong_donor_j:{wrong}",
                ))
                logit_desired = model.donor_coordinates(
                    target_capture["activations"], logit_directions,
                    rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                logit_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], logit_directions,
                    logit_desired, rtol=float(config["lens"]["pseudoinverse_rtol"]),
                )
                model.apply_without_retaining_logits(
                    source_prepared, patcher=logit_patcher
                )
                intervention_rows.extend(intervention_rows_for(
                    logit_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="logit_lens_all24",
                ))
                for arm in ("non_j_a", "non_j_b"):
                    patcher = numeric_control_patcher(
                        model, source_prepared, reference, directions, config=config,
                        task_id=str(task["task_id"]), source=source, target=target,
                        probe=probe, arm=arm,
                    )
                    model.apply_without_retaining_logits(source_prepared, patcher=patcher)
                    rows.extend(numeric_rows(
                        patcher, task_id=str(task["task_id"]), source=source, target=target,
                        probe=probe, arm=arm,
                    ))
        donor_immutable = donor_immutable and all(
            tensor_sha256(captures[(probe, alias)]["activations"][layer])
            == capture_hashes[(probe, alias, layer)]
            for probe in ("direct", "consequence")
            for alias in aliases
            for layer in band
        )
    controls = config["controls"]
    expected = len(selected_tasks) * (11 if full else 1) * 2 * 2 * len(band)
    causal_max = max(causal_differences, default=float("inf"))
    numeric_pass = bool(
        len(rows) == expected
        and all(row["passed"] for row in rows)
        and max(row["norm_relative_error"] for row in rows) <= float(
            controls["post_bf16_norm_relative_tolerance"]
        )
        and max(row["realized_span_projection_fraction"] for row in rows) <= float(
            controls["post_bf16_non_j_span_projection_max"]
        )
    )
    passed = bool(
        numeric_pass
        and all(
            row["position_equal"] and row["length_equal"]
            and row["whole_scaffold_tokenization_pass"]
            for row in position_contracts
        )
        and causal_max <= float(controls["causal_activation_atol"])
        and min(j_nonzero, default=0.0) > 0.0
        and max(full_nonzero, default=0.0) > 0.0
        and all(row["passed"] for row in intervention_rows)
        and donor_immutable
    )
    stage = "control_calibration" if full else "model_smoke"
    summary = {
        "schema_version": 1,
        "stage": stage,
        "passed": passed,
        "decision": (
            "CONTROL_CALIBRATION_PASS" if full and passed
            else "MODEL_SMOKE_PASS" if passed
            else "CONTROL_CALIBRATION_FAIL" if full
            else "MODEL_SMOKE_FAIL"
        ),
        "scientific_result": False,
        "model_id": config["model"]["id"],
        "model_revision": config["model"]["revision"],
        "dtype": config["model"]["dtype"],
        "attention": config["model"]["attention"],
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_device": (
            torch.cuda.get_device_name(torch.cuda.current_device())
            if torch.cuda.is_available() else None
        ),
        "batch_size": 1,
        "native_prefix_use_cache": True,
        "anchor_mechanics_use_cache": False,
        "public_data_path": str(public_path.relative_to(ROOT)),
        "public_data_sha256": sha256(public_path),
        "scientific_config_sha256": scientific_config_sha256(config),
        "lens_sha256": sha256(EXP / config["lens"]["path"]),
        "model_smoke_sha256": model_smoke_sha,
        "tasks": len(selected_tasks),
        "targets_per_task": 11 if full else 1,
        "probes": ["direct", "consequence"],
        "control_arms": ["non_j_a", "non_j_b"],
        "numeric_rows": len(rows),
        "expected_numeric_rows": expected,
        "numeric_pass": numeric_pass,
        "max_norm_relative_error": max(row["norm_relative_error"] for row in rows),
        "max_realized_span_projection_fraction": max(
            row["realized_span_projection_fraction"] for row in rows
        ),
        "max_lattice_pair_steps": max(row["lattice_pair_steps"] for row in rows),
        "causal_activation_max_abs": causal_max,
        "position_contract_pass": all(
            row["position_equal"] and row["length_equal"]
            and row["whole_scaffold_tokenization_pass"]
            for row in position_contracts
        ),
        "minimum_j_delta_norm": min(j_nonzero),
        "maximum_full_donor_delta_norm": max(full_nonzero),
        "intervention_rows": len(intervention_rows),
        "intervention_rows_pass": all(row["passed"] for row in intervention_rows),
        "donor_tensors_immutable": donor_immutable,
        "outcomes_loaded": False,
        "correct_alias_loaded": False,
        "first_op_loaded": False,
        "hidden_loaded": False,
        "target_pipeline_loaded": False,
        "logits_recorded": False,
        "probabilities_recorded": False,
        "confirmation_opened": False,
        "prefixes": prefixes,
        "design_boundary": design,
        "implementation_boundary": implementation,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    directory = EXP / "runs" / stage
    write_json(directory / "summary.json", summary)
    write_jsonl(directory / "numeric_rows.jsonl", rows)
    write_jsonl(directory / "position_contracts.jsonl", position_contracts)
    write_jsonl(directory / "prefixes.jsonl", prefixes)
    write_jsonl(directory / "intervention_rows.jsonl", intervention_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not passed:
        raise RuntimeError(f"{stage} failed closed")
    return summary


def run_mechanics(config: dict) -> dict:
    """Run public diagnostic mechanics only after the full numeric firewall."""

    import time
    from branch_geometry import balanced_j_branches
    from mechanics import ARMS, evaluate, scored_row, wrong_derangement
    from model_ops import AddDeltaPatcher, CoordinateClampPatcher, FullActivationPatcher

    design = validate_design_boundary(config)
    mechanics_boundary = validate_mechanics_boundary(config)
    started = time.perf_counter()
    model, lens, band, directions, implementation = load_model_context(config)
    public_path = EXP / "data" / "procedural" / "mechanics_public.jsonl"
    tasks = read_jsonl(public_path)
    allowed = {
        "task_id", "visible", "alias_to_operation", "source_alias",
        "result_label_by_operation",
    }
    if len(tasks) != 4 or any(set(task) != allowed for task in tasks):
        raise RuntimeError("mechanics public firewall changed")
    if sha256(public_path) != design["hashes"]["mechanics_public_sha256"]:
        raise RuntimeError("mechanics public bytes changed")
    locked_prefixes = {
        row["task_id"]: row
        for row in read_jsonl(EXP / "runs" / "control_calibration" / "prefixes.jsonl")
    }
    if set(locked_prefixes) != {task["task_id"] for task in tasks}:
        raise RuntimeError("locked prefix task set differs from public mechanics")
    aliases = list(config["data"]["alias_tokens"])
    result_labels = list(config["anchor"]["result_labels"])
    rtol = float(config["lens"]["pseudoinverse_rtol"])
    additive_banks = {
        layer: balanced_j_branches(
            directions[layer],
            public_concepts=int(config["lens"]["public_alias_concepts"]),
            target_rms_norm=float(config["lens"]["replicated_median_delta_norms"][layer]),
        )
        for layer in band
    }
    logit_dictionary = model.lm_head.weight[list(lens.token_ids)].float().T.detach().cpu()
    logit_directions = {layer: logit_dictionary for layer in band}
    outcome_rows: list[dict] = []
    numeric: list[dict] = []
    interventions: list[dict] = []
    prefix_receipts = []
    donor_immutable = True
    for task in tasks:
        native = model.prepare_native_prompt(
            task_prompt(task), max_length=int(config["generation"]["prompt_max_tokens"])
        )
        locked = locked_prefixes[str(task["task_id"])]
        token_ids = [int(value) for value in locked["thought_token_ids"]]
        observed_hash = hashlib.sha256(
            json.dumps(token_ids, separators=(",", ":")).encode()
        ).hexdigest()
        if (
            len(token_ids) != int(config["generation"]["prefix_tokens"])
            or int(native["sequence_tokens"]) != int(locked["prompt_tokens"])
            or observed_hash != locked["thought_sha256"]
        ):
            raise RuntimeError("locked native prefix receipt changed")
        trace = {"token_ids": token_ids}
        prefix_receipts.append({
            "task_id": task["task_id"],
            "prompt_tokens": int(native["sequence_tokens"]),
            "thought_tokens": len(token_ids),
            "thought_sha256": observed_hash,
            "locked_match": True,
        })
        prepared = {}
        captures = {}
        for probe in ("direct", "consequence"):
            for alias in aliases:
                value = prepare_anchor(
                    model, native, trace, task, config, alias=alias, probe=probe
                )
                prepared[(probe, alias)] = value
                captures[(probe, alias)] = model.capture(
                    value, layers=band, retain_logits=False
                )
        capture_hashes = {
            (probe, alias, layer): tensor_sha256(
                captures[(probe, alias)]["activations"][layer]
            )
            for probe in ("direct", "consequence")
            for alias in aliases
            for layer in band
        }
        source = str(task["source_alias"])
        targets = [alias for alias in aliases if alias != source]
        wrong_by_target = wrong_derangement(source, aliases)
        for probe in ("direct", "consequence"):
            registered = aliases if probe == "direct" else result_labels
            source_prepared = prepared[(probe, source)]
            source_score = model.score(source_prepared)
            all_coordinates = [
                model.donor_coordinates(
                    captures[(probe, alias)]["activations"], directions, rtol=rtol
                )
                for alias in aliases
            ]
            mean_desired = {
                layer: torch.stack([value[layer] for value in all_coordinates]).mean(dim=0)
                for layer in band
            }
            mean_patcher = CoordinateClampPatcher(
                model.layers, source_prepared["position"], directions,
                mean_desired, rtol=rtol,
            )
            mean_score = model.score(source_prepared, patcher=mean_patcher)
            interventions.extend(intervention_rows_for(
                mean_patcher, task_id=str(task["task_id"]), source=source,
                target="__mean_all12__", probe=probe, arm="mean_donor_j",
            ))
            for target in targets:
                wrong = wrong_by_target[target]
                target_capture = captures[(probe, target)]
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="source", score=source_score, registered_tokens=registered,
                ))
                text_score = model.score(prepared[(probe, target)])
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="text_target", score=text_score, registered_tokens=registered,
                ))
                full_patcher = FullActivationPatcher(
                    model.layers, source_prepared["position"],
                    {layer: target_capture["activations"][layer] for layer in band},
                )
                full_score = model.score(source_prepared, patcher=full_patcher)
                interventions.extend(intervention_rows_for(
                    full_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="full_donor", allow_zero=True,
                ))
                if max(float(value.float().norm()) for value in full_patcher.deltas.values()) <= 0.0:
                    raise RuntimeError("mechanics full donor is zero at every layer")
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="full_donor", score=full_score, registered_tokens=registered,
                ))
                desired = model.donor_coordinates(
                    target_capture["activations"], directions, rtol=rtol
                )
                j_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], directions, desired, rtol=rtol
                )
                j_score = model.score(source_prepared, patcher=j_patcher)
                reference = dict(j_patcher.deltas)
                interventions.extend(intervention_rows_for(
                    j_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="donor_j",
                ))
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="donor_j", score=j_score, registered_tokens=registered,
                ))
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="mean_donor_j", score=mean_score, registered_tokens=registered,
                ))
                target_index = aliases.index(target)
                additive_patcher = AddDeltaPatcher(
                    model.layers,
                    source_prepared["position"],
                    {layer: additive_banks[layer][:, target_index] for layer in band},
                )
                additive_score = model.score(source_prepared, patcher=additive_patcher)
                interventions.extend(intervention_rows_for(
                    additive_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="additive_j",
                ))
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="additive_j", score=additive_score, registered_tokens=registered,
                ))
                for arm in ("non_j_a", "non_j_b"):
                    control = numeric_control_patcher(
                        model, source_prepared, reference, directions, config=config,
                        task_id=str(task["task_id"]), source=source, target=target,
                        probe=probe, arm=arm,
                    )
                    control_score = model.score(source_prepared, patcher=control)
                    numeric.extend(numeric_rows(
                        control, task_id=str(task["task_id"]), source=source,
                        target=target, probe=probe, arm=arm,
                    ))
                    outcome_rows.append(scored_row(
                        model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                        arm=arm, score=control_score, registered_tokens=registered,
                    ))
                wrong_desired = model.donor_coordinates(
                    captures[(probe, wrong)]["activations"], directions, rtol=rtol
                )
                wrong_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], directions,
                    wrong_desired, rtol=rtol,
                )
                wrong_score = model.score(source_prepared, patcher=wrong_patcher)
                interventions.extend(intervention_rows_for(
                    wrong_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm=f"wrong_donor_j:{wrong}",
                ))
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="wrong_donor_j", score=wrong_score, registered_tokens=registered,
                ))
                logit_desired = model.donor_coordinates(
                    target_capture["activations"], logit_directions, rtol=rtol
                )
                logit_patcher = CoordinateClampPatcher(
                    model.layers, source_prepared["position"], logit_directions,
                    logit_desired, rtol=rtol,
                )
                logit_score = model.score(source_prepared, patcher=logit_patcher)
                interventions.extend(intervention_rows_for(
                    logit_patcher, task_id=str(task["task_id"]), source=source,
                    target=target, probe=probe, arm="logit_lens_all24",
                ))
                outcome_rows.append(scored_row(
                    model, task, target_alias=target, wrong_alias=wrong, probe=probe,
                    arm="logit_lens_all24", score=logit_score,
                    registered_tokens=registered,
                ))
        donor_immutable = donor_immutable and all(
            tensor_sha256(captures[(probe, alias)]["activations"][layer])
            == capture_hashes[(probe, alias, layer)]
            for probe in ("direct", "consequence")
            for alias in aliases
            for layer in band
        )
    calibration_numeric = read_jsonl(
        EXP / "runs" / "control_calibration" / "numeric_rows.jsonl"
    )
    calibration_interventions = read_jsonl(
        EXP / "runs" / "control_calibration" / "intervention_rows.jsonl"
    )
    numeric_key = lambda row: (
        row["task_id"], row["target_alias"], row["probe"], row["arm"], row["layer"]
    )
    intervention_key = lambda row: (
        row["task_id"], row["target_alias"], row["probe"], row["arm"], row["layer"]
    )
    numeric_matches = sorted(numeric, key=numeric_key) == sorted(
        calibration_numeric, key=numeric_key
    )
    interventions_match = sorted(interventions, key=intervention_key) == sorted(
        calibration_interventions, key=intervention_key
    )
    firewall_pass = bool(
        len(numeric) == 880
        and all(row["passed"] for row in numeric)
        and all(row["passed"] for row in interventions)
        and numeric_matches
        and interventions_match
        and donor_immutable
    )
    if not firewall_pass:
        failure = {
            "schema_version": 1,
            "stage": "mechanics",
            "passed": False,
            "decision": "INVALID_MECHANICS_CONTROL",
            "outcomes_recorded": False,
            "probabilities_recorded": False,
            "numeric_rows": len(numeric),
            "numeric_matches_calibration": numeric_matches,
            "interventions_match_calibration": interventions_match,
            "donor_tensors_immutable": donor_immutable,
        }
        write_json(EXP / "runs" / "mechanics_invalid_control.json", failure)
        raise RuntimeError("mechanics numeric firewall failed before outcome retention")
    evaluation = evaluate(outcome_rows, numeric, interventions, config)
    summary = {
        "schema_version": 1,
        "stage": "mechanics",
        **evaluation,
        "scientific_result": True,
        "task_correctness_loaded": False,
        "correct_alias_loaded": False,
        "first_op_loaded": False,
        "hidden_loaded": False,
        "target_pipeline_loaded": False,
        "confirmation_opened": False,
        "outcomes_recorded": True,
        "probabilities_recorded": True,
        "numeric_matches_calibration": numeric_matches,
        "interventions_match_calibration": interventions_match,
        "donor_tensors_immutable": donor_immutable,
        "prefixes": prefix_receipts,
        "arms": list(ARMS),
        "public_data_sha256": sha256(public_path),
        "scientific_config_sha256": scientific_config_sha256(config),
        "design_boundary": design,
        "implementation_boundary": implementation,
        "mechanics_boundary": mechanics_boundary,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    directory = EXP / "runs" / "mechanics"
    write_json(directory / "summary.json", summary)
    write_jsonl(directory / "outcome_rows.jsonl", outcome_rows)
    write_jsonl(directory / "numeric_rows.jsonl", numeric)
    write_jsonl(directory / "intervention_rows.jsonl", interventions)
    write_jsonl(directory / "prefixes.jsonl", prefix_receipts)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "smoke", "model-smoke", "control-calibration", "mechanics",
            "qualification", "confirmation",
        ),
    )
    args = parser.parse_args()

    if args.stage == "smoke":
        smoke()
        return 0
    config = yaml.safe_load(CONFIG.read_text())
    if args.stage == "model-smoke":
        run_numeric_stage(config, full=False)
        return 0
    if args.stage == "control-calibration":
        smoke_path = EXP / "runs" / "model_smoke" / "summary.json"
        smoke_prefix_path = EXP / "runs" / "model_smoke" / "prefixes.jsonl"
        if (
            not smoke_path.is_file()
            or not smoke_prefix_path.is_file()
            or json.loads(smoke_path.read_text()).get("passed") is not True
        ):
            raise RuntimeError("control calibration requires a passing model-smoke receipt")
        run_numeric_stage(config, full=True)
        return 0
    if args.stage == "mechanics":
        run_mechanics(config)
        return 0
    raise RuntimeError(
        f"stage {args.stage!r} is unavailable before an audited implementation boundary"
    )


if __name__ == "__main__":
    sys.exit(main())
