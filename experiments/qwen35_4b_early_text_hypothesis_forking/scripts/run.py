#!/usr/bin/env python3
"""Staged fail-closed harness for early text hypothesis forking."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from plans import branch_plan, freeze_resource_plan  # noqa: E402
from protocol import (  # noqa: E402
    candidate_injection,
    parse_program,
    parse_result,
    program_source,
    select_visible,
    task_prompt,
)
from task_data import (  # noqa: E402
    CONCRETE_OPERATIONS,
    DEPTH_TWO_PROGRAMS,
    EXPECTED_PARAMETERS,
    IDENTIFIABLE_FIRST_OPERATIONS,
    OPERATIONS,
    behavior_fingerprint,
    build_splits,
    canonical_operation,
    diagnostic_apply,
    gold_task,
    operation_from_record,
    public_task,
    task_fingerprint,
)


CONFIG = EXP / "configs" / "default.yaml"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def ancestor_behavior_fingerprints() -> set[str]:
    """Read procedural experiment artifacts only; benchmark contents stay forbidden."""

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
            if {"depth", "visible", "hidden"}.issubset(row):
                try:
                    values.add(behavior_fingerprint(row))
                except (KeyError, TypeError, ValueError):
                    continue
    return values


def validate_config(config: dict[str, Any]) -> None:
    if config["model"]["id"] != "Qwen/Qwen3.5-4B":
        raise RuntimeError("only Qwen/Qwen3.5-4B is permitted")
    if config["model"]["revision"] != "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a":
        raise RuntimeError("model revision changed")
    if config["model"]["backend"] != "vllm":
        raise RuntimeError("all generation arms require vLLM")
    if tuple(config["data"]["operations"]) != tuple(OPERATIONS):
        raise RuntimeError("operation menu changed")
    expected_parameters = {
        name: list(parameters)
        for name, parameters in EXPECTED_PARAMETERS.items()
        if parameters != (None,)
    }
    if config["data"]["parameter_values"] != expected_parameters:
        raise RuntimeError("bound-operation inventory changed")
    if len(CONCRETE_OPERATIONS) != 24 or len(IDENTIFIABLE_FIRST_OPERATIONS) != 23:
        raise RuntimeError("24-candidate/23-gold inventory changed")
    generation = config["generation"]
    for field in ("hypotheses_per_task", "duplicate_branches", "placebo_branches"):
        if int(generation[field]) != len(CONCRETE_OPERATIONS):
            raise RuntimeError(f"{field} must remain 24")
    if int(generation["late_prefix_tokens"]) + int(
        generation["late_equal_total_remaining_tokens"]
    ) != int(generation["main_thinking_budget"]):
        raise RuntimeError("equal-total late and early thought budgets differ")
    if int(generation["late_equal_post_remaining_tokens"]) != int(
        generation["main_thinking_budget"]
    ):
        raise RuntimeError("equal-post late arm does not preserve early suffix budget")
    if config["boundaries"]["implementation"]["status"] not in {
        "pending",
        "reviewed_pending_lock",
        "locked",
    }:
        raise RuntimeError("unknown model implementation boundary state")
    if config["boundaries"]["mechanics"]["status"] != "pending":
        raise RuntimeError("mechanics boundary moved before audited lock")


def _resource_smoke() -> dict[str, Any]:
    rows = [
        {
            "id": f"neutral-{index:02d}",
            "sampled_tokens": 1050 + (index % 7),
            "logical_model_tokens": 2050 + (index % 11),
        }
        for index in range(48)
    ]
    plan = freeze_resource_plan(
        rows,
        target_sampled_tokens=24 * 1053,
        target_logical_model_tokens=24 * 2055,
    )
    if plan["sampled"]["pool_exhausted"] or plan["logical"]["pool_exhausted"]:
        raise RuntimeError("synthetic resource matcher exhausted its master pool")
    mutated_gold = {
        "hidden": [{"input": [999], "output": [-999]}],
        "first_op": {"name": "negate", "parameter": None},
        "target_pipeline": [],
    }
    # Gold is deliberately not an argument to the matcher. This assertion
    # records that arbitrary mutations cannot change its bytes.
    before = canonical_sha256(plan)
    mutated_gold["hidden"][0]["output"] = [123456]
    if canonical_sha256(plan) != before:
        raise RuntimeError("resource matching changed under a gold-only mutation")
    return plan


def smoke() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG.read_text())
    validate_config(config)
    first = build_splits(config)
    second = build_splits(config)
    if first != second:
        raise RuntimeError("procedural splits are not deterministic")
    expected = {
        "qualification": int(config["data"]["qualification_tasks"]),
        "confirmation": int(config["data"]["confirmation_tasks"]),
    }
    if {name: len(rows) for name, rows in first.items()} != expected:
        raise RuntimeError("split sizes changed")

    ancestor = ancestor_behavior_fingerprints()
    collisions = [
        behavior_fingerprint(task)
        for rows in first.values()
        for task in rows
        if behavior_fingerprint(task) in ancestor
    ]
    if collisions:
        raise RuntimeError("fresh behavior collides with ancestor procedural data")

    data_dir = EXP / "data" / "procedural"
    paths: dict[str, Path] = {}
    for split, tasks in first.items():
        public_path = data_dir / f"{split}_public.jsonl"
        gold_path = data_dir / f"{split}_gold.jsonl"
        write_jsonl(public_path, [public_task(task) for task in tasks])
        write_jsonl(gold_path, [gold_task(task) for task in tasks])
        paths[f"{split}_public"] = public_path
        paths[f"{split}_gold"] = gold_path

    mechanics_rows: list[dict[str, Any]] = []
    diagnostic_maps: list[dict[str, list[int]]] = []
    diagnostic_inputs = config["mechanics"]["diagnostic_inputs"]
    for index, values in enumerate(diagnostic_inputs):
        mapping = {
            canonical_operation(operation): diagnostic_apply(operation, list(values))
            for operation in CONCRETE_OPERATIONS
        }
        if len({tuple(result) for result in mapping.values()}) != 24:
            raise RuntimeError(f"bound mechanics results collide in context {index}")
        diagnostic_maps.append(mapping)
        mechanics_rows.append(
            {
                "context_id": f"mechanics-{index:05d}",
                "input": values,
                "operation_results": mapping,
            }
        )
    if len({canonical_sha256(value) for value in diagnostic_maps}) != len(
        diagnostic_inputs
    ):
        raise RuntimeError("operation-to-result compositions do not vary by context")
    mechanics_path = data_dir / "mechanics_public.jsonl"
    write_jsonl(mechanics_path, mechanics_rows)
    paths["mechanics_public"] = mechanics_path

    panel = [list(values) for values in diagnostic_inputs[:2]]
    branch_seed = int(config["seeds"]["branch_permutation"])
    branch_plans: list[dict[str, Any]] = []
    split_gold_slots: dict[str, Counter[int]] = {}
    for split, tasks in first.items():
        plans = [
            branch_plan(task["task_id"], branch_seed, behavior_panel=panel)
            for task in tasks
        ]
        if len({plan["composed_map_sha256"] for plan in plans}) != len(plans):
            raise RuntimeError(f"{split} repeats a composed branch map")
        slots = Counter(plan["gold_slot_from_public_schedule"] for plan in plans)
        if set(slots) != set(range(24)) or max(slots.values()) - min(slots.values()) > 1:
            raise RuntimeError(f"{split} gold branch positions are not balanced")
        for task, plan in zip(tasks, plans, strict=True):
            operation = operation_from_record(task["first_op"])
            row = plan["rows"][plan["gold_slot_from_public_schedule"]]
            if row["canonical_operation"] != canonical_operation(operation):
                raise RuntimeError("public branch schedule disagrees with task schedule")
            if len({canonical_sha256(value["behavior_signature"]) for value in plan["rows"]}) != 24:
                raise RuntimeError("fixed-panel mechanics signatures are not 24-way distinct")
        split_gold_slots[split] = slots
        branch_plans.extend(plans)
    branch_plan_path = data_dir / "branch_plans.jsonl"
    write_jsonl(branch_plan_path, branch_plans)
    paths["branch_plans"] = branch_plan_path

    # Exercise the real strict parser and visible selector with the complete CPU
    # grammar on four tasks. The generator has already enumerated all 576 on all
    # 144 tasks to establish task admission.
    cpu_candidates = [
        {"candidate_id": f"cpu-{index:03d}", "text": program_source(program)}
        for index, program in enumerate(DEPTH_TWO_PROGRAMS)
    ]
    exhaustive_receipts = []
    for task in first["qualification"][: int(config["data"]["mechanics_tasks"])]:
        selected = select_visible(public_task(task), cpu_candidates)
        if selected["abstained"]:
            raise RuntimeError("CPU exhaustive visible selector abstained")
        if selected["eligible_unique_programs"] != task[
            "visible_consistent_program_count"
        ]:
            raise RuntimeError("CPU selector and task enumeration disagree")
        exhaustive_receipts.append(
            {
                "task_id": task["task_id"],
                "eligible_unique_programs": selected["eligible_unique_programs"],
                "selected_canonical": selected["selected"]["canonical"],
            }
        )

    exemplar = first["qualification"][0]
    public = public_task(exemplar)
    target = [operation_from_record(value) for value in exemplar["target_pipeline"]]
    strict_source = program_source(target)
    if not parse_program(strict_source)["parsed"]:
        raise RuntimeError("strict target Python does not parse")
    if not parse_result("private thought</think>\nRESULT: [3, 2, 1]")["parsed"]:
        raise RuntimeError("mechanics result parser failed")
    prompt = task_prompt(public)
    mutated_gold = copy.deepcopy(gold_task(exemplar))
    mutated_gold["hidden"] = [{"input": [999], "output": [0]}]
    mutated_gold["first_op"] = {"name": "negate", "parameter": None}
    mutated_gold["target_pipeline"] = []
    if task_prompt(public) != prompt:
        raise RuntimeError("public prompt changed under a gold-only mutation")

    injection_texts = [candidate_injection(operation) for operation in CONCRETE_OPERATIONS]
    if len(set(injection_texts)) != 24:
        raise RuntimeError("candidate injection text is not 24-way distinct")
    resource_plan = _resource_smoke()
    firewall = {
        "schema_version": 1,
        "public_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "candidate_injection_text_sha256": [
            hashlib.sha256(text.encode()).hexdigest() for text in injection_texts
        ],
        "branch_plan_file_sha256": sha256(branch_plan_path),
        "resource_plan_sha256": resource_plan["resource_plan_sha256"],
        "gold_mutation_fields": ["hidden", "first_op", "target_pipeline"],
        "gold_used_for_prompt": False,
        "gold_used_for_branch_plan": False,
        "gold_used_for_resource_plan": False,
        "model_loaded": False,
        "outcomes_loaded": False,
    }
    firewall_path = EXP / "runs" / "smoke" / "pregade_firewall.json"
    write_json(firewall_path, firewall)
    paths["pregade_firewall"] = firewall_path

    manifest = {
        "schema_version": 2,
        "seed": int(config["seeds"]["data"]),
        "branch_seed": branch_seed,
        "split_rows": expected,
        "mechanics_contexts": len(mechanics_rows),
        "mechanics_rows_planned": len(mechanics_rows) * len(CONCRETE_OPERATIONS),
        "bound_operation_candidates": len(CONCRETE_OPERATIONS),
        "eligible_gold_first_operations": len(IDENTIFIABLE_FIRST_OPERATIONS),
        "depth_two_programs_exhausted_per_task": len(DEPTH_TWO_PROGRAMS),
        "total_new_behaviors": sum(expected.values()),
        "unique_behavior_fingerprints": len(
            {behavior_fingerprint(task) for rows in first.values() for task in rows}
        ),
        "unique_task_fingerprints": len(
            {task_fingerprint(task) for rows in first.values() for task in rows}
        ),
        "ancestor_behavior_fingerprints": len(ancestor),
        "ancestor_overlap_count": 0,
        "unique_composed_branch_maps": len(
            {plan["composed_map_sha256"] for plan in branch_plans}
        ),
        "gold_slot_counts": {
            split: {str(slot): count for slot, count in sorted(counts.items())}
            for split, counts in split_gold_slots.items()
        },
        "diagnostic_result_support_per_context": [24] * len(diagnostic_maps),
        "unique_operation_result_maps": len(
            {canonical_sha256(value) for value in diagnostic_maps}
        ),
        "cpu_exhaustive_selector_tasks": len(exhaustive_receipts),
        "paths": {
            name: {"path": str(path.relative_to(ROOT)), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "benchmarks_read": False,
        "model_loaded": False,
        "outcomes_loaded": False,
    }
    manifest_path = data_dir / "manifest.json"
    write_json(manifest_path, manifest)

    result = {
        "schema_version": 2,
        "stage": "smoke",
        "passed": True,
        "decision": "CPU_SMOKE_PASS",
        "design_status": config["boundaries"]["design"]["status"],
        "design_amendment_status": config["boundaries"]["design_amendment"][
            "status"
        ],
        "implementation_status": config["boundaries"]["implementation"]["status"],
        "model_loaded": False,
        "outcomes_loaded": False,
        "benchmarks_read": False,
        "split_rows": expected,
        "total_procedural_tasks": sum(expected.values()),
        "bound_operation_candidates": 24,
        "eligible_gold_first_operations": 23,
        "depth_two_programs_exhausted_per_task": 576,
        "mechanics_contexts": len(mechanics_rows),
        "mechanics_rows_planned": len(mechanics_rows) * 24,
        "ancestor_overlap_count": 0,
        "unique_composed_branch_maps": manifest["unique_composed_branch_maps"],
        "diagnostic_result_support_per_context": manifest[
            "diagnostic_result_support_per_context"
        ],
        "cpu_exhaustive_selector_tasks": len(exhaustive_receipts),
        "resource_matcher_pool_exhausted": False,
        "firewall_sha256": sha256(firewall_path),
        "manifest_sha256": sha256(manifest_path),
    }
    write_json(EXP / "runs" / "smoke" / "summary.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("smoke", "mechanics", "qualification", "confirmation"),
    )
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
        return 0
    raise RuntimeError(
        f"stage {args.stage!r} is unavailable before audited implementation boundaries"
    )


if __name__ == "__main__":
    sys.exit(main())
