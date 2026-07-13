#!/usr/bin/env python3
"""Fail-closed staged harness for materialized residual sibling search."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from plans import (  # noqa: E402
    freeze_taskwise_matches,
    pool_cost,
    strict_taskwise_dominance,
    top4_cost,
)
from protocol import (  # noqa: E402
    direct_prompt,
    listwise_prompt,
    parse_program,
    select_visible,
    suffix_prompt,
    target_derangement,
    viability_prompt,
)
from stats import simulate_compound_power  # noqa: E402
from task_data import (  # noqa: E402
    ALIASES,
    CONCRETE_OPERATIONS,
    INVALID,
    alias_program,
    apply_pipeline,
    audit_task,
    behavior_fingerprint,
    build_common_panel,
    build_splits,
    canonical_operation,
    count_live_operations,
    gold_task,
    operation_from_record,
    public_task,
    valid_output_vector,
    validate_splits,
)


CONFIG = EXP / "configs" / "default.yaml"
DESIGN_LOCK_FILES = (
    "README.md",
    "configs/default.yaml",
    "reports/preregistration.md",
    "reports/design_review.md",
    "scripts/run.py",
    "src/plans.py",
    "src/protocol.py",
    "src/stats.py",
    "src/task_data.py",
    "src/vllm_runner.py",
    "tests/test_plans_stats.py",
    "tests/test_protocol.py",
    "tests/test_task_data.py",
    "tests/test_vllm_runner.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _design_lock_hashes() -> dict[str, str]:
    return {relative: _sha256(EXP / relative) for relative in DESIGN_LOCK_FILES}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    )
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if not line:
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _validate_config(config: dict[str, Any]) -> None:
    if config["model"] != {
        "id": "Qwen/Qwen3.5-4B",
        "revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "dtype": "bfloat16",
        "backend": "vllm",
        "vllm_version": "0.24.0+cu129",
    }:
        raise RuntimeError("the pinned model/backend identity changed")
    if tuple(config["aliases"]["operations"]) != ALIASES:
        raise RuntimeError("the fixed A-X alias map changed")
    configured = config["data"]["operations"]
    for parameters in configured.values():
        if not isinstance(parameters, list):
            raise RuntimeError("operation configuration must use lists")
    # parameter-free is a presentation group rather than an operation name.
    parameter_free = [(name, None) for name in configured["parameter_free"]]
    expanded = parameter_free + [
        (name, value)
        for name, values in configured.items()
        if name != "parameter_free"
        for value in values
    ]
    if tuple(expanded) != CONCRETE_OPERATIONS:
        raise RuntimeError("configured operation bank changed")
    data = config["data"]
    if (
        int(data["common_panel_per_length"])
        * (int(data["input_max_length"]) - int(data["input_min_length"]) + 1)
        != int(data["common_fingerprint_panel_inputs"])
    ):
        raise RuntimeError("common-panel geometry changed")
    if config["generation"]["ranking_thinking"] != "off":
        raise RuntimeError("ranking must remain no-think")
    if config["generation"]["logprobs_mode"] != "raw_logprobs":
        raise RuntimeError("ranking requires raw log probabilities")
    if config["top4_secondary"] != {
        "primary_gate_dependency": "none",
        "inferential_status": "descriptive_operational_only",
        "failure_seals": "top4_secondary_only",
        "qualification": {
            "coverage_noninferiority_margin": 0.10,
            "coverage_gain_vs_surface_min": 0.05,
            "sampled_tokens_must_be_below_all24_each_task": True,
            "logical_tokens_must_be_below_all24_each_task": True,
        },
        "confirmation": {
            "coverage_noninferiority_margin": 0.10,
            "coverage_gain_vs_surface_min": 0.05,
            "sampled_tokens_must_be_below_all24_each_task": True,
            "logical_tokens_must_be_below_all24_each_task": True,
        },
    }:
        raise RuntimeError("top-four secondary changed or became a primary veto")
    if config["boundaries"]["design"]["status"] != "adversarial_review_passed":
        raise RuntimeError("design review boundary is not passed")
    if config["boundaries"]["cpu_smoke"]["status"] != "passed_model_free":
        raise RuntimeError("CPU smoke boundary is not passed")
    if config["boundaries"]["mechanics_implementation"]["status"] != "absent":
        raise RuntimeError("mechanics moved before a published implementation lock")


def _compatible_pipeline(row: dict[str, Any]) -> tuple[tuple[str, int | None], ...] | None:
    value = row.get("target_pipeline")
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return tuple(operation_from_record(item) for item in value)
    except (TypeError, ValueError):
        return None


def prior_function_fingerprints(
    common: list[list[int]],
) -> tuple[set[str], dict[str, Any]]:
    """Read experiment procedural artifacts only; benchmarks stay forbidden."""

    patterns = (
        "*/data/procedural/*.jsonl",
        "*/data/**/*gold*.jsonl",
        "*/data/**/*oracle*.jsonl",
    )
    paths = {
        path
        for pattern in patterns
        for path in (ROOT / "experiments").glob(pattern)
        if EXP not in path.parents
    }
    fingerprints: set[str] = set()
    compatible_rows = 0
    read_paths: list[str] = []
    for path in sorted(paths):
        read_paths.append(str(path.relative_to(ROOT)))
        for row in _read_jsonl(path):
            pipeline = _compatible_pipeline(row)
            if pipeline is None:
                continue
            outputs = valid_output_vector(pipeline, common)
            if outputs is None:
                continue
            compatible_rows += 1
            fingerprints.add(behavior_fingerprint(outputs))
    return fingerprints, {
        "paths_read": read_paths,
        "path_count": len(read_paths),
        "compatible_depth_three_rows": compatible_rows,
        "unique_compatible_function_fingerprints": len(fingerprints),
        "benchmarks_read": False,
    }


def _audit_task_ids(splits: dict[str, list[dict[str, Any]]]) -> set[str]:
    result = {task["task_id"] for task in splits["mechanics"]}
    for split in ("qualification", "confirmation"):
        for index in range(0, len(splits[split]), 24):
            result.add(splits[split][index]["task_id"])
    return result


def _protocol_smoke(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    receipts: list[dict[str, Any]] = []
    seen_strata: set[str] = set()
    for task in tasks:
        if task["stratum"] in seen_strata:
            continue
        seen_strata.add(task["stratum"])
        public = public_task(task)
        live_rows = task["public_live"]
        candidates: list[dict[str, Any]] = []
        for index, live_row in enumerate(live_rows):
            candidate = operation_from_record(live_row["operation"])
            suffix = tuple(
                operation_from_record(value)
                for value in live_row["first_fitting_suffix"]
            )
            text = f"PROGRAM: {alias_program(suffix)}"
            parsed = parse_program(text, arity=2)
            if not parsed["parsed"]:
                raise RuntimeError("known suffix did not parse")
            candidates.append(
                {
                    "candidate_id": f"cpu-{index}",
                    "candidate": candidate,
                    "text": text,
                }
            )
            for representation in ("materialized", "name_only", "shuffled"):
                prompt = suffix_prompt(
                    public, candidate=candidate, representation=representation
                )
                if "hidden" in prompt.lower() or "probe" in prompt.lower():
                    raise RuntimeError("suffix prompt mentions a sealed field")
            viability_prompt(public, candidate=candidate, representation="materialized")
            echo = suffix_prompt(
                public,
                candidate=candidate,
                representation="echo",
                supplied_suffix=suffix,
            )
            if f"PROGRAM: {alias_program(suffix)}" not in echo:
                raise RuntimeError("echo prompt omitted the supplied suffix")
        selection = select_visible(public, candidates)
        if selection["abstained"]:
            raise RuntimeError("known public-live witnesses produced selector abstention")
        targets = [row["output"] for row in public["visible"]]
        permutation = target_derangement(targets, salt=task["task_id"])
        if sorted(permutation) != list(range(len(targets))) or any(
            targets[index] == targets[source]
            for index, source in enumerate(permutation)
        ):
            raise RuntimeError("target derangement retained a value-level pair")
        target = tuple(
            operation_from_record(value) for value in task["target_pipeline"]
        )
        direct = f"PROGRAM: {alias_program(target)}"
        if not parse_program(direct, arity=3)["parsed"]:
            raise RuntimeError("known direct target did not parse")
        prompt_before = direct_prompt(public)
        mutated = copy.deepcopy(task)
        mutated["hidden"] = [{"input": [9, 9, 9, 9], "output": [-999]}]
        if direct_prompt(public_task(mutated)) != prompt_before:
            raise RuntimeError("direct prompt changed under hidden mutation")
        receipts.append(
            {
                "task_id": task["task_id"],
                "stratum": task["stratum"],
                "live_count": len(live_rows),
                "selected_candidate_id": selection["selected_candidate_id"],
                "derangement": list(permutation),
            }
        )
    if seen_strata != {"single", "double", "triple", "quad"}:
        raise RuntimeError("protocol smoke did not cover every live-count stratum")
    return {"tasks": receipts, "strata": sorted(seen_strata)}


def _synthetic_completion(index: int, *, scale: int = 1) -> dict[str, int]:
    return {
        "n_sampled_tokens": scale * (40 + index % 5),
        "n_stage1_prompt_tokens": scale * (300 + index % 7),
        "n_stage2_prompt_tokens": scale * (340 + index % 11),
    }


def _resource_smoke() -> dict[str, Any]:
    treatment = [_synthetic_completion(index) for index in range(24)]
    direct = [_synthetic_completion(index, scale=2) for index in range(128)]
    plan = freeze_taskwise_matches(
        task_id="synthetic", treatment_outputs=treatment, direct_outputs=direct
    )
    if plan["sampled"]["pool_exhausted"] or plan["logical"]["pool_exhausted"]:
        raise RuntimeError("synthetic direct pool exhausted")
    ranking = [_synthetic_completion(index, scale=0) for index in range(24)]
    # A one-token rank call cannot use the zero-scale helper literally.
    for output in ranking:
        output.update(
            n_sampled_tokens=1,
            n_stage1_prompt_tokens=350,
            n_stage2_prompt_tokens=0,
        )
    suffix = [_synthetic_completion(index) for index in range(4)]
    top = top4_cost(ranking_outputs=ranking, suffix_outputs=suffix)
    all_cost = pool_cost(treatment)
    if not strict_taskwise_dominance(top, all_cost):
        raise RuntimeError("synthetic top-four policy is not resource-dominant")
    return {"direct_match": plan, "top4": top, "all24": all_cost}


def _tokenizer_smoke(
    config: dict[str, Any], splits: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
        use_fast=True,
    )
    alias_ids: dict[str, dict[str, list[int]]] = {}
    for alias in ALIASES:
        plain = tokenizer.encode(alias, add_special_tokens=False)
        leading = tokenizer.encode(" " + alias, add_special_tokens=False)
        if len(plain) != 1 or len(leading) != 1:
            raise RuntimeError(f"alias {alias} is not single-token in both forms")
        alias_ids[alias] = {"plain": plain, "leading_space": leading}
    if len({value["plain"][0] for value in alias_ids.values()}) != 24:
        raise RuntimeError("plain alias token IDs collide")

    think_open_ids = tokenizer.encode("<think>", add_special_tokens=False)
    think_close_ids = tokenizer.encode("</think>", add_special_tokens=False)
    close_ids = tokenizer.encode("</think>\n\n", add_special_tokens=False)
    if think_open_ids != [248068] or think_close_ids != [248069]:
        raise RuntimeError("pinned think-token IDs changed")
    if not close_ids or close_ids[0] != think_close_ids[0]:
        raise RuntimeError("forced close sequence changed")
    thinking_suffix_ids = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n", add_special_tokens=False
    )
    no_thinking_suffix_ids = tokenizer.encode(
        "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        add_special_tokens=False,
    )

    suffix_reserve = (
        int(config["generation"]["suffix_thinking_budget"])
        + len(close_ids)
        + int(config["generation"]["suffix_answer_max_tokens"])
    )
    direct_reserve = (
        int(config["generation"]["direct_thinking_budget"])
        + len(close_ids)
        + int(config["generation"]["direct_answer_max_tokens"])
    )
    conditions: dict[str, dict[str, Any]] = {
        "direct": {"enable_thinking": True, "reserve_tokens": direct_reserve, "prompts": []},
        "suffix_materialized": {
            "enable_thinking": True,
            "reserve_tokens": suffix_reserve,
            "prompts": [],
        },
        "suffix_name_only": {
            "enable_thinking": True,
            "reserve_tokens": suffix_reserve,
            "prompts": [],
        },
        "suffix_shuffled": {
            "enable_thinking": True,
            "reserve_tokens": suffix_reserve,
            "prompts": [],
        },
        "suffix_echo_mechanics_live": {
            "enable_thinking": True,
            "reserve_tokens": suffix_reserve,
            "prompts": [],
        },
        "viability_materialized": {
            "enable_thinking": False,
            "reserve_tokens": 1,
            "prompts": [],
        },
        "viability_name_only": {
            "enable_thinking": False,
            "reserve_tokens": 1,
            "prompts": [],
        },
        "viability_shuffled": {
            "enable_thinking": False,
            "reserve_tokens": 1,
            "prompts": [],
        },
        "listwise": {"enable_thinking": False, "reserve_tokens": 1, "prompts": []},
    }
    for split_name, tasks in splits.items():
        for task in tasks:
            public = public_task(task)
            conditions["direct"]["prompts"].append(direct_prompt(public))
            conditions["listwise"]["prompts"].append(listwise_prompt(public))
            for candidate in CONCRETE_OPERATIONS:
                for representation in ("materialized", "name_only", "shuffled"):
                    conditions[f"suffix_{representation}"]["prompts"].append(
                        suffix_prompt(
                            public,
                            candidate=candidate,
                            representation=representation,
                        )
                    )
                    conditions[f"viability_{representation}"]["prompts"].append(
                        viability_prompt(
                            public,
                            candidate=candidate,
                            representation=representation,
                        )
                    )
            if split_name == "mechanics":
                for live_row in task["public_live"]:
                    candidate = operation_from_record(live_row["operation"])
                    suffix = tuple(
                        operation_from_record(value)
                        for value in live_row["first_fitting_suffix"]
                    )
                    conditions["suffix_echo_mechanics_live"]["prompts"].append(
                        suffix_prompt(
                            public,
                            candidate=candidate,
                            representation="echo",
                            supplied_suffix=suffix,
                        )
                    )

    condition_receipts: dict[str, dict[str, Any]] = {}
    all_lengths: list[int] = []
    max_model_len = int(config["generation"]["max_model_len"])
    for name, condition in conditions.items():
        raw_prompts = condition.pop("prompts")
        if not raw_prompts:
            raise RuntimeError(f"tokenizer condition {name} has no prompts")
        rendered_prompts: list[str] = []
        token_rows: list[list[int]] = []
        for prompt in raw_prompts:
            rendered = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=bool(condition["enable_thinking"]),
            )
            if not isinstance(rendered, str):
                raise RuntimeError("chat template did not return rendered text")
            token_ids = tokenizer.encode(rendered, add_special_tokens=False)
            expected_suffix = (
                thinking_suffix_ids
                if condition["enable_thinking"]
                else no_thinking_suffix_ids
            )
            if token_ids[-len(expected_suffix) :] != expected_suffix:
                raise RuntimeError(f"{name} rendered through the wrong prompt channel")
            rendered_prompts.append(rendered)
            token_rows.append(token_ids)
        lengths = [len(values) for values in token_rows]
        maximum_total = max(lengths) + int(condition["reserve_tokens"])
        if maximum_total > max_model_len:
            raise RuntimeError(f"{name} exceeds the frozen context")
        all_lengths.extend(lengths)
        condition_receipts[name] = {
            "prompt_count": len(raw_prompts),
            "enable_thinking": condition["enable_thinking"],
            "reserve_tokens": condition["reserve_tokens"],
            "prompt_tokens_min": min(lengths),
            "prompt_tokens_max": max(lengths),
            "max_prompt_plus_reserve": maximum_total,
            "context_slack_min": max_model_len - maximum_total,
            "user_prompt_text_sha256": _canonical_sha256(raw_prompts),
            "rendered_prompt_text_sha256": _canonical_sha256(rendered_prompts),
            "prompt_token_ids_sha256": _canonical_sha256(token_rows),
        }
    return {
        "alias_token_ids": alias_ids,
        "think_open_token_ids": think_open_ids,
        "think_close_token_ids": think_close_ids,
        "forced_close_token_ids": close_ids,
        "thinking_prompt_suffix_ids": thinking_suffix_ids,
        "no_thinking_prompt_suffix_ids": no_thinking_suffix_ids,
        "conditions": condition_receipts,
        "prompt_count": sum(row["prompt_count"] for row in condition_receipts.values()),
        "prompt_tokens_min": min(all_lengths),
        "prompt_tokens_max": max(all_lengths),
        "model_loaded": False,
    }


def _gate_attainability(config: dict[str, Any], mechanics_live_rows: int) -> dict[str, Any]:
    mechanics_tasks = int(config["data"]["mechanics_tasks"])
    qualification_tasks = int(config["data"]["qualification_tasks"])
    confirmation_tasks = int(config["data"]["confirmation_tasks"])
    return {
        "mechanics_live_rows": mechanics_live_rows,
        "suffix_parse_successes_min": math.ceil(
            config["mechanics"]["suffix_parse_rate_min"] * mechanics_live_rows
        ),
        "suffix_cap_contacts_max": math.floor(
            config["mechanics"]["suffix_answer_limit_contact_max"]
            * mechanics_live_rows
        ),
        "materialized_any_live_tasks_min": math.ceil(
            config["mechanics"]["materialized_live_suffix_visible_pass_min"]
            * mechanics_tasks
        ),
        "qualification_selected_min": math.ceil(
            config["qualification"]["materialized_selected_accuracy_min"]
            * qualification_tasks
        ),
        "qualification_coverage_min": math.ceil(
            config["qualification"]["materialized_proposal_coverage_min"]
            * qualification_tasks
        ),
        "qualification_gain_tasks_min": math.ceil(
            config["qualification"]["selected_accuracy_gain_min"]
            * qualification_tasks
        ),
        "confirmation_gain_tasks_min": math.ceil(
            config["confirmation"]["selected_accuracy_gain_min"]
            * confirmation_tasks
        ),
        "random_recall_at_4_expectation": 4 / 24,
    }


def smoke() -> dict[str, Any]:
    started = time.perf_counter()
    config = yaml.safe_load(CONFIG.read_text())
    _validate_config(config)
    common = build_common_panel(config)
    prior_fingerprints, prior_receipt = prior_function_fingerprints(common)
    splits, construction, rebuilt_common = build_splits(
        config, excluded_fingerprints=prior_fingerprints
    )
    if common != rebuilt_common:
        raise RuntimeError("common panel changed inside split construction")
    second, second_receipt, second_common = build_splits(
        config, excluded_fingerprints=prior_fingerprints
    )
    if (
        _canonical_sha256(splits) != _canonical_sha256(second)
        or construction != second_receipt
        or common != second_common
    ):
        raise RuntimeError("procedural construction is not deterministic")

    audit_ids = _audit_task_ids(splits)
    validate_splits(
        splits,
        config,
        common_inputs=common,
        exhaustive_live_task_ids=audit_ids,
    )
    protocol_receipt = _protocol_smoke(splits["mechanics"])
    resource_receipt = _resource_smoke()
    tokenizer_receipt = _tokenizer_smoke(config, splits)
    power_receipt = simulate_compound_power(
        config["confirmation"]["power_simulation"],
        seed=int(config["seeds"]["bootstrap"]),
    )
    if power_receipt["compound_pass_rate"] < 0.80:
        raise RuntimeError("compound confirmation design has inadequate target power")

    data_dir = EXP / "data" / "procedural"
    paths: dict[str, Path] = {}
    common_path = data_dir / "common_panel.json"
    _write_json(common_path, common)
    paths["common_panel"] = common_path
    for split, tasks in splits.items():
        for kind, transform in (
            ("public", public_task),
            ("gold", gold_task),
            ("audit", audit_task),
        ):
            path = data_dir / f"{split}_{kind}.jsonl"
            _write_jsonl(path, [transform(task) for task in tasks])
            paths[f"{split}_{kind}"] = path

    mechanics_live_rows = sum(
        len(task["public_live"]) for task in splits["mechanics"]
    )
    gate_receipt = _gate_attainability(config, mechanics_live_rows)
    fingerprint_count = len(
        {
            task["common_fingerprint"]
            for tasks in splits.values()
            for task in tasks
        }
    )
    total_tasks = sum(len(tasks) for tasks in splits.values())
    if fingerprint_count != total_tasks:
        raise RuntimeError("function fingerprints collide across splits")
    manifest = {
        "schema_version": 1,
        "construction": construction,
        "prior_art": prior_receipt,
        "split_sha256": _canonical_sha256(splits),
        "deterministic_rebuild_sha256": _canonical_sha256(second),
        "independent_live_audit_tasks": len(audit_ids),
        "total_tasks": total_tasks,
        "unique_common_function_fingerprints": fingerprint_count,
        "live_operation_support": dict(
            sorted(
                count_live_operations(
                    task for tasks in splits.values() for task in tasks
                ).items()
            )
        ),
        "gate_attainability": gate_receipt,
        "protocol": protocol_receipt,
        "resource": resource_receipt,
        "tokenizer": tokenizer_receipt,
        "power": power_receipt,
        "design_lock_file_sha256": _design_lock_hashes(),
        "paths": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
        },
        "benchmarks_read": False,
        "model_loaded": False,
        "model_calls": 0,
    }
    manifest_path = data_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    result = {
        "schema_version": 1,
        "stage": "smoke",
        "passed": True,
        "decision": "CPU_SMOKE_PASS",
        "total_tasks": total_tasks,
        "split_rows": construction["split_rows"],
        "eligible_function_fingerprints": construction[
            "eligible_function_fingerprints"
        ],
        "excluded_prior_fingerprints": construction[
            "excluded_prior_fingerprints"
        ],
        "independent_live_audit_tasks": len(audit_ids),
        "compound_power_at_design_alternative": power_receipt[
            "compound_pass_rate"
        ],
        "common_panel_sha256": construction["common_panel_sha256"],
        "manifest_sha256": _sha256(manifest_path),
        "elapsed_seconds": time.perf_counter() - started,
        "benchmarks_read": False,
        "model_loaded": False,
        "model_calls": 0,
    }
    summary_path = EXP / "runs" / "smoke" / "summary.json"
    _write_json(summary_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
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
        f"stage {args.stage!r} is sealed pending implementation review and lock"
    )


if __name__ == "__main__":
    sys.exit(main())
