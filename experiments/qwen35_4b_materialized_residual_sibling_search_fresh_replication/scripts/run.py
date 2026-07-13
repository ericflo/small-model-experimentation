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
from identity import (  # noqa: E402
    EXPERIMENT_ID,
    PARENT_LINEAGE,
    REQUEST_NAMESPACE,
    SCIENTIFIC_PARENT_ID,
    TASK_NAMESPACE,
    canonical_sha256 as identity_sha256,
    public_instance_fingerprint,
    verified_manifest_file,
    verify_parent_lineage,
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
    "src/identity.py",
    "src/plans.py",
    "src/protocol.py",
    "src/stats.py",
    "src/task_data.py",
    "src/vllm_runner.py",
    "tests/test_plans_stats.py",
    "tests/test_identity.py",
    "tests/test_freshness.py",
    "tests/test_protocol.py",
    "tests/test_task_data.py",
    "tests/test_vllm_runner.py",
    "tests/test_scaffold.py",
)

PARENT_EXP = ROOT / "experiments" / SCIENTIFIC_PARENT_ID
PARENT_CONSTRUCTION_KEYS = {
    "common_panel",
    "mechanics_public",
    "mechanics_gold",
    "mechanics_audit",
    "qualification_public",
    "qualification_gold",
    "qualification_audit",
    "confirmation_public",
    "confirmation_gold",
    "confirmation_audit",
}
PARENT_PREPARED_KEYS = {
    "direct_requests.jsonl",
    "listwise_requests.jsonl",
    "random_scores.jsonl",
    "suffix_echo_requests.jsonl",
    "suffix_materialized_requests.jsonl",
    "suffix_name_only_requests.jsonl",
    "suffix_shuffled_requests.jsonl",
    "surface_folds.json",
    "surface_scores.jsonl",
    "viability_materialized_requests.jsonl",
    "viability_name_only_requests.jsonl",
    "viability_shuffled_requests.jsonl",
}
PARENT_REQUEST_KEYS = {
    name for name in PARENT_PREPARED_KEYS if name.endswith("_requests.jsonl")
}


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
    if config.get("identity") != {
        "experiment_id": EXPERIMENT_ID,
        "scientific_parent_id": SCIENTIFIC_PARENT_ID,
        "task_namespace": TASK_NAMESPACE,
        "request_namespace": REQUEST_NAMESPACE,
        "scientific_parent_function_exemption": "descriptive_overlap_only",
        "public_instance_overlap_allowed": False,
        "model_prompt_overlap_allowed": False,
    }:
        raise RuntimeError("fresh identity contract changed")
    fresh_seeds = {int(value) for value in config["seeds"].values()}
    if fresh_seeds != set(range(2026072700, 2026072710)):
        raise RuntimeError("fresh ten-seed domain changed")
    if fresh_seeds & set(range(2026072600, 2026072610)):
        raise RuntimeError("fresh seed domain overlaps the scientific parent")
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
    if (
        config["boundaries"]["cpu_smoke"]["status"]
        != "authorized_model_free_smoke"
    ):
        raise RuntimeError("model-free construction smoke is not authorized")
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


def _verified_unrelated_prior_path(
    path: Path, *, experiments_root: Path | None = None
) -> Path:
    base = experiments_root or (ROOT / "experiments")
    try:
        relative = path.relative_to(base)
    except ValueError as error:
        raise RuntimeError("unrelated prior path escaped experiments/") from error
    cursor = base
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise RuntimeError(f"unrelated prior path contains a symlink: {path}")
    if not path.is_file():
        raise RuntimeError(f"unrelated prior path is absent or non-regular: {path}")
    try:
        path.resolve().relative_to(base.resolve())
    except ValueError as error:
        raise RuntimeError("unrelated prior resolved outside experiments/") from error
    return path


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
        if PARENT_EXP in path.parents:
            continue
        path = _verified_unrelated_prior_path(path)
        read_paths.append(str(path.relative_to(ROOT)))
        for row in _strict_jsonl(path):
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
        "scientific_parent_exempted_from_function_rejection": True,
        "scientific_parent": SCIENTIFIC_PARENT_ID,
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
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
        use_fast=True,
    )
    model_config = AutoConfig.from_pretrained(
        config["model"]["id"],
        revision=config["model"]["revision"],
        trust_remote_code=True,
    )
    hf_model_eos_id = int(model_config.text_config.eos_token_id)
    tokenizer_eos_id = int(tokenizer.eos_token_id)
    if (
        hf_model_eos_id != 248044
        or tokenizer_eos_id != 248046
        or tokenizer.eos_token != "<|im_end|>"
        or tokenizer.encode("<|endoftext|>", add_special_tokens=False) != [248044]
        or tokenizer.encode("<|im_end|>", add_special_tokens=False) != [248046]
    ):
        raise RuntimeError("pinned model/tokenizer termination identity changed")
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
        "termination": {
            "hf_model_eos_token": "<|endoftext|>",
            "hf_model_eos_token_id": hf_model_eos_id,
            "tokenizer_eos_token": tokenizer.eos_token,
            "tokenizer_eos_token_id": tokenizer_eos_id,
            "ignore_eos": True,
            "explicit_stop_token_ids": [hf_model_eos_id],
            "trim_only_token_id": hf_model_eos_id,
            "preserve_tokenizer_eos_during_trim": True,
        },
        "conditions": condition_receipts,
        "prompt_count": sum(row["prompt_count"] for row in condition_receipts.values()),
        "prompt_tokens_min": min(all_lengths),
        "prompt_tokens_max": max(all_lengths),
        "model_loaded": False,
    }


def _strict_jsonl(path: Path, *, expected_rows: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line:
            raise RuntimeError(f"blank line in authenticated JSONL {path}:{line_number}")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"malformed authenticated JSONL {path}:{line_number}"
            ) from error
        if not isinstance(value, dict):
            raise RuntimeError(
                f"non-object authenticated JSONL row {path}:{line_number}"
            )
        rows.append(value)
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError(
            f"authenticated row count drift for {path}: {len(rows)} != {expected_rows}"
        )
    return rows


def _authenticated_parent_inputs(
    lineage: dict[str, dict[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, set[str]], dict[str, Any]]:
    manifest_path = ROOT / lineage["construction_manifest"]["path"]
    manifest = json.loads(manifest_path.read_text())
    construction_files = manifest.get("paths")
    if not isinstance(construction_files, dict) or set(construction_files) != PARENT_CONSTRUCTION_KEYS:
        raise RuntimeError("parent construction inventory changed")

    public_rows: list[dict[str, Any]] = []
    gold_rows: list[dict[str, Any]] = []
    construction_receipt: dict[str, dict[str, Any]] = {}
    for name, row in sorted(construction_files.items()):
        path = verified_manifest_file(ROOT, row)
        construction_receipt[name] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": row["sha256"],
        }
        if name.endswith("_public"):
            public_rows.extend(_strict_jsonl(path))
        elif name.endswith("_gold"):
            gold_rows.extend(_strict_jsonl(path))
    if len(public_rows) != 264 or len(gold_rows) != 264:
        raise RuntimeError("parent construction task count changed")
    if {row.get("task_id") for row in public_rows} != {
        row.get("task_id") for row in gold_rows
    }:
        raise RuntimeError("parent public/gold task identities do not align")

    preoutcome_path = ROOT / lineage["original_preoutcome"]["path"]
    preoutcome = json.loads(preoutcome_path.read_text())
    prepared_files = preoutcome.get("files")
    if not isinstance(prepared_files, dict) or set(prepared_files) != PARENT_PREPARED_KEYS:
        raise RuntimeError("parent prepared inventory changed")
    prompt_hashes: dict[str, set[str]] = {}
    prepared_receipt: dict[str, dict[str, Any]] = {}
    for name, row in sorted(prepared_files.items()):
        if not isinstance(row, dict) or set(row) != {"path", "rows", "sha256"}:
            raise RuntimeError(f"parent prepared row schema changed: {name}")
        path = verified_manifest_file(
            ROOT, {"path": row["path"], "sha256": row["sha256"]}
        )
        prepared_receipt[name] = {
            "path": str(path.relative_to(ROOT)),
            "rows": int(row["rows"]),
            "sha256": row["sha256"],
        }
        if name in PARENT_REQUEST_KEYS:
            requests = _strict_jsonl(path, expected_rows=int(row["rows"]))
            hashes: set[str] = set()
            for request in requests:
                if not isinstance(request.get("id"), str) or not isinstance(
                    request.get("messages"), list
                ):
                    raise RuntimeError(f"parent request schema changed: {name}")
                hashes.add(identity_sha256(request["messages"]))
            prompt_hashes[name] = hashes
    return public_rows, gold_rows, prompt_hashes, {
        "construction_files": construction_receipt,
        "prepared_files": prepared_receipt,
    }


def _fresh_mechanics_prompt_hashes(
    mechanics: list[dict[str, Any]],
) -> dict[str, set[str]]:
    result = {name: set() for name in PARENT_REQUEST_KEYS}

    def add(name: str, prompt: str) -> None:
        result[name].add(
            identity_sha256([{"role": "user", "content": prompt}])
        )

    for task in mechanics:
        public = public_task(task)
        add("direct_requests.jsonl", direct_prompt(public))
        add("listwise_requests.jsonl", listwise_prompt(public))
        for candidate in CONCRETE_OPERATIONS:
            for representation in ("materialized", "name_only", "shuffled"):
                add(
                    f"suffix_{representation}_requests.jsonl",
                    suffix_prompt(
                        public,
                        candidate=candidate,
                        representation=representation,
                    ),
                )
                add(
                    f"viability_{representation}_requests.jsonl",
                    viability_prompt(
                        public,
                        candidate=candidate,
                        representation=representation,
                    ),
                )
        for live_row in task["public_live"]:
            candidate = operation_from_record(live_row["operation"])
            suffix = tuple(
                operation_from_record(value)
                for value in live_row["first_fitting_suffix"]
            )
            add(
                "suffix_echo_requests.jsonl",
                suffix_prompt(
                    public,
                    candidate=candidate,
                    representation="echo",
                    supplied_suffix=suffix,
                ),
            )
    return result


def _prompt_overlap_counts(
    parent_prompts: dict[str, set[str]], fresh_prompts: dict[str, set[str]]
) -> dict[str, Any]:
    if set(parent_prompts) != PARENT_REQUEST_KEYS or set(fresh_prompts) != PARENT_REQUEST_KEYS:
        raise RuntimeError("prompt overlap inventory changed")
    if any(
        not isinstance(values, set)
        or any(not isinstance(value, str) or len(value) != 64 for value in values)
        for values in (*parent_prompts.values(), *fresh_prompts.values())
    ):
        raise RuntimeError("prompt overlap hashes have the wrong schema")
    by_same_file = {
        name: len(parent_prompts[name] & fresh_prompts[name])
        for name in sorted(PARENT_REQUEST_KEYS)
    }
    parent_union = set().union(*parent_prompts.values())
    fresh_union = set().union(*fresh_prompts.values())
    return {
        "by_same_request_file": by_same_file,
        "all_parent_vs_all_fresh": len(parent_union & fresh_union),
        "terminal_suffix_materialized": len(
            parent_prompts["suffix_materialized_requests.jsonl"]
            & fresh_prompts["suffix_materialized_requests.jsonl"]
        ),
    }


def _parent_freshness_receipt(
    splits: dict[str, list[dict[str, Any]]],
    common: list[list[int]],
    lineage: dict[str, dict[str, str]],
) -> dict[str, Any]:
    parent_public, parent_gold, parent_prompts, file_receipt = (
        _authenticated_parent_inputs(lineage)
    )
    fresh_tasks = [task for rows in splits.values() for task in rows]
    parent_task_ids = {str(row["task_id"]) for row in parent_public}
    fresh_task_ids = {str(row["task_id"]) for row in fresh_tasks}
    task_id_overlap = parent_task_ids & fresh_task_ids

    parent_instances = {public_instance_fingerprint(row) for row in parent_public}
    fresh_instances = {
        public_instance_fingerprint(public_task(row)) for row in fresh_tasks
    }
    public_instance_overlap = parent_instances & fresh_instances

    parent_triples = {
        tuple(operation_from_record(value) for value in row["target_pipeline"])
        for row in parent_gold
    }
    fresh_triples = {
        tuple(operation_from_record(value) for value in row["target_pipeline"])
        for row in fresh_tasks
    }
    parent_suffixes = {tuple(value[1:]) for value in parent_triples}
    fresh_suffixes = {tuple(value[1:]) for value in fresh_triples}
    parent_functions_on_fresh_panel: set[str] = set()
    invalid_parent_functions = 0
    for triple in parent_triples:
        outputs = valid_output_vector(triple, common)
        if outputs is None:
            invalid_parent_functions += 1
        else:
            parent_functions_on_fresh_panel.add(behavior_fingerprint(outputs))
    fresh_functions = {str(row["common_fingerprint"]) for row in fresh_tasks}

    fresh_prompts = _fresh_mechanics_prompt_hashes(splits["mechanics"])
    prompt_overlap = _prompt_overlap_counts(parent_prompts, fresh_prompts)
    required_zero = {
        "task_ids": len(task_id_overlap),
        "public_instance_payloads": len(public_instance_overlap),
        "model_facing_mechanics_prompts": prompt_overlap[
            "all_parent_vs_all_fresh"
        ],
        "terminal_suffix_materialized_prompts": prompt_overlap[
            "terminal_suffix_materialized"
        ],
    }
    _require_zero_freshness_intersections(required_zero)
    return {
        "schema_version": 1,
        "scientific_parent": SCIENTIFIC_PARENT_ID,
        "task_namespace": TASK_NAMESPACE,
        "request_namespace_reserved": REQUEST_NAMESPACE,
        "parent_seed_block": list(range(2026072600, 2026072610)),
        "fresh_seed_block": list(range(2026072700, 2026072710)),
        "required_zero_intersections": required_zero,
        "prompt_overlap": prompt_overlap,
        "descriptive_finite_dsl_reuse": {
            "parent_functions_re_evaluated_on_fresh_panel": len(
                parent_functions_on_fresh_panel
            ),
            "parent_functions_invalid_on_fresh_panel": invalid_parent_functions,
            "shared_behavior_functions": len(
                parent_functions_on_fresh_panel & fresh_functions
            ),
            "shared_concrete_triples": len(parent_triples & fresh_triples),
            "shared_two_operation_suffixes": len(parent_suffixes & fresh_suffixes),
            "rejection_policy": "reported_not_rejected",
        },
        "fresh_counts": {
            "tasks": len(fresh_tasks),
            "public_instance_payloads": len(fresh_instances),
            "mechanics_prompt_payloads_by_request_file": {
                name: len(values) for name, values in sorted(fresh_prompts.items())
            },
        },
        "authenticated_parent_inputs": file_receipt,
        "benchmarks_read": False,
    }


def _require_zero_freshness_intersections(required_zero: dict[str, int]) -> None:
    expected = {
        "task_ids",
        "public_instance_payloads",
        "model_facing_mechanics_prompts",
        "terminal_suffix_materialized_prompts",
    }
    if set(required_zero) != expected or any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in required_zero.values()
    ):
        raise RuntimeError("fresh identity intersection receipt has the wrong schema")
    if any(required_zero.values()):
        raise RuntimeError(f"fresh identity overlap gate failed: {required_zero}")


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
    parent_lineage = verify_parent_lineage(ROOT)
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
    freshness_receipt = _parent_freshness_receipt(
        splits, common, parent_lineage
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
        "parent_lineage": parent_lineage,
        "scientific_parent_freshness": freshness_receipt,
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
        "parent_required_zero_intersections": freshness_receipt[
            "required_zero_intersections"
        ],
        "parent_finite_dsl_reuse": freshness_receipt[
            "descriptive_finite_dsl_reuse"
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
