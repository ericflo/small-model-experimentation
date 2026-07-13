#!/usr/bin/env python3
"""Fail-closed receipt checks shared by the downstream white-box analyzers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Iterable


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import harness  # noqa: E402
import repo_tasks  # noqa: E402

HISTORY_POLICY = "canonical_first_valid_tool_call"
HEX_DIGITS = frozenset("0123456789abcdef")


def fail(message: str) -> None:
    raise SystemExit(message)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"cannot read JSON receipt {path}: {exc}")
    if not isinstance(value, dict):
        fail(f"JSON receipt is not an object: {path}")
    return value


@lru_cache(maxsize=128)
def _sha256_file_cached(resolved_path: str, size: int, mtime_ns: int) -> str:
    del size, mtime_ns  # cache-key provenance; bytes still come from the path
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    resolved = path.resolve()
    stat = resolved.stat()
    return _sha256_file_cached(str(resolved), stat.st_size, stat.st_mtime_ns)


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(HEX_DIGITS)
    )


def finite_rate(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        fail(f"{label} is not a finite rate in [0, 1]: {result!r}")
    return result


def finite_number(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(result):
        fail(f"{label} is not finite: {result!r}")
    return result


def _rate(rows: list[dict], key: str) -> float:
    return sum(bool(row.get(key)) for row in rows) / len(rows) if rows else 0.0


def _same_float(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _budget_key(scenario_set: str) -> str:
    if scenario_set in {"acquisition", "injected", "random"}:
        return "controlled"
    if scenario_set in {"normal", "recovery"}:
        return scenario_set
    fail(f"unknown scenario set: {scenario_set}")


def _validate_runner_summaries(
    payload: dict,
    cfg: dict,
    path: Path,
    *,
    mode: str,
    answer_max_tokens: int,
) -> None:
    summaries = payload.get("runner_summaries")
    if not isinstance(summaries, list) or not summaries:
        fail(f"behavior receipt has no runner summaries: {path}")
    expected_runner_sha = sha256_file(EXP / "src" / "vllm_runner.py")
    expected_greedy = mode == "deep"
    expected_seed = int(
        cfg["evaluation"]["deep_run_seed" if expected_greedy else "sample_run_seed"]
    )
    for index, summary in enumerate(summaries):
        label = f"{path} runner_summaries[{index}]"
        if not isinstance(summary, dict) or summary.get("runner_sha256") != expected_runner_sha:
            fail(f"runner implementation hash drifted at {label}")
        if summary.get("model") != payload.get("model"):
            fail(f"runner/model path mismatch at {label}")
        engine = summary.get("engine") or {}
        for key in (
            "max_model_len",
            "gpu_memory_utilization",
            "max_num_seqs",
            "max_num_batched_tokens",
        ):
            if engine.get(key) != cfg["engine"][key]:
                fail(f"engine drift at {key}: {label}")
        if engine.get("adapter") is not None:
            fail(f"behavior run unexpectedly used a runtime adapter: {label}")
        if engine.get("model_override") != payload.get("model"):
            fail(f"runner model_override mismatch at {label}")
        sampling = summary.get("sampling") or {}
        expected_sampling = {
            "thinking": "budget",
            "thinking_budget": int(cfg["evaluation"]["think_budget"]),
            "answer_max_tokens": answer_max_tokens,
            "greedy": expected_greedy,
            "run_seed": expected_seed,
        }
        for key, expected in expected_sampling.items():
            if sampling.get(key) != expected:
                fail(f"sampling drift at {key}: {label}")
        counts = summary.get("counts") or {}
        for key in ("sampled_tokens", "logical_model_input_tokens"):
            if not isinstance(counts.get(key), int) or counts[key] < 0:
                fail(f"invalid runner token count {key}: {label}")

    trajectories = payload.get("trajectories")
    if not isinstance(trajectories, list) or not trajectories:
        fail(f"behavior receipt has no trajectories: {path}")
    try:
        trajectory_sampled = sum(int(row["sampled_tokens"]) for row in trajectories)
        summary_sampled = sum(int(row["counts"]["sampled_tokens"]) for row in summaries)
        trajectory_input = sum(
            int(row["logical_model_input_tokens"]) for row in trajectories
        )
        summary_input = sum(
            int(row["counts"]["logical_model_input_tokens"]) for row in summaries
        )
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"malformed token accounting in {path}: {exc}")
    if trajectory_sampled != summary_sampled or trajectory_input != summary_input:
        fail(f"trajectory/runner token accounting mismatch: {path}")


def _validate_behavior_aggregate(payload: dict, path: Path) -> None:
    aggregate = payload.get("aggregate")
    if not isinstance(aggregate, dict):
        fail(f"behavior receipt has no aggregate object: {path}")
    cases = aggregate.get("cases")
    dyads = aggregate.get("dyads")
    if not isinstance(cases, list) or not cases:
        fail(f"behavior receipt has no aggregate cases: {path}")
    if not isinstance(dyads, list) or not dyads:
        fail(f"behavior receipt has no aggregate dyads: {path}")
    if aggregate.get("n_cases") != len(cases) or aggregate.get("n_dyads") != len(dyads):
        fail(f"aggregate case/dyad counts do not match rows: {path}")
    case_ids = [row.get("case_id") for row in cases]
    if any(not isinstance(value, str) or not value for value in case_ids):
        fail(f"aggregate case IDs are malformed: {path}")
    if len(set(case_ids)) != len(case_ids):
        fail(f"aggregate case IDs are not unique: {path}")
    dyad_ids = [(row.get("pair_id"), row.get("scenario")) for row in dyads]
    if len(set(dyad_ids)) != len(dyad_ids) or any(
        not isinstance(pair_id, str) or not pair_id or not isinstance(scenario, str)
        for pair_id, scenario in dyad_ids
    ):
        fail(f"aggregate dyad IDs are malformed or duplicated: {path}")
    by_dyad: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for case in cases:
        by_dyad[(case.get("pair_id"), case.get("scenario"))].append(case)
    if set(by_dyad) != set(dyad_ids):
        fail(f"case and dyad identities differ: {path}")
    for key, members in by_dyad.items():
        if len(members) != 2 or {row.get("branch") for row in members} != {0, 1}:
            fail(f"incomplete counterfactual dyad {key}: {path}")
    if not _same_float(
        aggregate.get("paired_preverifier_success"),
        _rate(dyads, "paired_preverifier_success"),
    ):
        fail(f"aggregate paired-preverifier rate does not match dyads: {path}")
    if not _same_float(aggregate.get("success"), _rate(cases, "success")):
        fail(f"aggregate terminal success does not match cases: {path}")
    for key in (
        "paired_preverifier_success",
        "success",
        "first_patch_full_correct",
        "unnecessary_evidence_before_first_patch",
        "invalid_action_rate_per_turn",
        "answer_cap_hit_rate_per_turn",
        "unusable_answer_cap_hit_rate_per_turn",
        "verified_given_success",
        "commit_given_verified",
    ):
        finite_rate(aggregate.get(key), f"{path} aggregate.{key}")
    finite_number(
        aggregate.get("mean_non_source_inspects_before_first_patch"),
        f"{path} aggregate.mean_non_source_inspects_before_first_patch",
    )

    dimensions = {
        "per_family": "family",
        "per_channel": "evidence_channel",
        "per_query_skin": "acquisition_query_skin",
    }
    for aggregate_key, row_key in dimensions.items():
        table = aggregate.get(aggregate_key)
        if not isinstance(table, dict) or not table:
            fail(f"missing {aggregate_key}: {path}")
        expected_names = {row.get(row_key) for row in dyads}
        if set(table) != expected_names or None in expected_names:
            fail(f"{aggregate_key} keys do not match dyads: {path}")
        for name, stats in table.items():
            members = [row for row in dyads if row[row_key] == name]
            if not isinstance(stats, dict):
                fail(f"malformed {aggregate_key}.{name}: {path}")
            if not _same_float(
                stats.get("paired_preverifier_success"),
                _rate(members, "paired_preverifier_success"),
            ):
                fail(f"{aggregate_key}.{name} does not match dyads: {path}")


def _registered_behavior_tasks(cfg: dict, block: str, contract: str) -> list:
    block_cfg = cfg["evaluation"]["blocks"].get(block)
    if not isinstance(block_cfg, dict) or block_cfg.get("legacy_retention"):
        fail(f"not a registered counterfactual behavior block: {block}")
    families_key = block_cfg.get("families")
    if families_key not in cfg["families"]:
        fail(f"behavior block has an unknown family registry: {block}")
    n_tasks = int(block_cfg["tasks_per_family"])
    if n_tasks < 2 or n_tasks % 2:
        fail(f"registered behavior task count is not positive/even: {block}")
    return repo_tasks.make_pairs(
        tuple(cfg["families"][families_key]),
        n_tasks // 2,
        int(block_cfg["seed"]),
        block,
        explicit_contract=contract == "explicit",
    )


def _registered_behavior_manifests(tasks: list) -> dict:
    content_digests = [repo_tasks.content_digest(task) for task in tasks]
    pair_static_digests = [repo_tasks.pair_static_digest(task) for task in tasks]
    mapping = [
        {
            "task_id": task.task_id,
            "pair_id": task.pair_id,
            "branch": task.branch,
            "family": task.family,
            "evidence_channel": task.evidence_channel,
            "evidence_path": task.evidence_path,
            "evidence_path_regime": task.evidence_path_regime,
            "acquisition_query_skin": task.acquisition_query_skin,
            "evidence_sha256": hashlib.sha256(
                task.files[task.evidence_path].encode()
            ).hexdigest(),
            "acquisition_query_sha256": hashlib.sha256(
                task.acquisition_query.encode()
            ).hexdigest(),
            "oracle_first_patch_sha256": hashlib.sha256(
                json.dumps(
                    task.oracle_patches[0].__dict__,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        }
        for task in tasks
    ]
    return {
        "task_manifest_sha256": repo_tasks.manifest_digest(tasks),
        "task_content_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(content_digests), separators=(",", ":")).encode()
        ).hexdigest(),
        "pair_static_manifest_sha256": hashlib.sha256(
            json.dumps(sorted(pair_static_digests), separators=(",", ":")).encode()
        ).hexdigest(),
        "composed_mapping_manifest": mapping,
    }


def _validate_local_provenance(payload: dict, cfg: dict, path: Path) -> None:
    config_path_value = cfg.get("__analysis_config_path__")
    if not isinstance(config_path_value, str):
        fail("analyzer config path was not registered before receipt validation")
    config_path = Path(config_path_value)
    expected_hashes = {
        "config_sha256": sha256_file(config_path),
        "evaluator_sha256": sha256_file(EXP / "scripts" / "eval_repo_agent.py"),
        "repo_agent_sha256": sha256_file(EXP / "src" / "repo_agent.py"),
        "task_generator_sha256": sha256_file(EXP / "src" / "repo_tasks.py"),
    }
    for key, expected in expected_hashes.items():
        if payload.get(key) != expected:
            fail(f"behavior provenance drift at {key}: {path}")
    try:
        model_path = Path(payload["model"]).resolve()
    except (KeyError, TypeError) as exc:
        fail(f"behavior receipt has no model path: {path}: {exc}")
    if not model_path.is_dir():
        fail(f"behavior model path is unavailable: {path}")
    files = {
        "model_weight_sha256": model_path / "model.safetensors",
        "model_config_sha256": model_path / "config.json",
        "model_generation_config_sha256": (
            model_path / "generation_config.json"
        ),
        "merge_receipt_sha256": model_path / "merge_receipt.json",
    }
    for key, model_file in files.items():
        if not model_file.is_file() or payload.get(key) != sha256_file(model_file):
            fail(f"behavior model provenance drift at {key}: {path}")
    try:
        tokenizer = harness.validate_registered_tokenizer_provenance(
            model_path, payload
        )
        merge_receipt = read_json(model_path / "merge_receipt.json")
        registered = harness.validate_registered_tokenizer_provenance(
            model_path, merge_receipt, allow_absent=True
        )
    except (OSError, ValueError) as exc:
        fail(f"behavior tokenizer provenance drift: {path}: {exc}")
    if tokenizer != registered:
        fail(f"behavior/merge tokenizer provenance mismatch: {path}")
    model_cfg = cfg.get("model", {})
    anchor_paths = set()
    for key in ("locality_anchor", "menagerie_incumbent"):
        value = model_cfg.get(key)
        if isinstance(value, str):
            registered_path = Path(value)
            anchor_paths.add(
                (registered_path if registered_path.is_absolute() else ROOT / registered_path)
                .resolve()
            )
    expected_manifest = (
        model_cfg.get("anchor_tokenizer_manifest_sha256")
        if model_path in anchor_paths
        else model_cfg.get("start_tokenizer_manifest_sha256")
    )
    if (
        tokenizer["tokenizer_manifest_sha256"]
        != expected_manifest
        or tokenizer["tokenizer_compatibility_sha256"]
        != model_cfg.get("tokenizer_compatibility_sha256")
    ):
        fail(f"behavior tokenizer differs from the frozen config identity: {path}")


def validate_behavior_receipt(
    path: Path,
    cfg: dict,
    *,
    block: str,
    contract: str,
    scenario_set: str,
    mode: str,
    arm: str,
    expected_weight_sha256: str | None = None,
    scaffold: bool = False,
) -> dict:
    payload = read_json(path)
    expected_identity = (block, contract, scenario_set, mode, scaffold)
    observed_identity = (
        payload.get("block"),
        payload.get("contract"),
        payload.get("scenario_set"),
        payload.get("mode"),
        payload.get("scaffold"),
    )
    if observed_identity != expected_identity:
        fail(f"wrong behavior receipt {path}: {observed_identity} != {expected_identity}")
    if payload.get("schema_version") != 1:
        fail(f"unsupported behavior receipt schema: {path}")
    if payload.get("arm") != arm:
        fail(f"wrong behavior arm in {path}: {payload.get('arm')!r} != {arm!r}")
    observed_hash = payload.get("model_weight_sha256")
    if not is_sha256(observed_hash):
        fail(f"invalid behavior model hash: {path}")
    if expected_weight_sha256 is not None and observed_hash != expected_weight_sha256:
        fail(f"wrong behavior model weights: {path}")
    if payload.get("history_policy") != HISTORY_POLICY:
        fail(f"history policy drifted: {path}")
    answer_max_tokens = payload.get("answer_max_tokens")
    if answer_max_tokens not in cfg["evaluation"]["interface_answer_rungs"]:
        fail(f"unregistered answer allowance in {path}: {answer_max_tokens!r}")
    if payload.get("think_budget") != int(cfg["evaluation"]["think_budget"]):
        fail(f"thinking allowance drifted: {path}")
    _validate_local_provenance(payload, cfg, path)
    registered_tasks = _registered_behavior_tasks(cfg, block, contract)
    registered_count = int(cfg["evaluation"]["blocks"][block]["tasks_per_family"])
    if payload.get("tasks_per_family") != registered_count:
        fail(f"behavior receipt used a tasks-per-family override: {path}")
    registered_manifests = _registered_behavior_manifests(registered_tasks)
    for key, expected in registered_manifests.items():
        if payload.get(key) != expected:
            fail(f"behavior receipt differs from registered tasks at {key}: {path}")
    mapping = payload.get("composed_mapping_manifest")
    if not isinstance(mapping, list) or not mapping:
        fail(f"missing composed mapping manifest: {path}")
    required_mapping = {
        "task_id",
        "pair_id",
        "branch",
        "family",
        "evidence_channel",
        "evidence_path",
        "evidence_path_regime",
        "acquisition_query_skin",
        "evidence_sha256",
        "acquisition_query_sha256",
        "oracle_first_patch_sha256",
    }
    task_ids = []
    for row in mapping:
        if not isinstance(row, dict) or not required_mapping.issubset(row):
            fail(f"malformed composed mapping row: {path}")
        if row.get("branch") not in (0, 1):
            fail(f"invalid composed mapping branch: {path}")
        for key in ("evidence_sha256", "acquisition_query_sha256", "oracle_first_patch_sha256"):
            if not is_sha256(row.get(key)):
                fail(f"invalid composed mapping hash {key}: {path}")
        task_ids.append(row.get("task_id"))
    if any(not isinstance(value, str) or not value for value in task_ids):
        fail(f"invalid composed mapping task ID: {path}")
    if len(set(task_ids)) != len(task_ids):
        fail(f"duplicate composed mapping task ID: {path}")

    budget = cfg["evaluation"][_budget_key(scenario_set)]
    per_call = int(cfg["evaluation"]["think_budget"]) + int(answer_max_tokens)
    if mode == "deep":
        trajectories = 1
        turns = int(budget["deep_turns"])
    elif mode == "sample_more":
        trajectories = int(budget["sample_more_trajectories"])
        turns = int(budget["sample_more_turns_each"])
    elif mode == "sample_pool":
        trajectories = int(cfg["evaluation"]["sample_more_pool_trajectories"])
        turns = int(budget["sample_more_turns_each"])
    else:
        fail(f"unregistered behavior mode: {mode}")
    expected_reserved = trajectories * turns * per_call
    expected_deep = int(budget["deep_turns"]) * per_call
    if (
        payload.get("reserved_sampled_tokens_per_case") != expected_reserved
        or payload.get("deep_reserved_sampled_tokens_per_case") != expected_deep
    ):
        fail(f"behavior compute reservation drifted: {path}")
    if mode != "sample_pool" and expected_reserved != expected_deep:
        fail(f"registered behavior modes are not compute matched: {path}")
    _validate_runner_summaries(
        payload,
        cfg,
        path,
        mode=mode,
        answer_max_tokens=int(answer_max_tokens),
    )
    _validate_behavior_aggregate(payload, path)
    aggregate_task_ids = {row.get("task_id") for row in payload["aggregate"]["cases"]}
    if aggregate_task_ids != set(task_ids):
        fail(f"aggregate tasks do not match composed mapping manifest: {path}")
    scenarios = {
        "acquisition": ("ambiguous_source",),
        "injected": ("evidence_injected",),
        "random": ("nondiscriminating_search_injected",),
        "normal": ("normal",),
        "recovery": ("rejected_patch", "failed_test"),
    }[scenario_set]
    expected_cases = len(registered_tasks) * len(scenarios)
    expected_dyads = expected_cases // 2
    if (
        payload["aggregate"].get("n_cases") != expected_cases
        or payload["aggregate"].get("n_dyads") != expected_dyads
    ):
        fail(f"behavior aggregate count differs from registered block: {path}")
    expected_trajectories = {
        (task.task_id, scenario, trajectory)
        for task in registered_tasks
        for scenario in scenarios
        for trajectory in range(trajectories)
    }
    try:
        observed_trajectories = {
            (row["task_id"], row["scenario"], int(row["trajectory"]))
            for row in payload["trajectories"]
        }
    except (KeyError, TypeError, ValueError) as exc:
        fail(f"malformed behavior trajectory identity: {path}: {exc}")
    if (
        observed_trajectories != expected_trajectories
        or len(payload["trajectories"]) != len(expected_trajectories)
    ):
        fail(f"behavior trajectories differ from registered task/run grid: {path}")
    summary_requests = sum(
        int(row["counts"].get("requests", -1)) for row in payload["runner_summaries"]
    )
    trajectory_turns = sum(int(row.get("turns", -1)) for row in payload["trajectories"])
    if summary_requests != trajectory_turns:
        fail(f"behavior runner request count differs from trajectory turns: {path}")
    return payload


def assert_behavior_peers(
    named_payloads: dict[str, dict],
    *,
    include_scenario: bool = True,
    include_scaffold: bool = True,
) -> None:
    keys = [
        "block",
        "contract",
        "mode",
        "history_policy",
        "think_budget",
        "answer_max_tokens",
        "task_manifest_sha256",
        "task_content_manifest_sha256",
        "pair_static_manifest_sha256",
        "composed_mapping_manifest",
    ]
    if include_scenario:
        keys.append("scenario_set")
    if include_scaffold:
        keys.append("scaffold")
    names = list(named_payloads)
    first_name = names[0]
    first = named_payloads[first_name]
    for name in names[1:]:
        payload = named_payloads[name]
        for key in keys:
            if payload.get(key) != first.get(key):
                fail(f"behavior peers {first_name}/{name} differ at {key}")


def dyad_map(payload: dict) -> dict[tuple[str, str], dict]:
    rows = payload["aggregate"]["dyads"]
    return {(row["pair_id"], row["scenario"]): row for row in rows}


def paired_differences(candidate: dict, comparator: dict) -> list[float]:
    left = dyad_map(candidate)
    right = dyad_map(comparator)
    if set(left) != set(right):
        fail("paired receipts contain different dyad identities")
    differences = []
    for key in sorted(left):
        for metadata in (
            "family",
            "evidence_channel",
            "evidence_path_regime",
            "acquisition_query_skin",
            "explicit_contract",
        ):
            if left[key].get(metadata) != right[key].get(metadata):
                fail(f"paired dyad metadata differs at {key} {metadata}")
        differences.append(
            float(bool(left[key]["paired_preverifier_success"]))
            - float(bool(right[key]["paired_preverifier_success"]))
        )
    return differences


def validate_selected_answer_allowance(payloads: Iterable[dict]) -> int:
    values = {int(payload["answer_max_tokens"]) for payload in payloads}
    if len(values) != 1:
        fail(f"behavior peers use different answer allowances: {sorted(values)}")
    return values.pop()
