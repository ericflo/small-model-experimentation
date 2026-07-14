"""Outcome-blind frozen-reservoir accounting and final matched-compute gate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from analyze import paired_bootstrap_interval
from provenance import validate_action_inputs, validate_generation_protocol, validate_sampling
from score_artifacts import validate_score_artifact
from scoring import score_generation_rows
from vllm_runner import SamplingConfig


EXPERIMENT_ID = "qwen35_4b_counterfactual_plan_reflection_transfer"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def artifact_ref(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ValueError(f"matched-compute source artifact does not exist: {resolved}")
    return {"path": str(resolved), "sha256": sha256_file(resolved)}


def path_from_ref(value: Any, label: str) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"matched-compute artifact has malformed {label} reference")
    path = Path(value["path"])
    if not path.is_absolute() or not path.is_file() or sha256_file(path) != value["sha256"]:
        raise ValueError(f"matched-compute {label} artifact is absent or changed")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def validate_training_compute(receipt: dict[str, Any]) -> dict[str, Any]:
    compute = receipt.get("compute")
    required = {
        "schema_version", "amortization_horizon", "forward_tokens",
        "forward_backward_multiplier", "token_forward_equivalents",
        "model_load_seconds", "training_seconds", "gpu_phase_wall_seconds",
    }
    if (
        not isinstance(compute, dict)
        or set(compute) != required
        or compute.get("schema_version") != 1
        or compute.get("amortization_horizon")
        != "full_training_charged_to_each_confirmation_split"
        or not isinstance(compute.get("forward_tokens"), int)
        or compute["forward_tokens"] < 1
        or compute.get("forward_backward_multiplier") != 3
        or compute.get("token_forward_equivalents") != compute["forward_tokens"] * 3
        or not isinstance(compute.get("model_load_seconds"), (int, float))
        or compute["model_load_seconds"] <= 0
        or not isinstance(compute.get("training_seconds"), (int, float))
        or compute["training_seconds"] <= 0
        or compute.get("gpu_phase_wall_seconds")
        != compute["model_load_seconds"] + compute["training_seconds"]
    ):
        raise ValueError("training receipt has invalid matched-compute accounting")
    return compute


def inference_compute(metadata: dict[str, Any]) -> dict[str, float | int]:
    counts = metadata.get("counts")
    timing = metadata.get("timing")
    if not isinstance(counts, dict) or not isinstance(timing, dict):
        raise ValueError("generation metadata lacks compute accounting")
    logical = counts.get("logical_model_input_tokens")
    sampled = counts.get("sampled_tokens")
    load = timing.get("model_load_seconds")
    generation = timing.get("generation_seconds")
    if (
        not isinstance(logical, int)
        or logical < 1
        or not isinstance(sampled, int)
        or sampled < 1
        or not isinstance(load, (int, float))
        or load <= 0
        or not isinstance(generation, (int, float))
        or generation <= 0
    ):
        raise ValueError("generation compute fields are invalid")
    return {
        "token_forward_equivalents": logical + sampled,
        "model_load_seconds": float(load),
        "generation_seconds": float(generation),
        "gpu_phase_wall_seconds": float(load) + float(generation),
    }


def target_compute_budget(
    targets: list[tuple[dict[str, Any], dict[str, Any]]],
    expected_seeds: set[int],
) -> dict[str, Any]:
    if len(targets) != len(expected_seeds):
        raise ValueError("matched-compute target cardinality changed")
    per_seed: dict[str, dict[str, float | int]] = {}
    for receipt, metadata in targets:
        seed = int(receipt.get("seed", -1))
        if seed in {int(value) for value in per_seed} or seed not in expected_seeds:
            raise ValueError("matched-compute targets do not contain both unique seeds")
        if receipt.get("arm") != "reflection_correct":
            raise ValueError("matched-compute target is not reflection-correct training")
        override = metadata.get("model_override")
        if not isinstance(override, dict) or override.get("source_seed") != seed:
            raise ValueError("trained confirmation metadata has the wrong source seed")
        training = validate_training_compute(receipt)
        inference = inference_compute(metadata)
        per_seed[str(seed)] = {
            "training_token_forward_equivalents": int(
                training["token_forward_equivalents"]
            ),
            "confirmation_token_forward_equivalents": int(
                inference["token_forward_equivalents"]
            ),
            "total_token_forward_equivalents": int(
                training["token_forward_equivalents"]
                + inference["token_forward_equivalents"]
            ),
            "training_gpu_phase_wall_seconds": float(
                training["gpu_phase_wall_seconds"]
            ),
            "confirmation_gpu_phase_wall_seconds": float(
                inference["gpu_phase_wall_seconds"]
            ),
            "total_gpu_phase_wall_seconds": float(
                training["gpu_phase_wall_seconds"]
                + inference["gpu_phase_wall_seconds"]
            ),
        }
    if set(map(int, per_seed)) != expected_seeds:
        raise ValueError("matched-compute target seeds changed")
    return {
        "schema_version": 1,
        "amortization_horizon": "full_training_charged_to_each_144_task_confirmation_split",
        "per_seed": per_seed,
        "required_token_forward_equivalents": max(
            int(value["total_token_forward_equivalents"])
            for value in per_seed.values()
        ),
        "required_gpu_phase_wall_seconds": max(
            float(value["total_gpu_phase_wall_seconds"])
            for value in per_seed.values()
        ),
    }


def cumulative_reservoir_compute(
    block_metadata: list[dict[str, Any]],
) -> list[dict[str, float | int]]:
    if not block_metadata:
        raise ValueError("frozen reservoir has no completed blocks")
    cumulative_tokens = 0
    cumulative_wall = 0.0
    result: list[dict[str, float | int]] = []
    first_load: float | None = None
    for index, metadata in enumerate(block_metadata):
        observed = inference_compute(metadata)
        load = float(observed["model_load_seconds"])
        if first_load is None:
            first_load = load
            cumulative_wall += load
        elif load != first_load:
            raise ValueError("reservoir blocks do not share one persistent model load")
        cumulative_wall += float(observed["generation_seconds"])
        cumulative_tokens += int(observed["token_forward_equivalents"])
        result.append(
            {
                "blocks": index + 1,
                "candidates_per_task": 16 * (index + 1),
                "token_forward_equivalents": cumulative_tokens,
                "gpu_phase_wall_seconds": cumulative_wall,
            }
        )
    return result


def first_budget_prefix(
    cumulative: list[dict[str, float | int]], target: dict[str, Any]
) -> int | None:
    required_tokens = int(target["required_token_forward_equivalents"])
    required_wall = float(target["required_gpu_phase_wall_seconds"])
    for index, value in enumerate(cumulative):
        if (
            int(value["token_forward_equivalents"]) >= required_tokens
            and float(value["gpu_phase_wall_seconds"]) >= required_wall
        ):
            return index
    return None


def _validate_block_outputs(path: Path, *, rows: int, candidates: int) -> None:
    generated = read_jsonl(path)
    if len(generated) != rows:
        raise ValueError("reservoir block row count changed")
    for row in generated:
        outputs = row.get("outputs")
        if (
            not isinstance(outputs, list)
            or len(outputs) != candidates
            or [value.get("sample_index") for value in outputs]
            != list(range(candidates))
        ):
            raise ValueError("reservoir block candidate/sample-index contract changed")


def validate_reservoir_manifest(
    path: Path,
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
    labels_path: Path,
    expected_frozen_generated: Path,
    expected_frozen_metadata: Path,
    expected_targets: dict[int, tuple[Path, Path]],
) -> dict[str, Any]:
    observed = json.loads(path.read_text())
    required = {
        "schema_version", "experiment_id", "config_sha256", "producer",
        "worktree", "stage_receipt", "input", "input_receipt", "targets",
        "target_budget", "block_contract", "blocks", "cumulative_compute",
        "stop", "outcome_blindness",
    }
    if (
        not isinstance(observed, dict)
        or set(observed) != required
        or observed.get("schema_version") != 1
        or observed.get("experiment_id") != EXPERIMENT_ID
        or observed.get("config_sha256") != sha256_file(config_path)
    ):
        raise ValueError("frozen reservoir manifest schema or identity changed")
    producer = observed["producer"]
    if (
        not isinstance(producer, dict)
        or set(producer)
        != {"script_sha256", "module_sha256", "runner_sha256", "git_commit"}
        or producer["script_sha256"]
        != sha256_file(experiment_root / "scripts" / "run_frozen_reservoir.py")
        or producer["module_sha256"] != sha256_file(Path(__file__).resolve())
        or producer["runner_sha256"]
        != sha256_file(experiment_root / "src" / "vllm_runner.py")
    ):
        raise ValueError("frozen reservoir producer identity changed")
    input_path = path_from_ref(observed["input"], "reservoir input")
    receipt_path = path_from_ref(observed["input_receipt"], "reservoir input receipt")
    stage_path = path_from_ref(observed["stage_receipt"], "reservoir stage receipt")
    split, _task_metadata, sealed = validate_action_inputs(
        config=config,
        config_path=config_path,
        receipt_path=receipt_path,
        labels_path=labels_path,
        expected_split="confirmation",
    )
    if split != "confirmation" or sha256_file(input_path) != sealed["prompt_sha256"]:
        raise ValueError("reservoir input differs from sealed confirmation prompts")
    target_refs = observed["targets"]
    if not isinstance(target_refs, list) or len(target_refs) != len(expected_targets):
        raise ValueError("reservoir target reference cardinality changed")
    targets: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen_seeds: set[int] = set()
    for value in target_refs:
        if not isinstance(value, dict) or set(value) != {
            "seed", "training_receipt", "correct_confirmation_metadata"
        }:
            raise ValueError("reservoir target reference schema changed")
        seed = int(value["seed"])
        if seed in seen_seeds or seed not in expected_targets:
            raise ValueError("reservoir target seed changed")
        seen_seeds.add(seed)
        training_path = path_from_ref(value["training_receipt"], "training receipt")
        metadata_path = path_from_ref(
            value["correct_confirmation_metadata"], "correct confirmation metadata"
        )
        if (
            training_path.resolve() != expected_targets[seed][0].resolve()
            or metadata_path.resolve() != expected_targets[seed][1].resolve()
        ):
            raise ValueError("reservoir target does not bind exact confirmation evidence")
        targets.append((json.loads(training_path.read_text()), json.loads(metadata_path.read_text())))
    target_budget = target_compute_budget(targets, set(expected_targets))
    if observed["target_budget"] != target_budget:
        raise ValueError("reservoir target budget differs from exact receipts")
    contract = config["evaluation"]["frozen_sample_more"]
    if observed["block_contract"] != contract:
        raise ValueError("reservoir block contract differs from preregistration")
    blocks = observed["blocks"]
    seeds = list(contract["block_run_seeds"])
    if (
        len(seeds) != int(contract["maximum_blocks"])
        or len(set(map(int, seeds))) != len(seeds)
        or int(contract["maximum_candidates_per_task"])
        != int(contract["block_candidate_count"]) * int(contract["maximum_blocks"])
        or int(contract["block_candidate_count"]) != 16
        or contract["prefix_caching"] is not False
        or contract["stop_inputs"]
        != "compute_receipts_only_no_labels_scores_or_correctness"
        or not isinstance(blocks, list)
        or not blocks
        or len(blocks) > int(contract["maximum_blocks"])
    ):
        raise ValueError("reservoir completed-block cardinality changed")
    metadata_values: list[dict[str, Any]] = []
    generated_paths: list[Path] = []
    runtime_protocols: set[str] = set()
    for index, block in enumerate(blocks):
        if (
            not isinstance(block, dict)
            or set(block) != {"index", "run_seed", "generated", "metadata"}
            or block["index"] != index
            or block["run_seed"] != seeds[index]
        ):
            raise ValueError("reservoir block order/seed contract changed")
        generated_path = path_from_ref(block["generated"], "reservoir generated block")
        metadata_path = path_from_ref(block["metadata"], "reservoir metadata block")
        metadata = json.loads(metadata_path.read_text())
        validate_sampling(
            metadata,
            SamplingConfig(
                thinking="budget",
                thinking_budget=int(config["evaluation"]["thinking_budget"]),
                n=int(contract["block_candidate_count"]),
                answer_max_tokens=int(config["evaluation"]["answer_max_tokens"]),
                temperature=float(config["evaluation"]["temperature"]),
                top_p=float(config["evaluation"]["top_p"]),
                top_k=int(config["evaluation"]["top_k"]),
                run_seed=int(block["run_seed"]),
            ),
        )
        protocol = validate_generation_protocol(
            metadata=metadata,
            config=config,
            experiment_root=experiment_root,
            generated_path=generated_path,
            expected_rows=int(json.loads(receipt_path.read_text())["rows"]),
            expect_merged=False,
            expected_stage="confirmation",
            expected_split="confirmation",
            expected_input_kind="action",
            expected_source_seed=None,
        )
        if metadata.get("input", {}).get("sha256") != sealed["prompt_sha256"]:
            raise ValueError("reservoir block used different confirmation prompts")
        _validate_block_outputs(
            generated_path,
            rows=int(json.loads(receipt_path.read_text())["rows"]),
            candidates=int(contract["block_candidate_count"]),
        )
        runtime_protocols.add(protocol)
        generated_paths.append(generated_path)
        metadata_values.append(metadata)
    if len(runtime_protocols) != 1:
        raise ValueError("reservoir blocks do not share one runtime protocol")
    first_runtime = metadata_values[0].get("runtime", {})
    expected_worktree = {
        "repo_root": first_runtime.get("git_root"),
        "git_commit": first_runtime.get("git_commit"),
        "head_mode": first_runtime.get("git_head_mode"),
        "cwd": first_runtime.get("cwd"),
    }
    if (
        observed["worktree"] != expected_worktree
        or producer["git_commit"] != expected_worktree["git_commit"]
        or any(
            metadata.get("runtime", {}).get("git_commit")
            != expected_worktree["git_commit"]
            or metadata.get("generation_stage", {}).get("stage_receipt_path")
            != str(stage_path.resolve())
            for metadata in metadata_values
        )
    ):
        raise ValueError("reservoir worktree/stage lineage differs across blocks")
    if (
        generated_paths[0].resolve() != expected_frozen_generated.resolve()
        or path_from_ref(blocks[0]["metadata"], "first reservoir metadata").resolve()
        != expected_frozen_metadata.resolve()
    ):
        raise ValueError("reservoir block zero is not the frozen confirmation comparator")
    cumulative = cumulative_reservoir_compute(metadata_values)
    prefix = first_budget_prefix(cumulative, target_budget)
    if observed["cumulative_compute"] != cumulative:
        raise ValueError("reservoir cumulative compute differs from raw metadata")
    expected_stop = {
        "decision": (
            "FIRST_COMPLETE_BLOCK_REACHES_BOTH_BUDGETS"
            if prefix == len(blocks) - 1
            else "MAXIMUM_BLOCKS_EXHAUSTED_WITHOUT_BOTH_BUDGETS"
        ),
        "first_budget_prefix_index": prefix,
        "completed_blocks": len(blocks),
    }
    if prefix is not None and prefix != len(blocks) - 1:
        raise ValueError("reservoir continued after an earlier compute-only stop")
    if prefix is None and len(blocks) != int(contract["maximum_blocks"]):
        raise ValueError("reservoir stopped before reaching compute or maximum blocks")
    if observed["stop"] != expected_stop:
        raise ValueError("reservoir stop receipt differs from compute-only replay")
    if observed["outcome_blindness"] != {
        "accepted_label_or_score_paths": False,
        "read_correctness_fields": False,
        "stop_fields": [
            "training_receipt.compute",
            "correct_confirmation_metadata.counts",
            "correct_confirmation_metadata.timing",
            "frozen_block_metadata.counts",
            "frozen_block_metadata.timing",
        ],
    }:
        raise ValueError("reservoir outcome-blindness contract changed")
    labels = read_jsonl(labels_path)
    block_scores = [
        score_generation_rows(
            read_jsonl(generated_path),
            labels,
            arm="frozen_matched_compute_action",
            candidate_counts=(int(contract["block_candidate_count"]),),
            answer_max_tokens=int(config["evaluation"]["answer_max_tokens"]),
            loop_detector=config["evaluation"]["loop_detector"],
        )
        for generated_path in generated_paths
    ]
    coverage: dict[str, float] = {}
    family: dict[str, str] = {}
    for rows in block_scores:
        for row in rows:
            task_id = str(row["task_id"])
            coverage[task_id] = max(
                coverage.get(task_id, 0.0),
                float(row[f"coverage_at_{contract['block_candidate_count']}"]),
            )
            family[task_id] = str(row["family"])
    return {
        "manifest": observed,
        "budget_pass": prefix == len(blocks) - 1,
        "runtime_protocol_sha256": next(iter(runtime_protocols)),
        "coverage": coverage,
        "family": family,
        "candidates_per_task": len(blocks) * int(contract["block_candidate_count"]),
    }


def _decision_scores(
    decision: dict[str, Any],
    *,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    result: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for reference in decision["invocation"]["scores"]:
        score_path = path_from_ref(reference, "confirmation score")
        rows = validate_score_artifact(
            score_path,
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
        arm = str(rows[0]["arm"])
        if arm in result:
            raise ValueError("confirmation decision duplicates an arm score artifact")
        result[arm] = (score_path, rows)
    return result


def _comparison(
    correct_rows: list[dict[str, Any]],
    reservoir: dict[str, Any],
    bootstrap: dict[str, Any],
    threshold: dict[str, Any],
    seed_offset: int,
) -> dict[str, Any]:
    correct = {str(row["task_id"]): row for row in correct_rows}
    if set(correct) != set(reservoir["coverage"]):
        raise ValueError("trained and matched-compute task sets differ")
    deltas = [
        float(correct[task_id]["coverage_at_16"])
        - float(reservoir["coverage"][task_id])
        for task_id in sorted(correct)
    ]
    lower, upper = paired_bootstrap_interval(
        deltas,
        int(bootstrap["paired_task_resamples"]),
        int(bootstrap["seed"]) + seed_offset,
    )
    families = sorted(set(reservoir["family"].values()))
    family_delta = {
        family: mean(
            float(correct[task_id]["coverage_at_16"])
            - float(reservoir["coverage"][task_id])
            for task_id in sorted(correct)
            if reservoir["family"][task_id] == family
        )
        for family in families
    }
    delta = mean(deltas)
    checks = {
        "strictly_beats_matched_frozen": delta
        > float(threshold["correct_at_16_minus_frozen_matched_delta_gt"]),
        "paired_lower_95": lower > float(threshold["paired_delta_lower_95_gt"]),
        "each_family_nonnegative": all(
            value >= float(threshold["each_family_delta_min"])
            for value in family_delta.values()
        ),
        "wall_and_token_budget": reservoir["budget_pass"] is True,
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "n": len(deltas),
        "delta": delta,
        "lower_95": lower,
        "upper_95": upper,
        "family_delta": family_delta,
        "trained_candidates_per_task": 16,
        "frozen_candidates_per_task": reservoir["candidates_per_task"],
    }


def build_matched_compute_artifact(
    *,
    confirmation_decision_paths: list[Path],
    reservoir_manifest_path: Path,
    config: dict[str, Any],
    config_path: Path,
    experiment_root: Path,
) -> dict[str, Any]:
    from gate_artifacts import validate_gate_artifact
    from stages import git_commit

    expected_seeds = set(map(int, config["training"]["staged_seeds"].values()))
    if len(confirmation_decision_paths) != 2:
        raise ValueError("matched-compute gate requires both confirmation decisions")
    decisions: dict[int, dict[str, Any]] = {}
    scores: dict[int, dict[str, tuple[Path, list[dict[str, Any]]]]] = {}
    for decision_path in confirmation_decision_paths:
        decision = validate_gate_artifact(
            decision_path,
            kind="decision",
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
        seed = int(decision.get("seed", -1))
        if (
            seed in decisions
            or seed not in expected_seeds
            or decision.get("block") != "confirmation"
            or decision.get("capability", {}).get("capability_pass") is not True
        ):
            raise ValueError("matched-compute input is not both passing confirmation seeds")
        decisions[seed] = decision
        scores[seed] = _decision_scores(
            decision,
            config=config,
            config_path=config_path,
            experiment_root=experiment_root,
        )
    if set(decisions) != expected_seeds:
        raise ValueError("matched-compute decisions do not contain both seeds")
    frozen_paths = {
        values["frozen_action"][0].resolve() for values in scores.values()
    }
    if len(frozen_paths) != 1:
        raise ValueError("both confirmation decisions must reuse one frozen block zero")
    frozen_path = next(iter(frozen_paths))
    frozen_rows = next(iter(scores.values()))["frozen_action"][1]
    frozen_provenance = frozen_rows[0]["score_provenance"]
    frozen_generated = path_from_ref(frozen_provenance["generated"], "frozen generated")
    frozen_metadata = path_from_ref(frozen_provenance["metadata"], "frozen metadata")
    labels_path = path_from_ref(frozen_provenance["labels"], "confirmation labels")
    expected_targets: dict[int, tuple[Path, Path]] = {}
    correct_rows: dict[int, list[dict[str, Any]]] = {}
    for seed, arm_scores in scores.items():
        correct_path, rows = arm_scores["reflection_correct_action"]
        del correct_path
        provenance = rows[0]["score_provenance"]
        metadata_path = path_from_ref(
            provenance["metadata"], "correct confirmation metadata"
        )
        metadata = json.loads(metadata_path.read_text())
        override_path = Path(metadata["model_override"]["path"])
        training_path = (
            override_path / "source_lineage" / "adapter_tree" / "training_receipt.json"
        )
        if not training_path.is_file():
            raise ValueError("correct confirmation model lacks embedded training receipt")
        expected_targets[seed] = (training_path, metadata_path)
        correct_rows[seed] = rows
    reservoir = validate_reservoir_manifest(
        reservoir_manifest_path,
        config=config,
        config_path=config_path,
        experiment_root=experiment_root,
        labels_path=labels_path,
        expected_frozen_generated=frozen_generated,
        expected_frozen_metadata=frozen_metadata,
        expected_targets=expected_targets,
    )
    threshold = config["decision_gates"]["final_matched_compute"]
    bootstrap = config["decision_gates"]["bootstrap"]
    by_seed = {
        str(seed): _comparison(
            correct_rows[seed], reservoir, bootstrap, threshold, offset
        )
        for offset, seed in enumerate(sorted(expected_seeds), 20)
    }
    gate = {
        "pass": reservoir["budget_pass"] is True
        and all(value["pass"] for value in by_seed.values()),
        "budget_pass": reservoir["budget_pass"],
        "candidates_per_task": reservoir["candidates_per_task"],
        "runtime_protocol_sha256": reservoir["runtime_protocol_sha256"],
        "by_seed": by_seed,
    }
    return {
        "schema_version": 2,
        "experiment_id": EXPERIMENT_ID,
        "config_sha256": sha256_file(config_path),
        "producer": {
            "script_sha256": sha256_file(
                experiment_root / "scripts" / "matched_compute_gate.py"
            ),
            "module_sha256": sha256_file(Path(__file__).resolve()),
            "git_commit": git_commit(),
        },
        "invocation": {
            "confirmation_decisions": [
                artifact_ref(path) for path in confirmation_decision_paths
            ],
            "reservoir_manifest": artifact_ref(reservoir_manifest_path),
        },
        "gate": gate,
    }
