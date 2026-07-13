"""Fail-closed phase analysis for state-formation capacity adjudication."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .config import MODEL_ID, MODEL_REVISION, config_sha256, require_confirmatory_config, source_contract_sha256
from .data_pipeline import load_contrast_access_ledger, read_jsonl, validate_data_manifest
from .design_boundary import design_lineage, validate_design_receipt
from .initialization import load_initialization_bundle
from .substrate import trajectory_targets, verify_example


ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ROOT.parents[1]
REQUIREMENTS_LOCK = REPO_ROOT / "requirements-training.lock.txt"
SEEDS = (7411, 7412, 7413)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identity(config: Mapping[str, Any], phase: str) -> dict[str, Any]:
    design = design_lineage(config)
    return {
        "experiment_id": config["experiment_id"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "backend": "transformers",
        "config_sha256": config_sha256(config),
        "source_contract_sha256": source_contract_sha256(ROOT),
        "requirements_training_lock_sha256": _sha256(REQUIREMENTS_LOCK),
        "design_receipt_sha256": design["sha256"],
        "design_receipt_identity_sha256": design["receipt_identity_sha256"],
        "phase": phase,
    }


def _with_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(payload)
    receipt["receipt_identity_sha256"] = _canonical_sha256(receipt)
    return receipt


def _checkpoint_identity(metadata: Mapping[str, Any]) -> str:
    return _canonical_sha256(
        {key: value for key, value in metadata.items() if key != "checkpoint_identity_sha256"}
    )


def _resolve_repo_path(value: str) -> Path:
    path = (REPO_ROOT / value).resolve()
    if not path.is_relative_to(REPO_ROOT):
        raise RuntimeError(f"lineage path escapes repository: {value}")
    return path


def _validate_lineage_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    required = {"path", "sha256", "receipt_identity_sha256", "status", "phase"}
    if set(entry) != required:
        raise RuntimeError("analysis lineage entry has the wrong fields")
    path = _resolve_repo_path(str(entry["path"]))
    if not path.is_file() or _sha256(path) != entry["sha256"]:
        raise RuntimeError("analysis lineage file changed")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    claimed = receipt.get("receipt_identity_sha256")
    identity_payload = {
        key: value for key, value in receipt.items() if key != "receipt_identity_sha256"
    }
    if claimed != _canonical_sha256(identity_payload) or claimed != entry["receipt_identity_sha256"]:
        raise RuntimeError("analysis lineage receipt identity changed")
    if (
        str(receipt.get("status", receipt.get("verdict", ""))) != entry["status"]
        or receipt.get("phase") != entry["phase"]
    ):
        raise RuntimeError("analysis lineage status/phase changed")
    return receipt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _stable_setup_receipt(setup: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(setup.get("environment"), Mapping) or not isinstance(
        setup.get("preflight_device"), Mapping
    ):
        raise RuntimeError("setup device/environment receipt is malformed")

    def stable_device(device: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: value for key, value in device.items()
            if key != "free_memory_gib_before_load"
        }

    result = dict(setup)
    environment = dict(setup["environment"])
    if not isinstance(environment.get("device"), Mapping):
        raise RuntimeError("setup environment device receipt is malformed")
    environment["device"] = stable_device(environment["device"])
    result["environment"] = environment
    result["preflight_device"] = stable_device(setup["preflight_device"])
    return result


def _pairing_setup_receipt(setup: Mapping[str, Any]) -> dict[str, Any]:
    stable = _stable_setup_receipt(setup)
    fields = {
        "model_seed", "tokenizer", "adaptation_targets",
        "adaptation_targets_sha256", "shared_initialization", "dropout_control",
        "environment", "installed_environment_lock", "preflight_device",
    }
    if not fields.issubset(stable):
        raise RuntimeError("setup pairing fields are incomplete")
    return {key: stable[key] for key in fields}


def _training_lr(config: Mapping[str, Any], step: int) -> float:
    training = config["training"]
    total_steps = int(training["train_steps"])
    warmup = max(1, int(total_steps * float(training["warmup_fraction"])))
    schedule_index = step - 1
    if schedule_index < warmup:
        multiplier = (schedule_index + 1) / warmup
    else:
        progress = (schedule_index - warmup) / max(total_steps - warmup, 1)
        multiplier = 0.5 * (
            1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0))
        )
    return float(training["learning_rate"]) * multiplier


def _expected_training_order_sha256(config: Mapping[str, Any], seed: int) -> str:
    data_dir = (ROOT / str(config["paths"]["data_dir"])).resolve()
    rows = read_jsonl(data_dir / "train.jsonl.gz")
    rng = random.Random(seed)
    rng.shuffle(rows)
    total_microbatches = int(config["training"]["train_steps"]) * int(
        config["training"]["gradient_accumulation"]
    )
    train_k = int(config["training"]["train_k"])
    digest = hashlib.sha256()
    for microbatch_index in range(1, total_microbatches + 1):
        row = rows[(microbatch_index - 1) % len(rows)]
        event = {
            "microbatch_index": microbatch_index,
            "id": row["id"],
            "k": train_k,
        }
        digest.update(
            json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def _validate_training_payloads(
    config: Mapping[str, Any],
    *,
    run: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    capacity: str,
    objective: str,
    seed: int,
) -> dict[str, Any]:
    expected_run = {
        "schema_version": 1,
        "status": "TRAINING_COMPLETE",
        **_identity(config, f"{capacity}_{objective}_training"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": seed,
        "steps": int(config["training"]["train_steps"]),
        "data_manifest_sha256": checkpoint["data_manifest_sha256"],
    }
    for key, value in expected_run.items():
        if run.get(key) != value:
            raise RuntimeError(f"training run {key} mismatch")
    canonical_run_dir = (
        ROOT
        / str(config["paths"]["large_artifacts_dir"])
        / f"{capacity}_{objective}_seed{seed}"
    ).resolve()
    expected_checkpoint_path = (
        canonical_run_dir / f"checkpoint_{int(config['training']['train_steps']):06d}"
    )
    if checkpoint_path != expected_checkpoint_path:
        raise RuntimeError("training checkpoint uses a noncanonical result path")
    expected_shared_fields = (
        "training_prompt_tokens",
        "training_layer_token_applications",
        "training_order_sha256",
        "dropout_schedule_sha256",
        "dropout_probes",
        "train_metrics_sha256",
        "train_metrics_rows",
        "train_metrics_path",
        "optimizer_steps_sha256",
        "optimizer_steps_rows",
        "optimizer_steps_path",
        "optimizer_state",
        "optimizer_step_receipt",
        "setup_sha256",
        "stable_setup",
    )
    for field in expected_shared_fields:
        if run.get(field) != checkpoint.get(field):
            raise RuntimeError(f"training run/checkpoint {field} mismatch")
    if run.get("training_order_sha256") != _expected_training_order_sha256(
        config, seed
    ):
        raise RuntimeError("training row order differs from the registered seeded schedule")
    setup = run.get("setup")
    if not isinstance(setup, Mapping) or run.get("setup_sha256") != _canonical_sha256(setup):
        raise RuntimeError("training setup receipt is malformed")
    if run.get("stable_setup") != _stable_setup_receipt(setup):
        raise RuntimeError("training stable setup receipt changed")
    expected_setup = {
        "capacity": capacity,
        "model_seed": seed,
        "shared_initialization": checkpoint["shared_initialization"],
        "trainable_parameters": checkpoint["trainable_parameters"],
        "adaptation_parameters": checkpoint["adaptation_parameters"],
        "adaptation_target_manifest_sha256": checkpoint[
            "adaptation_target_manifest_sha256"
        ],
        "environment": checkpoint["environment"],
    }
    for key, value in expected_setup.items():
        if setup.get(key) != value:
            raise RuntimeError(f"training setup/checkpoint {key} mismatch")

    metrics_path = _resolve_repo_path(str(run["train_metrics_path"]))
    optimizer_steps_path = _resolve_repo_path(str(run["optimizer_steps_path"]))
    if metrics_path != canonical_run_dir / "train_metrics.jsonl":
        raise RuntimeError("training metrics use a noncanonical path")
    if optimizer_steps_path != canonical_run_dir / "optimizer_steps.jsonl":
        raise RuntimeError("optimizer steps use a noncanonical path")
    for path, digest, rows, label in (
        (metrics_path, run["train_metrics_sha256"], run["train_metrics_rows"], "metrics"),
        (
            optimizer_steps_path,
            run["optimizer_steps_sha256"],
            run["optimizer_steps_rows"],
            "optimizer steps",
        ),
    ):
        if (
            not path.is_file()
            or _sha256(path) != digest
            or _line_count(path) != rows
        ):
            raise RuntimeError(f"training {label} payload changed")

    metric_rows = _read_jsonl(metrics_path)
    total_steps = int(config["training"]["train_steps"])
    expected_metric_steps = [
        step for step in range(1, total_steps + 1)
        if step == 1 or step % 10 == 0 or step == total_steps
    ]
    if [row.get("step") for row in metric_rows] != expected_metric_steps:
        raise RuntimeError("training metric step sequence changed")
    for row in metric_rows:
        if (
            row.get("capacity") != capacity
            or row.get("objective") != objective
            or row.get("model_seed") != seed
        ):
            raise RuntimeError("training metric cell identity changed")
        for field in (
            "loss", "answer", "state", "fixed", "adaptation_learning_rate",
            "common_state_learning_rate",
            "preclip_adaptation_gradient_norm", "preclip_common_gradient_norm",
            "adaptation_applied_clip_scale", "common_state_applied_clip_scale",
            "elapsed_seconds", "peak_allocated_gib",
        ):
            _strict_float(row.get(field), f"training metric {field}", low=0.0)
        expected_lr = _training_lr(config, int(row["step"]))
        for group in ("adaptation", "common_state"):
            if not math.isclose(
                float(row[f"{group}_learning_rate"]), expected_lr,
                rel_tol=0.0, abs_tol=1e-15,
            ):
                raise RuntimeError(
                    f"training metric {group} learning-rate schedule changed"
                )

    optimizer_rows = _read_jsonl(optimizer_steps_path)
    if [row.get("step") for row in optimizer_rows] != list(range(1, total_steps + 1)):
        raise RuntimeError("optimizer receipt does not cover every exact step")
    event_digest = hashlib.sha256()
    minimum_scales = {"adaptation": 1.0, "common_state": 1.0}
    for row in optimizer_rows:
        if set(row) != {
            "step", "adaptation_preclip_gradient_norm",
            "adaptation_applied_clip_scale", "common_state_preclip_gradient_norm",
            "common_state_applied_clip_scale", "adaptation_gradient_finite",
            "common_state_gradient_finite", "base_trainable_parameters",
            "adaptation_learning_rate", "common_state_learning_rate",
        }:
            raise RuntimeError("optimizer-step schema changed")
        if (
            row["adaptation_gradient_finite"] is not True
            or row["common_state_gradient_finite"] is not True
            or row["base_trainable_parameters"] != 0
        ):
            raise RuntimeError("optimizer-step finiteness/base-freeze receipt failed")
        adaptation_norm = _strict_float(
            row["adaptation_preclip_gradient_norm"], "adaptation norm", low=0.0
        )
        common_norm = _strict_float(
            row["common_state_preclip_gradient_norm"], "common norm", low=0.0
        )
        expected_adaptation_scale = min(
            1.0,
            float(config["training"]["adaptation_gradient_clip"])
            / (adaptation_norm + 1e-6),
        )
        expected_common_scale = min(
            1.0,
            float(config["training"]["common_gradient_clip"])
            / (common_norm + 1e-6),
        )
        for group, actual, expected_scale in (
            ("adaptation", row["adaptation_applied_clip_scale"], expected_adaptation_scale),
            ("common_state", row["common_state_applied_clip_scale"], expected_common_scale),
        ):
            actual_scale = _strict_float(actual, f"{group} clip scale", low=0.0, high=1.0)
            if not math.isclose(actual_scale, expected_scale, rel_tol=0.0, abs_tol=1e-12):
                raise RuntimeError(f"{group} applied clip scale is inconsistent")
            minimum_scales[group] = min(minimum_scales[group], actual_scale)
        for group in ("adaptation", "common_state"):
            actual_lr = _strict_float(
                row[f"{group}_learning_rate"],
                f"optimizer {group} learning rate", low=0.0,
            )
            if not math.isclose(
                actual_lr, _training_lr(config, int(row["step"])),
                rel_tol=0.0, abs_tol=1e-15,
            ):
                raise RuntimeError(f"optimizer {group} learning-rate schedule changed")
        event_digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
    step_receipt = run.get("optimizer_step_receipt")
    if not isinstance(step_receipt, Mapping):
        raise RuntimeError("optimizer step summary is malformed")
    expected_step_summary = {
        "schema_version": 1,
        "steps": total_steps,
        "rows": total_steps,
        "events_sha256": event_digest.hexdigest(),
        "group_names": ["adaptation", "common_state"],
        "clip_thresholds": {
            "adaptation": float(config["training"]["adaptation_gradient_clip"]),
            "common_state": float(config["training"]["common_gradient_clip"]),
        },
        "minimum_applied_clip_scales": minimum_scales,
        "all_gradients_finite": True,
        "base_trainable_parameters": 0,
        "probes": [
            optimizer_rows[index - 1]
            for index in (1, max(1, total_steps // 2), total_steps)
        ],
    }
    if dict(step_receipt) != expected_step_summary:
        raise RuntimeError("optimizer step summary changed")
    state_receipt = run.get("optimizer_state")
    if (
        not isinstance(state_receipt, Mapping)
        or state_receipt.get("delta_states_complete") is not True
        or state_receipt.get("delta_moment_tensors")
        != 2 * state_receipt.get("delta_parameters_audited", -1)
        or state_receipt.get("all_required_group_states_complete_and_finite") is not True
        or state_receipt.get("registered_missing_state_exemptions")
        != (1 if objective == "state_only" else 0)
    ):
        raise RuntimeError("optimizer state receipt is incomplete")

    tracked_dir = ROOT / "runs" / "training" / f"{capacity}_{objective}_seed{seed}"
    tracked_run = _resolve_repo_path(str(run["tracked_run_path"]))
    tracked_metrics = _resolve_repo_path(str(run["tracked_metrics_path"]))
    tracked_optimizer_steps = _resolve_repo_path(str(run["tracked_optimizer_steps_path"]))
    external_run = canonical_run_dir / "run.json"
    expected_tracked = (
        (tracked_run, tracked_dir / "run.json", external_run),
        (tracked_metrics, tracked_dir / "train_metrics.jsonl", metrics_path),
        (
            tracked_optimizer_steps,
            tracked_dir / "optimizer_steps.jsonl",
            optimizer_steps_path,
        ),
    )
    for tracked, expected_path, external in expected_tracked:
        if (
            tracked != expected_path.resolve()
            or not tracked.is_file()
            or tracked.read_bytes() != external.read_bytes()
        ):
            raise RuntimeError("tracked training receipt mirror is missing or changed")
    return {
        "status": "TRAINING_PROVENANCE_VALID",
        "metrics_rows": len(metric_rows),
        "optimizer_step_rows": len(optimizer_rows),
        "training_order_recomputed": True,
        "tracked_run_path": run["tracked_run_path"],
    }


EVALUATION_ROW_KEYS = {
    "id", "split", "depth", "family", "template", "query_kind",
    "capacity", "objective", "model_seed", "adaptation_mode",
    "node_target", "phase_target", "checksum_target",
    "node_prediction", "phase_prediction", "checksum_prediction",
    "node_final_correct", "phase_final_correct", "checksum_final_correct",
    "joint_final_correct", "node_trajectory_targets", "phase_trajectory_targets",
    "checksum_trajectory_targets", "node_trajectory_predictions",
    "phase_trajectory_predictions", "checksum_trajectory_predictions",
    "node_trajectory_accuracy", "phase_trajectory_accuracy",
    "checksum_trajectory_accuracy", "joint_trajectory_accuracy",
    "answer_choice_target", "answer_choice_prediction", "answer_correct",
    "full_top_is_answer", "answer_token_mass", "state_change_rms_by_transition",
    "mean_state_change_rms", "answer_loss", "state_loss", "fixed_point_loss",
    "prompt_tokens", "base_layer_token_applications",
    "extra_loop_layer_token_applications", "total_layer_token_applications",
    "adaptation_forward_macs", "adaptation_calls",
    "adaptation_call_manifest_sha256", "adaptation_cycles", "compute_proxy",
}


def _exact_eval_matrix(
    config: Mapping[str, Any], eval_set: str
) -> dict[str, tuple[tuple[int, ...], int]]:
    substrate = config["substrate"]
    if eval_set == "trigger":
        return {
            "validation": (
                tuple(map(int, substrate["train_depths"])),
                int(substrate["validation_examples"]) // len(substrate["train_depths"]),
            ),
            "depth_extrapolation": (
                tuple(map(int, substrate["extrapolation_depths"])),
                int(substrate["depth_examples"]) // len(substrate["extrapolation_depths"]),
            ),
            "joint_holdout": (
                tuple(map(int, substrate["extrapolation_depths"])),
                int(substrate["joint_examples"]) // len(substrate["extrapolation_depths"]),
            ),
        }
    if eval_set == "contrast":
        return {
            "contrast_validation": (
                (2, 3, 4),
                int(substrate["contrast_validation_examples"]) // 3,
            ),
            "contrast_depth": (
                tuple(map(int, substrate["extrapolation_depths"])),
                int(substrate["contrast_depth_examples"])
                // len(substrate["extrapolation_depths"]),
            ),
            "contrast_joint": (
                tuple(map(int, substrate["extrapolation_depths"])),
                int(substrate["contrast_joint_examples"])
                // len(substrate["extrapolation_depths"]),
            ),
        }
    raise RuntimeError(f"unknown evaluation set: {eval_set}")


def _strict_int(value: Any, label: str, *, low: int | None = None, high: int | None = None) -> int:
    if type(value) is not int:
        raise RuntimeError(f"{label} must be an exact JSON integer")
    if low is not None and value < low or high is not None and value >= high:
        raise RuntimeError(f"{label} is out of range")
    return value


def _strict_float(
    value: Any, label: str, *, low: float | None = None, high: float | None = None
) -> float:
    if type(value) not in {int, float} or isinstance(value, bool):
        raise RuntimeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise RuntimeError(f"{label} is nonfinite")
    if low is not None and number < low or high is not None and number > high:
        raise RuntimeError(f"{label} is out of range")
    return number


def _validate_evaluation_rows(
    config: Mapping[str, Any],
    rows_by_mode: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    capacity: str,
    objective: str,
    seed: int,
    eval_set: str,
) -> dict[str, Any]:
    if set(rows_by_mode) != {"intact", "disabled"}:
        raise RuntimeError("evaluation must contain exactly intact and disabled modes")
    if capacity not in {"lora", "fullrank"} or objective not in {"joint", "state_only"}:
        raise RuntimeError("evaluation cell is not registered")
    if seed not in SEEDS:
        raise RuntimeError("evaluation seed is not registered")
    matrix = _exact_eval_matrix(config, eval_set)
    train_families = tuple(map(str, config["substrate"]["train_families"]))
    train_templates = tuple(map(str, config["substrate"]["train_templates"]))
    heldout_family = str(config["substrate"]["heldout_family"])
    heldout_template = str(config["substrate"]["heldout_template"])
    node_count = int(config["substrate"]["node_count"])
    checksum_modulus = int(config["substrate"]["checksum_modulus"])
    answer_choices = int(config["substrate"]["num_choices"])
    target_count = int(config["architecture"]["adaptation"][capacity]["expected_targets"])
    adaptation_parameters = int(
        config["architecture"]["adaptation"][capacity]["expected_parameters"]
    )
    mode_keys: dict[str, list[tuple[str, str, str, str, int, str]]] = {}
    counts_receipt = {}
    for mode in ("intact", "disabled"):
        rows = rows_by_mode[mode]
        seen_ids: set[str] = set()
        keys = []
        counts: dict[tuple[str, int], int] = {}
        grid_counts: dict[tuple[str, int, str, str, str], int] = {}
        for index, row in enumerate(rows):
            if set(row) != EVALUATION_ROW_KEYS:
                raise RuntimeError(f"evaluation row schema changed at {mode}/{index}")
            row_id = row["id"]
            if not isinstance(row_id, str) or not row_id or row_id in seen_ids:
                raise RuntimeError("evaluation IDs must be nonempty and unique")
            seen_ids.add(row_id)
            split = row["split"]
            if split not in matrix:
                raise RuntimeError(f"unexpected evaluation split: {split}")
            depth = _strict_int(row["depth"], "depth", low=1)
            if depth not in matrix[split][0]:
                raise RuntimeError("evaluation depth is outside the exact split matrix")
            if (
                row["capacity"] != capacity
                or row["objective"] != objective
                or type(row["model_seed"]) is not int
                or row["model_seed"] != seed
                or row["adaptation_mode"] != mode
            ):
                raise RuntimeError("evaluation row cell identity mismatch")
            expected_families = (
                (heldout_family,) if split in {"joint_holdout", "contrast_joint"}
                else train_families
            )
            expected_templates = (
                (heldout_template,) if split in {"joint_holdout", "contrast_joint"}
                else train_templates
            )
            if row["family"] not in expected_families or row["template"] not in expected_templates:
                raise RuntimeError("evaluation split family/template geometry changed")
            if row["query_kind"] not in {"node", "checksum"}:
                raise RuntimeError("evaluation query kind is unregistered")
            scalar_specs = (
                ("node_target", node_count), ("node_prediction", node_count),
                ("phase_target", 2), ("phase_prediction", 2),
                ("checksum_target", checksum_modulus),
                ("checksum_prediction", checksum_modulus),
            )
            for field, bound in scalar_specs:
                _strict_int(row[field], field, low=0, high=bound)
            for field in (
                "node_final_correct", "phase_final_correct", "checksum_final_correct",
                "joint_final_correct", "answer_correct", "full_top_is_answer",
            ):
                if type(row[field]) is not bool:
                    raise RuntimeError(f"{field} must be an exact JSON boolean")
            trajectory_specs = (
                ("node_trajectory_targets", node_count),
                ("node_trajectory_predictions", node_count),
                ("phase_trajectory_targets", 2),
                ("phase_trajectory_predictions", 2),
                ("checksum_trajectory_targets", checksum_modulus),
                ("checksum_trajectory_predictions", checksum_modulus),
            )
            for field, bound in trajectory_specs:
                values = row[field]
                if not isinstance(values, list) or len(values) != depth:
                    raise RuntimeError(f"{field} has the wrong trajectory length")
                for value in values:
                    _strict_int(value, field, low=0, high=bound)
            node_correct = [
                left == right for left, right in zip(
                    row["node_trajectory_predictions"], row["node_trajectory_targets"]
                )
            ]
            phase_correct = [
                left == right for left, right in zip(
                    row["phase_trajectory_predictions"], row["phase_trajectory_targets"]
                )
            ]
            checksum_correct = [
                left == right for left, right in zip(
                    row["checksum_trajectory_predictions"],
                    row["checksum_trajectory_targets"],
                )
            ]
            joint_correct = [
                node and phase and checksum
                for node, phase, checksum in zip(
                    node_correct, phase_correct, checksum_correct
                )
            ]
            expected_tail = {
                "node_target": row["node_trajectory_targets"][-1],
                "node_prediction": row["node_trajectory_predictions"][-1],
                "phase_target": row["phase_trajectory_targets"][-1],
                "phase_prediction": row["phase_trajectory_predictions"][-1],
                "checksum_target": row["checksum_trajectory_targets"][-1],
                "checksum_prediction": row["checksum_trajectory_predictions"][-1],
                "node_final_correct": node_correct[-1],
                "phase_final_correct": phase_correct[-1],
                "checksum_final_correct": checksum_correct[-1],
                "joint_final_correct": joint_correct[-1],
            }
            if any(row[field] != value for field, value in expected_tail.items()):
                raise RuntimeError("evaluation terminal diagnostics contradict trajectories")
            expected_accuracies = {
                "node_trajectory_accuracy": sum(node_correct) / depth,
                "phase_trajectory_accuracy": sum(phase_correct) / depth,
                "checksum_trajectory_accuracy": sum(checksum_correct) / depth,
                "joint_trajectory_accuracy": sum(joint_correct) / depth,
            }
            for field, expected_value in expected_accuracies.items():
                actual = _strict_float(row[field], field, low=0.0, high=1.0)
                if not math.isclose(actual, expected_value, abs_tol=1e-12):
                    raise RuntimeError(f"{field} is internally inconsistent")
            _strict_int(row["answer_choice_target"], "answer_choice_target", low=0, high=answer_choices)
            _strict_int(
                row["answer_choice_prediction"], "answer_choice_prediction",
                low=0, high=answer_choices,
            )
            if row["answer_correct"] != (
                row["answer_choice_target"] == row["answer_choice_prediction"]
            ):
                raise RuntimeError("answer correctness is internally inconsistent")
            _strict_float(row["answer_token_mass"], "answer_token_mass", low=0.0, high=1.0)
            changes = row["state_change_rms_by_transition"]
            if not isinstance(changes, list) or len(changes) != depth - 1:
                raise RuntimeError("state-change trajectory has the wrong length")
            if any(type(value) is not float or not math.isfinite(value) or value < 0 for value in changes):
                raise RuntimeError("state-change trajectory must contain finite nonnegative floats")
            expected_change = sum(changes) / len(changes) if changes else 0.0
            actual_change = _strict_float(
                row["mean_state_change_rms"], "mean_state_change_rms", low=0.0
            )
            if not math.isclose(actual_change, expected_change, abs_tol=1e-12):
                raise RuntimeError("mean state change is internally inconsistent")
            for field in ("answer_loss", "state_loss", "fixed_point_loss"):
                _strict_float(row[field], field, low=0.0)
            prompt_tokens = _strict_int(row["prompt_tokens"], "prompt_tokens", low=1)
            base_compute = prompt_tokens * int(config["architecture"]["expected_num_layers"])
            extra_compute = prompt_tokens * (
                int(config["architecture"]["loop_end"])
                - int(config["architecture"]["loop_start"])
            ) * (depth - 1)
            active_cycles = depth - 1 if mode == "intact" else 0
            expected_compute = {
                "base_layer_token_applications": base_compute,
                "extra_loop_layer_token_applications": extra_compute,
                "total_layer_token_applications": base_compute + extra_compute,
                "adaptation_forward_macs": (
                    prompt_tokens * adaptation_parameters * active_cycles
                ),
                "adaptation_calls": target_count * active_cycles,
                "adaptation_cycles": active_cycles,
            }
            for field, expected_value in expected_compute.items():
                if _strict_int(row[field], field, low=0) != expected_value:
                    raise RuntimeError(f"{field} is inconsistent with registered geometry")
            digest = row["adaptation_call_manifest_sha256"]
            if (
                not isinstance(digest, str) or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise RuntimeError("adaptation call-manifest digest is malformed")
            if row["compute_proxy"] != (
                "exact_layer_token_applications_and_adapter_linear_macs;not_hardware_flops"
            ):
                raise RuntimeError("compute proxy definition changed")
            counts[(split, depth)] = counts.get((split, depth), 0) + 1
            grid_key = (
                split, depth, str(row["family"]), str(row["template"]),
                str(row["query_kind"]),
            )
            grid_counts[grid_key] = grid_counts.get(grid_key, 0) + 1
            keys.append(
                (
                    split, row_id, str(row["family"]), str(row["template"]),
                    depth, str(row["query_kind"]),
                )
            )
        expected_counts = {
            (split, depth): count
            for split, (depths, count) in matrix.items()
            for depth in depths
        }
        if counts != expected_counts:
            raise RuntimeError("evaluation split/depth cardinality matrix is incomplete")
        expected_grid = {}
        for split, (depths, count) in matrix.items():
            families = (
                (heldout_family,) if split in {"joint_holdout", "contrast_joint"}
                else train_families
            )
            templates = (
                (heldout_template,) if split in {"joint_holdout", "contrast_joint"}
                else train_templates
            )
            cell_count = count // (len(families) * len(templates) * 2)
            for depth in depths:
                for family in families:
                    for template in templates:
                        for query_kind in ("node", "checksum"):
                            expected_grid[(split, depth, family, template, query_kind)] = cell_count
        if grid_counts != expected_grid:
            raise RuntimeError("evaluation family/template/query grid is incomplete")
        mode_keys[mode] = keys
        counts_receipt[mode] = {
            f"{split}|depth={depth}": count
            for (split, depth), count in sorted(counts.items())
        }
    if mode_keys["intact"] != mode_keys["disabled"]:
        raise RuntimeError("intact/disabled task keys or order differ")
    return {
        "status": "EXACT_EVALUATION_MATRIX_VALID",
        "eval_set": eval_set,
        "counts": counts_receipt,
        "unique_tasks": len(mode_keys["intact"]),
    }


def _load_evaluation(
    config: Mapping[str, Any],
    runs_dir: Path,
    *,
    capacity: str,
    objective: str,
    seed: int,
    eval_set: str,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]]:
    directory = runs_dir / f"{capacity}_{objective}_seed{seed}_{eval_set}"
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise RuntimeError(f"evaluation summary is missing: {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {
        **_identity(config, f"{capacity}_{objective}_{eval_set}_evaluation"),
        "status": "STATE_EVALUATION_COMPLETE",
        "capacity": capacity,
        "objective": objective,
        "model_seed": seed,
        "eval_set": eval_set,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise RuntimeError(f"evaluation {summary_path} has mismatched {key}")
    claimed = summary.get("receipt_identity_sha256")
    payload = {key: value for key, value in summary.items() if key != "receipt_identity_sha256"}
    if claimed != _canonical_sha256(payload):
        raise RuntimeError(f"evaluation receipt identity mismatch: {summary_path}")
    parity = _strict_float(
        summary.get("k1_max_logit_abs_error"), "evaluation K=1 parity", low=0.0
    )
    if parity > float(config["gates"]["k1_max_logit_abs_error"]):
        raise RuntimeError("evaluation K=1 parity exceeds the frozen bound")
    if summary.get("k1_adaptation_calls") != 0:
        raise RuntimeError("evaluation K=1 unexpectedly called adaptation")
    data_dir = (ROOT / str(config["paths"]["data_dir"])).resolve()
    data_manifest_path = data_dir / "manifest.json"
    if not data_manifest_path.is_file():
        raise RuntimeError("analysis data manifest is missing")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    expected_splits = (
        list(config["evaluation"]["trigger_splits"])
        if eval_set == "trigger" else list(config["evaluation"]["sealed_contrast_splits"])
    )
    validate_data_manifest(
        config, data_dir, data_manifest, content_splits=set(expected_splits)
    )
    if summary.get("data_manifest_sha256") != _sha256(data_manifest_path):
        raise RuntimeError("evaluation/live data manifest mismatch")
    expected_payloads = {
        split: {
            "sha256": data_manifest["files"][split]["sha256"],
            "canonical_rows": data_manifest["files"][split]["canonical_rows"],
        }
        for split in expected_splits
    }
    if summary.get("split_payloads") != expected_payloads:
        raise RuntimeError("evaluation split payload lineage mismatch")
    checkpoint_path = _resolve_repo_path(str(summary["checkpoint_path"]))
    checkpoint_metadata = checkpoint_path / "checkpoint.json"
    if not checkpoint_metadata.is_file() or _sha256(checkpoint_metadata) != summary["checkpoint_metadata_sha256"]:
        raise RuntimeError("evaluation checkpoint metadata changed")
    checkpoint = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
    expected_checkpoint = {
        **_identity(config, f"{capacity}_{objective}_training"),
        "capacity": capacity,
        "objective": objective,
        "model_seed": seed,
        "step": int(config["training"]["train_steps"]),
        "data_manifest_sha256": summary["data_manifest_sha256"],
    }
    for key, value in expected_checkpoint.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"evaluation checkpoint {key} mismatch")
    if (
        checkpoint.get("checkpoint_identity_sha256") != _checkpoint_identity(checkpoint)
        or checkpoint.get("checkpoint_identity_sha256")
        != summary.get("checkpoint_identity_sha256")
    ):
        raise RuntimeError("evaluation checkpoint identity mismatch")
    setup_gate_receipts = {
        lineage_key: _validate_lineage_entry(checkpoint[lineage_key])
        for lineage_key in ("g0_lineage", "positive_control_lineage")
    }
    branch_lineage = checkpoint.get("branch_authorization_lineage")
    if branch_lineage is not None:
        _validate_lineage_entry(branch_lineage)
    shared_initialization = checkpoint.get("shared_initialization")
    if not isinstance(shared_initialization, Mapping):
        raise RuntimeError("checkpoint shared initialization is malformed")
    initialization_path = _resolve_repo_path(str(shared_initialization.get("bundle_path")))
    _, reopened_initialization = load_initialization_bundle(
        config, seed, initialization_path
    )
    if reopened_initialization != shared_initialization:
        raise RuntimeError("checkpoint shared initialization receipt changed")
    for filename, digest_key in (
        ("adaptation_state.pt", "adaptation_state_sha256"),
        ("loop_state.pt", "loop_state_sha256"),
    ):
        payload_path = checkpoint_path / filename
        if not payload_path.is_file() or _sha256(payload_path) != checkpoint[digest_key]:
            raise RuntimeError("checkpoint tensor payload changed")
    run_path = checkpoint_path.parent / "run.json"
    if not run_path.is_file():
        raise RuntimeError("training run index is missing")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run_claimed = run.get("receipt_identity_sha256")
    if run_claimed != _canonical_sha256(
        {key: value for key, value in run.items() if key != "receipt_identity_sha256"}
    ):
        raise RuntimeError("training run identity mismatch")
    if (
        run.get("checkpoint_metadata_sha256") != summary["checkpoint_metadata_sha256"]
        or run.get("checkpoint_identity_sha256") != summary["checkpoint_identity_sha256"]
        or run.get("checkpoint_path") != summary["checkpoint_path"]
    ):
        raise RuntimeError("training run/checkpoint lineage mismatch")
    run_setup = run.get("setup")
    if not isinstance(run_setup, Mapping):
        raise RuntimeError("training run omits deterministic setup")
    for label, gate_receipt in setup_gate_receipts.items():
        gate_setup = gate_receipt.get("setup")
        if (
            not isinstance(gate_setup, Mapping)
            or _stable_setup_receipt(gate_setup) != _stable_setup_receipt(run_setup)
        ):
            raise RuntimeError(f"training setup differs from exact {label}")
    training_provenance = _validate_training_payloads(
        config,
        run=run,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        capacity=capacity,
        objective=objective,
        seed=seed,
    )
    rows_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in ("intact", "disabled"):
        mode_receipt = summary["modes"][mode]
        if mode_receipt.get("rows_path") != f"rows_{mode}.jsonl":
            raise RuntimeError("evaluation row payload name changed")
        rows_path = directory / str(mode_receipt["rows_path"])
        if not rows_path.is_file() or _sha256(rows_path) != mode_receipt["rows_sha256"]:
            raise RuntimeError(f"evaluation row payload changed: {rows_path}")
        rows = _read_jsonl(rows_path)
        if len(rows) != int(mode_receipt["rows"]):
            raise RuntimeError("evaluation row count changed")
        rows_by_mode[mode] = rows
    intact_keys = {(row["split"], row["id"]) for row in rows_by_mode["intact"]}
    disabled_keys = {(row["split"], row["id"]) for row in rows_by_mode["disabled"]}
    if intact_keys != disabled_keys or len(intact_keys) != len(rows_by_mode["intact"]):
        raise RuntimeError("intact/disabled evaluation keys differ or duplicate")
    matrix_receipt = _validate_evaluation_rows(
        config, rows_by_mode, capacity=capacity, objective=objective,
        seed=seed, eval_set=eval_set,
    )
    corpus_rows = [
        (split, row)
        for split in expected_splits
        for row in read_jsonl(data_dir / f"{split}.jsonl.gz")
    ]
    expected_task_sequence = [
        (
            split, str(row["id"]), str(row["family"]), str(row["template"]),
            int(row["depth"]), str(row["query_kind"]),
        )
        for split, row in corpus_rows
    ]
    for _, row in corpus_rows:
        verify_example(
            row,
            str(config["architecture"]["state_token"]),
            int(config["architecture"]["state_slots"]),
        )
    for mode in ("intact", "disabled"):
        observed = [
            (
                str(row["split"]), str(row["id"]), str(row["family"]),
                str(row["template"]), int(row["depth"]), str(row["query_kind"]),
            )
            for row in rows_by_mode[mode]
        ]
        if observed != expected_task_sequence:
            raise RuntimeError("evaluation rows are detached from the exact bound task corpus")
        for evaluation_row, (_, corpus_row) in zip(rows_by_mode[mode], corpus_rows):
            targets = trajectory_targets(corpus_row, int(corpus_row["depth"]))
            expected_targets = {
                "node_trajectory_targets": targets["node"],
                "phase_trajectory_targets": targets["phase"],
                "checksum_trajectory_targets": targets["checksum"],
                "node_target": targets["node"][-1],
                "phase_target": targets["phase"][-1],
                "checksum_target": targets["checksum"][-1],
                "answer_choice_target": int(corpus_row["correct_choice"]),
            }
            if any(
                evaluation_row.get(field) != value
                for field, value in expected_targets.items()
            ):
                raise RuntimeError("evaluation targets differ from exact corpus truth")
    manifest = {
        "summary_path": summary_path.relative_to(REPO_ROOT).as_posix(),
        "summary_sha256": _sha256(summary_path),
        "receipt_identity_sha256": claimed,
        "checkpoint_identity_sha256": summary["checkpoint_identity_sha256"],
        "checkpoint_path": summary["checkpoint_path"],
        "checkpoint_metadata_sha256": summary["checkpoint_metadata_sha256"],
        "checkpoint": checkpoint,
        "run": run,
        "training_provenance": training_provenance,
        "rows": {
            mode: {
                "sha256": summary["modes"][mode]["rows_sha256"],
                "count": summary["modes"][mode]["rows"],
            }
            for mode in ("intact", "disabled")
        },
        "matrix_validation": matrix_receipt,
    }
    return summary, rows_by_mode, manifest


def _cells(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[int, float]]:
    totals: dict[str, dict[int, list[int]]] = {}
    for row in rows:
        if type(row.get("joint_final_correct")) is not bool:
            raise RuntimeError("joint_final_correct must be an exact boolean")
        split = str(row["split"])
        depth = int(row["depth"])
        bucket = totals.setdefault(split, {}).setdefault(depth, [0, 0])
        bucket[0] += int(row["joint_final_correct"])
        bucket[1] += 1
    return {
        split: {depth: correct / count for depth, (correct, count) in sorted(depths.items())}
        for split, depths in sorted(totals.items())
    }


def _required_depths(split: str) -> tuple[int, ...]:
    if split in {"validation", "contrast_validation"}:
        # K=1 has no trained extra-call adaptation and remains a diagnostic.
        return (2, 3, 4)
    return tuple(range(5, 13))


def _formation_summary(
    bundles: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    threshold: float,
    *, primary_mode: str = "intact",
) -> dict[str, Any]:
    if primary_mode not in {"intact", "disabled"}:
        raise ValueError(primary_mode)
    if set(bundles) != set(SEEDS):
        return {
            "passes": False, "status": "EVIDENCE_INCOMPLETE", "threshold": threshold,
            "primary_mode": primary_mode, "per_seed": {},
        }
    observed_splits = {
        str(row.get("split"))
        for seed in SEEDS for row in bundles[seed].get(primary_mode, [])
    }
    trigger_matrix = {
        "validation": (1, 2, 3, 4),
        "depth_extrapolation": tuple(range(5, 13)),
        "joint_holdout": tuple(range(5, 13)),
    }
    contrast_matrix = {
        "contrast_validation": (2, 3, 4),
        "contrast_depth": tuple(range(5, 13)),
        "contrast_joint": tuple(range(5, 13)),
    }
    if observed_splits == set(trigger_matrix):
        matrix = trigger_matrix
        matrix_kind = "trigger"
    elif observed_splits == set(contrast_matrix):
        matrix = contrast_matrix
        matrix_kind = "contrast"
    else:
        return {
            "passes": False, "status": "EVIDENCE_INCOMPLETE", "threshold": threshold,
            "primary_mode": primary_mode, "per_seed": {},
        }
    per_seed: dict[str, Any] = {}
    incomplete = False
    category_passes = {"trained": True, "depth": True, "joint": True}
    for seed in SEEDS:
        if set(bundles[seed]) != {"intact", "disabled"}:
            incomplete = True
            continue
        intact = _cells(bundles[seed]["intact"])
        disabled = _cells(bundles[seed]["disabled"])
        if set(intact) != set(matrix) or set(disabled) != set(matrix):
            incomplete = True
            continue
        cells = {}
        for split, expected_depths in matrix.items():
            if set(intact[split]) != set(expected_depths) or set(disabled[split]) != set(expected_depths):
                incomplete = True
                continue
            split_cells = {}
            for depth in expected_depths:
                accuracy = intact[split][depth]
                disabled_accuracy = disabled[split][depth]
                required = depth in _required_depths(split)
                primary_accuracy = accuracy if primary_mode == "intact" else disabled_accuracy
                passes = (primary_accuracy >= threshold) if required else None
                if required and not passes:
                    if split in {"validation", "contrast_validation"}:
                        category_passes["trained"] = False
                    elif split in {"depth_extrapolation", "contrast_depth"}:
                        category_passes["depth"] = False
                    else:
                        category_passes["joint"] = False
                split_cells[str(depth)] = {
                    "intact_joint_final_accuracy": accuracy,
                    "disabled_joint_final_accuracy": disabled_accuracy,
                    "intact_minus_disabled": accuracy - disabled_accuracy,
                    "primary_accuracy": primary_accuracy,
                    "required": required,
                    "passes": passes,
                }
            cells[split] = split_cells
        per_seed[str(seed)] = cells
    if incomplete or len(per_seed) != len(SEEDS):
        status = "EVIDENCE_INCOMPLETE"
    elif not category_passes["trained"]:
        status = "TRAINED_DEPTH_MISS"
    elif not category_passes["depth"]:
        status = "TRAINED_PASS_DEPTH_EXTRAPOLATION_MISS"
    elif not category_passes["joint"]:
        status = "TRAINED_AND_DEPTH_PASS_JOINT_SHIFT_MISS"
    else:
        status = "STATE_FORMATION_PASS"
    return {
        "passes": status == "STATE_FORMATION_PASS",
        "status": status,
        "threshold": threshold,
        "primary_mode": primary_mode,
        "matrix": matrix_kind,
        "category_passes": category_passes,
        "per_seed": per_seed,
    }


def _paired_records(
    bundles: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    split: str,
    *,
    left_mode: str = "intact",
    right_mode: str = "disabled",
) -> dict[int, dict[str, float]]:
    result = {}
    for seed in SEEDS:
        left = {
            str(row["id"]): float(bool(row["joint_final_correct"]))
            for row in bundles[seed][left_mode] if row["split"] == split
        }
        right = {
            str(row["id"]): float(bool(row["joint_final_correct"]))
            for row in bundles[seed][right_mode] if row["split"] == split
        }
        if left.keys() != right.keys():
            raise RuntimeError(f"paired task keys differ in {split}/seed{seed}")
        result[seed] = {task: left[task] - right[task] for task in left}
    return result


def _crossed_bootstrap(
    records: Mapping[int, Mapping[str, float]],
    *,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
    model_seeds = sorted(records)
    task_ids = sorted(next(iter(records.values())))
    if any(sorted(records[item]) != task_ids for item in model_seeds):
        raise RuntimeError("crossed bootstrap task sets differ across seeds")
    point = sum(records[s][task] for s in model_seeds for task in task_ids) / (
        len(model_seeds) * len(task_ids)
    )
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        sampled_seeds = [rng.choice(model_seeds) for _ in model_seeds]
        sampled_tasks = [rng.choice(task_ids) for _ in task_ids]
        samples.append(
            sum(records[s][task] for s in sampled_seeds for task in sampled_tasks)
            / (len(sampled_seeds) * len(sampled_tasks))
        )
    samples.sort()
    return {
        "point": point,
        "ci95": [samples[int(0.025 * resamples)], samples[min(resamples - 1, int(0.975 * resamples))]],
        "model_seeds": model_seeds,
        "tasks": len(task_ids),
        "bootstrap_unit": "crossed_model_seed_by_task",
        "resamples": resamples,
    }


def _adaptation_effects(
    config: Mapping[str, Any],
    bundles: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"])
    splits = sorted({str(row["split"]) for row in bundles[SEEDS[0]]["intact"]})
    effects = {}
    every_seed_positive = True
    no_depth_negative = True
    minimum_seed_effect = float(
        config["gates"]["min_adaptation_intact_minus_disabled_each_seed"]
    )
    for index, split in enumerate(splits):
        records = _paired_records(bundles, split)
        seed_points = {
            str(seed): sum(values.values()) / len(values)
            for seed, values in records.items()
        }
        if any(point <= minimum_seed_effect for point in seed_points.values()):
            every_seed_positive = False
        depths = sorted({
            int(row["depth"])
            for row in bundles[SEEDS[0]]["intact"] if row["split"] == split
        })
        depth_points = {}
        for depth in depths:
            values = []
            for seed in SEEDS:
                intact = {
                    str(row["id"]): int(row["joint_final_correct"])
                    for row in bundles[seed]["intact"]
                    if row["split"] == split and int(row["depth"]) == depth
                }
                disabled = {
                    str(row["id"]): int(row["joint_final_correct"])
                    for row in bundles[seed]["disabled"]
                    if row["split"] == split and int(row["depth"]) == depth
                }
                if intact.keys() != disabled.keys() or not intact:
                    raise RuntimeError("adaptation depth contrast task keys differ")
                values.extend(intact[key] - disabled[key] for key in intact)
            depth_points[str(depth)] = sum(values) / len(values)
        if any(point < 0.0 for point in depth_points.values()):
            no_depth_negative = False
        crossed = _crossed_bootstrap(
            records,
            resamples=resamples,
            seed=base_seed + index,
        )
        effects[split] = {
            **crossed,
            "per_seed_points": seed_points,
            "per_depth_points": depth_points,
        }
    all_lcb_positive = all(effects[split]["ci95"][0] > 0 for split in splits)
    threshold = float(config["gates"]["min_final_joint_accuracy_each_seed_depth"])
    intact_formation = _formation_summary(bundles, threshold, primary_mode="intact")
    disabled_formation = _formation_summary(bundles, threshold, primary_mode="disabled")
    if intact_formation["passes"] and disabled_formation["passes"]:
        status = "ADAPTATION_NOT_REQUIRED_AT_INFERENCE"
    elif not intact_formation["passes"] and disabled_formation["passes"]:
        # Branching still follows the preregistered intact checkpoint, but a
        # common-state path that passes only with the learned adapter removed
        # is scientifically different from generic contrast uncertainty.
        status = "ADAPTATION_DISABLED_REVERSAL"
    elif (
        intact_formation["passes"] and every_seed_positive
        and no_depth_negative and all_lcb_positive
    ):
        status = "ADAPTATION_REQUIRED"
    else:
        status = "ADAPTATION_CONTRAST_UNCERTAIN"
    return {
        "status": status,
        "intact_formation": intact_formation,
        "disabled_formation": disabled_formation,
        "every_seed_point_positive": every_seed_positive,
        "no_depth_point_negative": no_depth_negative,
        "every_split_crossed_lcb_positive": all_lcb_positive,
        "passes": status == "ADAPTATION_REQUIRED",
        "adapter_disabled_reversal": status == "ADAPTATION_DISABLED_REVERSAL",
        "splits": effects,
    }


def _failure_category_replication(
    trigger: Mapping[str, Any], sealed: Mapping[str, Any]
) -> dict[str, Any]:
    """Require every trigger-failed formation domain to fail again on fresh rows."""

    categories = ("trained", "depth", "joint")
    trigger_categories = trigger.get("category_passes")
    sealed_categories = sealed.get("category_passes")
    if (
        not isinstance(trigger_categories, Mapping)
        or not isinstance(sealed_categories, Mapping)
        or set(trigger_categories) != set(categories)
        or set(sealed_categories) != set(categories)
        or any(type(trigger_categories[item]) is not bool for item in categories)
        or any(type(sealed_categories[item]) is not bool for item in categories)
    ):
        raise RuntimeError("formation category receipts are incomplete")
    trigger_failed = [item for item in categories if not trigger_categories[item]]
    sealed_failed = [item for item in categories if not sealed_categories[item]]
    if not trigger_failed:
        raise RuntimeError("failure replication requires a trigger formation miss")
    missing_replications = [
        item for item in trigger_failed if sealed_categories[item]
    ]
    return {
        "status": (
            "TRIGGER_FAILURE_CATEGORIES_REPLICATED"
            if not missing_replications
            else "TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED"
        ),
        "trigger_failed_categories": trigger_failed,
        "sealed_failed_categories": sealed_failed,
        "missing_replications": missing_replications,
        "passes": not missing_replications,
    }


def _load_cell(
    config: Mapping[str, Any], runs_dir: Path, capacity: str, objective: str, eval_set: str
) -> tuple[dict[int, dict[str, list[dict[str, Any]]]], list[dict[str, Any]]]:
    bundles = {}
    manifests = []
    for seed in SEEDS:
        _, rows, manifest = _load_evaluation(
            config, runs_dir, capacity=capacity, objective=objective, seed=seed, eval_set=eval_set
        )
        bundles[seed] = rows
        manifests.append(manifest)
    for mode in ("intact", "disabled"):
        reference = [
            (row["split"], row["id"], row["depth"])
            for row in bundles[SEEDS[0]][mode]
        ]
        for seed in SEEDS[1:]:
            observed = [
                (row["split"], row["id"], row["depth"])
                for row in bundles[seed][mode]
            ]
            if observed != reference:
                raise RuntimeError("evaluation task keys/order differ across model seeds")
    return bundles, manifests


def _load_analysis_authorization(
    config: Mapping[str, Any], path: Path, statuses: set[str], phases: set[str]
) -> dict[str, Any]:
    path = _resolve_repo_path(str(path))
    if not path.is_file():
        raise RuntimeError(f"analysis authorization is missing: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    status = str(receipt.get("status", receipt.get("verdict", "")))
    phase = str(receipt.get("phase", ""))
    if status not in statuses or phase not in phases:
        raise RuntimeError("analysis authorization status/phase mismatch")
    for key, value in _identity(config, phase).items():
        if receipt.get(key) != value:
            raise RuntimeError(f"analysis authorization {key} mismatch")
    claimed = receipt.get("receipt_identity_sha256")
    payload = {key: value for key, value in receipt.items() if key != "receipt_identity_sha256"}
    if claimed != _canonical_sha256(payload):
        raise RuntimeError("analysis authorization identity mismatch")
    return {
        "path": path.relative_to(REPO_ROOT).as_posix(),
        "sha256": _sha256(path),
        "receipt_identity_sha256": claimed,
        "status": status,
        "phase": phase,
    }


def _fullrank_minus_lora_contrast(
    config: Mapping[str, Any],
    fullrank: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
    lora: Mapping[int, Mapping[str, Sequence[Mapping[str, Any]]]],
) -> dict[str, Any]:
    resamples = int(config["evaluation"]["bootstrap_resamples"])
    base_seed = int(config["evaluation"]["bootstrap_seed"]) + 100
    splits = sorted({str(row["split"]) for row in fullrank[SEEDS[0]]["intact"]})
    result = {}
    every_seed_positive = True
    no_depth_negative = True
    for index, split in enumerate(splits):
        records = {}
        for seed in SEEDS:
            left = {
                str(row["id"]): float(bool(row["joint_final_correct"]))
                for row in fullrank[seed]["intact"] if row["split"] == split
            }
            right = {
                str(row["id"]): float(bool(row["joint_final_correct"]))
                for row in lora[seed]["intact"] if row["split"] == split
            }
            if left.keys() != right.keys():
                raise RuntimeError("fullrank/LoRA sealed contrast task keys differ")
            records[seed] = {task: left[task] - right[task] for task in left}
        seed_points = {
            str(seed): sum(records[seed].values()) / len(records[seed]) for seed in SEEDS
        }
        if any(point <= 0.0 for point in seed_points.values()):
            every_seed_positive = False
        depths = sorted({
            int(row["depth"])
            for row in fullrank[SEEDS[0]]["intact"] if row["split"] == split
        })
        depth_points = {}
        for depth in depths:
            values = []
            for seed in SEEDS:
                left = {
                    str(row["id"]): int(row["joint_final_correct"])
                    for row in fullrank[seed]["intact"]
                    if row["split"] == split and int(row["depth"]) == depth
                }
                right = {
                    str(row["id"]): int(row["joint_final_correct"])
                    for row in lora[seed]["intact"]
                    if row["split"] == split and int(row["depth"]) == depth
                }
                if left.keys() != right.keys() or not left:
                    raise RuntimeError("fullrank/LoRA depth task keys differ")
                values.extend(left[key] - right[key] for key in left)
            depth_points[str(depth)] = sum(values) / len(values)
        if any(point < 0.0 for point in depth_points.values()):
            no_depth_negative = False
        crossed = _crossed_bootstrap(
            records, resamples=resamples, seed=base_seed + index
        )
        result[split] = {
            **crossed, "per_seed_points": seed_points, "per_depth_points": depth_points,
        }
    lcb_positive = all(item["ci95"][0] > 0 for item in result.values())
    return {
        "passes": every_seed_positive and no_depth_negative and lcb_positive,
        "every_seed_point_positive": every_seed_positive,
        "no_depth_point_negative": no_depth_negative,
        "every_split_crossed_lcb_positive": lcb_positive,
        "splits": result,
    }


def _manifest_by_seed(manifests: Sequence[Mapping[str, Any]]) -> dict[int, Mapping[str, Any]]:
    result = {}
    for manifest in manifests:
        checkpoint = manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("evaluation manifest omits checkpoint metadata")
        seed = checkpoint.get("model_seed")
        if type(seed) is not int or seed in result:
            raise RuntimeError("evaluation manifests have duplicate/invalid model seed")
        result[seed] = manifest
    if set(result) != set(SEEDS):
        raise RuntimeError("evaluation manifests do not cover all registered seeds")
    return result


def _evaluation_lineage(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "summary_path": manifest["summary_path"],
        "summary_sha256": manifest["summary_sha256"],
        "receipt_identity_sha256": manifest["receipt_identity_sha256"],
        "checkpoint_path": manifest["checkpoint_path"],
        "checkpoint_metadata_sha256": manifest["checkpoint_metadata_sha256"],
        "checkpoint_identity_sha256": manifest["checkpoint_identity_sha256"],
    }


def _stage_b_matching_receipt(
    lora_joint_manifests: Sequence[Mapping[str, Any]],
    lora_control_manifests: Sequence[Mapping[str, Any]],
    fullrank_joint_manifests: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {
        "lora_joint": _manifest_by_seed(lora_joint_manifests),
        "lora_state_only": _manifest_by_seed(lora_control_manifests),
        "fullrank_joint": _manifest_by_seed(fullrank_joint_manifests),
    }
    per_seed = {}
    for seed in SEEDS:
        checkpoints = {name: manifests[seed]["checkpoint"] for name, manifests in groups.items()}
        runs = {name: manifests[seed]["run"] for name, manifests in groups.items()}
        initializations = {
            name: checkpoint["shared_initialization"] for name, checkpoint in checkpoints.items()
        }
        if len({json.dumps(item, sort_keys=True) for item in initializations.values()}) != 1:
            raise RuntimeError("Stage-B shared initialization differs across reached arms")
        pairing_setups = {
            name: _pairing_setup_receipt(run["setup"])
            for name, run in runs.items()
        }
        if len({json.dumps(item, sort_keys=True) for item in pairing_setups.values()}) != 1:
            raise RuntimeError("Stage-B hardware/environment/target setup differs across arms")
        for field in ("data_manifest_sha256", "training_order_sha256", "dropout_schedule_sha256"):
            if len({str(run[field]) for run in runs.values()}) != 1:
                raise RuntimeError(f"Stage-B matched {field} differs across reached arms")
        if len({json.dumps(run["dropout_probes"], sort_keys=True) for run in runs.values()}) != 1:
            raise RuntimeError("Stage-B realized dropout probes differ across reached arms")
        if checkpoints["lora_joint"]["g0_lineage"] != checkpoints["lora_state_only"]["g0_lineage"]:
            raise RuntimeError("LoRA objectives do not share the exact G0 receipt")
        if (
            checkpoints["lora_joint"]["positive_control_lineage"]
            != checkpoints["lora_state_only"]["positive_control_lineage"]
        ):
            raise RuntimeError("LoRA objectives do not share the exact positive control")
        if checkpoints["lora_joint"].get("branch_authorization_lineage") is not None:
            raise RuntimeError("LoRA joint unexpectedly has branch authorization")
        for name in ("lora_state_only", "fullrank_joint"):
            if checkpoints[name].get("branch_authorization_lineage") != dict(authorization):
                raise RuntimeError("Stage-B checkpoint changed the LoRA-miss authorization")
        lora_g0 = _validate_lineage_entry(checkpoints["lora_joint"]["g0_lineage"])
        fullrank_g0 = _validate_lineage_entry(checkpoints["fullrank_joint"]["g0_lineage"])
        lora_probes = [
            step["dropout_probe"] for step in lora_g0["two_step_gradient_probe"]
        ]
        fullrank_probes = [
            step["dropout_probe"] for step in fullrank_g0["two_step_gradient_probe"]
        ]
        if lora_probes != fullrank_probes:
            raise RuntimeError("LoRA/full-rank G0 realized dropout schedules differ")
        per_seed[str(seed)] = {
            "shared_initialization_receipt_identity_sha256": initializations[
                "lora_joint"
            ]["receipt_identity_sha256"],
            "pairing_setup_sha256": _canonical_sha256(pairing_setups["lora_joint"]),
            "training_order_sha256": runs["lora_joint"]["training_order_sha256"],
            "dropout_schedule_sha256": runs["lora_joint"]["dropout_schedule_sha256"],
            "dropout_probes": runs["lora_joint"]["dropout_probes"],
            "lora_g0_lineage": checkpoints["lora_joint"]["g0_lineage"],
            "fullrank_g0_lineage": checkpoints["fullrank_joint"]["g0_lineage"],
            "checkpoint_identities": {
                name: checkpoint["checkpoint_identity_sha256"]
                for name, checkpoint in checkpoints.items()
            },
            "checkpoint_lineages": {
                name: {
                    "path": groups[name][seed]["checkpoint_path"],
                    "metadata_sha256": groups[name][seed]["checkpoint_metadata_sha256"],
                    "checkpoint_identity_sha256": checkpoints[name][
                        "checkpoint_identity_sha256"
                    ],
                }
                for name in ("lora_joint", "fullrank_joint")
            },
            "trigger_evaluation_lineages": {
                name: _evaluation_lineage(groups[name][seed])
                for name in ("lora_joint", "lora_state_only", "fullrank_joint")
            },
        }
    return {"status": "STAGE_B_MATCHING_VALID", "per_seed": per_seed}


def _require_stage_b_trigger_match(
    manifests: Sequence[Mapping[str, Any]],
    *,
    arm: str,
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    stage_b = _validate_lineage_entry(authorization)
    matching = stage_b.get("matching")
    if not isinstance(matching, Mapping) or matching.get("status") != "STAGE_B_MATCHING_VALID":
        raise RuntimeError("Stage-B authorization has no valid matching receipt")
    grouped = _manifest_by_seed(manifests)
    for seed in SEEDS:
        expected = (
            matching.get("per_seed", {})
            .get(str(seed), {})
            .get("trigger_evaluation_lineages", {})
            .get(arm)
        )
        if expected != _evaluation_lineage(grouped[seed]):
            raise RuntimeError("current trigger evaluation differs from the Stage-B seal")
    return {"status": "STAGE_B_TRIGGER_INPUTS_REOPENED", "arm": arm}


def _stage_c_matching_receipt(
    lora_control_manifests: Sequence[Mapping[str, Any]],
    fullrank_joint_manifests: Sequence[Mapping[str, Any]],
    fullrank_control_manifests: Sequence[Mapping[str, Any]],
    authorization: Mapping[str, Any],
) -> dict[str, Any]:
    current = _validate_lineage_entry(authorization)
    if authorization.get("phase") == "stage_b_seal_analysis":
        stage_b_lineage = dict(authorization)
    elif authorization.get("phase") == "fullrank_joint_analysis":
        upstream = current.get("authorization")
        if not isinstance(upstream, Mapping):
            raise RuntimeError("post-contrast Stage-C authorization omits Stage-B lineage")
        stage_b_lineage = dict(upstream)
    else:
        raise RuntimeError("Stage-C authorization has an unregistered phase")
    stage_b = _validate_lineage_entry(stage_b_lineage)
    if stage_b.get("status") not in {
        "FULLRANK_STATE_ONLY_REQUIRED", "STAGE_B_CONTRAST_AUTHORIZED"
    }:
        raise RuntimeError("Stage-C does not descend from a valid Stage-B seal")
    stage_b_matching = stage_b.get("matching")
    if (
        not isinstance(stage_b_matching, Mapping)
        or stage_b_matching.get("status") != "STAGE_B_MATCHING_VALID"
    ):
        raise RuntimeError("Stage-C lacks the exact Stage-B matching receipt")
    groups = {
        "lora_state_only": _manifest_by_seed(lora_control_manifests),
        "fullrank_joint": _manifest_by_seed(fullrank_joint_manifests),
        "fullrank_state_only": _manifest_by_seed(fullrank_control_manifests),
    }
    per_seed = {}
    for seed in SEEDS:
        checkpoints = {
            name: group[seed]["checkpoint"] for name, group in groups.items()
        }
        runs = {name: group[seed]["run"] for name, group in groups.items()}
        initializations = {
            name: checkpoint["shared_initialization"]
            for name, checkpoint in checkpoints.items()
        }
        if len({json.dumps(item, sort_keys=True) for item in initializations.values()}) != 1:
            raise RuntimeError("Stage-C shared initialization differs across reached arms")
        pairing_setups = {
            name: _pairing_setup_receipt(run["setup"])
            for name, run in runs.items()
        }
        if len({json.dumps(item, sort_keys=True) for item in pairing_setups.values()}) != 1:
            raise RuntimeError("Stage-C hardware/environment/target setup differs across arms")
        for field in (
            "data_manifest_sha256", "training_order_sha256", "dropout_schedule_sha256"
        ):
            if len({str(run[field]) for run in runs.values()}) != 1:
                raise RuntimeError(f"Stage-C matched {field} differs across reached arms")
        if len({json.dumps(run["dropout_probes"], sort_keys=True) for run in runs.values()}) != 1:
            raise RuntimeError("Stage-C realized dropout probes differ across reached arms")
        if (
            checkpoints["fullrank_joint"]["g0_lineage"]
            != checkpoints["fullrank_state_only"]["g0_lineage"]
            or checkpoints["fullrank_joint"]["positive_control_lineage"]
            != checkpoints["fullrank_state_only"]["positive_control_lineage"]
        ):
            raise RuntimeError("full-rank objectives do not reuse exact setup receipts")
        if checkpoints["fullrank_state_only"].get("branch_authorization_lineage") != dict(
            authorization
        ):
            raise RuntimeError("full-rank state-only changed its Stage-C authorization")
        original_lora_miss = stage_b.get("authorization")
        for name in ("lora_state_only", "fullrank_joint"):
            if checkpoints[name].get("branch_authorization_lineage") != original_lora_miss:
                raise RuntimeError("Stage-C predecessor changed its LoRA-miss authorization")
        stage_b_seed = stage_b_matching.get("per_seed", {}).get(str(seed), {})
        for name in ("lora_state_only", "fullrank_joint"):
            expected = stage_b_seed.get("trigger_evaluation_lineages", {}).get(name)
            if expected != _evaluation_lineage(groups[name][seed]):
                raise RuntimeError("Stage-C predecessor differs from the Stage-B seal")
        per_seed[str(seed)] = {
            "shared_initialization_receipt_identity_sha256": initializations[
                "fullrank_joint"
            ]["receipt_identity_sha256"],
            "pairing_setup_sha256": _canonical_sha256(pairing_setups["fullrank_joint"]),
            "training_order_sha256": runs["fullrank_joint"]["training_order_sha256"],
            "dropout_schedule_sha256": runs["fullrank_joint"]["dropout_schedule_sha256"],
            "fullrank_g0_lineage": checkpoints["fullrank_joint"]["g0_lineage"],
            "fullrank_positive_control_lineage": checkpoints["fullrank_joint"][
                "positive_control_lineage"
            ],
            "checkpoint_identities": {
                name: checkpoint["checkpoint_identity_sha256"]
                for name, checkpoint in checkpoints.items()
            },
        }
    return {
        "status": "STAGE_C_MATCHING_VALID",
        "stage_b_authorization": stage_b_lineage,
        "per_seed": per_seed,
    }


def _contrast_firewall_preopen(
    config: Mapping[str, Any], runs_dir: Path, authorization: Mapping[str, Any]
) -> dict[str, Any]:
    data_dir = (ROOT / str(config["paths"]["data_dir"])).resolve()
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("Stage-B data manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_data_manifest(config, data_dir, manifest, content_splits=set())
    ledger = load_contrast_access_ledger(config, data_dir, manifest)
    if ledger["events"]:
        raise RuntimeError("sealed contrast ledger is not empty before authorization")
    prohibited = [
        runs_dir / f"{capacity}_joint_seed{seed}_contrast"
        for capacity in ("lora", "fullrank") for seed in SEEDS
    ]
    existing = [path.relative_to(REPO_ROOT).as_posix() for path in prohibited if path.exists()]
    if existing:
        raise RuntimeError(f"contrast outputs exist before Stage-B seal: {existing}")
    return {
        "status": "CONTRAST_FIREWALL_UNOPENED",
        "data_manifest_sha256": _sha256(manifest_path),
        "ledger_path": (data_dir / "contrast_access_ledger.json").relative_to(REPO_ROOT).as_posix(),
        "ledger_sha256": _sha256(data_dir / "contrast_access_ledger.json"),
        "events": 0,
        "authorization": dict(authorization),
    }


def _contrast_ledger_guard(
    config: Mapping[str, Any], authorization: Mapping[str, Any]
) -> dict[str, Any]:
    stage_b = _validate_lineage_entry(authorization)
    matching = stage_b.get("matching")
    if not isinstance(matching, Mapping) or matching.get("status") != "STAGE_B_MATCHING_VALID":
        raise RuntimeError("Stage-B authorization has no matching receipt")
    data_dir = (ROOT / str(config["paths"]["data_dir"])).resolve()
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_data_manifest(config, data_dir, manifest, content_splits=set())
    ledger = load_contrast_access_ledger(config, data_dir, manifest)
    events = ledger["events"]
    expected_cells = {
        (capacity, "joint", seed)
        for capacity in ("lora", "fullrank") for seed in SEEDS
    }
    observed_cells = {
        (event.get("capacity"), event.get("objective"), event.get("model_seed"))
        for event in events
    }
    if len(events) != 6 or observed_cells != expected_cells:
        raise RuntimeError("contrast ledger does not contain the exact six registered cells")
    for event in events:
        if event.get("authorization") != dict(authorization):
            raise RuntimeError("contrast ledger mixes Stage-B authorizations")
        event_identity = event.get("event_identity_sha256")
        if event_identity != _canonical_sha256(
            {key: value for key, value in event.items() if key != "event_identity_sha256"}
        ):
            raise RuntimeError("contrast access event identity mismatch")
        output = _resolve_repo_path(str(event["evaluation_output"]))
        expected_output = (
            ROOT / "runs"
            / f"{event['capacity']}_joint_seed{event['model_seed']}_contrast"
        ).resolve()
        if output != expected_output:
            raise RuntimeError("contrast event uses a noncanonical evaluation output")
        summary_path = output / "summary.json"
        if not summary_path.is_file():
            raise RuntimeError("contrast access event has no complete evaluation summary")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if (
            summary.get("capacity") != event["capacity"]
            or summary.get("objective") != "joint"
            or summary.get("model_seed") != event["model_seed"]
            or summary.get("eval_set") != "contrast"
            or summary.get("contrast_authorization") != dict(authorization)
            or summary.get("contrast_access_event") != event
        ):
            raise RuntimeError("contrast event/evaluation cell lineage mismatch")
        checkpoint_lineage = event.get("checkpoint_lineage")
        if not isinstance(checkpoint_lineage, Mapping):
            raise RuntimeError("contrast event omits checkpoint lineage")
        if (
            summary.get("checkpoint_path") != checkpoint_lineage.get("path")
            or summary.get("checkpoint_metadata_sha256")
            != checkpoint_lineage.get("metadata_sha256")
            or summary.get("checkpoint_identity_sha256")
            != checkpoint_lineage.get("checkpoint_identity_sha256")
        ):
            raise RuntimeError("contrast event/evaluation checkpoint lineage mismatch")
        expected_checkpoint = (
            matching.get("per_seed", {})
            .get(str(event["model_seed"]), {})
            .get("checkpoint_lineages", {})
            .get(f"{event['capacity']}_joint")
        )
        if checkpoint_lineage != expected_checkpoint:
            raise RuntimeError("contrast event checkpoint differs from the Stage-B seal")
    return {
        "status": "CONTRAST_ACCESS_LEDGER_COMPLETE",
        "events": events,
        "ledger_sha256": _sha256(data_dir / "contrast_access_ledger.json"),
    }


def analyze_phase(
    config: Mapping[str, Any],
    runs_dir: Path,
    phase: str,
    output: Path,
    authorization_receipt: Path | None = None,
) -> dict[str, Any]:
    require_confirmatory_config(config)
    validate_design_receipt(config)
    if output.exists():
        raise RuntimeError(f"refusing to overwrite analysis: {output}")
    threshold = float(config["gates"]["min_final_joint_accuracy_each_seed_depth"])
    inputs: list[dict[str, Any]] = []
    authorization = None

    if phase == "lora_joint":
        if authorization_receipt is not None:
            raise RuntimeError("LoRA joint analysis accepts no authorization")
        bundles, manifests = _load_cell(config, runs_dir, "lora", "joint", "trigger")
        inputs.extend(manifests)
        formation = _formation_summary(bundles, threshold)
        adaptation = _adaptation_effects(config, bundles)
        if formation["status"] == "EVIDENCE_INCOMPLETE":
            status = "EVIDENCE_INVALID_REPAIR_REQUIRED"
            verdict = status
            next_stage = "repair_evidence_only"
        elif formation["passes"]:
            status = "LORA_DOES_NOT_PREVENT_STATE_FORMATION"
            verdict = status
            next_stage = "stop_capacity_branch"
        else:
            status = "LORA_JOINT_MISS_CONTROLS_REQUIRED"
            verdict = status
            next_stage = "run_lora_state_only_and_fullrank_joint"
        result = {"formation": formation, "adaptation_effect": adaptation}
        receipt_phase = "lora_joint_analysis"

    elif phase == "lora_control":
        auth_path = authorization_receipt or ROOT / "analysis" / "lora_joint_trigger.json"
        authorization = _load_analysis_authorization(
            config, auth_path, {"LORA_JOINT_MISS_CONTROLS_REQUIRED"},
            {"lora_joint_analysis"},
        )
        bundles, manifests = _load_cell(config, runs_dir, "lora", "state_only", "trigger")
        inputs.extend(manifests)
        formation = _formation_summary(bundles, threshold)
        adaptation = _adaptation_effects(config, bundles)
        if formation["status"] == "EVIDENCE_INCOMPLETE":
            status = "BRANCH_EVIDENCE_INCOMPLETE"
            verdict = status
            next_stage = "repair_evidence_only"
        else:
            status = "LORA_STATE_ONLY_CONTROL_COMPLETE"
            verdict = (
                "LORA_CAN_FORM_STATE_STATE_ONLY"
                if formation["passes"] else formation["status"]
            )
            next_stage = "continue_mandatory_stage_b_seal"
        result = {
            "formation": formation,
            "adaptation_effect": adaptation,
            "lora_state_only_annotation": (
                "LORA_CAN_FORM_STATE_STATE_ONLY" if formation["passes"] else None
            ),
        }
        receipt_phase = "lora_control_analysis"

    elif phase == "stage_b_seal":
        auth_path = authorization_receipt or ROOT / "analysis" / "lora_joint_trigger.json"
        authorization = _load_analysis_authorization(
            config, auth_path, {"LORA_JOINT_MISS_CONTROLS_REQUIRED"},
            {"lora_joint_analysis"},
        )
        lora_miss_receipt = _validate_lineage_entry(authorization)
        lora_control_authorization = _load_analysis_authorization(
            config, ROOT / "analysis" / "lora_control.json",
            {"LORA_STATE_ONLY_CONTROL_COMPLETE"}, {"lora_control_analysis"},
        )
        lora_control_receipt = _validate_lineage_entry(lora_control_authorization)
        if lora_control_receipt.get("authorization") != dict(authorization):
            raise RuntimeError("LoRA control analysis uses a different LoRA-miss receipt")
        lora_joint, lora_joint_manifests = _load_cell(
            config, runs_dir, "lora", "joint", "trigger"
        )
        lora_control, lora_control_manifests = _load_cell(
            config, runs_dir, "lora", "state_only", "trigger"
        )
        fullrank_joint, fullrank_joint_manifests = _load_cell(
            config, runs_dir, "fullrank", "joint", "trigger"
        )
        inputs.extend(
            lora_joint_manifests + lora_control_manifests + fullrank_joint_manifests
        )
        lora_joint_formation = _formation_summary(lora_joint, threshold)
        fullrank_formation = _formation_summary(fullrank_joint, threshold)
        lora_control_formation = _formation_summary(lora_control, threshold)
        if lora_joint_formation["passes"]:
            raise RuntimeError("current LoRA joint evidence contradicts the miss authorization")
        if lora_miss_receipt.get("formation") != lora_joint_formation:
            raise RuntimeError("LoRA miss authorization is detached from current LoRA evidence")
        if lora_control_receipt.get("formation") != lora_control_formation:
            raise RuntimeError("LoRA control analysis is detached from current control evidence")
        evidence_incomplete = any(
            item["status"] == "EVIDENCE_INCOMPLETE"
            for item in (
                lora_joint_formation, fullrank_formation, lora_control_formation
            )
        )
        if evidence_incomplete:
            status = "BRANCH_EVIDENCE_INCOMPLETE"
            verdict = status
            next_stage = "repair_evidence_only"
            matching = {"status": "NOT_EVALUATED_INCOMPLETE_EVIDENCE"}
            firewall = {"status": "NOT_EVALUATED_INCOMPLETE_EVIDENCE"}
        else:
            try:
                matching = _stage_b_matching_receipt(
                    lora_joint_manifests, lora_control_manifests,
                    fullrank_joint_manifests, authorization,
                )
                firewall = _contrast_firewall_preopen(config, runs_dir, authorization)
            except RuntimeError as exc:
                status = "CONTRAST_FIREWALL_NOT_READY"
                verdict = status
                next_stage = "repair_stage_b_evidence_only"
                matching = {"status": "INVALID", "error": str(exc)}
                firewall = {"status": "INVALID", "error": str(exc)}
            else:
                if fullrank_formation["passes"]:
                    status = "STAGE_B_CONTRAST_AUTHORIZED"
                    verdict = status
                    next_stage = "evaluate_exact_six_joint_contrast_cells"
                else:
                    status = "FULLRANK_STATE_ONLY_REQUIRED"
                    verdict = status
                    next_stage = "run_fullrank_state_only_control"
        result = {
            "lora_joint_formation": lora_joint_formation,
            "fullrank_trigger_formation": fullrank_formation,
            "lora_state_only_formation": lora_control_formation,
            "lora_state_only_annotation": (
                "LORA_CAN_FORM_STATE_STATE_ONLY"
                if lora_control_formation["passes"] else None
            ),
            "lora_control_analysis": lora_control_authorization,
            "matching": matching,
            "contrast_firewall": firewall,
        }
        receipt_phase = "stage_b_seal_analysis"

    elif phase == "fullrank_joint":
        auth_path = authorization_receipt or ROOT / "analysis" / "stage_b_seal.json"
        authorization = _load_analysis_authorization(
            config, auth_path, {"STAGE_B_CONTRAST_AUTHORIZED"},
            {"stage_b_seal_analysis"},
        )
        stage_b_receipt = _validate_lineage_entry(authorization)
        cached_lora_state_only_formation = stage_b_receipt.get(
            "lora_state_only_formation"
        )
        lora_state_only_annotation = stage_b_receipt.get("lora_state_only_annotation")
        cached_lora_trigger_formation = stage_b_receipt.get("lora_joint_formation")
        if not isinstance(cached_lora_state_only_formation, Mapping):
            raise RuntimeError("Stage-B receipt omits LoRA state-only formation")
        if not isinstance(cached_lora_trigger_formation, Mapping):
            raise RuntimeError("Stage-B receipt omits LoRA trigger formation")
        expected_lora_annotation = (
            "LORA_CAN_FORM_STATE_STATE_ONLY"
            if cached_lora_state_only_formation.get("passes") is True else None
        )
        if lora_state_only_annotation != expected_lora_annotation:
            raise RuntimeError("Stage-B LoRA state-only annotation is inconsistent")
        ledger_guard = _contrast_ledger_guard(config, authorization)
        fullrank_trigger, trigger_manifests = _load_cell(
            config, runs_dir, "fullrank", "joint", "trigger"
        )
        trigger_reopen = _require_stage_b_trigger_match(
            trigger_manifests, arm="fullrank_joint", authorization=authorization
        )
        lora_trigger, lora_trigger_manifests = _load_cell(
            config, runs_dir, "lora", "joint", "trigger"
        )
        lora_trigger_reopen = _require_stage_b_trigger_match(
            lora_trigger_manifests, arm="lora_joint", authorization=authorization
        )
        lora_state_only, lora_state_only_manifests = _load_cell(
            config, runs_dir, "lora", "state_only", "trigger"
        )
        lora_state_only_reopen = _require_stage_b_trigger_match(
            lora_state_only_manifests,
            arm="lora_state_only",
            authorization=authorization,
        )
        fullrank_contrast, fullrank_manifests = _load_cell(
            config, runs_dir, "fullrank", "joint", "contrast"
        )
        lora_contrast, lora_manifests = _load_cell(
            config, runs_dir, "lora", "joint", "contrast"
        )
        inputs.extend(
            trigger_manifests + lora_trigger_manifests
            + lora_state_only_manifests + fullrank_manifests + lora_manifests
        )
        trigger_formation = _formation_summary(fullrank_trigger, threshold)
        lora_trigger_formation = _formation_summary(lora_trigger, threshold)
        lora_state_only_formation = _formation_summary(lora_state_only, threshold)
        if lora_trigger_formation != cached_lora_trigger_formation:
            raise RuntimeError("reopened LoRA trigger formation differs from Stage-B")
        if lora_state_only_formation != cached_lora_state_only_formation:
            raise RuntimeError("reopened LoRA state-only formation differs from Stage-B")
        sealed_formation = _formation_summary(fullrank_contrast, threshold)
        lora_sealed_formation = _formation_summary(lora_contrast, threshold)
        sealed_adaptation = _adaptation_effects(config, fullrank_contrast)
        lora_sealed_adaptation = _adaptation_effects(config, lora_contrast)
        evidence_incomplete = any(
            item["status"] == "EVIDENCE_INCOMPLETE"
            for item in (
                trigger_formation, sealed_formation, lora_sealed_formation
            )
        )
        lora_failure_replication = (
            {
                "status": "EVIDENCE_INCOMPLETE",
                "trigger_failed_categories": [],
                "sealed_failed_categories": [],
                "missing_replications": [],
                "passes": False,
            }
            if evidence_incomplete
            else _failure_category_replication(
                lora_trigger_formation, lora_sealed_formation
            )
        )
        contrast = _fullrank_minus_lora_contrast(
            config, fullrank_contrast, lora_contrast
        )
        if evidence_incomplete:
            status = "BRANCH_EVIDENCE_INCOMPLETE"
            verdict = status
            next_stage = "repair_evidence_only"
        elif lora_sealed_formation["passes"]:
            status = "LORA_TRIGGER_MISS_NOT_REPLICATED_ON_SEALED_CONTRAST"
            verdict = status
            next_stage = "stop_capacity_branch"
        elif not trigger_formation["passes"] or not sealed_formation["passes"]:
            status = "FULLRANK_STATE_ONLY_REQUIRED"
            verdict = status
            next_stage = "run_fullrank_state_only_control"
        elif not lora_failure_replication["passes"]:
            status = (
                "LORA_TRIGGER_FAILURE_CATEGORIES_NOT_REPLICATED_"
                "ON_SEALED_CONTRAST"
            )
            verdict = status
            next_stage = "stop_capacity_branch"
        elif sealed_adaptation["passes"] and contrast["passes"]:
            status = "DIRECT_FULLSHAPE_RECIPE_RESCUE"
            verdict = status
            next_stage = "stop_capacity_branch"
        else:
            status = "DIRECT_FULLSHAPE_RECIPE_PASS_CONTRAST_UNCERTAIN"
            verdict = status
            next_stage = "stop_capacity_branch"
        result = {
            "trigger_formation": trigger_formation,
            "sealed_contrast_formation": sealed_formation,
            "lora_sealed_contrast_formation": lora_sealed_formation,
            "lora_trigger_failure_replication": lora_failure_replication,
            "sealed_contrast_adaptation": sealed_adaptation,
            "lora_sealed_contrast_adaptation": lora_sealed_adaptation,
            "fullrank_minus_lora_sealed_contrast": contrast,
            "contrast_access_ledger": ledger_guard,
            "stage_b_trigger_reopen": trigger_reopen,
            "lora_trigger_reopen": lora_trigger_reopen,
            "lora_state_only_reopen": lora_state_only_reopen,
            "lora_state_only_formation": lora_state_only_formation,
            "lora_state_only_annotation": lora_state_only_annotation,
            "objective_interaction_interpretation": (
                "consistent_with_objective_interaction"
                if lora_state_only_annotation else None
            ),
        }
        receipt_phase = "fullrank_joint_analysis"

    elif phase == "fullrank_control":
        auth_path = authorization_receipt or ROOT / "analysis" / "fullrank_joint.json"
        authorization = _load_analysis_authorization(
            config, auth_path, {"FULLRANK_STATE_ONLY_REQUIRED"},
            {"stage_b_seal_analysis", "fullrank_joint_analysis"},
        )
        fullrank, manifests = _load_cell(
            config, runs_dir, "fullrank", "state_only", "trigger"
        )
        lora, lora_manifests = _load_cell(
            config, runs_dir, "lora", "state_only", "trigger"
        )
        _, fullrank_joint_manifests = _load_cell(
            config, runs_dir, "fullrank", "joint", "trigger"
        )
        inputs.extend(manifests + lora_manifests + fullrank_joint_manifests)
        matching = _stage_c_matching_receipt(
            lora_manifests, fullrank_joint_manifests, manifests, authorization
        )
        fullrank_formation = _formation_summary(fullrank, threshold)
        lora_formation = _formation_summary(lora, threshold)
        fullrank_adaptation = _adaptation_effects(config, fullrank)
        lora_adaptation = _adaptation_effects(config, lora)
        if any(
            item["status"] == "EVIDENCE_INCOMPLETE"
            for item in (fullrank_formation, lora_formation)
        ):
            status = "BRANCH_EVIDENCE_INCOMPLETE"
            next_stage = "repair_evidence_only"
        elif fullrank_formation["passes"] and not lora_formation["passes"]:
            status = "DIRECT_FULLSHAPE_RECIPE_STATE_ONLY_RESCUE"
        elif fullrank_formation["passes"] and lora_formation["passes"]:
            status = "BOTH_CAPACITIES_FORM_STATE_WITHOUT_ANSWER"
        elif not fullrank_formation["passes"] and lora_formation["passes"]:
            status = "FULLRANK_CONTROL_REVERSAL"
        else:
            status = (
                "FULLRANK_RELIEF_NOT_SUFFICIENT_REGISTERED_RECIPE_"
                "BOTTLENECK_UNRESOLVED"
            )
        verdict = status
        if status != "BRANCH_EVIDENCE_INCOMPLETE":
            next_stage = "stop_capacity_branch"
        result = {
            "fullrank_formation": fullrank_formation,
            "lora_formation": lora_formation,
            "fullrank_adaptation_effect": fullrank_adaptation,
            "lora_adaptation_effect": lora_adaptation,
            "matching": matching,
            "lora_state_only_annotation": (
                "LORA_CAN_FORM_STATE_STATE_ONLY" if lora_formation["passes"] else None
            ),
            "unresolved_bottlenecks": (
                ["supervision_or_readout_architecture", "registered_optimization_recipe"]
                if not fullrank_formation["passes"] and not lora_formation["passes"] else []
            ),
        }
        receipt_phase = "fullrank_control_analysis"

    else:
        raise ValueError(f"unknown analysis phase: {phase}")

    summary = _with_identity({
        "schema_version": 1,
        "status": status,
        "verdict": verdict,
        **_identity(config, receipt_phase),
        "analysis_phase": phase,
        "next_stage": next_stage,
        "authorization": authorization,
        "input_manifest": inputs,
        **result,
        "warning": (
            "State readability only; no answer-use, capability, deployment, "
            "serial-advantage, rank-identification, or sample-more claim is licensed."
        ),
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary
