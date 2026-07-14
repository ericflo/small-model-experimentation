#!/usr/bin/env python3
"""Independently audit local provenance before authorizing benchmark events."""

from __future__ import annotations

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
CONFIG = EXP / "configs" / "default.yaml"
PREREGISTRATION = EXP / "runs" / "preregistration_receipt.json"
CONFIRMATION = EXP / "analysis" / "confirmation.json"
CONFIRMATION_ROOT = (EXP / "runs" / "confirmation").resolve()
BENCHMARK_ROOT = EXP / "runs" / "benchmark"
CONTROLS = EXP / "runs" / "controls.json"
GATEWAY = REPO / "scripts" / "run_benchmark_aggregate.py"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
BENCH = EXP / "scripts" / "bench.py"
ANALYZER = EXP / "scripts" / "analyze_benchmark.py"
CONFIRMATION_ANALYZER = EXP / "scripts" / "analyze_confirmation.py"
CONFIRMATION_EVALUATOR = EXP / "scripts" / "eval_policy.py"
CONTROL_REMATCH = EXP / "src" / "control_rematch.py"
CONTROL_RECEIPTS = EXP / "src" / "control_receipts.py"
PY = REPO / ".venv" / "bin" / "python"
FROZEN_FILES = (
    "configs/default.yaml",
    "idea_intake.md",
    "reports/preregistration.md",
    "reports/design_review.md",
    "reports/literature_review.md",
)
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from bench import (  # noqa: E402
    BENCHMARK_EVENT_COUNT,
    _expected_models,
    _publication_start_state,
    _tier_seeds,
    benchmark_source_inventory,
    model_provenance,
)
from confirmation_artifacts import (  # noqa: E402
    confirmation_admission_binding,
    confirmation_raw_dir,
    configured_confirmation_raw_root,
    controls_authorization_binding,
    validate_confirmation_campaign_tree,
    validate_confirmation_geometry,
    validate_confirmation_score_artifacts,
)
from confirmation_protocol import (  # noqa: E402
    _strict_json_equal,
    canonical_backend_protocol,
    expected_confirmation_sampling_protocol,
)
from control_rematch import (  # noqa: E402
    validate_control_overlay_cache,
    validate_control_rematch_manifest,
)
from control_receipts import (  # noqa: E402
    validate_control_training_receipt,
    validate_parameter_control_model,
)
from io_utils import (  # noqa: E402
    confirmation_evaluator_source_inventory,
    load_config,
    resolve_repo_path,
    sha256_file,
)
from model_provenance import (  # noqa: E402
    validate_model_checkpoint,
    validate_source_checkpoint_receipts,
)


def _load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid provenance receipt: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"provenance receipt is not an object: {path}")
    return payload


def _validated_confirmation_analysis_file(path: Path) -> Path:
    """Require the canonical, regular, symlink-free confirmation analysis."""

    canonical, existed = _publication_start_state(
        path,
        expected=EXP / "analysis" / "confirmation.json",
        label="confirmation analysis receipt",
    )
    if not existed:
        raise ValueError("canonical confirmation analysis receipt is missing")
    return canonical


def _confirmation_analysis_binding(path: Path, expected_payload: dict) -> dict[str, str]:
    """Bind canonical analysis bytes, rechecking path and content around hashing."""

    canonical = _validated_confirmation_analysis_file(path)
    expected_bytes = (
        json.dumps(expected_payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    observed_bytes = canonical.read_bytes()
    if (
        observed_bytes != expected_bytes
        or not _strict_json_equal(_load(canonical), expected_payload)
    ):
        raise ValueError("confirmation analysis changed during authorization")
    digest = sha256_file(canonical)
    if (
        _validated_confirmation_analysis_file(canonical) != canonical
        or canonical.read_bytes() != observed_bytes
        or not _strict_json_equal(_load(canonical), expected_payload)
        or sha256_file(canonical) != digest
    ):
        raise ValueError("confirmation analysis changed during authorization")
    return {"path": str(canonical), "sha256": digest}


def _validated_authorization_publication_path(path: Path) -> Path:
    """Return a lexical absolute output confined to the real analysis root."""

    output = Path(os.path.abspath(os.fspath(path)))
    analysis_root = Path(os.path.abspath(os.fspath(EXP / "analysis")))
    for label, candidate_path in (
        ("canonical analysis root", analysis_root),
        ("benchmark authorization output", output),
    ):
        for candidate in reversed((candidate_path, *candidate_path.parents)):
            try:
                metadata = candidate.lstat()
            except FileNotFoundError:
                break
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(
                    f"unsafe {label}: symlinked existing ancestor or output: "
                    f"{candidate}"
                )

    resolved_root = analysis_root.resolve(strict=False)
    resolved_parent = output.parent.resolve(strict=False)
    if not resolved_parent.is_relative_to(resolved_root):
        raise ValueError(
            "benchmark authorization output parent is outside the experiment "
            f"analysis root: {resolved_parent}"
        )
    return output


def _publish_authorization_no_clobber(path: Path, payload: dict) -> None:
    """Atomically seal one authorization; exact reruns only verify bytes."""

    try:
        encoded = json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError) as exc:
        raise ValueError("benchmark authorization is not canonical JSON") from exc
    path = _validated_authorization_publication_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path = _validated_authorization_publication_path(path)
    if path.is_file() and not path.is_symlink():
        existing = _load(path)
        try:
            existing_encoded = json.dumps(
                existing,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            ) + "\n"
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"existing benchmark authorization is not canonical JSON: {path}"
            ) from exc
        if existing_encoded != encoded:
            raise ValueError(
                f"refusing to overwrite stale benchmark authorization: {path}"
            )
        return
    if path.exists() or path.is_symlink():
        raise ValueError(f"unsafe benchmark authorization output: {path}")

    temporary_name = None
    try:
        path = _validated_authorization_publication_path(path)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            path = _validated_authorization_publication_path(path)
            os.link(temporary_name, path)
        except FileExistsError as exc:
            raise ValueError(
                f"benchmark authorization publication lost a race: {path}"
            ) from exc
        path = _validated_authorization_publication_path(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def _code_provenance() -> dict:
    source_inventory = benchmark_source_inventory(MENAGERIE.parent)
    evaluator_source = confirmation_evaluator_source_inventory()
    return {
        "aggregate_gateway_sha256": sha256_file(GATEWAY),
        "benchmark_runner_sha256": sha256_file(MENAGERIE),
        "benchmark_source_inventory_sha256": source_inventory["sha256"],
        "benchmark_source_file_count": source_inventory["file_count"],
        "bench_sha256": sha256_file(BENCH),
        "analyzer_sha256": sha256_file(ANALYZER),
        "confirmation_analyzer_sha256": sha256_file(CONFIRMATION_ANALYZER),
        "confirmation_evaluator_sha256": sha256_file(CONFIRMATION_EVALUATOR),
        "confirmation_evaluator_source_inventory_sha256": evaluator_source["sha256"],
        "confirmation_evaluator_source_file_count": evaluator_source["file_count"],
        "control_rematch_sha256": sha256_file(CONTROL_REMATCH),
        "control_receipts_sha256": sha256_file(CONTROL_RECEIPTS),
        "authorizer_sha256": sha256_file(Path(__file__)),
    }


def _audit_preregistration() -> dict:
    payload = _load(PREREGISTRATION)
    if (
        int(payload.get("schema_version", -1)) != 1
        or payload.get("status") != "locked"
        or payload.get("experiment_id") != EXP.name
        or tuple(payload.get("frozen_file_order") or ()) != FROZEN_FILES
        or payload.get("model_output_precedes_lock") is not False
    ):
        raise ValueError("preregistration is not locked")
    frozen = payload.get("frozen_files")
    if not isinstance(frozen, dict) or set(frozen) != set(FROZEN_FILES):
        raise ValueError("preregistration lacks frozen-file hashes")
    for relative in FROZEN_FILES:
        expected = frozen[relative]
        path = EXP / str(relative)
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError("a frozen preregistration file changed")
    design_commit = payload.get("design_commit")
    if not isinstance(design_commit, str) or not design_commit:
        raise ValueError("preregistration design commit is missing")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", design_commit, "HEAD"],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("preregistered design commit is not an ancestor")
    return payload


def _integration_model(config: dict, seed: int) -> Path:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    final_round = int(config["mopd"]["rounds"]) - 1
    return (
        root / "merged" / "primary" / f"seed_{seed}" / f"round_{final_round}"
    ).resolve()


def _control_model(config: dict, name: str) -> Path:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    final_round = int(config["mopd"]["rounds"]) - 1
    return (root / "merged" / "controls" / name / f"round_{final_round}").resolve()


def _same_path(value: object, expected: Path) -> bool:
    try:
        return isinstance(value, str) and Path(value).resolve() == expected.resolve()
    except (OSError, RuntimeError, ValueError):
        return False


def _audit_merge_receipt(merged: Path, base_model: Path, adapter: Path) -> str:
    """Authenticate the semantic inputs recorded for one LoRA merge."""

    provenance = validate_model_checkpoint(merged, profile="local")
    receipt_path = merged / "merge_receipt.json"
    receipt = _load(receipt_path)
    adapter_config = adapter / "adapter_config.json"
    adapter_weights = adapter / "adapter_model.safetensors"
    weight_rows = receipt.get("weight_files")
    if (
        receipt.get("method") != "explicit_composite_lora_merge"
        or not _same_path(receipt.get("base_model"), base_model)
        or not _same_path(receipt.get("adapter"), adapter)
        or not adapter_config.is_file()
        or not adapter_weights.is_file()
        or receipt.get("adapter_config_sha256") != sha256_file(adapter_config)
        or receipt.get("adapter_weights_sha256") != sha256_file(adapter_weights)
        or int(receipt.get("applied_lora_modules", 0)) < 1
        or int(receipt.get("nonzero_lora_modules", -1))
        != int(receipt.get("applied_lora_modules", -2))
        or not isinstance(weight_rows, list)
        or not weight_rows
    ):
        raise ValueError("round merge receipt is semantically stale")
    recorded_names = []
    for row in weight_rows:
        if not isinstance(row, dict) or set(row) != {"name", "sha256"}:
            raise ValueError("round merge weight inventory is malformed")
        name = row.get("name")
        expected = row.get("sha256")
        if (
            not isinstance(name, str)
            or not name
            or Path(name).name != name
            or not isinstance(expected, str)
            or len(expected) != 64
        ):
            raise ValueError("round merge weight inventory entry is invalid")
        weight = merged / name
        if not weight.is_file() or sha256_file(weight) != expected:
            raise ValueError("round merge weight hash is stale")
        recorded_names.append(name)
    actual_names = sorted(path.name for path in merged.glob("*.safetensors"))
    if (
        len(recorded_names) != len(set(recorded_names))
        or sorted(recorded_names) != actual_names
    ):
        raise ValueError("round merge weight inventory is incomplete")
    if provenance["model_merge_receipt_sha256"] != sha256_file(receipt_path):
        raise ValueError("round merge model inventory changed during audit")
    return provenance["model_merge_receipt_sha256"]


def _audit_mopd_training_receipt(
    path: Path,
    *,
    config: dict,
    config_path: Path,
    base_model: Path,
    round_manifest: Path,
    target_cache: Path,
    round_index: int,
    seed: int,
    arm: str,
    recorded_gate: object,
    expected_target_initial_loss: float | None = None,
) -> dict:
    """Validate the actual MOPD method and all round-defining inputs."""

    receipt = _load(path)
    updates = int(config["mopd"]["updates_per_round"])
    base_merge = base_model / "merge_receipt.json"
    if (
        int(receipt.get("schema_version", -1)) != 2
        or receipt.get("method")
        != "deep_advantage_routed_corrected_teacher_topk_reverse_kl"
        or receipt.get("arm") != arm
        or not _same_path(receipt.get("config"), config_path)
        or receipt.get("config_sha256") != sha256_file(config_path)
        or not _same_path(receipt.get("base_model"), base_model)
        or not base_merge.is_file()
        or receipt.get("base_merge_receipt_sha256") != sha256_file(base_merge)
        or not _same_path(receipt.get("target_cache"), target_cache)
        or not target_cache.is_file()
        or receipt.get("target_cache_sha256") != sha256_file(target_cache)
        or int(receipt.get("round", -1)) != round_index
        or int(receipt.get("seed", -1)) != seed
        or int(receipt.get("requested_updates", -1)) != updates
        or int(receipt.get("completed_updates", -1)) != updates
        or receipt.get("consume_once_verified") is not True
        or receipt.get("round_gate") != recorded_gate
        or not isinstance(receipt.get("round_gate"), dict)
        or not receipt.get("round_gate", {}).get("passed")
        or not receipt.get("round_gate", {}).get("completed_all_updates")
    ):
        raise ValueError("MOPD training receipt is semantically stale")
    if arm != "primary":
        if expected_target_initial_loss is None:
            raise ValueError("matched MOPD control lacks primary pressure")
        validate_control_training_receipt(
            receipt,
            config=config,
            arm=arm,
            expected_target_initial_loss=expected_target_initial_loss,
            source_manifest=round_manifest,
            target_cache=target_cache,
            round_index=round_index,
            seed=seed,
        )
    return receipt


def _audit_offpolicy_training_receipt(
    path: Path,
    *,
    config: dict,
    config_path: Path,
    base_model: Path,
    round_manifest: Path,
    round_index: int,
    seed: int,
    recorded_gate: object,
    expected_target_initial_loss: float,
) -> dict:
    """Validate the matched off-policy method and all round-defining inputs."""

    receipt = _load(path)
    updates = int(config["mopd"]["updates_per_round"])
    base_merge = base_model / "merge_receipt.json"
    if (
        int(receipt.get("schema_version", -1)) != 1
        or receipt.get("method") != "offpolicy_best_selection_continuation_sft"
        or not _same_path(receipt.get("config"), config_path)
        or receipt.get("config_sha256") != sha256_file(config_path)
        or not _same_path(receipt.get("base_model"), base_model)
        or not base_merge.is_file()
        or receipt.get("base_merge_receipt_sha256") != sha256_file(base_merge)
        or not _same_path(receipt.get("round_manifest"), round_manifest)
        or not round_manifest.is_file()
        or receipt.get("round_manifest_sha256") != sha256_file(round_manifest)
        or int(receipt.get("round", -1)) != round_index
        or int(receipt.get("seed", -1)) != seed
        or int(receipt.get("requested_updates", -1)) != updates
        or int(receipt.get("completed_updates", -1)) != updates
        or receipt.get("consume_once_verified") is not True
        or receipt.get("round_gate") != recorded_gate
        or not isinstance(receipt.get("round_gate"), dict)
        or not receipt.get("round_gate", {}).get("passed")
        or not receipt.get("round_gate", {}).get("completed_all_updates")
    ):
        raise ValueError("off-policy training receipt is semantically stale")
    validate_control_training_receipt(
        receipt,
        config=config,
        arm="offpolicy_sft",
        expected_target_initial_loss=expected_target_initial_loss,
        source_manifest=round_manifest,
        base_model=base_model,
        round_index=round_index,
        seed=seed,
    )
    return receipt


def _audit_parameter_control(
    config: dict,
    *,
    model: Path,
    row: object,
    expected_weight: float,
) -> None:
    """Independently authenticate one parameter-soup model and receipt binding."""

    validate_parameter_control_model(
        model,
        quick_adapter=resolve_repo_path(config["model"]["quick_adapter"]),
        deep_adapter=resolve_repo_path(config["model"]["deep_adapter"]),
        expected_deep_weight=expected_weight,
        expected_model_id=str(config["model"]["id"]),
        expected_revision=str(config["model"]["revision"]),
    )
    receipt_path = model / "merge_receipt.json"
    if (
        not isinstance(row, dict)
        or row.get("model") != str(model)
        or row.get("merge_receipt_sha256") != sha256_file(receipt_path)
    ):
        raise ValueError("parameter-control provenance is stale")


def _parameter_models(config: dict) -> dict[str, Path]:
    root = resolve_repo_path(config["model"]["artifacts_root"])
    return {
        f"soup{int(round(float(weight) * 100)):02d}": (
            root / "merged" / f"soup{int(round(float(weight) * 100)):02d}"
        ).resolve()
        for weight in config["controls"]["parameter_merge_deep_weights"]
    }


def _audit_integration(
    config: dict, config_path: Path, seed: int, primary: Path
) -> dict:
    receipt_path = EXP / "runs" / "integration" / f"seed_{seed}.json"
    receipt = _load(receipt_path)
    rounds = receipt.get("rounds")
    expected_rounds = int(config["mopd"]["rounds"])
    if (
        receipt.get("stage") != "four_round_deep_advantage_routed_mopd"
        or receipt.get("config_sha256") != sha256_file(config_path)
        or int(receipt.get("seed", -1)) != seed
        or not receipt.get("gate", {}).get("passed")
        or receipt.get("final_model") != str(primary.resolve())
        or not isinstance(rounds, list)
        or len(rounds) != expected_rounds
        or int(receipt.get("completed_rounds", -1)) != expected_rounds
    ):
        raise ValueError("primary integration receipt is stale")
    final = rounds[-1]
    if (
        int(final.get("round", -1)) != expected_rounds - 1
        or final.get("merged") != str(primary.resolve())
        or final.get("merge_receipt_sha256")
        != sha256_file(primary / "merge_receipt.json")
        or not all(row.get("round_gate", {}).get("passed") for row in rounds)
    ):
        raise ValueError("primary integration final-round provenance is stale")
    artifacts_root = resolve_repo_path(config["model"]["artifacts_root"])
    soup = resolve_repo_path(config["model"]["student_checkpoint"])
    primary_seed = int(config["seeds"]["integration_training"][0])
    for round_index, row in enumerate(rounds):
        adapter = (
            artifacts_root
            / "adapters"
            / "primary"
            / f"seed_{seed}"
            / f"round_{round_index}"
        ).resolve()
        training_receipt = adapter / "training_receipt.json"
        merged = (
            artifacts_root
            / "merged"
            / "primary"
            / f"seed_{seed}"
            / f"round_{round_index}"
        ).resolve()
        # All integration seeds start from the identical frozen soup, so the
        # runner reuses the primary seed's exact round-zero on-policy draw.
        # Once checkpoints diverge, every later round is seed-local.
        data_seed = primary_seed if round_index == 0 else seed
        data_root = (
            artifacts_root
            / "online"
            / "primary"
            / f"seed_{data_seed}"
            / f"round_{round_index}"
        ).resolve()
        round_manifest = data_root / "training_round.json"
        target_cache = data_root / "all_policy_targets.pt"
        base_model = (
            soup
            if round_index == 0
            else (
                artifacts_root
                / "merged"
                / "primary"
                / f"seed_{seed}"
                / f"round_{round_index - 1}"
            ).resolve()
        )
        if (
            int(row.get("round", -1)) != round_index
            or row.get("round_manifest") != str(round_manifest)
            or not round_manifest.is_file()
            or row.get("round_manifest_sha256") != sha256_file(round_manifest)
            or row.get("target_cache") != str(target_cache)
            or not target_cache.is_file()
            or row.get("target_cache_sha256") != sha256_file(target_cache)
            or row.get("training_receipt") != str(training_receipt)
            or not training_receipt.is_file()
            or row.get("training_receipt_sha256") != sha256_file(training_receipt)
            or row.get("merged") != str(merged)
            or not (merged / "merge_receipt.json").is_file()
            or row.get("merge_receipt_sha256")
            != sha256_file(merged / "merge_receipt.json")
            or not row.get("round_gate", {}).get("passed")
        ):
            raise ValueError("primary integration round provenance is stale")
        _audit_mopd_training_receipt(
            training_receipt,
            config=config,
            config_path=config_path,
            base_model=base_model,
            round_manifest=round_manifest,
            target_cache=target_cache,
            round_index=round_index,
            seed=seed,
            arm="primary",
            recorded_gate=row.get("round_gate"),
        )
        if row.get("merge_receipt_sha256") != _audit_merge_receipt(
            merged, base_model, adapter
        ):
            raise ValueError("primary integration merge provenance is stale")
    return {
        "path": str(receipt_path.resolve()),
        "sha256": sha256_file(receipt_path),
    }


def _audit_controls(config: dict, config_path: Path) -> tuple[dict, dict[str, Path]]:
    payload = _load(CONTROLS)
    control_names = ("non_advantage_route", "wrong_teacher", "offpolicy_sft")
    expected_parameters = _parameter_models(config)
    controls = payload.get("controls")
    parameters = payload.get("parameter_controls")
    if (
        payload.get("stage") != "matched_controls"
        or payload.get("config_sha256") != sha256_file(config_path)
        or int(payload.get("primary_seed", -1))
        != int(config["seeds"]["integration_training"][0])
        or not payload.get("gate", {}).get("passed")
        or not isinstance(controls, dict)
        or set(controls) != set(control_names)
        or not isinstance(parameters, dict)
        or set(parameters) != set(expected_parameters)
    ):
        raise ValueError("matched-control receipt is stale")

    artifacts_root = resolve_repo_path(config["model"]["artifacts_root"])
    soup = resolve_repo_path(config["model"]["student_checkpoint"])
    primary_seed = int(config["seeds"]["integration_training"][0])
    expected_rounds = int(config["mopd"]["rounds"])
    models: dict[str, Path] = {}
    for name in control_names:
        row = controls[name]
        model = _control_model(config, name)
        seed = int(config["seeds"][name])
        rounds = row.get("rounds")
        if (
            int(row.get("seed", -1)) != seed
            or not isinstance(rounds, list)
            or len(rounds) != expected_rounds
            or row.get("final_model") != str(model)
            or row.get("final_merge_receipt_sha256")
            != sha256_file(model / "merge_receipt.json")
            or not row.get("gate", {}).get("passed")
        ):
            raise ValueError("matched-control final provenance is stale")
        for round_index, round_row in enumerate(rounds):
            adapter = (
                artifacts_root
                / "adapters"
                / "controls"
                / name
                / f"round_{round_index}"
            ).resolve()
            training_receipt = adapter / "training_receipt.json"
            merged = (
                artifacts_root
                / "merged"
                / "controls"
                / name
                / f"round_{round_index}"
            ).resolve()
            base_model = (
                soup
                if round_index == 0
                else (
                    artifacts_root
                    / "merged"
                    / "controls"
                    / name
                    / f"round_{round_index - 1}"
                ).resolve()
            )
            data_root = (
                artifacts_root
                / "online"
                / "primary"
                / f"seed_{primary_seed}"
                / f"round_{round_index}"
            ).resolve()
            round_manifest = data_root / "training_round.json"
            source_cache = data_root / "all_policy_targets.pt"
            primary_training_receipt = (
                artifacts_root
                / "adapters"
                / "primary"
                / f"seed_{primary_seed}"
                / f"round_{round_index}"
                / "training_receipt.json"
            ).resolve()
            if (
                int(round_row.get("round", -1)) != round_index
                or not round_manifest.is_file()
                or round_row.get("primary_manifest_sha256")
                != sha256_file(round_manifest)
                or round_row.get("training_receipt") != str(training_receipt)
                or not training_receipt.is_file()
                or round_row.get("training_receipt_sha256")
                != sha256_file(training_receipt)
                or round_row.get("primary_training_receipt")
                != str(primary_training_receipt)
                or not primary_training_receipt.is_file()
                or round_row.get("primary_training_receipt_sha256")
                != sha256_file(primary_training_receipt)
                or round_row.get("merged") != str(merged)
                or not (merged / "merge_receipt.json").is_file()
                or round_row.get("merge_receipt_sha256")
                != sha256_file(merged / "merge_receipt.json")
                or not round_row.get("round_gate", {}).get("passed")
            ):
                raise ValueError("matched-control round provenance is stale")
            primary_pressure_receipt = _load(primary_training_receipt)
            if (
                primary_pressure_receipt.get("arm") != "primary"
                or int(primary_pressure_receipt.get("round", -1)) != round_index
                or int(primary_pressure_receipt.get("seed", -1)) != primary_seed
                or primary_pressure_receipt.get("config_sha256")
                != sha256_file(config_path)
                or not primary_pressure_receipt.get("round_gate", {}).get("passed")
            ):
                raise ValueError("primary pressure receipt is stale")
            try:
                primary_pressure = float(
                    primary_pressure_receipt["initial_probe"]["mean_loss"]
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("primary pressure receipt lacks initial loss") from exc
            if name == "offpolicy_sft":
                _audit_offpolicy_training_receipt(
                    training_receipt,
                    config=config,
                    config_path=config_path,
                    base_model=base_model,
                    round_manifest=round_manifest,
                    round_index=round_index,
                    seed=seed,
                    recorded_gate=round_row.get("round_gate"),
                    expected_target_initial_loss=primary_pressure,
                )
            else:
                if not source_cache.is_file() or round_row.get(
                    "primary_target_cache_sha256"
                ) != sha256_file(source_cache):
                    raise ValueError("matched-control source cache provenance is stale")
                expected_cache = source_cache
                if name == "non_advantage_route":
                    rematch_manifest = (
                        data_root
                        / "control_overlays"
                        / "non_advantage_route"
                        / "rematch_manifest.json"
                    )
                    if (
                        round_row.get("control_rematch_manifest")
                        != str(rematch_manifest)
                        or not rematch_manifest.is_file()
                        or round_row.get("control_rematch_manifest_sha256")
                        != sha256_file(rematch_manifest)
                    ):
                        raise ValueError("control rematch manifest provenance is stale")
                    rematch = validate_control_rematch_manifest(
                        rematch_manifest,
                        config=config,
                        config_path=config_path,
                        source_manifest=round_manifest,
                        source_cache=source_cache,
                    )
                    replacements = int(rematch.get("replacement_count", -1))
                    if replacements < 0:
                        raise ValueError("control rematch replacement count is invalid")
                    if replacements:
                        expected_cache = (
                            rematch_manifest.parent / "all_policy_targets.pt"
                        )
                        validate_control_overlay_cache(
                            expected_cache,
                            rematch_manifest=rematch_manifest,
                            source_cache=source_cache,
                            config=config,
                            config_path=config_path,
                        )
                elif (
                    round_row.get("control_rematch_manifest") is not None
                    or round_row.get("control_rematch_manifest_sha256") is not None
                ):
                    raise ValueError("wrong-teacher control unexpectedly used a rematch")
                if (
                    round_row.get("effective_target_cache") != str(expected_cache)
                    or not expected_cache.is_file()
                    or round_row.get("effective_target_cache_sha256")
                    != sha256_file(expected_cache)
                ):
                    raise ValueError("matched-control target cache provenance is stale")
                _audit_mopd_training_receipt(
                    training_receipt,
                    config=config,
                    config_path=config_path,
                    base_model=base_model,
                    round_manifest=round_manifest,
                    target_cache=expected_cache,
                    round_index=round_index,
                    seed=seed,
                    arm=name,
                    recorded_gate=round_row.get("round_gate"),
                    expected_target_initial_loss=primary_pressure,
                )
            if round_row.get("merge_receipt_sha256") != _audit_merge_receipt(
                merged, base_model, adapter
            ):
                raise ValueError("matched-control merge provenance is stale")
        models[name] = model

    for index, (name, model) in enumerate(expected_parameters.items()):
        row = parameters[name]
        expected_weight = float(config["controls"]["parameter_merge_deep_weights"][index])
        _audit_parameter_control(
            config,
            model=model,
            row=row,
            expected_weight=expected_weight,
        )
        models[name] = model
    return {
        "path": str(CONTROLS.resolve()),
        "sha256": sha256_file(CONTROLS),
    }, models


def _audit_confirmation(
    config: dict,
    config_path: Path,
    expected_arm_models: dict[str, Path],
) -> tuple[dict, list[dict], dict[str, dict[str, str]]]:
    confirmation_path = _validated_confirmation_analysis_file(CONFIRMATION)
    confirmation = _load(confirmation_path)
    if (
        confirmation.get("stage")
        != "two_block_same_prefix_advantage_confirmation"
        or confirmation.get("config_sha256") != sha256_file(config_path)
        or not confirmation.get("gate", {}).get("passed")
        or confirmation.get("downstream_authorization") != "benchmark_cli"
    ):
        raise ValueError("procedural confirmation did not authorize benchmarks")
    try:
        manifest_path = Path(confirmation["manifest"]).resolve()
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("confirmation manifest path is invalid") from exc
    expected_manifest = (CONFIRMATION_ROOT / "manifest.json").resolve()
    if manifest_path != expected_manifest or not manifest_path.is_file():
        raise ValueError("confirmation manifest is not the sealed experiment manifest")
    manifest_sha = sha256_file(manifest_path)
    if confirmation.get("manifest_sha256") != manifest_sha:
        raise ValueError("confirmation manifest hash mismatch")
    manifest = _load(manifest_path)
    recorded_authorization = manifest.get("controls_authorization")
    recorded_admission = manifest.get("confirmation_admission")
    if not isinstance(recorded_authorization, dict) or not isinstance(
        recorded_admission, dict
    ):
        raise ValueError("confirmation manifest lacks pre-admission provenance")
    authorization_binding = controls_authorization_binding(
        Path(str(recorded_authorization.get("path", ""))),
        expected_config_sha256=sha256_file(config_path),
    )
    admission_binding = confirmation_admission_binding(
        Path(str(recorded_admission.get("path", ""))),
        expected_config_sha256=sha256_file(config_path),
        expected_controls_authorization=authorization_binding,
    )
    expected_seeds = [int(value) for value in config["seeds"]["confirmatory_blocks"]]
    raw_root = configured_confirmation_raw_root(config)
    validate_confirmation_campaign_tree(
        Path(str(admission_binding["path"])),
        raw_root=raw_root,
        terminal=True,
        require_manifest=True,
    )
    arms = manifest.get("arms")
    model_receipts = manifest.get("model_merge_receipts")
    model_inventories = manifest.get("model_inference_inventories")
    evaluator_source = confirmation_evaluator_source_inventory()
    integration_seeds = [int(value) for value in config["seeds"]["integration_training"]]
    primary_arm = f"primary_seed{integration_seeds[0]}"
    replicate_arms = [f"primary_seed{seed}" for seed in integration_seeds[1:]]
    strict_arms = [
        "quick",
        "deep",
        "soup",
        "non_advantage_route",
        "wrong_teacher",
        "offpolicy_sft",
        *_parameter_models(config),
    ]
    if (
        manifest.get("stage") != "sealed_confirmation_manifest"
        or manifest.get("config_sha256") != sha256_file(config_path)
        or manifest.get("block_seeds") != expected_seeds
        or manifest.get("primary_arm") != primary_arm
        or manifest.get("replicate_arms") != replicate_arms
        or manifest.get("quick_arm") != "quick"
        or manifest.get("deep_arm") != "deep"
        or manifest.get("soup_arm") != "soup"
        or manifest.get("sample_more_arm") != "soup_best8"
        or manifest.get("evaluator_sha256") != sha256_file(CONFIRMATION_EVALUATOR)
        or manifest.get("evaluator_source_inventory") != evaluator_source
        or recorded_authorization != authorization_binding
        or recorded_admission != admission_binding
        or confirmation.get("controls_authorization") != authorization_binding
        or confirmation.get("confirmation_admission") != admission_binding
        or manifest.get("strict_comparator_arms") != strict_arms
        or not isinstance(arms, dict)
        or not isinstance(model_receipts, dict)
        or not isinstance(model_inventories, dict)
        or set(arms) != set(expected_arm_models)
        or set(arms) != set(model_receipts)
        or set(arms) != set(model_inventories)
    ):
        raise ValueError("confirmation manifest protocol is stale")

    score_artifacts = []
    seen_score_paths: set[Path] = set()
    observed_models: dict[Path, dict[str, str]] = {}
    arm_models: dict[str, dict[str, str]] = {}
    for arm, values in arms.items():
        if not isinstance(values, list) or len(values) != len(expected_seeds):
            raise ValueError("confirmation arm block inventory is stale")
        block_models = set()
        for block_index, (block_seed, value) in enumerate(zip(expected_seeds, values)):
            recorded_score_path = Path(value)
            score_path = recorded_score_path.resolve()
            if not score_path.is_relative_to(CONFIRMATION_ROOT):
                raise ValueError("confirmation artifact escaped its sealed directory")
            if score_path in seen_score_paths:
                raise ValueError("confirmation score artifact was reused across arms")
            seen_score_paths.add(score_path)
            score = validate_confirmation_score_artifacts(
                recorded_score_path,
                expected_tag=f"block_{block_index}_{arm}",
                raw_root=raw_root,
            )
            validate_confirmation_geometry(score, config)
            expected_decode = "sample8" if arm == "soup_best8" else "greedy"
            expected_k = (
                int(config["controls"]["sample_more_k"])
                if arm == "soup_best8"
                else 1
            )
            engine_protocol = score.get("engine_protocol")
            backend_protocol, backend_fingerprint = canonical_backend_protocol(
                score.get("runner_summary"),
                expected_engine=config["engine"],
                expected_model=str(score.get("model", "")),
            )
            expected_sampling = expected_confirmation_sampling_protocol(
                config, decode=expected_decode, block_seed=block_seed
            )
            if (
                score.get("stage") != "policy_eval"
                or score.get("scope") != "confirmatory"
                or score.get("evaluator_sha256")
                != sha256_file(CONFIRMATION_EVALUATOR)
                or score.get("evaluator_source_inventory_sha256")
                != evaluator_source["sha256"]
                or int(score.get("evaluator_source_file_count", -1))
                != evaluator_source["file_count"]
                or score.get("config_sha256") != sha256_file(config_path)
                or int(score.get("block_seed", -1)) != block_seed
                or score.get("model_merge_receipt_sha256")
                != model_receipts[arm]
                or score.get("model_inference_inventory_sha256")
                != model_inventories[arm]
                or score.get("decode") != expected_decode
                or int(score.get("k", -1)) != expected_k
                or not isinstance(engine_protocol, dict)
                or not engine_protocol
                or not all(engine_protocol.values())
                or score.get("backend_protocol") != backend_protocol
                or score.get("backend_fingerprint") != backend_fingerprint
                or score.get("sampling_protocol") != expected_sampling
                or not isinstance(score.get("task_manifest_sha256"), str)
                or not isinstance(score.get("ordered_plan_sha256"), str)
                or set(score.get("token_ledger") or {}) != {"sampled_tokens"}
                or score.get("controls_authorization") != authorization_binding
                or score.get("confirmation_admission") != admission_binding
                or not isinstance(score.get("items"), list)
                or not score["items"]
            ):
                raise ValueError("confirmation score provenance is stale")
            model = Path(score["model"]).resolve()
            artifacts_root = (REPO / "large_artifacts").resolve()
            if not model.is_relative_to(artifacts_root):
                raise ValueError("confirmation model escaped the artifact root")
            if model not in observed_models:
                observed_models[model] = model_provenance(model)
            if (
                observed_models[model]["model_merge_receipt_sha256"]
                != model_receipts[arm]
                or observed_models[model]["model_inference_inventory_sha256"]
                != model_inventories[arm]
                or observed_models[model]["model_config_sha256"]
                != score.get("model_config_sha256")
            ):
                raise ValueError("confirmation model changed after scoring")
            block_models.add(model)
            score_artifacts.append(
                {"path": str(score_path), "sha256": sha256_file(score_path)}
            )
            raw_dir = confirmation_raw_dir(recorded_score_path, raw_root=raw_root)
            score_artifacts.extend(
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in sorted(raw_dir.iterdir())
                if path.is_file() and not path.is_symlink()
            )
        if len(block_models) != 1:
            raise ValueError("confirmation arm used different models across blocks")
        arm_models[str(arm)] = observed_models[next(iter(block_models))]

    for arm, expected_model in expected_arm_models.items():
        if arm not in arm_models or Path(arm_models[arm]["model"]) != expected_model:
            raise ValueError("confirmation arm/model mapping is invalid")

    score_artifacts.extend(
        [
            {
                "path": authorization_binding["path"],
                "sha256": authorization_binding["sha256"],
            },
            {
                "path": admission_binding["path"],
                "sha256": admission_binding["sha256"],
            },
        ]
    )

    completed = subprocess.run(
        [
            str(PY),
            str(CONFIRMATION_ANALYZER),
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--out",
            str(confirmation_path),
        ],
        cwd=REPO,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("independent confirmation reanalysis did not pass")
    regenerated = _load(_validated_confirmation_analysis_file(confirmation_path))
    if not _strict_json_equal(regenerated, confirmation):
        raise ValueError("confirmation receipt differs from independent reanalysis")
    return confirmation, sorted(score_artifacts, key=lambda row: row["path"]), arm_models


def _build_authorization(config: dict, config_path: Path) -> dict:
    code_before = _code_provenance()
    preregistration = _audit_preregistration()
    validate_source_checkpoint_receipts(config)
    integration_artifacts = []
    integration_models: dict[str, Path] = {}
    for seed_value in config["seeds"]["integration_training"]:
        seed = int(seed_value)
        model = _integration_model(config, seed)
        integration_artifacts.append(
            _audit_integration(config, config_path, seed, model)
        )
        integration_models[f"primary_seed{seed}"] = model
    controls_artifact, control_models = _audit_controls(config, config_path)
    tier_models = {
        "quick": _expected_models(config, "quick")["visible"],
        "deep": _expected_models(config, "medium")["visible"],
        "soup": _expected_models(config, "quick")["soup"],
    }
    expected_arm_models = {
        **tier_models,
        **integration_models,
        **control_models,
        "soup_best8": tier_models["soup"],
    }
    confirmation, score_artifacts, confirmation_models = _audit_confirmation(
        config, config_path, expected_arm_models
    )
    confirmation_binding_before = _confirmation_analysis_binding(
        CONFIRMATION, confirmation
    )

    tier_seeds = _tier_seeds(config)
    events = []
    event_models: dict[Path, dict[str, str]] = {
        Path(value["model"]): value for value in confirmation_models.values()
    }
    for tier, seeds in tier_seeds.items():
        models = _expected_models(config, tier)
        for label, model in models.items():
            if model not in event_models:
                event_models[model] = model_provenance(model)
            provenance = event_models[model]
            for seed in seeds:
                events.append(
                    {
                        "tier": tier,
                        "seed": seed,
                        "label": label,
                        **provenance,
                    }
                )
    events.sort(key=lambda row: (row["tier"], row["seed"], row["label"]))
    if len(events) != BENCHMARK_EVENT_COUNT:
        raise ValueError("frozen benchmark inventory is not exactly 33 events")
    evidence_artifacts = sorted(
        [*integration_artifacts, controls_artifact, *score_artifacts],
        key=lambda row: row["path"],
    )
    code_after = _code_provenance()
    if code_after != code_before:
        raise ValueError("authorization code changed during the provenance audit")
    for arm, model in expected_arm_models.items():
        if model_provenance(model) != confirmation_models[arm]:
            raise ValueError("confirmation model changed during authorization")
    confirmation_binding_after = _confirmation_analysis_binding(
        CONFIRMATION, confirmation
    )
    if confirmation_binding_after != confirmation_binding_before:
        raise ValueError("confirmation analysis changed during authorization")
    return {
        "schema_version": 2,
        "stage": "benchmark_aggregate_authorization",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "preregistration": str(PREREGISTRATION.resolve()),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "design_commit": preregistration["design_commit"],
        "integration_receipts": integration_artifacts,
        "controls_receipt": controls_artifact,
        "confirmation": confirmation_binding_after["path"],
        "confirmation_sha256": confirmation_binding_after["sha256"],
        "confirmation_manifest_sha256": confirmation["manifest_sha256"],
        "confirmation_artifacts": score_artifacts,
        "evidence_artifacts": evidence_artifacts,
        "confirmation_models": confirmation_models,
        **code_after,
        "backend": "qwen_vllm",
        "events": events,
        "gate": {"passed": True},
        "downstream_authorization": "aggregate_only_benchmark_cli",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=EXP / "analysis" / "benchmark_authorization.json"
    )
    args = parser.parse_args()
    config, config_path = load_config(CONFIG)
    result = _build_authorization(config, config_path)
    existing_regular_file = args.out.is_file() and not args.out.is_symlink()
    if not existing_regular_file and (args.out.exists() or args.out.is_symlink()):
        raise SystemExit("benchmark authorization output path is not a file")
    if (
        not existing_regular_file
        and BENCHMARK_ROOT.exists()
        and any(BENCHMARK_ROOT.rglob("*"))
    ):
        raise SystemExit("benchmark artifacts exist before initial authorization")
    try:
        _publish_authorization_no_clobber(args.out, result)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
