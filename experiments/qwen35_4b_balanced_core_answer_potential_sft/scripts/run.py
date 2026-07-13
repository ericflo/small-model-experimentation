#!/usr/bin/env python3
"""Restartable long-horizon answer-potential experiment orchestrator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import yaml  # noqa: E402

from hf_scorer import HFAnswerPotentialScorer  # noqa: E402
from io_utils import read_json, read_jsonl, sha256_file, write_json, write_jsonl  # noqa: E402
from model_ops import AnswerPotentialModel, answer_mention  # noqa: E402
from pivot import choose_pivot, natural_checkpoint_indices  # noqa: E402
from selector import (  # noqa: E402
    deranged_sources,
    oversample_to,
    select_task,
    sft_record,
)
from shards import read_jsonl_gz, valid_receipt, write_jsonl_gz  # noqa: E402
from stats import kendall_tau_b, mean, paired_bootstrap, roc_auc  # noqa: E402
from task_data import build_all  # noqa: E402
from vllm_runner import (  # noqa: E402
    EngineConfig,
    MODEL_ID,
    MODEL_REVISION,
    SamplingConfig,
    VLLMRunner,
)

CONFIG_PATH = EXP / "configs" / "default.yaml"
DATA_DIR = EXP / "data" / "procedural"
RUNS_DIR = EXP / "runs"
AMENDMENT_RECEIPT_PATH = RUNS_DIR / "preselection_amendment_receipt.json"
AMENDMENT_FROZEN_FILES = tuple(
    sorted(
        {
            str(path.relative_to(ROOT))
            for directory, pattern in (
                (EXP / "configs", "*.yaml"),
                (EXP / "data" / "procedural", "*.json"),
                (EXP / "data" / "procedural", "*.jsonl"),
                (EXP / "scripts", "*.py"),
                (EXP / "src", "*.py"),
            )
            for path in directory.rglob(pattern)
            if not any(
                part.startswith(".") for part in path.relative_to(directory).parts
            )
        }
        | {
            str((EXP / "reports" / "design_review.md").relative_to(ROOT)),
            str((EXP / "reports" / "preregistration.md").relative_to(ROOT)),
            "requirements-training.lock.txt",
            "requirements-vllm.lock.txt",
        }
    )
)


def optional_mean(values: list[float | int | bool]) -> float | None:
    return sum(float(value) for value in values) / len(values) if values else None


def _run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def git_file_sha256(commit: str, path: str) -> str:
    payload = _run(["git", "show", f"{commit}:{path}"]).stdout.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def base_model_fingerprint() -> str:
    return canonical_sha256({"model": MODEL_ID, "revision": MODEL_REVISION})


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    excluded = {"merge_receipt.json", "merge_application_receipt.json"}
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file() and item.name not in excluded
    )
    if not (path / "config.json").is_file() or not any(
        item.suffix == ".safetensors" for item in files
    ):
        raise RuntimeError(f"incomplete merged checkpoint: {path}")
    entries = [
        {
            "name": str(item.relative_to(path)),
            "bytes": item.stat().st_size,
            "sha256": sha256_file(item),
        }
        for item in files
    ]
    return {"sha256": canonical_sha256(entries), "files": entries}


def sampling_contract(sampling: SamplingConfig) -> dict[str, Any]:
    return dict(vars(sampling))


def task_scope_sha256(config: dict[str, Any], split: str) -> str:
    items = load_core_train(config) if split == "train" else load_split(split)
    return canonical_sha256(
        [
            {
                "id": item["id"],
                "prompt": item["prompt"],
                "canonical_answer": item["canonical_answer"],
                "family": item["family"],
                "level": item["level"],
            }
            for item in items
        ]
    )


def score_operation_contract(config: dict[str, Any]) -> str:
    return canonical_sha256(
        {
            "model": config["model"],
            "task_scope_sha256": task_scope_sha256(config, "train"),
            "scoring": config["scoring"],
            "max_train_length": int(config["selector"]["max_train_length"]),
            "scorer_sha256": sha256_file(EXP / "src" / "hf_scorer.py"),
            "environment_lock_sha256": sha256_file(
                ROOT / "requirements-vllm.lock.txt"
            ),
            "canonical_only": True,
        }
    )


def rollout_operation_contract(
    config: dict[str, Any], *, source_stages: list[str], r: int
) -> str:
    sampling = config["sampling"]
    return canonical_sha256(
        {
            "model": config["model"],
            "task_scope_sha256": task_scope_sha256(config, "train"),
            "engine": config["engine"],
            "source_stages": source_stages,
            "r": r,
            "answer_max_tokens": int(sampling["answer_max_tokens"]),
            "run_seed": int(sampling["continuation_seed"]) + 101 * r,
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "top_k": int(sampling["top_k"]),
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
            "model_ops_sha256": sha256_file(EXP / "src" / "model_ops.py"),
            "vllm_lock_sha256": sha256_file(ROOT / "requirements-vllm.lock.txt"),
        }
    )


def raw_pool_operation_contract(
    config: dict[str, Any], *, split: str, n: int, stage: str
) -> str:
    sampling = config["sampling"]
    return canonical_sha256(
        {
            "model": config["model"],
            "task_scope_sha256": task_scope_sha256(config, split),
            "engine": config["engine"],
            "split": split,
            "stage": stage,
            "n": n,
            "natural_close_allowance": int(sampling["natural_close_allowance"]),
            "nonloop_continuation_tokens": int(
                sampling["nonloop_continuation_tokens"]
            ),
            "run_seed": int(sampling["run_seed"]),
            "continuation_seed": int(sampling["continuation_seed"]),
            "temperature": float(sampling["temperature"]),
            "top_p": float(sampling["top_p"]),
            "top_k": int(sampling["top_k"]),
            "logprobs": int(sampling["logprobs"]),
            "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
            "model_ops_sha256": sha256_file(EXP / "src" / "model_ops.py"),
            "environment_lock_sha256": sha256_file(
                ROOT / "requirements-vllm.lock.txt"
            ),
            "minimum_natural_per_task": int(
                config["selector"]["minimum_natural_per_task"]
            ),
            "topup": {
                "n": 16,
                "maximum_batches": 4,
                "seed_stride": 10_000,
            },
            "inherited_source_index_sha256": config.get("inherit", {}).get(
                "source_index_sha256"
            ),
        }
    )


def selection_evidence_contracts(config: dict[str, Any]) -> dict[str, str]:
    return {
        "independent": raw_pool_operation_contract(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        ),
        "independent_scores": score_operation_contract(config),
        "rollouts": rollout_operation_contract(
            config,
            source_stages=["train_independent"],
            r=int(config["sampling"]["train_rollouts_per_trace"]),
        ),
    }


def selection_evidence_bundle(
    config: dict[str, Any], *, allow_missing_contracts: bool
) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]
]:
    """Validate and fingerprint the exact three indexes consumed by selection."""
    root = external_root(config)
    source_names = {
        "independent": "train_independent",
        "independent_scores": "train_independent_scores",
        "rollouts": "train_rollouts_r1",
    }
    index_paths = {
        key: root / "pools" / stage / "index.json"
        for key, stage in source_names.items()
    }
    missing = [str(path) for path in index_paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"selection evidence index missing: {missing}")
    indices = {key: read_json(path) for key, path in index_paths.items()}
    expected_contracts = selection_evidence_contracts(config)
    for key, expected in expected_contracts.items():
        observed = indices[key].get("operation_contract_sha256")
        if observed not in (None, expected):
            raise RuntimeError(f"{key} operation contract mismatch")
        if observed is None and not allow_missing_contracts:
            raise RuntimeError(f"{key} operation contract is not sealed")

    for key, index in indices.items():
        if (
            index.get("stage") != source_names[key]
            or index.get("split") != "train"
            or index.get("model") != MODEL_ID
            or index.get("revision") != MODEL_REVISION
        ):
            raise RuntimeError(f"{key} evidence-index identity mismatch")
    if indices["independent_scores"].get("backend") != str(
        config["scoring"]["backend"]
    ):
        raise RuntimeError("selection score backend mismatch")

    task_ids = {str(item["id"]) for item in load_core_train(config)}
    expected_tasks = int(config["splits"]["full_train_tasks"])
    if len(task_ids) != expected_tasks:
        raise RuntimeError("frozen train scope count mismatch during evidence seal")
    if any(set(index.get("shards", {})) != task_ids for index in indices.values()):
        raise RuntimeError("selection evidence does not cover the exact frozen task scope")

    for task_id in sorted(task_ids):
        raw_entry = indices["independent"]["shards"][task_id]
        score_entry = indices["independent_scores"]["shards"][task_id]
        rollout_entry = indices["rollouts"]["shards"][task_id]
        for label, entry in (
            ("raw", raw_entry),
            ("score", score_entry),
            ("rollout", rollout_entry),
        ):
            if not valid_receipt(entry["artifact"]):
                raise RuntimeError(f"invalid {label} evidence artifact: {task_id}")
        raw_sha256 = raw_entry["artifact"]["sha256"]
        if score_entry.get("source_artifact_sha256") != raw_sha256:
            raise RuntimeError(f"score/raw provenance mismatch: {task_id}")
        if rollout_entry.get("source_sha256") != [raw_sha256]:
            raise RuntimeError(f"rollout/raw provenance mismatch: {task_id}")
        if int(score_entry.get("eligible", -1)) != int(
            score_entry["artifact"]["rows"]
        ):
            raise RuntimeError(f"score eligibility receipt mismatch: {task_id}")
        if int(rollout_entry.get("eligible", -1)) != int(
            rollout_entry["artifact"]["rows"]
        ):
            raise RuntimeError(f"rollout eligibility receipt mismatch: {task_id}")
        if int(score_entry["artifact"]["rows"]) != int(
            rollout_entry["artifact"]["rows"]
        ):
            raise RuntimeError(f"score/rollout row-count mismatch: {task_id}")
        if allow_missing_contracts:
            raw_rows = read_jsonl_gz(Path(raw_entry["artifact"]["path"]))
            score_rows = read_jsonl_gz(Path(score_entry["artifact"]["path"]))
            rollout_rows = read_jsonl_gz(Path(rollout_entry["artifact"]["path"]))
            if any(
                str(row.get("task_id")) != task_id
                for row in (*raw_rows, *score_rows, *rollout_rows)
            ):
                raise RuntimeError(f"evidence shard contains another task ID: {task_id}")
            raw_ids = [str(row["trace_id"]) for row in raw_rows]
            score_ids = [str(row["trace_id"]) for row in score_rows]
            rollout_ids = [str(row["trace_id"]) for row in rollout_rows]
            if (
                len(raw_ids) != len(set(raw_ids))
                or len(score_ids) != len(set(score_ids))
                or len(rollout_ids) != len(set(rollout_ids))
                or set(score_ids) != set(rollout_ids)
                or not set(score_ids).issubset(set(raw_ids))
            ):
                raise RuntimeError(f"evidence trace-ID join mismatch: {task_id}")
            eligible_raw_ids = {
                str(row["trace_id"])
                for row in raw_rows
                if bool(row["natural_close"]) and not bool(row["loop_flag"])
            }
            if set(score_ids) != eligible_raw_ids:
                raise RuntimeError(f"evidence eligibility-set mismatch: {task_id}")

    summaries = {
        key: {
            "stage": source_names[key],
            "index": str(index_paths[key]),
            "index_sha256": sha256_file(index_paths[key]),
            "tasks": len(indices[key]["shards"]),
            "rows": sum(
                int(entry["artifact"]["rows"])
                for entry in indices[key]["shards"].values()
            ),
            "operation_contract_sha256": expected_contracts[key],
        }
        for key in source_names
    }
    if summaries["independent_scores"]["rows"] != summaries["rollouts"]["rows"]:
        raise RuntimeError("score and rollout evidence totals differ")
    if summaries["independent"]["rows"] < (
        expected_tasks * int(config["sampling"]["train_independent_n"])
    ):
        raise RuntimeError("raw evidence total is below the frozen N per task")
    return indices, expected_contracts, summaries


def apply_selection_operation_contracts(
    config: dict[str, Any],
    indices: dict[str, dict[str, Any]],
    contracts: dict[str, str],
) -> None:
    """Apply a validated retrospective attestation, safely across restarts."""
    root = external_root(config)
    source_names = {
        "independent": "train_independent",
        "independent_scores": "train_independent_scores",
        "rollouts": "train_rollouts_r1",
    }
    for key, stage in source_names.items():
        expected = contracts[key]
        observed = indices[key].get("operation_contract_sha256")
        if observed not in (None, expected):
            raise RuntimeError(f"{key} operation contract changed before attestation")
        if observed is None:
            indices[key]["operation_contract_sha256"] = expected
            indices[key]["operation_contract_sealed_post_run"] = True
            write_json(root / "pools" / stage / "index.json", indices[key])


def committed_amendment_file_hashes(commit: str) -> dict[str, str]:
    """Require every frozen scientific input to match one committed tree."""
    frozen_files: dict[str, str] = {}
    for relative in AMENDMENT_FROZEN_FILES:
        path = ROOT / relative
        current_sha256 = sha256_file(path)
        if git_file_sha256(commit, relative) != current_sha256:
            raise RuntimeError(
                f"cannot seal an uncommitted amendment file: {relative}"
            )
        frozen_files[relative] = current_sha256
    return frozen_files


def seal_selection_evidence(config: dict[str, Any]) -> dict[str, Any]:
    """One-time post-run seal before the amendment receipt is committed."""
    root = external_root(config)
    stage_keys = {
        "independent": "train_independent",
        "independent_scores": "train_independent_scores",
        "rollouts": "train_rollouts_r1",
    }
    forbidden = [
        root / "selection",
        root / "sft",
        root / "adapters",
        root / "merged",
        EXP / "data" / "sft_manifest.json",
    ]
    present = [str(path) for path in forbidden if path.exists()]
    if present:
        raise RuntimeError(
            f"cannot create a pre-selection seal after downstream artifacts: {present}"
        )
    existing_seal_path = RUNS_DIR / "preselection_evidence_seal.json"
    if existing_seal_path.is_file():
        existing = read_json(existing_seal_path)
        _, _, current_summaries = selection_evidence_bundle(
            config, allow_missing_contracts=False
        )
        operation_path = RUNS_DIR / "evidence_operation_contracts.json"
        legacy_path = RUNS_DIR / "preselection_legacy_indexes.json"
        if (
            existing.get("passed") is not True
            or existing.get("selection_artifacts_absent") is not True
            or existing.get("adapter_artifacts_absent") is not True
            or existing.get("indexes") != current_summaries
            or not operation_path.is_file()
            or not legacy_path.is_file()
            or existing.get("evidence_operation_contracts_sha256")
            != sha256_file(operation_path)
            or existing.get("legacy_index_receipt_sha256")
            != sha256_file(legacy_path)
        ):
            raise RuntimeError("existing pre-selection evidence seal is stale")
        committed_amendment_file_hashes(
            _run(["git", "rev-parse", "HEAD"]).stdout.strip()
        )
        design_boundary_receipt(config)
        write_preselection_amendment_receipt(config, existing)
        return existing

    # This complete pass is deliberately read-only.  Incomplete or corrupt
    # evidence must leave neither a legacy receipt nor partially attested
    # upstream indexes behind.
    pre_indices, contracts, _ = selection_evidence_bundle(
        config, allow_missing_contracts=True
    )
    committed_amendment_file_hashes(
        _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    )
    design_boundary_receipt(config)
    legacy_path = RUNS_DIR / "preselection_legacy_indexes.json"
    if legacy_path.is_file():
        legacy = read_json(legacy_path)
    else:
        legacy = {
            "schema_version": 1,
            "classification": "retrospective_pre_operation_contract_attestation",
            "indexes": {
                stage: {
                    "path": str(root / "pools" / stage / "index.json"),
                    "sha256": sha256_file(root / "pools" / stage / "index.json"),
                    "observed_operation_contract_sha256": pre_indices[key].get(
                        "operation_contract_sha256"
                    ),
                }
                for key, stage in stage_keys.items()
            },
        }
        write_json(legacy_path, legacy)
    if set(legacy.get("indexes", {})) != set(stage_keys.values()):
        raise RuntimeError("legacy evidence-index receipt scope mismatch")
    for key, stage in stage_keys.items():
        legacy_entry = legacy["indexes"][stage]
        legacy_contract = legacy_entry.get("observed_operation_contract_sha256")
        if (
            legacy_entry.get("path")
            != str(root / "pools" / stage / "index.json")
            or len(str(legacy_entry.get("sha256", ""))) != 64
            or legacy_contract not in (None, contracts[key])
        ):
            raise RuntimeError(f"legacy evidence-index receipt mismatch: {stage}")
        current_contract = pre_indices[key].get("operation_contract_sha256")
        if current_contract is None or legacy_contract == contracts[key]:
            reconstructed_sha256 = sha256_file(
                root / "pools" / stage / "index.json"
            )
        else:
            reconstructed = dict(pre_indices[key])
            reconstructed.pop("operation_contract_sha256", None)
            reconstructed.pop("operation_contract_sealed_post_run", None)
            reconstructed_sha256 = hashlib.sha256(
                (
                    json.dumps(
                        reconstructed,
                        indent=2,
                        sort_keys=True,
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
            ).hexdigest()
        if reconstructed_sha256 != legacy_entry["sha256"]:
            raise RuntimeError(f"unattested legacy evidence-index drift: {stage}")
    original_index_sha256 = {
        stage: str(legacy["indexes"][stage]["sha256"])
        for stage in stage_keys.values()
    }
    apply_selection_operation_contracts(config, pre_indices, contracts)
    indices, contracts, summaries = selection_evidence_bundle(
        config, allow_missing_contracts=False
    )
    operation_receipt = {
        "schema_version": 1,
        "contracts": contracts,
        "score_backend": indices["independent_scores"].get("backend"),
        "indexes": summaries,
    }
    operation_path = RUNS_DIR / "evidence_operation_contracts.json"
    write_json(operation_path, operation_receipt)
    result = {
        "schema_version": 1,
        "passed": True,
        "indexes": summaries,
        "legacy_index_sha256_before_operation_seal": original_index_sha256,
        "legacy_index_receipt_sha256": sha256_file(legacy_path),
        "operation_contract_attestation": (
            "retrospective post-run attestation over preserved artifacts; "
            "not metadata emitted by the original generation/scoring process"
        ),
        "evidence_operation_contracts_sha256": sha256_file(operation_path),
        "selection_artifacts_absent": True,
        "adapter_artifacts_absent": True,
    }
    write_json(existing_seal_path, result)
    write_preselection_amendment_receipt(config, result)
    return result


def resolve_preselection_amendment_commit() -> str:
    """Keep commit A stable across later commits; recover it after a rebase."""
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    receipt_rel = str(AMENDMENT_RECEIPT_PATH.relative_to(ROOT))
    if AMENDMENT_RECEIPT_PATH.is_file():
        stored = str(read_json(AMENDMENT_RECEIPT_PATH).get("amendment_commit", ""))
        if stored and (
            _run(
                ["git", "merge-base", "--is-ancestor", stored, head],
                check=False,
            ).returncode
            == 0
        ):
            return stored
    receipt_is_committed = (
        _run(["git", "cat-file", "-e", f"HEAD:{receipt_rel}"], check=False).returncode
        == 0
    )
    if not receipt_is_committed:
        return head
    receipt_commit = _run(
        [
            "git",
            "log",
            "-1",
            "--diff-filter=A",
            "--format=%H",
            "--",
            receipt_rel,
        ]
    ).stdout.strip()
    if not receipt_commit:
        raise RuntimeError("cannot recover the amendment boundary after rebase")
    return _run(["git", "rev-parse", f"{receipt_commit}^"]).stdout.strip()


def write_preselection_amendment_receipt(
    config: dict[str, Any], evidence_seal: dict[str, Any]
) -> dict[str, Any]:
    """Materialize the commit-B receipt while selection remains impossible."""
    if config["preselection_amendment"]["classification"] != (
        "post_score_partial_rollout_pre_official_selection_deviation"
    ):
        raise RuntimeError("unexpected amendment classification")
    amendment_commit = resolve_preselection_amendment_commit()
    frozen_files = committed_amendment_file_hashes(amendment_commit)
    operation_path = RUNS_DIR / "evidence_operation_contracts.json"
    legacy_path = RUNS_DIR / "preselection_legacy_indexes.json"
    evidence_seal_path = RUNS_DIR / "preselection_evidence_seal.json"
    receipt = {
        "schema_version": 1,
        "passed": True,
        "classification": (
            "post_score_partial_rollout_pre_official_selection_deviation"
        ),
        "amendment_commit": amendment_commit,
        "frozen_files": frozen_files,
        "evidence_indexes": evidence_seal["indexes"],
        "evidence_operation_contracts_sha256": sha256_file(operation_path),
        "legacy_index_receipt_sha256": sha256_file(legacy_path),
        "evidence_seal_sha256": sha256_file(evidence_seal_path),
        "selection_artifacts_absent_at_seal": bool(
            evidence_seal["selection_artifacts_absent"]
        ),
        "adapter_artifacts_absent_at_seal": bool(
            evidence_seal["adapter_artifacts_absent"]
        ),
        "deviation_disclosure": {
            "candidate_scores_observed_before_commit": True,
            "original_helper_retained_tasks": 116,
            "official_selection_observed": False,
            "partial_rollout_labels_observed_before_commit": True,
            "partial_rollout_labels_used_for_cost_planning": True,
            "rollout_labels_informed_repair": False,
            "adapter_or_evaluation_outcome_observed": False,
        },
    }
    if (
        not AMENDMENT_RECEIPT_PATH.is_file()
        or read_json(AMENDMENT_RECEIPT_PATH) != receipt
    ):
        write_json(AMENDMENT_RECEIPT_PATH, receipt)
    return receipt


def validate_preselection_amendment_receipt(
    config: dict[str, Any],
) -> tuple[
    dict[str, dict[str, Any]], dict[str, str], dict[str, dict[str, Any]]
]:
    """Fail closed unless committed amendment code and evidence remain exact."""
    if not AMENDMENT_RECEIPT_PATH.is_file():
        raise RuntimeError(
            "missing committed pre-selection amendment receipt; run evidence-seal, "
            "commit its receipt, and do not select yet"
        )
    receipt = read_json(AMENDMENT_RECEIPT_PATH)
    receipt_rel = str(AMENDMENT_RECEIPT_PATH.relative_to(ROOT))
    try:
        committed_receipt_sha256 = git_file_sha256("HEAD", receipt_rel)
    except subprocess.CalledProcessError as error:
        raise RuntimeError("pre-selection amendment receipt is not committed") from error
    if committed_receipt_sha256 != sha256_file(AMENDMENT_RECEIPT_PATH):
        raise RuntimeError("pre-selection amendment receipt differs from committed HEAD")

    amendment_commit = str(receipt.get("amendment_commit", ""))
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    if not amendment_commit or (
        _run(
            ["git", "merge-base", "--is-ancestor", amendment_commit, head],
            check=False,
        ).returncode
        != 0
    ):
        raise RuntimeError("pre-selection amendment commit is not an ancestor of HEAD")
    frozen = receipt.get("frozen_files", {})
    if set(frozen) != set(AMENDMENT_FROZEN_FILES):
        raise RuntimeError("pre-selection amendment frozen-file set mismatch")
    for relative in AMENDMENT_FROZEN_FILES:
        expected = str(frozen[relative])
        path = ROOT / relative
        if (
            not path.is_file()
            or sha256_file(path) != expected
            or git_file_sha256(amendment_commit, relative) != expected
            or git_file_sha256("HEAD", relative) != expected
        ):
            raise RuntimeError(f"pre-selection amendment file drift: {relative}")

    committed_auxiliary = {
        RUNS_DIR / "evidence_operation_contracts.json": receipt.get(
            "evidence_operation_contracts_sha256"
        ),
        RUNS_DIR / "preselection_legacy_indexes.json": receipt.get(
            "legacy_index_receipt_sha256"
        ),
        RUNS_DIR / "preselection_evidence_seal.json": receipt.get(
            "evidence_seal_sha256"
        ),
    }
    for path, expected_sha256 in committed_auxiliary.items():
        relative = str(path.relative_to(ROOT))
        if (
            not path.is_file()
            or sha256_file(path) != expected_sha256
            or git_file_sha256("HEAD", relative) != expected_sha256
        ):
            raise RuntimeError(f"pre-selection committed auxiliary drift: {relative}")
    operation_receipt = read_json(RUNS_DIR / "evidence_operation_contracts.json")
    legacy_receipt = read_json(RUNS_DIR / "preselection_legacy_indexes.json")
    evidence_seal = read_json(RUNS_DIR / "preselection_evidence_seal.json")
    indices, contracts, summaries = selection_evidence_bundle(
        config, allow_missing_contracts=False
    )
    if (
        receipt.get("classification")
        != "post_score_partial_rollout_pre_official_selection_deviation"
        or receipt.get("passed") is not True
        or evidence_seal.get("passed") is not True
        or evidence_seal.get("selection_artifacts_absent") is not True
        or evidence_seal.get("adapter_artifacts_absent") is not True
        or receipt.get("selection_artifacts_absent_at_seal")
        != evidence_seal.get("selection_artifacts_absent")
        or receipt.get("adapter_artifacts_absent_at_seal")
        != evidence_seal.get("adapter_artifacts_absent")
        or receipt.get("evidence_indexes") != summaries
        or evidence_seal.get("indexes") != summaries
        or operation_receipt.get("contracts") != contracts
        or operation_receipt.get("indexes") != summaries
        or operation_receipt.get("score_backend")
        != indices["independent_scores"].get("backend")
        or evidence_seal.get("evidence_operation_contracts_sha256")
        != receipt.get("evidence_operation_contracts_sha256")
        or evidence_seal.get("legacy_index_receipt_sha256")
        != receipt.get("legacy_index_receipt_sha256")
        or legacy_receipt.get("classification")
        != "retrospective_pre_operation_contract_attestation"
        or set(legacy_receipt.get("indexes", {}))
        != {
            "train_independent",
            "train_independent_scores",
            "train_rollouts_r1",
        }
        or evidence_seal.get("legacy_index_sha256_before_operation_seal")
        != {
            stage: str(entry["sha256"])
            for stage, entry in legacy_receipt.get("indexes", {}).items()
        }
        or receipt.get("deviation_disclosure")
        != {
            "candidate_scores_observed_before_commit": True,
            "original_helper_retained_tasks": 116,
            "official_selection_observed": False,
            "partial_rollout_labels_observed_before_commit": True,
            "partial_rollout_labels_used_for_cost_planning": True,
            "rollout_labels_informed_repair": False,
            "adapter_or_evaluation_outcome_observed": False,
        }
    ):
        raise RuntimeError("pre-selection amendment seal semantics mismatch")
    return indices, contracts, summaries


def load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("default config must be a mapping")
    if value["model"] != {"id": MODEL_ID, "revision": MODEL_REVISION}:
        raise ValueError("one-model invariant mismatch")
    if value.get("preselection_amendment") != {
        "receipt": "runs/preselection_amendment_receipt.json",
        "classification": (
            "post_score_partial_rollout_pre_official_selection_deviation"
        ),
        "required_before": ["select", "train", "merge", "evaluate"],
    }:
        raise ValueError("pre-selection amendment contract mismatch")
    return value


def design_boundary_receipt(config: dict[str, Any]) -> dict[str, Any]:
    boundary = config["design_boundary"]
    commit = str(boundary["commit"])
    head = _run(["git", "rev-parse", "HEAD"]).stdout.strip()
    ancestor = (
        _run(["git", "merge-base", "--is-ancestor", commit, head], check=False).returncode
        == 0
    )
    paths = {
        "readme": "experiments/qwen35_4b_balanced_core_answer_potential_sft/README.md",
        "preregistration": "experiments/qwen35_4b_balanced_core_answer_potential_sft/reports/preregistration.md",
        "design_review": "experiments/qwen35_4b_balanced_core_answer_potential_sft/reports/design_review.md",
    }
    observed = {}
    for name, path in paths.items():
        payload = _run(["git", "show", f"{commit}:{path}"]).stdout.encode("utf-8")
        observed[name] = hashlib.sha256(payload).hexdigest()
    expected = {name: str(boundary[f"{name}_sha256"]) for name in paths}
    passed = ancestor and observed == expected
    receipt = {
        "schema_version": 1,
        "passed": passed,
        "design_commit": commit,
        "current_head": head,
        "design_is_ancestor": ancestor,
        "observed_sha256": observed,
        "expected_sha256": expected,
        "scientific_gpu_work_preceded_by_design_commit": True,
    }
    write_json(RUNS_DIR / "design_boundary_receipt.json", receipt)
    if not passed:
        raise RuntimeError(f"immutable design boundary failed: {receipt}")
    return receipt


def engine_config(
    config: dict[str, Any], *, model_override: Path | None = None
) -> EngineConfig:
    value = config["engine"]
    return EngineConfig(
        max_model_len=int(value["max_model_len"]),
        gpu_memory_utilization=float(value["gpu_memory_utilization"]),
        max_num_seqs=int(value["max_num_seqs"]),
        max_num_batched_tokens=int(value["max_num_batched_tokens"]),
        enable_prefix_caching=bool(value["prefix_caching"]),
        cudagraph_capture_sizes=tuple(
            int(item) for item in value["cudagraph_capture_sizes"]
        ),
        model_override=model_override,
    )


def external_root(config: dict[str, Any]) -> Path:
    return Path(str(config["artifacts"]["external_root"]))


def build_data() -> dict[str, Any]:
    manifest = build_all(DATA_DIR)
    print(f"[data] {manifest['audit']['split_counts']}", flush=True)
    return manifest


def load_split(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.is_file():
        build_data()
    return read_jsonl(path)


def load_core_train(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the prospectively fixed 3 x 3 x 40 balanced train core."""
    split = config["splits"]
    families = set(str(value) for value in split["core_families"])
    levels = set(int(value) for value in split["train_levels"])
    items = [
        item
        for item in load_split("train")
        if str(item["family"]) in families and int(item["level"]) in levels
    ]
    expected_cell = int(split["train_per_family_level"])
    cells = {
        (family, level): sum(
            str(item["family"]) == family and int(item["level"]) == level
            for item in items
        )
        for family in sorted(families)
        for level in sorted(levels)
    }
    if any(count != expected_cell for count in cells.values()):
        raise RuntimeError(f"unbalanced core train cells: {cells}")
    if len(items) != int(split["full_train_tasks"]):
        raise RuntimeError(f"core train count mismatch: {len(items)}")
    return items


def load_evaluation_scope(
    config: dict[str, Any], split: str
) -> list[dict[str, Any]]:
    """Materialize frozen Stage-A subsets or full inherited eval splits."""
    core = set(str(value) for value in config["splits"]["core_families"])
    if split == "core_iid":
        items = [item for item in load_split("iid_eval") if item["family"] in core]
        expected = int(config["splits"]["core_iid_tasks"])
    elif split == "core_hard":
        items = [item for item in load_split("hard_eval") if item["family"] in core]
        expected = int(config["splits"]["core_hard_tasks"])
    elif split == "held_stage_a":
        per_cell = int(config["splits"]["held_stage_a_per_family_level"])
        counts: dict[tuple[str, int], int] = {}
        items = []
        for item in load_split("held_family_eval"):
            key = (str(item["family"]), int(item["level"]))
            ordinal = counts.get(key, 0)
            counts[key] = ordinal + 1
            if ordinal < per_cell:
                items.append(item)
        expected = int(config["splits"]["held_stage_a_tasks"])
    else:
        items = load_split(split)
        expected = len(items)
    if len(items) != expected:
        raise RuntimeError(f"evaluation scope {split} count mismatch: {len(items)} != {expected}")
    return items


def import_inherited_train_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Create a new immutable-provenance index over the parent's 331 shards."""
    inherit = config["inherit"]
    source_path = (
        Path(str(inherit["external_root"]))
        / "pools"
        / str(inherit["stage"])
        / "index.json"
    )
    observed_sha = sha256_file(source_path)
    expected_sha = str(inherit["source_index_sha256"])
    if observed_sha != expected_sha:
        raise RuntimeError(
            f"inherited source index changed: {observed_sha} != {expected_sha}"
        )
    source = read_json(source_path)
    if len(source["shards"]) != int(inherit["expected_tasks"]):
        raise RuntimeError("inherited task count mismatch")
    if sum(int(row["summary"]["rows"]) for row in source["shards"].values()) != int(
        inherit["expected_rows"]
    ):
        raise RuntimeError("inherited row count mismatch")
    if sum(
        int(row["summary"]["sampled_tokens"]) for row in source["shards"].values()
    ) != int(inherit["expected_sampled_tokens"]):
        raise RuntimeError("inherited sampled-token count mismatch")

    core_ids = {str(item["id"]) for item in load_core_train(config)}
    if not set(source["shards"]).issubset(core_ids):
        raise RuntimeError("parent checkpoint contains a task outside the balanced core")
    for task_id, entry in source["shards"].items():
        if not valid_receipt(entry["artifact"]):
            raise RuntimeError(f"invalid inherited shard receipt: {task_id}")

    target_root = external_root(config) / "pools" / "train_independent"
    target_path = target_root / "index.json"
    target = _stage_index(target_path, stage="train_independent", split="train")
    for task_id, entry in source["shards"].items():
        existing = target["shards"].get(task_id)
        if existing and existing["artifact"]["sha256"] != entry["artifact"]["sha256"]:
            raise RuntimeError(f"inherited/generated shard collision: {task_id}")
        target["shards"][task_id] = {
            **entry,
            "inherited": True,
            "inherited_from_experiment": inherit["experiment_id"],
            "inherited_source_index_sha256": observed_sha,
        }
    target["inheritance"] = {
        "experiment_id": inherit["experiment_id"],
        "source_index": str(source_path),
        "source_index_sha256": observed_sha,
        "tasks": len(source["shards"]),
        "rows": int(inherit["expected_rows"]),
        "sampled_tokens": int(inherit["expected_sampled_tokens"]),
    }
    for key in ("logical_counts", "runtime", "engine", "resolved_cudagraph"):
        if key in source and key not in target:
            target[key] = source[key]
    write_json(target_path, target)
    receipt = {
        "schema_version": 1,
        "passed": True,
        **target["inheritance"],
        "target_index": str(target_path),
    }
    write_json(RUNS_DIR / "inherited_pool_receipt.json", receipt)
    return receipt


def _stage_index(path: Path, *, stage: str, split: str) -> dict[str, Any]:
    if path.is_file():
        value = read_json(path)
        if value.get("stage") != stage or value.get("split") != split:
            raise RuntimeError(f"stage-index identity mismatch: {path}")
        return value
    return {
        "schema_version": 1,
        "stage": stage,
        "split": split,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "shards": {},
    }


def _summarize_traces(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = sorted(int(row["n_tokens"]) for row in rows)
    prior_available = sum(row.get("prior_logprob_mean") is not None for row in rows)
    return {
        "rows": len(rows),
        "natural_close": sum(bool(row["natural_close"]) for row in rows),
        "loop": sum(bool(row["loop_flag"]) for row in rows),
        "continued": sum(bool(row.get("continued")) for row in rows),
        "prior_available": prior_available,
        "sampled_tokens": sum(int(row["n_sampled_tokens"]) for row in rows),
        "min_tokens": lengths[0] if lengths else None,
        "median_tokens": lengths[len(lengths) // 2] if lengths else None,
        "max_tokens": lengths[-1] if lengths else None,
    }


def generate_pool(
    config: dict[str, Any], *, split: str, n: int, stage: str
) -> dict[str, Any]:
    design_boundary_receipt(config)
    items = load_core_train(config) if split == "train" else load_split(split)
    sampling = config["sampling"]
    root = external_root(config) / "pools" / stage
    shard_dir = root / "traces"
    index_path = root / "index.json"
    index = _stage_index(index_path, stage=stage, split=split)
    operation_contract = raw_pool_operation_contract(
        config, split=split, n=n, stage=stage
    )
    observed_contract = index.get("operation_contract_sha256")
    if observed_contract not in (None, operation_contract):
        raise RuntimeError("refusing to mix raw-pool operation contracts")
    if observed_contract is None and index["shards"]:
        index["operation_contract_sealed_post_run"] = True
    index["operation_contract_sha256"] = operation_contract
    write_json(index_path, index)
    started = time.perf_counter()
    pending_task_ids = [
        str(item["id"])
        for item in items
        if not (
            (previous := index["shards"].get(str(item["id"])))
            and valid_receipt(previous["artifact"])
        )
    ]
    model_context = (
        AnswerPotentialModel(engine_config(config))
        if pending_task_ids
        else nullcontext(None)
    )
    with model_context as model:
        for item_number, item in enumerate(items, 1):
            task_id = str(item["id"])
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            if model is None:
                raise RuntimeError("internal raw-pool pending-task mismatch")
            print(f"[{stage}] {item_number}/{len(items)} {task_id}", flush=True)
            rows, generation_meta = model.generate_thoughts(
                [item],
                n=n,
                max_tokens=int(sampling["natural_close_allowance"]),
                run_seed=int(sampling["run_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
                logprobs=int(sampling["logprobs"]),
                stage=stage,
                chunk_size=1,
            )
            rows, continuation_meta = model.continue_unclosed_thoughts(
                [item],
                rows,
                max_tokens=int(sampling["nonloop_continuation_tokens"]),
                run_seed=int(sampling["continuation_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            if any(
                row["n_tokens"] and row.get("prior_logprob_mean") is None
                for row in rows
            ):
                raise RuntimeError("sampled trace prior logprob missing")
            artifact = write_jsonl_gz(shard_dir / f"{task_id}.jsonl.gz", rows)
            index["shards"][task_id] = {
                "artifact": artifact,
                "summary": _summarize_traces(rows),
                "generation_elapsed_seconds": generation_meta["elapsed_seconds"],
                "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
            }
            index["logical_counts"] = continuation_meta["logical_counts"]
            index["runtime"] = generation_meta["runtime"]
            index["engine"] = generation_meta["engine"]
            index["resolved_cudagraph"] = generation_meta["resolved_cudagraph"]
            write_json(index_path, index)
    summaries = [entry["summary"] for entry in index["shards"].values()]
    total_rows = sum(int(value["rows"]) for value in summaries)
    summary = {
        "schema_version": 1,
        "stage": stage,
        "split": split,
        "tasks": len(index["shards"]),
        "rows": total_rows,
        "natural_close": sum(int(value["natural_close"]) for value in summaries),
        "loops": sum(int(value["loop"]) for value in summaries),
        "prior_available": sum(int(value["prior_available"]) for value in summaries),
        "sampled_tokens": sum(int(value["sampled_tokens"]) for value in summaries),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / f"{stage}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def ensure_minimum_natural_train_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen mechanical N=16 top-up rule only where needed."""
    root = external_root(config) / "pools" / "train_independent"
    index_path = root / "index.json"
    index = read_json(index_path)
    expected_contract = raw_pool_operation_contract(
        config,
        split="train",
        n=int(config["sampling"]["train_independent_n"]),
        stage="train_independent",
    )
    observed_contract = index.get("operation_contract_sha256")
    if observed_contract not in (None, expected_contract):
        raise RuntimeError("top-up input raw-pool operation contract mismatch")
    if observed_contract is None:
        index["operation_contract_sha256"] = expected_contract
        index["operation_contract_sealed_post_run"] = True
        write_json(index_path, index)
    items = {str(item["id"]): item for item in load_core_train(config)}
    minimum = int(config["selector"]["minimum_natural_per_task"])

    def eligible_count(task_id: str) -> int:
        rows = read_jsonl_gz(Path(index["shards"][task_id]["artifact"]["path"]))
        return sum(row["natural_close"] and not row["loop_flag"] for row in rows)

    counts = {task_id: eligible_count(task_id) for task_id in sorted(index["shards"])}
    deficient = [task_id for task_id, count in counts.items() if count < minimum]
    topup_rows = 0
    if deficient:
        sampling = config["sampling"]
        with AnswerPotentialModel(engine_config(config)) as model:
            for batch in range(1, 5):
                active = [task_id for task_id in deficient if counts[task_id] < minimum]
                if not active:
                    break
                for task_id in active:
                    prior_rows = read_jsonl_gz(
                        Path(index["shards"][task_id]["artifact"]["path"])
                    )
                    new_rows, generation_meta = model.generate_thoughts(
                        [items[task_id]],
                        n=16,
                        max_tokens=int(sampling["natural_close_allowance"]),
                        run_seed=int(sampling["run_seed"]) + 10_000 * batch,
                        temperature=float(sampling["temperature"]),
                        top_p=float(sampling["top_p"]),
                        top_k=int(sampling["top_k"]),
                        logprobs=int(sampling["logprobs"]),
                        stage=f"train_topup_b{batch}",
                        chunk_size=1,
                    )
                    new_rows, continuation_meta = model.continue_unclosed_thoughts(
                        [items[task_id]],
                        new_rows,
                        max_tokens=int(sampling["nonloop_continuation_tokens"]),
                        run_seed=int(sampling["continuation_seed"]) + 10_000 * batch,
                        temperature=float(sampling["temperature"]),
                        top_p=float(sampling["top_p"]),
                        top_k=int(sampling["top_k"]),
                    )
                    combined = [*prior_rows, *new_rows]
                    artifact = write_jsonl_gz(
                        root / "traces" / f"{task_id}.jsonl.gz", combined
                    )
                    index["shards"][task_id] = {
                        "artifact": artifact,
                        "summary": _summarize_traces(combined),
                        "topup_batches": batch,
                        "generation_elapsed_seconds": generation_meta["elapsed_seconds"],
                        "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
                    }
                    counts[task_id] = sum(
                        row["natural_close"] and not row["loop_flag"] for row in combined
                    )
                    topup_rows += len(new_rows)
                    index["logical_counts"] = continuation_meta["logical_counts"]
                    write_json(index_path, index)
    remaining = sorted(task_id for task_id, count in counts.items() if count < minimum)
    result = {
        "schema_version": 1,
        "minimum": minimum,
        "initially_deficient": deficient,
        "topup_rows": topup_rows,
        "remaining_deficient": remaining,
        "counts": counts,
    }
    write_json(RUNS_DIR / "train_natural_minimum.json", result)
    print(
        f"[topup] initially_deficient={len(deficient)} rows={topup_rows} "
        f"remaining={len(remaining)}",
        flush=True,
    )
    return result


def analyze_termination_pilot(config: dict[str, Any]) -> dict[str, Any]:
    """Summarize termination mechanics without reading correctness."""
    index = read_json(
        external_root(config) / "pools" / "termination_pilot" / "index.json"
    )
    rows = [
        row
        for task_id in sorted(index["shards"])
        for row in read_jsonl_gz(
            Path(index["shards"][task_id]["artifact"]["path"])
        )
    ]
    lengths = sorted(int(row["n_tokens"]) for row in rows)
    result = {
        "schema_version": 1,
        "tasks": len(index["shards"]),
        "traces": len(rows),
        "natural_close": sum(bool(row["natural_close"]) for row in rows),
        "natural_close_rate": mean([bool(row["natural_close"]) for row in rows]),
        "exact_periodic_loops": sum(bool(row["loop_flag"]) for row in rows),
        "initial_allowance_contacts": sum(bool(row.get("continued")) for row in rows),
        "still_unclosed_after_continuation": sum(
            bool(row.get("continued")) and not bool(row["natural_close"]) for row in rows
        ),
        "greater_than_512_tokens": sum(int(row["n_tokens"]) > 512 for row in rows),
        "median_tokens": lengths[len(lengths) // 2],
        "p95_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "max_tokens": lengths[-1],
        "sampled_tokens": sum(int(row["n_sampled_tokens"]) for row in rows),
        "correctness_inspected": False,
    }
    write_json(RUNS_DIR / "termination_pilot_analysis.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def score_pool(
    config: dict[str, Any], *, source_stage: str, output_stage: str, split: str
) -> dict[str, Any]:
    design_boundary_receipt(config)
    backend = str(config["scoring"]["backend"])
    if backend != "transformers_bf16_sdpa_single_context":
        raise RuntimeError(f"unsupported amended bulk scoring backend: {backend}")
    items = {
        str(item["id"]): item
        for item in (load_core_train(config) if split == "train" else load_split(split))
    }
    source_root = external_root(config) / "pools" / source_stage
    source_index = read_json(source_root / "index.json")
    output_root = external_root(config) / "pools" / output_stage
    output_index_path = output_root / "index.json"
    output_index = _stage_index(
        output_index_path, stage=output_stage, split=split
    )
    if output_index.get("backend", backend) != backend:
        raise RuntimeError("refusing to mix scoring backends in one stage index")
    output_index["backend"] = backend
    operation_contract = score_operation_contract(config)
    if output_index.get("operation_contract_sha256", operation_contract) != operation_contract:
        raise RuntimeError("refusing to reuse a score index from another operation contract")
    output_index["operation_contract_sha256"] = operation_contract
    output_index["vllm_candidate_instrument"] = {
        "receipt": str(RUNS_DIR / "scorer_parity_joint_32.json"),
        "passed": False,
        "retired_before_bulk_scoring": True,
    }
    write_json(output_index_path, output_index)
    started = time.perf_counter()
    pending_task_ids = [
        task_id
        for task_id in sorted(source_index["shards"])
        if not (
            (previous := output_index["shards"].get(task_id))
            and valid_receipt(previous["artifact"])
            and previous.get("source_artifact_sha256")
            == source_index["shards"][task_id]["artifact"]["sha256"]
        )
    ]
    scorer = HFAnswerPotentialScorer() if pending_task_ids else None
    try:
        for task_number, task_id in enumerate(sorted(source_index["shards"]), 1):
            previous = output_index["shards"].get(task_id)
            source_sha256 = source_index["shards"][task_id]["artifact"]["sha256"]
            if (
                previous
                and valid_receipt(previous["artifact"])
                and previous.get("source_artifact_sha256") == source_sha256
            ):
                continue
            rows = read_jsonl_gz(
                Path(source_index["shards"][task_id]["artifact"]["path"])
            )
            if scorer is None:
                raise RuntimeError("internal scorer pending-task mismatch")
            prompt_ids = scorer.prompt_ids(items[task_id])
            answer_ids = scorer.tokenizer.encode(
                str(items[task_id]["canonical_answer"]), add_special_tokens=False
            )
            max_train_length = int(config["selector"]["max_train_length"])
            eligible = [
                row
                for row in rows
                if row["natural_close"] and not row["loop_flag"]
                and row.get("prior_logprob_mean") is not None
                and math.isfinite(float(row["prior_logprob_mean"]))
                and len(prompt_ids)
                + len(row["token_ids"])
                + len(scorer.boundary_ids)
                + len(answer_ids)
                <= max_train_length
            ]
            print(
                f"[{output_stage}] {task_number}/{len(source_index['shards'])} "
                f"{task_id}: {len(eligible)} eligible",
                flush=True,
            )
            task_started = time.perf_counter()
            scored = [scorer.score_trace(items[task_id], row) for row in eligible]
            artifact = write_jsonl_gz(
                output_root / "scores" / f"{task_id}.jsonl.gz", scored
            )
            output_index["shards"][task_id] = {
                "artifact": artifact,
                "source_artifact_sha256": source_sha256,
                "eligible": len(eligible),
                "excluded": len(rows) - len(eligible),
                "scoring_elapsed_seconds": time.perf_counter() - task_started,
            }
            output_index["readout"] = (
                "transformers_bf16_single_context_full_prefix_target_logits"
            )
            write_json(output_index_path, output_index)
    finally:
        if scorer is not None:
            scorer.close()
    summary = {
        "schema_version": 1,
        "stage": output_stage,
        "split": split,
        "tasks": len(output_index["shards"]),
        "rows": sum(int(entry["artifact"]["rows"]) for entry in output_index["shards"].values()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(output_index_path),
    }
    write_json(RUNS_DIR / f"{output_stage}_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def analyze_calibration(config: dict[str, Any]) -> dict[str, Any]:
    """Report selector informativeness without gating the SFT matrix."""
    raw_index = read_json(
        external_root(config) / "pools" / "calibration_independent" / "index.json"
    )
    score_index = read_json(
        external_root(config) / "pools" / "calibration_scores" / "index.json"
    )
    rollout_index = read_json(
        external_root(config) / "pools" / "calibration_rollouts_r4" / "index.json"
    )
    items = {str(item["id"]): item for item in load_split("calibration")}
    per_task = []
    top_ks = (1, 2, 4, 8)
    metric_names = {
        "answer_gain": "answer_gain_per_answer_token",
        "joint_gain": "joint_gain_per_answer_token",
        "negative_length": "negative_length",
        "trace_prior": "prior_logprob_mean",
    }
    top_curves: dict[str, dict[int, list[float]]] = {
        name: {k: [] for k in top_ks} for name in (*metric_names, "seeded_random")
    }
    for task_id in sorted(items):
        raw = read_jsonl_gz(Path(raw_index["shards"][task_id]["artifact"]["path"]))
        scores = read_jsonl_gz(Path(score_index["shards"][task_id]["artifact"]["path"]))
        rollouts = read_jsonl_gz(Path(rollout_index["shards"][task_id]["artifact"]["path"]))
        raw_by_id = {str(row["trace_id"]): row for row in raw}
        rollout_by_id = {str(row["trace_id"]): row for row in rollouts}
        joined = []
        for score in scores:
            trace_id = str(score["trace_id"])
            if trace_id not in rollout_by_id:
                continue
            raw_row = raw_by_id[trace_id]
            rollout = rollout_by_id[trace_id]
            joined.append(
                {
                    **score,
                    "negative_length": -float(raw_row["n_tokens"]),
                    "success_fraction": float(rollout["success_fraction"]),
                    "any_success": bool(rollout["any_success"]),
                }
            )
        labels = [row["any_success"] for row in joined]
        aucs = {}
        for name, field in metric_names.items():
            finite = [
                row for row in joined if row.get(field) is not None and math.isfinite(float(row[field]))
            ]
            aucs[name] = roc_auc(
                [row["any_success"] for row in finite],
                [float(row[field]) for row in finite],
            )
            ranked = sorted(
                finite,
                key=lambda row: (-float(row[field]), str(row["trace_id"])),
            )
            for k in top_ks:
                if ranked:
                    top_curves[name][k].append(
                        mean([row["success_fraction"] for row in ranked[:k]])
                    )
        seeded = list(joined)
        random_seed = int.from_bytes(
            hashlib.blake2b(task_id.encode("utf-8"), digest_size=8).digest(), "big"
        ) + int(config["sampling"]["control_seed"])
        import random

        random.Random(random_seed).shuffle(seeded)
        for k in top_ks:
            if seeded:
                top_curves["seeded_random"][k].append(
                    mean([row["success_fraction"] for row in seeded[:k]])
                )
        tau = kendall_tau_b(
            [float(row["answer_gain_per_answer_token"]) for row in joined],
            [float(row["joint_gain_per_answer_token"]) for row in joined],
        )
        format_tau = kendall_tau_b(
            [float(row["answer_gain_per_answer_token"]) for row in joined],
            [
                float(row["format_variant_answer_gain_per_answer_token"])
                for row in joined
            ],
        )
        per_task.append(
            {
                "task_id": task_id,
                "family": items[task_id]["family"],
                "level": items[task_id]["level"],
                "raw_traces": len(raw),
                "natural_close_rate": mean([bool(row["natural_close"]) for row in raw]),
                "loop_rate": mean([bool(row["loop_flag"]) for row in raw]),
                "answer_mention_rate": mean(
                    [
                        answer_mention(
                            str(row.get("text", "")),
                            str(items[task_id]["canonical_answer"]),
                        )
                        is not None
                        for row in raw
                    ]
                ),
                "eligible_scored": len(scores),
                "rollout_traces": len(joined),
                "success_rate": optional_mean(
                    [row["success_fraction"] for row in joined]
                ),
                "auroc": aucs,
                "answer_joint_kendall_tau_b": tau,
                "canonical_format_kendall_tau_b": format_tau,
            }
        )
    result = {
        "schema_version": 1,
        "tasks": len(per_task),
        "raw_candidates": sum(row["raw_traces"] for row in per_task),
        "eligible_candidates": sum(row["eligible_scored"] for row in per_task),
        "task_macro_auroc": {
            name: optional_mean(
                [row["auroc"][name] for row in per_task if row["auroc"][name] is not None]
            )
            for name in metric_names
        },
        "top_k_mean_rollout_success": {
            name: {str(k): optional_mean(values) for k, values in curves.items()}
            for name, curves in top_curves.items()
        },
        "task_macro_answer_joint_kendall_tau_b": optional_mean(
            [
                row["answer_joint_kendall_tau_b"]
                for row in per_task
                if row["answer_joint_kendall_tau_b"] is not None
            ]
        ),
        "task_macro_canonical_format_kendall_tau_b": optional_mean(
            [
                row["canonical_format_kendall_tau_b"]
                for row in per_task
                if row["canonical_format_kendall_tau_b"] is not None
            ]
        ),
        "family": {
            family: {
                "tasks": sum(row["family"] == family for row in per_task),
                "mean_success": optional_mean(
                    [
                        row["success_rate"]
                        for row in per_task
                        if row["family"] == family and row["success_rate"] is not None
                    ]
                ),
                "answer_auroc": optional_mean(
                    [
                        row["auroc"]["answer_gain"]
                        for row in per_task
                        if row["family"] == family and row["auroc"]["answer_gain"] is not None
                    ]
                ),
            }
            for family in sorted({row["family"] for row in per_task})
        },
        "per_task": per_task,
        "effectiveness_gate_used": False,
    }
    write_json(RUNS_DIR / "calibration_analysis.json", result)
    print(json.dumps({key: result[key] for key in ("raw_candidates", "eligible_candidates", "task_macro_auroc", "top_k_mean_rollout_success")}, indent=2), flush=True)
    return result


def run_scorer_parity(
    config: dict[str, Any], *, source_stage: str, split: str
) -> dict[str, Any]:
    """Gate answer, joint, empty-baseline, and gain parity on 32 fixed rows."""
    design_boundary_receipt(config)
    source_index = read_json(
        Path(str(config["inherit"]["external_root"]))
        / "pools"
        / source_stage
        / "index.json"
    )
    items = {
        str(item["id"]): item
        for item in (load_core_train(config) if split == "train" else load_split(split))
    }
    traces: list[dict[str, Any]] = []
    for task_id in sorted(source_index["shards"]):
        rows = read_jsonl_gz(
            Path(source_index["shards"][task_id]["artifact"]["path"])
        )
        eligible = [
            row for row in rows if row["natural_close"] and not row["loop_flag"]
        ]
        if eligible:
            traces.append(eligible[0])
        if len(traces) >= int(config["scoring"]["parity_rows"]):
            break
    count = int(config["scoring"]["parity_rows"])
    traces = traces[:count]
    if len(traces) != count:
        raise RuntimeError(f"parity requires {count} eligible rows; found {len(traces)}")
    parity_items = [items[str(trace["task_id"])] for trace in traces]
    unique_items = {str(item["id"]): item for item in parity_items}
    with AnswerPotentialModel(engine_config(config)) as model:
        vllm_rows, vllm_meta = model.score_canonical_joint(
            list(unique_items.values()),
            traces,
            chunk_size=int(config["scoring"]["chunk_size"]),
        )
    scorer = HFAnswerPotentialScorer()
    try:
        hf_rows = [scorer.score_trace(items[str(row["task_id"])], row) for row in traces]
    finally:
        scorer.close()
    vllm_by_id = {str(row["trace_id"]): row for row in vllm_rows}
    comparisons = []
    for row in hf_rows:
        vllm_row = vllm_by_id[str(row["trace_id"])]
        answer_tokens = len(row["answer_token_ids"])
        joint_tokens = len(row["joint_token_logprobs"])
        deltas = {
            "answer_ll_mean_token_delta": (
                float(row["answer_ll_sum"]) - float(vllm_row["answer_ll_sum"])
            )
            / answer_tokens,
            "joint_ll_mean_token_delta": (
                float(row["joint_ll_sum"]) - float(vllm_row["joint_ll_sum"])
            )
            / joint_tokens,
            "empty_answer_ll_mean_token_delta": (
                float(row["empty_answer_ll_sum"])
                - float(vllm_row["empty_answer_ll_sum"])
            )
            / answer_tokens,
            "empty_joint_ll_mean_token_delta": (
                float(row["empty_joint_ll_sum"])
                - float(vllm_row["empty_joint_ll_sum"])
            )
            / joint_tokens,
            "answer_gain_per_answer_token_delta": float(
                row["answer_gain_per_answer_token"]
            )
            - float(vllm_row["answer_gain_per_answer_token"]),
            "joint_gain_per_answer_token_delta": float(
                row["joint_gain_per_answer_token"]
            )
            - float(vllm_row["joint_gain_per_answer_token"]),
        }
        comparisons.append(
            {
                "trace_id": row["trace_id"],
                "task_id": row["task_id"],
                "answer_tokens": answer_tokens,
                "joint_tokens": joint_tokens,
                **deltas,
                "max_abs_registered_delta": max(abs(value) for value in deltas.values()),
            }
        )
    maximum = max(row["max_abs_registered_delta"] for row in comparisons)
    threshold = float(config["scoring"]["max_abs_mean_token_delta"])
    result = {
        "schema_version": 1,
        "source_stage": source_stage,
        "rows": len(comparisons),
        "max_abs_mean_token_delta": maximum,
        "threshold": threshold,
        "passed": maximum <= threshold,
        "vllm_meta": vllm_meta,
        "comparisons": comparisons,
    }
    write_json(RUNS_DIR / "scorer_parity_joint_32.json", result)
    if not result["passed"]:
        raise RuntimeError(f"32-row scorer parity failed: {maximum} > {threshold}")
    print(f"[parity] rows={count} max_delta={maximum:.6f} passed", flush=True)
    return result


def plan_pivots(config: dict[str, Any]) -> dict[str, Any]:
    """Score natural root checkpoints and freeze one pivot per train task."""
    design_boundary_receipt(config)
    raw_index = read_json(
        external_root(config) / "pools" / "train_independent" / "index.json"
    )
    score_index = read_json(
        external_root(config) / "pools" / "train_independent_scores" / "index.json"
    )
    items = {str(item["id"]): item for item in load_split("train")}
    output_root = external_root(config) / "pools" / "train_pivots"
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage="train_pivots", split="train")
    branch = config["branch"]
    started = time.perf_counter()
    scorer = HFAnswerPotentialScorer()
    try:
        for task_number, task_id in enumerate(sorted(raw_index["shards"]), 1):
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            traces = read_jsonl_gz(
                Path(raw_index["shards"][task_id]["artifact"]["path"])
            )
            scores = read_jsonl_gz(
                Path(score_index["shards"][task_id]["artifact"]["path"])
            )
            if not scores:
                index.setdefault("excluded", {})[task_id] = (
                    "no eligible natural independent root after registered top-ups"
                )
                write_json(index_path, index)
                print(
                    f"[train_pivots] {task_number}/{len(raw_index['shards'])} "
                    f"{task_id} excluded: no natural root",
                    flush=True,
                )
                continue
            root_score = max(
                scores,
                key=lambda row: (
                    float(row["joint_gain_per_answer_token"]),
                    str(row["trace_id"]),
                ),
            )
            trace_by_id = {str(row["trace_id"]): row for row in traces}
            root = trace_by_id[str(root_score["trace_id"])]
            indices = natural_checkpoint_indices(
                scorer.tokenizer,
                root["token_ids"],
                max_checkpoints=int(branch["max_checkpoints"]),
            )
            checkpoints = []
            for token_index in indices:
                if token_index == int(root["n_tokens"]):
                    checkpoint_score = root_score
                else:
                    partial = {
                        **root,
                        "trace_id": f"{root['trace_id']}::checkpoint::{token_index}",
                        "token_ids": root["token_ids"][:token_index],
                        "n_tokens": token_index,
                    }
                    checkpoint_score = scorer.score_trace(items[task_id], partial)
                checkpoints.append(
                    {
                        "token_index": token_index,
                        "answer_gain_per_answer_token": checkpoint_score[
                            "answer_gain_per_answer_token"
                        ],
                        "joint_gain_per_answer_token": checkpoint_score[
                            "joint_gain_per_answer_token"
                        ],
                    }
                )
            decision = choose_pivot(
                checkpoints,
                minimum_positive_jump=float(
                    branch["minimum_positive_jump_per_answer_token"]
                ),
                fallback_fraction=float(branch["fallback_fraction"]),
                full_length=int(root["n_tokens"]),
            )
            pivot_index = int(decision["pivot_token_index"])
            plan = {
                "schema_version": 1,
                "task_id": task_id,
                "family": root["family"],
                "level": root["level"],
                "root_trace_id": root["trace_id"],
                "root_tokens": root["n_tokens"],
                "root_joint_gain_per_answer_token": root_score[
                    "joint_gain_per_answer_token"
                ],
                "prefix_token_ids": root["token_ids"][:pivot_index],
                "checkpoints": checkpoints,
                **decision,
            }
            artifact = write_jsonl_gz(
                output_root / "plans" / f"{task_id}.jsonl.gz", [plan]
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "root_source_sha256": raw_index["shards"][task_id]["artifact"][
                    "sha256"
                ],
                "score_source_sha256": score_index["shards"][task_id]["artifact"][
                    "sha256"
                ],
                "pivot_token_index": pivot_index,
            }
            write_json(index_path, index)
            print(
                f"[train_pivots] {task_number}/{len(raw_index['shards'])} "
                f"{task_id} root={root['n_tokens']} pivot={pivot_index}",
                flush=True,
            )
    finally:
        scorer.close()
    summary = {
        "schema_version": 1,
        "stage": "train_pivots",
        "tasks": len(index["shards"]),
        "excluded_tasks": len(index.get("excluded", {})),
        "exclusions": index.get("excluded", {}),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / "train_pivots_summary.json", summary)
    return summary


def generate_branch_pool(config: dict[str, Any]) -> dict[str, Any]:
    """Generate the registered sixteen pivot suffixes for every train task."""
    design_boundary_receipt(config)
    items = load_split("train")
    item_by_id = {str(item["id"]): item for item in items}
    plan_index = read_json(
        external_root(config) / "pools" / "train_pivots" / "index.json"
    )
    output_root = external_root(config) / "pools" / "train_branches"
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage="train_branches", split="train")
    sampling = config["sampling"]
    started = time.perf_counter()
    with AnswerPotentialModel(engine_config(config)) as model:
        for task_number, task_id in enumerate(sorted(plan_index["shards"]), 1):
            previous = index["shards"].get(task_id)
            if previous and valid_receipt(previous["artifact"]):
                continue
            plan = read_jsonl_gz(
                Path(plan_index["shards"][task_id]["artifact"]["path"])
            )[0]
            rows, branch_meta = model.generate_pivot_branches(
                [item_by_id[task_id]],
                [plan],
                n=int(sampling["train_branch_n"]),
                total_allowance=int(sampling["natural_close_allowance"]),
                run_seed=int(sampling["branch_seed"]),
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            rows, continuation_meta = model.continue_unclosed_thoughts(
                [item_by_id[task_id]],
                rows,
                max_tokens=int(sampling["nonloop_continuation_tokens"]),
                run_seed=int(sampling["continuation_seed"]) + 1,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            artifact = write_jsonl_gz(
                output_root / "traces" / f"{task_id}.jsonl.gz", rows
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "summary": _summarize_traces(rows),
                "plan_sha256": plan_index["shards"][task_id]["artifact"]["sha256"],
                "branch_elapsed_seconds": branch_meta["elapsed_seconds"],
                "continuation_elapsed_seconds": continuation_meta["elapsed_seconds"],
            }
            index["logical_counts"] = continuation_meta["logical_counts"]
            index["runtime"] = branch_meta["runtime"]
            index["engine"] = branch_meta["engine"]
            write_json(index_path, index)
            print(
                f"[train_branches] {task_number}/{len(plan_index['shards'])} {task_id}",
                flush=True,
            )
    summaries = [row["summary"] for row in index["shards"].values()]
    summary = {
        "schema_version": 1,
        "stage": "train_branches",
        "tasks": len(index["shards"]),
        "rows": sum(int(row["rows"]) for row in summaries),
        "natural_close": sum(int(row["natural_close"]) for row in summaries),
        "loops": sum(int(row["loop"]) for row in summaries),
        "sampled_tokens": sum(int(row["sampled_tokens"]) for row in summaries),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / "train_branches_summary.json", summary)
    return summary


def rollout_pool(
    config: dict[str, Any],
    *,
    split: str,
    source_stages: list[str],
    output_stage: str,
    r: int,
) -> dict[str, Any]:
    """Generate restartable per-task answer rollouts over one or more pools."""
    design_boundary_receipt(config)
    items = {
        str(item["id"]): item
        for item in (load_core_train(config) if split == "train" else load_split(split))
    }
    source_indices = [
        read_json(external_root(config) / "pools" / stage / "index.json")
        for stage in source_stages
    ]
    task_ids = sorted(set.intersection(*(set(index["shards"]) for index in source_indices)))
    output_root = external_root(config) / "pools" / output_stage
    index_path = output_root / "index.json"
    index = _stage_index(index_path, stage=output_stage, split=split)
    sampling = config["sampling"]
    operation_contract = rollout_operation_contract(
        config, source_stages=source_stages, r=r
    )
    if index.get("operation_contract_sha256", operation_contract) != operation_contract:
        raise RuntimeError("refusing to reuse rollouts from another operation contract")
    index["operation_contract_sha256"] = operation_contract
    write_json(index_path, index)
    started = time.perf_counter()
    pending_task_ids = [
        task_id
        for task_id in task_ids
        if not (
            (previous := index["shards"].get(task_id))
            and valid_receipt(previous["artifact"])
            and previous.get("source_sha256")
            == [
                source_index["shards"][task_id]["artifact"]["sha256"]
                for source_index in source_indices
            ]
        )
    ]
    model_context = (
        AnswerPotentialModel(engine_config(config))
        if pending_task_ids
        else nullcontext(None)
    )
    with model_context as model:
        for task_number, task_id in enumerate(task_ids, 1):
            previous = index["shards"].get(task_id)
            source_sha256 = [
                source_index["shards"][task_id]["artifact"]["sha256"]
                for source_index in source_indices
            ]
            if (
                previous
                and valid_receipt(previous["artifact"])
                and previous.get("source_sha256") == source_sha256
            ):
                continue
            traces: list[dict[str, Any]] = []
            for source_index in source_indices:
                entry = source_index["shards"][task_id]
                traces.extend(read_jsonl_gz(Path(entry["artifact"]["path"])))
            traces = [
                row for row in traces if row["natural_close"] and not row["loop_flag"]
            ]
            if model is None:
                raise RuntimeError("internal rollout pending-task mismatch")
            rows, metadata = model.generate_answer_rollouts(
                [items[task_id]],
                traces,
                r=r,
                max_tokens=int(sampling["answer_max_tokens"]),
                run_seed=int(sampling["continuation_seed"]) + 101 * r,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
            )
            artifact = write_jsonl_gz(
                output_root / "rollouts" / f"{task_id}.jsonl.gz", rows
            )
            index["shards"][task_id] = {
                "artifact": artifact,
                "source_sha256": source_sha256,
                "eligible": len(traces),
                "elapsed_seconds": metadata["elapsed_seconds"],
            }
            index["logical_counts"] = metadata["logical_counts"]
            index["runtime"] = metadata["runtime"]
            write_json(index_path, index)
            print(
                f"[{output_stage}] {task_number}/{len(task_ids)} {task_id}: {len(traces)} traces",
                flush=True,
            )
    summary = {
        "schema_version": 1,
        "stage": output_stage,
        "tasks": len(index["shards"]),
        "rows": sum(int(row["artifact"]["rows"]) for row in index["shards"].values()),
        "rollouts": sum(int(row["artifact"]["rows"]) * r for row in index["shards"].values()),
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "external_index": str(index_path),
    }
    write_json(RUNS_DIR / f"{output_stage}_summary.json", summary)
    return summary


def build_sft_datasets(config: dict[str, Any]) -> dict[str, Any]:
    """Join independent evidence and materialize all six balanced-core arms."""
    indices, expected_contracts, evidence_summaries = (
        validate_preselection_amendment_receipt(config)
    )
    from transformers import AutoTokenizer

    design_boundary_receipt(config)
    root = external_root(config)
    items = {str(item["id"]): item for item in load_core_train(config)}
    selector_config = config["selector"]
    seed = int(config["sampling"]["control_seed"])
    selections: dict[str, dict[str, list[dict[str, Any]]]] = {}
    details_root = root / "selection" / "tasks"
    available_tasks = sorted(
        set.intersection(*(set(index["shards"]) for index in indices.values()))
    )
    expected_tasks = int(config["splits"]["full_train_tasks"])
    if len(available_tasks) != expected_tasks or set(available_tasks) != set(items):
        raise RuntimeError(
            "selection inputs do not cover the frozen balanced task set: "
            f"available={len(available_tasks)} expected={expected_tasks}"
        )
    for task_number, task_id in enumerate(available_tasks, 1):
        raw_entry = indices["independent"]["shards"][task_id]
        score_entry = indices["independent_scores"]["shards"][task_id]
        rollout_entry = indices["rollouts"]["shards"][task_id]
        if not all(
            valid_receipt(entry["artifact"])
            for entry in (raw_entry, score_entry, rollout_entry)
        ):
            raise RuntimeError(f"invalid selection input artifact: {task_id}")
        raw_sha256 = raw_entry["artifact"]["sha256"]
        if score_entry.get("source_artifact_sha256") != raw_sha256:
            raise RuntimeError(f"score/raw provenance mismatch: {task_id}")
        if rollout_entry.get("source_sha256") != [raw_sha256]:
            raise RuntimeError(f"rollout/raw provenance mismatch: {task_id}")
        traces: list[dict[str, Any]] = []
        scores: list[dict[str, Any]] = []
        traces.extend(
            read_jsonl_gz(
                Path(indices["independent"]["shards"][task_id]["artifact"]["path"])
            )
        )
        scores.extend(
            read_jsonl_gz(
                Path(
                    indices["independent_scores"]["shards"][task_id]["artifact"][
                        "path"
                    ]
                )
            )
        )
        rollouts = read_jsonl_gz(
            Path(indices["rollouts"]["shards"][task_id]["artifact"]["path"])
        )
        if any(str(row.get("task_id")) != task_id for row in traces):
            raise RuntimeError(f"raw shard contains another task ID: {task_id}")
        if any(str(row.get("task_id")) != task_id for row in scores):
            raise RuntimeError(f"score shard contains another task ID: {task_id}")
        if any(str(row.get("task_id")) != task_id for row in rollouts):
            raise RuntimeError(f"rollout shard contains another task ID: {task_id}")
        trace_ids = [str(row["trace_id"]) for row in traces]
        score_ids = [str(row["trace_id"]) for row in scores]
        rollout_ids = [str(row["trace_id"]) for row in rollouts]
        if len(trace_ids) != len(set(trace_ids)):
            raise RuntimeError(f"duplicate raw trace IDs: {task_id}")
        if len(score_ids) != len(set(score_ids)) or len(rollout_ids) != len(
            set(rollout_ids)
        ):
            raise RuntimeError(f"duplicate evidence trace IDs: {task_id}")
        if set(score_ids) != set(rollout_ids):
            raise RuntimeError(f"score/rollout trace join mismatch: {task_id}")
        if not set(score_ids).issubset(set(trace_ids)):
            raise RuntimeError(f"evidence contains unknown trace IDs: {task_id}")
        if (
            len(score_ids) != int(score_entry["eligible"])
            or len(rollout_ids) != int(rollout_entry["eligible"])
        ):
            raise RuntimeError(f"evidence eligibility count mismatch: {task_id}")
        selected = select_task(
            traces,
            scores,
            rollouts,
            selector_config=selector_config,
            seed=seed,
        )
        selections[task_id] = selected
        write_jsonl_gz(
            details_root / f"{task_id}.jsonl.gz",
            [
                {
                    "task_id": task_id,
                    "arm": arm,
                    "trace_ids": [row["trace_id"] for row in rows],
                    "trace_tokens": [row["n_tokens"] for row in rows],
                    "source_kinds": [row.get("source_kind") for row in rows],
                    "selection_modes": [row.get("selection_mode") for row in rows],
                    "selection_gaps_from_best": [
                        row.get("selection_gap_from_best") for row in rows
                    ],
                }
                for arm, rows in selected.items()
                if arm != "eligible"
            ],
        )
        if task_number % 50 == 0:
            print(f"[selection] {task_number}/{len(available_tasks)}", flush=True)

    required = int(selector_config["minimum_natural_per_task"])
    core_tasks = [
        task_id
        for task_id, selected in selections.items()
        if all(
            len(selected[arm]) >= required
            for arm in (
                "answer_potential",
                "joint_potential",
                "random_natural",
                "shortest_natural",
            )
        )
    ]
    deficient = sorted(set(items) - set(core_tasks))
    if deficient or len(core_tasks) != expected_tasks:
        raise RuntimeError(
            "balanced selection lost required tasks: "
            f"core={len(core_tasks)} deficient={deficient[:10]}"
        )
    cell_task_counts: dict[str, int] = {}
    for task_id in core_tasks:
        item = items[task_id]
        cell = f"{item['family']}-L{int(item['level'])}"
        cell_task_counts[cell] = cell_task_counts.get(cell, 0) + 1
    expected_per_cell = int(config["splits"]["train_per_family_level"])
    expected_cells = {
        f"{family}-L{int(level)}"
        for family in config["splits"]["core_families"]
        for level in config["splits"]["train_levels"]
    }
    if set(cell_task_counts) != expected_cells or any(
        count != expected_per_cell for count in cell_task_counts.values()
    ):
        raise RuntimeError(
            "selection broke frozen family/level balance: "
            f"counts={cell_task_counts} expected_per_cell={expected_per_cell}"
        )

    chosen: dict[str, list[dict[str, Any] | None]] = {
        "answer_potential": [],
        "joint_potential": [],
        "random_natural": [],
        "success_rft": [],
        "shortest_natural": [],
    }
    row_tasks: dict[str, list[str]] = {key: [] for key in chosen}
    for task_id in core_tasks:
        selected = selections[task_id]
        for arm in (
            "answer_potential",
            "joint_potential",
            "random_natural",
            "shortest_natural",
        ):
            rows = selected[arm][:required]
            chosen[arm].extend(rows)
            row_tasks[arm].extend([task_id] * len(rows))
        success = selected["success_rft"][:required]
        chosen["success_rft"].extend(success)
        row_tasks["success_rft"].extend([task_id] * len(success))

    answer_rows = [row for row in chosen["answer_potential"] if row is not None]
    shuffle_rows = deranged_sources(answer_rows)
    chosen["potential_shuffle"] = shuffle_rows
    row_tasks["potential_shuffle"] = [str(row["task_id"]) for row in shuffle_rows]

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    max_length = int(config["sft"]["max_length"])
    encoded: dict[str, list[dict[str, Any]]] = {}
    for arm in config["sft"]["arms"]:
        rows = []
        for ordinal, (task_id, trace) in enumerate(
            zip(row_tasks[arm], chosen[arm]), 1
        ):
            rows.append(
                sft_record(
                    arm=arm,
                    item=items[task_id],
                    trace=trace,
                    tokenizer=tokenizer,
                    ordinal=ordinal,
                    max_length=max_length,
                )
            )
        encoded[arm] = rows
    target_rows = len(encoded["answer_potential"])
    expected_rows = expected_tasks * required
    if target_rows != expected_rows:
        raise RuntimeError(
            f"balanced row target mismatch: {target_rows} != {expected_rows}"
        )
    success_unique_rows = len(encoded["success_rft"])
    success_unique_tasks = len({row["task_id"] for row in encoded["success_rft"]})
    encoded["success_rft"] = oversample_to(
        encoded["success_rft"], target_rows, seed=seed
    )
    for arm, rows in encoded.items():
        if len(rows) != target_rows:
            raise RuntimeError(f"row-matching failure for {arm}: {len(rows)} != {target_rows}")

    dataset_root = root / "sft"
    receipts = {
        arm: write_jsonl_gz(dataset_root / f"{arm}.jsonl.gz", rows)
        for arm, rows in encoded.items()
    }
    arm_summary = {}
    for arm, rows in encoded.items():
        selection_modes: dict[str, int] = {}
        for row in rows:
            mode = str(row.get("selection_mode") or "not_applicable")
            selection_modes[mode] = selection_modes.get(mode, 0) + 1
        gap_values_by_mode: dict[str, list[float]] = {}
        for row in rows:
            if row.get("selection_gap_from_best") is None:
                continue
            gap_values_by_mode.setdefault(str(row["selection_mode"]), []).append(
                float(row["selection_gap_from_best"])
            )
        selection_gap_summary = {}
        for mode, values in sorted(gap_values_by_mode.items()):
            ordered = sorted(values)
            selection_gap_summary[mode] = {
                "rows": len(ordered),
                "mean": sum(ordered) / len(ordered),
                "median": ordered[len(ordered) // 2],
                "p90": ordered[min(len(ordered) - 1, int(0.90 * len(ordered)))],
                "max": ordered[-1],
            }
        shuffle_gaps = [
            abs(int(row["trace_tokens"]) - int(row["shuffle_target_trace_tokens"]))
            for row in rows
            if row.get("shuffle_target_trace_tokens") is not None
        ]
        arm_summary[arm] = {
            "rows": len(rows),
            "unique_record_ids": len({row["record_id"].split("::repeat", 1)[0] for row in rows}),
            "unique_tasks": len({row["task_id"] for row in rows}),
            "trace_tokens": sum(int(row["trace_tokens"]) for row in rows),
            "forward_tokens": sum(int(row["total_tokens"]) for row in rows),
            "supervised_weighted_tokens": sum(
                len(row["prompt_token_ids"])
                * float(config["sft"]["weight_prompt"])
                + len(row["trace_token_ids"])
                * float(config["sft"]["weight_think"])
                + (
                    len(row["answer_boundary_token_ids"])
                    + len(row["answer_token_ids"])
                    + 1
                )
                * float(config["sft"]["weight_close_answer"])
                for row in rows
            ),
            "branch_rows": sum(row.get("source_kind") == "pivot_branch" for row in rows),
            "selection_modes": selection_modes,
            "selection_gap_from_best_by_mode": selection_gap_summary,
            "shuffle_length_mismatch": (
                {
                    "total_tokens": sum(shuffle_gaps),
                    "mean_tokens": sum(shuffle_gaps) / len(shuffle_gaps),
                    "max_tokens": max(shuffle_gaps),
                }
                if shuffle_gaps
                else None
            ),
            "artifact": receipts[arm],
        }
    raw_trace_ids = {
        arm: {
            str(row.get("source_trace_id"))
            for row in rows
            if row.get("source_trace_id") is not None
        }
        for arm, rows in encoded.items()
    }
    pairwise_trace_overlap = {
        f"{left}__{right}": len(raw_trace_ids[left] & raw_trace_ids[right])
        for index, left in enumerate(config["sft"]["arms"])
        for right in config["sft"]["arms"][index + 1 :]
    }
    summary = {
        "schema_version": 1,
        "preselection_amendment": {
            "receipt": str(AMENDMENT_RECEIPT_PATH),
            "receipt_sha256": sha256_file(AMENDMENT_RECEIPT_PATH),
            "amendment_commit": read_json(AMENDMENT_RECEIPT_PATH)[
                "amendment_commit"
            ],
            "evidence_indexes": evidence_summaries,
            "operation_contracts": expected_contracts,
            "selector_sha256": sha256_file(EXP / "src" / "selector.py"),
            "run_sha256": sha256_file(Path(__file__)),
            "config_sha256": sha256_file(CONFIG_PATH),
        },
        "core_tasks": len(core_tasks),
        "cell_task_counts": cell_task_counts,
        "deficient_tasks": deficient,
        "target_rows_per_arm": target_rows,
        "success_unique_rows_before_oversampling": success_unique_rows,
        "success_unique_tasks": success_unique_tasks,
        "pairwise_source_trace_overlap": pairwise_trace_overlap,
        "arms": arm_summary,
    }
    write_json(EXP / "data" / "sft_manifest.json", summary)
    write_json(RUNS_DIR / "selection_summary.json", summary)
    print(
        f"[selection] core_tasks={len(core_tasks)} rows/arm={target_rows} "
        f"success_tasks={success_unique_tasks}",
        flush=True,
    )
    return summary


def _score_evaluation_rows(
    items: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from collections import Counter

    from gym import base
    from gym.families import load as load_train_family
    from gym.heldout_families import load as load_held_family

    scored_rows = []
    all_thought_lengths: list[int] = []
    all_completion_lengths: list[int] = []
    for row in rows:
        task_id = str(row["id"])
        item = items[task_id]
        module = (
            load_held_family(str(item["family"]))
            if item["family"] in {"brinework", "spindle"}
            else load_train_family(str(item["family"]))
        )
        outcomes = []
        for output in row["outputs"]:
            answer_value = base.extract_answer(str(output["text"]))
            score = float(module.score_atom(item, str(output["text"])))
            outcomes.append(
                {
                    **output,
                    "answer_value": answer_value,
                    "score": score,
                    "correct": score == 1.0,
                    "parsed": answer_value is not None,
                }
            )
            all_thought_lengths.append(int(output["n_thinking_tokens"]))
            all_completion_lengths.append(int(output["n_sampled_tokens"]))
        parsed = [outcome["answer_value"] for outcome in outcomes if outcome["answer_value"] is not None]
        majority_value = Counter(parsed).most_common(1)[0][0] if parsed else None
        majority_correct = any(
            outcome["answer_value"] == majority_value and outcome["correct"]
            for outcome in outcomes
        ) if majority_value is not None else False
        scored_rows.append(
            {
                **{key: value for key, value in row.items() if key != "outputs"},
                "family": item["family"],
                "level": item["level"],
                "outputs": outcomes,
                "any_correct": any(outcome["correct"] for outcome in outcomes),
                "majority_answer": majority_value,
                "majority_correct": majority_correct,
                "unique_answers": len(set(parsed)),
            }
        )
    tasks = len(scored_rows)
    outputs = [outcome for row in scored_rows for outcome in row["outputs"]]
    lengths = sorted(all_thought_lengths)
    summary = {
        "tasks": tasks,
        "samples": len(outputs),
        "sample_accuracy": sum(outcome["correct"] for outcome in outputs) / len(outputs),
        "parse_rate": sum(outcome["parsed"] for outcome in outputs) / len(outputs),
        "parse_conditional_accuracy": (
            sum(outcome["correct"] for outcome in outputs if outcome["parsed"])
            / sum(outcome["parsed"] for outcome in outputs)
            if any(outcome["parsed"] for outcome in outputs)
            else None
        ),
        "natural_close_rate": sum(outcome["thinking_closed"] for outcome in outputs) / len(outputs),
        "pass_at_n": sum(row["any_correct"] for row in scored_rows) / tasks,
        "majority_at_n": sum(row["majority_correct"] for row in scored_rows) / tasks,
        "mean_unique_answers": sum(row["unique_answers"] for row in scored_rows) / tasks,
        "mean_thinking_tokens": sum(all_thought_lengths) / len(all_thought_lengths),
        "median_thinking_tokens": lengths[len(lengths) // 2],
        "p95_thinking_tokens": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))],
        "sampled_tokens": sum(all_completion_lengths),
        "logical_prompt_tokens": sum(int(row["n_prompt_tokens"]) * len(row["outputs"]) for row in scored_rows),
    }
    return scored_rows, summary


def evaluate_matrix(
    config: dict[str, Any],
    *,
    mode: str,
    phase: str = "stage_a",
    arms_override: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the mandatory core or conditional expanded matrix."""
    if mode not in {"greedy", "sample8"}:
        raise ValueError("mode must be greedy or sample8")
    if phase not in {"stage_a", "stage_b"}:
        raise ValueError("phase must be stage_a or stage_b")
    validate_preselection_amendment_receipt(config)
    design_boundary_receipt(config)
    artifact_root = external_root(config)
    arms = arms_override or ["base", *config["sft"]["arms"]]
    if phase == "stage_a":
        splits = tuple(config["evaluation"]["stage_a_splits"])
    elif mode == "greedy":
        splits = tuple(config["evaluation"]["stage_b_full_splits"])
    else:
        splits = ("core_iid",)
    eval_root = artifact_root / "evaluation" / "seed42" / phase / mode
    index_path = eval_root / "index.json"
    index = read_json(index_path) if index_path.is_file() else {
        "schema_version": 1,
        "mode": mode,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "arms": {},
    }
    evaluation = config["evaluation"]
    if any(arm != "base" for arm in arms):
        probe = behavioral_difference_probe(config)
        if not probe.get("passed"):
            raise RuntimeError("fresh behavioral deployment probe did not pass")
    for arm in arms:
        model_override = (
            None
            if arm == "base"
            else artifact_root / "merged" / "seed42" / arm
        )
        model_fingerprint = (
            base_model_fingerprint()
            if arm == "base"
            else validated_merged_fingerprint(config, arm)
        )
        prepared: dict[str, dict[str, Any]] = {}
        for split in splits:
            split_items = load_evaluation_scope(config, split)
            records = [
                {
                    "id": item["id"],
                    "messages": [{"role": "user", "content": item["prompt"]}],
                    "meta": {
                        "family": item["family"],
                        "level": item["level"],
                        "split": split,
                    },
                }
                for item in split_items
            ]
            sampling = SamplingConfig(
                thinking="natural",
                n=1 if mode == "greedy" else int(evaluation["sampled_k"]),
                max_tokens=int(evaluation["natural_max_tokens"]),
                greedy=mode == "greedy",
                temperature=(
                    None
                    if mode == "greedy"
                    else float(evaluation["sampling_temperature"])
                ),
                top_p=(
                    None if mode == "greedy" else float(evaluation["sampling_top_p"])
                ),
                top_k=(
                    None if mode == "greedy" else int(evaluation["sampling_top_k"])
                ),
                run_seed=int(
                    evaluation["greedy_seed" if mode == "greedy" else "sample_seed"]
                ),
            )
            contract = canonical_sha256(
                {
                    "phase": phase,
                    "mode": mode,
                    "arm": arm,
                    "split": split,
                    "model_fingerprint": model_fingerprint,
                    "scope": [
                        {
                            "id": item["id"],
                            "prompt": item["prompt"],
                            "canonical_answer": item["canonical_answer"],
                            "family": item["family"],
                            "level": item["level"],
                        }
                        for item in split_items
                    ],
                    "sampling": sampling_contract(sampling),
                    "engine": config["engine"],
                    "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
                    "environment_lock_sha256": sha256_file(
                        ROOT / "requirements-vllm.lock.txt"
                    ),
                    "analysis_script_sha256": sha256_file(Path(__file__)),
                }
            )
            prepared[split] = {
                "items": split_items,
                "records": records,
                "sampling": sampling,
                "contract": contract,
            }
        pending = [
            split
            for split in splits
            if not (
                split in index["arms"].get(arm, {})
                and valid_receipt(index["arms"][arm][split]["artifact"])
                and index["arms"][arm][split].get("contract_sha256")
                == prepared[split]["contract"]
                and index["arms"][arm][split].get("model_fingerprint")
                == model_fingerprint
            )
        ]
        if not pending:
            continue
        if model_override is not None and not (model_override / "config.json").is_file():
            raise RuntimeError(f"missing merged checkpoint for {arm}: {model_override}")
        print(f"[eval:{mode}] loading {arm}", flush=True)
        with VLLMRunner(
            engine_config(config, model_override=model_override)
        ) as runner:
            for split in pending:
                split_items = prepared[split]["items"]
                item_by_id = {str(item["id"]): item for item in split_items}
                records = prepared[split]["records"]
                sampling = prepared[split]["sampling"]
                raw_rows, metadata = runner.generate(records, sampling)
                scored, summary = _score_evaluation_rows(item_by_id, raw_rows)
                artifact = write_jsonl_gz(
                    eval_root / arm / f"{split}.jsonl.gz", scored
                )
                index.setdefault("arms", {}).setdefault(arm, {})[split] = {
                    "artifact": artifact,
                    "summary": summary,
                    "generation_metadata": metadata,
                    "model_override": None if model_override is None else str(model_override),
                    "model_fingerprint": model_fingerprint,
                    "contract_sha256": prepared[split]["contract"],
                }
                write_json(index_path, index)
                print(
                    f"[eval:{mode}] {arm}/{split} "
                    f"accuracy={summary['sample_accuracy']:.4f} pass={summary['pass_at_n']:.4f}",
                    flush=True,
                )
    compact = {
        arm: {
            split: entry["summary"]
            for split, entry in split_entries.items()
        }
        for arm, split_entries in index["arms"].items()
    }
    write_json(RUNS_DIR / f"evaluation_{phase}_{mode}_summary.json", compact)
    return compact


def validate_selected_dataset_manifest(
    config: dict[str, Any], *, arm: str, dataset: Path
) -> dict[str, Any]:
    """Bind training to the committed, amendment-sealed selection output."""
    manifest_path = EXP / "data" / "sft_manifest.json"
    summary_path = RUNS_DIR / "selection_summary.json"
    if not manifest_path.is_file() or not summary_path.is_file():
        raise RuntimeError("selected-dataset manifest is missing")
    manifest_sha256 = sha256_file(manifest_path)
    if manifest_sha256 != sha256_file(summary_path):
        raise RuntimeError("tracked selection manifest and summary differ")
    for path in (manifest_path, summary_path):
        relative = str(path.relative_to(ROOT))
        if git_file_sha256("HEAD", relative) != sha256_file(path):
            raise RuntimeError(f"selected-dataset manifest is not committed: {relative}")
    manifest = read_json(manifest_path)
    if arm not in manifest.get("arms", {}):
        raise RuntimeError(f"selected-dataset manifest lacks arm: {arm}")
    manifest_arm = manifest["arms"][arm]
    artifact = manifest_arm.get("artifact", {})
    if (
        artifact.get("path") != str(dataset)
        or not valid_receipt(artifact)
        or int(artifact.get("rows", -1)) != int(manifest_arm.get("rows", -2))
    ):
        raise RuntimeError(f"selected dataset artifact mismatch for {arm}")

    amendment = read_json(AMENDMENT_RECEIPT_PATH)
    selected_from = manifest.get("preselection_amendment", {})
    if (
        selected_from.get("receipt") != str(AMENDMENT_RECEIPT_PATH)
        or selected_from.get("receipt_sha256")
        != sha256_file(AMENDMENT_RECEIPT_PATH)
        or selected_from.get("amendment_commit")
        != amendment.get("amendment_commit")
        or selected_from.get("evidence_indexes")
        != amendment.get("evidence_indexes")
        or selected_from.get("operation_contracts")
        != selection_evidence_contracts(config)
        or selected_from.get("selector_sha256")
        != sha256_file(EXP / "src" / "selector.py")
        or selected_from.get("run_sha256") != sha256_file(Path(__file__))
        or selected_from.get("config_sha256") != sha256_file(CONFIG_PATH)
    ):
        raise RuntimeError("selected-dataset amendment provenance mismatch")
    return manifest_arm


def validate_training_receipt(
    config: dict[str, Any],
    *,
    arm: str,
    seed: int,
    dataset: Path,
    output: Path,
    receipt: dict[str, Any],
) -> None:
    manifest_arm = validate_selected_dataset_manifest(
        config, arm=arm, dataset=dataset
    )
    expected_rows = int(manifest_arm["rows"])
    expected_forward_tokens = int(manifest_arm["forward_tokens"])
    expected_supervised_tokens = float(manifest_arm["supervised_weighted_tokens"])
    expected_epochs = float(config["sft"]["epochs"])
    expected_steps = math.ceil(
        expected_rows
        / (
            int(config["sft"]["batch_size"])
            * int(config["sft"]["gradient_accumulation"])
        )
    ) * int(expected_epochs)
    seed_contract = receipt.get("seed_contract", {})
    expected_weights = {
        "prompt": float(config["sft"]["weight_prompt"]),
        "think": float(config["sft"]["weight_think"]),
        "close_answer": float(config["sft"]["weight_close_answer"]),
    }
    expected_training_contract = {
        "script_sha256": sha256_file(EXP / "scripts" / "train_think.py"),
        "rank": int(config["sft"]["rank"]),
        "alpha": int(config["sft"]["alpha"]),
        "dropout": float(config["sft"]["dropout"]),
        "learning_rate": float(config["sft"]["learning_rate"]),
        "batch_size": int(config["sft"]["batch_size"]),
        "gradient_accumulation": int(config["sft"]["gradient_accumulation"]),
        "max_length": int(config["sft"]["max_length"]),
        "optimizer": "paged_adamw_8bit",
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
    }
    checks = {
        "arm": receipt.get("arm") == arm,
        "seed": int(receipt.get("seed", -1)) == seed,
        "non_smoke": receipt.get("smoke") is False,
        "model": receipt.get("model") == MODEL_ID,
        "revision": receipt.get("revision") == MODEL_REVISION,
        "dataset_sha256": receipt.get("dataset_sha256") == sha256_file(dataset),
        "rows": int(receipt.get("rows", -1)) == expected_rows,
        "epochs": float(receipt.get("epochs", -1)) == expected_epochs,
        "completed_epochs": float(receipt.get("completed_epochs", -1)) == expected_epochs,
        "optimizer_steps": int(receipt.get("optimizer_steps", -1)) == expected_steps,
        "skipped_rows": int(receipt.get("skipped_rows", -1)) == 0,
        "loss_weights": receipt.get("loss_weights") == expected_weights,
        "training_contract": receipt.get("training_contract")
        == expected_training_contract,
        "global_seed": seed_contract.get("global_seed_before_model") is True,
        "adapter_seed": seed_contract.get("lora_seed_reset_before_adapter_init") is True,
        "trainer_seed": int(seed_contract.get("trainer_seed", -1)) == seed,
        "data_seed": int(seed_contract.get("data_seed", -1)) == seed,
        "initial_hash": len(str(receipt.get("adapter_initial_state", {}).get("sha256", ""))) == 64,
        "initial_tensors": int(receipt.get("adapter_initial_state", {}).get("tensors", 0)) > 0,
        "adapter_manifest_tensors": int(receipt.get("adapter_tensor_manifest", {}).get("tensors", -1)) == 256,
        "adapter_manifest_hash": len(
            str(receipt.get("adapter_tensor_manifest", {}).get("sha256", ""))
        )
        == 64,
        "training_lock": receipt.get("training_lock", {}).get("sha256")
        == sha256_file(ROOT / "requirements-training.lock.txt"),
        "selection_manifest": receipt.get("selection_manifest_sha256")
        == sha256_file(EXP / "data" / "sft_manifest.json"),
        "amendment_receipt": receipt.get(
            "preselection_amendment_receipt_sha256"
        )
        == sha256_file(AMENDMENT_RECEIPT_PATH),
        "dataset_forward_tokens": int(
            receipt.get("dataset_forward_tokens_one_pass", -1)
        )
        == expected_forward_tokens,
        "actual_examples_seen": int(receipt.get("actual_examples_seen", -1))
        == expected_rows * int(expected_epochs),
        "actual_forward_tokens_seen": int(
            receipt.get("actual_forward_tokens_seen", -1)
        )
        == expected_forward_tokens * int(expected_epochs),
        "dataset_supervised_tokens": math.isclose(
            float(receipt.get("dataset_supervised_weighted_tokens_one_pass", -1)),
            expected_supervised_tokens,
            rel_tol=0,
            abs_tol=1e-6,
        ),
        "actual_supervised_tokens_seen": math.isclose(
            float(receipt.get("actual_supervised_weighted_tokens_seen", -1)),
            expected_supervised_tokens * int(expected_epochs),
            rel_tol=0,
            abs_tol=1e-6,
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(
            f"stale or mismatched training receipt for {arm}: {', '.join(failed)}"
        )
    artifacts = receipt.get("artifacts", {})
    if "adapter_model.safetensors" not in artifacts:
        raise RuntimeError(f"training receipt lacks adapter weights for {arm}")
    for name, artifact in artifacts.items():
        path = output / name
        if not path.is_file() or sha256_file(path) != artifact.get("sha256"):
            raise RuntimeError(f"invalid cached training artifact for {arm}: {path}")


def train_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Train every mandatory seed-42 arm, restarting at completed receipts."""
    validate_preselection_amendment_receipt(config)
    design_boundary_receipt(config)
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError(
            "missing separate Transformers training environment at .venv; "
            "create it per docs/compute_environment.md"
        )
    artifact_root = external_root(config)
    results = {}
    for arm in config["sft"]["arms"]:
        dataset = artifact_root / "sft" / f"{arm}.jsonl.gz"
        output = artifact_root / "adapters" / "seed42" / arm
        receipt_path = output / "training_receipt.json"
        validate_selected_dataset_manifest(config, arm=arm, dataset=dataset)
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            validate_training_receipt(
                config,
                arm=arm,
                seed=int(config["sft"]["screen_seed"]),
                dataset=dataset,
                output=output,
                receipt=receipt,
            )
            results[arm] = receipt
            continue
        print(f"[train-matrix] starting {arm}", flush=True)
        subprocess.run(
            [
                str(python),
                str(EXP / "scripts" / "train_think.py"),
                "--arm",
                arm,
                "--dataset",
                str(dataset),
                "--out",
                str(output),
                "--seed",
                str(config["sft"]["screen_seed"]),
            ],
            cwd=ROOT,
            check=True,
        )
        receipt = read_json(receipt_path)
        validate_training_receipt(
            config,
            arm=arm,
            seed=int(config["sft"]["screen_seed"]),
            dataset=dataset,
            output=output,
            receipt=receipt,
        )
        results[arm] = receipt
    initial_hashes = {
        str(receipt["adapter_initial_state"]["sha256"])
        for receipt in results.values()
    }
    if len(initial_hashes) != 1:
        raise RuntimeError(
            f"seed-42 arms did not share one LoRA initialization: {sorted(initial_hashes)}"
        )
    write_json(RUNS_DIR / "training_matrix_summary.json", results)
    return results


def merge_matrix(config: dict[str, Any]) -> dict[str, Any]:
    """Merge all adapters into deployable composite checkpoints."""
    validate_preselection_amendment_receipt(config)
    design_boundary_receipt(config)
    python = ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise RuntimeError("missing .venv for adapter merge")
    artifact_root = external_root(config)
    receipts = {}
    for arm in config["sft"]["arms"]:
        adapter = artifact_root / "adapters" / "seed42" / arm
        output = artifact_root / "merged" / "seed42" / arm
        receipt_path = output / "merge_receipt.json"
        adapter_receipt = read_json(adapter / "training_receipt.json")
        validate_training_receipt(
            config,
            arm=arm,
            seed=int(config["sft"]["screen_seed"]),
            dataset=artifact_root / "sft" / f"{arm}.jsonl.gz",
            output=adapter,
            receipt=adapter_receipt,
        )
        adapter_hash = adapter_receipt["artifacts"]["adapter_model.safetensors"]["sha256"]
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            current_fingerprint = checkpoint_fingerprint(output)
            if (
                receipt.get("adapter_sha256") == adapter_hash
                and receipt.get("merged_checkpoint_fingerprint")
                == current_fingerprint
            ):
                receipts[arm] = receipt
                continue
            raise RuntimeError(f"stale merged checkpoint receipt: {receipt_path}")
        print(f"[merge-matrix] starting {arm}", flush=True)
        subprocess.run(
            [
                str(python),
                str(EXP / "scripts" / "merge_adapter.py"),
                "--adapter",
                str(adapter),
                "--out",
                str(output),
            ],
            cwd=ROOT,
            check=True,
        )
        application_path = output / "merge_application_receipt.json"
        application = read_json(application_path)
        if (
            application.get("adapter_sha256") != adapter_hash
            or application.get("adapter_tensor_manifest")
            != adapter_receipt.get("adapter_tensor_manifest")
            or int(application.get("applied_lora_pairs", -1)) != 128
        ):
            raise RuntimeError(f"merge application contract failed for {arm}")
        merged_fingerprint = checkpoint_fingerprint(output)
        receipt = {
            "schema_version": 1,
            "arm": arm,
            "seed": int(config["sft"]["screen_seed"]),
            "adapter": str(adapter),
            "adapter_sha256": adapter_hash,
            "adapter_tensor_manifest": adapter_receipt["adapter_tensor_manifest"],
            "output": str(output),
            "merge_application_receipt_sha256": sha256_file(application_path),
            "applied_lora_pairs": 128,
            "merged_checkpoint_fingerprint": merged_fingerprint,
        }
        write_json(receipt_path, receipt)
        receipts[arm] = receipt
    write_json(RUNS_DIR / "merge_matrix_summary.json", receipts)
    return receipts


def validated_merged_fingerprint(
    config: dict[str, Any], arm: str, *, seed: int = 42
) -> str:
    artifact_root = external_root(config)
    root = artifact_root / "merged" / f"seed{seed}" / arm
    receipt_path = root / "merge_receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError(f"missing merge receipt for {arm} seed {seed}")
    receipt = read_json(receipt_path)
    adapter = artifact_root / "adapters" / f"seed{seed}" / arm
    training_receipt = read_json(adapter / "training_receipt.json")
    validate_training_receipt(
        config,
        arm=arm,
        seed=seed,
        dataset=artifact_root / "sft" / f"{arm}.jsonl.gz",
        output=adapter,
        receipt=training_receipt,
    )
    adapter_sha256 = training_receipt["artifacts"]["adapter_model.safetensors"][
        "sha256"
    ]
    application_path = root / "merge_application_receipt.json"
    application = read_json(application_path)
    current = checkpoint_fingerprint(root)
    if (
        receipt.get("arm") != arm
        or int(receipt.get("seed", seed)) != seed
        or receipt.get("adapter_sha256") != adapter_sha256
        or receipt.get("adapter_tensor_manifest")
        != training_receipt.get("adapter_tensor_manifest")
        or int(receipt.get("applied_lora_pairs", -1)) != 128
        or receipt.get("merge_application_receipt_sha256")
        != sha256_file(application_path)
        or application.get("adapter_sha256") != adapter_sha256
        or int(application.get("applied_lora_pairs", -1)) != 128
        or receipt.get("merged_checkpoint_fingerprint") != current
    ):
        raise RuntimeError(f"merged checkpoint fingerprint mismatch for {arm} seed {seed}")
    return str(current["sha256"])


def behavioral_difference_probe(config: dict[str, Any]) -> dict[str, Any]:
    """Reject no-op merged deployments before result-bearing evaluation."""
    validate_preselection_amendment_receipt(config)
    design_boundary_receipt(config)
    artifact_root = external_root(config)
    items = load_split("termination_pilot")[:8]
    records = [
        {
            "id": item["id"],
            "messages": [{"role": "user", "content": item["prompt"]}],
            "meta": {"family": item["family"], "level": item["level"]},
        }
        for item in items
    ]
    sampling = SamplingConfig(
        thinking="natural",
        n=1,
        max_tokens=4096,
        greedy=True,
        run_seed=int(config["evaluation"]["greedy_seed"]) + 901,
    )

    def generate(
        label: str, model_override: Path | None, model_fingerprint: str
    ) -> tuple[list[dict[str, Any]], str]:
        path = artifact_root / "deployment_probe" / f"{label}.jsonl.gz"
        receipt_path = artifact_root / "deployment_probe" / f"{label}.receipt.json"
        contract = canonical_sha256(
            {
                "label": label,
                "model_fingerprint": model_fingerprint,
                "records": records,
                "sampling": sampling_contract(sampling),
                "engine": config["engine"],
                "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
                "environment_lock_sha256": sha256_file(
                    ROOT / "requirements-vllm.lock.txt"
                ),
            }
        )
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if (
                valid_receipt(receipt)
                and receipt.get("contract_sha256") == contract
                and receipt.get("model_fingerprint") == model_fingerprint
            ):
                return read_jsonl_gz(path), contract
        with VLLMRunner(
            engine_config(config, model_override=model_override)
        ) as runner:
            rows, metadata = runner.generate(records, sampling)
        receipt = write_jsonl_gz(path, rows)
        write_json(
            receipt_path,
            {
                **receipt,
                "generation_metadata": metadata,
                "label": label,
                "contract_sha256": contract,
                "model_fingerprint": model_fingerprint,
            },
        )
        return rows, contract

    base_fingerprint = base_model_fingerprint()
    base0, base0_contract = generate("base0", None, base_fingerprint)
    base1, base1_contract = generate("base1", None, base_fingerprint)
    base_tokens = {
        str(row["id"]): row["outputs"][0]["token_ids"] for row in base0
    }
    base_repeat_tokens = {
        str(row["id"]): row["outputs"][0]["token_ids"] for row in base1
    }
    null_differences = sum(
        base_tokens[task_id] != base_repeat_tokens[task_id]
        for task_id in base_tokens
    )
    if null_differences:
        raise RuntimeError(
            f"greedy base/base deployment probe is nondeterministic on {null_differences} tasks"
        )
    arms = {}
    arm_fingerprints = {}
    arm_contracts = {}
    for arm in config["sft"]["arms"]:
        fingerprint = validated_merged_fingerprint(config, arm)
        rows, contract = generate(
            arm,
            artifact_root / "merged" / "seed42" / arm,
            fingerprint,
        )
        arm_fingerprints[arm] = fingerprint
        arm_contracts[arm] = contract
        installed = {
            str(row["id"]): row["outputs"][0]["token_ids"] for row in rows
        }
        differences = sum(
            base_tokens[task_id] != installed[task_id] for task_id in base_tokens
        )
        arms[arm] = {"tasks": len(items), "token_sequence_differences": differences}
        if differences == 0:
            raise RuntimeError(f"merged deployment for {arm} is a behavioral no-op")
    result = {
        "schema_version": 1,
        "passed": True,
        "base_base_differences": null_differences,
        "base_model_fingerprint": base_fingerprint,
        "base_contracts": [base0_contract, base1_contract],
        "arm_model_fingerprints": arm_fingerprints,
        "arm_contracts": arm_contracts,
        "arms": arms,
    }
    write_json(RUNS_DIR / "behavioral_difference_probe.json", result)
    return result


def analyze_evaluation(config: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen Stage-A funnel and emit the conditional arm list."""
    # Re-enter the cache-contract gate so standalone analysis cannot consume
    # stale prompt/sampling/runner or checkpoint-bound generations.
    evaluate_matrix(config, mode="greedy", phase="stage_a")
    artifact_root = external_root(config)
    greedy_index = read_json(
        artifact_root
        / "evaluation"
        / "seed42"
        / "stage_a"
        / "greedy"
        / "index.json"
    )
    arms = ["base", *config["sft"]["arms"]]
    splits = tuple(config["evaluation"]["stage_a_splits"])
    task_scores: dict[str, dict[str, dict[str, float]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    for arm in arms:
        task_scores[arm] = {}
        metrics[arm] = {}
        for split in splits:
            entry = greedy_index.get("arms", {}).get(arm, {}).get(split)
            if not entry or not valid_receipt(entry["artifact"]):
                raise RuntimeError(f"missing or invalid Stage-A artifact: {arm}/{split}")
            expected_fingerprint = (
                base_model_fingerprint()
                if arm == "base"
                else validated_merged_fingerprint(config, arm)
            )
            if (
                entry.get("model_fingerprint") != expected_fingerprint
                or len(str(entry.get("contract_sha256", ""))) != 64
            ):
                raise RuntimeError(f"stale Stage-A model contract: {arm}/{split}")
            rows = read_jsonl_gz(
                Path(entry["artifact"]["path"])
            )
            row_ids = [str(row["id"]) for row in rows]
            expected_ids = {
                str(item["id"]) for item in load_evaluation_scope(config, split)
            }
            if len(row_ids) != len(set(row_ids)) or set(row_ids) != expected_ids:
                raise RuntimeError(
                    f"Stage-A task identity mismatch: {arm}/{split} "
                    f"rows={len(row_ids)} unique={len(set(row_ids))} "
                    f"expected={len(expected_ids)}"
                )
            if any(len(row.get("outputs", [])) != 1 for row in rows):
                raise RuntimeError(f"greedy Stage-A row has non-unit outputs: {arm}/{split}")
            if any(
                int(row["outputs"][0].get("n_stage2_prompt_tokens", -1)) != 0
                for row in rows
            ):
                raise RuntimeError(f"natural Stage-A unexpectedly used a second prefill: {arm}/{split}")
            if any(
                not math.isfinite(float(row["outputs"][0]["score"]))
                or not isinstance(row["outputs"][0].get("parsed"), bool)
                for row in rows
            ):
                raise RuntimeError(f"invalid Stage-A score/parse value: {arm}/{split}")
            scores = {
                str(row["id"]): float(row["outputs"][0]["score"])
                for row in rows
            }
            parsed = [bool(row["outputs"][0]["parsed"]) for row in rows]
            closed = [bool(row["outputs"][0]["thinking_closed"]) for row in rows]
            by_family: dict[str, list[float]] = {}
            for row in rows:
                by_family.setdefault(str(row["family"]), []).append(
                    float(row["outputs"][0]["score"])
                )
            parsed_scores = [
                float(row["outputs"][0]["score"])
                for row in rows
                if row["outputs"][0]["parsed"]
            ]
            task_scores[arm][split] = scores
            metrics[arm][split] = {
                "tasks": len(rows),
                "accuracy": mean(list(scores.values())),
                "parse_rate": mean(parsed),
                "parse_conditional_accuracy": (
                    mean(parsed_scores) if parsed_scores else None
                ),
                "natural_close_rate": mean(closed),
                "family_macro": mean([mean(values) for values in by_family.values()]),
                "family_accuracy": {
                    family: mean(values) for family, values in sorted(by_family.items())
                },
                "mean_thinking_tokens": mean(
                    [float(row["outputs"][0]["n_thinking_tokens"]) for row in rows]
                ),
                "actual_forward_tokens": sum(
                    int(row["outputs"][0]["n_stage1_prompt_tokens"])
                    + int(row["outputs"][0]["n_stage2_prompt_tokens"])
                    + int(row["outputs"][0]["n_sampled_tokens"])
                    for row in rows
                ),
            }

    comparisons: dict[str, Any] = {}
    baselines = ("random_natural", "success_rft", "shortest_natural")
    treatments = ("answer_potential", "joint_potential")
    contrast_pairs = [
        (treatment, baseline)
        for treatment in treatments
        for baseline in (*baselines, "potential_shuffle", "base")
    ]
    contrast_pairs.extend(
        ("shortest_natural", baseline)
        for baseline in ("random_natural", "success_rft", "base")
    )
    for treatment, baseline in contrast_pairs:
        key = f"{treatment}_minus_{baseline}"
        comparisons[key] = paired_bootstrap(
            {
                task_id: (
                    task_scores[treatment]["core_iid"][task_id],
                    task_scores[baseline]["core_iid"][task_id],
                )
                for task_id in task_scores[treatment]["core_iid"]
            },
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"])
            + _stable_analysis_offset(key),
        )

    reachability = {}
    treatment_verdicts = {}
    for treatment in treatments:
        baseline_accuracy = max(
            float(metrics[arm]["core_iid"]["accuracy"]) for arm in baselines
        )
        tied_baselines = [
            arm
            for arm in baselines
            if abs(float(metrics[arm]["core_iid"]["accuracy"]) - baseline_accuracy)
            <= 1e-12
        ]
        baseline = tied_baselines[0]
        positive_delta = float(config["evaluation"]["positive_delta"])
        reachable = baseline_accuracy + positive_delta <= 1.0 + 1e-12
        reachability[treatment] = {
            "metric": "core_iid_accuracy",
            "hard_lower": 0.0,
            "hard_upper": 1.0,
            "observed_strongest_baseline": baseline_accuracy,
            "tied_strongest_baselines": tied_baselines,
            "required_delta": positive_delta,
            "required_treatment_value": baseline_accuracy + positive_delta,
            "reachable": reachable,
        }
        primary_contrasts = {
            arm: comparisons[f"{treatment}_minus_{arm}"] for arm in tied_baselines
        }
        primary = primary_contrasts[baseline]
        shuffled = comparisons[f"{treatment}_minus_potential_shuffle"]
        parse_deltas = {
            arm: metrics[treatment]["core_iid"]["parse_rate"]
            - metrics[arm]["core_iid"]["parse_rate"]
            for arm in tied_baselines
        }
        family_deltas = {
            arm: metrics[treatment]["core_iid"]["family_macro"]
            - metrics[arm]["core_iid"]["family_macro"]
            for arm in tied_baselines
        }
        positive = (
            reachable
            and all(
                float(contrast["mean_delta"]) >= positive_delta
                and float(contrast["ci95_low"]) > 0
                for contrast in primary_contrasts.values()
            )
            and float(shuffled["mean_delta"]) > 0
            and all(
                delta >= float(config["evaluation"]["noninferiority_delta"])
                for delta in parse_deltas.values()
            )
            and all(
                delta >= float(config["evaluation"]["noninferiority_delta"])
                for delta in family_deltas.values()
            )
        )
        treatment_verdicts[treatment] = {
            "strongest_trace_baseline": baseline,
            "tied_strongest_baselines": tied_baselines,
            "strongest_baseline_tie_rule": (
                "require every accuracy tie; listed-order arm used for replication"
            ),
            "primary_contrast": primary,
            "primary_contrasts_for_all_ties": primary_contrasts,
            "shuffle_contrast": shuffled,
            "parse_delta": parse_deltas[baseline],
            "parse_deltas_for_all_ties": parse_deltas,
            "family_macro_delta": family_deltas[baseline],
            "family_macro_deltas_for_all_ties": family_deltas,
            "stage_b_triggered": positive,
        }

    triggered = [
        treatment
        for treatment, verdict in treatment_verdicts.items()
        if verdict["stage_b_triggered"]
    ]
    trace_arms = [
        "random_natural",
        "success_rft",
        "shortest_natural",
        "answer_potential",
        "joint_potential",
        "potential_shuffle",
    ]
    strongest_trace_accuracy = max(
        float(metrics[arm]["core_iid"]["accuracy"]) for arm in trace_arms
    )
    tied_strongest_trace_arms = [
        arm
        for arm in trace_arms
        if abs(float(metrics[arm]["core_iid"]["accuracy"]) - strongest_trace_accuracy)
        <= 1e-12
    ]
    strongest_trace_arm = tied_strongest_trace_arms[0]
    result = {
        "schema_version": 1,
        "phase": "stage_a",
        "metrics": metrics,
        "paired_core_iid_comparisons": comparisons,
        "gate_reachability": reachability,
        "treatment_verdicts": treatment_verdicts,
        "stage_b_triggered_treatments": triggered,
        "strongest_trace_arm": strongest_trace_arm,
        "tied_strongest_trace_arms": tied_strongest_trace_arms,
        "shortest_banking_leads": strongest_trace_arm == "shortest_natural",
        "verdict": (
            "STAGE_B_TRIGGERED" if triggered else "CORE_BANKING_NEGATIVE"
        ),
        "complete_seed42_stage_a_matrix": all(
            arm in metrics and all(split in metrics[arm] for split in splits)
            for arm in arms
        ),
    }
    write_json(RUNS_DIR / "gate_reachability.json", reachability)
    write_json(RUNS_DIR / "stage_a_analysis.json", result)
    print(
        json.dumps(
            {
                "core_iid": {
                    arm: metrics[arm]["core_iid"]["accuracy"] for arm in arms
                },
                "verdict": result["verdict"],
                "triggered": triggered,
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def _stable_analysis_offset(text_value: str) -> int:
    return int.from_bytes(
        hashlib.blake2b(text_value.encode("utf-8"), digest_size=4).digest(), "big"
    ) % 100_000


def run_conditional_stage_b(
    config: dict[str, Any], stage_a: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run and analyze only the expansion licensed by the frozen Stage-A gate."""
    validate_preselection_amendment_receipt(config)
    if stage_a is None:
        stage_a = analyze_evaluation(config)
    triggered = list(stage_a["stage_b_triggered_treatments"])
    if not triggered:
        result = {
            "schema_version": 1,
            "triggered": False,
            "reason": "no potential treatment cleared the frozen Stage-A funnel",
        }
        write_json(RUNS_DIR / "stage_b_analysis.json", result)
        return result

    evaluate_matrix(config, mode="greedy", phase="stage_b")
    sample_arms = {"base", "shortest_natural", "potential_shuffle"}
    for treatment in triggered:
        sample_arms.add(treatment)
        sample_arms.add(
            str(stage_a["treatment_verdicts"][treatment]["strongest_trace_baseline"])
        )
    ordered_sample_arms = [
        arm for arm in ["base", *config["sft"]["arms"]] if arm in sample_arms
    ]
    evaluate_matrix(
        config,
        mode="sample8",
        phase="stage_b",
        arms_override=ordered_sample_arms,
    )

    root = external_root(config) / "evaluation" / "seed42" / "stage_b"
    greedy_index = read_json(root / "greedy" / "index.json")
    sample_index = read_json(root / "sample8" / "index.json")
    arms = ["base", *config["sft"]["arms"]]
    splits = tuple(config["evaluation"]["stage_b_full_splits"])
    metrics: dict[str, dict[str, Any]] = {}
    scores: dict[str, dict[str, dict[str, float]]] = {}
    for arm in arms:
        metrics[arm] = {}
        scores[arm] = {}
        for split in splits:
            rows = read_jsonl_gz(
                Path(greedy_index["arms"][arm][split]["artifact"]["path"])
            )
            task_values = {
                str(row["id"]): float(row["outputs"][0]["score"])
                for row in rows
            }
            by_family: dict[str, list[float]] = {}
            for row in rows:
                by_family.setdefault(str(row["family"]), []).append(
                    float(row["outputs"][0]["score"])
                )
            scores[arm][split] = task_values
            metrics[arm][split] = {
                "tasks": len(rows),
                "accuracy": mean(list(task_values.values())),
                "parse_rate": mean(
                    [bool(row["outputs"][0]["parsed"]) for row in rows]
                ),
                "natural_close_rate": mean(
                    [bool(row["outputs"][0]["thinking_closed"]) for row in rows]
                ),
                "family_macro": mean([mean(value) for value in by_family.values()]),
                "actual_forward_tokens": sum(
                    int(row["outputs"][0]["n_stage1_prompt_tokens"])
                    + int(row["outputs"][0]["n_stage2_prompt_tokens"])
                    + int(row["outputs"][0]["n_sampled_tokens"])
                    for row in rows
                ),
            }

    treatment_results = {}
    for treatment in triggered:
        baseline = str(
            stage_a["treatment_verdicts"][treatment]["strongest_trace_baseline"]
        )
        key = f"stage_b::{treatment}_minus_{baseline}"
        paired = paired_bootstrap(
            {
                task_id: (
                    scores[treatment]["iid_eval"][task_id],
                    scores[baseline]["iid_eval"][task_id],
                )
                for task_id in scores[treatment]["iid_eval"]
            },
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"])
            + _stable_analysis_offset(key),
        )
        shuffle = paired_bootstrap(
            {
                task_id: (
                    scores[treatment]["iid_eval"][task_id],
                    scores["potential_shuffle"]["iid_eval"][task_id],
                )
                for task_id in scores[treatment]["iid_eval"]
            },
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"])
            + _stable_analysis_offset(f"stage_b::{treatment}::shuffle"),
        )
        parse_delta = (
            metrics[treatment]["iid_eval"]["parse_rate"]
            - metrics[baseline]["iid_eval"]["parse_rate"]
        )
        held_delta = (
            metrics[treatment]["held_family_eval"]["accuracy"]
            - metrics[baseline]["held_family_eval"]["accuracy"]
        )
        rendering_delta = (
            metrics[treatment]["rendering_eval"]["accuracy"]
            - metrics[baseline]["rendering_eval"]["accuracy"]
        )
        positive = (
            float(paired["mean_delta"])
            >= float(config["evaluation"]["positive_delta"])
            and float(paired["ci95_low"]) > 0
            and float(shuffle["mean_delta"]) > 0
            and parse_delta >= float(config["evaluation"]["noninferiority_delta"])
            and held_delta >= float(config["evaluation"]["noninferiority_delta"])
            and rendering_delta >= float(config["evaluation"]["noninferiority_delta"])
        )
        treatment_results[treatment] = {
            "baseline": baseline,
            "full_iid_contrast": paired,
            "shuffle_contrast": shuffle,
            "parse_delta": parse_delta,
            "held_family_accuracy_delta": held_delta,
            "rendering_accuracy_delta": rendering_delta,
            "potential_banking_positive": positive,
        }

    base_rows = read_jsonl_gz(
        Path(sample_index["arms"]["base"]["core_iid"]["artifact"]["path"])
    )
    sample_more = {}
    for k in (1, 2, 4, 8):
        sample_more[str(k)] = {
            "pass_at_k": mean(
                [
                    any(output["correct"] for output in row["outputs"][:k])
                    for row in base_rows
                ]
            ),
            "actual_forward_tokens": sum(
                sum(
                    int(output["n_stage1_prompt_tokens"])
                    + int(output["n_stage2_prompt_tokens"])
                    + int(output["n_sampled_tokens"])
                    for output in row["outputs"][:k]
                )
                for row in base_rows
            ),
        }

    replication = run_seed43_replication(config, stage_a)
    for treatment, values in treatment_results.items():
        rep = replication.get("contrasts", {}).get(treatment)
        values["replicated_banking_positive"] = bool(
            values["potential_banking_positive"]
            and rep
            and rep["replicated_banking_positive"]
        )
        stage_a_cost = int(
            stage_a["metrics"][treatment]["core_iid"]["actual_forward_tokens"]
        )
        stage_a_accuracy = float(
            stage_a["metrics"][treatment]["core_iid"]["accuracy"]
        )
        matched_wins = [
            {"k": int(k), **point}
            for k, point in sample_more.items()
            if int(point["actual_forward_tokens"]) <= stage_a_cost
            and stage_a_accuracy > float(point["pass_at_k"])
        ]
        values["matched_compute_sample_more_wins"] = matched_wins
        values["mission_positive"] = bool(
            values["replicated_banking_positive"] and matched_wins
        )

    result = {
        "schema_version": 1,
        "triggered": True,
        "triggered_treatments": triggered,
        "metrics": metrics,
        "sample_more_curve": sample_more,
        "replication": replication,
        "treatment_results": treatment_results,
        "verdict": (
            "MISSION_POSITIVE"
            if any(row["mission_positive"] for row in treatment_results.values())
            else "REPLICATED_BANKING_POSITIVE"
            if any(
                row["replicated_banking_positive"]
                for row in treatment_results.values()
            )
            else "POTENTIAL_BANKING_POSITIVE"
            if any(row["potential_banking_positive"] for row in treatment_results.values())
            else "STAGE_B_NOT_CONFIRMED"
        ),
    }
    write_json(RUNS_DIR / "stage_b_analysis.json", result)
    write_json(RUNS_DIR / "final_analysis.json", result)
    return result


def run_seed43_replication(
    config: dict[str, Any], analysis: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Run only the preregistered treatment/baseline pairs that triggered."""
    validate_preselection_amendment_receipt(config)
    if analysis is None:
        analysis = analyze_evaluation(config)
    triggered = {
        treatment: analysis["treatment_verdicts"][treatment][
            "strongest_trace_baseline"
        ]
        for treatment in analysis["stage_b_triggered_treatments"]
    }
    if not triggered:
        result = {
            "schema_version": 1,
            "triggered": False,
            "reason": "no seed-42 treatment met the frozen replication trigger",
        }
        write_json(RUNS_DIR / "replication_analysis.json", result)
        return result

    python = ROOT / ".venv" / "bin" / "python"
    artifact_root = external_root(config)
    arms = sorted(set(triggered) | set(triggered.values()))
    seed = int(config["sft"]["replication_seed"])
    seed43_receipts: dict[str, dict[str, Any]] = {}
    for arm in arms:
        dataset = artifact_root / "sft" / f"{arm}.jsonl.gz"
        adapter = artifact_root / "adapters" / "seed43" / arm
        train_receipt = adapter / "training_receipt.json"
        if not train_receipt.is_file():
            print(f"[replication] training seed43 {arm}", flush=True)
            subprocess.run(
                [
                    str(python),
                    str(EXP / "scripts" / "train_think.py"),
                    "--arm",
                    arm,
                    "--dataset",
                    str(dataset),
                    "--out",
                    str(adapter),
                    "--seed",
                    str(seed),
                ],
                cwd=ROOT,
                check=True,
            )
        adapter_receipt = read_json(train_receipt)
        validate_training_receipt(
            config,
            arm=arm,
            seed=seed,
            dataset=dataset,
            output=adapter,
            receipt=adapter_receipt,
        )
        seed43_receipts[arm] = adapter_receipt
        merged = artifact_root / "merged" / "seed43" / arm
        merge_receipt_path = merged / "merge_receipt.json"
        adapter_hash = adapter_receipt["artifacts"]["adapter_model.safetensors"][
            "sha256"
        ]
        if merge_receipt_path.is_file():
            validated_merged_fingerprint(config, arm, seed=43)
            existing_merge = read_json(merge_receipt_path)
            if existing_merge.get("adapter_sha256") != adapter_hash:
                raise RuntimeError(f"stale seed43 merge receipt for {arm}")
        else:
            print(f"[replication] merging seed43 {arm}", flush=True)
            subprocess.run(
                [
                    str(python),
                    str(EXP / "scripts" / "merge_adapter.py"),
                    "--adapter",
                    str(adapter),
                    "--out",
                    str(merged),
                ],
                cwd=ROOT,
                check=True,
            )
            application_path = merged / "merge_application_receipt.json"
            application = read_json(application_path)
            if (
                application.get("adapter_sha256") != adapter_hash
                or application.get("adapter_tensor_manifest")
                != adapter_receipt.get("adapter_tensor_manifest")
                or int(application.get("applied_lora_pairs", -1)) != 128
            ):
                raise RuntimeError(f"seed43 merge application contract failed for {arm}")
            merged_fingerprint = checkpoint_fingerprint(merged)
            write_json(
                merge_receipt_path,
                {
                    "schema_version": 1,
                    "arm": arm,
                    "seed": seed,
                    "adapter_sha256": adapter_hash,
                    "adapter_tensor_manifest": adapter_receipt[
                        "adapter_tensor_manifest"
                    ],
                    "merge_application_receipt_sha256": sha256_file(
                        application_path
                    ),
                    "applied_lora_pairs": 128,
                    "merged_checkpoint_fingerprint": merged_fingerprint,
                },
            )

    seed43_initial_hashes = {
        str(receipt["adapter_initial_state"]["sha256"])
        for receipt in seed43_receipts.values()
    }
    if len(seed43_initial_hashes) != 1:
        raise RuntimeError(
            f"seed43 arms did not share one LoRA initialization: {seed43_initial_hashes}"
        )
    seed42_initial_hashes = {
        str(
            read_json(
                artifact_root / "adapters" / "seed42" / arm / "training_receipt.json"
            )["adapter_initial_state"]["sha256"]
        )
        for arm in arms
    }
    if len(seed42_initial_hashes) != 1 or seed42_initial_hashes == seed43_initial_hashes:
        raise RuntimeError(
            "seed42/seed43 initialization hashes do not establish distinct replications"
        )

    iid_items = load_evaluation_scope(config, "core_iid")
    item_by_id = {str(item["id"]): item for item in iid_items}
    records = [
        {
            "id": item["id"],
            "messages": [{"role": "user", "content": item["prompt"]}],
            "meta": {"family": item["family"], "level": item["level"]},
        }
        for item in iid_items
    ]
    sampling = SamplingConfig(
        thinking="natural",
        n=1,
        max_tokens=int(config["evaluation"]["natural_max_tokens"]),
        greedy=True,
        run_seed=int(config["evaluation"]["greedy_seed"]),
    )
    evaluation_rows = {}
    for arm in arms:
        model_fingerprint = validated_merged_fingerprint(config, arm, seed=43)
        contract = canonical_sha256(
            {
                "phase": "seed43_replication",
                "arm": arm,
                "model_fingerprint": model_fingerprint,
                "scope": [
                    {
                        "id": item["id"],
                        "prompt": item["prompt"],
                        "canonical_answer": item["canonical_answer"],
                        "family": item["family"],
                        "level": item["level"],
                    }
                    for item in iid_items
                ],
                "sampling": sampling_contract(sampling),
                "engine": config["engine"],
                "runner_sha256": sha256_file(EXP / "src" / "vllm_runner.py"),
                "environment_lock_sha256": sha256_file(
                    ROOT / "requirements-vllm.lock.txt"
                ),
                "analysis_script_sha256": sha256_file(Path(__file__)),
            }
        )
        output_path = (
            artifact_root / "evaluation" / "seed43" / "core_iid" / f"{arm}.jsonl.gz"
        )
        receipt_path = output_path.with_suffix(".receipt.json")
        receipt = read_json(receipt_path) if receipt_path.is_file() else None
        if (
            receipt
            and valid_receipt(receipt)
            and receipt.get("contract_sha256") == contract
            and receipt.get("model_fingerprint") == model_fingerprint
        ):
            scored = read_jsonl_gz(output_path)
        else:
            with VLLMRunner(
                engine_config(
                    config,
                    model_override=artifact_root / "merged" / "seed43" / arm,
                )
            ) as runner:
                raw, metadata = runner.generate(records, sampling)
            scored, summary = _score_evaluation_rows(item_by_id, raw)
            artifact = write_jsonl_gz(output_path, scored)
            write_json(
                receipt_path,
                {
                    **artifact,
                    "summary": summary,
                    "generation_metadata": metadata,
                    "contract_sha256": contract,
                    "model_fingerprint": model_fingerprint,
                },
            )
        row_ids = [str(row["id"]) for row in scored]
        if (
            len(row_ids) != len(set(row_ids))
            or set(row_ids) != set(item_by_id)
            or any(len(row.get("outputs", [])) != 1 for row in scored)
        ):
            raise RuntimeError(f"seed43 evaluation identity mismatch for {arm}")
        evaluation_rows[arm] = scored

    seed42_index = read_json(
        artifact_root
        / "evaluation"
        / "seed42"
        / "stage_a"
        / "greedy"
        / "index.json"
    )
    base_rows = read_jsonl_gz(
        Path(seed42_index["arms"]["base"]["core_iid"]["artifact"]["path"])
    )
    base_tokens = {
        str(row["id"]): row["outputs"][0]["token_ids"] for row in base_rows
    }
    behavior = {
        arm: sum(
            base_tokens[str(row["id"])] != row["outputs"][0]["token_ids"]
            for row in evaluation_rows[arm]
        )
        for arm in arms
    }
    if any(count == 0 for count in behavior.values()):
        raise RuntimeError(f"seed43 merged behavioral no-op: {behavior}")

    contrasts = {}
    for treatment, baseline in triggered.items():
        t43 = {
            str(row["id"]): float(row["outputs"][0]["score"])
            for row in evaluation_rows[treatment]
        }
        b43 = {
            str(row["id"]): float(row["outputs"][0]["score"])
            for row in evaluation_rows[baseline]
        }
        seed43_pairs = {
            task_id: (t43[task_id], b43[task_id]) for task_id in t43
        }
        seed42_t = task_scores_from_eval_index(
            seed42_index, treatment, "core_iid"
        )
        seed42_b = task_scores_from_eval_index(seed42_index, baseline, "core_iid")
        pooled = {
            **{
                f"seed42::{task_id}": (seed42_t[task_id], seed42_b[task_id])
                for task_id in seed42_t
            },
            **{
                f"seed43::{task_id}": pair for task_id, pair in seed43_pairs.items()
            },
        }
        seed43_stats = paired_bootstrap(
            seed43_pairs,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + 43,
        )
        pooled_stats = paired_bootstrap(
            pooled,
            resamples=int(config["evaluation"]["bootstrap_resamples"]),
            seed=int(config["evaluation"]["bootstrap_seed"]) + 4243,
        )
        contrasts[treatment] = {
            "baseline": baseline,
            "seed43": seed43_stats,
            "pooled": pooled_stats,
            "replicated_banking_positive": (
                float(seed43_stats["mean_delta"]) > 0
                and float(pooled_stats["ci95_low"]) > 0
            ),
        }
    result = {
        "schema_version": 1,
        "triggered": True,
        "seed": seed,
        "arms": arms,
        "behavioral_differences_vs_base": behavior,
        "contrasts": contrasts,
    }
    write_json(RUNS_DIR / "replication_analysis.json", result)
    return result


def task_scores_from_eval_index(
    index: dict[str, Any], arm: str, split: str
) -> dict[str, float]:
    rows = read_jsonl_gz(Path(index["arms"][arm][split]["artifact"]["path"]))
    return {
        str(row["id"]): float(row["outputs"][0]["score"])
        for row in rows
    }


def run_smoke(config: dict[str, Any]) -> dict[str, Any]:
    design_boundary_receipt(config)
    build_data()
    import_inherited_train_pool(config)
    index = read_json(
        external_root(config) / "pools" / "train_independent" / "index.json"
    )
    item_by_id = {str(item["id"]): item for item in load_core_train(config)}
    traces: list[dict[str, Any]] = []
    for task_id in sorted(index["shards"]):
        traces.extend(
            row
            for row in read_jsonl_gz(
                Path(index["shards"][task_id]["artifact"]["path"])
            )
            if row["natural_close"] and not row["loop_flag"]
        )
        if len(traces) >= 2:
            break
    traces = traces[:2]
    if len(traces) != 2:
        raise RuntimeError("smoke requires two inherited natural traces")
    items = [item_by_id[str(trace["task_id"])] for trace in traces]
    unique_items = {str(item["id"]): item for item in items}
    smoke_dir = RUNS_DIR / "smoke"
    with AnswerPotentialModel(engine_config(config)) as model:
        vllm_scores, score_meta = model.score_canonical_joint(
            list(unique_items.values()),
            traces,
            chunk_size=int(config["scoring"]["chunk_size"]),
        )
    write_jsonl(smoke_dir / "traces.jsonl", traces)
    write_jsonl(smoke_dir / "vllm_scores.jsonl", vllm_scores)
    scorer = HFAnswerPotentialScorer()
    try:
        hf_scores = [
            scorer.score_trace(item_by_id[str(trace["task_id"])], trace)
            for trace in traces
        ]
    finally:
        scorer.close()
    write_jsonl(smoke_dir / "hf_scores.jsonl", hf_scores)
    vllm_by_trace = {row["trace_id"]: row for row in vllm_scores}
    deltas = []
    for row in hf_scores:
        other = vllm_by_trace[str(row["trace_id"])]
        deltas.extend(
            [
                abs(float(row["answer_ll_sum"]) - float(other["answer_ll_sum"]))
                / len(row["answer_token_ids"]),
                abs(float(row["joint_ll_sum"]) - float(other["joint_ll_sum"]))
                / len(row["joint_token_logprobs"]),
                abs(
                    float(row["answer_gain_per_answer_token"])
                    - float(other["answer_gain_per_answer_token"])
                ),
                abs(
                    float(row["joint_gain_per_answer_token"])
                    - float(other["joint_gain_per_answer_token"])
                ),
            ]
        )
    result = {
        "schema_version": 1,
        "passed": (
            len(traces) == 2
            and all(row.get("prior_logprob_mean") is not None for row in traces)
            and all(math.isfinite(value) for value in deltas)
            and max(deltas) <= float(config["scoring"]["max_abs_mean_token_delta"])
        ),
        "traces": _summarize_traces(traces),
        "max_abs_mean_token_delta_hf_vllm": max(deltas),
        "parity_threshold": config["scoring"]["max_abs_mean_token_delta"],
        "vllm_score_meta": score_meta,
        "new_generation_performed": False,
    }
    write_json(smoke_dir / "result.json", result)
    print(json.dumps(result["traces"], indent=2), flush=True)
    print(
        f"[smoke] parity={result['max_abs_mean_token_delta_hf_vllm']:.6f} "
        f"passed={result['passed']}",
        flush=True,
    )
    if not result["passed"]:
        raise RuntimeError("long-horizon smoke failed")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--stage",
        choices=(
            "data",
            "import",
            "smoke",
            "harvest",
            "parity",
            "score",
            "rollouts",
            "evidence-seal",
            "select",
            "train",
            "merge",
            "deployment-probe",
            "evaluate-stage-a",
            "analyze-stage-a",
            "stage-b",
            "replicate",
            "full",
        ),
        default="smoke",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke:
        args.stage = "smoke"
    config = load_config()
    if args.stage == "data":
        design_boundary_receipt(config)
        build_data()
        load_core_train(config)
        for split in config["evaluation"]["stage_a_splits"]:
            load_evaluation_scope(config, split)
    elif args.stage == "import":
        design_boundary_receipt(config)
        import_inherited_train_pool(config)
    elif args.stage == "smoke":
        run_smoke(config)
    elif args.stage == "harvest":
        import_inherited_train_pool(config)
        generate_pool(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        )
        ensure_minimum_natural_train_pool(config)
    elif args.stage == "parity":
        run_scorer_parity(
            config,
            source_stage=str(config["scoring"]["parity_source_stage"]),
            split="calibration",
        )
    elif args.stage == "score":
        score_pool(
            config,
            source_stage="train_independent",
            output_stage="train_independent_scores",
            split="train",
        )
    elif args.stage == "rollouts":
        rollout_pool(
            config,
            split="train",
            source_stages=["train_independent"],
            output_stage="train_rollouts_r1",
            r=int(config["sampling"]["train_rollouts_per_trace"]),
        )
    elif args.stage == "evidence-seal":
        seal_selection_evidence(config)
    elif args.stage == "select":
        build_sft_datasets(config)
    elif args.stage == "train":
        train_matrix(config)
    elif args.stage == "merge":
        merge_matrix(config)
    elif args.stage == "deployment-probe":
        behavioral_difference_probe(config)
    elif args.stage == "evaluate-stage-a":
        evaluate_matrix(config, mode="greedy", phase="stage_a")
    elif args.stage == "analyze-stage-a":
        analyze_evaluation(config)
    elif args.stage == "stage-b":
        run_conditional_stage_b(config)
    elif args.stage == "replicate":
        run_seed43_replication(config)
    else:
        build_data()
        import_inherited_train_pool(config)
        generate_pool(
            config,
            split="train",
            n=int(config["sampling"]["train_independent_n"]),
            stage="train_independent",
        )
        ensure_minimum_natural_train_pool(config)
        run_scorer_parity(
            config,
            source_stage=str(config["scoring"]["parity_source_stage"]),
            split="calibration",
        )
        score_pool(
            config,
            source_stage="train_independent",
            output_stage="train_independent_scores",
            split="train",
        )
        rollout_pool(
            config,
            split="train",
            source_stages=["train_independent"],
            output_stage="train_rollouts_r1",
            r=int(config["sampling"]["train_rollouts_per_trace"]),
        )
        build_sft_datasets(config)
        train_matrix(config)
        merge_matrix(config)
        behavioral_difference_probe(config)
        evaluate_matrix(config, mode="greedy", phase="stage_a")
        analysis = analyze_evaluation(config)
        if analysis["stage_b_triggered_treatments"]:
            run_conditional_stage_b(config, analysis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
