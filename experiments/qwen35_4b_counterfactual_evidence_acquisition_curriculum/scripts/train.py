#!/usr/bin/env python3
"""Low-dose QLoRA over complete conditional transition supercycles."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from harness import (  # noqa: E402
    REQUIRED_FROZEN_FILES,
    tokenizer_provenance,
    validate_registered_checkpoint,
    validate_registered_tokenizer_provenance,
)

MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_REVISION = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"
]
TRANSITIONS = (
    "start_to_inspect_source",
    "ambiguous_source_to_inspect_evidence",
    "evidence_to_policy_patch",
    "explicit_source_to_patch",
    "rejected_patch_to_changed_patch",
    "failed_test_to_diagnose",
    "diagnosis_to_changed_patch",
    "patch_ok_to_verify",
    "passed_test_to_commit",
)


def checkpointed_weighted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    signed_weights: torch.Tensor,
    chunk_positions: int,
) -> torch.Tensor:
    """Exact weighted CE without retaining a full-vocabulary FP32 temporary."""
    if chunk_positions < 1:
        raise ValueError("chunk_positions must be positive")
    flat_logits = logits.reshape(-1, logits.size(-1))
    flat_labels = labels.reshape(-1)
    flat_weights = signed_weights.reshape(-1)

    def chunk_loss(
        chunk_logits: torch.Tensor,
        chunk_labels: torch.Tensor,
        chunk_weights: torch.Tensor,
    ) -> torch.Tensor:
        losses = torch.nn.functional.cross_entropy(
            chunk_logits, chunk_labels.clamp(min=0), reduction="none"
        )
        return (losses * chunk_weights).sum()

    total = logits.new_zeros((), dtype=torch.float32)
    for start in range(0, flat_logits.size(0), chunk_positions):
        end = min(start + chunk_positions, flat_logits.size(0))
        total = total + checkpoint(
            chunk_loss,
            flat_logits[start:end],
            flat_labels[start:end],
            flat_weights[start:end],
            use_reentrant=True,
        )
    return total


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def unpadded_row_lengths(attention_mask: torch.Tensor) -> list[int]:
    """Validate prefix-contiguous masks and return physical batch-one lengths."""
    if attention_mask.ndim != 2:
        raise ValueError("attention mask must be rank two")
    lengths = []
    for row in attention_mask:
        length = int(row.sum().item())
        expected = torch.cat((
            torch.ones(length, dtype=row.dtype, device=row.device),
            torch.zeros(row.numel() - length, dtype=row.dtype, device=row.device),
        ))
        if not torch.equal(row, expected):
            raise ValueError("training attention mask is not a contiguous right-pad suffix")
        lengths.append(length)
    return lengths


def validate_design_lock(
    path: Path,
    *,
    bank_receipt_path: Path,
    arm: str,
    train_sha256: str,
) -> dict:
    """Refuse a model mutation unless the frozen design authorizes it."""
    expected_path = (EXP / "runs" / "preregistration_receipt.json").resolve()
    if path.resolve() != expected_path or not path.is_file():
        raise SystemExit("design lock is missing; real training is not authorized")
    payload = json.loads(path.read_text())
    frozen_order = payload.get("frozen_file_order")
    frozen_files = payload.get("frozen_files")
    design_commit = payload.get("design_commit")
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "locked"
        or payload.get("experiment_id") != EXP.name
        or payload.get("model_output_precedes_lock") is not False
        or not isinstance(frozen_order, list)
        or not isinstance(frozen_files, dict)
        or tuple(frozen_order) != REQUIRED_FROZEN_FILES
        or set(frozen_files) != set(REQUIRED_FROZEN_FILES)
        or not isinstance(design_commit, str)
        or len(design_commit) != 40
        or any(character not in "0123456789abcdef" for character in design_commit)
    ):
        raise SystemExit("design lock is not the registered immutable preregistration")
    expected_self = frozen_files.get("scripts/train.py")
    if expected_self != sha256_file(Path(__file__).resolve()):
        raise SystemExit("trainer differs from the frozen preregistered implementation")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", design_commit, "HEAD"],
        cwd=ROOT,
        check=False,
    )
    if ancestry.returncode:
        raise SystemExit("design commit is not an ancestor of HEAD")
    tracked = subprocess.run(
        [
            "git", "ls-files", "--error-unmatch",
            str(path.resolve().relative_to(ROOT)),
        ],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--short", "--", str(path.resolve())],
        cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.strip()
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    origin = subprocess.run(
        ["git", "rev-parse", "origin/main"], cwd=ROOT, text=True,
        capture_output=True, check=True,
    ).stdout.strip()
    if tracked.returncode or dirty or head != origin:
        raise SystemExit("design lock must be committed and pushed to origin/main")
    for relative, expected in frozen_files.items():
        if sha256_file(EXP / relative) != expected:
            raise SystemExit(f"frozen design changed after lock: {relative}")
    if payload.get("bank_receipt_sha256") != sha256_file(bank_receipt_path):
        raise SystemExit("bank receipt differs from the receipt frozen at design lock")
    smoke_path = EXP / "reports" / "smoke_receipt.json"
    if payload.get("smoke_receipt_sha256") != sha256_file(smoke_path):
        raise SystemExit("model-free smoke receipt differs from the design lock")
    if payload.get("bank_sha256", {}).get(arm) != train_sha256:
        raise SystemExit("training bank differs from the arm hash frozen at design lock")
    return payload


def validate_preflight_encoding(
    arm: str, current: dict, design_lock: dict
) -> dict:
    """Bind the real deterministic schedule to the locked encode-only preflight."""
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    artifact_root = Path(cfg["artifacts"]["root"])
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    path = artifact_root / "preflight" / "encode" / arm / "encoding_receipt.json"
    smoke_path = EXP / "reports" / "smoke_receipt.json"
    smoke = json.loads(smoke_path.read_text())
    if (
        design_lock.get("smoke_receipt_sha256") != sha256_file(smoke_path)
        or smoke.get("encoding_receipt_sha256", {}).get(arm) != sha256_file(path)
    ):
        raise SystemExit("preflight encoding receipt is not bound by the design lock")
    preflight = json.loads(path.read_text())
    dynamic = {
        "design_lock_sha256",
        "design_commit",
        "training_authorization_sha256",
        "selected_answer_max_tokens",
    }
    normalize = lambda value: {  # noqa: E731
        key: row for key, row in value.items() if key not in dynamic
    }
    if normalize(current) != normalize(preflight):
        raise SystemExit("real training schedule differs from locked encode preflight")
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def validate_training_contract(args, design_lock: dict, bank_receipt: dict) -> None:
    """Bind every model-affecting CLI degree of freedom to frozen config."""
    config_path = EXP / "configs" / "default.yaml"
    if design_lock["frozen_files"].get("configs/default.yaml") != sha256_file(config_path):
        raise SystemExit("training config differs from the design lock")
    cfg = yaml.safe_load(config_path.read_text())
    tcfg = cfg["training"]
    expected = {
        "epochs": int(tcfg["epochs"]),
        "lr": float(tcfg["learning_rate"]),
        "rank": int(tcfg["rank"]),
        "alpha": int(tcfg["alpha"]),
        "batch_size": int(tcfg["batch_size"]),
        "grad_accum": int(tcfg["gradient_accumulation_steps"]),
        "loss_chunk_positions": int(tcfg["loss_chunk_positions"]),
        "max_length": int(tcfg["max_length"]),
        "seed": int(tcfg["seed"]),
    }
    observed = {key: getattr(args, key) for key in expected}
    if observed != expected:
        raise SystemExit(f"training CLI differs from frozen config: {observed} != {expected}")
    registered_base = Path(cfg["model"]["start_checkpoint"])
    registered_base = (
        registered_base if registered_base.is_absolute() else ROOT / registered_base
    )
    if (
        args.base_model.resolve() != registered_base.resolve()
        or args.expected_base_weight_sha256 != cfg["model"]["start_weight_sha256"]
    ):
        raise SystemExit("training base checkpoint differs from frozen config")
    if args.smoke:
        raise SystemExit("post-lock training smoke is not a preregistered training arm")
    try:
        observed_tokenizer = tokenizer_provenance(args.base_model)
    except (OSError, ValueError) as exc:
        raise SystemExit(f"training tokenizer provenance is invalid: {exc}") from exc
    registered_tokenizer = {
        key: bank_receipt.get(key)
        for key in (
            "tokenizer_files",
            "tokenizer_manifest_sha256",
            "tokenizer_compatibility_sha256",
        )
    }
    if observed_tokenizer != registered_tokenizer:
        raise SystemExit("training tokenizer differs from bank calibration")
    if (
        observed_tokenizer["tokenizer_manifest_sha256"]
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or observed_tokenizer["tokenizer_compatibility_sha256"]
        != cfg["model"]["tokenizer_compatibility_sha256"]
    ):
        raise SystemExit("training tokenizer differs from frozen config identity")


def validate_training_authorization(path: Path, design_lock_path: Path) -> dict:
    expected_keys = {
        "schema_version", "stage", "experiment_id", "design_lock_sha256",
        "issuer_sha256", "config_sha256",
        "ancestor_receipts", "selected_answer_max_tokens", "checks", "gate",
        "training_authorized", "menagerie_authorized",
    }
    payload = json.loads(path.read_text())
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("stage") != "training_authorization"
        or payload.get("experiment_id") != EXP.name
        or payload.get("issuer_sha256")
        != sha256_file(EXP / "scripts" / "run.py")
        or payload.get("config_sha256")
        != sha256_file(EXP / "configs" / "default.yaml")
        or payload.get("design_lock_sha256") != sha256_file(design_lock_path)
        or payload.get("gate") != {
            "passed": True, "verdict": "TRAINING_AUTHORIZED"
        }
        or payload.get("training_authorized") is not True
        or payload.get("menagerie_authorized") is not False
        or payload.get("checks") != {
            "acquisition_qualified": True,
            "lineage_locality_feasible": True,
        }
    ):
        raise SystemExit("training authorization did not pass the frozen ancestor gates")
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    if payload.get("selected_answer_max_tokens") not in cfg["evaluation"][
        "interface_answer_rungs"
    ]:
        raise SystemExit("training authorization names an unregistered answer rung")
    expected_paths = {
        "qualification_gate": EXP / "analysis" / "qualification_gate.json",
        "lineage_locality_gate": EXP / "analysis" / "locality_start_vs_anchor.json",
    }
    receipts = payload.get("ancestor_receipts")
    if not isinstance(receipts, dict) or set(receipts) != set(expected_paths):
        raise SystemExit("training authorization has the wrong ancestor receipt set")
    ancestors = {}
    for name, expected_path in expected_paths.items():
        receipt = receipts[name]
        if (
            not isinstance(receipt, dict)
            or set(receipt) != {"path", "sha256"}
            or Path(receipt["path"]).resolve() != expected_path.resolve()
            or sha256_file(expected_path) != receipt["sha256"]
        ):
            raise SystemExit(f"training ancestor receipt changed: {name}")
        ancestors[name] = json.loads(expected_path.read_text())
        if ancestors[name].get("gate", {}).get("passed") is not True:
            raise SystemExit(f"training ancestor gate is not passed: {name}")
    qualification = ancestors["qualification_gate"]
    if (
        qualification.get("schema_version") != 1
        or qualification.get("stage")
        != "counterfactual_evidence_acquisition_qualification"
        or qualification.get("analyzer_sha256")
        != sha256_file(EXP / "scripts" / "analyze_qualification.py")
        or qualification.get("config_sha256")
        != sha256_file(EXP / "configs" / "default.yaml")
        or qualification.get("model") != cfg["model"]
        or qualification.get("training_authorized") is not True
        or qualification.get("gate") != {
            "passed": True, "verdict": "ACQUISITION_QUALIFIED"
        }
        or qualification.get("selected_answer_max_tokens")
        != payload["selected_answer_max_tokens"]
    ):
        raise SystemExit("qualification receipt does not authorize this training rung")
    if (
        set(qualification.get("blocks", {}))
        != {"qualification_a", "qualification_b"}
        or qualification.get("block_checks") != {
            "qualification_a": True, "qualification_b": True
        }
        or qualification.get("interface_checks") != {
            "invalid_actions": True,
            "answer_limit_contact": True,
            "content_disjoint": True,
        }
    ):
        raise SystemExit("qualification receipt omits a required passing block")
    interface_registration = qualification.get("interface_receipt")
    interface_path = EXP / "analysis" / "interface_answer_band.json"
    if (
        not isinstance(interface_registration, dict)
        or set(interface_registration) != {"path", "sha256"}
        or Path(interface_registration["path"]).resolve() != interface_path.resolve()
        or sha256_file(interface_path) != interface_registration["sha256"]
    ):
        raise SystemExit("qualification interface ancestor is stale")
    interface = json.loads(interface_path.read_text())
    if (
        interface.get("schema_version") != 1
        or interface.get("stage") != "interface_answer_band_selection"
        or interface.get("selector_sha256")
        != sha256_file(EXP / "scripts" / "select_interface_band.py")
        or interface.get("config_sha256")
        != sha256_file(EXP / "configs" / "default.yaml")
        or interface.get("selected_answer_max_tokens")
        != payload["selected_answer_max_tokens"]
        or interface.get("gate", {}).get("passed") is not True
        or interface.get("qualification_authorized") is not True
    ):
        raise SystemExit("qualification interface selection is not registered")
    registrations = qualification.get("input_receipts")
    answer_tokens = payload["selected_answer_max_tokens"]
    artifact_root = Path(cfg["artifacts"]["root"])
    artifact_root = (
        artifact_root if artifact_root.is_absolute() else ROOT / artifact_root
    )
    expected_inputs = {}
    for short, block in (("a", "qualification_a"), ("b", "qualification_b")):
        for label, contract, scenario in (
            ("unassisted", "inferred", "acquisition"),
            ("injected", "inferred", "injected"),
            ("control_search", "inferred", "random"),
            ("explicit", "explicit", "acquisition"),
        ):
            expected_inputs[f"{short}_{label}"] = artifact_root / "eval" / block / (
                f"start_{contract}_{scenario}_deep_a{answer_tokens}.json"
            )
    if not isinstance(registrations, dict) or set(registrations) != set(expected_inputs):
        raise SystemExit("qualification raw receipt set is incomplete")
    for name, expected_path in expected_inputs.items():
        registration = registrations[name]
        if (
            not isinstance(registration, dict)
            or set(registration) != {"path", "sha256"}
            or Path(registration["path"]).resolve() != expected_path.resolve()
            or not expected_path.is_file()
            or sha256_file(expected_path) != registration["sha256"]
        ):
            raise SystemExit(f"qualification raw receipt changed: {name}")
    locality = ancestors["lineage_locality_gate"]
    locality_contexts = Path(cfg["locality"]["contexts"])
    locality_contexts = (
        locality_contexts
        if locality_contexts.is_absolute()
        else ROOT / locality_contexts
    )
    if (
        locality.get("schema_version") != 1
        or locality.get("before_model_weight_sha256")
        != cfg["model"]["anchor_weight_sha256"]
        or locality.get("after_model_weight_sha256")
        != cfg["model"]["start_weight_sha256"]
        or Path(locality.get("before_model", "")).resolve()
        != (ROOT / cfg["model"]["locality_anchor"]).resolve()
        or Path(locality.get("after_model", "")).resolve()
        != (ROOT / cfg["model"]["start_checkpoint"]).resolve()
        or locality.get("auditor_sha256")
        != sha256_file(EXP / "scripts" / "audit_locality.py")
        or locality.get("contexts_sha256")
        != sha256_file(locality_contexts)
        or locality.get("tokenized_context_ids_equal") is not True
        or locality.get("rendered_prompts_equal") is not True
        or locality.get("before_tokenizer_compatibility_sha256")
        != cfg["model"]["tokenizer_compatibility_sha256"]
        or locality.get("after_tokenizer_compatibility_sha256")
        != cfg["model"]["tokenizer_compatibility_sha256"]
        or locality.get("before_tokenizer_manifest_sha256")
        != cfg["model"]["anchor_tokenizer_manifest_sha256"]
        or locality.get("after_tokenizer_manifest_sha256")
        != cfg["model"]["start_tokenizer_manifest_sha256"]
        or locality.get("before_rendered_prompts_sha256")
        != locality.get("after_rendered_prompts_sha256")
        or locality.get("before_tokenized_contexts_sha256")
        != locality.get("after_tokenized_contexts_sha256")
        or locality.get("ceiling")
        != float(cfg["locality"]["median_non_target_logit_drift_max"])
        or locality.get("entropy_delta_min")
        != float(cfg["locality"]["mean_entropy_delta_min"])
        or locality.get("checks") != {
            "finite": True,
            "context_count": True,
            "within_drift_ceiling": True,
            "entropy_retained": True,
        }
    ):
        raise SystemExit("lineage-locality ancestor provenance is invalid")
    return payload


def validate_start_checkpoint(path: Path, expected_weight_sha256: str) -> dict:
    cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
    try:
        checkpoint = validate_registered_checkpoint(
            EXP,
            path,
            cfg,
            EXP / "runs" / "preregistration_receipt.json",
            "start",
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"start checkpoint registration is invalid: {exc}") from exc
    if checkpoint["model_weight_sha256"] != expected_weight_sha256:
        raise SystemExit("start weight hash differs from the registered invocation")
    merge = json.loads((path / "merge_receipt.json").read_text())
    return {
        "path": str(path.resolve()),
        "config_sha256": checkpoint["model_config_sha256"],
        "generation_config_sha256": checkpoint["generation_config_sha256"],
        "merge_receipt_sha256": checkpoint["merge_receipt_sha256"],
        "weight_sha256": checkpoint["model_weight_sha256"],
        **{
            key: checkpoint[key]
            for key in (
                "tokenizer_files",
                "tokenizer_manifest_sha256",
                "tokenizer_compatibility_sha256",
            )
        },
        "recorded_weight_files": merge.get("weight_files", []),
        "model_lineage": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def encode_row(record: dict, tokenizer, max_length: int) -> dict | None:
    prompt = tokenizer.apply_chat_template(
        record["messages"], tokenize=False, add_generation_prompt=True,
        enable_thinking=True,
    )
    think_part = record["think"].strip() + "\n</think>\n\n"
    answer_part = record["answer"].strip() + tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    think_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_length:
        return None
    if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[:len(think_ids)] != think_ids:
        return None
    row_weight = float(record.get("row_weight", 1.0))
    think_weight = float(record.get("think_weight", 0.0))
    weights = (
        [0.0] * len(prompt_ids)
        + [think_weight * abs(row_weight)] * (len(think_ids) - len(prompt_ids))
        + [row_weight] * (len(full_ids) - len(think_ids))
    )
    labels = [token if weight != 0.0 else -100 for token, weight in zip(full_ids, weights)]
    answer_mask = [0.0] * len(think_ids) + [1.0] * (len(full_ids) - len(think_ids))
    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
        "loss_weights": weights,
        "answer_mask": answer_mask,
        "operator": record["operator"],
        "transition": record["transition"],
        "family": record["family"],
        "task_id": record["task_id"],
        "row_id": record["id"],
        "pair_id": record.get("pair_id"),
        "branch": record.get("branch"),
        "prompt_tokens": len(prompt_ids),
        "total_tokens": len(full_ids),
        "think_tokens": len(think_ids) - len(prompt_ids),
        "answer_tokens": len(full_ids) - len(think_ids),
        "weighted_plan_mass": (
            (len(think_ids) - len(prompt_ids)) * think_weight * abs(row_weight)
        ),
        "weighted_action_mass": (len(full_ids) - len(think_ids)) * abs(row_weight),
    }


def make_batches(
    encoded: list[dict], batch_size: int, gradient_accumulation_steps: int, seed: int
) -> tuple[list[dict], dict]:
    """Create optimizer steps containing one microbatch from every transition."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if gradient_accumulation_steps != len(TRANSITIONS):
        raise ValueError("registered accumulation must equal the nine transition strata")
    by_transition: dict[str, list[dict]] = defaultdict(list)
    for row in encoded:
        by_transition[str(row["transition"])].append(row)
    if set(by_transition) != set(TRANSITIONS):
        raise ValueError(f"missing transition strata: {set(TRANSITIONS) - set(by_transition)}")
    counts = {transition: len(rows) for transition, rows in by_transition.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"transition rows are not pre-balanced: {counts}")
    rows_per_transition = next(iter(counts.values()))
    if rows_per_transition % batch_size:
        raise ValueError("pre-balanced transition rows do not form full microbatches")
    rng = random.Random(seed)
    chunks_by_transition: dict[str, list[list[dict]]] = {}
    pair_cooccurrence_checks = 0
    for transition in TRANSITIONS:
        rows = list(by_transition[transition])
        paired: dict[str, list[dict]] = defaultdict(list)
        unpaired = []
        for row in rows:
            pair_id = row.get("pair_id")
            if pair_id:
                paired[str(pair_id)].append(row)
            else:
                unpaired.append(row)
        pair_rounds: dict[int, list[list[dict]]] = defaultdict(list)
        for pair_id, pair_rows in sorted(paired.items()):
            branches = {row.get("branch") for row in pair_rows}
            if len(pair_rows) % 2 or branches != {0, 1}:
                raise ValueError(f"unbalanced counterfactual rows: {pair_id}")
            branch_zero = sorted(
                (row for row in pair_rows if row.get("branch") == 0),
                key=lambda row: row["row_id"],
            )
            branch_one = sorted(
                (row for row in pair_rows if row.get("branch") == 1),
                key=lambda row: row["row_id"],
            )
            for copy_index, (a, b) in enumerate(zip(branch_zero, branch_one)):
                pair_rounds[copy_index].append([a, b])
        segments: list[list[dict]] = []
        for copy_index in sorted(pair_rounds):
            groups = pair_rounds[copy_index]
            rng.shuffle(groups)
            segment = [row for group in groups for row in group]
            if len(segment) % batch_size:
                raise ValueError(
                    f"paired copy round does not fill microbatches: "
                    f"{transition} copy={copy_index} rows={len(segment)}"
                )
            segments.append(segment)
            pair_cooccurrence_checks += len(groups)

        unpaired_by_task: dict[str, list[dict]] = defaultdict(list)
        for row in unpaired:
            unpaired_by_task[str(row["task_id"])].append(row)
        unpaired_rounds = max(
            (len(items) for items in unpaired_by_task.values()), default=0
        )
        for copy_index in range(unpaired_rounds):
            segment = [
                sorted(items, key=lambda row: row["row_id"])[copy_index]
                for _task_id, items in sorted(unpaired_by_task.items())
                if copy_index < len(items)
            ]
            rng.shuffle(segment)
            if len(segment) % batch_size:
                raise ValueError(
                    f"unpaired copy round does not fill microbatches: "
                    f"{transition} copy={copy_index} rows={len(segment)}"
                )
            segments.append(segment)
        ordered = [row for segment in segments for row in segment]
        if len(ordered) != rows_per_transition:
            raise AssertionError("transition ordering lost rows")
        transition_chunks = []
        for segment in segments:
            transition_chunks.extend(
                segment[index:index + batch_size]
                for index in range(0, len(segment), batch_size)
            )
        if any(len(chunk) != batch_size for chunk in transition_chunks):
            raise AssertionError("transition balancing did not form full microbatches")
        for chunk in transition_chunks:
            if len({row["task_id"] for row in chunk}) != len(chunk):
                raise AssertionError("a duplicated task entered one microbatch")
            pair_counts = Counter(
                str(row["pair_id"]) for row in chunk if row.get("pair_id")
            )
            if any(count != 2 for count in pair_counts.values()):
                raise AssertionError("counterfactual branches are not co-located exactly once")
            for pair_id, count in pair_counts.items():
                branches = {
                    row.get("branch") for row in chunk
                    if str(row.get("pair_id")) == pair_id
                }
                if count != 2 or branches != {0, 1}:
                    raise AssertionError("counterfactual branch pair is malformed")
        chunks_by_transition[transition] = transition_chunks
    supercycles = []
    chunks_per_transition = len(next(iter(chunks_by_transition.values())))
    for index in range(chunks_per_transition):
        cycle = [chunks_by_transition[transition][index] for transition in TRANSITIONS]
        rng.shuffle(cycle)
        supercycles.append(cycle)
    rng.shuffle(supercycles)
    chunks = [chunk for cycle in supercycles for chunk in cycle]
    if len(chunks) % gradient_accumulation_steps:
        raise AssertionError("transition supercycle is not optimizer-step aligned")
    flattened = [row for chunk in chunks for row in chunk]
    return flattened, {
        "original_tasks": len({row["task_id"] for row in encoded}),
        "transition_balance_padding_rows": sum(
            "transition-balance-pad" in row["row_id"] for row in encoded
        ),
        "effective_tasks_per_epoch": len({row["task_id"] for row in encoded}),
        "rows_per_epoch": len(flattened),
        "microbatches_per_epoch": len(chunks),
        "optimizer_steps_per_epoch": len(chunks) // gradient_accumulation_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": batch_size * gradient_accumulation_steps,
        "complete_transition_supercycle": True,
        "every_optimizer_step_contains_all_transitions": True,
        "counterfactual_pair_cooccurrence_checks": pair_cooccurrence_checks,
        "microbatch_unique_task_checks": len(chunks),
        "transition_exposures_per_epoch": {
            transition: len(by_transition[transition]) for transition in TRANSITIONS
        },
    }


def loss_mass_receipt(encoded: list[dict], microbatches_per_epoch: int) -> dict:
    """Return the realized fixed-denominator transition/operator allocation."""
    if microbatches_per_epoch < 1:
        raise ValueError("microbatches_per_epoch must be positive")
    total = sum(float(row["weighted_action_mass"]) for row in encoded)
    denominator = total / float(microbatches_per_epoch)
    if denominator <= 0:
        raise ValueError("registered fixed loss denominator is not positive")
    by_operator = {
        operator: sum(
            float(row["weighted_action_mass"]) for row in encoded
            if row["operator"] == operator
        )
        for operator in sorted({row["operator"] for row in encoded})
    }
    by_transition = {
        transition: sum(
            float(row["weighted_action_mass"]) for row in encoded
            if row["transition"] == transition
        )
        for transition in TRANSITIONS
    }
    return {
        "weighted_action_mass": total,
        "weighted_action_mass_by_operator": by_operator,
        "weighted_action_mass_by_transition": by_transition,
        "fixed_loss_denominator_per_microbatch": denominator,
        "effective_normalized_mass_by_operator": {
            key: value / denominator for key, value in by_operator.items()
        },
        "effective_normalized_mass_by_transition": {
            key: value / denominator for key, value in by_transition.items()
        },
    }


class WeightedDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        return self.rows[index]


class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        maximum = max(len(row["input_ids"]) for row in features)
        pad = self.tokenizer.pad_token_id
        return {
            "input_ids": torch.tensor([
                row["input_ids"] + [pad] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "attention_mask": torch.tensor([
                row["attention_mask"] + [0] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "labels": torch.tensor([
                row["labels"] + [-100] * (maximum - len(row["input_ids"]))
                for row in features
            ]),
            "loss_weights": torch.tensor([
                row["loss_weights"] + [0.0] * (maximum - len(row["input_ids"]))
                for row in features
            ], dtype=torch.float32),
            "answer_mask": torch.tensor([
                row["answer_mask"] + [0.0] * (maximum - len(row["input_ids"]))
                for row in features
            ], dtype=torch.float32),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm", choices=[
            "evidence_binding", "explicit_redundant", "shuffled_binding"
        ],
        required=True,
    )
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--expected-base-weight-sha256", required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--expected-train-sha256", required=True)
    parser.add_argument("--bank-receipt", type=Path, required=True)
    parser.add_argument(
        "--design-lock",
        type=Path,
        help="immutable preregistration receipt; required unless --encode-only",
    )
    parser.add_argument(
        "--training-authorization",
        type=Path,
        help="passed qualification+lineage receipt; required unless --encode-only",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=9)
    parser.add_argument("--loss-chunk-positions", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--encode-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1:
        raise SystemExit("epochs must be positive")
    base_receipt = validate_start_checkpoint(
        args.base_model, args.expected_base_weight_sha256
    )
    observed_train_sha256 = sha256_file(args.train)
    if observed_train_sha256 != args.expected_train_sha256:
        raise SystemExit(
            f"training bank hash mismatch: {observed_train_sha256}"
        )
    bank_receipt = json.loads(args.bank_receipt.read_text())
    if bank_receipt.get("bank_sha256", {}).get(args.arm) != observed_train_sha256:
        raise SystemExit("bank receipt does not register the selected training bank")
    design_lock = None
    training_authorization = None
    if not args.encode_only:
        if args.design_lock is None or args.training_authorization is None:
            raise SystemExit(
                "--design-lock and --training-authorization are required for real training"
            )
        design_lock = validate_design_lock(
            args.design_lock,
            bank_receipt_path=args.bank_receipt,
            arm=args.arm,
            train_sha256=observed_train_sha256,
        )
        validate_training_contract(args, design_lock, bank_receipt)
        training_authorization = validate_training_authorization(
            args.training_authorization, args.design_lock
        )
    rows = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    if not rows or any(row.get("kind") != f"repo_{args.arm}" for row in rows):
        raise SystemExit(f"{args.train} is not a pure {args.arm} bank")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=True, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    probe = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}], tokenize=False,
        add_generation_prompt=True, enable_thinking=True,
    )
    if not probe.endswith("<think>\n"):
        raise SystemExit(f"unexpected thinking template tail: {probe[-40:]!r}")

    encoded = [encode_row(record, tokenizer, args.max_length) for record in rows]
    if any(row is None for row in encoded):
        raise SystemExit("a registered row truncates or merges at a token boundary")
    encoded = [row for row in encoded if row is not None]
    ordered, batch_receipt = make_batches(
        encoded, args.batch_size, args.grad_accum, args.seed
    )
    mass_receipt = loss_mass_receipt(
        encoded, int(batch_receipt["microbatches_per_epoch"])
    )
    fixed_loss_denominator = mass_receipt["fixed_loss_denominator_per_microbatch"]
    weighted_mass_by_operator = mass_receipt["weighted_action_mass_by_operator"]
    weighted_mass_by_transition = mass_receipt["weighted_action_mass_by_transition"]
    expected_operator_mass = bank_receipt["weighted_action_mass_by_operator"][args.arm]
    for operator, expected in expected_operator_mass.items():
        if abs(weighted_mass_by_operator.get(operator, 0.0) - float(expected)) > 1e-5:
            raise SystemExit(
                f"encoded operator mass disagrees with bank receipt: {operator}"
            )
    expected_transition_mass = bank_receipt[
        "weighted_action_mass_by_transition"
    ][args.arm]
    for transition, expected in expected_transition_mass.items():
        if abs(weighted_mass_by_transition.get(transition, 0.0) - float(expected)) > 1e-5:
            raise SystemExit(
                f"encoded transition mass disagrees with bank receipt: {transition}"
            )
    if len({round(value, 6) for value in weighted_mass_by_transition.values()}) != 1:
        raise SystemExit("conditional transition loss mass is not exactly balanced")
    if sum(row["weighted_plan_mass"] for row in encoded) != 0.0:
        raise SystemExit("zero-think-loss registration was violated during encoding")
    padded_tokens_per_epoch = sum(
        max(len(row["input_ids"]) for row in ordered[index:index + args.batch_size])
        * args.batch_size
        for index in range(0, len(ordered), args.batch_size)
    )
    steps_per_epoch = int(batch_receipt["optimizer_steps_per_epoch"])
    max_steps = steps_per_epoch * args.epochs
    schedule_sha256 = hashlib.sha256(
        json.dumps(
            [row["row_id"] for row in ordered], separators=(",", ":")
        ).encode()
    ).hexdigest()
    encoding_receipt = {
        "schema_version": 1,
        "arm": args.arm,
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "start_checkpoint": base_receipt,
        "training_file": {
            "path": str(args.train.resolve()),
            "sha256": observed_train_sha256,
            "rows": len(rows),
        },
        "bank_receipt_sha256": sha256_file(args.bank_receipt),
        "design_lock_sha256": (
            sha256_file(args.design_lock) if design_lock is not None else None
        ),
        "design_commit": (
            design_lock.get("design_commit") if design_lock is not None else None
        ),
        "training_authorization_sha256": (
            sha256_file(args.training_authorization)
            if training_authorization is not None else None
        ),
        "selected_answer_max_tokens": (
            training_authorization.get("selected_answer_max_tokens")
            if training_authorization is not None else None
        ),
        "encoded_rows": len(encoded),
        "max_encoded_tokens": max(len(row["input_ids"]) for row in encoded),
        "prompt_tokens": sum(row["prompt_tokens"] for row in encoded),
        "total_unpadded_tokens": sum(row["total_tokens"] for row in encoded),
        "serial_forward_tokens_per_epoch": sum(
            len(row["input_ids"]) for row in encoded
        ),
        "dynamically_padded_tokens_per_epoch": padded_tokens_per_epoch,
        "think_tokens": sum(row["think_tokens"] for row in encoded),
        "answer_tokens": sum(row["answer_tokens"] for row in encoded),
        "weighted_plan_mass": sum(row["weighted_plan_mass"] for row in encoded),
        **mass_receipt,
        "operator_rows": dict(Counter(row["operator"] for row in encoded)),
        "transition_rows": dict(Counter(row["transition"] for row in encoded)),
        "epochs": args.epochs,
        "max_steps": max_steps,
        "ordered_schedule_sha256": schedule_sha256,
        **batch_receipt,
    }
    if design_lock is not None:
        encoding_receipt["preflight_encoding_receipt"] = validate_preflight_encoding(
            args.arm, encoding_receipt, design_lock
        )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "encoding_receipt.json").write_text(
        json.dumps(encoding_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(encoding_receipt, indent=2), flush=True)
    if args.encode_only:
        return 0
    if args.smoke:
        needed = args.batch_size * args.grad_accum * 2
        ordered = (ordered * ((needed + len(ordered) - 1) // len(ordered)))[:needed]
        max_steps = 2

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=True,
        device_map="cuda",
        dtype=torch.bfloat16,
        quantization_config=quantization,
        attn_implementation="sdpa",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lora_module_names = sorted(
        name for name, module in model.named_modules()
        if hasattr(module, "lora_A") and getattr(module, "lora_A")
    )
    if not lora_module_names or trainable_parameter_count <= 0:
        raise SystemExit("LoRA trainable-module geometry is empty")

    training_args = TrainingArguments(
        output_dir=str(args.out),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=max(1, min(10, max_steps // 4)),
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        seed=args.seed,
        data_seed=args.seed,
        remove_unused_columns=False,
    )

    class TransitionBalancedTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            from torch.utils.data import SequentialSampler
            return SequentialSampler(self.train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            del kwargs
            weights = inputs.pop("loss_weights")
            answer_mask = inputs.pop("answer_mask")
            labels = inputs.pop("labels")
            attention = inputs["attention_mask"]
            numerator = None
            outputs = None
            # Qwen3.5's hybrid GDN/Mamba blocks are not logit-equivalent under
            # right padding (C52 measured 0.30--0.44).  Keep the logical batch
            # and its dyad/supercycle schedule, but execute every row as an
            # unpadded physical batch-of-one forward and sum its graph.
            physical_lengths = unpadded_row_lengths(attention)
            for row_index, length in enumerate(physical_lengths):
                if length < 2:
                    raise RuntimeError("training row has fewer than two real tokens")
                single_inputs = {
                    key: value[row_index:row_index + 1, :length]
                    for key, value in inputs.items()
                }
                outputs = model(**single_inputs)
                logits = outputs.logits[:, :-1, :]
                shifted_labels = labels[row_index:row_index + 1, 1:length].contiguous()
                shifted_weights = weights[row_index:row_index + 1, 1:length].contiguous()
                shifted_answer_mask = answer_mask[
                    row_index:row_index + 1, 1:length
                ].contiguous()
                signed = (shifted_labels != -100).float() * shifted_weights
                row_numerator = checkpointed_weighted_cross_entropy(
                    logits, shifted_labels, signed, args.loss_chunk_positions
                )
                numerator = (
                    row_numerator if numerator is None else numerator + row_numerator
                )
            if numerator is None or outputs is None:
                raise RuntimeError("empty logical training microbatch")
            # All arms use the same registered action-token denominator; plan
            # weights are identically zero. A single bank-wide denominator
            # preserves the calibrated operator
            # mass across transition-homogeneous microbatches.  Per-microbatch
            # token means would double-normalize short VERIFY/COMMIT actions.
            loss = numerator / fixed_loss_denominator
            return (loss, outputs) if return_outputs else loss

    started = time.perf_counter()
    result = TransitionBalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=WeightedDataset(ordered),
        data_collator=Collator(tokenizer),
    ).train()
    if int(result.global_step) != int(max_steps):
        raise RuntimeError(
            f"trainer stopped at {int(result.global_step)} of {int(max_steps)} steps"
        )
    model.save_pretrained(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    receipt = {
        **encoding_receipt,
        "method": "warm_start_transition_balanced_qlora",
        "adapter_path": str(args.out.resolve()),
        "adapter_config_sha256": sha256_file(args.out / "adapter_config.json"),
        "adapter_weights_sha256": sha256_file(
            args.out / "adapter_model.safetensors"
        ),
        "learning_rate": args.lr,
        "rank": args.rank,
        "alpha": args.alpha,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps_actual": args.grad_accum,
        "seed": args.seed,
        "smoke": bool(args.smoke),
        "optimizer_steps": int(result.global_step),
        "training_loss": float(result.training_loss),
        "wall_seconds": time.perf_counter() - started,
        "gpu": torch.cuda.get_device_name(0),
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "trainable_parameter_count": trainable_parameter_count,
        "total_parameter_count_loaded": total_parameter_count,
        "lora_module_count": len(lora_module_names),
        "lora_module_names_sha256": hashlib.sha256(
            json.dumps(lora_module_names, separators=(",", ":")).encode()
        ).hexdigest(),
        "normalization": "fixed bank-wide weighted action mass per microbatch",
        "physical_forward_batch_size": 1,
        "logical_microbatch_size": args.batch_size,
        "padding_policy": "logical right-padding is sliced away before four serial unpadded forwards",
        "loss_chunk_positions": args.loss_chunk_positions,
        "loss_implementation": "exact_sequence_chunked_checkpointed_cross_entropy",
    }
    (args.out / "training_receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[train] saved {args.arm} adapter to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
