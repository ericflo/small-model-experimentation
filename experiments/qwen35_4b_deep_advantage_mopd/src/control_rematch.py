"""Pure selection, hashing, and provenance checks for control-only overlays."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from io_utils import (
    canonical_hash,
    resolve_repo_path,
    sha256_file,
    validate_policy_cache_provenance,
)
from route_control_matching import MATCH_TIERS, matched_non_advantage_route_units


REMATCH_STAGE = "full_prefix_non_advantage_route_rematch"
CACHE_VARIANT = "full_prefix_non_advantage_route_overlay"


def _ordered_controls(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda row: str(row["state_id"]),
    )


def control_mapping(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return the compact canonical identity needed to audit a rematch."""

    return sorted(
        (
            {
                "state_id": str(row["state_id"]),
                "matched_primary_state_id": str(row["matched_primary_state_id"]),
                "match_tier": str(row["match_tier"]),
                "observed_route": str(row["observed_route"]),
                "family": str(row["family"]),
                "kind": str(row["kind"]),
                "level": int(row["level"]),
            }
            for row in rows
        ),
        key=lambda row: row["matched_primary_state_id"],
    )


def control_mapping_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(control_mapping(rows))


def rematch_full_prefix_controls(
    *,
    capability_units: Sequence[Mapping[str, Any]],
    original_controls: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    eligible_state_ids: set[str],
    match_order: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Replay the frozen matcher after a length-only eligibility filter.

    The unfiltered replay must exactly reproduce the frozen controls.  The
    filtered replay is then permitted to change only controls that failed the
    full-prefix predicate.  A greedy cascade into any otherwise legal control
    is a hard error rather than an unregistered control redesign.
    """

    order = tuple(str(value) for value in match_order)
    if order != MATCH_TIERS:
        raise ValueError(f"invalid frozen match order: {order}")
    candidate_ids = [str(row["state_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate controls must have unique state IDs")
    if not eligible_state_ids <= set(candidate_ids):
        raise ValueError("eligible-state inventory is not a subset of candidates")

    replayed = matched_non_advantage_route_units(
        capability_units, candidates, order
    )
    if control_mapping_sha256(replayed) != control_mapping_sha256(original_controls):
        raise ValueError("unfiltered control replay does not reproduce frozen manifest")

    filtered = [
        row for row in candidates if str(row["state_id"]) in eligible_state_ids
    ]
    rematched = matched_non_advantage_route_units(
        capability_units, filtered, order
    )
    original_by_primary = {
        str(row["matched_primary_state_id"]): row for row in original_controls
    }
    rematched_by_primary = {
        str(row["matched_primary_state_id"]): row for row in rematched
    }
    if set(original_by_primary) != set(rematched_by_primary):
        raise ValueError("rematch changed the primary-state inventory")
    offender_primary_ids = {
        str(row["matched_primary_state_id"])
        for row in original_controls
        if str(row["state_id"]) not in eligible_state_ids
    }
    changed_primary_ids = {
        primary_id
        for primary_id in original_by_primary
        if str(original_by_primary[primary_id]["state_id"])
        != str(rematched_by_primary[primary_id]["state_id"])
    }
    if changed_primary_ids != offender_primary_ids:
        raise ValueError(
            "full-prefix filter changed nonoffending control matches: "
            f"changed={sorted(changed_primary_ids)} "
            f"offenders={sorted(offender_primary_ids)}"
        )
    tier_changed_primary_ids = {
        primary_id
        for primary_id in changed_primary_ids
        if str(original_by_primary[primary_id]["match_tier"])
        != str(rematched_by_primary[primary_id]["match_tier"])
    }
    if tier_changed_primary_ids:
        raise ValueError(
            "full-prefix rematch changed offender match tiers: "
            f"{sorted(tier_changed_primary_ids)}"
        )
    if any(
        str(row["state_id"]) not in eligible_state_ids for row in rematched
    ):
        raise ValueError("rematch retained an ineligible control")

    original_tier_counts = {
        tier: sum(str(row["match_tier"]) == tier for row in original_controls)
        for tier in order
    }
    rematched_tier_counts = {
        tier: sum(str(row["match_tier"]) == tier for row in rematched)
        for tier in order
    }
    if rematched_tier_counts != original_tier_counts:
        raise ValueError(
            "full-prefix rematch changed aggregate match-tier geometry: "
            f"original={original_tier_counts} rematched={rematched_tier_counts}"
        )

    audit = {
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(filtered),
        "ineligible_candidate_count": len(candidates) - len(filtered),
        "original_control_count": len(original_controls),
        "rematched_control_count": len(rematched),
        "offender_primary_state_ids": sorted(offender_primary_ids),
        "changed_primary_state_ids": sorted(changed_primary_ids),
        "unchanged_control_count": len(original_controls) - len(changed_primary_ids),
        "original_mapping_sha256": control_mapping_sha256(original_controls),
        "rematched_mapping_sha256": control_mapping_sha256(rematched),
        "original_match_tier_counts": original_tier_counts,
        "match_tier_counts": rematched_tier_counts,
    }
    return _ordered_controls(rematched), audit


def prompt_truncation_summary(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize completion-preserving prompt cuts, including role identity."""

    truncated = [
        sample
        for sample in samples
        if int(sample["meta"].get("prompt_tokens_truncated", 0)) > 0
    ]

    def role_summary(role: str) -> dict[str, Any]:
        selected = [sample for sample in truncated if sample["meta"]["role"] == role]
        values = [int(sample["meta"]["prompt_tokens_truncated"]) for sample in selected]
        return {
            "sample_count": len(selected),
            "total_tokens": sum(values),
            "maximum_tokens": max(values, default=0),
            "state_ids_sha256": hashlib.sha256(
                "\n".join(sorted(str(sample["id"]) for sample in selected)).encode()
            ).hexdigest(),
        }

    values = [int(sample["meta"]["prompt_tokens_truncated"]) for sample in truncated]
    return {
        "sample_count": len(truncated),
        "total_tokens": sum(values),
        "maximum_tokens": max(values, default=0),
        "state_ids_sha256": hashlib.sha256(
            "\n".join(sorted(str(sample["id"]) for sample in truncated)).encode()
        ).hexdigest(),
        "by_role": {
            role: role_summary(role)
            for role in ("capability", "anchor", "route_control")
        },
    }


def _semantic_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.view(torch.uint8).numpy().tobytes()
        return {
            "tensor_dtype": str(tensor.dtype),
            "tensor_shape": list(tensor.shape),
            "tensor_sha256": hashlib.sha256(raw).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _semantic_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_semantic_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported semantic-hash value: {type(value).__name__}")


def samples_semantic_sha256(samples: Sequence[Mapping[str, Any]]) -> str:
    return canonical_hash(_semantic_value(list(samples)))


def validate_control_rematch_manifest(
    path: Path,
    *,
    config: Mapping[str, Any],
    config_path: Path,
    source_manifest: Path,
    source_cache: Path,
) -> dict[str, Any]:
    """Validate a control-rematch selection receipt and all frozen inputs."""

    if not path.is_file():
        raise ValueError(f"control-rematch manifest missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if payload.get("stage") != REMATCH_STAGE:
        errors.append("stage")
    if payload.get("config_sha256") != sha256_file(config_path):
        errors.append("config_sha256")
    source_manifest_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
    if source_manifest_payload.get("stage") != "online_advantage_training_round":
        errors.append("source.round_manifest.stage")
    if source_manifest_payload.get("config_sha256") != sha256_file(config_path):
        errors.append("source.round_manifest.config_sha256")
    source = payload.get("source") or {}
    expected_sources = {
        "round_manifest": source_manifest.resolve(),
        "target_cache": source_cache.resolve(),
    }
    for key, expected_path in expected_sources.items():
        record = source.get(key) or {}
        if Path(record.get("path", "/")).resolve() != expected_path:
            errors.append(f"source.{key}.path")
        elif record.get("sha256") != sha256_file(expected_path):
            errors.append(f"source.{key}.sha256")
    source_receipt = source_cache.with_suffix(source_cache.suffix + ".receipt.json")
    receipt_record = source.get("target_cache_receipt") or {}
    if (
        Path(receipt_record.get("path", "/")).resolve() != source_receipt.resolve()
        or not source_receipt.is_file()
        or receipt_record.get("sha256") != sha256_file(source_receipt)
    ):
        errors.append("source.target_cache_receipt")
    if tuple(payload.get("match_order") or ()) != MATCH_TIERS:
        errors.append("match_order")
    if int(payload.get("round", -1)) != int(source_manifest_payload.get("round", -2)):
        errors.append("round")
    if int(payload.get("max_length", -1)) != int(config["mopd"]["max_length"]):
        errors.append("max_length")
    audit = payload.get("audit") or {}
    if audit.get("offender_primary_state_ids") != audit.get("changed_primary_state_ids"):
        errors.append("audit.changed_primary_state_ids")
    if int(audit.get("original_control_count", -1)) != int(
        config["mopd"]["capability_units_per_round"]
    ):
        errors.append("audit.original_control_count")
    if int(audit.get("rematched_control_count", -1)) != int(
        config["mopd"]["capability_units_per_round"]
    ):
        errors.append("audit.rematched_control_count")
    if int(payload.get("replacement_count", -1)) != len(
        audit.get("changed_primary_state_ids") or []
    ):
        errors.append("replacement_count")
    if int(audit.get("candidate_count", -1)) != (
        int(audit.get("eligible_candidate_count", -2))
        + int(audit.get("ineligible_candidate_count", -3))
    ):
        errors.append("audit.candidate_count")
    candidate_counts = source_manifest_payload.get("candidate_counts") or {}
    expected_candidate_count = int(candidate_counts.get("quick_routed", -1)) + int(
        candidate_counts.get("abstained", -1)
    )
    if int(audit.get("candidate_count", -2)) != expected_candidate_count:
        errors.append("audit.candidate_count.source_manifest")
    if int(audit.get("unchanged_control_count", -1)) + len(
        audit.get("changed_primary_state_ids") or []
    ) != int(audit.get("original_control_count", -2)):
        errors.append("audit.unchanged_control_count")
    observed_original_tier_counts = audit.get("original_match_tier_counts") or {}
    observed_rematched_tier_counts = audit.get("match_tier_counts") or {}
    if set(observed_original_tier_counts) != set(MATCH_TIERS):
        errors.append("audit.original_match_tier_counts.inventory")
    if set(observed_rematched_tier_counts) != set(MATCH_TIERS):
        errors.append("audit.match_tier_counts.inventory")
    if sum(int(value) for value in observed_original_tier_counts.values()) != int(
        audit.get("original_control_count", -1)
    ):
        errors.append("audit.original_match_tier_counts")
    if sum(int(value) for value in observed_rematched_tier_counts.values()) != int(
        audit.get("rematched_control_count", -1)
    ):
        errors.append("audit.match_tier_counts")
    if observed_original_tier_counts != observed_rematched_tier_counts:
        errors.append("audit.match_tier_geometry")
    original_mapping = payload.get("original_control_mapping") or []
    rematched_mapping = payload.get("rematched_control_mapping") or []
    expected_original_mapping = control_mapping(
        source_manifest_payload.get("control_units") or []
    )
    if original_mapping != expected_original_mapping:
        errors.append("original_control_mapping.source_manifest")
    if canonical_hash(original_mapping) != audit.get("original_mapping_sha256"):
        errors.append("original_control_mapping")
    if canonical_hash(rematched_mapping) != audit.get("rematched_mapping_sha256"):
        errors.append("rematched_control_mapping")
    original_primary_ids = [
        str(row.get("matched_primary_state_id")) for row in original_mapping
    ]
    rematched_primary_ids = [
        str(row.get("matched_primary_state_id")) for row in rematched_mapping
    ]
    original_state_ids = [str(row.get("state_id")) for row in original_mapping]
    rematched_state_ids = [str(row.get("state_id")) for row in rematched_mapping]
    if (
        len(original_mapping) != int(audit.get("original_control_count", -1))
        or len(original_primary_ids) != len(set(original_primary_ids))
        or len(original_state_ids) != len(set(original_state_ids))
    ):
        errors.append("original_control_mapping.inventory")
    if (
        len(rematched_mapping) != int(audit.get("rematched_control_count", -1))
        or len(rematched_primary_ids) != len(set(rematched_primary_ids))
        or len(rematched_state_ids) != len(set(rematched_state_ids))
        or set(rematched_primary_ids) != set(original_primary_ids)
    ):
        errors.append("rematched_control_mapping.inventory")
    replacements = payload.get("replacements") or []
    if len(replacements) != int(payload.get("replacement_count", -1)):
        errors.append("replacements")
    original_by_primary = {
        str(row.get("matched_primary_state_id")): row for row in original_mapping
    }
    rematched_by_primary = {
        str(row.get("matched_primary_state_id")): row for row in rematched_mapping
    }
    actual_changed_primary_ids = {
        primary_id
        for primary_id in original_by_primary
        if primary_id in rematched_by_primary
        and original_by_primary[primary_id].get("state_id")
        != rematched_by_primary[primary_id].get("state_id")
    }
    audit_changed_primary_ids = {
        str(value) for value in audit.get("changed_primary_state_ids") or []
    }
    audit_offender_primary_ids = {
        str(value) for value in audit.get("offender_primary_state_ids") or []
    }
    replacement_primary_ids = [
        str(row.get("matched_primary_state_id")) for row in replacements
    ]
    if (
        len(replacement_primary_ids) != len(set(replacement_primary_ids))
        or set(replacement_primary_ids) != actual_changed_primary_ids
        or audit_changed_primary_ids != actual_changed_primary_ids
        or audit_offender_primary_ids != actual_changed_primary_ids
    ):
        errors.append("replacement_primary_inventory")
    for replacement in replacements:
        primary_id = str(replacement.get("matched_primary_state_id"))
        if (
            primary_id not in original_by_primary
            or primary_id not in rematched_by_primary
            or replacement.get("old_state_id")
            != original_by_primary[primary_id].get("state_id")
            or replacement.get("new_state_id")
            != rematched_by_primary[primary_id].get("state_id")
            or int((replacement.get("old_fit") or {}).get("prompt_tokens_truncated", 0))
            <= 0
            or int((replacement.get("new_fit") or {}).get("prompt_tokens_truncated", -1))
            != 0
            or original_by_primary[primary_id].get("match_tier")
            != rematched_by_primary[primary_id].get("match_tier")
            or replacement.get("old_match_tier")
            != original_by_primary[primary_id].get("match_tier")
            or replacement.get("new_match_tier")
            != rematched_by_primary[primary_id].get("match_tier")
        ):
            errors.append("replacements")
            break
    actual_original_tier_counts = {
        tier: sum(str(row.get("match_tier")) == tier for row in original_mapping)
        for tier in MATCH_TIERS
    }
    actual_rematched_tier_counts = {
        tier: sum(str(row.get("match_tier")) == tier for row in rematched_mapping)
        for tier in MATCH_TIERS
    }
    if actual_original_tier_counts != observed_original_tier_counts:
        errors.append("audit.original_match_tier_counts.mapping")
    if actual_rematched_tier_counts != observed_rematched_tier_counts:
        errors.append("audit.match_tier_counts.mapping")
    try:
        source_payload = torch.load(
            source_cache, map_location="cpu", weights_only=False
        )
        source_samples = list(source_payload["samples"])
    except Exception as error:  # noqa: BLE001 - convert corrupt caches to evidence
        errors.append(f"source.target_cache.payload:{type(error).__name__}")
        source_samples = []
    source_semantic_sha = (payload.get("source") or {}).get(
        "samples_semantic_sha256"
    )
    if (
        not source_semantic_sha
        or samples_semantic_sha256(source_samples) != source_semantic_sha
    ):
        errors.append("source.samples_semantic_sha256")
    primary_by_control_state = {
        str(row.get("state_id")): str(row.get("matched_primary_state_id"))
        for row in original_mapping
    }
    truncated_control_primary_ids = {
        primary_by_control_state[str(sample.get("id"))]
        for sample in source_samples
        if str((sample.get("meta") or {}).get("role")) == "route_control"
        and int((sample.get("meta") or {}).get("prompt_tokens_truncated", 0)) > 0
        and str(sample.get("id")) in primary_by_control_state
    }
    if truncated_control_primary_ids != actual_changed_primary_ids:
        errors.append("source.truncated_control_inventory")
    if not payload.get("gate", {}).get("passed"):
        errors.append("gate")
    expected_artifacts = []
    for batch in source_manifest_payload.get("artifacts") or []:
        records = [
            ("states", batch["states"]),
            ("anchors", batch["anchors"]),
        ]
        records.extend(
            (f"branches.{policy}", batch["branches"][policy])
            for policy in ("quick", "deep", "student")
        )
        expected_artifacts.extend(
            {
                "batch": int(batch["batch"]),
                "role": role,
                "path": str(Path(record["path"]).resolve()),
                "sha256": str(record["sha256"]),
            }
            for role, record in records
        )
    observed_artifacts = payload.get("candidate_artifacts") or []
    if canonical_hash(
        sorted(expected_artifacts, key=lambda row: (row["batch"], row["role"]))
    ) != canonical_hash(observed_artifacts):
        errors.append("candidate_artifacts.inventory")
    for record in observed_artifacts:
        artifact = Path(record.get("path", "/"))
        if not artifact.is_file() or record.get("sha256") != sha256_file(artifact):
            errors.append("candidate_artifacts")
            break
    tokenizer = payload.get("tokenizer") or {}
    tokenizer_root = resolve_repo_path(config["model"]["student_checkpoint"]).resolve()
    expected_tokenizer_files = [
        tokenizer_root / filename
        for filename in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    ]
    if Path(tokenizer.get("path", "/")).resolve() != tokenizer_root:
        errors.append("tokenizer.path")
    tokenizer_files = tokenizer.get("files") or []
    if {
        Path(record.get("path", "/")).resolve() for record in tokenizer_files
    } != set(expected_tokenizer_files):
        errors.append("tokenizer.inventory")
    for record in tokenizer_files:
        artifact = Path(record.get("path", "/"))
        if not artifact.is_file() or record.get("sha256") != sha256_file(artifact):
            errors.append("tokenizer.files")
            break
    if errors:
        raise ValueError(
            "control-rematch provenance mismatch: " + ", ".join(sorted(set(errors)))
        )
    return payload


def validate_control_overlay_cache(
    cache: Path,
    *,
    rematch_manifest: Path,
    source_cache: Path,
    config: Mapping[str, Any],
    config_path: Path,
) -> dict[str, Any]:
    """Validate the derived cache receipt and its transitive source binding."""

    receipt_path = cache.with_suffix(cache.suffix + ".receipt.json")
    if not cache.is_file() or not receipt_path.is_file():
        raise ValueError(f"partial control overlay cache: {cache}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_policy_cache_provenance(receipt, config, config_path)
    errors = []
    if receipt.get("cache_variant") != CACHE_VARIANT:
        errors.append("cache_variant")
    if receipt.get("cache_sha256") != sha256_file(cache):
        errors.append("cache_sha256")
    if receipt.get("source_target_cache_sha256") != sha256_file(source_cache):
        errors.append("source_target_cache_sha256")
    if Path(receipt.get("source_target_cache", "/")).resolve() != source_cache.resolve():
        errors.append("source_target_cache")
    source_receipt = source_cache.with_suffix(source_cache.suffix + ".receipt.json")
    if (
        not source_receipt.is_file()
        or receipt.get("source_target_cache_receipt_sha256")
        != sha256_file(source_receipt)
    ):
        errors.append("source_target_cache_receipt_sha256")
    if receipt.get("control_rematch_manifest_sha256") != sha256_file(rematch_manifest):
        errors.append("control_rematch_manifest_sha256")
    if (
        Path(receipt.get("control_rematch_manifest", "/")).resolve()
        != rematch_manifest.resolve()
    ):
        errors.append("control_rematch_manifest")
    if int(receipt.get("sample_count", -1)) != (
        int(config["mopd"]["updates_per_round"])
        * int(config["mopd"]["grad_accum"])
        + int(config["mopd"]["capability_units_per_round"])
    ):
        errors.append("sample_count")
    if int((receipt.get("prompt_truncation") or {}).get("sample_count", -1)) != 0:
        errors.append("prompt_truncation")
    rematch = json.loads(rematch_manifest.read_text(encoding="utf-8"))
    replacements = int(receipt.get("replacement_count", -1))
    if replacements != int(rematch.get("replacement_count", -2)):
        errors.append("replacement_count")
    if int(receipt.get("copied_sample_count", -1)) != int(
        receipt.get("sample_count", -2)
    ) - replacements:
        errors.append("copied_sample_count")
    source_manifest = (rematch.get("source") or {}).get("round_manifest") or {}
    if receipt.get("round_manifest_sha256") != source_manifest.get("sha256"):
        errors.append("round_manifest_sha256")
    for field in (
        "copied_samples_semantic_sha256",
        "replacement_samples_semantic_sha256",
        "replacement_state_ids_sha256",
    ):
        value = str(receipt.get(field, ""))
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            errors.append(field)
    try:
        derived_payload = torch.load(cache, map_location="cpu", weights_only=False)
        source_payload = torch.load(
            source_cache, map_location="cpu", weights_only=False
        )
        if not isinstance(derived_payload, Mapping) or not isinstance(
            source_payload, Mapping
        ):
            raise TypeError("cache payload is not a mapping")
        derived_samples = list(derived_payload["samples"])
        source_samples = list(source_payload["samples"])
    except Exception as error:  # noqa: BLE001 - convert corrupt caches to evidence
        errors.append(f"serialized_payload:{type(error).__name__}")
        derived_payload = {}
        source_payload = {}
        derived_samples = []
        source_samples = []
    else:
        try:
            validate_policy_cache_provenance(
                derived_payload, config, config_path
            )
            validate_policy_cache_provenance(source_payload, config, config_path)
        except ValueError:
            errors.append("serialized_payload.policy_provenance")
        try:
            source_receipt_payload = json.loads(
                source_receipt.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"source_receipt.payload:{type(error).__name__}")
            source_receipt_payload = {}
        for prefix, metadata, recorded in (
            ("derived", derived_payload, receipt),
            ("source", source_payload, source_receipt_payload),
        ):
            for key, value in metadata.items():
                if key != "samples" and recorded.get(key) != value:
                    errors.append(f"{prefix}_payload.{key}")
        if source_payload.get("models") != source_receipt_payload.get("models"):
            errors.append("source_payload.models")
        if derived_payload.get("models") != receipt.get("models"):
            errors.append("derived_payload.models")

        replacement_rows = list(rematch.get("replacements") or [])
        replacement_indices = [
            int(row.get("source_cache_index", -1)) for row in replacement_rows
        ]
        if (
            len(replacement_indices) != replacements
            or len(replacement_indices) != len(set(replacement_indices))
            or any(index < 0 or index >= len(source_samples) for index in replacement_indices)
            or len(derived_samples) != len(source_samples)
        ):
            errors.append("serialized_payload.replacement_indices")
            replacement_indices = []
        replacement_index_set = set(replacement_indices)
        if replacement_indices:
            for row, index in zip(replacement_rows, replacement_indices):
                source_sample = source_samples[index]
                derived_sample = derived_samples[index]
                expected_mapping = next(
                    (
                        mapping
                        for mapping in rematch.get("rematched_control_mapping") or []
                        if str(mapping.get("matched_primary_state_id"))
                        == str(row.get("matched_primary_state_id"))
                    ),
                    {},
                )
                if (
                    str(source_sample.get("id")) != str(row.get("old_state_id"))
                    or str(derived_sample.get("id")) != str(row.get("new_state_id"))
                    or str((derived_sample.get("meta") or {}).get("role"))
                    != "route_control"
                    or str(
                        (derived_sample.get("meta") or {}).get(
                            "matched_primary_state_id"
                        )
                    )
                    != str(row.get("matched_primary_state_id"))
                    or str((derived_sample.get("meta") or {}).get("match_tier"))
                    != str(expected_mapping.get("match_tier"))
                ):
                    errors.append("serialized_payload.replacement_identity")
                    break

        copied_source = [
            sample
            for index, sample in enumerate(source_samples)
            if index not in replacement_index_set
        ]
        copied_derived = [
            sample
            for index, sample in enumerate(derived_samples)
            if index not in replacement_index_set
        ]
        copied_sha = samples_semantic_sha256(copied_source)
        if (
            copied_sha != receipt.get("copied_samples_semantic_sha256")
            or samples_semantic_sha256(copied_derived) != copied_sha
        ):
            errors.append("serialized_payload.copied_samples_semantic_sha256")
        derived_replacements = [
            derived_samples[index]
            for index in replacement_indices
            if 0 <= index < len(derived_samples)
        ]
        if samples_semantic_sha256(derived_replacements) != receipt.get(
            "replacement_samples_semantic_sha256"
        ):
            errors.append("serialized_payload.replacement_samples_semantic_sha256")
        replacement_state_ids_sha = hashlib.sha256(
            "\n".join(
                sorted(str(sample.get("id")) for sample in derived_replacements)
            ).encode()
        ).hexdigest()
        if replacement_state_ids_sha != receipt.get("replacement_state_ids_sha256"):
            errors.append("serialized_payload.replacement_state_ids_sha256")

        source_ids = [str(sample.get("id")) for sample in source_samples]
        derived_ids = [str(sample.get("id")) for sample in derived_samples]
        expected_derived_ids = set(source_ids)
        for row in replacement_rows:
            expected_derived_ids.discard(str(row.get("old_state_id")))
            expected_derived_ids.add(str(row.get("new_state_id")))
        if (
            len(source_ids) != len(set(source_ids))
            or len(derived_ids) != len(set(derived_ids))
            or set(derived_ids) != expected_derived_ids
        ):
            errors.append("serialized_payload.sample_identity_inventory")
        expected_role_counts = {
            "capability": int(config["mopd"]["capability_units_per_round"]),
            "anchor": int(config["mopd"]["anchor_units_per_round"]),
            "route_control": int(config["mopd"]["capability_units_per_round"]),
        }
        for label, samples in (
            ("source", source_samples),
            ("derived", derived_samples),
        ):
            role_counts = {
                role: sum(
                    str((sample.get("meta") or {}).get("role")) == role
                    for sample in samples
                )
                for role in ("capability", "anchor", "route_control")
            }
            if role_counts != expected_role_counts:
                errors.append(f"serialized_payload.{label}_role_inventory")
            for sample in samples:
                role = str((sample.get("meta") or {}).get("role"))
                expected_targets = {"soup"} if role == "anchor" else {
                    "quick", "deep", "soup"
                }
                if set(sample.get("targets") or {}) != expected_targets:
                    errors.append(f"serialized_payload.{label}_target_inventory")
                    break
        actual_truncation = prompt_truncation_summary(derived_samples)
        if (
            actual_truncation != receipt.get("prompt_truncation")
            or actual_truncation != derived_payload.get("prompt_truncation")
        ):
            errors.append("serialized_payload.prompt_truncation")
        if int(derived_payload.get("sample_count", -1)) != len(derived_samples):
            errors.append("serialized_payload.sample_count")
        if int(source_payload.get("sample_count", -1)) != len(source_samples):
            errors.append("serialized_payload.source_sample_count")
        actual_positions = sum(
            int(sample["positions"].numel()) for sample in derived_samples
        )
        if int(derived_payload.get("active_positions", -1)) != actual_positions:
            errors.append("serialized_payload.active_positions")
        source_positions = sum(
            int(sample["positions"].numel()) for sample in source_samples
        )
        if int(source_payload.get("active_positions", -1)) != source_positions:
            errors.append("serialized_payload.source_active_positions")
        source_semantic_sha = (rematch.get("source") or {}).get(
            "samples_semantic_sha256"
        )
        if samples_semantic_sha256(source_samples) != source_semantic_sha:
            errors.append("serialized_payload.source_samples_semantic_sha256")
    if errors:
        raise ValueError(
            "control overlay cache provenance mismatch: "
            + ", ".join(sorted(set(errors)))
        )
    return receipt
