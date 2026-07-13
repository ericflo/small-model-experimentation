#!/usr/bin/env python3
"""Build a full-prefix, control-only cache overlay from frozen candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from advantage_routing import select_teacher  # noqa: E402
from control_rematch import (  # noqa: E402
    CACHE_VARIANT,
    REMATCH_STAGE,
    control_mapping,
    prompt_truncation_summary,
    rematch_full_prefix_controls,
    samples_semantic_sha256,
)
from io_utils import (  # noqa: E402
    canonical_hash,
    expected_policy_paths,
    load_config,
    read_jsonl,
    sha256_file,
    validate_policy_cache_provenance,
    write_json,
)
from training_units import make_sparse_sample  # noqa: E402

# Reuse the exact dense scorer that created the primary target cache.  Importing
# this CLI module is side-effect free; its main entry point remains guarded.
from cache_policy_targets import _score_policy  # noqa: E402


def _flatten_artifacts(manifest: dict) -> list[dict]:
    flattened = []
    for batch in manifest["artifacts"]:
        records = [
            ("states", batch["states"]),
            ("anchors", batch["anchors"]),
        ]
        records.extend(
            (f"branches.{policy}", batch["branches"][policy])
            for policy in ("quick", "deep", "student")
        )
        for role, record in records:
            flattened.append(
                {
                    "batch": int(batch["batch"]),
                    "role": role,
                    "path": str(Path(record["path"]).resolve()),
                    "sha256": str(record["sha256"]),
                }
            )
    return sorted(flattened, key=lambda row: (row["batch"], row["role"]))


def _verify_artifacts(records: list[dict]) -> None:
    for record in records:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"candidate artifact provenance failed: {path}")


def _reconstruct_non_deep_candidates(
    manifest: dict, *, selection_branches: int
) -> tuple[list[dict], dict[str, int]]:
    state_map: dict[str, dict] = {}
    state_batches: dict[str, int] = {}
    branch_maps: dict[str, dict[tuple[str, int], dict]] = {
        policy: {} for policy in ("quick", "deep", "student")
    }
    for batch in manifest["artifacts"]:
        batch_index = int(batch["batch"])
        states = read_jsonl(Path(batch["states"]["path"]))
        for state in states:
            state_id = str(state["state_id"])
            if state_id in state_map:
                raise ValueError(f"duplicate candidate state: {state_id}")
            state_map[state_id] = state
            state_batches[state_id] = batch_index
        for policy in branch_maps:
            branches = read_jsonl(Path(batch["branches"][policy]["path"]))
            expected = {
                (str(state["state_id"]), branch_index)
                for state in states
                for branch_index in range(selection_branches)
            }
            observed = {
                (str(row["state_id"]), int(row["branch_index"])) for row in branches
            }
            if observed != expected or len(observed) != len(branches):
                raise ValueError(
                    f"{policy} candidate branch inventory mismatch in batch {batch_index}"
                )
            if any(str(row.get("policy")) != policy for row in branches):
                raise ValueError(f"{policy} candidate branch policy tag mismatch")
            for row in branches:
                key = (str(row["state_id"]), int(row["branch_index"]))
                if key in branch_maps[policy]:
                    raise ValueError(f"duplicate {policy} candidate branch: {key}")
                branch_maps[policy][key] = row

    candidates = []
    for state_id, state in sorted(state_map.items()):
        scores = {
            policy: [
                float(branch_maps[policy][(state_id, branch_index)]["score"])
                for branch_index in range(selection_branches)
            ]
            for policy in branch_maps
        }
        teacher = select_teacher(scores)
        if teacher == "deep":
            continue
        candidates.append(
            {
                "state_id": state_id,
                "family": str(state["family"]),
                "kind": str(state["kind"]),
                "level": int(state["level"]),
                "state": state,
                "selection_scores": scores,
                "selection_means": {
                    policy: sum(values) / len(values)
                    for policy, values in scores.items()
                },
                "primary_teacher": teacher,
            }
        )
    return candidates, state_batches


def _candidate_fit_sample(unit: dict, tokenizer, config: dict) -> dict:
    row = dict(unit)
    row.update(
        {
            "role": "route_control",
            "observed_route": (
                unit.get("observed_route")
                or unit.get("primary_teacher")
                or "abstain"
            ),
            "primary_teacher": "deep",
            "offpolicy_target": None,
            "matched_primary_state_id": unit.get("matched_primary_state_id"),
            "match_tier": unit.get("match_tier"),
        }
    )
    return make_sparse_sample(
        row,
        tokenizer,
        max_positions=int(config["mopd"]["max_target_positions"]),
        max_length=int(config["mopd"]["max_length"]),
    )


def _tokenizer_receipt(model: Path) -> dict:
    filenames = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja")
    files = []
    for filename in filenames:
        path = model / filename
        if not path.is_file():
            raise ValueError(f"frozen tokenizer file missing: {path}")
        files.append(
            {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
            }
        )
    return {"path": str(model.resolve()), "files": files}


def _source_inputs(
    *,
    config: dict,
    config_path: Path,
    round_manifest: Path,
    source_cache: Path,
    models: dict[str, Path],
) -> tuple[dict, dict, dict, list[dict]]:
    manifest = json.loads(round_manifest.read_text(encoding="utf-8"))
    if manifest.get("stage") != "online_advantage_training_round":
        raise ValueError("invalid source training-round manifest")
    if manifest.get("config_sha256") != sha256_file(config_path):
        raise ValueError("source training-round config mismatch")
    receipt_path = source_cache.with_suffix(source_cache.suffix + ".receipt.json")
    if not source_cache.is_file() or not receipt_path.is_file():
        raise ValueError(f"source target cache is incomplete: {source_cache}")
    cache_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if cache_receipt.get("cache_sha256") != sha256_file(source_cache):
        raise ValueError("source target cache checksum mismatch")
    validate_policy_cache_provenance(cache_receipt, config, config_path)
    source_payload = torch.load(source_cache, map_location="cpu", weights_only=False)
    validate_policy_cache_provenance(source_payload, config, config_path)
    if source_payload.get("models") != cache_receipt.get("models"):
        raise ValueError("source target cache/receipt model mismatch")
    manifest_sha = sha256_file(round_manifest)
    if (
        source_payload.get("round_manifest_sha256") != manifest_sha
        or cache_receipt.get("round_manifest_sha256") != manifest_sha
        or int(source_payload.get("round", -1)) != int(manifest["round"])
    ):
        raise ValueError("source target cache is not bound to the round manifest")
    expected_models = expected_policy_paths(config)
    if {policy: path.resolve() for policy, path in models.items()} != expected_models:
        raise ValueError("overlay policy paths differ from frozen config")

    samples = list(source_payload["samples"])
    expected_primary = int(config["mopd"]["updates_per_round"]) * int(
        config["mopd"]["grad_accum"]
    )
    expected_control = int(config["mopd"]["capability_units_per_round"])
    manifest_ids = {
        str(row["state_id"])
        for row in [*manifest.get("units", []), *manifest.get("control_units", [])]
    }
    sample_ids = {str(sample["id"]) for sample in samples}
    if (
        len(samples) != expected_primary + expected_control
        or len(sample_ids) != len(samples)
        or sample_ids != manifest_ids
    ):
        raise ValueError("source cache sample inventory differs from frozen manifest")
    primary_cuts = [
        str(sample["id"])
        for sample in samples
        if sample["meta"]["role"] in {"capability", "anchor"}
        and int(sample["meta"].get("prompt_tokens_truncated", 0)) > 0
    ]
    if primary_cuts:
        raise ValueError(f"source cache contains truncated primary samples: {primary_cuts}")
    return manifest, cache_receipt, source_payload, samples


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--round-manifest", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--quick", type=Path, required=True)
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--soup", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    config, config_path = load_config(args.config)
    round_manifest = args.round_manifest.resolve()
    source_cache = args.source_cache.resolve()
    out_dir = args.out_dir.resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"control overlay output is not empty: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rematch_path = out_dir / "rematch_manifest.json"
    derived_cache = out_dir / "all_policy_targets.pt"
    models = {
        "quick": args.quick.resolve(),
        "deep": args.deep.resolve(),
        "soup": args.soup.resolve(),
    }

    manifest_sha_before = sha256_file(round_manifest)
    cache_sha_before = sha256_file(source_cache)
    cache_receipt_path = source_cache.with_suffix(source_cache.suffix + ".receipt.json")
    cache_receipt_sha_before = sha256_file(cache_receipt_path)
    manifest, source_receipt, source_payload, source_samples = _source_inputs(
        config=config,
        config_path=config_path,
        round_manifest=round_manifest,
        source_cache=source_cache,
        models=models,
    )
    source_samples_semantic_before = samples_semantic_sha256(source_samples)
    artifact_records = _flatten_artifacts(manifest)
    _verify_artifacts(artifact_records)
    candidates, candidate_batches = _reconstruct_non_deep_candidates(
        manifest,
        selection_branches=int(config["route"]["selection_branches_per_policy"]),
    )
    expected_non_deep = int(manifest["candidate_counts"]["quick_routed"]) + int(
        manifest["candidate_counts"]["abstained"]
    )
    if len(candidates) != expected_non_deep:
        raise SystemExit("reconstructed non-deep candidate count mismatch")

    tokenizer = AutoTokenizer.from_pretrained(
        models["soup"],
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    fit_by_state = {}
    eligible_state_ids = set()
    for candidate in candidates:
        sample = _candidate_fit_sample(candidate, tokenizer, config)
        fit_by_state[str(candidate["state_id"])] = {
            "original_prompt_tokens": int(sample["meta"]["original_prompt_tokens"]),
            "completion_tokens": int(sample["completion_ids"].numel()),
            "input_tokens": int(sample["meta"]["input_tokens"]),
            "prompt_tokens_truncated": int(
                sample["meta"]["prompt_tokens_truncated"]
            ),
        }
        if int(sample["meta"]["prompt_tokens_truncated"]) == 0:
            eligible_state_ids.add(str(candidate["state_id"]))

    capability_units = [
        row for row in manifest["units"] if row.get("role") == "capability"
    ]
    original_controls = list(manifest["control_units"])
    try:
        rematched_controls, audit = rematch_full_prefix_controls(
            capability_units=capability_units,
            original_controls=original_controls,
            candidates=candidates,
            eligible_state_ids=eligible_state_ids,
            match_order=config["controls"]["non_advantage_route_match_order"],
        )
    except ValueError as error:
        failure = {
            "schema_version": 1,
            "stage": REMATCH_STAGE,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "round": int(manifest["round"]),
            "source": {
                "round_manifest": {
                    "path": str(round_manifest), "sha256": manifest_sha_before
                },
                "target_cache": {
                    "path": str(source_cache), "sha256": cache_sha_before
                },
                "target_cache_receipt": {
                    "path": str(cache_receipt_path),
                    "sha256": cache_receipt_sha_before,
                },
            },
            "match_order": list(config["controls"]["non_advantage_route_match_order"]),
            "max_length": int(config["mopd"]["max_length"]),
            "candidate_artifacts": artifact_records,
            "tokenizer": _tokenizer_receipt(models["soup"]),
            "replacement_count": None,
            "error": str(error),
            "gate": {"passed": False},
        }
        write_json(rematch_path, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 3

    original_by_primary = {
        str(row["matched_primary_state_id"]): row for row in original_controls
    }
    rematched_by_primary = {
        str(row["matched_primary_state_id"]): row for row in rematched_controls
    }
    source_sample_index = {
        str(sample["id"]): index for index, sample in enumerate(source_samples)
    }
    replacement_rows = []
    for primary_id in audit["changed_primary_state_ids"]:
        old = original_by_primary[primary_id]
        new = rematched_by_primary[primary_id]
        old_id = str(old["state_id"])
        new_id = str(new["state_id"])
        old_sample = source_samples[source_sample_index[old_id]]
        replacement_rows.append(
            {
                "matched_primary_state_id": primary_id,
                "primary_source_order": next(
                    index
                    for index, row in enumerate(
                        sorted(capability_units, key=lambda value: str(value["state_id"]))
                    )
                    if str(row["state_id"]) == primary_id
                ),
                "source_cache_index": source_sample_index[old_id],
                "old_state_id": old_id,
                "new_state_id": new_id,
                "family": str(new["family"]),
                "kind": str(new["kind"]),
                "level": int(new["level"]),
                "old_match_tier": str(old["match_tier"]),
                "new_match_tier": str(new["match_tier"]),
                "old_observed_route": str(old["observed_route"]),
                "new_observed_route": str(new["observed_route"]),
                "candidate_batch": int(candidate_batches[new_id]),
                "old_fit": {
                    "original_prompt_tokens": int(
                        old_sample["meta"]["original_prompt_tokens"]
                    ),
                    "completion_tokens": int(old_sample["completion_ids"].numel()),
                    "input_tokens": int(old_sample["meta"]["input_tokens"]),
                    "prompt_tokens_truncated": int(
                        old_sample["meta"]["prompt_tokens_truncated"]
                    ),
                },
                "new_fit": fit_by_state[new_id],
                "new_selection_scores_sha256": canonical_hash(new["selection_scores"]),
                "new_selection_means": new["selection_means"],
            }
        )
    replacement_rows.sort(key=lambda row: row["matched_primary_state_id"])
    replacement_source_indices = {
        int(row["source_cache_index"]) for row in replacement_rows
    }
    if len(replacement_source_indices) != len(replacement_rows):
        raise SystemExit("replacement controls do not have unique source-cache indices")
    copied_samples_semantic_before = samples_semantic_sha256(
        [
            sample
            for index, sample in enumerate(source_samples)
            if index not in replacement_source_indices
        ]
    )
    cache_offender_ids = {
        str(sample["id"])
        for sample in source_samples
        if sample["meta"]["role"] == "route_control"
        and int(sample["meta"].get("prompt_tokens_truncated", 0)) > 0
    }
    selected_offender_ids = {
        str(original_by_primary[primary_id]["state_id"])
        for primary_id in audit["offender_primary_state_ids"]
    }
    if cache_offender_ids != selected_offender_ids:
        raise SystemExit(
            "cache truncation inventory differs from deterministic eligibility replay"
        )

    selection_manifest = {
        "schema_version": 1,
        "stage": REMATCH_STAGE,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "round": int(manifest["round"]),
        "source": {
            "round_manifest": {
                "path": str(round_manifest), "sha256": manifest_sha_before
            },
            "target_cache": {"path": str(source_cache), "sha256": cache_sha_before},
            "target_cache_receipt": {
                "path": str(cache_receipt_path),
                "sha256": cache_receipt_sha_before,
            },
            "samples_semantic_sha256": source_samples_semantic_before,
        },
        "match_order": list(config["controls"]["non_advantage_route_match_order"]),
        "max_length": int(config["mopd"]["max_length"]),
        "eligibility": "zero completion-preserving prompt truncation",
        "candidate_artifacts": artifact_records,
        "tokenizer": _tokenizer_receipt(models["soup"]),
        "audit": audit,
        "original_control_mapping": control_mapping(original_controls),
        "rematched_control_mapping": control_mapping(rematched_controls),
        "replacement_count": len(replacement_rows),
        "replacements": replacement_rows,
        "gate": {"passed": True},
    }
    write_json(rematch_path, selection_manifest)
    if not replacement_rows:
        print(json.dumps(selection_manifest, indent=2, sort_keys=True))
        return 0

    replacement_samples = []
    replacement_index = {}
    for row in replacement_rows:
        unit = rematched_by_primary[row["matched_primary_state_id"]]
        sample = _candidate_fit_sample(unit, tokenizer, config)
        if int(sample["meta"]["prompt_tokens_truncated"]) != 0:
            raise SystemExit(f"replacement unexpectedly truncates: {sample['id']}")
        replacement_index[str(sample["id"])] = len(replacement_samples)
        replacement_samples.append(sample)
    replacement_ledgers = {}
    for policy in ("quick", "deep", "soup"):
        replacement_ledgers[policy] = _score_policy(
            models[policy],
            replacement_samples,
            policy,
            int(config["mopd"]["top_k"]),
        )
    for sample in replacement_samples:
        if set(sample["targets"]) != {"quick", "deep", "soup"}:
            raise SystemExit(f"replacement target cache incomplete: {sample['id']}")

    derived_samples = list(source_samples)
    replaced_indices = set()
    for row in replacement_rows:
        index = int(row["source_cache_index"])
        if str(derived_samples[index]["id"]) != row["old_state_id"]:
            raise SystemExit("source cache index changed during overlay construction")
        derived_samples[index] = replacement_samples[
            replacement_index[row["new_state_id"]]
        ]
        replaced_indices.add(index)
    copied_derived = [
        sample for index, sample in enumerate(derived_samples) if index not in replaced_indices
    ]
    if replaced_indices != replacement_source_indices:
        raise SystemExit("replacement source-cache index inventory changed")
    if samples_semantic_sha256(source_samples) != source_samples_semantic_before:
        raise SystemExit("overlay construction mutated the loaded source-cache samples")
    reloaded_source_payload = torch.load(
        source_cache, map_location="cpu", weights_only=False
    )
    reloaded_source_samples = list(reloaded_source_payload["samples"])
    if samples_semantic_sha256(reloaded_source_samples) != source_samples_semantic_before:
        raise SystemExit("source-cache samples changed during overlay construction")
    reloaded_copied = [
        sample
        for index, sample in enumerate(reloaded_source_samples)
        if index not in replacement_source_indices
    ]
    if samples_semantic_sha256(reloaded_copied) != copied_samples_semantic_before:
        raise SystemExit("source-cache copied-sample snapshot changed")
    if samples_semantic_sha256(copied_derived) != copied_samples_semantic_before:
        raise SystemExit("overlay modified a nonoffending source-cache sample")
    if len({str(sample["id"]) for sample in derived_samples}) != len(derived_samples):
        raise SystemExit("overlay cache contains duplicate sample IDs")
    role_counts = {
        role: sum(sample["meta"]["role"] == role for sample in derived_samples)
        for role in ("capability", "anchor", "route_control")
    }
    if role_counts != {
        "capability": int(config["mopd"]["capability_units_per_round"]),
        "anchor": int(config["mopd"]["anchor_units_per_round"]),
        "route_control": int(config["mopd"]["capability_units_per_round"]),
    }:
        raise SystemExit(f"overlay cache role inventory changed: {role_counts}")
    truncation = prompt_truncation_summary(derived_samples)
    if int(truncation["sample_count"]) != 0:
        raise SystemExit("overlay cache still contains a truncated sample")

    derived_payload = dict(source_payload)
    derived_payload.update(
        {
            "cache_variant": CACHE_VARIANT,
            "source_target_cache": str(source_cache),
            "source_target_cache_sha256": cache_sha_before,
            "source_target_cache_receipt_sha256": cache_receipt_sha_before,
            "control_rematch_manifest": str(rematch_path),
            "control_rematch_manifest_sha256": sha256_file(rematch_path),
            "replacement_count": len(replacement_rows),
            "copied_sample_count": len(reloaded_copied),
            "copied_samples_semantic_sha256": copied_samples_semantic_before,
            "replacement_samples_semantic_sha256": samples_semantic_sha256(
                replacement_samples
            ),
            "replacement_state_ids_sha256": hashlib.sha256(
                "\n".join(sorted(str(sample["id"]) for sample in replacement_samples)).encode()
            ).hexdigest(),
            "sample_count": len(derived_samples),
            "active_positions": sum(
                int(sample["positions"].numel()) for sample in derived_samples
            ),
            "prompt_truncation": truncation,
            "ledgers": {
                "source_cache": source_payload.get("ledgers"),
                "copied_sample_count": len(reloaded_copied),
                "replacement_scoring": replacement_ledgers,
            },
            "samples": derived_samples,
        }
    )
    validate_policy_cache_provenance(derived_payload, config, config_path)
    torch.save(derived_payload, derived_cache)
    receipt = {key: value for key, value in derived_payload.items() if key != "samples"}
    receipt["cache_sha256"] = sha256_file(derived_cache)
    receipt["cache_bytes"] = derived_cache.stat().st_size
    write_json(
        derived_cache.with_suffix(derived_cache.suffix + ".receipt.json"), receipt
    )
    if (
        sha256_file(round_manifest) != manifest_sha_before
        or sha256_file(source_cache) != cache_sha_before
        or sha256_file(cache_receipt_path) != cache_receipt_sha_before
    ):
        raise SystemExit("source primary artifacts changed during overlay construction")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
