#!/usr/bin/env python3
"""Apply the frozen trained-calibration feasibility or candidate gate."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml

from downstream_common import (
    EXP,
    assert_behavior_peers,
    fail,
    finite_number,
    is_sha256,
    read_json,
    sha256_file,
    validate_behavior_receipt,
    validate_selected_answer_allowance,
)

import harness  # noqa: E402


ROOT = EXP.parents[1]


def _resolve_registered(path: str) -> Path:
    value = Path(path)
    return value.resolve() if value.is_absolute() else (ROOT / value).resolve()


def _validate_locality(locality_path: Path, cfg: dict, candidate: dict) -> dict:
    locality = read_json(locality_path)
    if locality.get("schema_version") != 1:
        fail("unsupported locality receipt schema")
    expected_before = _resolve_registered(cfg["model"]["locality_anchor"])
    expected_contexts = _resolve_registered(cfg["locality"]["contexts"])
    try:
        before = Path(locality["before_model"]).resolve()
        after = Path(locality["after_model"]).resolve()
        contexts = Path(locality["contexts"]).resolve()
    except (KeyError, TypeError) as exc:
        fail(f"malformed locality provenance: {exc}")
    if before != expected_before:
        fail("locality receipt does not use the registered anchor")
    if after != Path(candidate["model"]).resolve():
        fail("locality receipt does not evaluate the candidate checkpoint")
    if contexts != expected_contexts:
        fail("locality receipt does not use the registered context bank")
    if locality.get("auditor_sha256") != sha256_file(
        EXP / "scripts" / "audit_locality.py"
    ):
        fail("locality auditor implementation hash drifted")
    if (
        not (before / "model.safetensors").is_file()
        or sha256_file(before / "model.safetensors")
        != cfg["model"]["anchor_weight_sha256"]
        or locality.get("before_model_weight_sha256")
        != cfg["model"]["anchor_weight_sha256"]
    ):
        fail("locality anchor weights do not match the registered anchor hash")
    if (
        not (after / "model.safetensors").is_file()
        or sha256_file(after / "model.safetensors")
        != candidate["model_weight_sha256"]
        or locality.get("after_model_weight_sha256")
        != candidate["model_weight_sha256"]
    ):
        fail("locality candidate weights do not match the behavior receipts")
    expected_model_hashes = {
        "before_model_weight_sha256": cfg["model"]["anchor_weight_sha256"],
        "after_model_weight_sha256": candidate["model_weight_sha256"],
        "before_model_config_sha256": sha256_file(before / "config.json"),
        "after_model_config_sha256": sha256_file(after / "config.json"),
        "before_model_generation_config_sha256": sha256_file(
            before / "generation_config.json"
        ),
        "after_model_generation_config_sha256": sha256_file(
            after / "generation_config.json"
        ),
        "before_merge_receipt_sha256": sha256_file(before / "merge_receipt.json"),
        "after_merge_receipt_sha256": sha256_file(after / "merge_receipt.json"),
    }
    for key, expected in expected_model_hashes.items():
        if locality.get(key) != expected:
            fail(f"locality model provenance drifted at {key}")
    expected_hashes = {
        "auditor_sha256": sha256_file(EXP / "scripts" / "audit_locality.py"),
        "before_model_config_sha256": sha256_file(before / "config.json"),
        "after_model_config_sha256": sha256_file(after / "config.json"),
        "before_model_generation_config_sha256": sha256_file(
            before / "generation_config.json"
        ),
        "after_model_generation_config_sha256": sha256_file(
            after / "generation_config.json"
        ),
        "before_merge_receipt_sha256": sha256_file(before / "merge_receipt.json"),
        "after_merge_receipt_sha256": sha256_file(after / "merge_receipt.json"),
    }
    for key, expected_hash in expected_hashes.items():
        if locality.get(key) != expected_hash:
            fail(f"locality provenance drift at {key}")
    checkpoint_tokenizers = {}
    for prefix, model_path in (("before", before), ("after", after)):
        try:
            observed = harness.tokenizer_provenance(model_path)
            merge_receipt = read_json(model_path / "merge_receipt.json")
            registered = harness.validate_registered_tokenizer_provenance(
                model_path, merge_receipt, allow_absent=True
            )
        except (OSError, ValueError) as exc:
            fail(f"locality {prefix} tokenizer provenance drifted: {exc}")
        if observed != registered:
            fail(f"locality {prefix} tokenizer/merge provenance mismatch")
        expected_tokenizer = {
            "tokenizer_files": locality.get(f"{prefix}_tokenizer_files"),
            "tokenizer_manifest_sha256": locality.get(
                f"{prefix}_tokenizer_manifest_sha256"
            ),
            "tokenizer_compatibility_sha256": locality.get(
                f"{prefix}_tokenizer_compatibility_sha256"
            ),
        }
        if observed != expected_tokenizer:
            fail(f"locality {prefix} tokenizer receipt drifted")
        checkpoint_tokenizers[prefix] = observed
    configured_tokenizer_hashes = {
        "before": cfg["model"]["anchor_tokenizer_manifest_sha256"],
        "after": cfg["model"]["start_tokenizer_manifest_sha256"],
    }
    for prefix, expected_manifest in configured_tokenizer_hashes.items():
        if (
            checkpoint_tokenizers[prefix]["tokenizer_manifest_sha256"]
            != expected_manifest
            or checkpoint_tokenizers[prefix]["tokenizer_compatibility_sha256"]
            != cfg["model"]["tokenizer_compatibility_sha256"]
        ):
            fail(f"locality {prefix} tokenizer differs from the frozen config identity")
    if (
        checkpoint_tokenizers["before"]["tokenizer_compatibility_sha256"]
        != checkpoint_tokenizers["after"]["tokenizer_compatibility_sha256"]
    ):
        fail("locality checkpoint tokenizers are not behavior-compatible")
    if locality.get("rendered_prompts_equal") is not True:
        fail("locality receipt did not establish exact rendered-prompt equality")
    if locality.get("tokenized_context_ids_equal") is not True:
        fail("locality receipt did not establish exact token-ID equality")
    for before_key, after_key, label in (
        (
            "before_rendered_prompts_sha256",
            "after_rendered_prompts_sha256",
            "rendered prompts",
        ),
        (
            "before_tokenized_contexts_sha256",
            "after_tokenized_contexts_sha256",
            "token IDs",
        ),
    ):
        if (
            not is_sha256(locality.get(before_key))
            or not is_sha256(locality.get(after_key))
            or locality[before_key] != locality[after_key]
        ):
            fail(f"locality {label} digest mismatch")
    if locality.get("max_context_tokens") != int(cfg["locality"]["max_context_tokens"]):
        fail("locality context-token limit drifted")
    if locality.get("n_contexts") != int(cfg["locality"]["count"]):
        fail("locality context count drifted")
    if not is_sha256(locality.get("contexts_sha256")):
        fail("locality receipt has no context-bank hash")
    if expected_contexts.is_file() and locality["contexts_sha256"] != sha256_file(expected_contexts):
        fail("locality context-bank hash drifted")
    context_payload = read_json(expected_contexts)
    expected_ids = [row.get("id") for row in context_payload.get("contexts", [])]
    rows = locality.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != int(cfg["locality"]["count"])
        or [row.get("id") for row in rows] != expected_ids
    ):
        fail("locality rows do not match the registered context bank")
    if not math.isclose(
        float(locality.get("ceiling")),
        float(cfg["locality"]["median_non_target_logit_drift_max"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("locality drift ceiling was not the registered ceiling")
    if not math.isclose(
        float(locality.get("entropy_delta_min")),
        float(cfg["locality"]["mean_entropy_delta_min"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        fail("locality entropy floor was not the registered floor")
    if locality.get("max_context_tokens") != int(cfg["locality"]["max_context_tokens"]):
        fail("locality context-token ceiling drifted")
    if not isinstance(locality.get("checks"), dict) or not locality.get("gate", {}).get("passed"):
        fail("candidate did not pass the registered locality gate")
    for key in (
        "median_non_target_centered_logit_drift",
        "mean_entropy_delta",
        "mean_varentropy_delta",
    ):
        finite_number(locality.get(key), f"locality.{key}")
    observed_drift = statistics.median(
        finite_number(
            row.get("median_non_target_centered_logit_drift"),
            "locality.rows.median_non_target_centered_logit_drift",
        )
        for row in rows
    )
    entropy_before = statistics.mean(
        finite_number(row.get("entropy_before"), "locality.rows.entropy_before")
        for row in rows
    )
    entropy_after = statistics.mean(
        finite_number(row.get("entropy_after"), "locality.rows.entropy_after")
        for row in rows
    )
    varentropy_before = statistics.mean(
        finite_number(row.get("varentropy_before"), "locality.rows.varentropy_before")
        for row in rows
    )
    varentropy_after = statistics.mean(
        finite_number(row.get("varentropy_after"), "locality.rows.varentropy_after")
        for row in rows
    )
    recomputed = {
        "median_non_target_centered_logit_drift": observed_drift,
        "mean_entropy_delta": entropy_after - entropy_before,
        "mean_varentropy_delta": varentropy_after - varentropy_before,
    }
    if any(
        not math.isclose(
            float(locality[key]), value, rel_tol=0.0, abs_tol=1e-9
        )
        for key, value in recomputed.items()
    ):
        fail("locality summary does not match its per-context rows")
    return locality


def analyze(
    cfg: dict,
    *,
    start: dict,
    explicit_control: dict,
    shuffled_control: dict,
    candidate: dict | None = None,
    candidate_explicit: dict | None = None,
    locality: dict | None = None,
) -> dict:
    gates = cfg["calibration_gates"]
    peers = {
        "start": start,
        "explicit_control": explicit_control,
        "shuffled_control": shuffled_control,
    }
    if candidate is not None:
        peers["candidate"] = candidate
    assert_behavior_peers(peers)
    answer_max_tokens = validate_selected_answer_allowance(peers.values())
    aggregates = {name: payload["aggregate"] for name, payload in peers.items()}
    upper = 1.0

    if candidate is None:
        if candidate_explicit is not None or locality is not None:
            fail("feasibility mode cannot consume candidate-only receipts")
        checks = {
            "paired_absolute_attainable": (
                upper >= float(gates["paired_preverifier_success_absolute_min"])
            ),
            "delta_vs_start_attainable": (
                upper - float(aggregates["start"]["paired_preverifier_success"])
                >= float(gates["paired_preverifier_delta_vs_start_min"])
            ),
            "delta_vs_explicit_control_attainable": (
                upper - float(aggregates["explicit_control"]["paired_preverifier_success"])
                >= float(gates["paired_preverifier_delta_vs_explicit_control_min"])
            ),
            "delta_vs_shuffled_attainable": (
                upper - float(aggregates["shuffled_control"]["paired_preverifier_success"])
                >= float(gates["paired_preverifier_delta_vs_shuffled_min"])
            ),
            "invalid_action_attainable": (
                -float(aggregates["start"]["invalid_action_rate_per_turn"])
                <= float(gates["invalid_action_delta_vs_start_max"])
            ),
            "unusable_cap_attainable": (
                -float(aggregates["start"]["unusable_answer_cap_hit_rate_per_turn"])
                <= float(gates["unusable_cap_delta_vs_start_max"])
            ),
            "explicit_retention_attainable": (
                upper >= float(gates["explicit_first_patch_retention_min"])
            ),
        }
        return {
            "schema_version": 1,
            "stage": "trained_calibration_feasibility",
            "answer_max_tokens": answer_max_tokens,
            "task_manifest_sha256": start["task_manifest_sha256"],
            "control_paired_preverifier_success": {
                name: aggregates[name]["paired_preverifier_success"]
                for name in ("start", "explicit_control", "shuffled_control")
            },
            "checks": checks,
            "gate": {
                "passed": all(checks.values()),
                "verdict": "CALIBRATION_FEASIBLE" if all(checks.values()) else "NO_CALIBRATION_HEADROOM",
            },
            "training_candidate_authorized": all(checks.values()),
            "menagerie_authorized": False,
        }

    if candidate_explicit is None or locality is None:
        fail("candidate calibration requires candidate-explicit and locality receipts")
    if candidate_explicit.get("model_weight_sha256") != candidate.get("model_weight_sha256"):
        fail("explicit-retention receipt is not from the candidate checkpoint")
    validate_selected_answer_allowance((candidate, candidate_explicit))
    candidate_aggregate = aggregates["candidate"]
    deltas = {
        name: float(candidate_aggregate["paired_preverifier_success"])
        - float(aggregates[name]["paired_preverifier_success"])
        for name in ("start", "explicit_control", "shuffled_control")
    }
    invalid_delta = float(candidate_aggregate["invalid_action_rate_per_turn"]) - float(
        aggregates["start"]["invalid_action_rate_per_turn"]
    )
    unusable_cap_delta = float(
        candidate_aggregate["unusable_answer_cap_hit_rate_per_turn"]
    ) - float(aggregates["start"]["unusable_answer_cap_hit_rate_per_turn"])
    explicit_retention = float(candidate_explicit["aggregate"]["first_patch_full_correct"])
    checks = {
        "locality": bool(locality.get("gate", {}).get("passed")),
        "paired_absolute": (
            float(candidate_aggregate["paired_preverifier_success"])
            >= float(gates["paired_preverifier_success_absolute_min"])
        ),
        "paired_delta_vs_start": deltas["start"]
        >= float(gates["paired_preverifier_delta_vs_start_min"]),
        "paired_delta_vs_explicit_control": deltas["explicit_control"]
        >= float(gates["paired_preverifier_delta_vs_explicit_control_min"]),
        "paired_delta_vs_shuffled": deltas["shuffled_control"]
        >= float(gates["paired_preverifier_delta_vs_shuffled_min"]),
        "invalid_action": invalid_delta
        <= float(gates["invalid_action_delta_vs_start_max"]),
        "unusable_cap": unusable_cap_delta
        <= float(gates["unusable_cap_delta_vs_start_max"]),
        "explicit_first_patch_retention": explicit_retention
        >= float(gates["explicit_first_patch_retention_min"]),
    }
    return {
        "schema_version": 1,
        "stage": "trained_calibration",
        "answer_max_tokens": answer_max_tokens,
        "task_manifest_sha256": start["task_manifest_sha256"],
        "candidate_model_weight_sha256": candidate["model_weight_sha256"],
        "paired_preverifier_success": {
            name: aggregates[name]["paired_preverifier_success"] for name in peers
        },
        "candidate_deltas": deltas,
        "invalid_action_delta_vs_start": invalid_delta,
        "unusable_cap_delta_vs_start": unusable_cap_delta,
        "explicit_first_patch_retention": explicit_retention,
        "locality": {
            key: locality[key]
            for key in (
                "median_non_target_centered_logit_drift",
                "mean_entropy_delta",
                "mean_varentropy_delta",
            )
        },
        "checks": checks,
        "gate": {
            "passed": all(checks.values()),
            "verdict": "CALIBRATION_PASS" if all(checks.values()) else "CALIBRATION_FAIL",
        },
        "transfer_authorized": all(checks.values()),
        "menagerie_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=EXP / "configs" / "default.yaml")
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--explicit-control", type=Path, required=True)
    parser.add_argument("--shuffled-control", type=Path, required=True)
    parser.add_argument("--expected-explicit-control-weight-sha256", required=True)
    parser.add_argument("--expected-shuffled-control-weight-sha256", required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--expected-candidate-weight-sha256")
    parser.add_argument("--candidate-explicit", type=Path)
    parser.add_argument("--locality", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    cfg["__analysis_config_path__"] = str(args.config.resolve())
    common = {
        "block": "trained_calibration",
        "contract": "inferred",
        "scenario_set": "acquisition",
        "mode": "deep",
        "scaffold": False,
    }
    start = validate_behavior_receipt(
        args.start,
        cfg,
        arm="start",
        expected_weight_sha256=cfg["model"]["start_weight_sha256"],
        **common,
    )
    explicit_control = validate_behavior_receipt(
        args.explicit_control, cfg, arm="explicit_redundant",
        expected_weight_sha256=args.expected_explicit_control_weight_sha256,
        **common,
    )
    shuffled_control = validate_behavior_receipt(
        args.shuffled_control, cfg, arm="shuffled_binding",
        expected_weight_sha256=args.expected_shuffled_control_weight_sha256,
        **common,
    )
    candidate = None
    candidate_explicit = None
    locality = None
    if args.candidate is not None:
        if (
            args.candidate_explicit is None
            or args.locality is None
            or args.expected_candidate_weight_sha256 is None
        ):
            fail("candidate mode requires candidate hash, explicit, and locality receipts")
        candidate = validate_behavior_receipt(
            args.candidate, cfg, arm="candidate",
            expected_weight_sha256=args.expected_candidate_weight_sha256,
            **common,
        )
        candidate_explicit = validate_behavior_receipt(
            args.candidate_explicit,
            cfg,
            block="explicit_retention",
            contract="explicit",
            scenario_set="acquisition",
            mode="deep",
            arm="candidate",
            expected_weight_sha256=candidate["model_weight_sha256"],
            scaffold=False,
        )
        locality = _validate_locality(args.locality, cfg, candidate)
    elif (
        args.candidate_explicit is not None
        or args.locality is not None
        or args.expected_candidate_weight_sha256 is not None
    ):
        fail("candidate-only inputs cannot be used without --candidate")
    result = analyze(
        cfg,
        start=start,
        explicit_control=explicit_control,
        shuffled_control=shuffled_control,
        candidate=candidate,
        candidate_explicit=candidate_explicit,
        locality=locality,
    )
    result["analyzer_sha256"] = sha256_file(Path(__file__).resolve())
    result["config_sha256"] = sha256_file(args.config)
    result["receipts"] = {
        "start": {"path": str(args.start.resolve()), "sha256": sha256_file(args.start)},
        "explicit_control": {
            "path": str(args.explicit_control.resolve()),
            "sha256": sha256_file(args.explicit_control),
        },
        "shuffled_control": {
            "path": str(args.shuffled_control.resolve()),
            "sha256": sha256_file(args.shuffled_control),
        },
    }
    if args.candidate is not None:
        result["receipts"].update({
            "candidate": {
                "path": str(args.candidate.resolve()),
                "sha256": sha256_file(args.candidate),
            },
            "candidate_explicit": {
                "path": str(args.candidate_explicit.resolve()),
                "sha256": sha256_file(args.candidate_explicit),
            },
            "locality": {
                "path": str(args.locality.resolve()),
                "sha256": sha256_file(args.locality),
            },
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["gate"]["passed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
