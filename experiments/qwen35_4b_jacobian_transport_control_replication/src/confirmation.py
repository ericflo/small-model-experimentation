"""Pure contracts and frozen decision logic for untouched confirmation."""

from __future__ import annotations

import hashlib
from typing import Any

from stats import paired_bootstrap_mean_ci


PROMPT_KINDS = ("direct", "consequence")
CONDITIONS = (
    "baseline",
    "full_target_donor",
    "j_all24",
    "random_a",
    "random_b",
    "j_wrong_donor",
    "j_pair",
    "logit_lens_all24",
)
CALIBRATION_ROW_KEYS = {
    "item_id",
    "prompt_kind",
    "arm",
    "layer",
    "passed",
    "j_delta_norm",
    "control_delta_norm",
    "norm_relative_error",
    "realized_span_projection_fraction",
    "chosen_candidate_index",
    "correction_iterations",
    "lattice_pair_steps",
}


def answer_contract(model, row: dict[str, Any], *, kind: str, concepts, digits):
    if kind == "direct":
        return {
            "source_id": model.concept_token_id(row["source"]),
            "target_id": model.concept_token_id(row["target"]),
            "wrong_id": model.concept_token_id(row["wrong"]),
            "parse_ids": {model.concept_token_id(concept) for concept in concepts},
        }
    if kind == "consequence":
        return {
            "source_id": model.bare_token_id(row["source_digit"]),
            "target_id": model.bare_token_id(row["target_digit"]),
            "wrong_id": model.bare_token_id(row["wrong_digit"]),
            "parse_ids": {model.bare_token_id(digit) for digit in digits},
        }
    raise ValueError(f"unknown prompt kind: {kind}")


def scored_row(
    model,
    item: dict[str, Any],
    *,
    kind: str,
    condition: str,
    band: tuple[int, ...],
    score: dict[str, Any],
    concepts,
    digits,
) -> dict[str, Any]:
    contract = answer_contract(
        model, item, kind=kind, concepts=concepts, digits=digits
    )
    logits = score["logits"]
    top_id = int(score.get("top_id", int(__import__("torch").argmax(logits).item())))
    delta_norms = {
        str(layer): float(delta.float().norm())
        for layer, delta in score.get("deltas", {}).items()
    }
    return {
        "item_id": item["item_id"],
        "split": "confirmation",
        "prompt_kind": kind,
        "condition": condition,
        "band": list(band),
        "source": item["source"],
        "target": item["target"],
        "wrong": item["wrong"],
        "source_answer": item["source"] if kind == "direct" else item["source_digit"],
        "target_answer": item["target"] if kind == "direct" else item["target_digit"],
        "wrong_answer": item["wrong"] if kind == "direct" else item["wrong_digit"],
        "source_id": contract["source_id"],
        "target_id": contract["target_id"],
        "wrong_id": contract["wrong_id"],
        "top_id": top_id,
        "top_text": model.tokenizer.decode([top_id]),
        "source_correct": top_id == contract["source_id"],
        "target_selected": top_id == contract["target_id"],
        "wrong_selected": top_id == contract["wrong_id"],
        "parsed": top_id in contract["parse_ids"],
        "target_minus_source_logit": float(
            logits[contract["target_id"]] - logits[contract["source_id"]]
        ),
        "wrong_minus_source_logit": float(
            logits[contract["wrong_id"]] - logits[contract["source_id"]]
        ),
        "delta_norms": delta_norms,
        "total_delta_norm": float(
            sum(value * value for value in delta_norms.values()) ** 0.5
        ),
        "sequence_tokens": int(score["sequence_tokens"]),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty rows")
    n = len(rows)
    return {
        "n": n,
        "source_accuracy": sum(bool(row["source_correct"]) for row in rows) / n,
        "target_rate": sum(bool(row["target_selected"]) for row in rows) / n,
        "wrong_rate": sum(bool(row["wrong_selected"]) for row in rows) / n,
        "parse_rate": sum(bool(row["parsed"]) for row in rows) / n,
        "mean_target_minus_source_logit": sum(
            float(row["target_minus_source_logit"]) for row in rows
        ) / n,
        "mean_total_delta_norm": sum(
            float(row["total_delta_norm"]) for row in rows
        ) / n,
    }


def _stable_seed(base: int, part: str) -> int:
    payload = f"{base}\0{part}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") % (
        2**31
    )


def validate_calibration_contract(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    intervention = config["intervention"]
    expected_items = int(config["data"]["control_calibration_items"])
    arms = tuple(intervention["random_arms"])
    band = tuple(int(layer) for layer in intervention["band"])
    expected_rows = expected_items * len(PROMPT_KINDS) * len(arms) * len(band)
    if not (
        summary.get("passed") is True
        and summary.get("decision") == "CONTROL_CALIBRATION_PASS"
        and summary.get("stage") == "control_calibration"
        and summary.get("scientific_result") is False
        and summary.get("outcomes_recorded") is False
        and summary.get("logits_recorded") is False
        and int(summary.get("numeric_rows", -1)) == expected_rows
        and int(summary.get("expected_numeric_rows", -1)) == expected_rows
        and len(rows) == expected_rows
    ):
        raise RuntimeError("committed calibration summary is not an eligible firewall")
    if any(set(row) != CALIBRATION_ROW_KEYS for row in rows):
        raise RuntimeError("calibration row schema changed after the frozen firewall")
    item_ids = {str(row["item_id"]) for row in rows}
    if len(item_ids) != expected_items:
        raise RuntimeError("calibration item cardinality mismatch")
    identities = {
        (
            str(row["item_id"]),
            str(row["prompt_kind"]),
            str(row["arm"]),
            int(row["layer"]),
        )
        for row in rows
    }
    expected_identities = {
        (item_id, kind, arm, layer)
        for item_id in item_ids
        for kind in PROMPT_KINDS
        for arm in arms
        for layer in band
    }
    if identities != expected_identities or len(identities) != len(rows):
        raise RuntimeError("calibration rows are incomplete or duplicated")
    norm_max = max(float(row["norm_relative_error"]) for row in rows)
    projection_max = max(
        float(row["realized_span_projection_fraction"]) for row in rows
    )
    if (
        not all(bool(row["passed"]) for row in rows)
        or norm_max > float(intervention["norm_relative_tolerance"])
        or projection_max > float(intervention["realized_span_projection_max"])
        or float(summary.get("max_norm_relative_error", float("inf"))) != norm_max
        or float(
            summary.get("max_realized_span_projection_fraction", float("inf"))
        )
        != projection_max
        or float(summary.get("causal_activation_max_abs", float("inf")))
        > float(intervention["causal_activation_atol"])
    ):
        raise RuntimeError("calibration numeric geometry no longer satisfies the gate")
    return {
        "passed": True,
        "items": len(item_ids),
        "numeric_rows": len(rows),
        "max_norm_relative_error": norm_max,
        "max_realized_span_projection_fraction": projection_max,
        "max_lattice_pair_steps": max(int(row["lattice_pair_steps"]) for row in rows),
    }


def evaluate_confirmation(
    result_rows: list[dict[str, Any]],
    control_rows: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    design_pass: bool,
    calibration_pass: bool,
    causal_max_abs: float,
) -> dict[str, Any]:
    expected_items = int(config["data"]["confirmation_items"])
    item_ids = {str(row["item_id"]) for row in result_rows}
    if len(item_ids) != expected_items:
        raise RuntimeError("confirmation item cardinality mismatch")
    expected_outcome_identities = {
        (item_id, kind, condition)
        for item_id in item_ids
        for kind in PROMPT_KINDS
        for condition in CONDITIONS
    }
    observed_outcome_identities = {
        (str(row["item_id"]), str(row["prompt_kind"]), str(row["condition"]))
        for row in result_rows
    }
    if (
        observed_outcome_identities != expected_outcome_identities
        or len(observed_outcome_identities) != len(result_rows)
    ):
        raise RuntimeError("confirmation arms are incomplete or duplicated")

    band = tuple(int(layer) for layer in config["intervention"]["band"])
    arms = tuple(config["intervention"]["random_arms"])
    expected_control_identities = {
        (item_id, kind, arm, layer)
        for item_id in item_ids
        for kind in PROMPT_KINDS
        for arm in arms
        for layer in band
    }
    observed_control_identities = {
        (
            str(row["item_id"]),
            str(row["prompt_kind"]),
            str(row["arm"]),
            int(row["layer"]),
        )
        for row in control_rows
    }
    if (
        observed_control_identities != expected_control_identities
        or len(observed_control_identities) != len(control_rows)
    ):
        raise RuntimeError("confirmation numeric controls are incomplete or duplicated")

    def subset(*, kind: str, condition: str) -> list[dict[str, Any]]:
        return [
            row
            for row in result_rows
            if row["prompt_kind"] == kind and row["condition"] == condition
        ]

    summaries = {
        condition: {
            kind: summarize(subset(kind=kind, condition=condition))
            for kind in PROMPT_KINDS
        }
        for condition in CONDITIONS
    }
    baseline = summaries["baseline"]
    primary = summaries["j_all24"]
    wrong_control = summaries["j_wrong_donor"]
    gates = config["gates"]
    clean_pass = all(
        baseline[kind]["source_accuracy"] >= float(gates["clean_accuracy_min"])
        and baseline[kind]["parse_rate"] >= float(gates["clean_parse_rate_min"])
        for kind in PROMPT_KINDS
    )
    donor_pass = bool(
        summaries["full_target_donor"]["direct"]["target_rate"]
        >= float(gates["donor_direct_target_rate_min"])
        and summaries["full_target_donor"]["consequence"]["target_rate"]
        >= float(gates["donor_consequence_target_rate_min"])
    )
    norm_limit = float(config["intervention"]["norm_relative_tolerance"])
    projection_limit = float(
        config["intervention"]["realized_span_projection_max"]
    )
    numeric_pass = bool(
        len(control_rows) == len(expected_control_identities)
        and all(bool(row["passed"]) for row in control_rows)
        and max(float(row["norm_relative_error"]) for row in control_rows)
        <= norm_limit
        and max(
            float(row["realized_span_projection_fraction"])
            for row in control_rows
        )
        <= projection_limit
    )
    causal_pass = causal_max_abs <= float(
        config["intervention"]["causal_activation_atol"]
    )
    direct_shift = (
        primary["direct"]["target_rate"] - baseline["direct"]["target_rate"]
    )
    consequence_shift = (
        primary["consequence"]["target_rate"]
        - baseline["consequence"]["target_rate"]
    )
    random_rates = {
        arm: summaries[arm]["consequence"]["target_rate"] for arm in arms
    }
    worse_random_arm = max(arms, key=lambda arm: (random_rates[arm], arm))
    j_minus_worse_random = (
        primary["consequence"]["target_rate"] - random_rates[worse_random_arm]
    )
    j_minus_wrong_target = (
        primary["consequence"]["target_rate"]
        - wrong_control["consequence"]["target_rate"]
    )
    wrong_own_shift = (
        wrong_control["consequence"]["wrong_rate"]
        - baseline["consequence"]["wrong_rate"]
    )
    parse_drop = (
        baseline["consequence"]["parse_rate"]
        - primary["consequence"]["parse_rate"]
    )
    primary_by_item = {
        str(row["item_id"]): float(row["target_selected"])
        for row in subset(kind="consequence", condition="j_all24")
    }
    bootstraps = {}
    for arm in arms:
        random_by_item = {
            str(row["item_id"]): float(row["target_selected"])
            for row in subset(kind="consequence", condition=arm)
        }
        if set(primary_by_item) != set(random_by_item):
            raise RuntimeError(f"paired J/{arm} item sets disagree")
        differences = [
            primary_by_item[item_id] - random_by_item[item_id]
            for item_id in sorted(primary_by_item)
        ]
        bootstraps[arm] = paired_bootstrap_mean_ci(
            differences,
            resamples=int(gates["bootstrap_resamples"]),
            seed=_stable_seed(int(config["seeds"]["bootstrap"]), arm),
        )
    bootstrap_pass = all(
        result["lower"] > float(gates["bootstrap_lower_bound_min"])
        for result in bootstraps.values()
    )
    primary_pass = bool(
        design_pass
        and calibration_pass
        and clean_pass
        and donor_pass
        and numeric_pass
        and causal_pass
        and direct_shift >= float(gates["j_direct_shift_min"])
        and consequence_shift >= float(gates["j_consequence_shift_min"])
        and j_minus_worse_random >= float(gates["j_minus_random_min"])
        and j_minus_wrong_target >= float(gates["j_minus_wrong_target_min"])
        and wrong_own_shift >= float(gates["wrong_own_digit_shift_min"])
        and parse_drop <= float(gates["max_parse_rate_drop"])
        and bootstrap_pass
    )
    if not (
        design_pass and calibration_pass and numeric_pass and causal_pass and donor_pass
    ):
        decision = "INVALID_CONTROL"
    elif primary_pass:
        decision = "REPLICATED_J_TRANSPORT"
    elif direct_shift >= float(gates["j_direct_shift_min"]):
        decision = "DIRECT_ONLY"
    else:
        decision = "NO_REPLICATION"
    return {
        "passed": primary_pass,
        "decision": decision,
        "summaries": summaries,
        "gate_metrics": {
            "design_pass": design_pass,
            "calibration_pass": calibration_pass,
            "clean_pass": clean_pass,
            "donor_pass": donor_pass,
            "causal_invariance_pass": causal_pass,
            "numeric_control_pass": numeric_pass,
            "direct_target_shift": direct_shift,
            "consequence_target_shift": consequence_shift,
            "random_consequence_target_rates": random_rates,
            "worse_random_arm": worse_random_arm,
            "consequence_j_minus_worse_random": j_minus_worse_random,
            "consequence_j_minus_wrong_target": j_minus_wrong_target,
            "wrong_donor_own_digit_shift": wrong_own_shift,
            "consequence_parse_drop": parse_drop,
            "paired_bootstrap_j_minus_random": bootstraps,
            "paired_bootstrap_pass": bootstrap_pass,
        },
        "control_audit": {
            "max_norm_relative_error": max(
                float(row["norm_relative_error"]) for row in control_rows
            ),
            "max_realized_span_projection_fraction": max(
                float(row["realized_span_projection_fraction"])
                for row in control_rows
            ),
            "lattice_repair_rows": sum(
                int(row["lattice_pair_steps"]) > 0 for row in control_rows
            ),
            "max_lattice_pair_steps": max(
                int(row["lattice_pair_steps"]) for row in control_rows
            ),
        },
    }
