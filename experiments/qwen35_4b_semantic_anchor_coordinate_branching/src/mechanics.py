"""Pure label-free mechanics scoring and frozen decision logic."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch


ARMS = (
    "source",
    "text_target",
    "full_donor",
    "donor_j",
    "mean_donor_j",
    "additive_j",
    "non_j_a",
    "non_j_b",
    "wrong_donor_j",
    "logit_lens_all24",
)


def wrong_derangement(source: str, aliases: list[str]) -> dict[str, str]:
    targets = [alias for alias in aliases if alias != source]
    if len(targets) != 11:
        raise ValueError("wrong-donor derangement requires 12 aliases")
    result = {
        target: targets[(index + 1) % len(targets)]
        for index, target in enumerate(targets)
    }
    if set(result) != set(targets) or set(result.values()) != set(targets):
        raise AssertionError("wrong donor is not bijective")
    if any(target == wrong or wrong == source for target, wrong in result.items()):
        raise AssertionError("wrong donor is not a non-source derangement")
    return result


def expected_token(task: dict[str, Any], alias: str, probe: str) -> str:
    if probe == "direct":
        return alias
    if probe == "consequence":
        operation = task["alias_to_operation"][alias]
        return task["result_label_by_operation"][operation]
    raise ValueError(f"unknown probe {probe!r}")


def scored_row(
    model,
    task: dict[str, Any],
    *,
    target_alias: str,
    wrong_alias: str,
    probe: str,
    arm: str,
    score: dict[str, Any],
    registered_tokens: list[str],
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError(f"unknown mechanics arm {arm!r}")
    token_ids = [model.leading_space_token_id(token) for token in registered_tokens]
    logits = score["logits"].float()
    constrained = logits[token_ids]
    probabilities = torch.softmax(constrained, dim=-1)
    full_normalizer = torch.logsumexp(logits, dim=-1)
    full_probabilities = torch.exp(constrained - full_normalizer)
    choice_index = int(torch.argmax(constrained))
    full_top_id = int(torch.argmax(logits))
    target_token = expected_token(task, target_alias, probe)
    wrong_token = expected_token(task, wrong_alias, probe)
    target_index = registered_tokens.index(target_token)
    wrong_index = registered_tokens.index(wrong_token)
    return {
        "task_id": str(task["task_id"]),
        "source_alias": str(task["source_alias"]),
        "target_alias": target_alias,
        "wrong_alias": wrong_alias,
        "probe": probe,
        "arm": arm,
        "registered_tokens": registered_tokens,
        "registered_token_ids": token_ids,
        "constrained_probabilities": {
            token: float(probabilities[index])
            for index, token in enumerate(registered_tokens)
        },
        "full_vocabulary_probabilities": {
            token: float(full_probabilities[index])
            for index, token in enumerate(registered_tokens)
        },
        "registered_probability_mass": float(full_probabilities.sum()),
        "constrained_choice": registered_tokens[choice_index],
        "full_top_id": full_top_id,
        "full_top_text": model.tokenizer.decode([full_top_id]),
        "parsed": full_top_id in set(token_ids),
        "target_token": target_token,
        "wrong_token": wrong_token,
        "target_selected": choice_index == target_index,
        "wrong_own_selected": choice_index == wrong_index,
        "target_probability": float(probabilities[target_index]),
        "target_full_probability": float(full_probabilities[target_index]),
        "wrong_probability": float(probabilities[wrong_index]),
        "finite": bool(torch.isfinite(logits).all() and torch.isfinite(probabilities).all()),
        "sequence_tokens": int(score["sequence_tokens"]),
    }


def _rate(rows: list[dict[str, Any]], arm: str, probe: str, key: str) -> float:
    selected = [row for row in rows if row["arm"] == arm and row["probe"] == probe]
    if not selected:
        raise ValueError(f"no rows for {arm}/{probe}")
    return sum(bool(row[key]) for row in selected) / len(selected)


def evaluate(
    rows: list[dict[str, Any]],
    numeric_rows: list[dict[str, Any]],
    intervention_rows: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    expected = 4 * 11 * 2 * len(ARMS)
    identities = {
        (row["task_id"], row["target_alias"], row["probe"], row["arm"])
        for row in rows
    }
    complete = len(rows) == expected and len(identities) == expected
    arm_probe_rates = {
        arm: {
            probe: _rate(rows, arm, probe, "target_selected")
            for probe in ("direct", "consequence")
        }
        for arm in ARMS
    }
    wrong_own = _rate(rows, "wrong_donor_j", "consequence", "wrong_own_selected")
    wrong_target = _rate(rows, "wrong_donor_j", "consequence", "target_selected")
    j_rows = [
        row for row in rows if row["arm"] == "donor_j" and row["probe"] == "consequence"
    ]
    source_by_key = {
        (row["task_id"], row["target_alias"]): row
        for row in rows if row["arm"] == "source" and row["probe"] == "consequence"
    }
    mean_lift = sum(
        row["target_probability"]
        - source_by_key[(row["task_id"], row["target_alias"])]["target_probability"]
        for row in j_rows
    ) / len(j_rows)
    worse_non_j = max(
        arm_probe_rates["non_j_a"]["consequence"],
        arm_probe_rates["non_j_b"]["consequence"],
    )
    j_minus_non_j = arm_probe_rates["donor_j"]["consequence"] - worse_non_j
    successful = [row for row in j_rows if row["target_selected"]]
    alias_support = len({row["target_alias"] for row in successful})
    label_support = len({row["target_token"] for row in successful})
    task_support = len({row["task_id"] for row in successful})
    parse_rate = sum(bool(row["parsed"]) for row in rows) / len(rows)
    numeric_pass = bool(
        len(numeric_rows) == 880
        and all(row["passed"] for row in numeric_rows)
        and intervention_rows
        and all(row["passed"] for row in intervention_rows)
    )
    gates = config["gates"]["mechanics"]
    metrics = {
        "text_direct_rate": arm_probe_rates["text_target"]["direct"],
        "text_consequence_rate": arm_probe_rates["text_target"]["consequence"],
        "full_donor_direct_rate": arm_probe_rates["full_donor"]["direct"],
        "full_donor_consequence_rate": arm_probe_rates["full_donor"]["consequence"],
        "coordinate_direct_rate": arm_probe_rates["donor_j"]["direct"],
        "coordinate_consequence_rate": arm_probe_rates["donor_j"]["consequence"],
        "coordinate_consequence_probability_lift": mean_lift,
        "coordinate_minus_worse_non_j_rate": j_minus_non_j,
        "wrong_donor_own_consequence_rate": wrong_own,
        "wrong_donor_registered_target_rate": wrong_target,
        "coordinate_candidate_alias_support": alias_support,
        "coordinate_result_label_support": label_support,
        "coordinate_successful_tasks": task_support,
        "parse_rate": parse_rate,
        "numeric_rows_pass_rate": (
            sum(bool(row["passed"]) for row in numeric_rows) / len(numeric_rows)
            if numeric_rows else 0.0
        ),
    }
    gate_results = {
        "text": bool(
            metrics["text_direct_rate"] >= float(gates["text_direct_rate_min"])
            and metrics["text_consequence_rate"] >= float(gates["text_consequence_rate_min"])
        ),
        "full_donor": bool(
            metrics["full_donor_direct_rate"] >= float(gates["full_donor_direct_rate_min"])
            and metrics["full_donor_consequence_rate"]
            >= float(gates["full_donor_consequence_rate_min"])
        ),
        "coordinate_direct": bool(
            metrics["coordinate_direct_rate"] >= float(gates["coordinate_direct_rate_min"])
        ),
        "coordinate_complete": bool(
            metrics["coordinate_consequence_rate"]
            >= float(gates["coordinate_consequence_rate_min"])
            and metrics["coordinate_consequence_probability_lift"]
            >= float(gates["coordinate_consequence_probability_lift_min"])
            and metrics["coordinate_minus_worse_non_j_rate"]
            >= float(gates["coordinate_minus_worse_non_j_rate_min"])
            and metrics["wrong_donor_own_consequence_rate"]
            >= float(gates["wrong_donor_own_consequence_rate_min"])
            and metrics["wrong_donor_registered_target_rate"]
            <= float(gates["wrong_donor_registered_target_rate_max"])
            and metrics["coordinate_candidate_alias_support"]
            >= int(gates["coordinate_candidate_alias_support_min"])
            and metrics["coordinate_result_label_support"]
            >= int(gates["coordinate_result_label_support_min"])
            and metrics["coordinate_successful_tasks"]
            >= int(gates["coordinate_successful_tasks_min"])
        ),
        "parse": bool(metrics["parse_rate"] >= float(gates["parse_rate_min"])),
        "numeric": numeric_pass,
        "complete": complete and all(row["finite"] for row in rows),
    }
    additive = bool(
        arm_probe_rates["additive_j"]["direct"]
        >= float(gates["coordinate_direct_rate_min"])
        and arm_probe_rates["additive_j"]["consequence"]
        >= float(gates["coordinate_consequence_rate_min"])
    )
    if not gate_results["numeric"] or not gate_results["parse"] or not gate_results["complete"]:
        decision = "INVALID_MECHANICS_CONTROL"
    elif not gate_results["text"]:
        decision = "ANCHOR_PROBE_UNREACHABLE"
    elif not gate_results["full_donor"]:
        decision = "NO_NATIVE_ANCHOR_STATE_TRANSPORT"
    elif not gate_results["coordinate_direct"]:
        decision = "NO_NATIVE_ANCHOR_J_TRANSPORT"
    elif not gate_results["coordinate_complete"]:
        decision = "DIRECT_ONLY_NATIVE_ANCHOR_J"
    else:
        decision = "NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT"
    return {
        "passed": decision == "NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT",
        "decision": decision,
        "additive_decision": (
            "ADDITIVE_ANCHOR_TRANSPORT" if additive else "NO_ADDITIVE_ANCHOR_TRANSPORT"
        ),
        "metrics": metrics,
        "gate_results": gate_results,
        "arm_probe_rates": arm_probe_rates,
        "outcome_rows": len(rows),
        "numeric_rows": len(numeric_rows),
        "intervention_rows": len(intervention_rows),
    }
