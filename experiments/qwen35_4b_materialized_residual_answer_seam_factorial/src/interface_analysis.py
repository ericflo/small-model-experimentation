"""Integer calibration/transport gates and the frozen interface winner rule."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from protocol import score_echo


def answer_cap_contact(output: dict[str, Any], cap: int) -> bool:
    if cap < 1:
        raise ValueError("answer cap must be positive")
    tokens = output.get("n_answer_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise ValueError("output has invalid n_answer_tokens")
    return tokens >= cap or output.get("finish_reason") == "length"


def thinking_cap_contact(output: dict[str, Any], budget: int) -> bool:
    if budget < 1:
        raise ValueError("thinking budget must be positive")
    tokens = output.get("n_thinking_tokens")
    if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens < 0:
        raise ValueError("output has invalid n_thinking_tokens")
    if output.get("seed_domain_stage1") == "answer":
        if tokens != 0:
            raise ValueError("no-think answer seed reported thinking tokens")
        return False
    return tokens >= budget or output.get("stage1_finish_reason") == "length"


def score_interface_rows(
    rows: Sequence[dict[str, Any]],
    *,
    answer_cap: int,
    thinking_budget: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("interface scoring requires rows")
    scored: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        row_id = row.get("id")
        meta = row.get("meta")
        outputs = row.get("outputs")
        if (
            not isinstance(row_id, str)
            or row_id in seen
            or not isinstance(meta, dict)
            or not isinstance(outputs, list)
            or len(outputs) != 1
        ):
            raise ValueError("interface result row schema changed")
        seen.add(row_id)
        arity = int(meta["arity"])
        expected = str(meta["expected"])
        output = outputs[0]
        seed_domain = output.get("seed_domain_stage1")
        if seed_domain not in {"thought", "answer"}:
            raise ValueError("interface result has an invalid stage-one seed domain")
        echo = score_echo(
            output["text"],
            expected=expected,
            arity=arity,
            thinking_expected=seed_domain == "thought",
        )
        scored.append(
            {
                "id": row_id,
                "task_id": meta["task_id"],
                "arity": arity,
                "expected": expected,
                "exact_echo": bool(echo["exact_echo"]),
                "parsed": bool(echo["parsed"]),
                "answer_body": echo["answer_body"],
                "parse_error": echo["error"],
                "answer_cap_contact": answer_cap_contact(output, answer_cap),
                "thinking_cap_contact": thinking_cap_contact(
                    output, thinking_budget
                ),
                "n_answer_tokens": output["n_answer_tokens"],
                "n_thinking_tokens": output["n_thinking_tokens"],
                "finish_reason": output["finish_reason"],
                "stage1_finish_reason": output["stage1_finish_reason"],
            }
        )
    by_arity: dict[str, dict[str, int]] = {}
    for arity in (2, 3):
        subset = [row for row in scored if row["arity"] == arity]
        by_arity[str(arity)] = {
            "rows": len(subset),
            "exact_echo_successes": sum(row["exact_echo"] for row in subset),
            "parse_successes": sum(row["parsed"] for row in subset),
            "answer_cap_contacts": sum(
                row["answer_cap_contact"] for row in subset
            ),
            "thinking_cap_contacts": sum(
                row["thinking_cap_contact"] for row in subset
            ),
        }
    return {
        "rows": len(scored),
        "exact_echo_successes": sum(row["exact_echo"] for row in scored),
        "parse_successes": sum(row["parsed"] for row in scored),
        "answer_cap_contacts": sum(row["answer_cap_contact"] for row in scored),
        "thinking_cap_contacts": sum(
            row["thinking_cap_contact"] for row in scored
        ),
        "arity_counts": dict(sorted(Counter(row["arity"] for row in scored).items())),
        "by_arity": by_arity,
        "scored": scored,
    }


def calibration_qualifies(metrics: dict[str, Any], gate: dict[str, int]) -> bool:
    if metrics["rows"] != gate["rows"]:
        raise ValueError("calibration row denominator changed")
    return bool(
        metrics["exact_echo_successes"] >= gate["exact_echo_successes_min"]
        and metrics["parse_successes"] >= gate["parse_successes_min"]
        and metrics["answer_cap_contacts"] <= gate["answer_cap_contacts_max"]
        and all(
            metrics["by_arity"][str(arity)]["rows"]
            == gate["suffix_rows" if arity == 2 else "direct_rows"]
            and metrics["by_arity"][str(arity)]["exact_echo_successes"]
            >= gate["each_arity_exact_successes_min"]
            and metrics["by_arity"][str(arity)]["parse_successes"]
            >= gate["each_arity_parse_successes_min"]
            and metrics["by_arity"][str(arity)]["answer_cap_contacts"]
            <= gate["each_arity_answer_cap_contacts_max"]
            for arity in (2, 3)
        )
    )


def choose_interface(
    metrics_by_arm: dict[str, dict[str, Any]],
    *,
    priority: Sequence[str],
    gate: dict[str, int],
) -> dict[str, Any]:
    if set(metrics_by_arm) != set(priority) or len(set(priority)) != len(priority):
        raise ValueError("interface arm inventory differs from frozen priority")
    qualification = {
        arm: calibration_qualifies(metrics_by_arm[arm], gate) for arm in priority
    }
    winner = next((arm for arm in priority if qualification[arm]), None)
    return {
        "decision": (
            "CALIBRATION_INTERFACE_SELECTED"
            if winner is not None
            else "NO_VALID_RESIDUAL_ANSWER_SEAM"
        ),
        "winner": winner,
        "fixed_priority": list(priority),
        "qualification": qualification,
        "selection_uses_metric_ranking": False,
    }
