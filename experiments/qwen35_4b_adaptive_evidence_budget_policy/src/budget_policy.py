from __future__ import annotations

from typing import Any

import numpy as np

from .operator_env import (
    Operator,
    all_cases,
    bucket_stats,
    candidate_mask,
    case_to_text,
    decoded_output,
    eval_pair,
    first_prior_pair,
    full_pool_choice,
    hidden_equivalent_count,
    operator_values,
    pair_hidden_matches,
    pair_output_matrix,
    rank_queries_by_expected_split,
    template_text,
)


STOP_LETTER = "A"
MORE_LETTER = "B"
ACTION_LETTERS = [STOP_LETTER, MORE_LETTER]


def observed_executions(record: dict[str, Any], operators: list[Operator], used_query_indices: list[int]) -> list[dict[str, Any]]:
    target_left, target_right = record["target_pair"]
    observed = list(record["visible_cases"])
    observed.extend(record["query_pool"][idx] for idx in used_query_indices)
    return [
        {
            "case": case,
            "output": eval_pair(record, operators, target_left, target_right, case),
        }
        for case in observed
    ]


def current_state_metrics(record: dict[str, Any], operators: list[Operator], used_query_indices: list[int]) -> dict[str, Any]:
    mask = candidate_mask(record, operators, used_query_indices)
    selected = first_prior_pair(mask)
    return {
        "candidate_count": int(mask.sum()),
        "selected_pair": list(selected) if selected is not None else None,
        "selected_hidden_all": pair_hidden_matches(record, operators, selected),
        "hidden_equivalent_candidates": hidden_equivalent_count(record, operators, mask),
        "target_reachable": bool(mask[tuple(record["target_pair"])]),
    }


def next_probe_summary(record: dict[str, Any], operators: list[Operator], used_query_indices: list[int]) -> dict[str, Any] | None:
    unused = [idx for idx in range(len(record["query_pool"])) if idx not in set(used_query_indices)]
    if not unused:
        return None
    ranked = rank_queries_by_expected_split(record, operators, used_query_indices, unused)
    if not ranked:
        return None
    row = ranked[0]
    return {
        "query_index": int(row["query_index"]),
        "case": row["case"],
        "unique": int(row["unique"]),
        "largest": int(row["largest"]),
        "expected_remaining": float(row["expected_remaining"]),
        "entropy": float(row["entropy"]),
        "top": row["top"],
    }


def prompt_for_state(
    record: dict[str, Any],
    operators: list[Operator],
    used_query_indices: list[int],
    max_budget: int,
) -> str:
    metrics = current_state_metrics(record, operators, used_query_indices)
    next_probe = next_probe_summary(record, operators, used_query_indices)
    lines: list[str] = []
    lines.append("You are controlling an executable verifier.")
    lines.append("Choose whether to stop and commit the current selected program, or request one more probe.")
    lines.append("A = STOP and commit now. B = MORE and run one additional probe.")
    lines.append("Reply with exactly A or B.")
    lines.append("")
    lines.append(f"Task: {template_text(record)}")
    lines.append(f"Library size: {record['library_size']} typed operators. Each operator maps list[int] -> int.")
    lines.append(f"Probe budget used: {len(used_query_indices)} of {max_budget}.")
    lines.append(f"Current surviving candidate programs: {metrics['candidate_count']}.")
    lines.append(f"Current selected program by deterministic verifier prior: {metrics['selected_pair']}.")
    lines.append("")
    lines.append("Observed executions:")
    for i, obs in enumerate(observed_executions(record, operators, used_query_indices), start=1):
        lines.append(f"{i}. {case_to_text(obs['case'])} -> {obs['output']}")
    lines.append("")
    if next_probe is None or len(used_query_indices) >= max_budget:
        lines.append("No further probe is available within the budget.")
    else:
        lines.append("Best available next probe under target-independent expected split:")
        lines.append(
            f"run {case_to_text(next_probe['case'])}; "
            f"unique={next_probe['unique']}; largest_bucket={next_probe['largest']}; "
            f"expected_remaining={next_probe['expected_remaining']:.1f}; entropy={next_probe['entropy']:.2f}; "
            f"top_buckets={next_probe['top']}"
        )
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)


def rollout_greedy_until(record: dict[str, Any], operators: list[Operator], budget: int, initial_used: list[int] | None = None) -> list[int]:
    used = list(initial_used or [])
    while len(used) < budget:
        choice = full_pool_choice(record, operators, used, "fullpool_max_split")
        used.append(int(choice["query_index"]))
    return used


def build_rollout_states(record: dict[str, Any], operators: list[Operator], max_budget: int) -> list[dict[str, Any]]:
    used: list[int] = []
    states: list[dict[str, Any]] = []
    for budget in range(max_budget + 1):
        metrics = current_state_metrics(record, operators, used)
        label = STOP_LETTER if metrics["selected_hidden_all"] or budget >= max_budget else MORE_LETTER
        next_probe = next_probe_summary(record, operators, used) if budget < max_budget else None
        states.append(
            {
                "record_id": record["record_id"],
                "split": record["split"],
                "library_size": record["library_size"],
                "template": record["template"],
                "budget": budget,
                "max_budget": max_budget,
                "used_query_indices": list(used),
                "candidate_count": metrics["candidate_count"],
                "selected_pair": metrics["selected_pair"],
                "selected_hidden_all": metrics["selected_hidden_all"],
                "hidden_equivalent_candidates": metrics["hidden_equivalent_candidates"],
                "target_reachable": metrics["target_reachable"],
                "next_probe": next_probe,
                "label": label,
                "label_index": 0 if label == STOP_LETTER else 1,
                "prompt": prompt_for_state(record, operators, used, max_budget),
            }
        )
        if budget >= max_budget:
            break
        choice = full_pool_choice(record, operators, used, "fullpool_max_split")
        used.append(int(choice["query_index"]))
    return states


def evaluate_commit(record: dict[str, Any], operators: list[Operator], used_query_indices: list[int]) -> dict[str, Any]:
    metrics = current_state_metrics(record, operators, used_query_indices)
    return {
        "used_probes": len(used_query_indices),
        **metrics,
    }
