from __future__ import annotations

from typing import Any

from .humaneval_env import current_state_metrics, greedy_next_probe, selected_candidate, survivors, test_text


STOP_LETTER = "A"
MORE_LETTER = "B"
ACTION_LETTERS = [STOP_LETTER, MORE_LETTER]


def code_excerpt(code: str, limit: int = 1200) -> str:
    if len(code) <= limit:
        return code
    return code[:limit] + "\n# ... truncated ..."


def prompt_for_state(record: dict[str, Any], used_probe_indices: list[int], max_budget: int) -> str:
    metrics = current_state_metrics(record, used_probe_indices)
    selected = selected_candidate(record, used_probe_indices)
    next_probe = greedy_next_probe(record, used_probe_indices) if len(used_probe_indices) < max_budget else None
    lines: list[str] = []
    lines.append("You are controlling an executable Python verifier.")
    lines.append("The verifier has candidate implementations for a programming task.")
    lines.append("Visible tests and already-run probes have been enforced.")
    lines.append("Choose whether to stop and commit the current selected candidate, or request one more probe.")
    lines.append("A = STOP and commit now. B = MORE and run one additional probe.")
    lines.append("Reply with exactly A or B.")
    lines.append("")
    lines.append(f"Task id: {record['task_id']}")
    lines.append("Prompt:")
    lines.append(record["prompt"].strip())
    lines.append("")
    lines.append(f"Probe budget used: {len(used_probe_indices)} of {max_budget}.")
    lines.append(f"Current surviving candidates: {metrics['candidate_count']}.")
    lines.append(f"Current output-agreement clusters: {metrics['agreement_cluster_count']}.")
    lines.append(
        f"Selected cluster size: {metrics['selected_cluster_size']} "
        f"({100 * metrics['selected_cluster_fraction']:.1f}% of survivors)."
    )
    lines.append(f"Current selected candidate: {metrics['selected_candidate_id']} from {metrics['selected_source']}.")
    lines.append("")
    lines.append("Visible tests:")
    for test in record["visible_tests"]:
        lines.append(f"- {test_text(test)}")
    if used_probe_indices:
        lines.append("Executed probes:")
        for idx in used_probe_indices:
            lines.append(f"- probe_{idx}: {test_text(record['probe_tests'][idx])}")
    else:
        lines.append("Executed probes: none")
    lines.append("")
    if selected is None:
        lines.append("Selected candidate code: none")
    else:
        lines.append("Selected candidate code:")
        lines.append("```python")
        lines.append(code_excerpt(selected["code"]))
        lines.append("```")
    lines.append("")
    if next_probe is None:
        lines.append("No further probe is available within the budget.")
    else:
        lines.append("Best available next probe under target-independent expected split:")
        lines.append(
            f"probe_{next_probe['probe_index']}: {test_text(next_probe['test'])}; "
            f"unique={next_probe['unique']}; largest_bucket={next_probe['largest']}; "
            f"expected_remaining={next_probe['expected_remaining']:.2f}; entropy={next_probe['entropy']:.2f}; "
            f"top_buckets={next_probe['top']}"
        )
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)


def rollout_greedy_until(record: dict[str, Any], budget: int, initial_used: list[int] | None = None) -> list[int]:
    used = list(initial_used or [])
    while len(used) < budget:
        choice = greedy_next_probe(record, used)
        if choice is None:
            break
        used.append(int(choice["probe_index"]))
    return used


def build_rollout_states(record: dict[str, Any], max_budget: int) -> list[dict[str, Any]]:
    used: list[int] = []
    states: list[dict[str, Any]] = []
    for budget in range(max_budget + 1):
        metrics = current_state_metrics(record, used)
        label = STOP_LETTER if metrics["selected_hidden_correct"] or budget >= max_budget else MORE_LETTER
        next_probe = greedy_next_probe(record, used) if budget < max_budget else None
        states.append(
            {
                "record_id": record["record_id"],
                "task_id": record["task_id"],
                "split": record["split"],
                "budget": budget,
                "max_budget": max_budget,
                "used_probe_indices": list(used),
                "candidate_count": metrics["candidate_count"],
                "agreement_cluster_count": metrics["agreement_cluster_count"],
                "selected_cluster_size": metrics["selected_cluster_size"],
                "selected_cluster_fraction": metrics["selected_cluster_fraction"],
                "selected_candidate_id": metrics["selected_candidate_id"],
                "selected_source": metrics["selected_source"],
                "selected_hidden_correct": metrics["selected_hidden_correct"],
                "hidden_correct_survivors": metrics["hidden_correct_survivors"],
                "target_reachable": metrics["target_reachable"],
                "next_probe": next_probe,
                "label": label,
                "label_index": 0 if label == STOP_LETTER else 1,
                "prompt": prompt_for_state(record, used, max_budget),
            }
        )
        if budget >= max_budget:
            break
        choice = greedy_next_probe(record, used)
        if choice is None:
            break
        used.append(int(choice["probe_index"]))
    return states


def evaluate_commit(record: dict[str, Any], used_probe_indices: list[int]) -> dict[str, Any]:
    metrics = current_state_metrics(record, used_probe_indices)
    live = survivors(record, used_probe_indices)
    return {
        "used_probes": len(used_probe_indices),
        "survivor_ids": [cand["candidate_id"] for cand in live],
        **metrics,
    }
