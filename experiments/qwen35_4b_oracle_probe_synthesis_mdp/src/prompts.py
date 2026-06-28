from __future__ import annotations

from typing import Any

from .operator_env import case_to_text, template_text


def process_prompt(
    record: dict[str, Any],
    diagnostics: dict[str, Any],
    feature_permutation: list[int] | None = None,
) -> str:
    """Create a compact action-choice prompt for one verifier process state."""
    actions = diagnostics["actions"]
    display_actions = actions
    if feature_permutation is not None:
        display_actions = [actions[i] for i in feature_permutation]

    lines: list[str] = []
    lines.append("You are controlling a verified program-search process.")
    lines.append("Choose the next probe input that will shrink the surviving program set the most.")
    lines.append("Reply with exactly one letter from A through H.")
    lines.append("")
    lines.append(f"Task: {template_text(record)}")
    lines.append(f"Library size: {record['library_size']} typed operators. Each operator maps list[int] -> int.")
    lines.append(f"Current surviving candidate programs: {diagnostics['candidate_count']}")
    lines.append("")
    lines.append("Observed executions:")
    target_left, target_right = record["target_pair"]
    # The target pair is not disclosed; it is used only to compute visible outputs.
    from .operator_env import build_operator_library, eval_pair

    operators = build_operator_library(record["library_size"])
    observed_cases = list(record["visible_cases"])
    for query_index in diagnostics.get("used_query_indices", []):
        observed_cases.append(record["query_pool"][query_index])
    for i, case in enumerate(observed_cases, start=1):
        output = eval_pair(record, operators, target_left, target_right, case)
        lines.append(f"{i}. {case_to_text(case)} -> {output}")
    lines.append("")
    lines.append("Probe choices. Bucket counts are candidate outputs before observing the true result:")
    for shown, actual in zip(display_actions, actions):
        # Keep actual letters fixed while optionally corrupting the displayed features.
        lines.append(
            f"{actual['letter']}. run {case_to_text(shown['case'])}; "
            f"unique={shown['unique']}; largest_bucket={shown['largest']}; "
            f"expected_remaining={shown['expected_remaining']:.1f}; entropy={shown['entropy']:.2f}; "
            f"top_buckets={shown['top']}"
        )
    lines.append("")
    lines.append("Answer: ")
    return "\n".join(lines)
