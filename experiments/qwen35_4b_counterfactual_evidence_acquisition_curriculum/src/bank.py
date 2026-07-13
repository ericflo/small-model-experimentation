"""Executable, transition-balanced evidence-acquisition banking."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from repo_agent import SYSTEM, assistant_content, execute_action, initial_messages, operator_for
from repo_tasks import Patch, RepoEnv, RepoTask


OPERATORS = ("INSPECT", "PATCH", "VERIFY", "COMMIT")
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
RECOVERY_TRANSITIONS = (
    "start_to_inspect_source",
    "explicit_source_to_patch",
    "rejected_patch_to_changed_patch",
    "failed_test_to_diagnose",
    "diagnosis_to_changed_patch",
    "patch_ok_to_verify",
    "passed_test_to_commit",
)
TRANSITION_OPERATOR = {
    "start_to_inspect_source": "INSPECT",
    "ambiguous_source_to_inspect_evidence": "INSPECT",
    "evidence_to_policy_patch": "PATCH",
    "explicit_source_to_patch": "PATCH",
    "rejected_patch_to_changed_patch": "PATCH",
    "failed_test_to_diagnose": "INSPECT",
    "diagnosis_to_changed_patch": "PATCH",
    "patch_ok_to_verify": "VERIFY",
    "passed_test_to_commit": "COMMIT",
}


def _action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _valid_patch_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(step["action"])
        for step in trajectory["steps"]
        if (step.get("action") or {}).get("tool") == "patch"
        and step.get("before_digest") != step.get("after_digest")
    ]


def replay_patch_set(task: RepoTask, patches: list[dict[str, Any]]) -> dict[str, Any]:
    env = RepoEnv(task)
    try:
        observations = []
        for action in patches:
            observation, done, _ = execute_action(env, action)
            observations.append(observation)
            if done or not observation.startswith("PATCH_OK"):
                return {
                    "visible": False,
                    "hidden": False,
                    "observations": observations,
                    "digest": env.workspace_digest(),
                    "snapshot": env.source_snapshot(),
                }
        visible = env.visible_pass()
        hidden = env.hidden_pass()
        return {
            "visible": visible,
            "hidden": hidden,
            "observations": observations,
            "digest": env.workspace_digest(),
            "snapshot": env.source_snapshot(),
        }
    finally:
        env.close()


def minimize_patches(task: RepoTask, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept = copy.deepcopy(patches)
    index = 0
    while index < len(kept):
        candidate = kept[:index] + kept[index + 1 :]
        replay = replay_patch_set(task, candidate)
        if replay["visible"] and replay["hidden"]:
            kept = candidate
        else:
            index += 1
    return kept


def collapse_patches_by_file(task: RepoTask, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env = RepoEnv(task)
    try:
        initial = env.source_snapshot()
        for action in patches:
            observation, done, _ = execute_action(env, action)
            if done or not observation.startswith("PATCH_OK"):
                raise AssertionError(f"cannot collapse failed patch: {observation}")
        if not (env.visible_pass() and env.hidden_pass()):
            raise AssertionError("cannot collapse an incorrect patch sequence")
        final = env.source_snapshot()
        result = [
            {"tool": "patch", "path": path, "old": initial[path], "new": final[path]}
            for path in sorted(initial)
            if initial[path] != final[path]
        ]
        replay = replay_patch_set(task, result)
        if not (replay["visible"] and replay["hidden"]):
            raise AssertionError("collapsed per-file patches do not replay")
        return result
    finally:
        env.close()


def select_success(task: RepoTask, trajectories: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in trajectories if row.get("workspace_success")]
    ranked = []
    for row in candidates:
        patches = minimize_patches(task, _valid_patch_steps(row))
        try:
            collapsed = collapse_patches_by_file(task, patches)
        except AssertionError:
            continue
        if len(collapsed) != 1:
            continue
        replay = replay_patch_set(task, collapsed)
        if replay["visible"] and replay["hidden"]:
            ranked.append((len(patches), row.get("sampled_tokens", 0), row.get("turns", 0), row, collapsed))
    if not ranked:
        return None
    _n, _tokens, _turns, selected, patches = min(ranked, key=lambda item: item[:3])
    return {"trajectory": selected, "patches": patches}


def oracle_final_patch(task: RepoTask) -> list[dict[str, Any]]:
    patches = [
        {"tool": "patch", "path": patch.path, "old": patch.old, "new": patch.new}
        for patch in task.oracle_patches
    ]
    return collapse_patches_by_file(task, patches)


def _prefix_plan(action: dict[str, Any], observation: str | None = None) -> str:
    tool = action["tool"]
    if tool == "read":
        return "Inspect the current implementation before choosing the next edit."
    if tool == "search":
        return "Search public repository evidence for the issue's compatibility reference."
    if tool == "patch":
        return "Apply the currently proposed issue-directed edit with an exact replacement."
    if tool == "test":
        return "Run the visible tests to obtain executable feedback before committing."
    if tool == "submit":
        return "Visible tests pass after the edit, so commit the verified workspace."
    raise KeyError((tool, observation))


def _transition_plan(transition: str) -> str:
    return {
        "start_to_inspect_source": "Inspect the relevant implementation before changing source.",
        "ambiguous_source_to_inspect_evidence": (
            "The source leaves an edge-case contract underdetermined. Inspect the public "
            "repository evidence that distinguishes the plausible policies before patching."
        ),
        "evidence_to_policy_patch": (
            "Bind the observed public contract to the implementation and make the first "
            "policy-faithful patch."
        ),
        "explicit_source_to_patch": (
            "The issue explicitly determines the edge behavior; make the scoped repair "
            "from the inspected source without unnecessary acquisition."
        ),
        "rejected_patch_to_changed_patch": (
            "The exact patch was rejected. Change the replacement anchor using the current source; "
            "do not repeat the rejected patch."
        ),
        "failed_test_to_diagnose": (
            "The visible suite failed after an edit. Inspect the current implementation and the "
            "reported assertion before revising."
        ),
        "diagnosis_to_changed_patch": (
            "The inspected state is an underfix. Revise the current source with a changed patch "
            "that addresses the remaining requirement."
        ),
        "patch_ok_to_verify": "Source changed successfully; run the visible tests now.",
        "passed_test_to_commit": "Visible tests pass after the final edit; submit the workspace.",
    }[transition]


def _messages_after(task: RepoTask, actions: list[dict[str, Any]]) -> tuple[list[dict[str, str]], dict]:
    env = RepoEnv(task)
    messages = initial_messages(task, env)
    observations = []
    try:
        for action in actions:
            observation, done, _submitted = execute_action(env, action)
            if done:
                raise AssertionError("training prefix cannot submit")
            messages.append({
                "role": "assistant",
                "content": assistant_content(_prefix_plan(action, observation), _action_key(action)),
            })
            messages.append({
                "role": "user",
                "content": f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call.",
            })
            observations.append(observation)
        state = {
            "visible_pass": env.visible_pass(),
            "hidden_pass": env.hidden_pass(),
            "workspace_digest": env.workspace_digest(),
            "observations": observations,
        }
        return copy.deepcopy(messages), state
    finally:
        env.close()


def _row(
    task: RepoTask,
    transition: str,
    prefix_actions: list[dict[str, Any]],
    target_action: dict[str, Any],
    conditioning: str,
) -> dict[str, Any]:
    operator = operator_for(target_action)
    if operator != TRANSITION_OPERATOR[transition]:
        raise AssertionError((transition, operator))
    messages, state = _messages_after(task, prefix_actions)
    return {
        "id": f"{task.task_id}-{conditioning}-{transition}",
        "task_id": task.task_id,
        "family": task.family,
        "split": task.split,
        "kind": f"repo_{conditioning}",
        "conditioning": conditioning,
        "transition": transition,
        "operator": operator,
        "messages": messages,
        "think": _transition_plan(transition),
        "answer": _action_key(target_action),
        "think_weight": 0.0,
        "prefix_actions": copy.deepcopy(prefix_actions),
        "target_action": copy.deepcopy(target_action),
        "state_receipt": state,
        "task_manifest": task.public_manifest(),
    }


def transition_rows(task: RepoTask, final_patches: list[dict[str, Any]]) -> tuple[list[dict], list[dict], dict]:
    if len(final_patches) != 1 or len(task.partial_patches) != 1:
        raise AssertionError("recovery banking requires one changed file")
    happy_patch = copy.deepcopy(final_patches[0])
    partial = task.partial_patches[0]
    if happy_patch["path"] != partial.path:
        raise AssertionError("partial and final repairs must edit the same file")
    partial_action = {"tool": "patch", "path": partial.path, "old": partial.old, "new": partial.new}
    partial_replay = replay_patch_set(task, [partial_action])
    if partial_replay["visible"] or partial_replay["hidden"]:
        raise AssertionError("partial repair unexpectedly passes")
    final_replay = replay_patch_set(task, [happy_patch])
    if not (final_replay["visible"] and final_replay["hidden"]):
        raise AssertionError("selected final repair does not pass")
    partial_source = partial_replay["snapshot"][partial.path]
    final_source = final_replay["snapshot"][partial.path]
    changed_patch = {
        "tool": "patch", "path": partial.path, "old": partial_source, "new": final_source,
    }
    if not (replay_patch_set(task, [partial_action, changed_patch])["visible"]
            and replay_patch_set(task, [partial_action, changed_patch])["hidden"]):
        raise AssertionError("changed recovery patch does not pass")
    read_action = {"tool": "read", "path": partial.path}
    test_action = {"tool": "test"}
    submit_action = {"tool": "submit"}
    rejected_action = {
        "tool": "patch",
        "path": partial.path,
        "old": partial.old + "\n# stale recovery anchor",
        "new": partial.new,
    }
    recovery_specs = {
        "start_to_inspect_source": ([], read_action),
        "explicit_source_to_patch": ([read_action], happy_patch),
        "rejected_patch_to_changed_patch": ([read_action, rejected_action], happy_patch),
        "failed_test_to_diagnose": ([read_action, partial_action, test_action], read_action),
        "diagnosis_to_changed_patch": ([read_action, partial_action, test_action, read_action], changed_patch),
        "patch_ok_to_verify": ([read_action, happy_patch], test_action),
        "passed_test_to_commit": ([read_action, happy_patch, test_action], submit_action),
    }
    happy_by_operator = {
        "INSPECT": ([], read_action),
        "PATCH": ([read_action], happy_patch),
        "VERIFY": ([read_action, happy_patch], test_action),
        "COMMIT": ([read_action, happy_patch, test_action], submit_action),
    }
    recovery = [
        _row(task, transition, *recovery_specs[transition], conditioning="recovery")
        for transition in RECOVERY_TRANSITIONS
    ]
    happy = [
        _row(
            task,
            transition,
            *happy_by_operator[TRANSITION_OPERATOR[transition]],
            conditioning="happy_control",
        )
        for transition in RECOVERY_TRANSITIONS
    ]
    receipt = {
        "task_id": task.task_id,
        "family": task.family,
        "final_patch_sha256": hashlib.sha256(_action_key(happy_patch).encode()).hexdigest(),
        "partial_visible_pass": partial_replay["visible"],
        "partial_hidden_pass": partial_replay["hidden"],
        "final_visible_pass": final_replay["visible"],
        "final_hidden_pass": final_replay["hidden"],
        "transitions": list(RECOVERY_TRANSITIONS),
    }
    return recovery, happy, receipt


def evidence_transition_rows(task: RepoTask) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build three executable rows that install acquisition and binding.

    VERIFY and COMMIT remain fully represented by prior complete-loop replay.
    Keeping aligned oracle patches out of later new-task prefixes makes the
    within-dyad shuffled-label arm a clean binding-direction control.
    """
    final_patches = oracle_final_patch(task)
    if len(final_patches) != 1:
        raise AssertionError("evidence task must collapse to one patch")
    patch_action = copy.deepcopy(final_patches[0])
    source_action = {"tool": "read", "path": patch_action["path"]}
    evidence_action = {"tool": "search", "query": task.acquisition_query}
    final_replay = replay_patch_set(task, [patch_action])
    if not (final_replay["visible"] and final_replay["hidden"]):
        raise AssertionError("evidence target patch does not pass")
    specs = {
        "start_to_inspect_source": ([], source_action),
        "ambiguous_source_to_inspect_evidence": ([source_action], evidence_action),
        "evidence_to_policy_patch": ([source_action, evidence_action], patch_action),
    }
    rows = [
        _row(task, transition, *specs[transition], conditioning="evidence_binding")
        for transition in specs
    ]
    for row in rows:
        row["pair_id"] = task.pair_id
        row["branch"] = task.branch
        row["evidence_channel"] = task.evidence_channel
        row["evidence_path"] = task.evidence_path
        row["evidence_path_regime"] = task.evidence_path_regime
        row["acquisition_query_skin"] = task.acquisition_query_skin
        row["acquisition_query"] = task.acquisition_query
        row["explicit_contract"] = task.explicit_contract
        row["think_weight"] = 0.0
    receipt = {
        "task_id": task.task_id,
        "pair_id": task.pair_id,
        "branch": task.branch,
        "family": task.family,
        "evidence_channel": task.evidence_channel,
        "evidence_path": task.evidence_path,
        "evidence_path_regime": task.evidence_path_regime,
        "acquisition_query_skin": task.acquisition_query_skin,
        "acquisition_query_sha256": hashlib.sha256(
            task.acquisition_query.encode()
        ).hexdigest(),
        "explicit_contract": task.explicit_contract,
        "final_patch_sha256": hashlib.sha256(
            _action_key(patch_action).encode()
        ).hexdigest(),
        "final_visible_pass": final_replay["visible"],
        "final_hidden_pass": final_replay["hidden"],
        "transitions": list(specs),
    }
    return rows, receipt


def build_banks(tasks: list[RepoTask], trajectories: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in trajectories or []:
        by_task[item["task_id"]].append(item)
    recovery_rows: list[dict[str, Any]] = []
    happy_rows: list[dict[str, Any]] = []
    receipts = []
    selections = []
    uncovered = []
    for task in tasks:
        if trajectories is None:
            selected = {"trajectory": None, "patches": oracle_final_patch(task)}
        else:
            selected = select_success(task, by_task.get(task.task_id, []))
        if selected is None:
            uncovered.append(task.task_id)
            continue
        recovery, happy, receipt = transition_rows(task, selected["patches"])
        recovery_rows.extend(recovery)
        happy_rows.extend(happy)
        receipts.append(receipt)
        source = selected["trajectory"]
        selections.append({
            "task_id": task.task_id,
            "source_trajectory": None if source is None else source["trajectory"],
            "source_turns": None if source is None else source["turns"],
            "source_sampled_tokens": None if source is None else source["sampled_tokens"],
            "source_patch_count": None if source is None else len(_valid_patch_steps(source)),
            "compressed_patch_count": len(selected["patches"]),
        })
    action_rows = copy.deepcopy(recovery_rows)
    reason_rows = copy.deepcopy(recovery_rows)
    for row in action_rows:
        row["id"] = row["id"].replace("-recovery-", "-recovery-action-")
        row["kind"] = "repo_recovery_action"
        row["think_weight"] = 0.0
    for row in reason_rows:
        row["id"] = row["id"].replace("-recovery-", "-recovery-reason-")
        row["kind"] = "repo_recovery_reason"
    for row in happy_rows:
        row["kind"] = "repo_happy_action"
        row["think_weight"] = 0.0
    return {
        "recovery_action_rows": action_rows,
        "recovery_reason_rows": reason_rows,
        "happy_action_rows": happy_rows,
        "replay_receipts": receipts,
        "selections": selections,
        "uncovered_task_ids": uncovered,
        "transition_counts": dict(Counter(row["transition"] for row in action_rows)),
        "operator_counts": dict(Counter(row["operator"] for row in action_rows)),
    }


def calibrate_transition_loss_mass(
    rows: list[dict[str, Any]],
    tokenizer,
    *,
    target_transition_action_mass: float,
    plan_mass_fraction: float,
    max_length: int,
) -> dict[str, Any]:
    token_rows = []
    for row in rows:
        prompt = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
        think_part = row["think"].strip() + "\n</think>\n\n"
        answer_part = row["answer"].strip() + tokenizer.eos_token
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        think_ids = tokenizer(prompt + think_part, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt + think_part + answer_part, add_special_tokens=False)["input_ids"]
        if full_ids[:len(prompt_ids)] != prompt_ids or full_ids[:len(think_ids)] != think_ids:
            raise AssertionError(f"token boundary merge: {row['id']}")
        token_rows.append({
            "row": row,
            "operator": row["operator"],
            "transition": row["transition"],
            "prompt_tokens": len(prompt_ids),
            "think_tokens": len(think_ids) - len(prompt_ids),
            "answer_tokens": len(full_ids) - len(think_ids),
            "total_tokens": len(full_ids),
        })
    overlength = [item["row"]["id"] for item in token_rows if item["total_tokens"] > max_length]
    if overlength:
        raise AssertionError(f"overlength recovery rows: {overlength[:5]}")
    strata = {
        operator: sorted({item["transition"] for item in token_rows if item["operator"] == operator})
        for operator in OPERATORS
    }
    if any(not values for values in strata.values()):
        raise AssertionError(f"missing operator strata: {strata}")
    raw_action = Counter()
    raw_plan = Counter()
    for item in token_rows:
        raw_action[item["transition"]] += item["answer_tokens"]
        raw_plan[item["transition"]] += item["think_tokens"]
    action_target = {
        transition: target_transition_action_mass for transition in TRANSITIONS
    }
    action_weight = {
        transition: action_target[transition] / raw_action[transition]
        for transition in TRANSITIONS
    }
    plan_weight = {}
    for transition in TRANSITIONS:
        if plan_mass_fraction == 0:
            plan_weight[transition] = 0.0
        else:
            weighted_raw = raw_plan[transition] * action_weight[transition]
            plan_weight[transition] = (
                plan_mass_fraction * action_target[transition] / weighted_raw
            )
    for item in token_rows:
        row = item["row"]
        transition = item["transition"]
        row["row_weight"] = action_weight[transition]
        row["think_weight"] = plan_weight[transition]
        row["token_counts"] = {
            "prompt": item["prompt_tokens"],
            "think": item["think_tokens"],
            "answer": item["answer_tokens"],
            "total": item["total_tokens"],
        }
    transition_action_mass = {
        transition: sum(
            item["answer_tokens"] * item["row"]["row_weight"]
            for item in token_rows if item["transition"] == transition
        )
        for transition in TRANSITIONS
    }
    operator_action_mass = {
        operator: sum(
            transition_action_mass[transition] for transition in strata[operator]
        )
        for operator in OPERATORS
    }
    transition_plan_mass = {
        transition: sum(
            item["think_tokens"] * item["row"]["row_weight"] * item["row"]["think_weight"]
            for item in token_rows if item["transition"] == transition
        )
        for transition in TRANSITIONS
    }
    return {
        "rows": len(rows),
        "tasks": len({row["task_id"] for row in rows}),
        "operator_transition_strata": strata,
        "raw_action_tokens_by_transition": dict(raw_action),
        "raw_plan_tokens_by_transition": dict(raw_plan),
        "action_row_weight_by_transition": action_weight,
        "plan_token_weight_by_transition": plan_weight,
        "target_transition_action_mass": target_transition_action_mass,
        "plan_mass_fraction": plan_mass_fraction,
        "weighted_action_mass_by_transition": transition_action_mass,
        "weighted_action_mass_by_operator": operator_action_mass,
        "weighted_plan_mass_by_transition": transition_plan_mass,
        "max_total_tokens": max(item["total_tokens"] for item in token_rows),
        "overlength_row_ids": overlength,
    }


def assert_firewall_clean(value: Any, tasks: Iterable[RepoTask]) -> None:
    rendered = json.dumps(value, sort_keys=True)
    forbidden_keys = (
        '"hidden_test"', '"oracle_patches"', '"partial_patches"', '"oracle"',
    )
    for key in forbidden_keys:
        if key in rendered:
            raise AssertionError(f"private field leaked: {key}")
    for task in tasks:
        if task.hidden_test and task.hidden_test in rendered:
            raise AssertionError(f"hidden executable leaked for {task.task_id}")
