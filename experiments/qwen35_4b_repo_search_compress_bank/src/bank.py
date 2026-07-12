"""Replay-verified trajectory compression and operator-balanced SFT banking."""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from repo_agent import SYSTEM, assistant_content, execute_action, initial_messages, operator_for
from repo_tasks import RepoEnv, RepoTask


OPERATORS = ("INSPECT", "PATCH", "VERIFY", "COMMIT")


def _action_key(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def _valid_patch_steps(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        copy.deepcopy(step["action"])
        for step in trajectory["steps"]
        if step.get("action", {}).get("tool") == "patch"
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
                }
        visible = env.visible_pass()
        hidden = env.hidden_pass()
        return {
            "visible": visible,
            "hidden": hidden,
            "observations": observations,
            "digest": env.workspace_digest(),
        }
    finally:
        env.close()


def minimize_patches(task: RepoTask, patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Greedily delete trajectory patches while executable correctness survives."""
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


def select_success(task: RepoTask, trajectories: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in trajectories if row.get("workspace_success")]
    if not candidates:
        return None
    ranked = []
    for row in candidates:
        patches = _valid_patch_steps(row)
        minimized = minimize_patches(task, patches)
        replay = replay_patch_set(task, minimized)
        if replay["visible"] and replay["hidden"]:
            ranked.append((len(minimized), row.get("sampled_tokens", 0), row.get("turns", 0), row, minimized))
    if not ranked:
        return None
    _n, _tokens, _turns, selected, patches = min(ranked, key=lambda item: item[:3])
    return {"trajectory": selected, "patches": patches}


def canonical_actions(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    paths = list(dict.fromkeys(str(action["path"]) for action in patches))
    return (
        [{"tool": "read", "path": path} for path in paths]
        + copy.deepcopy(patches)
        + [{"tool": "test"}, {"tool": "submit"}]
    )


def compact_plan(action: dict[str, Any], observation_before: str | None = None) -> str:
    tool = action["tool"]
    if tool == "read":
        return f"Inspect {action['path']} to locate the issue-relevant implementation before editing."
    if tool == "patch":
        return f"Apply the minimal exact repair in {action['path']}; verification must follow the final edit."
    if tool == "test":
        return "The intended edit is in place. Run the visible tests before committing the workspace."
    if tool == "submit":
        if not (observation_before or "").startswith("PASS"):
            raise AssertionError("canonical submit is not immediately preceded by passing tests")
        return "The visible tests pass after the final edit. Commit the verified workspace now."
    raise KeyError(tool)


def replay_canonical(task: RepoTask, patches: list[dict[str, Any]]) -> tuple[list[dict], dict]:
    """Replay a compact trace and emit one supervised row at every operator."""
    env = RepoEnv(task)
    messages = initial_messages(task, env)
    rows: list[dict[str, Any]] = []
    previous_observation: str | None = None
    try:
        for step_index, action in enumerate(canonical_actions(patches)):
            operator = operator_for(action)
            if operator not in OPERATORS:
                raise AssertionError(operator)
            plan = compact_plan(action, previous_observation)
            answer = _action_key(action)
            row = {
                "id": f"{task.task_id}-compact-{step_index:02d}",
                "task_id": task.task_id,
                "family": task.family,
                "split": task.split,
                "kind": "repo_compact_plan_action",
                "operator": operator,
                "messages": copy.deepcopy(messages),
                "think": plan,
                "answer": answer,
                "think_weight": 0.2,
                "task_manifest": task.public_manifest(),
            }
            rows.append(row)
            observation, done, submitted_success = execute_action(env, action)
            if action["tool"] == "patch" and not observation.startswith("PATCH_OK"):
                raise AssertionError(f"canonical patch failed: {observation}")
            messages.append({"role": "assistant", "content": assistant_content(plan, answer)})
            if not done:
                messages.append(
                    {
                        "role": "user",
                        "content": f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call.",
                    }
                )
            previous_observation = observation
        visible = env.visible_pass()
        hidden = env.hidden_pass()
        if not (visible and hidden and submitted_success):
            raise AssertionError(
                f"canonical replay failed: visible={visible} hidden={hidden} submit={submitted_success}"
            )
        receipt = {
            "task_id": task.task_id,
            "family": task.family,
            "row_ids": [row["id"] for row in rows],
            "patch_count": len(patches),
            "final_workspace_digest": env.workspace_digest(),
            "visible_pass": visible,
            "hidden_pass": hidden,
            "submitted_success": submitted_success,
        }
        return rows, receipt
    finally:
        env.close()


def balance_operator_weights(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(row["operator"] for row in rows)
    missing = set(OPERATORS) - set(counts)
    if missing:
        raise AssertionError(f"missing operator targets: {sorted(missing)}")
    target_mass = len(rows) / len(OPERATORS)
    weights = {operator: target_mass / counts[operator] for operator in OPERATORS}
    for row in rows:
        row["row_weight"] = weights[row["operator"]]
    masses = {
        operator: sum(row["row_weight"] for row in rows if row["operator"] == operator)
        for operator in OPERATORS
    }
    return {"counts": dict(counts), "weights": weights, "loss_mass": masses}


def action_only_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = copy.deepcopy(rows)
    for row in result:
        row["id"] = row["id"].replace("-compact-", "-action-only-")
        row["kind"] = "repo_action_only"
        row["think"] = "Choose the next tool action."
        row["think_weight"] = 0.0
    return result


def build_banks(tasks: list[RepoTask], trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        by_task[row["task_id"]].append(row)
    compact: list[dict[str, Any]] = []
    receipts = []
    uncovered = []
    selections = []
    for task in tasks:
        selected = select_success(task, by_task.get(task.task_id, []))
        if selected is None:
            uncovered.append(task.task_id)
            continue
        rows, receipt = replay_canonical(task, selected["patches"])
        compact.extend(rows)
        receipts.append(receipt)
        selections.append(
            {
                "task_id": task.task_id,
                "source_trajectory": selected["trajectory"]["trajectory"],
                "source_turns": selected["trajectory"]["turns"],
                "source_sampled_tokens": selected["trajectory"]["sampled_tokens"],
                "source_patch_count": len(_valid_patch_steps(selected["trajectory"])),
                "compressed_patch_count": len(selected["patches"]),
            }
        )
    balance = balance_operator_weights(compact) if compact else None
    action_only = action_only_rows(compact)
    payload = json.dumps(compact, sort_keys=True, separators=(",", ":")).encode()
    return {
        "compact_rows": compact,
        "action_only_rows": action_only,
        "replay_receipts": receipts,
        "selections": selections,
        "uncovered_task_ids": uncovered,
        "operator_balance": balance,
        "compact_sha256": hashlib.sha256(payload).hexdigest(),
    }


def assert_firewall_clean(value: Any, tasks: Iterable[RepoTask]) -> None:
    """Fail if serialized output contains private code or private field names."""
    rendered = json.dumps(value, sort_keys=True)
    forbidden_keys = ('"hidden_test"', '"oracle_patches"', '"oracle"')
    for key in forbidden_keys:
        if key in rendered:
            raise AssertionError(f"private field leaked: {key}")
    for task in tasks:
        if task.hidden_test and task.hidden_test in rendered:
            raise AssertionError(f"hidden executable leaked for {task.task_id}")
