"""Batched Qwen coding-agent loop over private-test procedural repositories.

Only public repository state and visible-test output enter the conversation.  The
hidden executable is evaluated host-side at submission or budget exhaustion and
is reduced to booleans in serialized receipts.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from typing import Any

from repo_tasks import RepoEnv, RepoTask


SYSTEM = """You are a coding agent working in a small Python repository.
At every turn, reason privately and then emit exactly one JSON tool call with no markdown.
Available calls:
{"tool":"tree"}
{"tool":"read","path":"src/file.py"}
{"tool":"search","query":"literal text"}
{"tool":"patch","path":"src/file.py","old":"exact old text","new":"replacement text"}
{"tool":"test"}
{"tool":"submit"}
Patch performs one exact string replacement. Inspect before editing. After the final edit,
run the visible tests and submit only after they pass. Hidden tests are evaluated privately.
Do not emit prose outside the JSON object."""

TOOL_OPERATOR = {
    "tree": "INSPECT",
    "read": "INSPECT",
    "search": "INSPECT",
    "patch": "PATCH",
    "test": "VERIFY",
    "submit": "COMMIT",
}


def initial_messages(task: RepoTask, env: RepoEnv) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": (
                f"ISSUE\n{task.issue}\n\nFILES\n{env.tree()}\n\n"
                "Repair the repository with the available tools. The listed visible tests "
                "may be inspected and run; hidden checks are private."
            ),
        },
    ]


def parse_action(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first JSON tool object from the answer region only."""
    answer = text.rsplit("</think>", 1)[-1]
    decoder = json.JSONDecoder()
    for index, char in enumerate(answer):
        if char != "{":
            continue
        try:
            value, _end = decoder.raw_decode(answer[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("tool"), str):
            return value, "ok"
    return None, "no_json_tool_call_in_answer"


def operator_for(action: dict[str, Any] | None) -> str:
    if not action:
        return "INVALID"
    return TOOL_OPERATOR.get(str(action.get("tool", "")), "INVALID")


def execute_action(env: RepoEnv, action: dict[str, Any]) -> tuple[str, bool, bool]:
    """Execute one public tool call.

    Returns ``(observation, done, submitted_success)``.  The hidden result is not
    included in the observation.
    """
    tool = str(action.get("tool", ""))
    env.tool_calls += 1
    if tool == "tree":
        return env.tree(), False, False
    if tool == "read":
        return env.read(str(action.get("path", ""))), False, False
    if tool == "search":
        return env.search(str(action.get("query", ""))), False, False
    if tool == "patch":
        return env.patch(
            str(action.get("path", "")),
            str(action.get("old", "")),
            str(action.get("new", "")),
        ), False, False
    if tool == "test":
        return env.run_visible(), False, False
    if tool == "submit":
        # Submission grading is the conjunction of public regression tests and
        # private edge cases. Neither result is revealed to the model.
        return "SUBMITTED", True, env.visible_pass() and env.hidden_pass()
    return f"TOOL_ERROR: unknown tool {tool!r}", False, False


def assistant_content(think: str, answer: str) -> str:
    """Render the generated suffix used as an earlier assistant message."""
    return f"{think.strip()}\n</think>\n\n{answer.strip()}"


@dataclass
class Episode:
    task: RepoTask
    trajectory: int
    env: RepoEnv = field(init=False)
    messages: list[dict[str, str]] = field(init=False)
    turns: int = 0
    sampled_tokens: int = 0
    invalid_actions: int = 0
    submitted: bool = False
    submitted_success: bool = False
    done: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.env = RepoEnv(self.task)
        self.messages = initial_messages(self.task, self.env)

    @property
    def record_id(self) -> str:
        return f"{self.task.task_id}-traj{self.trajectory}-turn{self.turns + 1}"

    def consume(self, output: dict[str, Any]) -> None:
        self.turns += 1
        raw = str(output.get("text", ""))
        self.sampled_tokens += int(output.get("n_sampled_tokens", 0))
        before = self.env.workspace_digest()
        context = copy.deepcopy(self.messages)
        self.messages.append({"role": "assistant", "content": raw})
        action, parse_status = parse_action(raw)
        if action is None:
            self.invalid_actions += 1
            observation = (
                f"TOOL_ERROR: {parse_status}. Emit exactly one valid JSON tool call."
            )
            done = False
            submitted_success = False
        else:
            observation, done, submitted_success = execute_action(self.env, action)
            if operator_for(action) == "INVALID" or observation.startswith("TOOL_ERROR"):
                self.invalid_actions += 1
        after = self.env.workspace_digest()
        if action and action.get("tool") == "submit":
            self.submitted = True
            self.submitted_success = submitted_success
        self.done = done
        self.steps.append(
            {
                "turn": self.turns,
                "context": context,
                "raw": raw,
                "action": action,
                "parse_status": parse_status,
                "operator": operator_for(action),
                "observation": observation[-6000:],
                "before_digest": before,
                "after_digest": after,
                "n_thinking_tokens": int(output.get("n_thinking_tokens", 0)),
                "n_answer_tokens": int(output.get("n_answer_tokens", 0)),
                "n_sampled_tokens": int(output.get("n_sampled_tokens", 0)),
                "thinking_closed": bool(output.get("thinking_closed", False)),
                "forced_close": bool(output.get("forced_close", False)),
            }
        )
        if not self.done:
            self.messages.append(
                {
                    "role": "user",
                    "content": f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call.",
                }
            )

    def finish(self) -> dict[str, Any]:
        final_visible_pass = self.env.visible_pass()
        final_hidden_pass = self.env.hidden_pass()
        workspace_success = final_visible_pass and final_hidden_pass
        patch_turns = [
            step["turn"]
            for step in self.steps
            if (step.get("action") or {}).get("tool") == "patch"
            and step["before_digest"] != step["after_digest"]
        ]
        final_patch_turn = max(patch_turns, default=None)
        passing_test_turns = [
            step["turn"]
            for step in self.steps
            if (step.get("action") or {}).get("tool") == "test"
            and step["observation"].startswith("PASS")
            and final_patch_turn is not None
            and step["turn"] > final_patch_turn
        ]
        verified_after_final_patch = bool(passing_test_turns)
        submit_turns = [
            step["turn"]
            for step in self.steps
            if (step.get("action") or {}).get("tool") == "submit"
        ]
        commit_after_pass = bool(
            passing_test_turns
            and submit_turns
            and max(submit_turns) > min(passing_test_turns)
        )
        result = {
            "task": self.task.public_manifest(),
            "task_id": self.task.task_id,
            "family": self.task.family,
            "split": self.task.split,
            "trajectory": self.trajectory,
            "success": workspace_success,
            "workspace_success": workspace_success,
            "final_visible_pass": final_visible_pass,
            "final_hidden_pass": final_hidden_pass,
            "submitted": self.submitted,
            "submitted_success": self.submitted_success,
            "turns": self.turns,
            "sampled_tokens": self.sampled_tokens,
            "tool_calls": self.env.tool_calls,
            "patch_calls": self.env.patch_calls,
            "test_calls": self.env.test_calls,
            "invalid_actions": self.invalid_actions,
            "verified_after_final_patch": verified_after_final_patch,
            "commit_after_pass": commit_after_pass,
            "operator_sequence": [step["operator"] for step in self.steps],
            "final_workspace_digest": self.env.workspace_digest(),
            "steps": self.steps,
        }
        self.env.close()
        return result


def run_episodes(
    runner,
    specs: list[tuple[RepoTask, int]],
    sampling,
    max_turns: int,
    progress: bool = True,
) -> list[dict[str, Any]]:
    """Continuously batch independent repositories until submit or turn cap."""
    episodes = [Episode(task, trajectory) for task, trajectory in specs]
    active = episodes
    loop_round = 0
    while active:
        loop_round += 1
        records = [{"id": ep.record_id, "messages": ep.messages} for ep in active]
        rows, _summary = runner.generate(records, sampling)
        if len(rows) != len(active):
            raise RuntimeError("runner changed episode count")
        next_active = []
        for episode, row in zip(active, rows):
            outputs = row.get("outputs", [])
            if len(outputs) != 1:
                raise RuntimeError("agent loop requires exactly one output per state")
            episode.consume(outputs[0])
            if not episode.done and episode.turns < max_turns:
                next_active.append(episode)
        if progress:
            completed = len(episodes) - len(next_active)
            print(
                f"[repo-agent] turn={loop_round} "
                f"active_before={len(active)} active_after={len(next_active)} "
                f"completed={completed}/{len(episodes)}",
                flush=True,
            )
        active = next_active
    return [episode.finish() for episode in episodes]
