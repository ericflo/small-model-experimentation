"""Minimal batched coding-agent loop over the procedural repository environment."""

from __future__ import annotations

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
{"tool":"test"}
{"tool":"patch","path":"src/file.py","old":"exact old text","new":"replacement text"}
{"tool":"submit"}
Patch uses one exact string replacement. You may inspect and test repeatedly. Submit only when the repair is ready.
Do not emit prose outside the JSON object."""


def parse_action(text: str) -> tuple[dict[str, Any] | None, str]:
    answer = text.split("</think>")[-1]
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
    return None, "no_json_tool_call"


@dataclass
class Episode:
    task: RepoTask
    trajectory: int
    env: RepoEnv = field(init=False)
    messages: list[dict] = field(init=False)
    turns: int = 0
    sampled_tokens: int = 0
    invalid_actions: int = 0
    submitted: bool = False
    success: bool = False
    done: bool = False
    actions: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.env = RepoEnv(self.task)
        self.messages = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": (
                f"ISSUE\n{self.task.issue}\n\nFILES\n{self.env.tree()}\n\n"
                "Repair the repository. Visible tests may be run with the test tool; hidden tests are used only after submission."
            )},
        ]

    @property
    def record_id(self) -> str:
        return f"{self.task.task_id}-traj{self.trajectory}-turn{self.turns}"

    def consume(self, output: dict) -> None:
        self.turns += 1
        self.sampled_tokens += int(output.get("n_sampled_tokens", 0))
        raw = output["text"]
        self.messages.append({"role": "assistant", "content": raw})
        action, parse_status = parse_action(raw)
        if action is None:
            self.invalid_actions += 1
            observation = f"TOOL_ERROR: {parse_status}. Emit exactly one valid JSON tool call."
            self.actions.append({"turn": self.turns, "parse": parse_status, "raw": raw[-1000:]})
            self.messages.append({"role": "user", "content": observation})
            return
        tool = action["tool"]
        self.env.tool_calls += 1
        observation: str
        if tool == "tree":
            observation = self.env.tree()
        elif tool == "read":
            observation = self.env.read(str(action.get("path", "")))
        elif tool == "search":
            observation = self.env.search(str(action.get("query", "")))
        elif tool == "test":
            observation = self.env.run_visible()
        elif tool == "patch":
            observation = self.env.patch(
                str(action.get("path", "")), str(action.get("old", "")),
                str(action.get("new", "")))
        elif tool == "submit":
            self.submitted = True
            self.success = self.env.hidden_pass()
            self.done = True
            observation = "SUBMITTED"
        else:
            self.invalid_actions += 1
            observation = f"TOOL_ERROR: unknown tool {tool!r}"
        self.actions.append({"turn": self.turns, "action": action,
                             "observation": observation[-4000:]})
        if not self.done:
            self.messages.append({"role": "user", "content":
                                  f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call."})

    def finish(self) -> dict:
        patch_correct = self.env.hidden_pass()
        result = {
            "task_id": self.task.task_id, "family": self.task.family,
            # Primary coding success is final-workspace hidden-test success at
            # budget exhaustion, as in software-engineering benchmarks. An
            # explicit submit can end early but is reported separately.
            "trajectory": self.trajectory, "success": patch_correct,
            "submitted_success": self.success,
            "patch_correct": patch_correct, "submitted": self.submitted,
            "turns": self.turns, "sampled_tokens": self.sampled_tokens,
            "tool_calls": self.env.tool_calls, "patch_calls": self.env.patch_calls,
            "invalid_actions": self.invalid_actions, "actions": self.actions,
        }
        self.env.close()
        return result


def run_episodes(runner, specs: list[tuple[RepoTask, int]], sampling,
                 max_turns: int) -> list[dict]:
    """Continuously batch active episodes, preserving independent repo states."""
    episodes = [Episode(task, trajectory) for task, trajectory in specs]
    active = episodes
    while active:
        records = [{"id": ep.record_id, "messages": ep.messages} for ep in active]
        rows, _summary = runner.generate(records, sampling)
        if len(rows) != len(active):
            raise RuntimeError("runner changed episode count")
        next_active = []
        for ep, row in zip(active, rows):
            outputs = row.get("outputs", [])
            if len(outputs) != 1:
                raise RuntimeError("agent loop requires exactly one output per state")
            ep.consume(outputs[0])
            if not ep.done and ep.turns < max_turns:
                next_active.append(ep)
        active = next_active
    return [ep.finish() for ep in episodes]
