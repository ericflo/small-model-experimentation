"""Batched coding-agent loop with counterfactual public-evidence states.

Only public repository state and visible tool output enter the conversation.
Host-private executables score first-patch and terminal behavior without
returning labels to the model.  Controlled starts isolate ambiguous source,
evidence-injected feasibility, rejected patches, and failed visible tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from repo_tasks import RepoEnv, RepoTask


SYSTEM = """You are a coding agent working in a small Python repository.
At every turn, reason privately and then emit exactly one JSON tool call with no markdown.
Available calls:
{"tool":"tree"}
{"tool":"read","path":"public/file.py"}
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


def action_text(action: dict[str, Any]) -> str:
    return json.dumps(action, sort_keys=True, separators=(",", ":"))


def recovery_hint(observation: str) -> str:
    if observation.startswith("ERROR: old text matched"):
        return (
            "RECOVERY RULE: the exact patch was rejected. Do not repeat it byte-for-byte; "
            "use the current file contents and issue a changed patch."
        )
    if observation.startswith("FAIL"):
        return (
            "RECOVERY RULE: visible tests failed. Diagnose the reported failure, revise "
            "the source with a changed patch, and re-run tests before submitting."
        )
    return ""


def _record_prefix_action(
    env: RepoEnv,
    messages: list[dict[str, str]],
    action: dict[str, Any],
    plan: str,
) -> dict[str, Any]:
    before = env.workspace_digest()
    observation, done, _submitted_success = execute_action(env, action)
    if done:
        raise AssertionError("recovery prefix cannot submit")
    after = env.workspace_digest()
    messages.append({"role": "assistant", "content": assistant_content(plan, action_text(action))})
    messages.append({
        "role": "user",
        "content": f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call.",
    })
    return {
        "action": copy.deepcopy(action),
        "operator": operator_for(action),
        "observation": observation[-6000:],
        "before_digest": before,
        "after_digest": after,
    }


def bootstrap_scenario(
    task: RepoTask,
    env: RepoEnv,
    messages: list[dict[str, str]],
    scenario: str,
) -> list[dict[str, Any]]:
    if scenario == "normal":
        return []
    if scenario in (
        "ambiguous_source", "evidence_injected",
        "nondiscriminating_search_injected",
    ):
        if len(task.oracle_patches) != 1:
            raise AssertionError("controlled specification state requires one target patch")
        read_action = {"tool": "read", "path": task.oracle_patches[0].path}
        steps = [_record_prefix_action(
            env,
            messages,
            read_action,
            "Inspect the unresolved implementation before deciding how to change it.",
        )]
        if scenario == "evidence_injected":
            evidence = _record_prefix_action(
                env,
                messages,
                {"tool": "search", "query": task.acquisition_query},
                "Search the public repository for the issue's compatibility reference.",
            )
            if (
                task.evidence_path not in evidence["observation"]
                or task.evidence_marker not in evidence["observation"]
            ):
                raise AssertionError(
                    "evidence-injected prefix did not expose the discriminator"
                )
            steps.append(evidence)
        elif scenario == "nondiscriminating_search_injected":
            source_path = task.oracle_patches[0].path
            candidates = sorted(
                path for path in task.files
                if path != source_path
                and path != task.evidence_path
                and path not in ("README.md", "tests/test_visible.py")
                and not path.endswith("/__init__.py")
            )
            if not candidates:
                raise AssertionError("search control has no non-evidence public file")
            query = None
            for path in candidates:
                for line in task.files[path].splitlines():
                    candidate = line.strip()
                    if not (8 <= len(candidate) <= 120):
                        continue
                    observation = env.search(candidate)
                    if (
                        observation != "NO_MATCHES"
                        and task.evidence_path not in observation
                        and task.evidence_marker not in observation
                    ):
                        query = candidate
                        break
                if query is not None:
                    break
            if query is None:
                raise AssertionError("could not construct a non-discriminating search")
            control_search = _record_prefix_action(
                env,
                messages,
                {"tool": "search", "query": query},
                "Run one deterministic public search that does not expose the policy discriminator.",
            )
            if (
                task.evidence_path in control_search["observation"]
                or task.evidence_marker in control_search["observation"]
            ):
                raise AssertionError("non-discriminating search leaked the evidence")
            steps.append(control_search)
        return steps
    if scenario not in ("rejected_patch", "failed_test"):
        raise ValueError(f"unknown recovery scenario: {scenario}")
    if len(task.partial_patches) != 1:
        raise AssertionError("registered recovery scenarios require one partial patch")
    partial = task.partial_patches[0]
    read_action = {"tool": "read", "path": partial.path}
    steps = [_record_prefix_action(
        env, messages, read_action,
        "Inspect the implementation before attempting the issue-directed edit.",
    )]
    if scenario == "rejected_patch":
        rejected = {
            "tool": "patch",
            "path": partial.path,
            "old": partial.old + "\n# stale recovery anchor",
            "new": partial.new,
        }
        step = _record_prefix_action(
            env, messages, rejected,
            "Attempt a plausible partial repair using an exact replacement anchor.",
        )
        if not step["observation"].startswith("ERROR: old text matched 0 times"):
            raise AssertionError(f"rejected-patch scenario did not reject: {step['observation']}")
        steps.append(step)
        return steps
    partial_action = {
        "tool": "patch", "path": partial.path, "old": partial.old, "new": partial.new,
    }
    step = _record_prefix_action(
        env, messages, partial_action,
        "Apply a plausible partial repair, then verify it against visible tests.",
    )
    if not step["observation"].startswith("PATCH_OK"):
        raise AssertionError(f"partial patch did not apply: {step['observation']}")
    steps.append(step)
    test_step = _record_prefix_action(
        env, messages, {"tool": "test"},
        "Run the visible suite after the partial repair.",
    )
    if not test_step["observation"].startswith("FAIL"):
        raise AssertionError("partial patch must remain a visible-test failure")
    if env.hidden_pass():
        raise AssertionError("partial patch must remain a hidden-test failure")
    steps.append(test_step)
    return steps


@dataclass
class Episode:
    task: RepoTask
    trajectory: int
    scenario: str = "normal"
    scaffold: bool = False
    env: RepoEnv = field(init=False)
    messages: list[dict[str, str]] = field(init=False)
    turns: int = 0
    sampled_tokens: int = 0
    logical_model_input_tokens: int = 0
    injected_tokens: int = 0
    invalid_actions: int = 0
    submitted: bool = False
    submitted_success: bool = False
    done: bool = False
    steps: list[dict[str, Any]] = field(default_factory=list)
    prefix_steps: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.env = RepoEnv(self.task)
        self.messages = initial_messages(self.task, self.env)
        self.prefix_steps = bootstrap_scenario(
            self.task, self.env, self.messages, self.scenario
        )
        if self.scaffold and self.prefix_steps:
            hint = recovery_hint(self.prefix_steps[-1]["observation"])
            if hint:
                self.messages[-1]["content"] += f"\n\n{hint}"

    @property
    def record_id(self) -> str:
        return (
            f"{self.task.task_id}-{self.scenario}-traj{self.trajectory}"
            f"-turn{self.turns + 1}"
        )

    def consume(self, output: dict[str, Any]) -> None:
        self.turns += 1
        raw = str(output.get("text", ""))
        self.sampled_tokens += int(output.get("n_sampled_tokens", 0))
        self.logical_model_input_tokens += int(
            output.get("n_stage1_prompt_tokens", 0)
        ) + int(output.get("n_stage2_prompt_tokens", 0))
        self.injected_tokens += int(output.get("n_injected_tokens", 0))
        before = self.env.workspace_digest()
        context = copy.deepcopy(self.messages)
        action, parse_status = parse_action(raw)
        if action is None:
            history_content = raw
        else:
            think = raw.rsplit("</think>", 1)[0] if "</think>" in raw else ""
            history_content = assistant_content(think, action_text(action))
        # Preserve raw output in the receipt, but prevent post-call answer run-on
        # from becoming a spurious later-turn instruction in every arm.
        self.messages.append({"role": "assistant", "content": history_content})
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
        patch_changed = bool(
            action
            and action.get("tool") == "patch"
            and before != after
        )
        if patch_changed:
            workspace_visible_pass, workspace_hidden_pass = self.env.score_workspace()
        else:
            workspace_visible_pass = None
            workspace_hidden_pass = None
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
                "n_stage1_prompt_tokens": int(
                    output.get("n_stage1_prompt_tokens", 0)
                ),
                "n_stage2_prompt_tokens": int(
                    output.get("n_stage2_prompt_tokens", 0)
                ),
                "n_injected_tokens": int(output.get("n_injected_tokens", 0)),
                "logical_model_tokens": (
                    int(output.get("n_stage1_prompt_tokens", 0))
                    + int(output.get("n_stage2_prompt_tokens", 0))
                    + int(output.get("n_sampled_tokens", 0))
                ),
                "thinking_closed": bool(output.get("thinking_closed", False)),
                "forced_close": bool(output.get("forced_close", False)),
                "finish_reason": output.get("finish_reason"),
                "stage1_finish_reason": output.get("stage1_finish_reason"),
                "stage2_finish_reason": output.get("stage2_finish_reason"),
                "history_canonicalized": action is not None,
                "workspace_visible_pass_after": workspace_visible_pass,
                "workspace_hidden_pass_after": workspace_hidden_pass,
            }
        )
        if not self.done:
            hint = recovery_hint(observation) if self.scaffold else ""
            suffix = f"\n\n{hint}" if hint else ""
            self.messages.append(
                {
                    "role": "user",
                    "content": (
                        f"TOOL RESULT\n{observation}\n\nContinue with one JSON tool call."
                        f"{suffix}"
                    ),
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
        rejected_first_two = self.steps[:2] if self.scenario == "rejected_patch" else []
        changed_patch_steps = [
            step for step in self.steps
            if step["operator"] == "PATCH"
            and step["before_digest"] != step["after_digest"]
        ]
        first_changed_patch = changed_patch_steps[0] if changed_patch_steps else None
        before_first_patch = (
            self.steps[: self.steps.index(first_changed_patch)]
            if first_changed_patch is not None
            else self.steps
        )

        def acquired_evidence(step: dict[str, Any]) -> bool:
            action = step.get("action") or {}
            if action.get("tool") == "read" and action.get("path") == self.task.evidence_path:
                return True
            return bool(
                action.get("tool") == "search"
                and self.task.evidence_path in step.get("observation", "")
                and self.task.evidence_marker in step.get("observation", "")
            )

        evidence_steps_before_patch = [
            step for step in before_first_patch if acquired_evidence(step)
        ]
        prefix_evidence_steps = [
            step for step in self.prefix_steps if acquired_evidence(step)
        ]
        evidence_paths_read = [
            str((step.get("action") or {}).get("path"))
            for step in [*self.prefix_steps, *before_first_patch]
            if (step.get("action") or {}).get("tool") == "read"
        ]
        rejected_changed_within_two = any(
            step["operator"] == "PATCH"
            and step["before_digest"] != step["after_digest"]
            for step in rejected_first_two
        )
        rejected_valid_changed_within_two = bool(
            rejected_changed_within_two
            and (
                rejected_first_two[0]["operator"] == "PATCH"
                or (
                    len(rejected_first_two) >= 2
                    and rejected_first_two[0]["operator"] == "INSPECT"
                    and rejected_first_two[1]["operator"] == "PATCH"
                    and rejected_first_two[1]["before_digest"]
                    != rejected_first_two[1]["after_digest"]
                )
            )
        )
        result = {
            "task": self.task.public_manifest(),
            "task_id": self.task.task_id,
            "case_id": f"{self.task.task_id}::{self.scenario}",
            "family": self.task.family,
            "pair_id": self.task.pair_id,
            "branch": self.task.branch,
            "evidence_channel": self.task.evidence_channel,
            "evidence_path": self.task.evidence_path,
            "explicit_contract": self.task.explicit_contract,
            "split": self.task.split,
            "scenario": self.scenario,
            "scaffold": self.scaffold,
            "trajectory": self.trajectory,
            "success": workspace_success,
            "workspace_success": workspace_success,
            "final_visible_pass": final_visible_pass,
            "final_hidden_pass": final_hidden_pass,
            "submitted": self.submitted,
            "submitted_success": self.submitted_success,
            "turns": self.turns,
            "sampled_tokens": self.sampled_tokens,
            "logical_model_input_tokens": self.logical_model_input_tokens,
            "injected_tokens": self.injected_tokens,
            "logical_model_tokens": (
                self.logical_model_input_tokens
                + self.sampled_tokens
            ),
            "tool_calls": len(self.steps),
            "prefix_tool_calls": len(self.prefix_steps),
            "patch_calls": sum(
                (step.get("action") or {}).get("tool") == "patch" for step in self.steps
            ),
            "test_calls": sum(
                (step.get("action") or {}).get("tool") == "test" for step in self.steps
            ),
            "invalid_actions": self.invalid_actions,
            "verified_after_final_patch": verified_after_final_patch,
            "commit_after_pass": commit_after_pass,
            "operator_sequence": [step["operator"] for step in self.steps],
            "prefix_operator_sequence": [step["operator"] for step in self.prefix_steps],
            "first_generated_operator": self.steps[0]["operator"] if self.steps else None,
            "first_generated_action": self.steps[0].get("action") if self.steps else None,
            "first_changed_patch_turn": (
                first_changed_patch["turn"] if first_changed_patch else None
            ),
            "first_changed_patch_action": (
                first_changed_patch.get("action") if first_changed_patch else None
            ),
            "first_changed_patch_visible_pass": bool(
                first_changed_patch
                and first_changed_patch.get("workspace_visible_pass_after")
            ),
            "first_changed_patch_hidden_pass": bool(
                first_changed_patch
                and first_changed_patch.get("workspace_hidden_pass_after")
            ),
            "first_changed_patch_full_correct": bool(
                first_changed_patch
                and first_changed_patch.get("workspace_visible_pass_after")
                and first_changed_patch.get("workspace_hidden_pass_after")
            ),
            "first_changed_patch_before_generated_test": bool(
                first_changed_patch
                and not any(
                    (step.get("action") or {}).get("tool") == "test"
                    and step["turn"] < first_changed_patch["turn"]
                    for step in self.steps
                )
            ),
            "evidence_acquired_before_first_patch": bool(
                prefix_evidence_steps or evidence_steps_before_patch
            ),
            "evidence_injected_by_prefix": bool(prefix_evidence_steps),
            "evidence_acquisition_turn": (
                evidence_steps_before_patch[0]["turn"]
                if evidence_steps_before_patch else 0 if prefix_evidence_steps else None
            ),
            "evidence_paths_read_before_first_patch": evidence_paths_read,
            "unnecessary_evidence_before_first_patch": bool(
                self.task.explicit_contract
                and any(
                    (step.get("action") or {}).get("tool") == "search"
                    or (
                        (step.get("action") or {}).get("tool") == "read"
                        and (step.get("action") or {}).get("path")
                        != self.task.oracle_patches[0].path
                    )
                    for step in before_first_patch
                )
            ),
            "any_search_before_first_patch": any(
                (step.get("action") or {}).get("tool") == "search"
                for step in before_first_patch
            ),
            "non_source_inspects_before_first_patch": sum(
                step["operator"] == "INSPECT"
                and not (
                    (step.get("action") or {}).get("tool") == "read"
                    and (step.get("action") or {}).get("path")
                    == self.task.oracle_patches[0].path
                )
                for step in before_first_patch
            ),
            "rejected_patch_changed_immediately": bool(
                self.scenario == "rejected_patch"
                and self.steps
                and self.steps[0]["operator"] == "PATCH"
                and self.steps[0]["before_digest"] != self.steps[0]["after_digest"]
                and action_text(self.steps[0].get("action") or {})
                != action_text(self.prefix_steps[-1].get("action") or {})
            ),
            "rejected_patch_changed_within_two": rejected_changed_within_two,
            "rejected_patch_valid_changed_within_two": rejected_valid_changed_within_two,
            "rejected_patch_first_two_operators": [
                step["operator"] for step in rejected_first_two
            ],
            "failed_test_diagnose_or_revise_immediately": bool(
                self.scenario == "failed_test"
                and self.steps
                and (
                    self.steps[0]["operator"] == "INSPECT"
                    or (
                        self.steps[0]["operator"] == "PATCH"
                        and self.steps[0]["before_digest"] != self.steps[0]["after_digest"]
                    )
                )
            ),
            "failed_test_changed_patch_within_two": bool(
                self.scenario == "failed_test"
                and any(
                    step["operator"] == "PATCH"
                    and step["before_digest"] != step["after_digest"]
                    for step in self.steps[:2]
                )
            ),
            "final_workspace_digest": self.env.workspace_digest(),
            "prefix_steps": self.prefix_steps,
            "steps": self.steps,
        }
        self.env.close()
        return result


def run_episodes(
    runner,
    specs: list[tuple],
    sampling,
    max_turns: int,
    scaffold: bool = False,
    progress: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Continuously batch independent repositories until submit or turn cap."""
    episodes = []
    for spec in specs:
        if len(spec) == 2:
            task, trajectory = spec
            scenario = "normal"
        elif len(spec) == 3:
            task, trajectory, scenario = spec
        else:
            raise ValueError(f"invalid episode spec: {spec!r}")
        episodes.append(Episode(task, trajectory, scenario=scenario, scaffold=scaffold))
    active = episodes
    loop_round = 0
    runner_summaries = []
    while active:
        loop_round += 1
        records = [{"id": ep.record_id, "messages": ep.messages} for ep in active]
        rows, summary = runner.generate(records, sampling)
        runner_summaries.append(summary)
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
    return [episode.finish() for episode in episodes], runner_summaries
