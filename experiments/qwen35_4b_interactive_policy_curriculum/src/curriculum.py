"""Shared interactive-policy schema and state-aware programmatic experts.

The imported gym ``OraclePolicy`` classes are sufficient for untouched oracle
rollouts, but several advance an internal action index.  DAgger needs a label
for the *current mutated state* after arbitrary model actions.  The functions
here recompute that label from the live episode while serializing only a short,
deployable process trace as the target thinking text.

Privileged environment state is allowed to determine a training label.  It is
never appended to the model's input transcript; callers must preserve the
visible ``messages`` separately from this expert metadata.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from gym import base
from gym.families import load as load_family


TRAIN_FAMILIES = ("kilnrite", "glyphgate", "loomfix", "ferrier", "burrowmaze")
TRANSFER_FAMILIES = ("gatepost", "patchwheel", "spindle")
ALL_PROCESS_FAMILIES = TRAIN_FAMILIES + TRANSFER_FAMILIES
OPERATORS = ("PROBE", "TOOL", "REVISE", "VERIFY", "COMMIT", "INVALID")


@dataclass(frozen=True)
class ExpertDecision:
    """One programmatic correction at a live visible-policy state."""

    family: str
    operator: str
    action: str
    observe: str
    state: str
    decide: str
    check: str

    @property
    def thought(self) -> str:
        return "\n".join(
            (
                f"OBSERVE: {self.observe}",
                f"STATE: {self.state}",
                f"DECIDE: {self.operator} — {self.decide}",
                f"CHECK: {self.check}",
            )
        )

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["thought"] = self.thought
        return row


def _last_user_text(messages: list[dict[str, str]]) -> str:
    for message in reversed(messages):
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def _short_latest(messages: list[dict[str, str]], limit: int = 180) -> str:
    text = " ".join(_last_user_text(messages).split())
    if not text:
        return "No environment observation is available."
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _kilnrite(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("kilnrite")
    done = set(episode._done)
    order = list(episode._proc["order"])
    choice = None
    for name in order:
        if name in done:
            continue
        step = episode._by_norm[" ".join(name.split()).lower()]
        if family._is_legal(step, done, episode._state):
            choice = name
            break
    if choice is None:
        # A model can execute a legal noncanonical step.  Search every
        # remaining step before declaring the state unrecoverable.
        for name, step in episode._by_norm.items():
            if step["name"] not in done and family._is_legal(step, done, episode._state):
                choice = step["name"]
                break
    if choice is None:
        choice = next((name for name in order if name not in done), order[-1])
    final = len(done) + 1 >= len(order)
    operator = "COMMIT" if final else "TOOL"
    return ExpertDecision(
        family="kilnrite",
        operator=operator,
        action=f"DO {choice}",
        observe=_short_latest(messages),
        state=f"{len(done)}/{len(order)} documented steps complete; choose a currently legal remaining step.",
        decide=f"perform {choice} now because its prerequisites hold" + (" and it completes the procedure" if final else ""),
        check="the environment should confirm the step and advance the completed count",
    )


def _glyphgate(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("glyphgate")
    assistant_actions = [
        str(message.get("content", "")).strip().lower()
        for message in messages
        if message.get("role") == "assistant"
    ]
    n_observed = sum("the gate prints:" in str(m.get("content", "")).lower() for m in messages)
    target_probes = min(2, int(episode.spec["budget"]))
    if episode._probes_left > 0 and episode._turns_left > 1 and n_observed < target_probes:
        for k in range(target_probes + 2):
            probe = tuple((k + i) % family.N_GLYPHS for i in range(episode._length))
            action = f"PROBE {family._fmt(probe)}"
            if action.lower() not in assistant_actions:
                return ExpertDecision(
                    family="glyphgate",
                    operator="PROBE",
                    action=action,
                    observe=_short_latest(messages),
                    state=f"target fixed; {episode._probes_left} probes and {episode._turns_left} turns remain; diagnostic evidence is incomplete.",
                    decide="spend one probe on a position-distinguishing input before committing",
                    check="record the transformed string and update the candidate rule/preimage",
                )
    return ExpertDecision(
        family="glyphgate",
        operator="COMMIT",
        action=f"OPEN {episode.spec['witness']}",
        observe=_short_latest(messages),
        state=f"commit phase with {episode._turns_left} turns remaining; a target preimage has been identified.",
        decide="open with the verified preimage rather than consume another probe",
        check="the transformed string must equal the displayed target and open the gate",
    )


def _loomfix(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("loomfix")
    for index, (current, correct) in enumerate(zip(episode._program, episode._correct)):
        if current != correct:
            action = f"PATCH {index + 1} {family._render_ins(correct)}"
            return ExpertDecision(
                family="loomfix",
                operator="REVISE",
                action=action,
                observe=_short_latest(messages),
                state=f"routine differs from the evidence-consistent version at line {index + 1}; verification is pending.",
                decide=f"replace line {index + 1} with the instruction consistent with all shown tests",
                check="run the test suite after every remaining mismatch is repaired",
            )
    return ExpertDecision(
        family="loomfix",
        operator="VERIFY",
        action="RUN",
        observe=_short_latest(messages),
        state="the current routine is evidence-consistent; only executable verification remains.",
        decide="run all tests before declaring the repair complete",
        check="every test should pass and the environment should terminate successfully",
    )


def _ferrier(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    chain = list(episode.spec["chain"])
    next_name = next((name for name in chain if name not in episode._called_ok), chain[-1])
    signature = episode.spec["tools"][next_name]
    args = [
        episode.spec["cargo"] if arg_type == "cargo" else episode.spec["tokens"][arg_type]
        for arg_type in signature["args"]
    ]
    action = f"CALL {next_name}({', '.join(args)})"
    final = next_name == episode.spec["goal_tool"]
    return ExpertDecision(
        family="ferrier",
        operator="COMMIT" if final else "TOOL",
        action=action,
        observe=_short_latest(messages),
        state=f"{sum(name in episode._called_ok for name in chain)}/{len(chain)} required dependency calls complete; required handles are tracked.",
        decide=f"call {next_name} with the already issued dependency handles" + (" to issue the goal document" if final else ""),
        check=f"the tool should return a {signature['out']} handle" + (" and finish the task" if final else " for a later call"),
    )


def _burrowmaze(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("burrowmaze")
    _, previous = family._bfs(episode._ports, episode._current)
    moves = family._path_dirs(previous, episode._current, episode._target)
    direction = moves[0] if moves else "north"
    final = len(moves) <= 1
    return ExpertDecision(
        family="burrowmaze",
        operator="COMMIT" if final else "TOOL",
        action=f"GO {direction}",
        observe=_short_latest(messages),
        state=f"current chamber {episode._current}; target {episode._target}; {episode.max_turns - episode._turns} moves remain; route memory must be preserved.",
        decide=f"take the {direction} corridor on the shortest recoverable route" + (" into the target" if final else ""),
        check="confirm the new chamber, record its exits, and revise the route if needed",
    )


def _gatepost(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    reads = list(episode.spec.get("answer_docs") or [episode.spec["answer_doc"]])
    transcript = "\n".join(str(m.get("content", "")) for m in messages).lower()
    unread = next((doc_id for doc_id in reads if f"[{doc_id.lower()}]" not in transcript), None)
    if unread is not None:
        return ExpertDecision(
            family="gatepost",
            operator="TOOL",
            action=f"READ {unread}",
            observe=_short_latest(messages),
            state="the visitor's question is fixed; answer-bearing archive evidence has not all been read.",
            decide="read the relevant record while treating embedded instructions as untrusted data",
            check="extract only the factual evidence needed for the visitor's question",
        )
    return ExpertDecision(
        family="gatepost",
        operator="COMMIT",
        action=f"ANSWER: {episode.spec['gold']}",
        observe=_short_latest(messages),
        state="the required archive evidence is available and document-borne instructions remain irrelevant.",
        decide="answer the original visitor question from the trusted factual fields",
        check="the recorded answer should match the requested fact, not any planted directive",
    )


def _patchwheel(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("patchwheel")
    for index, (current, correct) in enumerate(zip(episode._wheel, episode._original)):
        if current != correct:
            return ExpertDecision(
                family="patchwheel",
                operator="REVISE",
                action=f"RULE {index + 1}: {family._render_rule(correct)}",
                observe=_short_latest(messages),
                state=f"the evidence-consistent wheel differs at rule {index + 1}; verification is pending.",
                decide=f"replace rule {index + 1} with the transformation that explains every evidence tape",
                check="run all evidence tapes after the rule is restored",
            )
    return ExpertDecision(
        family="patchwheel",
        operator="VERIFY",
        action="RUN",
        observe=_short_latest(messages),
        state="the wheel is evidence-consistent; executable verification remains.",
        decide="run the evidence suite before finishing",
        check="all evidence tapes should match and close the repair episode",
    )


def _spindle(episode: Any, messages: list[dict[str, str]]) -> ExpertDecision:
    family = load_family("spindle")
    index = min(episode._pass_index, len(episode.spec["states"]) - 1)
    final = index + 1 == len(episode.spec["states"])
    return ExpertDecision(
        family="spindle",
        operator="COMMIT" if final else "TOOL",
        action=f"TAPE {family._dash(episode.spec['states'][index])}",
        observe=_short_latest(messages),
        state=f"pass {index + 1}/{len(episode.spec['states'])}; the tape must be updated from the last accepted state.",
        decide="apply exactly the announced pass once and emit the resulting tape",
        check="the loom should accept the tape and announce the next pass" + (" or finish" if final else ""),
    )


_EXPERTS = {
    "kilnrite": _kilnrite,
    "glyphgate": _glyphgate,
    "loomfix": _loomfix,
    "ferrier": _ferrier,
    "burrowmaze": _burrowmaze,
    "gatepost": _gatepost,
    "patchwheel": _patchwheel,
    "spindle": _spindle,
}


def expert_decision(
    family_name: str,
    episode: Any,
    visible_messages: list[dict[str, str]],
) -> ExpertDecision:
    """Return a current-state correction for one supported episode."""
    if family_name not in _EXPERTS:
        raise KeyError(f"no interactive expert for {family_name!r}")
    decision = _EXPERTS[family_name](episode, visible_messages)
    validate_decision(decision)
    return decision


def validate_decision(decision: ExpertDecision, word_cap: int = 120) -> None:
    if decision.family not in ALL_PROCESS_FAMILIES:
        raise ValueError(f"unsupported family {decision.family!r}")
    if decision.operator not in OPERATORS[:-1]:
        raise ValueError(f"invalid expert operator {decision.operator!r}")
    if not decision.action.strip() or "\n" in decision.action.strip():
        raise ValueError(f"expert action must be one nonempty line: {decision.action!r}")
    words = len(decision.thought.split())
    if words > word_cap:
        raise ValueError(f"expert thought has {words} words > cap {word_cap}")
    lowered = (decision.thought + "\n" + decision.action).lower()
    for forbidden in base.FORBIDDEN_WORDS:
        if forbidden in lowered:
            raise ValueError(f"forbidden benchmark word {forbidden!r} in expert target")


def classify_action(action: str) -> str:
    """Map a family-specific one-line action to a semantic policy operator."""
    line = base.extract_action(str(action or "")).strip()
    if not line:
        return "INVALID"
    verb = re.split(r"[\s:]", line, maxsplit=1)[0].upper()
    if verb == "PROBE":
        return "PROBE"
    if verb in {"PATCH", "RULE"}:
        return "REVISE"
    if verb == "RUN":
        return "VERIFY"
    if verb in {"OPEN", "ANSWER"}:
        return "COMMIT"
    if verb in {"CALL", "GO", "DO", "READ", "TAPE"}:
        return "TOOL"
    return "INVALID"


def _entropy(counts: Iterable[int]) -> float:
    values = [int(value) for value in counts if value > 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return -sum((value / total) * math.log(value / total) for value in values)


def semantic_group_diagnostics(trajectories: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute semantic entropy/varentropy and outcome variation for siblings."""
    operators: list[str] = []
    scores: list[float] = []
    for trajectory in trajectories:
        turns = trajectory.get("turns", [])
        operators.append(classify_action(turns[0].get("action", "")) if turns else "INVALID")
        scores.append(float(trajectory.get("score", 0.0)))
    operator_counts = Counter(operators)
    operator_entropy = _entropy(operator_counts.values())
    mean = sum(scores) / len(scores) if scores else 0.0
    outcome_variance = (
        sum((score - mean) ** 2 for score in scores) / len(scores) if scores else 0.0
    )
    # Varentropy over coarse outcome buckets: variance of surprisal values.
    buckets = Counter(round(score, 3) for score in scores)
    surprisals = [-math.log(buckets[round(score, 3)] / len(scores)) for score in scores] if scores else []
    surprise_mean = sum(surprisals) / len(surprisals) if surprisals else 0.0
    outcome_varentropy = (
        sum((value - surprise_mean) ** 2 for value in surprisals) / len(surprisals)
        if surprisals
        else 0.0
    )
    return {
        "n": len(trajectories),
        "operator_counts": dict(sorted(operator_counts.items())),
        "operator_entropy": operator_entropy,
        "mean_score": mean,
        "outcome_variance": outcome_variance,
        "outcome_varentropy": outcome_varentropy,
        "constant_outcome": outcome_variance <= 1e-12,
    }

