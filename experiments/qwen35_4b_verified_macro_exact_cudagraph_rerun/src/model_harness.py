#!/usr/bin/env python3
"""Prompt, parsing, and exact-accounting helpers for the macro experiment.

All model generation in this module goes through the experiment-local
``vllm_runner.py``.  Prompt construction and response parsing are deliberately
model-free so dataset construction, dry runs, and unit tests never allocate a
GPU or load a checkpoint.
"""

from __future__ import annotations

import dataclasses
import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Protocol

try:  # Package import (for callers importing ``src.model_harness``).
    from .vllm_runner import (
        MODEL_ID,
        MODEL_REVISION,
        EngineConfig,
        SamplingConfig,
        VLLMRunner,
    )
except ImportError:  # Script/test import with this experiment's src/ on sys.path.
    from vllm_runner import (  # type: ignore[no-redef]
        MODEL_ID,
        MODEL_REVISION,
        EngineConfig,
        SamplingConfig,
        VLLMRunner,
    )


REQUIRED_MODEL_ID = "Qwen/Qwen3.5-4B"
if MODEL_ID != REQUIRED_MODEL_ID:
    raise RuntimeError(
        f"macro harness requires exactly {REQUIRED_MODEL_ID}, local runner has {MODEL_ID}"
    )

_TOKEN_PATTERN = r"[A-Z][A-Z0-9_]*"
_TOKEN_RE = re.compile(rf"^{_TOKEN_PATTERN}$")
_MACRO_LINE_RE = re.compile(
    rf"^MACRO\s*:\s*({_TOKEN_PATTERN})\s*=\s*"
    rf"({_TOKEN_PATTERN}(?:\s*\|\s*{_TOKEN_PATTERN}){{1,2}})\s*$"
)
_PROGRAM_LINE_RE = re.compile(
    rf"^PROGRAM\s*:\s*({_TOKEN_PATTERN}(?:\s*\|\s*{_TOKEN_PATTERN})*)\s*$"
)
_FENCE_RE = re.compile(
    r"^```(?:[A-Za-z0-9_-]+)?[ \t]*\r?\n(?P<body>.*?)\r?\n```$", re.DOTALL
)
_TERMINAL_MARKERS = ("<|endoftext|>", "<|im_end|>")


class RunnerLike(Protocol):
    """The small part of :class:`VLLMRunner` used by this harness."""

    def generate(
        self,
        records: Sequence[dict[str, Any]],
        sampling: SamplingConfig,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]: ...


@dataclasses.dataclass(frozen=True)
class MacroProposal:
    """One model-proposed name and its 2--3 primitive composition."""

    name: str
    expansion: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class MacroDefinition:
    """A frozen solver-facing macro token and its verified expansion."""

    token: str
    expansion: tuple[str, ...]
    description: str = ""


@dataclasses.dataclass(frozen=True)
class CompletionText:
    """Raw semantic completion retained with its vLLM request identity."""

    record_id: str
    sample_index: int
    text: str
    token_ids: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class ParsedProgramCompletion:
    """A solver completion; malformed generations retain their parse error."""

    record_id: str
    sample_index: int
    raw_text: str
    program: tuple[str, ...] | None
    parse_error: str | None


@dataclasses.dataclass(frozen=True)
class ParsedMacroCompletion:
    """A macro-invention completion; malformed generations remain auditable."""

    record_id: str
    sample_index: int
    raw_text: str
    proposals: tuple[MacroProposal, ...] | None
    parse_error: str | None


@dataclasses.dataclass(frozen=True)
class RejectedMacroLine:
    """One nonblank proposal line rejected by the tolerant line extractor."""

    line_number: int
    text: str
    reason: str


@dataclasses.dataclass(frozen=True)
class MacroLineExtraction:
    """Auditable line-local macro harvesting from one answer region.

    ``proposals`` contains the first ``max_macros`` independently valid lines.
    Later valid lines are counted and set ``extra_valid_lines_capped`` rather
    than turning an otherwise useful completion into an all-or-nothing failure.
    """

    proposals: tuple[MacroProposal, ...]
    rejected_nonblank_lines: tuple[RejectedMacroLine, ...]
    extra_valid_lines_capped: bool
    total_valid_lines: int


@dataclasses.dataclass(frozen=True)
class CompletionAccounting:
    """Exact counts copied from one vLLM completion (never re-tokenized)."""

    record_id: str
    sample_index: int
    unique_request_prompt_tokens: int
    stage1_logical_prompt_tokens: int
    stage2_logical_prompt_tokens: int
    sampled_tokens: int
    injected_tokens: int
    completion_tokens: int
    thinking_tokens: int
    answer_tokens: int
    terminal_tokens_trimmed: int


@dataclasses.dataclass(frozen=True)
class BatchAccounting:
    """Exact aggregate compute counts and their per-completion provenance."""

    requests: int
    completions: int
    unique_input_prompt_tokens: int
    stage1_logical_prompt_tokens: int
    stage2_logical_prompt_tokens: int
    logical_model_input_tokens: int
    sampled_tokens: int
    injected_tokens: int
    completion_tokens: int
    thinking_tokens: int
    answer_tokens: int
    terminal_tokens_trimmed: int
    per_completion: tuple[CompletionAccounting, ...]


@dataclasses.dataclass(frozen=True)
class VLLMBatchResult:
    """vLLM rows, runner metadata, and independently checked exact counts."""

    rows: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    accounting: BatchAccounting


def _require_token(token: Any, *, where: str) -> str:
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise ValueError(f"{where} must be an uppercase DSL token, got {token!r}")
    return token


def _single_line(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be a non-empty string")
    if "\n" in value or "\r" in value:
        raise ValueError(f"{where} must be one line")
    return value.strip()


def _primitive_items(primitives: Mapping[str, str]) -> list[tuple[str, str]]:
    if not isinstance(primitives, Mapping) or not primitives:
        raise ValueError("primitives must be a non-empty token-to-description mapping")
    items: list[tuple[str, str]] = []
    for token, description in primitives.items():
        items.append(
            (
                _require_token(token, where="primitive name"),
                _single_line(description, where=f"description for {token}"),
            )
        )
    return items


def _program_tokens(
    value: Any,
    *,
    allowed: set[str],
    where: str,
    min_length: int = 1,
) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ValueError(f"{where} must be a sequence of DSL tokens")
    tokens = tuple(_require_token(token, where=where) for token in value)
    if len(tokens) < min_length:
        raise ValueError(f"{where} must contain at least {min_length} token(s)")
    unknown = sorted(set(tokens) - allowed)
    if unknown:
        raise ValueError(f"{where} uses tokens outside its inventory: {unknown}")
    return tokens


def _json(value: Any, *, where: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where} is not JSON-serializable") from exc


def _io_pairs(value: Any, *, where: str) -> list[tuple[Any, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not value:
        raise ValueError(f"{where} must be a non-empty sequence of I/O pairs")
    pairs: list[tuple[Any, Any]] = []
    for index, pair in enumerate(value):
        if isinstance(pair, Mapping):
            if "input" not in pair or "output" not in pair:
                raise ValueError(f"{where}[{index}] needs input and output keys")
            input_value, output_value = pair["input"], pair["output"]
        elif not isinstance(pair, (str, bytes)) and isinstance(pair, Sequence) and len(pair) == 2:
            input_value, output_value = pair
        else:
            raise ValueError(f"{where}[{index}] must be a pair or input/output mapping")
        _json(input_value, where=f"{where}[{index}].input")
        _json(output_value, where=f"{where}[{index}].output")
        pairs.append((input_value, output_value))
    return pairs


def _verified_train_examples(
    examples: Sequence[Mapping[str, Any]], primitive_names: set[str]
) -> list[tuple[str, tuple[str, ...], list[tuple[Any, Any]]]]:
    if isinstance(examples, (str, bytes)) or not examples:
        raise ValueError("verified_programs must be a non-empty sequence")
    normalized: list[tuple[str, tuple[str, ...], list[tuple[Any, Any]]]] = []
    seen_ids: set[str] = set()
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError(f"verified_programs[{index}] must be a mapping")
        if example.get("split") != "train":
            raise ValueError(
                f"verified_programs[{index}] is not explicitly split='train'; refusing leakage"
            )
        if example.get("verified") is not True:
            raise ValueError(
                f"verified_programs[{index}] is not explicitly verified=True"
            )
        example_id = _single_line(example.get("id"), where=f"verified_programs[{index}].id")
        if example_id in seen_ids:
            raise ValueError(f"duplicate verified program id: {example_id!r}")
        seen_ids.add(example_id)
        program = _program_tokens(
            example.get("program"),
            allowed=primitive_names,
            where=f"program for {example_id}",
        )
        io_pairs = _io_pairs(example.get("io"), where=f"io for {example_id}")
        normalized.append((example_id, program, io_pairs))
    return normalized


def build_macro_proposal_messages(
    *,
    primitives: Mapping[str, str],
    verified_programs: Sequence[Mapping[str, Any]],
    max_macros: int,
) -> list[dict[str, str]]:
    """Build a leakage-checked prompt from verified, train-only programs.

    Every input example must explicitly carry ``split='train'`` and
    ``verified=True``.  This fails closed if an evaluation or merely proposed
    program is accidentally passed to the invention stage.
    """

    if not isinstance(max_macros, int) or isinstance(max_macros, bool) or max_macros < 1:
        raise ValueError("max_macros must be a positive integer")
    primitive_items = _primitive_items(primitives)
    primitive_names = {name for name, _ in primitive_items}
    examples = _verified_train_examples(verified_programs, primitive_names)

    system = (
        "You discover reusable compositions in a straight-line transformation DSL. "
        "The supplied programs are verified training data only. Propose concise macros "
        "that recur or plausibly compress reusable structure. A macro expansion must use "
        "exactly 2 or 3 BASE primitive tokens. In the final answer, return only lines of "
        "the form `MACRO: DESCRIPTIVE_NAME = OP1 | OP2` or "
        "`MACRO: DESCRIPTIVE_NAME = OP1 | OP2 | OP3`. Names and tokens must be uppercase "
        "identifiers. Return exactly the requested number of macros. Do not return Python, "
        "markdown, commentary, scores, `NONE`, or prose."
    )
    lines = ["BASE PRIMITIVES (the only tokens allowed in expansions):"]
    lines.extend(f"{name}: {description}" for name, description in primitive_items)
    lines.append("")
    lines.append("VERIFIED TRAIN-ONLY SOLVED PROGRAMS:")
    # I/O was validated above but is intentionally omitted here: recurrence is
    # defined over verified program tokens, and hundreds of rendered I/O pairs
    # made the v1 prompt spend its entire reasoning budget re-reading examples.
    for example_id, program, _validated_io_pairs in examples:
        lines.append(f"[{example_id}] PROGRAM: " + " | ".join(program))
    lines.extend(
        [
            "",
            f"PROPOSE EXACTLY {max_macros} MACROS.",
            "Prefer reusable fragments over one-off aliases for an entire program.",
            f"FINAL ANSWER: exactly {max_macros} macro lines only.",
        ]
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_macro_proposal_record(
    record_id: str,
    *,
    primitives: Mapping[str, str],
    verified_programs: Sequence[Mapping[str, Any]],
    max_macros: int,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one JSONL-compatible vLLM record for macro invention."""

    record_id = _single_line(record_id, where="record_id")
    return {
        "id": record_id,
        "messages": build_macro_proposal_messages(
            primitives=primitives,
            verified_programs=verified_programs,
            max_macros=max_macros,
        ),
        "meta": {"prompt_kind": "macro_proposal", **dict(meta or {})},
    }


def _macro_definitions(
    macros: Sequence[MacroDefinition] | Mapping[str, Sequence[str]],
    *,
    primitive_names: set[str],
) -> list[MacroDefinition]:
    if isinstance(macros, Mapping):
        raw = [MacroDefinition(str(token), tuple(expansion)) for token, expansion in macros.items()]
    elif isinstance(macros, (str, bytes)) or not isinstance(macros, Sequence):
        raise ValueError("macros must be MacroDefinitions or a token-to-expansion mapping")
    else:
        raw = list(macros)
    normalized: list[MacroDefinition] = []
    seen: set[str] = set()
    for index, macro in enumerate(raw):
        if not isinstance(macro, MacroDefinition):
            raise ValueError(f"macros[{index}] must be a MacroDefinition")
        token = _require_token(macro.token, where=f"macros[{index}].token")
        if token in seen or token in primitive_names:
            raise ValueError(f"duplicate/colliding macro token: {token}")
        seen.add(token)
        expansion = _program_tokens(
            macro.expansion,
            allowed=primitive_names,
            where=f"expansion for {token}",
            min_length=2,
        )
        description = ""
        if macro.description:
            description = _single_line(
                macro.description, where=f"description for macro {token}"
            )
        normalized.append(MacroDefinition(token, expansion, description))
    return normalized


def _solver_demonstrations(
    demonstrations: Sequence[Mapping[str, Any]] | None,
    *,
    allowed_tokens: set[str],
    macro_items: Sequence[MacroDefinition],
    max_surface_calls: int,
    max_expanded_primitive_depth: int,
) -> list[tuple[str, tuple[str, ...], list[tuple[Any, Any]]]]:
    """Validate optional, solved train-only interface demonstrations."""

    if demonstrations is None:
        return []
    if isinstance(demonstrations, (str, bytes)) or not isinstance(
        demonstrations, Sequence
    ):
        raise ValueError("solved_demonstrations must be a sequence or None")
    macro_expansions = {macro.token: macro.expansion for macro in macro_items}
    normalized: list[tuple[str, tuple[str, ...], list[tuple[Any, Any]]]] = []
    seen_ids: set[str] = set()
    for index, demonstration in enumerate(demonstrations):
        if not isinstance(demonstration, Mapping):
            raise ValueError(f"solved_demonstrations[{index}] must be a mapping")
        if demonstration.get("split") != "train":
            raise ValueError(
                f"solved_demonstrations[{index}] is not explicitly split='train'; "
                "refusing leakage"
            )
        if demonstration.get("verified") is not True:
            raise ValueError(
                f"solved_demonstrations[{index}] is not explicitly verified=True"
            )
        demonstration_id = _single_line(
            demonstration.get("id"), where=f"solved_demonstrations[{index}].id"
        )
        if demonstration_id in seen_ids:
            raise ValueError(f"duplicate solved demonstration id: {demonstration_id!r}")
        seen_ids.add(demonstration_id)
        program = _program_tokens(
            demonstration.get("program"),
            allowed=allowed_tokens,
            where=f"program for demonstration {demonstration_id}",
        )
        if len(program) > max_surface_calls:
            raise ValueError(
                f"demonstration {demonstration_id} has {len(program)} surface calls, "
                f"limit is {max_surface_calls}"
            )
        expanded_depth = sum(
            len(macro_expansions[token]) if token in macro_expansions else 1
            for token in program
        )
        if expanded_depth > max_expanded_primitive_depth:
            raise ValueError(
                f"demonstration {demonstration_id} has expanded depth {expanded_depth}, "
                f"limit is {max_expanded_primitive_depth}"
            )
        pairs = _io_pairs(
            demonstration.get("io"), where=f"io for demonstration {demonstration_id}"
        )
        normalized.append((demonstration_id, program, pairs))
    return normalized


def build_solver_messages(
    *,
    primitives: Mapping[str, str],
    macros: Sequence[MacroDefinition] | Mapping[str, Sequence[str]],
    macros_callable: bool = True,
    solved_demonstrations: Sequence[Mapping[str, Any]] | None = None,
    io_examples: Sequence[Any],
    max_surface_calls: int,
    max_expanded_primitive_depth: int,
) -> list[dict[str, str]]:
    """Build one I/O induction prompt with an explicit frozen inventory."""

    if not isinstance(macros_callable, bool):
        raise ValueError("macros_callable must be a boolean")
    for name, value in (
        ("max_surface_calls", max_surface_calls),
        ("max_expanded_primitive_depth", max_expanded_primitive_depth),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    primitive_items = _primitive_items(primitives)
    primitive_names = {name for name, _ in primitive_items}
    macro_items = _macro_definitions(macros, primitive_names=primitive_names)
    allowed_tokens = set(primitive_names)
    if macros_callable:
        allowed_tokens.update(macro.token for macro in macro_items)
    demonstrations = _solver_demonstrations(
        solved_demonstrations,
        allowed_tokens=allowed_tokens,
        macro_items=macro_items,
        max_surface_calls=max_surface_calls,
        max_expanded_primitive_depth=max_expanded_primitive_depth,
    )
    pairs = _io_pairs(io_examples, where="io_examples")

    inventory_permission = (
        "Use only tokens in the supplied BASE and MACRO inventories. Macro tokens are "
        "legal calls and stand for their exact displayed primitive expansion. Prefer a "
        "macro alias whenever its exact expansion is present in a candidate; the alias is "
        "one surface call."
        if macros_callable
        else "Use only tokens in the supplied BASE inventory. Macro aliases are reference "
        "chunks, are not legal calls, and must not appear in the returned program. Keep any "
        "matching expansion as BASE tokens."
    )
    system = (
        "Infer a straight-line DSL program that maps every supplied input to its output. "
        f"{inventory_permission} Respect both limits. In the final answer, "
        "return exactly one line `PROGRAM: TOKEN | TOKEN | ...`. Do not return Python, markdown, "
        "analysis, tests, commentary, or any prose."
    )
    lines = ["BASE PRIMITIVES:"]
    lines.extend(f"{name}: {description}" for name, description in primitive_items)
    macro_header = (
        "FROZEN VERIFIED MACROS:"
        if macros_callable
        else "VERIFIED COMMON CHUNKS (not legal output tokens):"
    )
    lines.extend(["", macro_header])
    if macro_items:
        for macro in macro_items:
            suffix = f" ({macro.description})" if macro.description else ""
            lines.append(f"{macro.token} := {' | '.join(macro.expansion)}{suffix}")
    else:
        lines.append("NONE")
    lines.extend(
        [
            "",
            "ABSTRACT SURFACE-SYNTAX EXAMPLE (A, B, C, and X are not task tokens):",
            "If X := A | B, then X | C and A | B | C have the same exact expansion.",
            "X | C has 2 surface calls and expanded primitive depth 3.",
            "",
            "SURFACE-FIRST PROCEDURE (follow in private reasoning):",
            "1. Propose the shortest legal surface programs first.",
            "2. Expand every legal macro alias to its exact displayed BASE sequence.",
            "3. Reject any candidate whose expanded primitive depth exceeds "
            f"{max_expanded_primitive_depth}.",
            "4. Execute each expanded candidate against every visible example.",
            "5. Return the shortest candidate that passes every visible example.",
        ]
    )
    if demonstrations:
        lines.extend(["", "SOLVED TRAIN-ONLY FORMAT DEMONSTRATIONS:"])
        for demonstration_id, program, demonstration_pairs in demonstrations:
            lines.append(f"[{demonstration_id}]")
            for pair_index, (input_value, output_value) in enumerate(
                demonstration_pairs, start=1
            ):
                lines.append(
                    f"IO {pair_index}: IN={_json(input_value, where='input')} "
                    f"OUT={_json(output_value, where='output')}"
                )
            lines.append("ANSWER:")
            lines.append("PROGRAM: " + " | ".join(program))
    lines.extend(
        [
            "",
            "LIMITS:",
            f"- Maximum surface calls in returned PROGRAM: {max_surface_calls}",
            "- Maximum expanded primitive depth after replacing every macro: "
            f"{max_expanded_primitive_depth}",
            "",
            "VISIBLE INPUT/OUTPUT EXAMPLES:",
        ]
    )
    for pair_index, (input_value, output_value) in enumerate(pairs, start=1):
        lines.append(
            f"{pair_index}. IN={_json(input_value, where='input')} "
            f"OUT={_json(output_value, where='output')}"
        )
    lines.extend(["", "FINAL ANSWER: one PROGRAM line only."])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(lines)},
    ]


def build_solver_record(
    record_id: str,
    *,
    primitives: Mapping[str, str],
    macros: Sequence[MacroDefinition] | Mapping[str, Sequence[str]],
    macros_callable: bool = True,
    solved_demonstrations: Sequence[Mapping[str, Any]] | None = None,
    io_examples: Sequence[Any],
    max_surface_calls: int,
    max_expanded_primitive_depth: int,
    meta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return one JSONL-compatible vLLM record for task solving."""

    record_id = _single_line(record_id, where="record_id")
    return {
        "id": record_id,
        "messages": build_solver_messages(
            primitives=primitives,
            macros=macros,
            macros_callable=macros_callable,
            solved_demonstrations=solved_demonstrations,
            io_examples=io_examples,
            max_surface_calls=max_surface_calls,
            max_expanded_primitive_depth=max_expanded_primitive_depth,
        ),
        "meta": {
            "prompt_kind": "solve_program",
            "macros_callable": macros_callable,
            **dict(meta or {}),
        },
    }


def _answer_region(text: str) -> str:
    """Remove only Qwen thinking, terminal markers, one fence, and whitespace."""

    if not isinstance(text, str):
        raise ValueError("completion text must be a string")
    answer = text
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    elif "<think>" in answer:
        raise ValueError("thinking channel did not close")
    answer = answer.strip()
    removed = True
    while removed:
        removed = False
        for marker in _TERMINAL_MARKERS:
            if answer.endswith(marker):
                answer = answer[: -len(marker)].rstrip()
                removed = True
    fence = _FENCE_RE.fullmatch(answer)
    if fence is not None:
        answer = fence.group("body").strip()
    elif "```" in answer:
        raise ValueError("markdown fence must enclose the entire final answer")
    if not answer:
        raise ValueError("empty final answer")
    return answer


def parse_macro_proposals(
    text: str,
    *,
    allowed_primitives: Sequence[str] | set[str],
    max_macros: int | None = None,
) -> tuple[MacroProposal, ...]:
    """Strictly parse macro lines after limited answer-channel extraction."""

    allowed = {_require_token(token, where="allowed primitive") for token in allowed_primitives}
    if not allowed:
        raise ValueError("allowed_primitives cannot be empty")
    if max_macros is not None and (
        not isinstance(max_macros, int) or isinstance(max_macros, bool) or max_macros < 1
    ):
        raise ValueError("max_macros must be positive when supplied")
    answer = _answer_region(text)
    if answer == "NONE":
        return ()
    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if max_macros is not None and len(lines) > max_macros:
        raise ValueError(f"returned {len(lines)} macros, limit is {max_macros}")
    proposals: list[MacroProposal] = []
    seen: set[str] = set()
    for line_index, line in enumerate(lines, start=1):
        match = _MACRO_LINE_RE.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid macro line {line_index}: {line!r}")
        name = match.group(1)
        if name in seen or name in allowed:
            raise ValueError(f"duplicate/colliding macro name: {name}")
        seen.add(name)
        expansion = tuple(part.strip() for part in match.group(2).split("|"))
        unknown = sorted(set(expansion) - allowed)
        if unknown:
            raise ValueError(f"macro {name} uses non-primitive tokens: {unknown}")
        proposals.append(MacroProposal(name, expansion))
    return tuple(proposals)


def extract_macro_proposal_lines(
    text: str,
    *,
    allowed_primitives: Sequence[str] | set[str],
    max_macros: int,
) -> MacroLineExtraction:
    """Harvest valid macro lines independently while retaining every rejection.

    This is intentionally separate from :func:`parse_macro_proposals`.  The
    strict parser remains appropriate for an exact-format endpoint; proposal
    construction is a verified candidate-harvesting stage where one prose line
    must not erase other locally valid candidates.  Markdown fence lines are
    recorded as rejected nonblank lines, while valid macro lines inside them
    remain eligible.
    """

    if not isinstance(text, str):
        raise ValueError("completion text must be a string")
    if not isinstance(max_macros, int) or isinstance(max_macros, bool) or max_macros < 1:
        raise ValueError("max_macros must be a positive integer")
    allowed = {_require_token(token, where="allowed primitive") for token in allowed_primitives}
    if not allowed:
        raise ValueError("allowed_primitives cannot be empty")

    answer = text
    if "</think>" in answer:
        answer = answer.rsplit("</think>", 1)[1]
    elif "<think>" in answer:
        raise ValueError("thinking channel did not close")
    answer = answer.strip()
    removed = True
    while removed:
        removed = False
        for marker in _TERMINAL_MARKERS:
            if answer.endswith(marker):
                answer = answer[: -len(marker)].rstrip()
                removed = True

    proposals: list[MacroProposal] = []
    rejected: list[RejectedMacroLine] = []
    seen_names: set[str] = set()
    total_valid = 0
    capped = False
    for line_number, raw_line in enumerate(answer.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = _MACRO_LINE_RE.fullmatch(line)
        if match is None:
            rejected.append(
                RejectedMacroLine(line_number, line, "does not match MACRO line grammar")
            )
            continue
        name = match.group(1)
        if name in allowed:
            rejected.append(
                RejectedMacroLine(line_number, line, "macro name collides with a primitive")
            )
            continue
        if name in seen_names:
            rejected.append(RejectedMacroLine(line_number, line, "duplicate macro name"))
            continue
        expansion = tuple(part.strip() for part in match.group(2).split("|"))
        unknown = sorted(set(expansion) - allowed)
        if unknown:
            rejected.append(
                RejectedMacroLine(
                    line_number,
                    line,
                    f"uses non-primitive tokens: {unknown}",
                )
            )
            continue
        seen_names.add(name)
        total_valid += 1
        if len(proposals) < max_macros:
            proposals.append(MacroProposal(name, expansion))
        else:
            capped = True
    return MacroLineExtraction(
        proposals=tuple(proposals),
        rejected_nonblank_lines=tuple(rejected),
        extra_valid_lines_capped=capped,
        total_valid_lines=total_valid,
    )


def parse_program(
    text: str,
    *,
    allowed_tokens: Sequence[str] | set[str],
    max_surface_calls: int | None = None,
) -> tuple[str, ...]:
    """Parse exactly one solver ``PROGRAM`` line and validate its inventory."""

    allowed = {_require_token(token, where="allowed solver token") for token in allowed_tokens}
    if not allowed:
        raise ValueError("allowed_tokens cannot be empty")
    if max_surface_calls is not None and (
        not isinstance(max_surface_calls, int)
        or isinstance(max_surface_calls, bool)
        or max_surface_calls < 1
    ):
        raise ValueError("max_surface_calls must be positive when supplied")
    answer = _answer_region(text)
    nonblank = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(nonblank) != 1:
        raise ValueError("solver answer must contain exactly one non-empty line")
    match = _PROGRAM_LINE_RE.fullmatch(nonblank[0])
    if match is None:
        raise ValueError(f"invalid program line: {nonblank[0]!r}")
    program = tuple(part.strip() for part in match.group(1).split("|"))
    unknown = sorted(set(program) - allowed)
    if unknown:
        raise ValueError(f"program uses tokens outside its supplied inventory: {unknown}")
    if max_surface_calls is not None and len(program) > max_surface_calls:
        raise ValueError(
            f"program has {len(program)} surface calls, limit is {max_surface_calls}"
        )
    return program


def extract_completion_texts(rows: Sequence[Mapping[str, Any]]) -> tuple[CompletionText, ...]:
    """Extract semantic text/token IDs while preserving request and sample identity."""

    completions: list[CompletionText] = []
    seen: set[tuple[str, int]] = set()
    for row_index, row in enumerate(rows):
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"vLLM row {row_index} has no valid id")
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"vLLM row {record_id!r} has no outputs")
        for output_index, output in enumerate(outputs):
            if not isinstance(output, Mapping):
                raise ValueError(f"output {record_id}/{output_index} is not a mapping")
            sample_index = output.get("sample_index")
            text = output.get("text")
            token_ids = output.get("token_ids")
            if (
                not isinstance(sample_index, int)
                or isinstance(sample_index, bool)
                or sample_index < 0
            ):
                raise ValueError(f"output {record_id}/{output_index} has invalid sample_index")
            if not isinstance(text, str):
                raise ValueError(f"output {record_id}/{sample_index} has no text")
            if not isinstance(token_ids, list) or any(
                not isinstance(token, int) or isinstance(token, bool) for token in token_ids
            ):
                raise ValueError(f"output {record_id}/{sample_index} has invalid token_ids")
            key = (record_id, sample_index)
            if key in seen:
                raise ValueError(f"duplicate completion identity: {key}")
            seen.add(key)
            completions.append(
                CompletionText(record_id, sample_index, text, tuple(token_ids))
            )
    return tuple(completions)


def parse_program_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_tokens: Sequence[str] | set[str],
    max_surface_calls: int | None = None,
) -> tuple[ParsedProgramCompletion, ...]:
    """Parse every solver sample without discarding malformed completions."""

    parsed: list[ParsedProgramCompletion] = []
    for completion in extract_completion_texts(rows):
        try:
            program = parse_program(
                completion.text,
                allowed_tokens=allowed_tokens,
                max_surface_calls=max_surface_calls,
            )
            error = None
        except ValueError as exc:
            program = None
            error = str(exc)
        parsed.append(
            ParsedProgramCompletion(
                completion.record_id,
                completion.sample_index,
                completion.text,
                program,
                error,
            )
        )
    return tuple(parsed)


def parse_macro_outputs(
    rows: Sequence[Mapping[str, Any]],
    *,
    allowed_primitives: Sequence[str] | set[str],
    max_macros: int | None = None,
) -> tuple[ParsedMacroCompletion, ...]:
    """Parse every proposal sample without hiding formatting failures."""

    parsed: list[ParsedMacroCompletion] = []
    for completion in extract_completion_texts(rows):
        try:
            proposals = parse_macro_proposals(
                completion.text,
                allowed_primitives=allowed_primitives,
                max_macros=max_macros,
            )
            error = None
        except ValueError as exc:
            proposals = None
            error = str(exc)
        parsed.append(
            ParsedMacroCompletion(
                completion.record_id,
                completion.sample_index,
                completion.text,
                proposals,
                error,
            )
        )
    return tuple(parsed)


def _count(mapping: Mapping[str, Any], key: str, *, where: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{where}.{key} must be a non-negative integer")
    return value


def extract_token_accounting(
    rows: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None = None,
) -> BatchAccounting:
    """Copy exact vLLM counters and check runner-summary agreement.

    ``sampled_tokens`` is the matched-generation-compute quantity. Injected
    force-close tokens are intentionally separate. Prompt compute uses logical
    stage-one plus continuation-prefill tokens, exactly as recorded by the
    local runner.
    """

    per_completion: list[CompletionAccounting] = []
    unique_input_prompt_tokens = 0
    for row_index, row in enumerate(rows):
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"vLLM row {row_index} has no valid id")
        prompt_tokens = _count(row, "n_prompt_tokens", where=f"row {record_id}")
        unique_input_prompt_tokens += prompt_tokens
        outputs = row.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            raise ValueError(f"vLLM row {record_id!r} has no outputs")
        for output_index, output in enumerate(outputs):
            if not isinstance(output, Mapping):
                raise ValueError(f"output {record_id}/{output_index} is not a mapping")
            sample_index = _count(output, "sample_index", where=f"output {record_id}")
            stage1_prompt = _count(
                output, "n_stage1_prompt_tokens", where=f"output {record_id}/{sample_index}"
            )
            if stage1_prompt != prompt_tokens:
                raise ValueError(
                    f"output {record_id}/{sample_index} stage1 prompt count disagrees with row"
                )
            per_completion.append(
                CompletionAccounting(
                    record_id=record_id,
                    sample_index=sample_index,
                    unique_request_prompt_tokens=prompt_tokens,
                    stage1_logical_prompt_tokens=stage1_prompt,
                    stage2_logical_prompt_tokens=_count(
                        output,
                        "n_stage2_prompt_tokens",
                        where=f"output {record_id}/{sample_index}",
                    ),
                    sampled_tokens=_count(
                        output, "n_sampled_tokens", where=f"output {record_id}/{sample_index}"
                    ),
                    injected_tokens=_count(
                        output, "n_injected_tokens", where=f"output {record_id}/{sample_index}"
                    ),
                    completion_tokens=_count(
                        output, "n_completion_tokens", where=f"output {record_id}/{sample_index}"
                    ),
                    thinking_tokens=_count(
                        output, "n_thinking_tokens", where=f"output {record_id}/{sample_index}"
                    ),
                    answer_tokens=_count(
                        output, "n_answer_tokens", where=f"output {record_id}/{sample_index}"
                    ),
                    terminal_tokens_trimmed=_count(
                        output,
                        "n_terminal_tokens_trimmed",
                        where=f"output {record_id}/{sample_index}",
                    ),
                )
            )

    def total(field: str) -> int:
        return sum(getattr(item, field) for item in per_completion)

    stage1 = total("stage1_logical_prompt_tokens")
    stage2 = total("stage2_logical_prompt_tokens")
    result = BatchAccounting(
        requests=len(rows),
        completions=len(per_completion),
        unique_input_prompt_tokens=unique_input_prompt_tokens,
        stage1_logical_prompt_tokens=stage1,
        stage2_logical_prompt_tokens=stage2,
        logical_model_input_tokens=stage1 + stage2,
        sampled_tokens=total("sampled_tokens"),
        injected_tokens=total("injected_tokens"),
        completion_tokens=total("completion_tokens"),
        thinking_tokens=total("thinking_tokens"),
        answer_tokens=total("answer_tokens"),
        terminal_tokens_trimmed=total("terminal_tokens_trimmed"),
        per_completion=tuple(per_completion),
    )
    if summary is not None:
        if summary.get("model") != REQUIRED_MODEL_ID:
            raise ValueError(
                f"runner summary model must be {REQUIRED_MODEL_ID}, got {summary.get('model')!r}"
            )
        if summary.get("model_revision") != MODEL_REVISION:
            raise ValueError("runner summary model revision disagrees with local pinned runner")
        counts = summary.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError("runner summary has no counts mapping")
        expected = {
            "requests": result.requests,
            "completions": result.completions,
            "unique_input_prompt_tokens": result.unique_input_prompt_tokens,
            "stage1_logical_prompt_tokens": result.stage1_logical_prompt_tokens,
            "stage2_logical_prompt_tokens": result.stage2_logical_prompt_tokens,
            "logical_model_input_tokens": result.logical_model_input_tokens,
            "sampled_tokens": result.sampled_tokens,
            "injected_tokens": result.injected_tokens,
        }
        for key, value in expected.items():
            if counts.get(key) != value:
                raise ValueError(
                    f"runner summary count {key}={counts.get(key)!r}, extracted {value}"
                )
    return result


def generate_vllm_batch(
    runner: RunnerLike,
    records: Sequence[dict[str, Any]],
    sampling: SamplingConfig,
) -> VLLMBatchResult:
    """Generate all records in one vLLM call and validate exact accounting."""

    if not records:
        raise ValueError("records cannot be empty")
    rows, summary = runner.generate(records, sampling)
    accounting = extract_token_accounting(rows, summary)
    return VLLMBatchResult(tuple(rows), summary, accounting)


def run_vllm_batch(
    records: Sequence[dict[str, Any]],
    sampling: SamplingConfig,
    *,
    engine_config: EngineConfig = EngineConfig(),
    runner_factory: Callable[[EngineConfig], RunnerLike] = VLLMRunner,
) -> VLLMBatchResult:
    """Own one local vLLM engine, batch-generate, then release it.

    Long experiments should construct one :class:`VLLMRunner` and call
    :func:`generate_vllm_batch` repeatedly so model-load time is not paid per
    arm. ``runner_factory`` exists for no-model tests, not for another backend.
    """

    runner = runner_factory(engine_config)
    try:
        return generate_vllm_batch(runner, records, sampling)
    finally:
        close = getattr(runner, "close", None)
        if callable(close):
            close()
