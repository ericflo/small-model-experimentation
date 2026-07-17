#!/usr/bin/env python3
"""Designed count-don't-walk curriculum: teach INDEX ARITHMETIC, not walking.

Lifecycle 27 — the evidence-backed successor to the enumerative-repair
cell (lifecycle 26), changing ONLY the expression pedagogy. The evidence
(committed at experiments/qwen35_4b_enumerative_repair_protocol/analysis/
truncation_forensics.md): the enumeration discipline INSTALLED (9/40
canonical-next versus both controls at 0/40) but expressed as a VERBOSE
LINEAR WALK — 20 of 21 unparseable gate rows were 1,024-token cap
truncations caught mid-CORRECT walk, a token cost that grows with the
tried-list depth ``k`` and the grammar size. The fix taught here:
COUNT, DON'T WALK — the tried list has ``k`` entries in canonical
order, so the target is entry ``k + 1`` of the frozen order; locating
it is pure index arithmetic over ranges that are RENDERED IN THE
PROMPT, constant token cost in ``k``.

MACHINERY: the eight legality-bounded machine formalisms of the menders
dose-scale cell (``troughline``, ``trinketcord``, ``crankwheel``,
``sigilslate``, ``barrowyoke``, ``balesled``, ``millround``,
``skeinreel``) are REUSED via a byte-identical copy of that cell's
reviewed generator (``gen_feedloop_curriculum.py`` — machinery imported,
never forked): the machine builders, describe/apply semantics, the
banned-vocabulary inventory, and the legal-candidate enumeration idea.
The task shape (partial enumeration episode, frozen canonical order,
verified tried prefix, canonical-next target, K_CYCLE, uniqueness
invariants) is byte-equivalent to the enumerative-repair cell's. ONLY
the expression pedagogy changes:

- THE ORDER STATEMENT gains the rendered per-step candidate counts:
  after the byte-identical frozen rule text, every prompt now renders
  the per-step ranges ("step 1 offers 7 changes (numbers 1-7); step 2
  offers 7 (numbers 8-14); ..."), each step contributing
  ``len(action list) - 1`` candidates when its written action is in the
  list (always true by construction, computed generically, never
  assumed) and ``len(action list)`` otherwise. The generator verifies
  the rendered ranges against its own exhaustive enumeration exactly,
  range by range.
- THE THINK TARGET is a fixed-shape compact computation, identical
  structure in every row — exactly five short lines: (a) count the
  tried entries -> k; (b) the target is change number k+1 in the frozen
  order; (c) locate change k+1 by the rendered range arithmetic (find
  the step whose cumulative range contains k+1, then the explicit
  offset subtraction); (d) resolve the offset to the action-list slot
  (skipping the step's written action) and emit ``STEP <n>: <action>``.
  Never a walk; the token cost is constant in ``k``.
- A THINK LENGTH BUDGET is enforced in the generator: every think
  target must satisfy the frozen caps (``THINK_TOKEN_CAP`` estimated
  tokens and ``THINK_CHAR_CAP`` characters plus the frozen five-line
  shape, tested per row); the REAL tokenizer bound (<= THINK_TOKEN_CAP
  Qwen tokens per think span) is enforced fail-closed by
  scripts/measure_source_tokens.py over the frozen corpus.

Everything else is the reference cell's: the answer is exact-match
gradable (``STEP <k>: <corrected step>``); the generator verifies per
row, by exhaustive re-derivation over the full candidate space, that
the target IS the canonical-next untried legal candidate, that exactly
ONE candidate repairs both trials, and that every tried entry is legal,
canonically ordered, and genuinely failing. A banned-vocabulary scan
(the menders cell's full inventory) rejects any leak.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_feedloop_curriculum as feedloop  # noqa: E402


ANSWER_LINE = feedloop.ANSWER_LINE
BANNED_PROMPT_TOKENS = feedloop.BANNED_PROMPT_TOKENS
ENUM_FORMALISMS = feedloop.FEEDLOOP_FORMALISMS
# Every surface token in this cell is INHERITED from the menders
# dose-scale cell by design (same eight formalisms; freshness is at the
# ROW level, enforced by the corpus builder's overlap receipts). Nothing
# new is invented, so the fresh-surface claim is deliberately EMPTY.
INHERITED_SURFACE_TOKENS = tuple(
    dict.fromkeys(
        feedloop.INHERITED_SURFACE_TOKENS + feedloop.FRESH_SURFACE_TOKENS
    )
)
FRESH_SURFACE_TOKENS: tuple[str, ...] = ()

# The tried-list length schedule: per formalism, row index i takes
# K_CYCLE[i % 5]. Includes k=0 (first-candidate rows) and DEEP_K
# (deep-in-the-list rows); the 160-row arm gives each formalism each
# value exactly four times and the 40-row holdout exactly once.
K_CYCLE = (0, 1, 3, 6, 10)
DEEP_K = K_CYCLE[-1]

# The frozen canonical-order statement, byte-identical in EVERY row AND
# byte-identical to the enumerative-repair reference cell's rule text.
# It names exactly the generator's enumeration order: step index
# ascending, then the changed-to action's position in the rendered
# action list.
CANONICAL_ORDER_STATEMENT = (
    "Single-step changes are worked through in one frozen order: first "
    "by step number, lowest first; within a step, by the changed-to "
    "action's position in the numbered action list, top to bottom; a "
    "change never leaves the step as already written."
)
ASK_LINE = "Name the next untried single-step change in the frozen order."
NONE_TRIED_LINE = "No single-step change has been tried yet."
TRIED_HEADER = (
    "Changes already tried, in that frozen order, every one ruled out by "
    "its own two-trial run:"
)
ANSWER_FORMAT_RE = re.compile(r"^STEP (\d+): (.+)$")

# --- the count-don't-walk expression contract ------------------------------
# Frozen think length budget: the REAL tokenizer cap (verified
# fail-closed per row by scripts/measure_source_tokens.py over the
# frozen corpus and reported in data/source_token_lengths.json) plus the
# model-free per-row proxies the generator itself enforces (the
# machinery estimate len//4 and a hard character cap). The think target
# is five short lines whose only variable content is small integers and
# one action description, so its cost is CONSTANT in k.
THINK_TOKEN_CAP = 120
THINK_CHAR_CAP = 380
THINK_LINE_COUNT = 5
# Frozen per-line shape (one regex per line; identical structure in
# every row — only digits and the action description vary).
THINK_LINE_PATTERNS = (
    r"^Tried entries: \d+\.$",
    r"^Target: change number \d+ in the frozen order\.$",
    r"^Number \d+ sits in step \d+'s range \d+-\d+: offset \d+ - \d+ \+ 1 = \d+\.$",
    r"^Step \d+'s written action is list number \d+; skipping it, offset "
    r"\d+ is list number \d+: .+\.$",
    r"^STEP \d+: .+$",
)


# ---------------------------------------------------------------------------
# machine reconstruction (deterministic from formalism + vocabulary)
# ---------------------------------------------------------------------------

_GRAMMAR_BUILDERS = {
    "troughline": feedloop._trough_grammar,
    "trinketcord": feedloop._cord_grammar,
    "crankwheel": feedloop._wheel_grammar,
    "sigilslate": feedloop._slate_grammar,
    "barrowyoke": feedloop._yoke_grammar,
    "balesled": feedloop._sled_grammar,
    "millround": feedloop._round_grammar,
    "skeinreel": feedloop._reel_grammar,
}
_DESCRIBERS = {
    "troughline": feedloop._trough_describe,
    "trinketcord": feedloop._cord_describe,
    "crankwheel": feedloop._wheel_describe,
    "sigilslate": feedloop._slate_describe,
    "barrowyoke": feedloop._yoke_describe,
    "balesled": feedloop._sled_describe,
    "millround": feedloop._round_describe,
    "skeinreel": feedloop._reel_describe,
}
_APPLIERS = {
    "troughline": feedloop._trough_apply,
    "trinketcord": feedloop._cord_apply,
    "crankwheel": feedloop._wheel_apply,
    "sigilslate": feedloop._slate_apply,
    "barrowyoke": feedloop._yoke_apply,
    "balesled": feedloop._sled_apply,
    "millround": feedloop._round_apply,
    "skeinreel": feedloop._reel_apply,
}
_DICT_STATE = ("troughline", "barrowyoke")


def rebuild_grammar(formalism: str, vocabulary: list[str]) -> list[tuple]:
    if formalism not in _GRAMMAR_BUILDERS:
        raise ValueError(f"unknown count-walk formalism: {formalism}")
    return _GRAMMAR_BUILDERS[formalism](list(vocabulary))


def canonical_state(formalism: str, value) -> object:
    """JSON-roundtrip-safe state normalization for re-simulation."""
    if formalism in _DICT_STATE:
        if not isinstance(value, dict):
            raise ValueError(f"{formalism} state must be a mapping")
        return dict(value)
    return tuple(value)


def canonical_op(op) -> tuple:
    return tuple(op)


def candidate_space(grammar: list[tuple], written: list[tuple]) -> list[tuple[int, tuple]]:
    """The full legal single-step-change space in CANONICAL ORDER:
    step index ascending, then grammar (action-list) position."""
    return [
        (index, op)
        for index in range(len(written))
        for op in grammar
        if op != written[index]
    ]


def per_step_candidate_counts(grammar: list[tuple], written: list[tuple]) -> list[int]:
    """Each step contributes len(grammar) - 1 candidates when its written
    action is in the list (always true by construction) and
    len(grammar) otherwise — computed generically, never assumed."""
    return [
        sum(1 for op in grammar if op != written[index])
        for index in range(len(written))
    ]


def step_ranges(counts: list[int]) -> list[tuple[int, int]]:
    """Inclusive 1-based (start, end) cumulative range per step."""
    ranges: list[tuple[int, int]] = []
    start = 1
    for count in counts:
        end = start + count - 1
        ranges.append((start, end))
        start = end + 1
    return ranges


def render_range_statement(counts: list[int]) -> str:
    """The rendered per-step candidate counts appended to the frozen
    order statement — the mapping the compact computation reads back."""
    ranges = step_ranges(counts)
    parts = []
    for index, (count, (start, end)) in enumerate(zip(counts, ranges)):
        noun = " changes" if index == 0 else ""
        parts.append(
            f"step {index + 1} offers {count}{noun} (numbers {start}-{end})"
        )
    total = ranges[-1][1] if ranges else 0
    return "In that order, " + "; ".join(parts) + f" — {total} changes in all."


def locate_candidate(counts: list[int], number: int) -> tuple[int, int]:
    """Map a 1-based canonical candidate number to (step index, offset):
    find the step whose cumulative range contains it, then the 1-based
    offset inside that step."""
    for index, (start, end) in enumerate(step_ranges(counts)):
        if start <= number <= end:
            return index, number - start + 1
    raise ValueError(f"candidate number out of range: {number}")


def offset_to_list_number(
    grammar: list[tuple], written_op: tuple, offset: int
) -> int:
    """Map a 1-based within-step offset to the 1-based action-list slot,
    skipping the step's written action."""
    seen = 0
    for position, op in enumerate(grammar, start=1):
        if op == written_op:
            continue
        seen += 1
        if seen == offset:
            return position
    raise ValueError(f"offset out of range: {offset}")


def build_think(
    grammar: list[tuple],
    written: list[tuple],
    k: int,
    describe,
) -> str:
    """The fixed-shape compact computation (count -> index -> locate ->
    slot -> emit), a pure function of the machine and k; five short
    lines, constant token cost in k."""
    counts = per_step_candidate_counts(grammar, written)
    number = k + 1
    step_index, offset = locate_candidate(counts, number)
    start, end = step_ranges(counts)[step_index]
    written_position = grammar.index(written[step_index]) + 1
    list_number = offset_to_list_number(grammar, written[step_index], offset)
    action = describe(grammar[list_number - 1])
    lines = (
        f"Tried entries: {k}.",
        f"Target: change number {number} in the frozen order.",
        f"Number {number} sits in step {step_index + 1}'s range "
        f"{start}-{end}: offset {number} - {start} + 1 = {offset}.",
        f"Step {step_index + 1}'s written action is list number "
        f"{written_position}; skipping it, offset {offset} is list number "
        f"{list_number}: {action}.",
        f"STEP {step_index + 1}: {action}",
    )
    return "\n".join(lines)


def check_think_budget(think: str) -> None:
    """The frozen model-free budget + constant-shape contract, per row."""
    lines = think.split("\n")
    if len(lines) != THINK_LINE_COUNT:
        raise ValueError(
            f"think target is not the frozen {THINK_LINE_COUNT}-line shape"
        )
    for line, pattern in zip(lines, THINK_LINE_PATTERNS):
        if re.fullmatch(pattern, line) is None:
            raise ValueError(f"think line breaks the frozen shape: {line!r}")
    if len(think) > THINK_CHAR_CAP:
        raise ValueError(
            f"think target exceeds the frozen character cap: {len(think)}"
        )
    if max(1, len(think) // 4) > THINK_TOKEN_CAP:
        raise ValueError("think target exceeds the frozen estimated-token cap")


def run_patched(apply_fn, written: list[tuple], index: int, op: tuple, start):
    patched = written[:index] + [op] + written[index + 1 :]
    return feedloop._run_steps(apply_fn, patched, start)


def repairs_both(apply_fn, written, index, op, start_a, wanted_a, start_b, wanted_b) -> bool:
    return (
        run_patched(apply_fn, written, index, op, start_a) == wanted_a
        and run_patched(apply_fn, written, index, op, start_b) == wanted_b
    )


def make_row(
    *, prompt: str, think: str, answer: str, kind: str, surface: str, level: int, audit: dict
) -> dict:
    """The machinery's row shape (feedloop.make_row) with ONE designed
    delta: the think target is the fixed-shape five-line compact
    computation, so the machinery's single-line think assertion is
    replaced by the frozen count-don't-walk shape/budget contract.
    Every other field and estimate is byte-equivalent."""
    assert answer and "\n" not in answer
    assert audit.get("truth_valid") is True
    check_think_budget(think)
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": f"ANSWER: {answer}",
        "kind": f"u_{kind}",
        "family": "universal",
        "surface": surface,
        "level": level,
        "n_think_tokens": max(1, len(think) // 4),
        "row_weight": 1.0,
        "_audit": audit,
    }


def verify_ranges_against_enumeration(
    counts: list[int], candidates: list[tuple[int, tuple]], steps: int
) -> None:
    """The rendered ranges must equal the exhaustive enumeration exactly:
    the range slice for each step is exactly that step's candidates."""
    if len(counts) != steps:
        raise ValueError("per-step counts do not cover every written step")
    ranges = step_ranges(counts)
    if sum(counts) != len(candidates) or not ranges or ranges[-1][1] != len(candidates):
        raise ValueError("rendered ranges do not cover the candidate space")
    for index, (start, end) in enumerate(ranges):
        block = candidates[start - 1 : end]
        if len(block) != counts[index] or any(
            candidate_index != index for candidate_index, _ in block
        ):
            raise ValueError(
                f"rendered range for step {index + 1} disagrees with the enumeration"
            )


# ---------------------------------------------------------------------------
# the lesson
# ---------------------------------------------------------------------------

def _tried_line(
    position: int,
    index: int,
    op: tuple,
    describe,
    render,
    apply_fn,
    written,
    start_a,
    wanted_a,
    start_b,
    wanted_b,
) -> tuple[str, str]:
    """Render one tried entry with its observed two-trial outcome; the
    returned tag records WHERE it failed ('first' or 'second')."""
    out_a = run_patched(apply_fn, written, index, op, start_a)
    if out_a != wanted_a:
        return (
            f"  {position}. step {index + 1} to '{describe(op)}' — the "
            f"first trial finished at {render(out_a)}, not the wanted "
            f"{render(wanted_a)}.",
            "first",
        )
    out_b = run_patched(apply_fn, written, index, op, start_b)
    if out_b == wanted_b:
        raise RuntimeError("a tried candidate repairs both trials")
    return (
        f"  {position}. step {index + 1} to '{describe(op)}' — the first "
        f"trial came out as wanted, but the second trial finished at "
        f"{render(out_b)}, not the wanted {render(wanted_b)}.",
        "second",
    )


def count_walk_lesson(rng: random.Random, formalism: str, k_target: int) -> dict:
    if k_target < 0 or k_target > DEEP_K:
        raise ValueError(f"tried-list length out of range: {k_target}")
    for _ in range(5000):
        machine = feedloop._feedloop_machine(rng, formalism)
        grammar = machine["grammar"]
        describe = machine["describe"]
        apply_fn = machine["apply"]
        render = machine["render"]
        start_a, start_b = machine["start_a"], machine["start_b"]
        if start_a == start_b:
            continue
        length = rng.randint(4, 5)
        intended = [rng.choice(grammar) for _ in range(length)]
        bug_at = rng.randrange(length)
        wrong_op = rng.choice([op for op in grammar if op != intended[bug_at]])
        written = list(intended)
        written[bug_at] = wrong_op

        wanted_a = feedloop._run_steps(apply_fn, intended, start_a)
        finished_a = feedloop._run_steps(apply_fn, written, start_a)
        wanted_b = feedloop._run_steps(apply_fn, intended, start_b)
        finished_b = feedloop._run_steps(apply_fn, written, start_b)
        # Failure evidence on BOTH trials for the written sequence.
        if wanted_a == finished_a or wanted_b == finished_b:
            continue

        candidates = candidate_space(grammar, written)
        successes = [
            rank
            for rank, (index, op) in enumerate(candidates)
            if repairs_both(
                apply_fn, written, index, op, start_a, wanted_a, start_b, wanted_b
            )
        ]
        true_fix = (bug_at, intended[bug_at])
        # Exactly ONE candidate repairs both trials (the true fix): the
        # perfect enumerator's episode length is then well defined.
        if len(successes) != 1 or candidates[successes[0]] != true_fix:
            continue
        success_index = successes[0]
        if success_index < k_target:
            continue

        # Legality clauses must be documented in the spec (machinery
        # invariant carried from the reviewed reference cell).
        legality_clauses = machine["legality_clauses"]
        if not legality_clauses or any(
            clause not in machine["rules"] for clause in legality_clauses
        ):
            raise RuntimeError("a parameterized operation is not bounded in the spec")

        tried = candidates[:k_target]
        tried_lines: list[str] = []
        tried_audit: list[dict] = []
        for position, (index, op) in enumerate(tried, start=1):
            line, failed_on = _tried_line(
                position, index, op, describe, render, apply_fn,
                written, start_a, wanted_a, start_b, wanted_b,
            )
            tried_lines.append(line)
            tried_audit.append(
                {"index": index, "op": list(op), "failed_on": failed_on}
            )
        target_index, target_op = candidates[k_target]

        # The rendered per-step candidate counts: verified against the
        # exhaustive enumeration exactly (range by range) before any row
        # is emitted.
        counts = per_step_candidate_counts(grammar, written)
        verify_ranges_against_enumeration(counts, candidates, len(written))
        range_statement = render_range_statement(counts)

        action_list = "\n".join(
            f"  {position + 1}. {describe(op)}" for position, op in enumerate(grammar)
        )
        listing = "\n".join(
            f"  {index + 1}. {describe(op)}" for index, op in enumerate(written)
        )
        if tried_lines:
            tried_block = TRIED_HEADER + "\n" + "\n".join(tried_lines)
        else:
            tried_block = NONE_TRIED_LINE
        prompt = (
            f"{machine['scene']}\n{machine['rules']}\n"
            "Numbered action list (every action the machine accepts, in "
            "its frozen order):\n"
            f"{action_list}\n"
            f"Steps as written:\n{listing}\n"
            f"First trial, starting from {render(start_a)}: the crew "
            f"wanted {render(wanted_a)}, but the run finished at "
            f"{render(finished_a)}.\n"
            f"Second trial, starting from {render(start_b)}: the crew "
            f"wanted {render(wanted_b)}, but the run finished at "
            f"{render(finished_b)}.\n"
            "Exactly one written step is off.\n"
            f"{CANONICAL_ORDER_STATEMENT}\n"
            f"{range_statement}\n"
            f"{tried_block}\n"
            f"{ASK_LINE}\n"
            f"{ANSWER_LINE} (format: STEP <k>: <corrected step>)"
        )
        think = build_think(grammar, written, k_target, describe)
        check_think_budget(think)
        answer = f"STEP {target_index + 1}: {describe(target_op)}"
        if not think.endswith(answer):
            raise RuntimeError("the compact computation does not emit the answer")
        return make_row(
            prompt=prompt,
            think=think,
            answer=answer,
            kind="count_walk",
            surface=formalism,
            level=length,
            audit={
                "truth_valid": True,
                "formalism": formalism,
                "steps": length,
                "grammar_size": len(grammar),
                "candidate_count": len(candidates),
                "per_step_candidate_counts": counts,
                "k_tried": k_target,
                "success_index": success_index,
                "episode_success_turns": success_index + 1,
                "remaining_turns_after_tried": success_index - k_target + 1,
                "target_is_true_fix": candidates[k_target] == true_fix,
                "target": [target_index, list(target_op)],
                "true_fix": [bug_at, list(intended[bug_at])],
                "tried": tried_audit,
                "canonical_statement_in_prompt": True,
                "range_statement_in_prompt": True,
                "action_list_in_prompt": True,
                "legality_clauses": list(legality_clauses),
                "spec": {
                    "vocabulary": machine["vocabulary"],
                    "written": [list(op) for op in written],
                    "start_a": start_a,
                    "start_b": start_b,
                    "wanted_a": wanted_a,
                    "wanted_b": wanted_b,
                    "finished_a": finished_a,
                    "finished_b": finished_b,
                },
            },
        )
    raise RuntimeError(
        f"could not construct a {formalism} count-walk lesson at k={k_target}"
    )


# ---------------------------------------------------------------------------
# per-row exhaustive re-derivation (the validator and the fidelity readout)
# ---------------------------------------------------------------------------

def rederive_candidates(audit: dict) -> tuple[list[tuple[int, tuple]], dict]:
    """Rebuild the machine from the audit and re-derive the canonical
    candidate space, the unique both-trials fix, and the target."""
    formalism = audit["formalism"]
    spec = audit["spec"]
    grammar = rebuild_grammar(formalism, spec["vocabulary"])
    apply_fn = _APPLIERS[formalism]
    written = [canonical_op(op) for op in spec["written"]]
    if any(op not in grammar for op in written):
        raise ValueError("a written step is not in the bounded grammar")
    start_a = canonical_state(formalism, spec["start_a"])
    start_b = canonical_state(formalism, spec["start_b"])
    wanted_a = canonical_state(formalism, spec["wanted_a"])
    wanted_b = canonical_state(formalism, spec["wanted_b"])
    finished_a = canonical_state(formalism, spec["finished_a"])
    finished_b = canonical_state(formalism, spec["finished_b"])
    if feedloop._run_steps(apply_fn, written, start_a) != finished_a:
        raise ValueError("first-trial observed outcome does not re-derive")
    if feedloop._run_steps(apply_fn, written, start_b) != finished_b:
        raise ValueError("second-trial observed outcome does not re-derive")
    if wanted_a == finished_a or wanted_b == finished_b:
        raise ValueError("the written sequence does not fail both trials")
    candidates = candidate_space(grammar, written)
    successes = [
        rank
        for rank, (index, op) in enumerate(candidates)
        if repairs_both(
            apply_fn, written, index, op, start_a, wanted_a, start_b, wanted_b
        )
    ]
    true_fix = (audit["true_fix"][0], canonical_op(audit["true_fix"][1]))
    if len(successes) != 1 or candidates[successes[0]] != true_fix:
        raise ValueError("the both-trials fix is not unique or is not the true fix")
    return candidates, {
        "grammar": grammar,
        "written": written,
        "describe": _DESCRIBERS[formalism],
        "apply": apply_fn,
        "start_a": start_a,
        "wanted_a": wanted_a,
        "start_b": start_b,
        "wanted_b": wanted_b,
        "success_index": successes[0],
    }


def verify_row_audit(row: dict) -> None:
    """Exhaustive per-row re-derivation of every frozen invariant."""
    audit = row["_audit"]
    candidates, machine = rederive_candidates(audit)
    describe = machine["describe"]
    grammar = machine["grammar"]
    written = machine["written"]
    k = audit["k_tried"]
    success_index = machine["success_index"]
    if not isinstance(k, int) or not 0 <= k <= min(success_index, DEEP_K):
        raise ValueError("tried-list length is out of the frozen range")
    if audit["success_index"] != success_index:
        raise ValueError("recorded success index does not re-derive")
    if audit["episode_success_turns"] != success_index + 1:
        raise ValueError("episode-success turn count does not re-derive")
    if audit["remaining_turns_after_tried"] != success_index - k + 1:
        raise ValueError("remaining-turn count does not re-derive")
    if audit["candidate_count"] != len(candidates):
        raise ValueError("candidate count does not re-derive")
    if audit["grammar_size"] != len(grammar):
        raise ValueError("grammar size does not re-derive")
    # The tried list is EXACTLY the first k candidates in canonical
    # order (legal + canonically ordered by construction of the space),
    # and every entry genuinely fails — re-simulated against both trials.
    tried = audit["tried"]
    if len(tried) != k:
        raise ValueError("tried-list length disagrees with k")
    prompt = row["messages"][0]["content"]
    for position, entry in enumerate(tried):
        index, op = entry["index"], canonical_op(entry["op"])
        if (index, op) != candidates[position]:
            raise ValueError("a tried entry is not the canonical candidate at its rank")
        out_a = run_patched(
            machine["apply"], machine["written"], index, op, machine["start_a"]
        )
        if out_a != machine["wanted_a"]:
            failed_on = "first"
        else:
            out_b = run_patched(
                machine["apply"], machine["written"], index, op, machine["start_b"]
            )
            if out_b == machine["wanted_b"]:
                raise ValueError("a tried entry repairs both trials")
            failed_on = "second"
        if entry["failed_on"] != failed_on:
            raise ValueError("a tried entry's recorded failing trial does not re-derive")
        if f"step {index + 1} to '{describe(op)}'" not in prompt:
            raise ValueError("a tried entry is not rendered in the prompt")
    # The target IS the canonical-next untried legal candidate.
    target = (audit["target"][0], canonical_op(audit["target"][1]))
    if target != candidates[k]:
        raise ValueError("the target is not the canonical-next untried candidate")
    if audit["target_is_true_fix"] is not (k == success_index):
        raise ValueError("target_is_true_fix does not re-derive")
    expected_answer = f"ANSWER: STEP {target[0] + 1}: {describe(target[1])}"
    if row["answer"] != expected_answer:
        raise ValueError("the answer is not the canonical-next candidate")
    match = ANSWER_FORMAT_RE.fullmatch(row["answer"].removeprefix("ANSWER: "))
    if match is None or int(match.group(1)) != target[0] + 1:
        raise ValueError("the answer does not carry the frozen STEP format")
    # The rendered per-step ranges: recorded counts re-derive, agree with
    # the exhaustive enumeration exactly, and the rendered statement sits
    # in the prompt byte-exactly, directly after the frozen rule text.
    counts = per_step_candidate_counts(grammar, written)
    if audit["per_step_candidate_counts"] != counts:
        raise ValueError("per-step candidate counts do not re-derive")
    verify_ranges_against_enumeration(counts, candidates, len(written))
    range_statement = render_range_statement(counts)
    if (
        f"{CANONICAL_ORDER_STATEMENT}\n{range_statement}\n" not in prompt
        or prompt.count(range_statement) != 1
    ):
        raise ValueError(
            "the rendered range statement is missing or detached from the "
            "frozen order statement"
        )
    # The think target IS the fixed-shape compact computation — a pure
    # function of the machine and k — under the frozen budget.
    expected_think = build_think(grammar, written, k, describe)
    if row["think"] != expected_think:
        raise ValueError("the think target is not the frozen compact computation")
    check_think_budget(row["think"])
    # In-prompt documentation of the enumeration.
    if CANONICAL_ORDER_STATEMENT not in prompt:
        raise ValueError("the frozen canonical-order statement is missing")
    action_list = "\n".join(
        f"  {position + 1}. {describe(op)}"
        for position, op in enumerate(grammar)
    )
    if action_list not in prompt or ASK_LINE not in prompt:
        raise ValueError("the numbered action list or the ask line is missing")
    if (k == 0) is not (NONE_TRIED_LINE in prompt):
        raise ValueError("the empty-tried line does not match k")
    if (k > 0) is not (TRIED_HEADER in prompt):
        raise ValueError("the tried-list header does not match k")
    if not audit["legality_clauses"] or any(
        clause not in prompt for clause in audit["legality_clauses"]
    ):
        raise ValueError("a legality clause is not documented in the prompt")


# ---------------------------------------------------------------------------
# enumeration-fidelity readout (the preregistered non-gating mechanism read)
# ---------------------------------------------------------------------------

def parse_step_answer(parsed: str | None) -> tuple[int, str] | None:
    """Parse 'STEP <k>: <corrected step>' with whitespace tolerance."""
    if parsed is None:
        return None
    text = re.sub(r"\s+", " ", str(parsed)).strip()
    match = re.fullmatch(r"STEP (\d+)\s*:\s*(.+)", text, re.IGNORECASE)
    if match is None:
        return None
    return int(match.group(1)), match.group(2).strip()


def enumeration_fidelity(audit: dict, parsed: str | None) -> dict:
    """Three booleans per row: is the proposed candidate (a) LEGAL —
    a bounded single-step change that differs from the written step;
    (b) UNTRIED — not among the k already-tried candidates; and
    (c) CANONICAL-NEXT — exactly the target. Unparseable answers score
    False on all three."""
    candidates, machine = rederive_candidates(audit)
    describe = machine["describe"]
    parsed_step = parse_step_answer(parsed)
    result = {"parseable": parsed_step is not None,
              "legal": False, "untried": False, "canonical_next": False}
    if parsed_step is None:
        return result
    step_number, description = parsed_step
    matches = [
        (index, op)
        for index, op in candidates
        if index + 1 == step_number and describe(op) == description
    ]
    if not matches:
        return result
    proposal = matches[0]
    result["legal"] = True
    k = audit["k_tried"]
    result["untried"] = proposal not in candidates[:k]
    result["canonical_next"] = proposal == candidates[k]
    return result


def episode_success_turns(audit: dict) -> dict:
    """Analytic perfect-enumerator turn counts (no model)."""
    return {
        "from_scratch": audit["episode_success_turns"],
        "remaining_after_tried": audit["remaining_turns_after_tried"],
    }


# ---------------------------------------------------------------------------
# corpus assembly
# ---------------------------------------------------------------------------

def _count_walk_at(rng: random.Random, index: int) -> dict:
    formalism = ENUM_FORMALISMS[index % len(ENUM_FORMALISMS)]
    k_target = K_CYCLE[(index // len(ENUM_FORMALISMS)) % len(K_CYCLE)]
    return count_walk_lesson(rng, formalism, k_target)


SKILLS = {
    "count_walk": _count_walk_at,
}
ARM_MIX = "count_walk=160"
HOLDOUT_MIX = "count_walk=40"
SMOKE_MIX = "count_walk=16"


def parse_mix(specification: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    seen: set[str] = set()
    for part in specification.split(","):
        if not part.strip():
            continue
        name, separator, raw_count = part.partition("=")
        name = name.strip()
        if not separator or name not in SKILLS or name in seen:
            raise ValueError(f"invalid mix entry {part!r}")
        count = int(raw_count)
        if count <= 0:
            raise ValueError(f"count must be positive: {part}")
        result.append((name, count))
        seen.add(name)
    if not result:
        raise ValueError("empty mix")
    return result


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def generate_curriculum(specification: str, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for skill, count in parse_mix(specification):
        for index in range(count):
            row = SKILLS[skill](rng, index)
            row["task_id"] = f"cdw_{skill}_{index:05d}"
            rows.append(row)
    rng.shuffle(rows)
    return rows


def validate_generated(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("curriculum is empty")
    required = {
        "messages", "think", "answer", "kind", "family", "surface", "level",
        "n_think_tokens", "row_weight", "task_id", "_audit",
    }
    serialized: set[str] = set()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {index} schema mismatch: {sorted(row)}")
        if row["family"] != "universal" or row["row_weight"] != 1.0:
            raise ValueError(f"row {index} family/weight mismatch")
        if len(row["messages"]) != 1 or row["messages"][0].get("role") != "user":
            raise ValueError(f"row {index} message schema mismatch")
        if not row["think"].strip():
            raise ValueError(f"row {index} malformed think target")
        try:
            check_think_budget(row["think"])
        except ValueError as error:
            raise ValueError(f"row {index} think budget/shape violation: {error}")
        if not row["answer"].startswith("ANSWER: ") or "\n" in row["answer"]:
            raise ValueError(f"row {index} malformed answer")
        if row["kind"] != "u_count_walk":
            raise ValueError(f"row {index} has an unknown kind: {row['kind']}")
        if row["surface"] not in ENUM_FORMALISMS:
            raise ValueError(f"row {index} has an unknown surface: {row['surface']}")
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        if audit.get("formalism") != row["surface"]:
            raise ValueError(f"row {index} formalism/surface mismatch")
        try:
            verify_row_audit(row)
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f"row {index} count-walk re-derivation failed: {error}")
        canonical = json.dumps(public_row(row), sort_keys=True, ensure_ascii=False)
        prompt = row["messages"][0]["content"]
        if canonical in serialized or prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {index} duplicate")
        serialized.add(canonical)
        prompts.add(prompt)
        task_ids.add(row["task_id"])
    return {
        "rows": len(rows),
        "kinds": dict(sorted(Counter(row["kind"] for row in rows).items())),
        "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
        "max_estimated_think_tokens": max(row["n_think_tokens"] for row in rows),
        "max_think_chars": max(len(row["think"]) for row in rows),
        "think_token_cap": THINK_TOKEN_CAP,
        "think_char_cap": THINK_CHAR_CAP,
    }


def check_banned_vocabulary(rows: list[dict]) -> None:
    patterns = [
        (re.compile(rf"\b{re.escape(token)}\b", re.IGNORECASE), token)
        for token in BANNED_PROMPT_TOKENS
    ]
    for index, row in enumerate(rows):
        haystack = "\n".join((row["messages"][0]["content"], row["think"], row["answer"]))
        for pattern, token in patterns:
            if pattern.search(haystack):
                raise ValueError(f"row {index} leaks banned vocabulary: {token!r}")


def check_corpus_balance(rows: list[dict]) -> dict:
    """Formalism balance and the frozen k-distribution coverage."""
    treatment_rows = [row for row in rows if row["kind"] == "u_count_walk"]
    formalisms = Counter(row["_audit"]["formalism"] for row in treatment_rows)
    k_by_formalism: dict[str, Counter] = {
        formalism: Counter() for formalism in formalisms
    }
    for row in treatment_rows:
        k_by_formalism[row["_audit"]["formalism"]][row["_audit"]["k_tried"]] += 1
    k_overall = Counter(row["_audit"]["k_tried"] for row in treatment_rows)
    turns = Counter(row["_audit"]["episode_success_turns"] for row in treatment_rows)
    balance = {
        "count_walk_formalisms": dict(sorted(formalisms.items())),
        "k_cycle": list(K_CYCLE),
        "k_tried_counts": {str(key): value for key, value in sorted(k_overall.items())},
        "k_tried_by_formalism": {
            formalism: {str(key): value for key, value in sorted(counts.items())}
            for formalism, counts in sorted(k_by_formalism.items())
        },
        "target_is_true_fix_rows": sum(
            bool(row["_audit"]["target_is_true_fix"]) for row in treatment_rows
        ),
        "episode_success_turns_counts": {
            str(key): value for key, value in sorted(turns.items())
        },
    }
    if treatment_rows:
        if (
            set(formalisms) != set(ENUM_FORMALISMS)
            or len(set(formalisms.values())) != 1
        ):
            raise ValueError(f"count-walk formalism balance out of range: {balance}")
        per_formalism = next(iter(formalisms.values()))
        if per_formalism % len(K_CYCLE) == 0:
            expected = {
                key: per_formalism // len(K_CYCLE) for key in K_CYCLE
            }
            for formalism, counts in k_by_formalism.items():
                if dict(counts) != expected:
                    raise ValueError(
                        f"count-walk k-distribution out of range for {formalism}: "
                        f"{balance}"
                    )
        if per_formalism >= len(K_CYCLE) and (
            0 not in k_overall or DEEP_K not in k_overall
        ):
            raise ValueError(f"count-walk k coverage lost its ends: {balance}")
        if 0 not in k_overall:
            raise ValueError(f"count-walk k coverage lost k=0: {balance}")
    return balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77191)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if sum(bool(value) for value in (args.mix, args.smoke)) != 1:
        parser.error("choose exactly one of --mix, --smoke")
    mix = SMOKE_MIX if args.smoke else args.mix
    rows = generate_curriculum(mix, args.seed)
    summary = validate_generated(rows)
    check_banned_vocabulary(rows)
    balance = check_corpus_balance(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"mix": parse_mix(mix), **summary, "balance": balance},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
