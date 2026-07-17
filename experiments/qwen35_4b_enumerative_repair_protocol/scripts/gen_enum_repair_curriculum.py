#!/usr/bin/env python3
"""Designed enumerative-repair curriculum: teach SYSTEMATIC ENUMERATION.

Lifecycle 26 — the new-mechanism attack on the last goal-gating family.
Every failed predecessor dose taught the model to INFER the right fix
(eliminative inference — closed at every dose 80-800; even 2AFC
verification sat at chance). This dose teaches something never tried:
given failure evidence, propose the legal single-step candidates one per
turn in a FROZEN CANONICAL ORDER, let trial feedback decide, stop at
first success. Grounded in the repo's laws: C34 (brute-force search
dominates the model's reasoning — a model-level law) and the line's only
reliable installs being protocols (hygiene/explore/termination/
statechain).

MACHINERY: the eight legality-bounded machine formalisms of the menders
dose-scale cell (``troughline``, ``trinketcord``, ``crankwheel``,
``sigilslate``, ``barrowyoke``, ``balesled``, ``millround``,
``skeinreel``) are REUSED via a byte-identical copy of that cell's
reviewed generator (``gen_feedloop_curriculum.py`` — machinery imported,
never forked): the machine builders, describe/apply semantics, the
banned-vocabulary inventory, and the legal-candidate enumeration idea.
The LESSON changes completely.

EACH ROW is one PARTIAL enumeration episode rendered as plain text in
ONE user message:

- the machine spec with its legality clauses, PLUS a numbered ACTION
  LIST rendering the machine's full bounded grammar in its frozen order
  (this documents the enumeration order in-prompt);
- the broken written step sequence (exactly one step is off);
- BOTH trials' wanted+observed outcomes for the written sequence
  (failure evidence — the written run fails BOTH trials by
  construction);
- a FROZEN statement of the CANONICAL ORDER, byte-identical in every
  row: candidates are ranked by step number ascending, then by the
  changed-to action's position in the numbered action list (this is
  exactly the generator's enumeration order);
- a list of candidates ALREADY TRIED — the first ``k`` candidates in
  canonical order (``k`` varies over K_CYCLE, including k=0 rows and
  deep-in-the-list rows), each with its observed two-trial outcome; all
  are failures by construction (the generator simulates each against
  both trials);
- the ask: name the NEXT untried legal candidate in canonical order.

The think target narrates the protocol: enumerate the legal set, cross
off the tried ones, emit the next — never infer. The answer is
exact-match gradable in the predecessor cells' format
(``STEP <k>: <corrected step>``). The generator verifies per row, by
exhaustive re-derivation over the full candidate space, that the target
IS the canonical-next untried legal candidate, that exactly ONE
candidate repairs both trials (so a perfect enumerator's episode length
is well defined), and that every tried entry is legal, canonically
ordered, and genuinely failing. A banned-vocabulary scan (the menders
cell's full inventory, which bans the public blocker-family description
nouns — repair/rerun/test/debug/protocol/... — and every prior surface
pool) rejects any leak.
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

# The frozen canonical-order statement, byte-identical in EVERY row. It
# names exactly the generator's enumeration order: step index ascending,
# then the changed-to action's position in the rendered action list
# (the action list renders the bounded grammar in its frozen order, so
# op-grammar position and parameter order are documented in-prompt).
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
        raise ValueError(f"unknown enum-repair formalism: {formalism}")
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


def run_patched(apply_fn, written: list[tuple], index: int, op: tuple, start):
    patched = written[:index] + [op] + written[index + 1 :]
    return feedloop._run_steps(apply_fn, patched, start)


def repairs_both(apply_fn, written, index, op, start_a, wanted_a, start_b, wanted_b) -> bool:
    return (
        run_patched(apply_fn, written, index, op, start_a) == wanted_a
        and run_patched(apply_fn, written, index, op, start_b) == wanted_b
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


def enum_repair_lesson(rng: random.Random, formalism: str, k_target: int) -> dict:
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
            f"{tried_block}\n"
            f"{ASK_LINE}\n"
            f"{ANSWER_LINE} (format: STEP <k>: <corrected step>)"
        )
        if k_target == 0:
            crossed = (
                "Nothing has been tried yet, so the next candidate is the "
                "very first change in that order."
            )
        else:
            crossed = (
                f"The {k_target} tried changes are exactly the first "
                f"{k_target} in that order, and every one failed its "
                "two-trial run, so cross them off."
            )
        think = (
            f"The frozen order walks step 1 through step {length}, and "
            "within a step walks the numbered action list top to bottom, "
            "skipping only the action already written there; that yields "
            f"{len(candidates)} legal single-step changes in all. "
            f"{crossed} The next untried change in the frozen order is "
            f"step {target_index + 1} to '{describe(target_op)}'. Propose "
            "exactly that and let the next two-trial run judge it."
        )
        answer = f"STEP {target_index + 1}: {describe(target_op)}"
        return feedloop.make_row(
            prompt=prompt,
            think=think,
            answer=answer,
            kind="enum_repair",
            surface=formalism,
            level=length,
            audit={
                "truth_valid": True,
                "formalism": formalism,
                "steps": length,
                "grammar_size": len(grammar),
                "candidate_count": len(candidates),
                "k_tried": k_target,
                "success_index": success_index,
                "episode_success_turns": success_index + 1,
                "remaining_turns_after_tried": success_index - k_target + 1,
                "target_is_true_fix": candidates[k_target] == true_fix,
                "target": [target_index, list(target_op)],
                "true_fix": [bug_at, list(intended[bug_at])],
                "tried": tried_audit,
                "canonical_statement_in_prompt": True,
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
        f"could not construct a {formalism} enum-repair lesson at k={k_target}"
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
    if audit["grammar_size"] != len(machine["grammar"]):
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
    # In-prompt documentation of the enumeration.
    if CANONICAL_ORDER_STATEMENT not in prompt:
        raise ValueError("the frozen canonical-order statement is missing")
    action_list = "\n".join(
        f"  {position + 1}. {describe(op)}"
        for position, op in enumerate(machine["grammar"])
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

def _enum_repair_at(rng: random.Random, index: int) -> dict:
    formalism = ENUM_FORMALISMS[index % len(ENUM_FORMALISMS)]
    k_target = K_CYCLE[(index // len(ENUM_FORMALISMS)) % len(K_CYCLE)]
    return enum_repair_lesson(rng, formalism, k_target)


SKILLS = {
    "enum_repair": _enum_repair_at,
}
ARM_MIX = "enum_repair=160"
HOLDOUT_MIX = "enum_repair=40"
SMOKE_MIX = "enum_repair=16"


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
            row["task_id"] = f"erp_{skill}_{index:05d}"
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
        if not row["think"].strip() or "\n" in row["think"]:
            raise ValueError(f"row {index} malformed think target")
        if not row["answer"].startswith("ANSWER: ") or "\n" in row["answer"]:
            raise ValueError(f"row {index} malformed answer")
        if row["kind"] != "u_enum_repair":
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
            raise ValueError(f"row {index} enum-repair re-derivation failed: {error}")
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
    enum_rows = [row for row in rows if row["kind"] == "u_enum_repair"]
    formalisms = Counter(row["_audit"]["formalism"] for row in enum_rows)
    k_by_formalism: dict[str, Counter] = {
        formalism: Counter() for formalism in formalisms
    }
    for row in enum_rows:
        k_by_formalism[row["_audit"]["formalism"]][row["_audit"]["k_tried"]] += 1
    k_overall = Counter(row["_audit"]["k_tried"] for row in enum_rows)
    turns = Counter(row["_audit"]["episode_success_turns"] for row in enum_rows)
    balance = {
        "enum_repair_formalisms": dict(sorted(formalisms.items())),
        "k_cycle": list(K_CYCLE),
        "k_tried_counts": {str(key): value for key, value in sorted(k_overall.items())},
        "k_tried_by_formalism": {
            formalism: {str(key): value for key, value in sorted(counts.items())}
            for formalism, counts in sorted(k_by_formalism.items())
        },
        "target_is_true_fix_rows": sum(
            bool(row["_audit"]["target_is_true_fix"]) for row in enum_rows
        ),
        "episode_success_turns_counts": {
            str(key): value for key, value in sorted(turns.items())
        },
    }
    if enum_rows:
        if (
            set(formalisms) != set(ENUM_FORMALISMS)
            or len(set(formalisms.values())) != 1
        ):
            raise ValueError(f"enum-repair formalism balance out of range: {balance}")
        per_formalism = next(iter(formalisms.values()))
        if per_formalism % len(K_CYCLE) == 0:
            expected = {
                key: per_formalism // len(K_CYCLE) for key in K_CYCLE
            }
            for formalism, counts in k_by_formalism.items():
                if dict(counts) != expected:
                    raise ValueError(
                        f"enum-repair k-distribution out of range for {formalism}: "
                        f"{balance}"
                    )
        if per_formalism >= len(K_CYCLE) and (
            0 not in k_overall or DEEP_K not in k_overall
        ):
            raise ValueError(f"enum-repair k coverage lost its ends: {balance}")
        if 0 not in k_overall:
            raise ValueError(f"enum-repair k coverage lost k=0: {balance}")
    return balance


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mix", default=None)
    parser.add_argument("--seed", type=int, default=77190)
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
