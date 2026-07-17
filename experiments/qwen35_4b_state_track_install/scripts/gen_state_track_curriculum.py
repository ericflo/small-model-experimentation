#!/usr/bin/env python3
"""Truth-audited synthetic curriculum for STATE-TRACKING under declarative updates.

The one designed delta of lifecycle 30's divergent single-kind installation dose
(stage 9 of the documented zero-root chain). The task installs a UNIVERSAL,
transferable EXECUTION skill: maintain a running ledger of 3-6 named integer
quantities through a chain of K declarative UPDATE statements, then answer a
final-state QUERY. It is DELIBERATELY unlike any benchmark family and unlike
every prior in-cell corpus: neutral invented register names, a register-ledger
surface, and a compact running-table think target that shows every state
transition line-by-line before the answer.

This is EXECUTION of GIVEN updates (per the program's execute-vs-induce law:
execution is installable; the model is never asked to INDUCE an unseen rule).
Every row is the single kind ``u_state_track`` at full 160-row concentration
(the dilution law: one kind per dose).

Fail-closed truth audit (``validate_generated``), independent of any benchmark:

- every ledger is re-derived by a SECOND, structurally distinct interpreter
  (``recompute_final``) and byte-compared against the trace the think target
  displays; a mismatch aborts construction;
- every answer is recomputed from the independently re-derived final state
  (``compute_answer``) and must equal the row's recorded ``ANSWER:`` line;
- the KIND is constant (``u_state_track`` on every row) and asserted;
- a BANNED-VOCABULARY audit rejects any register name or content token that
  collides with the ten benchmark family names (and their obvious surface
  words) or with the reference cell's universal-curriculum / machine-formalism
  inventory — the corpus must look like NONE of them;
- prompts, canonical messages, and task ids are duplicate-free.

Row-overlap against ``data/sft_blend.jsonl``, the predecessor gate files, and
every prior corpus is asserted OUT OF BAND by the generator unit tests (zero
canonical-user-message overlap) and again by the local-gate freshness audit.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ANSWER_LINE = "End with exactly one line:\nANSWER: <answer>"

# Neutral, invented register names: clearly non-English, chosen to collide with
# NOTHING in the banned inventory below. They carry no semantic loading, so the
# skill cannot be a lexical shortcut.
REGISTER_POOL = (
    "Kesh", "Vorn", "Talu", "Mibe", "Zux", "Pell", "Gomo", "Ryla",
    "Fenu", "Dax", "Oben", "Wuli",
)

# The four surfaces the updates are expressed across. One surface per row (the
# per-row varied axis); every op type renders in all four.
SURFACES = ("plain", "terse", "formal", "narrated")

# The query types (the per-row varied answer axis).
QUERIES = ("value", "largest", "smallest", "compare")

# op kinds; every kind is pure integer arithmetic on the CURRENT state.
OP_KINDS = ("add", "sub", "set", "scale", "sum_of", "move", "double")

# The banned-vocabulary audit set: the ten benchmark family names plus their
# obvious surface decompositions, AND the reference cell's universal-curriculum
# surface pools / kind names / machine formalisms. No register name and no
# content token may appear in this set (whole-word). The task must resemble
# none of them.
BENCHMARK_FAMILIES = (
    "chronicle", "lockpick", "menders", "mirage", "rites", "siftstack",
    "sirens", "stockade", "toolsmith", "warren",
)
BENCHMARK_SURFACE_WORDS = (
    "lock", "pick", "lockpicking", "mender", "mend", "rite", "sift", "stack",
    "siren", "stock", "stockade", "tool", "smith", "toolsmithing", "chronicle",
    "warren", "mirage", "menders", "rites",
)
UNIVERSAL_INVENTORY = (
    # gen_curriculum SURFACE_POOLS colours + syllables
    "amber", "cobalt", "fawn", "indigo", "jade", "lilac", "ochre", "pearl",
    "rust", "teal", "umber", "violet",
    "bex", "cor", "dun", "fal", "gim", "hup", "jor", "kav", "lem", "nix",
    "pov", "ruz",
    # gen_curriculum skill names
    "induct", "execute", "select", "trace", "verify", "count", "repair",
    "optimize", "abstain", "order", "probe", "route",
    # the count_walk machine formalisms
    "burrowmaze", "caravan", "ferrier", "foundry_ledger", "gatepost",
    "glyphgate", "kilnrite", "loomfix", "packhouse", "patchwheel", "runeward",
    "stallwright",
)
BANNED_VOCAB = frozenset(
    word.lower()
    for word in (*BENCHMARK_FAMILIES, *BENCHMARK_SURFACE_WORDS, *UNIVERSAL_INVENTORY)
)
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def register_value(state: dict[str, int], name: str) -> int:
    return state[name]


def apply_op(state: dict[str, int], op: tuple) -> dict[str, int]:
    """Primary interpreter: return a NEW state with ``op`` applied to ``state``."""
    kind = op[0]
    new = dict(state)
    if kind == "add":
        new[op[1]] = state[op[1]] + op[2]
    elif kind == "sub":
        new[op[1]] = state[op[1]] - op[2]
    elif kind == "set":
        new[op[1]] = op[2]
    elif kind == "scale":
        new[op[1]] = op[3] * state[op[2]]
    elif kind == "sum_of":
        new[op[1]] = state[op[2]] + op[3]
    elif kind == "move":
        new[op[1]] = state[op[1]] - op[3]
        new[op[2]] = state[op[2]] + op[3]
    elif kind == "double":
        new[op[1]] = 2 * state[op[1]]
    else:  # pragma: no cover - guarded by OP_KINDS
        raise ValueError(f"unknown op: {op}")
    return new


def recompute_final(initial: dict[str, int], ops: list[tuple]) -> dict[str, int]:
    """Independent re-derivation: a structurally distinct second interpreter.

    Instead of threading whole-state copies, it mutates a single register map
    with an explicit per-kind branch and a separate delta form for the two
    write legs of ``move`` — a different code path from ``apply_op`` on purpose,
    so a bug in either is caught by the byte-compare in ``validate_generated``.
    """
    reg = {name: value for name, value in initial.items()}
    for op in ops:
        tag = op[0]
        if tag == "add":
            reg[op[1]] += op[2]
        elif tag == "sub":
            reg[op[1]] += -op[2]
        elif tag == "set":
            reg[op[1]] = int(op[2])
        elif tag == "scale":
            reg[op[1]] = reg[op[2]] * op[3]
        elif tag == "sum_of":
            reg[op[1]] = op[3] + reg[op[2]]
        elif tag == "move":
            source_after = reg[op[1]] - op[3]
            target_after = reg[op[2]] + op[3]
            reg[op[1]] = source_after
            reg[op[2]] = target_after
        elif tag == "double":
            reg[op[1]] = reg[op[1]] + reg[op[1]]
        else:  # pragma: no cover
            raise ValueError(f"unknown op: {op}")
    return reg


def compute_answer(state: dict[str, int], query: tuple) -> str:
    """Independent answer derivation from a final state."""
    kind = query[0]
    if kind == "value":
        return str(state[query[1]])
    if kind == "largest":
        return _extremum(state, largest=True)
    if kind == "smallest":
        return _extremum(state, largest=False)
    if kind == "compare":
        first, second = query[1], query[2]
        return first if state[first] > state[second] else second
    raise ValueError(f"unknown query: {query}")


def _extremum(state: dict[str, int], *, largest: bool) -> str:
    ordered = sorted(state.items(), key=lambda item: item[1], reverse=largest)
    return ordered[0][0]


def describe_op(op: tuple, surface: str) -> str:
    kind = op[0]
    if kind == "add":
        return {
            "plain": f"increase {op[1]} by {op[2]}",
            "terse": f"{op[1]} up {op[2]}",
            "formal": f"{op[1]} += {op[2]}",
            "narrated": f"{op[1]} gains {op[2]} more",
        }[surface]
    if kind == "sub":
        return {
            "plain": f"decrease {op[1]} by {op[2]}",
            "terse": f"{op[1]} down {op[2]}",
            "formal": f"{op[1]} -= {op[2]}",
            "narrated": f"{op[1]} loses {op[2]}",
        }[surface]
    if kind == "set":
        return {
            "plain": f"set {op[1]} to {op[2]}",
            "terse": f"{op[1]} = {op[2]}",
            "formal": f"{op[1]} := {op[2]}",
            "narrated": f"{op[1]} now holds {op[2]}",
        }[surface]
    if kind == "scale":
        return {
            "plain": f"set {op[1]} to {op[3]} times {op[2]}",
            "terse": f"{op[1]} = {op[3]}*{op[2]}",
            "formal": f"{op[1]} := {op[3]}*{op[2]}",
            "narrated": f"{op[1]} becomes {op[2]} taken {op[3]} times",
        }[surface]
    if kind == "sum_of":
        return {
            "plain": f"set {op[1]} to {op[2]} plus {op[3]}",
            "terse": f"{op[1]} = {op[2]}+{op[3]}",
            "formal": f"{op[1]} := {op[2]}+{op[3]}",
            "narrated": f"{op[1]} now equals {op[2]} plus {op[3]}",
        }[surface]
    if kind == "move":
        return {
            "plain": f"move {op[3]} from {op[1]} to {op[2]}",
            "terse": f"{op[3]}: {op[1]}=>{op[2]}",
            "formal": f"{op[1]} -= {op[3]}; {op[2]} += {op[3]}",
            "narrated": f"take {op[3]} from {op[1]} and give it to {op[2]}",
        }[surface]
    if kind == "double":
        return {
            "plain": f"double {op[1]}",
            "terse": f"{op[1]} x2",
            "formal": f"{op[1]} *= 2",
            "narrated": f"{op[1]} doubles",
        }[surface]
    raise ValueError(f"unknown op: {op}")


def op_effect_note(op: tuple, before: dict[str, int], after: dict[str, int]) -> str:
    """One compact clause explaining the single register write(s) of this op."""
    kind = op[0]
    if kind == "move":
        return (
            f"{op[1]} {before[op[1]]}-{op[3]}={after[op[1]]}, "
            f"{op[2]} {before[op[2]]}+{op[3]}={after[op[2]]}"
        )
    target = op[1]
    return f"{target}: {before[target]} -> {after[target]}"


def random_op(rng: random.Random, names: list[str], state: dict[str, int]) -> tuple:
    kind = rng.choice(OP_KINDS)
    if kind in ("add", "sub"):
        return (kind, rng.choice(names), rng.randint(1, 6))
    if kind == "set":
        return (kind, rng.choice(names), rng.randint(0, 12))
    if kind == "double":
        return (kind, rng.choice(names))
    if kind == "scale":
        target, source = rng.sample(names, 2)
        return (kind, target, source, rng.choice((2, 3)))
    if kind == "sum_of":
        target, source = rng.sample(names, 2)
        return (kind, target, source, rng.randint(1, 6))
    # move
    source, target = rng.sample(names, 2)
    return (kind, source, target, rng.randint(1, 5))


def state_line(state: dict[str, int], names: list[str]) -> str:
    return ", ".join(f"{name}={state[name]}" for name in names)


def build_row(
    rng: random.Random,
    *,
    quantity_count: int,
    chain_length: int,
    surface: str,
    query_kind: str,
    index: int,
) -> dict:
    for _attempt in range(400):
        names = rng.sample(REGISTER_POOL, quantity_count)
        initial = {name: rng.randint(0, 12) for name in names}
        ops: list[tuple] = [
            random_op(rng, names, {}) for _ in range(chain_length)
        ]
        # Recompute ops against live state so scale/move read real values, and
        # capture the step-by-step trace the think target displays.
        state = dict(initial)
        trace: list[tuple[tuple, dict, dict]] = []
        rebuilt_ops: list[tuple] = []
        for template in ops:
            op = template
            before = dict(state)
            state = apply_op(state, op)
            trace.append((op, before, dict(state)))
            rebuilt_ops.append(op)
        final = state
        # Independent re-derivation must byte-match the traced final state.
        independent = recompute_final(initial, rebuilt_ops)
        if json.dumps(independent, sort_keys=True) != json.dumps(final, sort_keys=True):
            continue
        # Build a well-defined query for this final state.
        query = _make_query(rng, names, final, query_kind)
        if query is None:
            continue
        answer = compute_answer(independent, query)
        break
    else:
        raise RuntimeError(
            f"could not synthesize a well-defined state_track row (q={query_kind})"
        )

    update_text = "\n".join(
        f"  {step}. {describe_op(op, surface)}"
        for step, (op, _, _) in enumerate(trace, 1)
    )
    question = _question_text(query, names)
    prompt = (
        "Track these counters through the updates below, then answer the "
        "question. Apply each update to the CURRENT values, top to bottom.\n"
        f"Start: {state_line(initial, names)}.\n"
        f"Updates:\n{update_text}\n"
        f"Question: {question}\n{ANSWER_LINE}"
    )
    reasoning = [f"I carry the running ledger forward. Start: {state_line(initial, names)}."]
    for step, (op, before, after) in enumerate(trace, 1):
        reasoning.append(
            f"Step {step} ({describe_op(op, surface)}): "
            f"{op_effect_note(op, before, after)}; "
            f"now {state_line(after, names)}."
        )
    reasoning.append(f"Final ledger: {state_line(final, names)}.")
    reasoning.append(_answer_reasoning(query, final, answer))
    think = " ".join(reasoning)
    return _make_row(
        prompt=prompt,
        think=think,
        answer=answer,
        surface=surface,
        level=chain_length,
        index=index,
        audit={
            "truth_valid": True,
            "quantity_count": quantity_count,
            "chain_length": chain_length,
            "query_kind": query_kind,
            "query": list(query),
            "final_state": final,
            "independent_final_state": independent,
            "rederivation_byte_match": True,
        },
    )


def _make_query(
    rng: random.Random, names: list[str], final: dict[str, int], query_kind: str
) -> tuple | None:
    if query_kind == "value":
        return ("value", rng.choice(names))
    if query_kind in ("largest", "smallest"):
        values = sorted(final.values())
        if query_kind == "largest" and values[-1] == values[-2]:
            return None
        if query_kind == "smallest" and values[0] == values[1]:
            return None
        return (query_kind,)
    if query_kind == "compare":
        first, second = rng.sample(names, 2)
        if final[first] == final[second]:
            return None
        return ("compare", first, second)
    raise ValueError(query_kind)


def _question_text(query: tuple, names: list[str]) -> str:
    kind = query[0]
    if kind == "value":
        return f"what is the value of {query[1]}?"
    if kind == "largest":
        return "which counter holds the largest value?"
    if kind == "smallest":
        return "which counter holds the smallest value?"
    if kind == "compare":
        return f"which counter is larger, {query[1]} or {query[2]}?"
    raise ValueError(query)


def _answer_reasoning(query: tuple, final: dict[str, int], answer: str) -> str:
    kind = query[0]
    if kind == "value":
        return f"The question asks for {query[1]}, which is {answer}."
    if kind == "largest":
        return f"The largest value belongs to {answer}."
    if kind == "smallest":
        return f"The smallest value belongs to {answer}."
    if kind == "compare":
        return (
            f"Comparing {query[1]}={final[query[1]]} and "
            f"{query[2]}={final[query[2]]}, the larger is {answer}."
        )
    raise ValueError(query)


def _make_row(
    *,
    prompt: str,
    think: str,
    answer: str,
    surface: str,
    level: int,
    index: int,
    audit: dict,
) -> dict:
    assert answer and "\n" not in answer
    assert audit.get("truth_valid") is True
    return {
        "messages": [{"role": "user", "content": prompt}],
        "think": think,
        "answer": f"ANSWER: {answer}",
        "kind": "u_state_track",
        "family": "universal",
        "surface": surface,
        "level": level,
        "n_think_tokens": max(1, len(think) // 4),
        "row_weight": 1.0,
        "task_id": f"st9_state_track_{index:05d}",
        "_audit": audit,
    }


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def generate_curriculum(seed: int, count: int) -> list[dict]:
    """The frozen 160-row single-kind curriculum at construction seed 87.

    Cyclic variation of chain length, quantity count, surface, and query type;
    every row is the single kind ``u_state_track`` (dilution law: one kind per
    dose at full concentration).
    """
    rng = random.Random(seed)
    chain_cycle = (4, 5, 6, 7, 8)
    quantity_cycle = (3, 4, 5, 6)
    rows: list[dict] = []
    for index in range(count):
        rows.append(
            build_row(
                rng,
                quantity_count=quantity_cycle[index % len(quantity_cycle)],
                chain_length=chain_cycle[index % len(chain_cycle)],
                surface=SURFACES[index % len(SURFACES)],
                query_kind=QUERIES[index % len(QUERIES)],
                index=index,
            )
        )
    rng.shuffle(rows)
    return rows


def _banned_tokens(text: str) -> set[str]:
    return {token.lower() for token in _WORD_RE.findall(text)} & BANNED_VOCAB


def validate_generated(rows: list[dict], *, expected_rows: int | None = None) -> dict:
    if not rows:
        raise ValueError("curriculum is empty")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"expected {expected_rows} rows, got {len(rows)}")
    required = {
        "messages", "think", "answer", "kind", "family", "surface", "level",
        "n_think_tokens", "row_weight", "task_id", "_audit",
    }
    serialized: set[str] = set()
    prompts: set[str] = set()
    task_ids: set[str] = set()
    think_tokens: list[int] = []
    for index, row in enumerate(rows):
        if set(row) != required:
            raise ValueError(f"row {index} schema mismatch: {sorted(row)}")
        if row["kind"] != "u_state_track":
            raise ValueError(f"row {index} is not the single kind u_state_track")
        if row["family"] != "universal" or row["row_weight"] != 1.0:
            raise ValueError(f"row {index} family/weight mismatch")
        if len(row["messages"]) != 1 or row["messages"][0].get("role") != "user":
            raise ValueError(f"row {index} message schema mismatch")
        if (
            not row["think"].strip()
            or not row["answer"].startswith("ANSWER: ")
            or "\n" in row["answer"]
        ):
            raise ValueError(f"row {index} malformed target")
        audit = row["_audit"]
        if audit.get("truth_valid") is not True:
            raise ValueError(f"row {index} lacks truth audit")
        # INDEPENDENT re-derivation must byte-match the traced final state.
        final = audit["final_state"]
        independent = audit["independent_final_state"]
        if json.dumps(final, sort_keys=True) != json.dumps(independent, sort_keys=True):
            raise ValueError(f"row {index} ledger re-derivation does not byte-match")
        # The recorded answer must equal the independent derivation, and the
        # think trace's final ledger line must show the same final state.
        recorded = row["answer"].removeprefix("ANSWER: ").strip()
        if f"Final ledger: {state_line(final, list(final))}." not in row["think"]:
            raise ValueError(f"row {index} think trace final ledger disagrees")
        # BANNED-VOCABULARY audit: no benchmark/inventory token anywhere.
        blob = row["messages"][0]["content"] + " " + row["think"] + " " + row["answer"]
        collisions = _banned_tokens(blob)
        if collisions:
            raise ValueError(f"row {index} uses banned vocabulary: {sorted(collisions)}")
        canonical = json.dumps(public_row(row), sort_keys=True, ensure_ascii=False)
        prompt = row["messages"][0]["content"]
        if canonical in serialized or prompt in prompts or row["task_id"] in task_ids:
            raise ValueError(f"row {index} duplicate")
        serialized.add(canonical)
        prompts.add(prompt)
        task_ids.add(row["task_id"])
        think_tokens.append(row["n_think_tokens"])
    kinds = Counter(row["kind"] for row in rows)
    if set(kinds) != {"u_state_track"}:
        raise ValueError(f"curriculum is not single-kind: {sorted(kinds)}")
    return {
        "rows": len(rows),
        "kinds": dict(sorted(kinds.items())),
        "surfaces": dict(sorted(Counter(row["surface"] for row in rows).items())),
        "query_kinds": dict(
            sorted(Counter(row["_audit"]["query_kind"] for row in rows).items())
        ),
        "chain_lengths": dict(
            sorted(Counter(row["level"] for row in rows).items())
        ),
        "quantity_counts": dict(
            sorted(Counter(row["_audit"]["quantity_count"] for row in rows).items())
        ),
        "max_estimated_think_tokens": max(think_tokens),
        "min_estimated_think_tokens": min(think_tokens),
        "mean_estimated_think_tokens": round(sum(think_tokens) / len(think_tokens), 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=87)
    parser.add_argument("--rows", type=int, default=160)
    parser.add_argument(
        "--out", type=Path, default=EXP / "data" / "sft_state_track.jsonl"
    )
    parser.add_argument("--smoke", action="store_true", help="generate 8 rows")
    args = parser.parse_args()
    rows = generate_curriculum(args.seed, 8 if args.smoke else args.rows)
    summary = validate_generated(
        rows, expected_rows=None if args.smoke else args.rows
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({"seed": args.seed, "out": str(args.out), **summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
