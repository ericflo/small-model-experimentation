from __future__ import annotations

import json
import random
from typing import Any


DSL_SPEC = """Available DSL:
- Variables are bare names from the input schema.
- String constants use double quotes.
- Expressions use prefix parentheses.
- Ops: sum, len, mod, add, sub, mul, format, contains, count_eq, tuple_get, sort, first, last, gt, ge, lt, eq, and, or, not, if, join.
- Output exactly one DSL expression and nothing else."""


def _case_line(index: int, case: dict[str, Any]) -> str:
    return (
        f"{index}. input={json.dumps(case['input'], sort_keys=True)} "
        f"expected={json.dumps(case['expected'], sort_keys=True)} "
        f"current_got={json.dumps(case.get('got'), sort_keys=True)}"
    )


def visible_block(record: dict[str, Any]) -> str:
    return "\n".join(_case_line(i + 1, case) for i, case in enumerate(record["visible"]))


def prompt_for_record(
    record: dict[str, Any],
    *,
    prompt_mode: str = "trace",
    trace_override: list[dict[str, Any]] | None = None,
) -> str:
    if prompt_mode not in {"trace", "no_trace", "shuffled_trace"}:
        raise ValueError(f"unknown prompt mode: {prompt_mode}")
    visible = trace_override if trace_override is not None else record["visible"]
    trace_text = "\n".join(_case_line(i + 1, case) for i, case in enumerate(visible))
    if prompt_mode == "no_trace":
        trace_text = "<withheld>"
    user = f"""{DSL_SPEC}

Task:
Repair the current DSL program so it matches the observed expected outputs.

Input schema:
{record['schema']}

Current DSL program:
{record['wrong_program']}

Visible execution cases:
{trace_text}

Corrected DSL program:"""
    return user


def messages_for_record(
    record: dict[str, Any],
    *,
    prompt_mode: str = "trace",
    trace_override: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You configure a deterministic executable DSL. Return only the corrected DSL expression.",
        },
        {"role": "user", "content": prompt_for_record(record, prompt_mode=prompt_mode, trace_override=trace_override)},
    ]


def shuffled_visible(records: list[dict[str, Any]], *, seed: int) -> list[list[dict[str, Any]]]:
    rng = random.Random(seed)
    traces = [row["visible"] for row in records]
    shuffled = traces[:]
    rng.shuffle(shuffled)
    if len(shuffled) > 1:
        for i, trace in enumerate(shuffled):
            if trace is traces[i]:
                j = (i + 1) % len(shuffled)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
    return shuffled
