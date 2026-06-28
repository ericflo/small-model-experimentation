from __future__ import annotations

import json
import random
from typing import Any


DSL_SPEC = """Available DSL:
- Variables are bare names from the input schema.
- String constants use double quotes.
- Expressions use prefix parentheses.
- Ops: sum, len, mod, add, sub, format, contains, count_eq, tuple_get, sort, first, last, gt, ge, lt, eq, and, or, not, if, join.
- Output exactly one DSL expression and nothing else."""


GRAPHIR_SPEC = """Available GraphIR:
- One assignment per line.
- Registers are r0, r1, r2, ...
- The final line must assign out.
- Arguments may be registers, input variables, integers, or double-quoted strings.
- Ops: SUM, LEN, MOD, ADD, SUB, FORMAT, CONTAINS, COUNT_EQ, GET, SORT, FIRST, LAST, GT, GE, LT, EQ, AND, OR, NOT, IF, JOIN.
- Example:
r0 = SORT values
r1 = GET r0 index
r2 = SUM values
r3 = ADD r1 r2
r4 = GT r3 threshold
out = IF r4 high_label low_label
- Output only GraphIR assignments and nothing else."""


def _case_line(index: int, case: dict[str, Any], *, got_key: str = "got") -> str:
    got = case.get(got_key, case.get("got"))
    return (
        f"{index}. input={json.dumps(case['input'], sort_keys=True)} "
        f"expected={json.dumps(case['expected'], sort_keys=True)} "
        f"got={json.dumps(got, sort_keys=True)}"
    )


def _trace_text(record: dict[str, Any], *, prompt_mode: str, trace_override=None, got_key: str = "got") -> str:
    visible = trace_override if trace_override is not None else record["visible"]
    if prompt_mode == "no_trace":
        return "<withheld>"
    return "\n".join(_case_line(i + 1, case, got_key=got_key) for i, case in enumerate(visible))


def dsl_prompt(record: dict[str, Any], *, prompt_mode: str, trace_override=None) -> str:
    return f"""{DSL_SPEC}

Task:
Repair the current DSL program so it matches the observed expected outputs.

Input schema:
{record['schema']}

Current DSL program:
{record['wrong_program']}

Visible execution cases:
{_trace_text(record, prompt_mode=prompt_mode, trace_override=trace_override)}

Corrected DSL program:"""


def graph_construct_prompt(record: dict[str, Any], *, prompt_mode: str, trace_override=None) -> str:
    return f"""{GRAPHIR_SPEC}

Task:
Configure an executable register graph that matches the observed expected outputs.

Input schema:
{record['schema']}

Current DSL program:
{record['wrong_program']}

Visible execution cases:
{_trace_text(record, prompt_mode=prompt_mode, trace_override=trace_override)}

Corrected GraphIR:"""


def graph_repair_prompt(record: dict[str, Any], *, prompt_mode: str, trace_override=None) -> str:
    return f"""{GRAPHIR_SPEC}

Task:
Repair the candidate register graph using the visible execution mismatches.

Input schema:
{record['schema']}

Candidate GraphIR:
{record['candidate_graph']}

Visible execution cases:
{_trace_text(record, prompt_mode=prompt_mode, trace_override=trace_override, got_key='candidate_got')}

Corrected GraphIR:"""


def prompt_for_record(record: dict[str, Any], *, prompt_mode: str, trace_override=None) -> str:
    task = record.get("task", "graph_construct")
    if task == "dsl":
        return dsl_prompt(record, prompt_mode=prompt_mode, trace_override=trace_override)
    if task == "graph_construct":
        return graph_construct_prompt(record, prompt_mode=prompt_mode, trace_override=trace_override)
    if task == "graph_repair":
        return graph_repair_prompt(record, prompt_mode=prompt_mode, trace_override=trace_override)
    raise ValueError(f"unknown task: {task}")


def messages_for_record(record: dict[str, Any], *, prompt_mode: str = "trace", trace_override=None):
    task = record.get("task", "graph_construct")
    if task == "dsl":
        system = "You repair deterministic DSL programs. Return only the corrected DSL expression."
    elif task == "graph_repair":
        system = "You repair deterministic executable register graphs. Return only GraphIR assignments."
    else:
        system = "You configure deterministic executable register graphs. Return only GraphIR assignments."
    return [
        {"role": "system", "content": system},
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
