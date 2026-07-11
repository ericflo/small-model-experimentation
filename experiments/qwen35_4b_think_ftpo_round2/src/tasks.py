"""Unified task sourcing and scoring for harvest and whitebox evals.

Two procedural sources, both self-contained in this experiment:
- gym atoms from the 10 TRAINED gauntlet families (brinework/spindle are the
  corpus's held-out families and are deliberately never harvested here);
- list-transform code tasks (gen_tasks port), scored by sandboxed execution
  against hidden examples.

Every item carries a stable id, a single user-message prompt, and a scorer
mapping full model output text -> score in [0, 1].
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import code_env  # noqa: E402
import gen_tasks  # noqa: E402
from gym import base as gym_base  # noqa: E402
from gym.families import TRAINED_FAMILIES, load as load_family  # noqa: E402


@dataclass
class TaskItem:
    item_id: str
    source: str            # "gym" | "code"
    family: str            # gym family name, or "listxform"
    level: int             # gym level, or code depth
    prompt: str
    payload: dict = field(default_factory=dict)


def gym_cells(levels: list[int]) -> list[tuple[str, int]]:
    return [(fam, lvl) for fam in TRAINED_FAMILIES for lvl in levels]


def make_gym_items(family: str, level: int, seed: int, n: int) -> list[TaskItem]:
    family_mod = load_family(family)
    items = family_mod.gen_atoms(seed, level, n)
    return [
        TaskItem(
            item_id=f"gym-{family}-L{level}-s{seed}-{i}",
            source="gym",
            family=family,
            level=level,
            prompt=item["prompt"],
            payload=item,
        )
        for i, item in enumerate(items)
    ]


def make_code_items(depth: int, seed: int, n: int) -> list[TaskItem]:
    import random

    rng = random.Random(seed)
    out: list[TaskItem] = []
    attempts = 0
    while len(out) < n and attempts < n * 30:
        attempts += 1
        task = gen_tasks.make_task(len(out), depth, rng)
        if task is None:
            continue
        out.append(
            TaskItem(
                item_id=f"code-d{depth}-s{seed}-{len(out)}",
                source="code",
                family="listxform",
                level=depth,
                prompt=gen_tasks.prompt_for(task),
                payload=task,
            )
        )
    return out


def _answer_region(text: str) -> str:
    if "</think>" in text:
        return text.split("</think>")[-1]
    return text


def score_item(item: TaskItem, output_text: str) -> float:
    if item.source == "gym":
        family_mod = load_family(item.family)
        return float(family_mod.score_atom(item.payload, output_text))
    # code: extract transform() from the answer region, grade on hidden examples
    answer = _answer_region(output_text)
    answer = gym_base.strip_terminal_markers(answer)
    code, _ = code_env.extract_candidate_code(answer, "transform")
    if not code:
        return 0.0
    ok, _reason = code_env.static_safety_check(code)
    if not ok:
        return 0.0
    result = code_env.execute_public_and_asserts(
        code,
        gen_tasks.to_public_cases(item.payload),
        gen_tasks.to_hidden_asserts(item.payload),
    )
    return 1.0 if result.get("full_pass") else 0.0


# --- format-shifted rendering for the transfer slice ------------------------

def render_format_variant(item: TaskItem, variant: int) -> str:
    """Re-render the same task under an alternate scaffold (transfer probe).

    variant 1: strip the family's terse answer-protocol line and ask for the
               answer as a plain final line.
    variant 2: chat-style framing with a preamble persona.
    """
    prompt = item.prompt
    if variant == 1:
        lines = [
            line for line in prompt.splitlines()
            if "ANSWER:" not in line and "one ```python block" not in line
        ]
        lines.append("")
        if item.source == "gym":
            lines.append("Finish with your final answer on the last line by itself.")
        else:
            lines.append("Finish with the complete Python function, nothing after it.")
        return "\n".join(lines)
    if variant == 2:
        return (
            "You are helping a colleague debug a puzzle they are stuck on. "
            "Here is what they sent you:\n\n---\n" + prompt + "\n---\n\n"
            "Work it out and give them the answer in the required format."
        )
    raise ValueError(f"unknown format variant: {variant}")
