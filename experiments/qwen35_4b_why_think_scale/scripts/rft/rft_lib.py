"""Rejection-sampling helpers: parse synthetic problems, filter native samples by execution."""
import json, re, sys
from pathlib import Path
ROOT = Path("/home/ericflo/Development/small-model-experimentation")
sys.path.insert(0, str(ROOT / "experiments/qwen35_4b_coding_fitness_harness/src"))
import code_env


def parse_problem(row):
    msg = row["messages"][0]["content"]
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", msg)
    entry = m.group(1) if m else None
    asserts = [l.strip() for l in msg.splitlines() if l.strip().startswith("assert ")]
    return entry, asserts


_PREAMBLE = (
    "import collections, functools, heapq, itertools, math, re, string\n"
    "from typing import *\n"
)


def code_passes(candidate_code, asserts, timeout_s=5.0):
    if not candidate_code:
        return False
    ok, _reason = code_env.static_safety_check(candidate_code)
    if not ok:
        return False
    script = _PREAMBLE + candidate_code + "\n" + "\n".join(asserts) + "\n"
    res = code_env.run_python_script(script, timeout_s=timeout_s)
    return bool(res.get("ok"))


def split_think_answer(full_text):
    """full completion is 'reasoning...</think>\\n\\n```python...```'. Return (think_inner, answer_code_fenced)."""
    text = full_text
    if "</think>" in text:
        think, after = text.split("</think>", 1)
    else:
        think, after = "", text
    think = think.replace("<think>", "").strip()
    return think, after.strip()
