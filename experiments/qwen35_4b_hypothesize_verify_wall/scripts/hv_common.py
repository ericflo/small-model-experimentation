"""Shared helpers for the hypothesize-verify-wall experiment: config, task (de)serialization,
the C36 behavioral skeleton metric + its probe-robust (mimicry-proof) extension.

Metric definitions (README review must-fix 1):
  legacy       -- C36 model_structure_correct: model program behavior on the 8 VISIBLE inputs
                  matches some param-fill of the TRUE op-type skeleton.
  probe_robust -- legacy AND the matching fill also reproduces the model program's behavior on
                  the 6 fresh PROBE inputs (labels never shown to the model).
  mimicry      -- legacy but not probe_robust (visible-match, probe-divergent-or-error).
  full         -- model program reproduces the true outputs on visible AND hidden examples.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from itertools import product
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402

OP_RE = re.compile(r"^([A-Za-z_]+)\((-?\d+)\)$")


def load_cfg():
    import yaml
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def sfx(smoke):
    return "_smoke" if smoke else ""


def fam_of(name):
    return FAM.FAMILIES[name]


def types_of(fam):
    return list(fam["prims"])


def opts_of(fam):
    return {n: o for n, (_, a, o) in fam["prims"].items() if a}


def parse_op_repr(s):
    m = OP_RE.match(s)
    return (m.group(1), int(m.group(2))) if m else (s, None)


def task_ops(t):
    """Concrete (op, param) pipeline from a task row (prefers the serialized 'ops' field)."""
    if "ops" in t:
        return [(op, k) for op, k in t["ops"]]
    return [parse_op_repr(s) for s in t["target_ops"]]


def true_types(t):
    return [s.split("(")[0] for s in t["target_ops"]]


def ident_prompt(fam, t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `{fam['sig']}` reproducing this for all such inputs. Only a ```python code block.")


def exec_seq(fam, ops, x):
    st = x
    for op, k in ops:
        st = FAM.apply_op(fam, op, k, st)
        if st is None:
            return None
    return st


def solves(fam, ops, exs):
    return all(exec_seq(fam, ops, e["input"]) == e["output"] for e in exs)


def all_fills(fam, optypes):
    OPTS = opts_of(fam)
    ranges = [OPTS[op] if op in OPTS else [None] for op in optypes]
    return [list(zip(optypes, combo)) for combo in product(*ranges)]


def fills_visible(fam, optypes, task):
    return [f for f in all_fills(fam, optypes) if solves(fam, f, task["visible"])]


def skeletonfill_hidden(fam, optypes, task):
    return any(solves(fam, f, task["hidden"]) for f in fills_visible(fam, optypes, task))


def to_public(t):
    return [{"call_expr": f"transform({e['input']!r})", "expected_expr": f"{e['output']!r}"} for e in t["visible"]]


def to_hidden(t):
    return [f"assert transform({e['input']!r}) == {e['output']!r}" for e in t["hidden"]]


def py_solves(code, t):
    if not code:
        return False
    try:
        return bool(E.execute_public_and_asserts(code, to_public(t), to_hidden(t))["full_pass"])
    except Exception:
        return False


def behavior(code, inputs):
    """Sandbox-exec the model program on `inputs`; parsed values with '__ERR__' per failing call,
    or None when the program cannot be run at all."""
    if not code:
        return None
    public = [{"call_expr": f"transform({x!r})", "expected_expr": "None"} for x in inputs]
    try:
        r = E.execute_public_and_asserts(code, public, [])
        outs = []
        for po in r.get("public_outputs", []):
            try:
                outs.append(ast.literal_eval(po))
            except Exception:
                outs.append("__ERR__")
        return outs if len(outs) == len(inputs) else None
    except Exception:
        return None


def fills_with_outputs(fam, t):
    """For every param-fill of the TRUE skeleton executable on all visible inputs: its visible
    outputs and (if executable on all probe inputs) its probe outputs."""
    vis_in = [e["input"] for e in t["visible"]]
    probe_in = t.get("probe_inputs", [])
    out = []
    for f in all_fills(fam, true_types(t)):
        vis = [exec_seq(fam, f, x) for x in vis_in]
        if None in vis:
            continue
        probe = [exec_seq(fam, f, x) for x in probe_in]
        out.append((vis, None if (None in probe) else probe))
    return out


def grade_candidate(fam, t, code, fills):
    """Grade one extracted model program against the four metrics; single sandbox exec."""
    res = {"parsed": code is not None, "legacy": False, "probe_robust": False,
           "mimicry": False, "full": False}
    if not code:
        return res
    vis_in = [e["input"] for e in t["visible"]]
    probe_in = t.get("probe_inputs", [])
    hid_in = [e["input"] for e in t["hidden"]]
    beh = behavior(code, vis_in + probe_in + hid_in)
    if beh is None:
        return res
    nv, np_ = len(vis_in), len(probe_in)
    beh_vis, beh_probe, beh_hid = beh[:nv], beh[nv:nv + np_], beh[nv + np_:]
    vis_out = [e["output"] for e in t["visible"]]
    hid_out = [e["output"] for e in t["hidden"]]
    res["full"] = beh_vis == vis_out and beh_hid == hid_out
    if "__ERR__" not in beh_vis:
        res["legacy"] = any(fv == beh_vis for fv, _ in fills)
        if res["legacy"]:
            pr = ("__ERR__" not in beh_probe) and any(
                fv == beh_vis and fp == beh_probe for fv, fp in fills if fp is not None)
            res["probe_robust"] = bool(pr)
            res["mimicry"] = not pr
    return res


def extract_code(gen_text, think_mode):
    """Answer-region ```python block. In think mode only text after the LAST </think> counts
    (the think region may sketch code; the graded answer is the post-think block)."""
    region = gen_text.split("</think>")[-1] if think_mode else gen_text
    code, _ = E.extract_candidate_code(region.strip(), "transform")
    return code


def load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def dump_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
