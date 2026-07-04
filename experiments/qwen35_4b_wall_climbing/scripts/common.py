"""Shared task-generation, prompt, and grading for the banking experiment. The identification prompt is
IDENTICAL to qwen35_4b_coverage_vs_selection (no op-menu) so harvest, training, and eval all match C17."""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
import families as FAM  # noqa: E402
import code_env as E  # noqa: E402


def ident_prompt(fam, t):
    lines = "\n".join(f"transform({e['input']!r}) == {e['output']!r}" for e in t["visible"])
    return (f"Infer the Python function `transform` from these input/output examples:\n{lines}\n\n"
            f"Write `{fam['sig']}` reproducing this for all such inputs. Only a ```python code block.")


EXEC_TMPL = """import json
{code}
vis_in = {vis_in!r}
vis_exp = {vis_exp!r}
hid_in = {hid_in!r}
hid_exp = {hid_exp!r}
def outs(inputs):
    r = []
    for x in inputs:
        try:
            r.append(repr(transform(x)))
        except BaseException as e:
            r.append("ERR:" + type(e).__name__)
    return r
vo = outs(vis_in); ho = outs(hid_in)
vis_pass = all(a == repr(b) for a, b in zip(vo, vis_exp))
hid_pass = all(a == repr(b) for a, b in zip(ho, hid_exp))
print(json.dumps({{"vis": vis_pass, "hid": hid_pass, "sig": ho}}))
"""


def grade(code, t):
    """Return (visible_pass, full_pass, hidden_behavior_signature) or (False, False, None)."""
    if not code:
        return (False, False, None)
    safe, _ = E.static_safety_check(code)
    if not safe:
        return (False, False, None)
    script = EXEC_TMPL.format(
        code=code,
        vis_in=[e["input"] for e in t["visible"]], vis_exp=[e["output"] for e in t["visible"]],
        hid_in=[e["input"] for e in t["hidden"]], hid_exp=[e["output"] for e in t["hidden"]])
    r = E.run_python_script(script, timeout_s=5.0)
    if not r["ok"]:
        return (False, False, None)
    try:
        p = json.loads(r["stdout"].strip().splitlines()[-1])
    except Exception:
        return (False, False, None)
    vis, hid = bool(p["vis"]), bool(p["hid"])
    return (vis, vis and hid, tuple(p["sig"]) if vis else None)


def gen_tasks(fam, depth_tasks, rng, start_id=0, k_visible=8, m_hidden=8):
    """depth_tasks: list of (depth, count). Returns verified tasks."""
    tasks = []
    for depth, count in depth_tasks:
        made = 0
        while made < count:
            t = FAM.make_task(fam, start_id + len(tasks), depth, rng, k_visible=k_visible, m_hidden=m_hidden)
            if t:
                tasks.append(t); made += 1
    return tasks
