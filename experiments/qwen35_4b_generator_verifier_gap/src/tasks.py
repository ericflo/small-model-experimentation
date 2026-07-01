"""MBPP task loading + sandboxed execution verifier.

Standard MBPP protocol: the model sees the NL description plus one example assert
(to fix the function signature); the candidate is verified against the full
`test_list` (the remaining asserts act as held-out checks of correctness).
"""
from __future__ import annotations

import ast
import multiprocessing as mp
import re
import sys
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class Task:
    task_id: int
    prompt: str
    test_list: list[str]
    test_imports: list[str] = field(default_factory=list)

    @property
    def entry_point(self) -> str | None:
        """Function name parsed from the first assert, e.g. assert foo(...) == ..."""
        for t in self.test_list:
            m = re.search(r"assert\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", t)
            if m:
                return m.group(1)
        return None


def load_mbpp(split: str = "test", limit: int | None = None, offset: int = 0) -> list[Task]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "sanitized")[split]
    tasks = [
        Task(
            task_id=int(r["task_id"]),
            prompt=str(r["prompt"]).strip(),
            test_list=list(r["test_list"]),
            test_imports=list(r.get("test_imports") or []),
        )
        for r in ds
    ]
    tasks = tasks[offset:]
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


# ---------------------------------------------------------------------------
# Code extraction from a model generation
# ---------------------------------------------------------------------------
_FENCE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_code(text: str) -> str:
    """Pull runnable Python out of a generation (handles markdown fences + prose)."""
    if "</think>" in text:  # keep only the post-thinking answer region
        text = text.split("</think>", 1)[1]
    blocks = _FENCE.findall(text)
    if blocks:
        # prefer the longest fenced block (the actual solution)
        return max(blocks, key=len).strip()
    # no fence: drop obvious prose lines, keep from the first def/import/class
    lines = text.splitlines()
    start = next((i for i, ln in enumerate(lines)
                  if re.match(r"\s*(def |class |import |from )", ln)), None)
    if start is None:
        return text.strip()
    return "\n".join(lines[start:]).strip()


def _is_parseable(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Sandboxed execution verifier
# ---------------------------------------------------------------------------
def _worker(code: str, imports: Sequence[str], tests: Sequence[str], q):
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (5, 6))
        resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
    except Exception:
        pass
    g: dict = {"__name__": "__candidate__"}
    try:
        for imp in imports:
            exec(imp, g)
        exec(code, g)
        for t in tests:
            exec(t, g)
        q.put(("pass", ""))
    except Exception as e:  # noqa: BLE001 - any failure is a fail
        q.put(("fail", f"{type(e).__name__}: {e}"))


def _verify_once(code: str, task: Task, timeout: float) -> tuple[bool, str]:
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_worker, args=(code, task.test_imports, task.test_list, q))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        return False, "timeout"
    try:
        status, detail = q.get_nowait()
    except Exception:
        return False, "crash"
    return status == "pass", detail


def verify(code: str, task: Task, timeout: float = 10.0) -> tuple[bool, str]:
    """Return (passed, detail). Runs candidate + full test_list in a subprocess.

    Retries once on timeout: a borderline-slow candidate can time out under CPU load
    (e.g. while the GPU job competes) and pass on a quieter retry — this removes most
    of the verification jitter without masking real infinite loops.
    """
    if not code or not _is_parseable(code):
        return False, "unparseable"
    ok, detail = _verify_once(code, task, timeout)
    if not ok and detail == "timeout":
        ok, detail = _verify_once(code, task, timeout * 1.5)
    return ok, detail


def verify_visible(code: str, task: Task, timeout: float = 8.0) -> tuple[bool, str]:
    """Deployable selector signal: pass the FIRST assert only (visible test)."""
    if not code or not _is_parseable(code):
        return False, "unparseable"
    if not task.test_list:
        return False, "no-tests"
    one = Task(task.task_id, task.prompt, task.test_list[:1], task.test_imports)
    return verify(code, one, timeout)


if __name__ == "__main__":  # quick self-test
    tasks = load_mbpp(limit=3)
    print(f"loaded {len(tasks)} tasks; first entry_point={tasks[0].entry_point}")
    ref = "def remove_Occ(s,ch):\n  return s.replace(ch,'',1)[::-1].replace(ch,'',1)[::-1]"
    ok, d = verify(ref, tasks[0])
    print("ref verify:", ok, d)
    bad, d2 = verify("def remove_Occ(s,ch):\n  return s", tasks[0])
    print("bad verify:", bad, d2)
