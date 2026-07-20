"""CodingEnv — a real-repo agentic coding environment for TRL 1.8 GRPO environment_factory.

TRL contract (verified in trl/trainer/grpo_trainer.py):
- environment_factory() -> instance (no args); instances are POOLED/REUSED across batches.
- reset(**kwargs) receives the FULL dataset row as kwargs; returns a string appended to the last
  user message (the task instructions + initial state). MUST fully re-initialize (pooled reuse).
- Every non-underscore method except reset/get_reward becomes a TOOL (needs type hints + docstring
  for the tool schema). TRL drives the multi-turn tool-calling loop and owns the token logprobs.
- get_reward() -> float, called once per completed rollout = the terminal reward.

Tools mirror pi-coding-agent's interface (read/write/list/bash) so training transfers to pi deploy.
Reward = fraction of the task's tests that pass (dense-ish signal for sparse binary success), with
a strict full-pass=1.0. Runs in the repo's own venv (isolated via cwd-insertion of the local copy).
"""
import functools, inspect, os, re, shutil, subprocess, tempfile
from pathlib import Path


def _tolerant(fn):
    """Drop tool arguments the tool does not declare, instead of raising TypeError.

    Measured: the policy emits a spurious `<parameter=content>None</parameter>` on EVERY tool call,
    so `read_file(path=..., content=None)` raised TypeError and 86% of all tool calls failed
    (tools/failure_frequency 0.8621). The agent then burned every iteration on rejected calls and
    never reached write_file -> every rollout scored exactly 0.0 -> reward_std 0 -> no GRPO gradient.
    That looked like a task-difficulty problem for a long time; it was a harness-strictness problem.

    Real agent harnesses (pi-coding-agent included) tolerate extra arguments, so tolerating them here
    also makes the training env behave like the deployment target. Genuine errors still surface:
    a MISSING required argument (e.g. write_file without content) still raises.

    functools.wraps sets __wrapped__, so inspect.signature() follows through to the ORIGINAL
    signature -- transformers still builds the correct tool JSON schema from it, which is why this
    does not repeat the earlier `*a, **k` mistake that produced DocstringParsingException.
    """
    params = set(inspect.signature(fn).parameters)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        dropped = [k for k in kwargs if k not in params]
        for k in dropped:
            kwargs.pop(k)
        out = fn(self, *args, **kwargs)
        if dropped and isinstance(out, str):
            out += f"\n[note: ignored unsupported argument(s): {', '.join(sorted(dropped))}]"
        return out

    return wrapper

RUN_TIMEOUT = 30
TEST_TIMEOUT = 120
MAX_OUT = 8000   # keep tool results REALISTIC (pi serves full files); memory is handled by the
                 # logprob-forward micro-batch (per_device_train_batch_size=1), NOT by truncating
                 # context — truncation would distort the task and not transfer to deployment.


def _stub_function(repo_root: Path, rel_file: str, func_name: str):
    import ast
    p = repo_root / rel_file
    src = p.read_text()
    tree = ast.parse(src)
    target = next((n for n in ast.walk(tree)
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == func_name), None)
    if target is None:
        return False
    lines = src.split("\n")
    body = target.body
    first = body[0]
    keep_doc = (isinstance(first, ast.Expr) and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(first.value.value, str))
    start = (body[1].lineno - 1) if (keep_doc and len(body) > 1) else (first.lineno - 1)
    end = body[-1].end_lineno
    indent = " " * target.body[0].col_offset
    lines[start:end] = [indent + "raise NotImplementedError  # TODO: implement"]
    p.write_text("\n".join(lines))
    return True


class CodingEnv:
    """Stub-a-function real-repo task. reset() kwargs: repo_dir, rel_file, func_name, python, test_cmd."""

    def __init__(self):
        self.project = None
        self.env = None
        self.test_cmd = None
        self._task = None

    # -- lifecycle -------------------------------------------------------------
    def reset(self, **kwargs) -> str:
        self._cleanup()
        self._task = kwargs
        repo_dir = kwargs["repo_dir"]
        rel_file = kwargs["rel_file"]
        func_name = kwargs["func_name"]
        python = kwargs["python"]
        self.test_cmd = kwargs.get("test_cmd", "python -m pytest -q")
        self.project = Path(tempfile.mkdtemp(prefix="grpoenv_"))
        shutil.copytree(repo_dir, self.project, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".git", ".venv*", "__pycache__", "*.pyc"))
        _stub_function(self.project, rel_file, func_name)
        venv_bin = str(Path(python).parent)
        self.env = {**os.environ, "PATH": venv_bin + os.pathsep + os.environ.get("PATH", ""),
                    "VIRTUAL_ENV": str(Path(python).parent.parent), "PYTHONDONTWRITEBYTECODE": "1"}
        return (f"You are in the project root of a Python repo. The function `{func_name}` in `{rel_file}` "
                f"has had its body replaced with `raise NotImplementedError`. Implement it so the tests pass. "
                f"Work methodically and DO NOT give up early:\n"
                f"1. Read `{rel_file}` and the function's tests (under the tests directory) to learn the exact "
                f"required behavior.\n"
                f"2. You MUST edit `{rel_file}` with write_file to implement `{func_name}` — never finish "
                f"without having written your implementation.\n"
                f"3. Run `{self.test_cmd}` with run_bash to check. If it fails, read the error, fix, and repeat.\n"
                f"4. Only stop once the tests actually pass. Only edit `{rel_file}`.")

    def get_reward(self) -> float:
        if self.project is None:
            return 0.0
        # engagement gate: did the agent actually implement the function (remove the stub)?
        try:
            edited = "raise NotImplementedError  # TODO: implement" not in (self.project / self._task["rel_file"]).read_text()
        except Exception:
            edited = False
        if not edited:
            return 0.0  # never wrote -> 0 (forces engagement, creates variance vs writers)
        r = self._run(self.test_cmd, TEST_TIMEOUT)
        out = r["out"]
        if r["code"] == 0:
            return 1.0
        m_pass = re.search(r"(\d+) passed", out)
        m_fail = re.search(r"(\d+) failed", out)
        p = int(m_pass.group(1)) if m_pass else 0
        f = int(m_fail.group(1)) if m_fail else 0
        if p + f == 0:
            return 0.05  # edited but crashes / collection error -> tiny credit for attempting
        return 0.1 + 0.5 * (p / (p + f))  # edited + partial pass: 0.1..0.6; full pass=1.0

    # -- tools (become the model's tool interface) -----------------------------
    @_tolerant
    def read_file(self, path: str) -> str:
        """Read a UTF-8 text file in the project and return its contents.

        Args:
            path: Project-relative path to the file to read.
        """
        f = self._safe(path)
        return f.read_text(errors="replace")[:MAX_OUT] if f and f.exists() else "(no such file)"

    @_tolerant
    def write_file(self, path: str, content: str) -> str:
        """Create or overwrite a UTF-8 text file in the project with the full new content.

        Args:
            path: Project-relative path to write.
            content: Full file contents to write.
        """
        f = self._safe(path)
        if f is None:
            return "(invalid path)"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(content)
        return f"wrote {path} ({len(content)} bytes)"

    @_tolerant
    def list_dir(self, path: str = ".") -> str:
        """List the entries of a directory in the project.

        Args:
            path: Project-relative directory path (default '.').
        """
        d = self._safe(path)
        if d is None or not d.exists():
            return "(no such directory)"
        return "\n".join(sorted(x.name + ("/" if x.is_dir() else "") for x in d.iterdir())) or "(empty)"

    @_tolerant
    def run_bash(self, command: str) -> str:
        """Run a shell command in the project root and return combined stdout/stderr and exit code.

        Args:
            command: The shell command to run.
        """
        r = self._run(command, RUN_TIMEOUT)
        return f"[exit {r['code']}]\n{r['out']}"

    # -- internals -------------------------------------------------------------
    def _safe(self, path):
        try:
            p = (self.project / (path or ".")).resolve()
            if not str(p).startswith(str(self.project.resolve())):
                return None
            return p
        except Exception:
            return None

    def _run(self, command, timeout):
        try:
            r = subprocess.run(command, shell=True, cwd=str(self.project), text=True, env=self.env,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
            return {"code": r.returncode, "out": (r.stdout or "")[:MAX_OUT]}
        except subprocess.TimeoutExpired:
            return {"code": 124, "out": f"[timeout after {timeout}s]"}
        except Exception as e:
            return {"code": 1, "out": f"[error: {type(e).__name__}: {e}]"}

    def _cleanup(self):
        if self.project is not None:
            shutil.rmtree(self.project, ignore_errors=True)
            self.project = None


class SynthEnv(CodingEnv):
    """Self-contained-scenario variant for TRL environment_factory.

    The difficulty calibration found the RLVR-usable band (per-task pass rate 0.25-0.75 for the
    warm-started policy) lives in the SELF-CONTAINED scenarios, not the real-repo stub tasks (which
    are uniformly 0.00). This env seeds a scenario's files into a temp project instead of stubbing a
    function in a repo checkout. Tools + reward semantics are inherited from CodingEnv, so the model
    sees the IDENTICAL pi-mirroring tool interface it was warm-started on.

    reset(**row) expects: scenario_id, files (dict path->content), check (shell cmd), prompt.
    """

    def reset(self, **kwargs) -> str:
        self._cleanup()
        self._task = kwargs
        import json as _json
        files = kwargs["files"]
        if isinstance(files, str):           # datasets may serialize the dict
            files = _json.loads(files)
        self.test_cmd = kwargs.get("check", "python3 test_solution.py")
        self.project = Path(tempfile.mkdtemp(prefix="grposynth_"))
        for rel, content in files.items():
            f = self._safe(rel); f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(content, encoding="utf-8")
        self.env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        # NOTE: TRL passes the WHOLE dataset row to reset(**row), so kwargs["prompt"] is the chat
        # messages list, not the task text. The task text is carried as `prompt_text`.
        return kwargs.get("prompt_text") or (
            "Implement the solution in `solution.py` so that the tests pass. Read `test_solution.py` "
            f"first, then implement and run `{self.test_cmd}` to verify. Iterate until it passes.")

    def get_reward(self) -> float:
        """Graded progress reward (dense enough that GRPO groups rarely tie).

        A near-binary reward makes both rollouts in a small group score 0.0 -> zero advantage ->
        the step is wasted (observed: only 1 of 7 steps had reward_std > 0). Ladder:
          0.00 no edit | 0.15 edited but crashes | 0.15 + 0.6*(tests_passed/total) | 1.00 all pass
        tests_passed is inferred from which test function the runner died in (they are named t0..tN).
        """
        if self.project is None:
            return 0.0
        sol = self.project / "solution.py"
        test = self.project / "test_solution.py"
        try:
            edited = "NotImplementedError" not in sol.read_text()
        except Exception:
            return 0.0
        if not edited:
            return self._log_reward(0.0, edited)
        r = self._run(self.test_cmd, TEST_TIMEOUT)
        if r["code"] == 0:
            return self._log_reward(1.0, edited, r)
        try:
            total = max(1, test.read_text().count("def t"))
        except Exception:
            total = 1
        import re as _re
        idx = _re.findall(r"in t(\d+)", r["out"])
        passed = max(int(x) for x in idx) if idx else 0   # died in tN => t0..t(N-1) passed
        return self._log_reward(round(0.15 + 0.6 * min(1.0, passed / total), 4), edited, r)

    def _log_reward(self, reward, edited, r=None):
        """Optional per-rollout reward trace (RLVR_REWARD_LOG=path).

        GRPO only ever logs the BATCH MEAN, which cannot distinguish "policy never engaged" from
        "policy engaged and failed" -- and that difference decides whether the blocker is the task
        band or the harness. One JSON line per scored rollout makes the distinction observable.
        """
        import os as _os
        path = _os.environ.get("RLVR_REWARD_LOG")
        if path:
            import json as _j
            try:
                with open(path, "a") as fh:
                    fh.write(_j.dumps({"task": (self._task or {}).get("scenario_id"),
                                       "reward": reward, "edited": bool(edited),
                                       "code": (r or {}).get("code"),
                                       "out": ((r or {}).get("out") or "")[:200]}) + "\n")
            except Exception:
                pass
        return reward
