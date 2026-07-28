"""Checkout copying, test execution, and per-test scoring — shared by task generation AND evaluation.

WHY THIS IS ONE MODULE: the program's three worst measurement failures were all harness mismatches
(real-repo ~0.00 vs pi 0.70; synthetic 0.486 vs pi 0.810; meanR-0.00 from a dict-default bug). If task
generation validates a task in one environment and evaluation scores it in a slightly different one,
the instrument lies. Both paths call `prepare_copy()` / `run_tests()` / `score_task()` here, and
nothing else may build a test environment or compute a reward.

--------------------------------------------------------------------------------------------------
THREE FAILURE MODES THIS FILE EXISTS TO PREVENT
--------------------------------------------------------------------------------------------------
1. THE EDITABLE-INSTALL TRAP (free rewards).
   `uv pip install -e .` + copy-per-episode is silently broken for `src/` layouts: an editable install
   resolves `import semver` through a .pth pointing at the ORIGINAL checkout, so pytest in the copy
   imports the ORIGINAL, unstubbed source. The stub has no effect, every test passes, and every
   episode scores a free 1.0 — an instrument reporting a perfect agent that never wrote a line.
   Guards: (a) install editable once to resolve deps, then UNINSTALL the distribution; (b) PYTHONPATH
   points at the copy's own roots; (c) a task is admitted only if stubbing it in a copy actually
   BREAKS tests, so an import leak yields zero tasks instead of free reward.

2. DRIVER-SIDE MEMORY BALLOON (killed the WSL VM three times on 2026-07-24/25).
   `subprocess.run(..., stdout=PIPE)` accumulates a child's ENTIRE output in the PARENT's memory. The
   capped child could not balloon itself, but nothing bounded how much it could PRINT, so the driver
   grew to ~15.7 GB anon-RSS, saturated the 16 GB VM cap plus 16 GB swap, and the kernel's global OOM
   condemned `/init.scope` — which under WSL2 is all of WSL. Fix: spool child output to DISK, stream
   it to extract per-test statuses, keep only a tail in memory, and kill any child that spews past
   MAX_OUT_BYTES. See docs/wsl_stability.md.

3. UNBOUNDED THIRD-PARTY ALLOCATION.
   We run test suites we do not control. `ulimit -v` (inside the shell string, NOT
   subprocess(preexec_fn=…), which is documented-unsafe in threaded drivers) turns a runaway suite
   into one rejected candidate instead of a dead box. Never apply this to torch/vLLM processes: CUDA
   reserves enormous virtual address space; bound those with systemd-run MemoryMax and bf16 loads.
"""
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

IGNORE = shutil.ignore_patterns(".git", ".venv*", "__pycache__", "*.pyc", ".pytest_cache",
                                ".tox", ".mypy_cache", ".ruff_cache", "*.egg-info")

# -v (not -q) so every test reports its own status: the instrument scores per-test sets, not totals.
# -rA is NOT redundant with -v: when a repo's own addopts enable pytest-xdist (`-n auto`), workers
# suppress the per-test progress lines and -v yields nothing parseable -- wcwidth silently baselined
# at "0 passed" until -rA was added. The -rA short-summary section lists every test's outcome in all
# execution modes, so it is the authoritative source and -v is the fallback.
# --tb=line keeps failure output to one line per failure, which is the difference between a 200 KB log
# and a 200 MB log on a suite where one stub breaks a hundred tests.
TEST_CMD = "python -m pytest -v -rA --tb=line -p no:cacheprovider"

MEM_CAP_KB = 6 * 1024 * 1024          # address space per test process (see failure mode 3)
MIN_AVAIL_MB = 3072                   # refuse to start new work below this MemAvailable
MAX_OUT_BYTES = 64 * 1024 * 1024      # kill a child that prints more than this (see failure mode 2)
TAIL_BYTES = 256 * 1024               # tail retained in memory for diagnosis

STATUS_RE = re.compile(
    r"^(?P<nid>[^\s:]+(?:::[^\s]+)?)\s+(?P<st>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS|SUBFAIL|SUBPASS)\b")
SUMMARY_RE = re.compile(
    r"^(?P<st>PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(?P<nid>[^\s:]+(?:::[^\s]+)?)")


# --------------------------------------------------------------------------------------------------
# memory / process hygiene
# --------------------------------------------------------------------------------------------------
def mem_available_mb() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) // 1024
    except Exception:
        pass
    return 1 << 30                    # unknown -> do not block


def wait_for_memory(min_avail_mb=MIN_AVAIL_MB, timeout=300, poll=5.0) -> bool:
    """Block until MemAvailable recovers. False if it never does inside `timeout`."""
    deadline = time.monotonic() + timeout
    while mem_available_mb() < min_avail_mb:
        if time.monotonic() > deadline:
            return False
        time.sleep(poll)
    return True


def capped(cmd: str, mem_cap_kb: int = MEM_CAP_KB) -> str:
    """Wrap a shell command with a hard address-space cap and no core dumps."""
    return f"ulimit -v {mem_cap_kb} 2>/dev/null; ulimit -c 0 2>/dev/null; {cmd}"


def run_spooled(cmd, cwd=None, env=None, timeout=300, spool_path=None,
                max_out_bytes=MAX_OUT_BYTES, tail_bytes=TAIL_BYTES):
    """Run a shell command, spooling output to DISK. Returns (code, tail, total_bytes, spool_path).

    If `spool_path` is given the file is left for the caller to stream and delete; otherwise it is
    removed before returning. Never holds more than `tail_bytes` in memory.
    """
    keep = spool_path is not None
    if not keep:
        fd, spool_path = tempfile.mkstemp(prefix="spool_", suffix=".log")
        os.close(fd)
    spool_path = str(spool_path)
    killed_for = None
    try:
        with open(spool_path, "w+b") as fh:
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd and str(cwd), env=env,
                                    stdout=fh, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
            deadline = time.monotonic() + timeout
            while proc.poll() is None:
                if time.monotonic() > deadline:
                    killed_for = f"[timeout after {timeout}s]"
                    break
                try:
                    if os.path.getsize(spool_path) > max_out_bytes:
                        killed_for = f"[killed: output exceeded {max_out_bytes // (1 << 20)} MB]"
                        break
                except OSError:
                    pass
                time.sleep(0.25)
            if killed_for:
                # start_new_session gave the child its own group; kill the GROUP or pytest's
                # subprocesses (and anything the suite spawned) keep running and keep printing.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    proc.kill()
            code = proc.wait(timeout=60)
        total = os.path.getsize(spool_path)
        with open(spool_path, "rb") as fh:
            fh.seek(max(0, total - tail_bytes))
            tail = fh.read().decode("utf-8", errors="replace")
        if killed_for:
            tail += "\n" + killed_for
            code = 124 if "timeout" in killed_for else 137
        return code, tail, total, spool_path
    finally:
        if not keep:
            try:
                os.unlink(spool_path)
            except OSError:
                pass


# --------------------------------------------------------------------------------------------------
# environment construction
# --------------------------------------------------------------------------------------------------
def venv_python(repo_dir) -> Path:
    return Path(repo_dir) / ".venv-test" / "bin" / "python"


def build_env(copy_dir, repo_dir) -> dict:
    """Environment for running the suite inside `copy_dir` using `repo_dir`'s test venv.

    PYTHONPATH forces imports to resolve inside the COPY (failure mode 1).
    """
    copy_dir, repo_dir = Path(copy_dir), Path(repo_dir)
    vbin = venv_python(repo_dir).parent
    roots = []
    if (copy_dir / "src").is_dir():
        roots.append(str(copy_dir / "src"))
    roots.append(str(copy_dir))
    return {**os.environ,
            "PATH": str(vbin) + os.pathsep + os.environ.get("PATH", ""),
            "VIRTUAL_ENV": str(vbin.parent),
            "PYTHONPATH": os.pathsep.join(roots),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0", "TZ": "UTC", "NO_COLOR": "1",
            "COLUMNS": "80", "LC_ALL": "C.UTF-8", "LANG": "C.UTF-8"}


def prepare_copy(repo_dir, prefix="episode_") -> Path:
    """Copy a checkout to a scratch dir (excluding .git/.venv/caches). Caller must clean up."""
    dst = Path(tempfile.mkdtemp(prefix=prefix))
    shutil.copytree(repo_dir, dst, dirs_exist_ok=True, ignore=IGNORE)
    return dst


# --------------------------------------------------------------------------------------------------
# execution + per-test status extraction
# --------------------------------------------------------------------------------------------------
def _statuses_from_file(path):
    """Stream a spooled pytest log and return {node_id: status}.

    Streaming (not slurping) is deliberate: a suite with a hundred broken tests can produce a log far
    larger than we ever want resident, and only the bounded status map is needed. Both `-v` progress
    lines and `-rA`-style summary lines are recognised, so a repo's own addopts cannot break parsing.
    """
    statuses = {}
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                m = STATUS_RE.match(line) or SUMMARY_RE.match(line)
                if not m:
                    continue
                nid, st = m.group("nid"), m.group("st")
                if "::" not in nid and not nid.endswith(".py"):
                    continue
                prev = statuses.get(nid)
                # A node can report twice (e.g. subtests, or setup ERROR then FAILED). Failure wins:
                # scoring must never count a test as passing because a later line said PASSED.
                if prev in ("FAILED", "ERROR") or (prev and st in ("SUBPASS",)):
                    continue
                statuses[nid] = st
    except OSError:
        pass
    return statuses


def run_tests(copy_dir, repo_dir, timeout=300, cmd=TEST_CMD, mem_cap_kb=MEM_CAP_KB) -> dict:
    """Run the suite inside a prepared copy, under a memory cap, with output spooled to disk.

    Returns {code, statuses, out (tail), out_bytes, counts…}. Never raises.
    """
    fd, spool = tempfile.mkstemp(prefix="pytest_", suffix=".log")
    os.close(fd)
    try:
        code, tail, total, _ = run_spooled(capped(cmd, mem_cap_kb), copy_dir,
                                           build_env(copy_dir, repo_dir), timeout, spool_path=spool)
        statuses = _statuses_from_file(spool)
    except Exception as exc:                                   # pragma: no cover - defensive
        return {"code": 1, "statuses": {}, "out": f"[error: {type(exc).__name__}: {exc}]",
                "out_bytes": 0, "passed": 0, "failed": 0, "errors": 0, "skipped": 0}
    finally:
        try:
            os.unlink(spool)
        except OSError:
            pass
    counts = {"passed": sum(1 for v in statuses.values() if v in ("PASSED", "XPASS", "SUBPASS")),
              "failed": sum(1 for v in statuses.values() if v in ("FAILED", "SUBFAIL")),
              "errors": sum(1 for v in statuses.values() if v == "ERROR"),
              "skipped": sum(1 for v in statuses.values() if v in ("SKIPPED", "XFAIL"))}
    return {"code": code, "statuses": statuses, "out": tail, "out_bytes": total, **counts}


def passing_ids(statuses) -> set:
    return {k for k, v in statuses.items() if v in ("PASSED", "XPASS", "SUBPASS")}


# --------------------------------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------------------------------
def score_task(statuses, fail_to_pass, pass_to_pass) -> dict:
    """Reward for one episode, from per-test sets (the SWE-bench-style protocol).

    WHY NOT "the whole suite must be green": requiring a globally green suite dropped 11 of 24
    candidate repos over one or two pre-existing environment-dependent failures (a missing package
    metadata check, a test that needs >6 GB). Those tests fail identically before and after the
    agent's edit, so they carry no information about the agent — but they would have silently capped
    every episode's reward at partial credit. Per-test sets exclude them by construction: a test that
    fails in the untouched baseline is in neither set.

    fail_to_pass: tests the stub BREAKS. The agent must make all of them pass — this is the task.
    pass_to_pass: tests that pass in the baseline and are unaffected by the stub. They guard against
                  "fix the test instead of the code": deleting or breaking them cannot raise reward.

    reward 1.0   all fail_to_pass pass AND no pass_to_pass regression
           0.1..0.6  partial credit on fail_to_pass, HALVED if anything regressed
           0.05  edited but the suite could not run (crash/collection error)
           0.0   handled by the caller's engagement gate (stub still present)
    """
    ftp, ptp = set(fail_to_pass), set(pass_to_pass)
    ok = passing_ids(statuses)
    if not statuses:
        return {"reward": 0.05, "ftp_passed": 0, "ftp_total": len(ftp), "regressed": [],
                "solved": False, "unrunnable": True}
    ftp_ok = len(ftp & ok)
    regressed = sorted(ptp - ok)
    solved = (ftp_ok == len(ftp) and not regressed)
    if solved:
        reward = 1.0
    else:
        frac = ftp_ok / max(1, len(ftp))
        reward = 0.1 + 0.5 * frac
        if regressed:
            reward *= 0.5
        reward = round(reward, 4)
    return {"reward": reward, "ftp_passed": ftp_ok, "ftp_total": len(ftp),
            "regressed": regressed[:20], "n_regressed": len(regressed),
            "solved": bool(solved), "unrunnable": False}
