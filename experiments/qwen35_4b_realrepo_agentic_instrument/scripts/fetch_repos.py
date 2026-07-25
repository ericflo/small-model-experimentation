"""Clone each candidate repo, build its test venv, and record its BASELINE per-test result set.

A repo becomes instrument substrate when its suite RUNS and yields enough passing tests — not when
the suite is globally green. Requiring global green dropped 11 of 24 repos over one or two
environment-dependent failures (missing package metadata, a test wanting >6 GB), which is pure
instrument loss: those tests fail identically before and after an agent's edit, so per-test scoring
(env_util.score_task) excludes them by construction.

What is NEVER done: patching a repo's tests or config to make it pass. The property that made claim
C64 credible is that the verifier is "the repository's OWN pytest"; a suite we repaired is no longer
that. Dependency INSTALLATION is fair game (it is environment, not code); editing test code is not.

Writes:
  data/repo_manifest.json   — per repo: resolved SHA, install strategy, dep repairs, counts, verdict
  data/repo_baselines.json  — per repo: the exact node ids that PASS / FAIL in an untouched copy
Together these are the reproduction path for this experiment (standalone-experiment directive).
"""
import argparse
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[2]
STORE = ROOT / "large_artifacts" / "_taskrepos"
DATA = HERE.parent / "data"

import env_util  # noqa: E402
from repos import REPOS  # noqa: E402

UV = shutil.which("uv") or "/home/ericflo/.local/bin/uv"
MIN_BASELINE_PASS = 10        # a suite with fewer passing tests cannot support meaningful scoring

# Missing-import -> distribution, for the dependency-repair loop (only where the names differ).
DIST_ALIAS = {"yaml": "pyyaml", "attr": "attrs", "dateutil": "python-dateutil",
              "pkg_resources": "setuptools", "pytest_cov": "pytest-cov"}

# A repo's own pytest config (addopts) often requires plugins it does not declare as test deps.
# pytest then exits 4 with "unrecognized arguments", which is not a missing-module error and needs
# its own repair path. Honouring the repo's config matters: stripping its addopts would mean scoring
# against a suite the repo does not itself run.
PLUGIN_FOR_FLAG = {"--cov": "pytest-cov", "--no-cov": "pytest-cov", "--timeout": "pytest-timeout",
                   "--numprocesses": "pytest-xdist", "-n": "pytest-xdist",
                   "--benchmark": "pytest-benchmark", "--forked": "pytest-forked",
                   "--asyncio-mode": "pytest-asyncio", "--mypy": "pytest-mypy",
                   "--flake8": "pytest-flake8", "--randomly": "pytest-randomly",
                   "--black": "pytest-black", "--freeze": "pytest-freezegun",
                   "--regtest": "pytest-regtest", "--subtests": "pytest-subtests",
                   "--snapshot": "syrupy", "--hypothesis": "hypothesis"}


def sh(cmd, cwd=None, timeout=900, env=None):
    """Run a helper command with output SPOOLED TO DISK, never into this process's memory.

    `subprocess.run(stdout=PIPE)` is what ballooned this driver to ~15.7 GB and killed the WSL VM
    three times (docs/wsl_stability.md). No command in this pipeline may buffer output in RAM.
    """
    try:
        code, out, _total, _p = env_util.run_spooled(cmd, cwd, env, timeout)
        return code, out
    except Exception as exc:                                   # pragma: no cover - defensive
        return 1, f"[error: {type(exc).__name__}: {exc}]"


def dist_name(repo_dir: Path):
    """Best-effort distribution name, needed to UNINSTALL the editable link (see env_util)."""
    pj = repo_dir / "pyproject.toml"
    if pj.exists():
        try:
            import tomllib
            data = tomllib.loads(pj.read_text())
            for key in (("project", "name"), ("tool", "poetry", "name")):
                cur = data
                for k in key:
                    cur = cur.get(k, {}) if isinstance(cur, dict) else {}
                if isinstance(cur, str):
                    return cur
        except Exception:
            pass
    cfg = repo_dir / "setup.cfg"
    if cfg.exists():
        import configparser
        cp = configparser.ConfigParser()
        try:
            cp.read(cfg)
            if cp.has_option("metadata", "name"):
                return cp.get("metadata", "name")
        except Exception:
            pass
    return repo_dir.name


def missing_dists(out: str):
    """Distributions implied by missing imports and by pytest 'unrecognized arguments'."""
    import re
    dists = set()
    for m in re.finditer(r"No module named '?([A-Za-z0-9_.]+)'?", out or ""):
        mod = m.group(1).split(".")[0]
        dists.add(DIST_ALIAS.get(mod, mod))
    m = re.search(r"unrecognized arguments:(.+)", out or "")
    if m:
        for tok in m.group(1).split():
            flag = tok.split("=")[0].strip()
            dist = PLUGIN_FOR_FLAG.get(flag)
            if dist is None:
                for known, d in PLUGIN_FOR_FLAG.items():
                    if d and flag.startswith(known):
                        dist = d
                        break
            if dist:
                dists.add(dist)
    return sorted(dists)


def setup_repo(spec, force=False, suite_timeout=600, max_repair=3):
    name, url = spec["name"], spec["url"]
    rec = {"name": name, "url": url, "group": spec["group"], "src": spec["src"],
           "installed_extra": None, "repairs": [], "verdict": "pending"}
    repo = STORE / name
    t0 = time.time()
    if not env_util.wait_for_memory():
        rec.update(verdict="skipped_low_memory", mem_available_mb=env_util.mem_available_mb())
        return rec, None

    # ---- clone -------------------------------------------------------------------------------
    if force and repo.exists():
        shutil.rmtree(repo, ignore_errors=True)
    if not (repo / ".git").exists():
        if repo.exists():
            rec["preexisting_no_git"] = True
        else:
            code, out = sh(f"git clone --depth 1 {url} {repo}", timeout=600)
            if code != 0:
                rec.update(verdict="clone_failed", detail=out[-400:])
                return rec, None
    code, out = sh("git rev-parse HEAD", cwd=repo)
    rec["sha"] = out.strip() if code == 0 else None
    code, out = sh("git log -1 --format=%cI", cwd=repo)
    rec["commit_date"] = out.strip() if code == 0 else None

    # ---- venv + dependency resolution --------------------------------------------------------
    vpy = env_util.venv_python(repo)
    if not vpy.exists():
        code, out = sh(f"{UV} venv {repo}/.venv-test", timeout=300)
        if code != 0:
            rec.update(verdict="venv_failed", detail=out[-400:])
            return rec, None
    pipx = f"{UV} pip install --python {vpy}"
    for extra in ('".[test]"', '".[tests]"', '".[dev]"', '".[testing]"', "."):
        code, out = sh(f"{pipx} -e {extra}", cwd=repo, timeout=900)
        if code == 0:
            rec["installed_extra"] = extra
            break
    else:
        rec.update(verdict="install_failed", detail=out[-400:])
        return rec, None
    sh(f"{pipx} pytest", cwd=repo, timeout=600)
    # ...then REMOVE the editable link so the only importable copy is the episode's own checkout.
    dist = dist_name(repo)
    rec["dist_name"] = dist
    code, _ = sh(f"{UV} pip uninstall --python {vpy} {dist}", cwd=repo, timeout=300)
    rec["uninstalled_editable"] = (code == 0)

    # ---- baseline capture, with a bounded dependency-repair loop ------------------------------
    for attempt in range(max_repair):
        copy = env_util.prepare_copy(repo, prefix=f"baseline_{name}_")
        try:
            res = env_util.run_tests(copy, repo, timeout=suite_timeout)
        finally:
            shutil.rmtree(copy, ignore_errors=True)
        rec["suite"] = {k: res[k] for k in ("code", "passed", "failed", "errors", "skipped")}
        rec["out_bytes"] = res["out_bytes"]
        if res["passed"] >= MIN_BASELINE_PASS:
            baseline = {"pass": sorted(env_util.passing_ids(res["statuses"])),
                        "fail": sorted(k for k, v in res["statuses"].items()
                                       if v in ("FAILED", "ERROR", "SUBFAIL"))}
            rec.update(verdict="usable", suite_secs=round(time.time() - t0, 1),
                       n_baseline_pass=len(baseline["pass"]),
                       n_baseline_fail=len(baseline["fail"]))
            return rec, baseline
        dists = missing_dists(res["out"])
        if not dists or attempt == max_repair - 1:
            rec.update(verdict="suite_unusable", detail=res["out"][-1500:],
                       suite_secs=round(time.time() - t0, 1))
            return rec, None
        rec["repairs"].append(dists)
        sh(f"{pipx} {' '.join(dists)}", cwd=repo, timeout=600)
    return rec, None


def main():
    ap = argparse.ArgumentParser()
    # Default 2, not 4: peak memory demand multiplies with workers, and the 2026-07-24 VM death ran
    # four third-party suites at once. Per-process caps + spooled output are the real fixes.
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", default=None, help="comma-separated repo names")
    ap.add_argument("--force", action="store_true", help="re-clone from scratch")
    ap.add_argument("--suite-timeout", type=int, default=600)
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-attempt repos previously recorded unusable")
    ap.add_argument("--rebaseline", action="store_true",
                    help="recapture EVERY baseline (keeps clones/venvs). Required after any change "
                         "to the test command: a baseline captured with a different command is not "
                         "comparable to an episode scored with the current one, and that mismatch is "
                         "precisely the class of bug that produced this program's false walls.")
    ap.add_argument("--out", default=str(DATA / "repo_manifest.json"))
    ap.add_argument("--baselines", default=str(DATA / "repo_baselines.json"))
    a = ap.parse_args()

    specs = REPOS
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        specs = [s for s in specs if s["name"] in want]
    STORE.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    # RESUME: a crash mid-fetch has already cost two full passes; never redo finished work.
    recs, keep = [], set()
    baselines = {}
    if Path(a.baselines).exists():
        try:
            baselines = json.loads(Path(a.baselines).read_text())
        except Exception:
            baselines = {}
    if a.rebaseline:
        baselines = {}
    if Path(a.out).exists() and not a.force and not a.rebaseline:
        try:
            prior = json.loads(Path(a.out).read_text()).get("repos", [])
        except Exception:
            prior = []
        for r in prior:
            usable = r.get("verdict") == "usable" and r["name"] in baselines
            if usable or (r.get("verdict") in ("clone_failed", "install_failed", "venv_failed")
                          and not a.retry_failed):
                recs.append(r)
                keep.add(r["name"])
        if keep:
            print(f"RESUME: skipping {len(keep)} already-recorded repos", flush=True)
    specs = [s for s in specs if s["name"] not in keep]
    print(f"fetching {len(specs)} repos into {STORE} | {a.jobs} workers | "
          f"MemAvailable {env_util.mem_available_mb()} MB", flush=True)

    def save():
        Path(a.out).write_text(json.dumps({"repos": recs}, indent=1))
        Path(a.baselines).write_text(json.dumps(baselines, indent=1))

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(setup_repo, s, a.force, a.suite_timeout): s["name"] for s in specs}
        for i, f in enumerate(as_completed(futs), 1):
            nm = futs[f]
            try:
                rec, baseline = f.result()
            except Exception as exc:
                rec, baseline = {"name": nm, "verdict": "exception",
                                 "detail": f"{type(exc).__name__}: {exc}"}, None
            recs.append(rec)
            if baseline:
                baselines[nm] = baseline
            s = rec.get("suite", {})
            print(f"  [{i}/{len(specs)}] {nm:20s} {rec['verdict']:16s} "
                  f"pass={s.get('passed','-')} fail={s.get('failed','-')} "
                  f"log={rec.get('out_bytes',0)/1e6:.1f}MB ({rec.get('suite_secs','?')}s)", flush=True)
            save()

    usable = [r for r in recs if r.get("verdict") == "usable"]
    print(f"\nUSABLE: {len(usable)}/{len(recs)} repos "
          f"(train={sum(1 for r in usable if r.get('group')=='train')}, "
          f"test={sum(1 for r in usable if r.get('group')=='test')})", flush=True)
    for r in recs:
        if r.get("verdict") != "usable":
            print(f"  DROPPED {r.get('name')}: {r.get('verdict')}", flush=True)
    biggest = sorted(recs, key=lambda r: -(r.get("out_bytes") or 0))[:3]
    print("  largest suite logs: " + ", ".join(
        f"{r.get('name')} {(r.get('out_bytes') or 0)/1e6:.1f}MB" for r in biggest), flush=True)
    save()
    print(f"saved -> {a.out} + {a.baselines}", flush=True)


if __name__ == "__main__":
    main()
