#!/usr/bin/env python3
"""Fail the build on measurement footguns this repo has ALREADY paid for.

WHY THIS EXISTS. Every rule below was written in prose first -- in AGENTS.md, docs/model_playbook.md or
docs/wsl_stability.md -- and then violated anyway, in some cases three times. Prose is a PULL mechanism:
it only fires if the agent happens to recall it at the moment it types a default value. A check is a
PUSH mechanism: it fires whether or not anyone remembers. The repo's own evidence is unambiguous --
everything enforced by `make check` (briefs, charts, dates, stray caches) has never regressed, while
prose rules about token budgets and process matching have regressed repeatedly.

Each rule cites the incident that motivated it, so a future reader can judge whether it still applies
rather than obeying it blindly (this repo has also been burned by stale advice whose rationale was
falsified: the vLLM `util 0.45` recommendation outlived the host-memory theory that justified it).

ESCAPE HATCH, deliberately explicit: put `# footgun-ok: <reason>` on the offending line. A justified
exception is fine; a silent one is not. The reason is required and must be non-empty.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCAN_DIRS = ["experiments", "scripts", "benchmarks"]
ALLOW = re.compile(r"#\s*footgun-ok:\s*\S+")

RULES = [
    dict(
        name="low-generation-cap",
        # 2026-07-26: max_new=512 on a reasoning arm -> mean 510 generated tokens, 95.5% never emitted
        # an answer, accuracy read 0.040 instead of ~0.235. Earlier: C45's 256 cap produced a false
        # 0.00. docs/model_playbook.md: "Budget the CoT generously... Do not transfer a budget."
        pattern=re.compile(r"max_new(?:_tokens)?\s*[=:]\s*(\d{1,4})\b"),
        test=lambda m: int(m.group(1)) < 2048,
        message=("generation cap {val} < 2048. A low cap has produced FALSE NEGATIVES three times in "
                 "this repo (C45 256-cap false 0.00; 512-cap 0.040-vs-0.235 on 2026-07-26). Use "
                 "stop_strings=[...] + tokenizer=tok so an episode ends when it COMMITS, keep the cap "
                 "generous (3072+ text / 8192 think), and record the parse rate. See "
                 "docs/model_playbook.md 'STOP-ON-COMMIT'."),
    ),
    dict(
        name="pipe-capture",
        # 8 WSL VM deaths: subprocess.run(stdout=PIPE) buffers a child's ENTIRE output in the PARENT.
        # One pi episode emits ~4MB of JSON events; a driver reached ~31GB of anonymous memory and the
        # kernel's global OOM condemned /init.scope, i.e. all of WSL. docs/wsl_stability.md.
        pattern=re.compile(r"stdout\s*=\s*subprocess\.PIPE|capture_output\s*=\s*True"),
        test=lambda m: True,
        message=("captures child output into THIS process's memory. That is the mechanism behind eight "
                 "WSL VM deaths (a driver reached ~31GB anon RSS; global OOM condemns /init.scope = all "
                 "of WSL). Spool to a file and read only a tail -- see "
                 "experiments/qwen35_4b_realrepo_agentic_instrument/scripts/env_util.py:run_spooled."),
    ),
    dict(
        name="unanchored-process-match",
        # docs/wsl_stability.md: an unanchored pgrep -f/pkill -f matches the agent's own shell wrapper.
        # It has killed the wrong process, and on 2026-07-25 it reported a vLLM server that was not
        # running, which would have silently produced an empty measurement run.
        pattern=re.compile(r"(?:pgrep|pkill)\s+(?:-[a-zA-Z]*\s+)*-f\s+[\"']?(?!\^)"),
        test=lambda m: True,
        message=("unanchored pgrep/pkill -f also matches the agent's own shell wrapper: it has killed "
                 "the wrong process and has falsely reported a server as UP (2026-07-25), which "
                 "silently empties a run. Anchor the pattern, e.g. pgrep -f \"^/home.*vllm serve\"."),
    ),
    dict(
        name="thinking-disabled",
        # Repo rule (memory 'think-is-the-default', AGENTS.md): measure WITH thinking by default. A
        # thinking-OFF harness bug made base HumanEval read 76.2% against a true 89.6% and invalidated
        # an entire line of coding experiments.
        pattern=re.compile(r"enable_thinking\s*=\s*False"),
        test=lambda m: True,
        message=("thinking disabled. The repo default is thinking-ON: a thinking-OFF harness made base "
                 "HumanEval read 76.2% vs a true 89.6% and voided a whole coding line. If no-think is "
                 "deliberate (e.g. matching a prior arm's protocol), say so with "
                 "'# footgun-ok: matches C59 forced-read protocol'."),
    ),
]


def scan_file(path: Path):
    hits = []
    if path.resolve() == Path(__file__).resolve():
        return hits          # this file DEFINES the patterns; it would always match itself
    try:
        lines = path.read_text(errors="replace").splitlines()
    except OSError:
        return hits
    for i, line in enumerate(lines, 1):
        if ALLOW.search(line) or line.strip().startswith("#"):
            continue         # annotated, or a comment (not executable)
        for rule in RULES:
            for m in rule["pattern"].finditer(line):
                if rule["test"](m):
                    val = m.group(1) if m.groups() else ""
                    hits.append((path, i, rule["name"], rule["message"].format(val=val), line.strip()))
                    break
    return hits


def changed_files():
    """Files changed in the working tree or the most recent commit.

    RATCHET, not retrofit. Scanning all of history yields 825 hits across 299 experiments -- real
    footguns, but gating on them would make `make check` unusable noise and the check would simply be
    disabled, which is how prose rules die too. New and changed code must be clean; history is
    grandfathered and can be cleaned opportunistically with --all.
    """
    import subprocess
    out = set()
    for cmd in (["git", "diff", "--name-only", "HEAD"],
                ["git", "diff", "--name-only", "--cached"],
                ["git", "show", "--name-only", "--pretty=format:", "HEAD"],
                ["git", "ls-files", "--others", "--exclude-standard"]):
        try:
            r = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,  # footgun-ok: tiny bounded git output, not a child we can flood
                               stderr=subprocess.DEVNULL, timeout=30)
            out |= {line.strip() for line in (r.stdout or "").splitlines() if line.strip()}
        except Exception:
            pass
    return [ROOT / f for f in sorted(out)]


def main():
    argv = [a for a in sys.argv[1:] if a != "--all"]
    scan_all = "--all" in sys.argv[1:]
    if argv:
        targets = [Path(a) for a in argv]
    elif scan_all:
        targets = [ROOT / d for d in SCAN_DIRS]
    else:
        targets = changed_files()
        if not targets:
            print("footgun check: no changed files to scan")
            return 0
    hits = []
    for t in targets:
        if t.is_file() and t.suffix in (".py", ".sh"):
            hits += scan_file(t)
        elif t.is_dir():
            for p in sorted(t.rglob("*")):
                if p.suffix in (".py", ".sh") and "__pycache__" not in p.parts:
                    hits += scan_file(p)
    if not hits:
        print(f"footgun check: clean ({len(RULES)} rules, "
              f"{'whole repo' if scan_all else 'changed files'})")
        return 0
    by_rule = {}
    for h in hits:
        by_rule.setdefault(h[2], []).append(h)
    print(f"footgun check: {len(hits)} hit(s) across {len(by_rule)} rule(s)\n")
    for name, group in sorted(by_rule.items()):
        print(f"[{name}] {len(group)} hit(s) -- {group[0][3]}")
        for path, ln, _, _, src in group[:12]:
            print(f"    {path.relative_to(ROOT)}:{ln}: {src[:100]}")
        if len(group) > 12:
            print(f"    ... and {len(group) - 12} more")
        print()
    print("Fix, or annotate the line with '# footgun-ok: <reason>' if the exception is justified.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
