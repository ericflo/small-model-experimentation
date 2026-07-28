#!/usr/bin/env python3
"""Export (task, spec, candidate implementation) pairs from logged pi trajectories.

STANDALONE DOCTRINE: this experiment must be reproducible from its own directory, so the pairs are
written into data/ as committed JSONL rather than referenced from the instrument's large_artifacts.
Each pair carries the VERBATIM stubbed region (`old_text`) alongside the spec, because reconstruction
works by exact text replacement: stub the function in a fresh checkout (which recreates old_text
byte-for-byte -- it is what the agent's own edit tool matched against), then replace old_text with the
candidate. Pairs whose old_text no longer matches after stubbing are dropped with a recorded reason
rather than silently patched.

Candidates come from BOTH measured arms (base and warm-start): different policies produce decorrelated
implementations, which is exactly the diversity a selection experiment needs. Episode-level outcomes are
carried as `episode_solved` for cross-checking only -- run_fidelity.py re-scores every reconstructed
candidate against the hidden suite itself, so label and measurement share one artifact.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTD = ROOT / "large_artifacts" / "qwen35_4b_realrepo_agentic_instrument"
INSTR = ROOT / "experiments" / "qwen35_4b_realrepo_agentic_instrument"
DATA = HERE.parent / "data"
STUB = "raise NotImplementedError  # TODO: implement"


def extract(traj: Path):
    """(old_text_with_stub, spec, final chained implementation) from one episode's edit calls."""
    old_text = spec = impl = None
    try:
        with open(traj, errors="replace") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except Exception:
                    continue
                if ev.get("type") != "tool_execution_start":
                    continue
                if (ev.get("toolName") or "") not in ("edit", "write", "str_replace"):
                    continue
                for ed in ((ev.get("args") or {}).get("edits") or []):
                    if not isinstance(ed, dict):
                        continue
                    old, new = ed.get("oldText") or "", ed.get("newText") or ""
                    if not new.strip():
                        continue
                    if STUB in old:
                        old_text, spec, impl = old, old.replace(STUB, "").rstrip(), new
                    elif impl and old and old in impl:
                        impl = impl.replace(old, new)
    except OSError:
        return None, None, None
    return old_text, spec, impl


def main():
    tasks = {t["task_id"]: t for t in
             json.load(open(INSTR / "data" / "tasks_split.json"))["test"]}
    DATA.mkdir(parents=True, exist_ok=True)
    rows, dropped = [], 0
    for arm in ("base", "warmstart"):
        rep = OUTD / "reports" / f"pi_{arm}_test.json"
        if not rep.exists():
            continue
        for i, e in enumerate(json.loads(rep.read_text()).get("episodes", [])):
            t = tasks.get(e.get("task_id"))
            if not t or not e.get("traj"):
                continue
            old_text, spec, impl = extract(OUTD / e["traj"])
            if not (old_text and spec and impl):
                dropped += 1
                continue
            rows.append({"pair_id": f"{arm}_{i}", "arm": arm, "task_id": t["task_id"],
                         "repo": t["repo"], "rel_file": t["rel_file"], "qual_name": t["qual_name"],
                         "fail_to_pass": t["fail_to_pass"], "test_cmd_hint": t.get("test_cmd_hint"),
                         "old_text": old_text, "spec": spec, "impl": impl,
                         "episode_solved": bool(e.get("solved"))})
    outp = DATA / "pairs.jsonl"
    with open(outp, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tasks_covered = {r["task_id"] for r in rows}
    print(f"exported {len(rows)} pairs over {len(tasks_covered)} tasks ({dropped} episodes unextractable)")
    print(f"episode-solved balance: {sum(r['episode_solved'] for r in rows)} / {len(rows)}")
    print(f"-> {outp}")

    # Task spec sheet for check GENERATION: spec + import hints ONLY. The generator must never see
    # candidate implementations or the repo's tests -- that is the firewall this experiment is about.
    spec_rows = {}
    for r in rows:
        if r["task_id"] in spec_rows:
            continue
        parts = Path(r["rel_file"]).parts
        pkg = parts[1] if parts[0] == "src" else parts[0]
        pkg = pkg[:-3] if pkg.endswith(".py") else pkg
        spec_rows[r["task_id"]] = {"task_id": r["task_id"], "repo": r["repo"],
                                   "rel_file": r["rel_file"], "qual_name": r["qual_name"],
                                   "package": pkg, "spec": r["spec"]}
    with open(DATA / "specs.jsonl", "w") as fh:
        for v in spec_rows.values():
            fh.write(json.dumps(v) + "\n")
    print(f"wrote {len(spec_rows)} spec sheets -> {DATA/'specs.jsonl'}")


if __name__ == "__main__":
    main()
