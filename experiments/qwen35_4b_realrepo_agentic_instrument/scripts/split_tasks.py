"""Turn validated candidates into the firewalled instrument: task ids, splits, difficulty strata.

THE FIREWALL IS BY REPO, and the side of every repo was fixed in repos.py BEFORE any task was
generated or any number measured, so the split cannot be tuned to a result later. A held-out task
therefore lives in a library whose code, conventions, and test style were never in any training
corpus — the only version of this firewall that survives contact with a harvest-and-train pipeline.
(Splitting tasks within a repo would leak: harvesting `funcy` teaches the model that repo's idioms,
its test layout, and often the neighbouring functions its held-out functions call.)

Balance matters as much as size. Capping tasks per repo in the eval split stops one large library
(boltons, more-itertools) from becoming 40% of the instrument, which would silently turn a
general-capability measurement into a measurement of one codebase's style.

Also emits, per task:
  * `task_id`      stable and unique: "<repo>::<rel_file>::<qual_name>"
  * `test_cmd_hint` the specific test files a developer would actually run (from the broken tests) --
                    on the slower suites the full run dominates the episode wall-clock
  * `stratum`      small/medium/large by body_lines, so difficulty can be reported without re-running
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE.parent / "data"

from repos import BY_NAME  # noqa: E402


def stratum(body_lines):
    if body_lines <= 6:
        return "small"
    return "medium" if body_lines <= 20 else "large"


def test_hint(failing_ids, limit=2):
    files, seen = [], set()
    for nid in failing_ids:
        f = nid.split("::")[0]
        if f not in seen and f.endswith(".py"):
            seen.add(f)
            files.append(f)
        if len(files) >= limit:
            break
    return f"python -m pytest {' '.join(files)} -q" if files else "python -m pytest -q"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", default=str(DATA / "tasks_all.json"))
    ap.add_argument("--max-per-repo-test", type=int, default=12,
                    help="balance cap for the eval split (0 = uncapped)")
    ap.add_argument("--max-per-repo-train", type=int, default=0, help="0 = keep all")
    ap.add_argument("--out", default=str(DATA / "tasks_split.json"))
    a = ap.parse_args()

    cands = json.load(open(a.tasks))["candidates"]
    admitted = [c for c in cands if c.get("admit")]
    print(f"{len(admitted)} admitted candidates from {len(cands)} validated", flush=True)

    by_repo = defaultdict(list)
    for c in admitted:
        c["task_id"] = f"{c['repo']}::{c['rel_file']}::{c['qual_name']}"
        c["stratum"] = stratum(c["body_lines"])
        c["fail_to_pass"] = c.pop("failing_ids", [])
        c["test_cmd_hint"] = test_hint(c["fail_to_pass"])
        by_repo[c["repo"]].append(c)

    split = {"train": [], "test": []}
    for repo, items in sorted(by_repo.items()):
        group = BY_NAME[repo]["group"]
        # Deterministic difficulty-spread selection: sort by body size, take an even stride, so the
        # balance cap keeps small AND large functions rather than the smallest N.
        items.sort(key=lambda c: (c["body_lines"], c["rel_file"], c["qual_name"]))
        cap = a.max_per_repo_test if group == "test" else a.max_per_repo_train
        if cap and len(items) > cap:
            step = len(items) / cap
            items = [items[int(i * step)] for i in range(cap)]
        split[group] += items

    meta = {}
    for side in ("train", "test"):
        rows = split[side]
        meta[side] = {"n_tasks": len(rows),
                      "n_repos": len({r["repo"] for r in rows}),
                      "by_repo": dict(Counter(r["repo"] for r in rows)),
                      "by_stratum": dict(Counter(r["stratum"] for r in rows)),
                      "median_body_lines": sorted(r["body_lines"] for r in rows)[len(rows) // 2] if rows else 0,
                      "median_fail_to_pass": sorted(len(r["fail_to_pass"]) for r in rows)[len(rows) // 2] if rows else 0}
        print(f"\n{side.upper()}: {meta[side]['n_tasks']} tasks over {meta[side]['n_repos']} repos"
              f" | strata {meta[side]['by_stratum']}"
              f" | median body {meta[side]['median_body_lines']} lines", flush=True)
        for r, n in sorted(meta[side]["by_repo"].items()):
            print(f"    {r:20s} {n}", flush=True)

    overlap = {r["repo"] for r in split["train"]} & {r["repo"] for r in split["test"]}
    assert not overlap, f"FIREWALL VIOLATION: repos on both sides: {overlap}"
    Path(a.out).write_text(json.dumps({"meta": meta, **split}, indent=1))
    print(f"\nFIREWALL OK (no repo on both sides)\nsaved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
