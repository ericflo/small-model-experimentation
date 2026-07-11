"""Fresh procedural mini-repository repair tasks for the agent-loop transfer gate."""

from __future__ import annotations

import random
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoTask:
    task_id: str
    family: str
    issue: str
    files: dict[str, str]
    hidden_test: str
    oracle_path: str
    oracle_old: str
    oracle_new: str


def _common(issue: str, module: str, body: str, visible: str) -> dict[str, str]:
    return {
        "README.md": f"# Repair task\n\n{issue}\n",
        "src/__init__.py": "",
        f"src/{module}.py": body,
        "src/constants.py": (
            '"""Shared defaults. This file is not necessarily involved in the bug."""\n'
            "DEFAULT_LIMIT = 100\nDEFAULT_LABEL = 'unrouted'\n"
        ),
        "tests/test_visible.py": visible,
    }


def _window(seed: int, task_id: str) -> RepoTask:
    width = 2 + seed % 3
    issue = ("`rollup(values, width)` sums consecutive chunks, including a final partial chunk. "
             "A `None` value is a hard segment boundary: flush the current partial chunk and start "
             "a new segment, never forming a window across the boundary. Empty segments add nothing.")
    old = '''    clean = [value for value in values if value is not None]\n    return [sum(clean[i:i + width]) for i in range(0, len(clean), width)]'''
    new = '''    result = []\n    segment = []\n    for value in values:\n        if value is None:\n            result.extend(sum(segment[i:i + width]) for i in range(0, len(segment), width))\n            segment = []\n        else:\n            segment.append(value)\n    result.extend(sum(segment[i:i + width]) for i in range(0, len(segment), width))\n    return result'''
    body = f'''"""Segment-aware window aggregation."""\n\ndef rollup(values, width):\n    if width <= 0:\n        raise ValueError("width must be positive")\n{old}\n'''
    values = list(range(1, width + 2)) + [None] + list(range(10, 10 + width + 1))
    left = values[:width + 1]; right = values[width + 2:]
    expected = ([sum(left[:width]), sum(left[width:])]
                + [sum(right[:width]), sum(right[width:])])
    visible = f"from src.windows import rollup\nassert rollup({values!r}, {width}) == {expected!r}\nassert rollup([None, None], {width}) == []\nprint('VISIBLE_OK')\n"
    hidden = "from src.windows import rollup\nassert rollup([1,None,2],2)==[1,2]\nassert rollup([1,2,3,None,4,5],2)==[3,3,9]\ntry:\n rollup([1],0)\n raise AssertionError('missing ValueError')\nexcept ValueError:\n pass\n"
    return RepoTask(task_id, "window_rollup", issue,
                    _common(issue, "windows", body, visible), hidden,
                    "src/windows.py", old, new)


def _routing(seed: int, task_id: str) -> RepoTask:
    threshold = 2 + seed % 4
    issue = ("`route(event, rules, default)` considers rules with matching `kind` whose "
             "`min_priority` is met. If several qualify, choose the most specific rule (largest "
             "min_priority); ties keep input order. Return default when none qualify.")
    old = '''    for rule in rules:\n        if event.get("kind") == rule["kind"] and priority >= rule.get("min_priority", 0):\n            return rule["target"]\n    return default'''
    new = '''    best = None\n    for rule in rules:\n        if event.get("kind") != rule["kind"] or priority < rule.get("min_priority", 0):\n            continue\n        if best is None or rule.get("min_priority", 0) > best.get("min_priority", 0):\n            best = rule\n    return best["target"] if best is not None else default'''
    body = f'''"""Specificity-ordered event routing."""\n\ndef route(event, rules, default="unrouted"):\n    priority = event.get("priority", 0)\n{old}\n'''
    rules = [{"kind": "build", "min_priority": 0, "target": "normal"},
             {"kind": "build", "min_priority": threshold, "target": "fast"},
             {"kind": "deploy", "min_priority": 1, "target": "release"}]
    visible = f"from src.routing import route\nrules = {rules!r}\nassert route({{'kind':'build','priority':{threshold + 1}}}, rules, 'hold') == 'fast'\nassert route({{'kind':'build','priority':1}}, rules, 'hold') == 'normal'\nprint('VISIBLE_OK')\n"
    hidden = f"from src.routing import route\nrules={rules!r}\nassert route({{'kind':'build','priority':99}},rules,'hold')=='fast'\nassert route({{'kind':'other','priority':99}},rules,'hold')=='hold'\ntied=[{{'kind':'x','min_priority':2,'target':'first'}},{{'kind':'x','min_priority':2,'target':'second'}}]\nassert route({{'kind':'x','priority':2}},tied,'hold')=='first'\n"
    return RepoTask(task_id, "routing_table", issue,
                    _common(issue, "routing", body, visible), hidden,
                    "src/routing.py", old, new)


def _dedup(seed: int, task_id: str) -> RepoTask:
    key = "id" if seed % 2 == 0 else "code"
    issue = (f"`merge_unique(groups)` keeps the first position for each `{key}`. Later duplicates "
             "must fill fields that are missing or currently `None`, but must never overwrite an "
             "existing non-None value. Inputs must not be mutated.")
    old = f'''    seen = set()\n    merged = []\n    for group in groups:\n        for item in group:\n            if item["{key}"] in seen:\n                continue\n            seen.add(item["{key}"])\n            merged.append(dict(item))\n    return merged'''
    new = f'''    positions = {{}}\n    merged = []\n    for group in groups:\n        for item in group:\n            marker = item["{key}"]\n            if marker not in positions:\n                positions[marker] = len(merged)\n                merged.append(dict(item))\n                continue\n            current = merged[positions[marker]]\n            for field, value in item.items():\n                if field not in current or current[field] is None:\n                    current[field] = value\n    return merged'''
    body = f'''"""Stable, field-aware collection merge."""\n\ndef merge_unique(groups):\n{old}\n'''
    groups = [[{key: "a", "v": None}, {key: "b", "v": 2}],
              [{key: "a", "v": 9, "note": "filled"}, {key: "c", "v": 3}]]
    expected = [{key: "a", "v": 9, "note": "filled"}, groups[0][1], groups[1][1]]
    visible = f"from src.merge import merge_unique\nassert merge_unique({groups!r}) == {expected!r}\nprint('VISIBLE_OK')\n"
    hidden = f"from src.merge import merge_unique\na=[[{{'{key}':'x','v':0}}],[{{'{key}':'x','v':1,'extra':2}}]]\nassert merge_unique(a)==[{{'{key}':'x','v':0,'extra':2}}]\nassert a==[[{{'{key}':'x','v':0}}],[{{'{key}':'x','v':1,'extra':2}}]]\nassert merge_unique([])==[]\n"
    return RepoTask(task_id, "dedup_merge", issue,
                    _common(issue, "merge", body, visible), hidden,
                    "src/merge.py", old, new)


def _quota(seed: int, task_id: str) -> RepoTask:
    total = 7 + seed % 11
    n_keys = 3 + seed % 3
    names = [f"q{i}" for i in range(n_keys)]
    weights = {name: (i % 3) + 1 for i, name in enumerate(names)}
    issue = ("`allocate(total, weights)` performs largest-remainder proportional allocation. "
             "Take floors of exact weighted shares, then give remaining units to the largest "
             "fractional remainders; ties follow mapping order. Reject negative totals, empty maps, "
             "and non-positive weights. The result must sum exactly to total.")
    old = '''    names = list(weights)\n    each, remainder = divmod(total, len(names))\n    return {name: each + (index < remainder) for index, name in enumerate(names)}'''
    new = '''    names = list(weights)\n    weight_sum = sum(weights.values())\n    exact = [total * weights[name] / weight_sum for name in names]\n    shares = [int(value) for value in exact]\n    remaining = total - sum(shares)\n    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - shares[i]), i))\n    for index in order[:remaining]:\n        shares[index] += 1\n    return {name: shares[index] for index, name in enumerate(names)}'''
    body = f'''"""Weighted integer quota allocation."""\n\ndef allocate(total, weights):\n    if total < 0 or not weights or any(value <= 0 for value in weights.values()):\n        raise ValueError("invalid total or weights")\n{old}\n'''
    exact = [total * v / sum(weights.values()) for v in weights.values()]
    shares = [int(x) for x in exact]
    order = sorted(range(n_keys), key=lambda i: (-(exact[i] - shares[i]), i))
    for i in order[:total - sum(shares)]: shares[i] += 1
    expected = {name: shares[i] for i, name in enumerate(names)}
    visible = f"from src.quota import allocate\nweights={weights!r}\nassert allocate({total}, weights) == {expected!r}\nassert sum(allocate({total}, weights).values()) == {total}\nprint('VISIBLE_OK')\n"
    hidden = "from src.quota import allocate\nassert allocate(7,{'a':1,'b':2,'c':1})=={'a':2,'b':3,'c':2}\nassert allocate(2,{'x':1,'y':1,'z':1})=={'x':1,'y':1,'z':0}\nfor bad in [({},1),({'a':0},1),({'a':1},-1)]:\n try:\n  allocate(bad[1],bad[0])\n  raise AssertionError('missing ValueError')\n except ValueError:\n  pass\n"
    return RepoTask(task_id, "quota_allocator", issue,
                    _common(issue, "quota", body, visible), hidden,
                    "src/quota.py", old, new)


def _interval(seed: int, task_id: str) -> RepoTask:
    offset = seed % 5
    issue = ("`compact(intervals)` receives closed `(start, end, label)` intervals. Merge overlap "
             "or touching endpoints only when labels match; a differently labelled interval must "
             "remain independent. Sort output by `(start, end, label)` and do not mutate input.")
    old = "if start <= current[1]:"
    new = "if label == current[2] and start <= current[1]:"
    body = f'''"""Label-aware closed interval compaction."""\n\ndef compact(intervals):\n    if not intervals:\n        return []\n    ordered = sorted([list(item) for item in intervals])\n    merged = [ordered[0]]\n    for start, end, label in ordered[1:]:\n        current = merged[-1]\n        {old}\n            current[1] = max(current[1], end)\n        else:\n            merged.append([start, end, label])\n    return [tuple(item) for item in merged]\n'''
    intervals = [(offset, offset + 3, "a"), (offset + 3, offset + 5, "a"),
                 (offset + 5, offset + 7, "b")]
    expected = [(offset, offset + 5, "a"), (offset + 5, offset + 7, "b")]
    visible = f"from src.intervals import compact\nassert compact({intervals!r}) == {expected!r}\nprint('VISIBLE_OK')\n"
    hidden = "from src.intervals import compact\nxs=[(0,3,'a'),(3,5,'b'),(7,8,'b'),(8,9,'b')]\nassert compact(xs)==[(0,3,'a'),(3,5,'b'),(7,9,'b')]\nassert xs==[(0,3,'a'),(3,5,'b'),(7,8,'b'),(8,9,'b')]\nassert compact([])==[]\n"
    return RepoTask(task_id, "interval_compactor", issue,
                    _common(issue, "intervals", body, visible), hidden,
                    "src/intervals.py", old, new)


def _slug(seed: int, task_id: str) -> RepoTask:
    sample = "  Blue\tRiver   Delta  " if seed % 2 else "  Blue   River\tDelta  "
    issue = ("`slug(text)` lowercases words and joins arbitrary whitespace runs with one hyphen. "
             "`build_index(labels)` must preserve every original label by mapping each slug to a "
             "list in encounter order, including normalization collisions.")
    old = '''def slug(text):\n    return "-".join(text.strip().lower().split(" "))\n\ndef build_index(labels):\n    return {slug(label): label for label in labels}'''
    new = '''def slug(text):\n    return "-".join(text.strip().lower().split())\n\ndef build_index(labels):\n    index = {}\n    for label in labels:\n        index.setdefault(slug(label), []).append(label)\n    return index'''
    body = f'''"""Collision-preserving human label normalization."""\n\n{old}\n'''
    labels = [sample, "Blue River Delta", "Other"]
    visible = f"from src.slugs import slug, build_index\nassert slug({sample!r}) == 'blue-river-delta'\nassert build_index({labels!r}) == {{'blue-river-delta': [{sample!r}, 'Blue River Delta'], 'other': ['Other']}}\nprint('VISIBLE_OK')\n"
    hidden = "from src.slugs import slug, build_index\nassert slug('  A\\n B  C ')=='a-b-c'\nassert slug('')==''\nassert build_index(['Red Fox','red   fox'])=={'red-fox':['Red Fox','red   fox']}\n"
    return RepoTask(task_id, "slug_index", issue,
                    _common(issue, "slugs", body, visible), hidden,
                    "src/slugs.py", old, new)


BUILDERS = {
    "window_rollup": _window,
    "routing_table": _routing,
    "dedup_merge": _dedup,
    "quota_allocator": _quota,
    "interval_compactor": _interval,
    "slug_index": _slug,
}


def make_tasks(families: list[str], tasks_per_family: int, seed: int) -> list[RepoTask]:
    rng = random.Random(seed)
    tasks = []
    for family in families:
        if family not in BUILDERS:
            raise KeyError(family)
        for index in range(tasks_per_family):
            item_seed = rng.randrange(1_000_000_000)
            task_id = f"repo-{family}-s{seed}-{index}"
            tasks.append(BUILDERS[family](item_seed, task_id))
    return tasks


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


class RepoEnv:
    """A constrained filesystem/test environment with exact-replacement patches."""

    def __init__(self, task: RepoTask):
        self.task = task
        self._tmp = tempfile.TemporaryDirectory(prefix="ftpo_repo_agent_")
        self.root = Path(self._tmp.name)
        for rel, text in task.files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.tool_calls = 0
        self.patch_calls = 0

    def close(self) -> None:
        self._tmp.cleanup()

    def tree(self) -> str:
        return "\n".join(str(p.relative_to(self.root)) for p in sorted(self.root.rglob("*"))
                         if p.is_file())

    def read(self, rel: str) -> str:
        path = self._safe_path(rel, allow_tests=True)
        if not path.is_file():
            return f"ERROR: no such file: {rel}"
        return path.read_text(encoding="utf-8")[:12000]

    def search(self, query: str) -> str:
        if not query:
            return "ERROR: empty query"
        hits = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    hits.append(f"{path.relative_to(self.root)}:{line_no}:{line}")
        return "\n".join(hits[:80]) or "NO_MATCHES"

    def patch(self, rel: str, old: str, new: str) -> str:
        try:
            path = self._safe_path(rel, allow_tests=False)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not path.is_file():
            return f"ERROR: no such file: {rel}"
        if not old or len(old) > 5000 or len(new) > 5000:
            return "ERROR: patch old/new must be non-empty and <=5000 characters"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            return f"ERROR: old text matched {count} times; expected exactly 1"
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        self.patch_calls += 1
        return f"PATCH_OK: {rel}"

    def run_visible(self) -> str:
        result = self._run([
            sys.executable, "-c",
            "exec(compile(open('tests/test_visible.py').read(), 'tests/test_visible.py', 'exec'))",
        ])
        return self._format_result(result)

    def hidden_pass(self) -> bool:
        result = self._run([sys.executable, "-c", self.task.hidden_test])
        return result.returncode == 0

    def initial_visible_fails(self) -> bool:
        return self._run([
            sys.executable, "-c",
            "exec(compile(open('tests/test_visible.py').read(), 'tests/test_visible.py', 'exec'))",
        ]).returncode != 0

    def apply_oracle(self) -> None:
        result = self.patch(self.task.oracle_path, self.task.oracle_old, self.task.oracle_new)
        if not result.startswith("PATCH_OK"):
            raise AssertionError(result)

    def _safe_path(self, rel: str, *, allow_tests: bool) -> Path:
        candidate = (self.root / rel).resolve()
        if self.root.resolve() not in candidate.parents:
            raise ValueError("path escapes repository")
        relative = candidate.relative_to(self.root).as_posix()
        allowed = relative.startswith("src/") or relative == "README.md"
        if allow_tests:
            allowed = allowed or relative.startswith("tests/")
        if not allowed:
            raise ValueError("path is outside the allowed tree")
        return candidate

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(cmd, cwd=self.root, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=5, check=False,
                              preexec_fn=_limit_child)

    @staticmethod
    def _format_result(result: subprocess.CompletedProcess) -> str:
        status = "PASS" if result.returncode == 0 else f"FAIL(exit={result.returncode})"
        output = (result.stdout + "\n" + result.stderr).strip()[-4000:]
        return f"{status}\n{output}".strip()
