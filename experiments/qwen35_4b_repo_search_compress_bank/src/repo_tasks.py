"""Fresh procedural Python repositories with private executable tests.

The generated repositories are deliberately small enough for fast agent loops but
exercise real file inspection, exact patches, subprocess execution, and hidden
edge cases.  Hidden tests and oracle edits live only in the host-side task object;
they are never written into the repository or shown to the model.
"""

from __future__ import annotations

import hashlib
import json
import random
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Patch:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class RepoTask:
    task_id: str
    family: str
    split: str
    issue: str
    files: dict[str, str]
    hidden_test: str
    oracle_patches: tuple[Patch, ...]
    difficulty: int

    def public_manifest(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "difficulty": self.difficulty,
            "issue_sha256": hashlib.sha256(self.issue.encode()).hexdigest(),
            "file_sha256": {
                path: hashlib.sha256(text.encode()).hexdigest()
                for path, text in sorted(self.files.items())
            },
        }


def _files(issue: str, module: str, body: str, visible: str, *, extras=None) -> dict[str, str]:
    result = {
        "README.md": f"# Cedar repair fixture\n\n{issue}\n",
        "src/__init__.py": "",
        f"src/{module}.py": body,
        "src/constants.py": (
            '"""Unrelated package defaults; inspect before assuming relevance."""\n'
            "DEFAULT_LIMIT = 100\nDEFAULT_LABEL = 'unrouted'\n"
        ),
        "src/models.py": (
            '"""Small shared data helpers used by some fixtures."""\n\n'
            "def copy_mapping(value):\n    return dict(value)\n"
        ),
        "tests/test_visible.py": visible,
    }
    result.update(extras or {})
    return result


def _window(seed: int, task_id: str, split: str) -> RepoTask:
    width = 2 + seed % 4
    issue = (
        "Fix `rollup(values, width)` in `src/windows.py`. It sums consecutive chunks, "
        "including a final partial chunk. A `None` is a hard segment boundary: flush the "
        "current partial chunk and start a new segment, never forming a chunk across the "
        "boundary. Preserve the existing positive-width validation."
    )
    old = """    clean = [value for value in values if value is not None]
    return [sum(clean[i:i + width]) for i in range(0, len(clean), width)]"""
    new = """    result = []
    segment = []
    for value in values:
        if value is None:
            result.extend(sum(segment[i:i + width]) for i in range(0, len(segment), width))
            segment = []
        else:
            segment.append(value)
    result.extend(sum(segment[i:i + width]) for i in range(0, len(segment), width))
    return result"""
    body = f'''"""Segment-aware window aggregation."""

def rollup(values, width):
    if width <= 0:
        raise ValueError("width must be positive")
{old}
'''
    left = list(range(1, width + 2))
    right = list(range(10, 10 + width + 1))
    values = left + [None] + right
    expected = [sum(left[:width]), sum(left[width:]), sum(right[:width]), sum(right[width:])]
    visible = (
        "from src.windows import rollup\n"
        f"assert rollup({values!r}, {width}) == {expected!r}\n"
        f"assert rollup([None, None], {width}) == []\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.windows import rollup\n"
        "assert rollup([1,None,2],2)==[1,2]\n"
        "assert rollup([1,2,3,None,4,5],2)==[3,3,9]\n"
        "try:\n rollup([1],0)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return RepoTask(task_id, "segmented_rollup", split, issue,
                    _files(issue, "windows", body, visible), hidden,
                    (Patch("src/windows.py", old, new),), 2)


def _routing(seed: int, task_id: str, split: str) -> RepoTask:
    threshold = 2 + seed % 5
    issue = (
        "Fix `route(event, rules, default)` in `src/routing.py`. Consider matching kinds "
        "whose `min_priority` is met. If several qualify, choose the most specific rule "
        "(largest `min_priority`); equal-specificity ties retain input order. Return the "
        "default when no rule qualifies."
    )
    old = """    for rule in rules:
        if event.get("kind") == rule["kind"] and priority >= rule.get("min_priority", 0):
            return rule["target"]
    return default"""
    new = """    best = None
    for rule in rules:
        if event.get("kind") != rule["kind"] or priority < rule.get("min_priority", 0):
            continue
        if best is None or rule.get("min_priority", 0) > best.get("min_priority", 0):
            best = rule
    return best["target"] if best is not None else default"""
    body = f'''"""Specificity-ordered event routing."""

def route(event, rules, default="unrouted"):
    priority = event.get("priority", 0)
{old}
'''
    rules = [
        {"kind": "build", "min_priority": 0, "target": "normal"},
        {"kind": "build", "min_priority": threshold, "target": "fast"},
        {"kind": "deploy", "min_priority": 1, "target": "release"},
    ]
    visible = (
        "from src.routing import route\n"
        f"rules={rules!r}\n"
        f"assert route({{'kind':'build','priority':{threshold + 1}}},rules,'hold')=='fast'\n"
        "assert route({'kind':'build','priority':1},rules,'hold')=='normal'\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.routing import route\n"
        f"rules={rules!r}\n"
        "assert route({'kind':'build','priority':99},rules,'hold')=='fast'\n"
        "assert route({'kind':'other','priority':99},rules,'hold')=='hold'\n"
        "t=[{'kind':'x','min_priority':2,'target':'first'},"
        "{'kind':'x','min_priority':2,'target':'second'}]\n"
        "assert route({'kind':'x','priority':2},t,'hold')=='first'\n"
    )
    return RepoTask(task_id, "specificity_router", split, issue,
                    _files(issue, "routing", body, visible), hidden,
                    (Patch("src/routing.py", old, new),), 2)


def _merge(seed: int, task_id: str, split: str) -> RepoTask:
    key = "id" if seed % 2 == 0 else "code"
    issue = (
        f"Fix `merge_unique(groups)` in `src/merge.py`. Keep the first position for each "
        f"`{key}`. Later duplicates fill fields that are absent or currently `None`, but "
        "must never overwrite an existing non-None value. Inputs must not be mutated."
    )
    old = f'''    seen = set()
    merged = []
    for group in groups:
        for item in group:
            if item["{key}"] in seen:
                continue
            seen.add(item["{key}"])
            merged.append(dict(item))
    return merged'''
    new = f'''    positions = {{}}
    merged = []
    for group in groups:
        for item in group:
            marker = item["{key}"]
            if marker not in positions:
                positions[marker] = len(merged)
                merged.append(dict(item))
                continue
            current = merged[positions[marker]]
            for field, value in item.items():
                if field not in current or current[field] is None:
                    current[field] = value
    return merged'''
    body = f'''"""Stable field-aware collection merge."""

def merge_unique(groups):
{old}
'''
    groups = [[{key: "a", "v": None}, {key: "b", "v": 2}],
              [{key: "a", "v": 9, "note": "filled"}, {key: "c", "v": 3}]]
    expected = [{key: "a", "v": 9, "note": "filled"}, groups[0][1], groups[1][1]]
    visible = (
        "from src.merge import merge_unique\n"
        f"assert merge_unique({groups!r}) == {expected!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.merge import merge_unique\n"
        f"a=[[{{'{key}':'x','v':0}}],[{{'{key}':'x','v':1,'extra':2}}]]\n"
        f"assert merge_unique(a)==[{{'{key}':'x','v':0,'extra':2}}]\n"
        f"assert a==[[{{'{key}':'x','v':0}}],[{{'{key}':'x','v':1,'extra':2}}]]\n"
        "assert merge_unique([])==[]\n"
    )
    return RepoTask(task_id, "stable_merge", split, issue,
                    _files(issue, "merge", body, visible), hidden,
                    (Patch("src/merge.py", old, new),), 2)


def _quota(seed: int, task_id: str, split: str) -> RepoTask:
    total = 7 + seed % 12
    names = [f"lane_{i}" for i in range(3 + seed % 3)]
    weights = {name: 1 + (i + seed) % 4 for i, name in enumerate(names)}
    issue = (
        "Fix `allocate(total, weights)` in `src/quota.py` to perform largest-remainder "
        "proportional allocation. Floor exact weighted shares, then assign leftover units "
        "by descending fractional remainder; ties follow mapping order. Reject negative "
        "totals, empty mappings, and non-positive weights. The allocation must sum to total."
    )
    old = """    names = list(weights)
    each, remainder = divmod(total, len(names))
    return {name: each + (index < remainder) for index, name in enumerate(names)}"""
    new = """    names = list(weights)
    weight_sum = sum(weights.values())
    exact = [total * weights[name] / weight_sum for name in names]
    shares = [int(value) for value in exact]
    remaining = total - sum(shares)
    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - shares[i]), i))
    for index in order[:remaining]:
        shares[index] += 1
    return {name: shares[index] for index, name in enumerate(names)}"""
    body = f'''"""Weighted integer quota allocation."""

def allocate(total, weights):
    if total < 0 or not weights or any(value <= 0 for value in weights.values()):
        raise ValueError("invalid total or weights")
{old}
'''
    exact = [total * value / sum(weights.values()) for value in weights.values()]
    shares = [int(value) for value in exact]
    order = sorted(range(len(names)), key=lambda i: (-(exact[i] - shares[i]), i))
    for index in order[: total - sum(shares)]:
        shares[index] += 1
    expected = {name: shares[index] for index, name in enumerate(names)}
    visible = (
        "from src.quota import allocate\n"
        f"weights={weights!r}\n"
        f"assert allocate({total},weights)=={expected!r}\n"
        f"assert sum(allocate({total},weights).values())=={total}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.quota import allocate\n"
        "assert allocate(7,{'a':1,'b':2,'c':1})=={'a':2,'b':3,'c':2}\n"
        "assert allocate(2,{'x':1,'y':1,'z':1})=={'x':1,'y':1,'z':0}\n"
        "for weights,total in [({},1),({'a':0},1),({'a':1},-1)]:\n"
        " try:\n  allocate(total,weights)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
    )
    return RepoTask(task_id, "weighted_quota", split, issue,
                    _files(issue, "quota", body, visible), hidden,
                    (Patch("src/quota.py", old, new),), 3)


def _intervals(seed: int, task_id: str, split: str) -> RepoTask:
    offset = seed % 7
    issue = (
        "Fix `compact(intervals)` in `src/intervals.py`. Intervals are closed "
        "`(start, end, label)` tuples. Merge overlapping or touching intervals only when "
        "their labels match; differently labelled intervals remain independent. Sort by "
        "`(start, end, label)` and do not mutate the input."
    )
    old = """    ordered = sorted([list(item) for item in intervals])
    merged = [ordered[0]]
    for start, end, label in ordered[1:]:
        current = merged[-1]
        if start <= current[1]:
            current[1] = max(current[1], end)
        else:
            merged.append([start, end, label])
    return [tuple(item) for item in merged]"""
    new = """    ordered = sorted([list(item) for item in intervals])
    merged = []
    for start, end, label in ordered:
        match = next((item for item in reversed(merged)
                      if item[2] == label and start <= item[1]), None)
        if match is not None:
            match[1] = max(match[1], end)
        else:
            merged.append([start, end, label])
    merged.sort(key=lambda item: (item[0], item[1], item[2]))
    return [tuple(item) for item in merged]"""
    body = f'''"""Label-aware closed interval compaction."""

def compact(intervals):
    if not intervals:
        return []
{old}
'''
    values = [(offset, offset + 3, "a"), (offset + 3, offset + 5, "a"),
              (offset + 4, offset + 7, "b")]
    expected = [(offset, offset + 5, "a"), (offset + 4, offset + 7, "b")]
    visible = (
        "from src.intervals import compact\n"
        f"assert compact({values!r})=={expected!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.intervals import compact\n"
        "xs=[(0,3,'a'),(3,5,'b'),(7,8,'b'),(8,9,'b')]\n"
        "assert compact(xs)==[(0,3,'a'),(3,5,'b'),(7,9,'b')]\n"
        "assert xs==[(0,3,'a'),(3,5,'b'),(7,8,'b'),(8,9,'b')]\n"
        "assert compact([])==[]\n"
    )
    return RepoTask(task_id, "label_intervals", split, issue,
                    _files(issue, "intervals", body, visible), hidden,
                    (Patch("src/intervals.py", old, new),), 3)


def _slugs(seed: int, task_id: str, split: str) -> RepoTask:
    sample = "  Blue\tRiver   Delta  " if seed % 2 else "  Blue   River\tDelta  "
    issue = (
        "Fix `slug(text)` and `build_index(labels)` in `src/slugs.py`. Normalize arbitrary "
        "whitespace runs to one hyphen after lowercasing. Preserve normalization collisions "
        "by mapping each slug to every original label in encounter order."
    )
    old = '''def slug(text):
    return "-".join(text.strip().lower().split(" "))

def build_index(labels):
    return {slug(label): label for label in labels}'''
    new = '''def slug(text):
    return "-".join(text.strip().lower().split())

def build_index(labels):
    index = {}
    for label in labels:
        index.setdefault(slug(label), []).append(label)
    return index'''
    body = f'''"""Collision-preserving human label normalization."""

{old}
'''
    labels = [sample, "Blue River Delta", "Other"]
    expected = {"blue-river-delta": [sample, "Blue River Delta"], "other": ["Other"]}
    visible = (
        "from src.slugs import slug,build_index\n"
        f"assert slug({sample!r})=='blue-river-delta'\n"
        f"assert build_index({labels!r})=={expected!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.slugs import slug,build_index\n"
        "assert slug('  A\\n B  C ')=='a-b-c'\n"
        "assert slug('')==''\n"
        "assert build_index(['Red Fox','red   fox'])=={'red-fox':['Red Fox','red   fox']}\n"
    )
    return RepoTask(task_id, "collision_index", split, issue,
                    _files(issue, "slugs", body, visible), hidden,
                    (Patch("src/slugs.py", old, new),), 2)


def _overlay(seed: int, task_id: str, split: str) -> RepoTask:
    delete_key = "obsolete" if seed % 2 else "retired"
    issue = (
        "Fix `overlay(base, override)` in `src/config.py`. Recursively merge nested mappings "
        "without mutating either input. A `None` override deletes that key. Non-mapping "
        "values replace the old value. Return an independent deep copy."
    )
    old = """def overlay(base, override):
    result = deepcopy(base)
    result.update(override)
    return result"""
    new = """def overlay(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if value is None:
            result.pop(key, None)
        elif isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = overlay(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result"""
    body = f'''"""Immutable recursive configuration overlay."""

from copy import deepcopy

{old}
'''
    base = {"service": {"port": 80, "tls": False}, delete_key: 3, "items": [1]}
    override = {"service": {"tls": True}, delete_key: None}
    expected = {"service": {"port": 80, "tls": True}, "items": [1]}
    visible = (
        "from src.config import overlay\n"
        f"base={base!r}; override={override!r}\n"
        f"assert overlay(base,override)=={expected!r}\n"
        f"assert base=={base!r} and override=={override!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.config import overlay\n"
        "a={'x':{'y':1,'z':2},'v':[1]}; b={'x':{'y':9},'v':[2]}\n"
        "out=overlay(a,b)\n"
        "assert out=={'x':{'y':9,'z':2},'v':[2]}\n"
        "out['v'].append(3); assert a['v']==[1] and b['v']==[2]\n"
        "assert overlay({'a':1},{'missing':None})=={'a':1}\n"
    )
    return RepoTask(task_id, "recursive_overlay", split, issue,
                    _files(issue, "config", body, visible,
                           extras={"src/schema.py": "REQUIRED_KEYS = ('service',)\n"}),
                    hidden, (Patch("src/config.py", old, new),), 4)


def _topology(seed: int, task_id: str, split: str) -> RepoTask:
    suffix = seed % 10
    issue = (
        "Fix `dependency_order(nodes, dependencies)` in `src/dependencies.py`. Return a "
        "topological order where every dependency precedes its node. When several nodes are "
        "ready, preserve their order from `nodes`. Dependencies may omit nodes with no edges. "
        "Raise `ValueError` for a cycle or dependency naming an unknown node."
    )
    old = """def dependency_order(nodes, dependencies):
    return sorted(nodes)"""
    new = """def dependency_order(nodes, dependencies):
    positions = {name: index for index, name in enumerate(nodes)}
    if len(positions) != len(nodes):
        raise ValueError("duplicate node")
    remaining = {}
    for node in nodes:
        deps = set(dependencies.get(node, ()))
        if any(dep not in positions for dep in deps):
            raise ValueError("unknown dependency")
        remaining[node] = deps
    result = []
    while remaining:
        ready = [node for node in nodes if node in remaining and not remaining[node]]
        if not ready:
            raise ValueError("dependency cycle")
        for node in ready:
            result.append(node)
            del remaining[node]
            for deps in remaining.values():
                deps.discard(node)
    return result"""
    body = f'''"""Stable dependency planning."""

{old}
'''
    nodes = [f"package_{suffix}", f"lint_{suffix}", f"build_{suffix}", f"test_{suffix}"]
    deps = {nodes[0]: [nodes[2], nodes[3]], nodes[3]: [nodes[2]]}
    expected = [nodes[1], nodes[2], nodes[3], nodes[0]]
    visible = (
        "from src.dependencies import dependency_order\n"
        f"nodes={nodes!r}; deps={deps!r}\n"
        f"assert dependency_order(nodes,deps)=={expected!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.dependencies import dependency_order\n"
        "assert dependency_order(['b','a','c'],{'c':['a']})==['b','a','c']\n"
        "for nodes,deps in [(['a','b'],{'a':['b'],'b':['a']}),(['a'],{'a':['x']})]:\n"
        " try:\n  dependency_order(nodes,deps)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
    )
    return RepoTask(task_id, "dependency_order", split, issue,
                    _files(issue, "dependencies", body, visible,
                           extras={"src/build.py": "from .dependencies import dependency_order\n"}),
                    hidden, (Patch("src/dependencies.py", old, new),), 4)


def _cache(seed: int, task_id: str, split: str) -> RepoTask:
    ttl = 2 + seed % 5
    issue = (
        "Fix `ExpiringCache.get` in `src/cache.py`. Stored falsy values such as `0` are valid. "
        "An entry expires exactly when `clock() >= expires_at`; expired entries are removed. "
        "Missing and expired keys return the caller's default."
    )
    old = """    def get(self, key, default=None):
        entry = self._items.get(key)
        if not entry:
            return default
        value, expires_at = entry
        if not value or self._clock() > expires_at:
            self._items.pop(key, None)
            return default
        return value"""
    new = """    def get(self, key, default=None):
        entry = self._items.get(key)
        if entry is None:
            return default
        value, expires_at = entry
        if self._clock() >= expires_at:
            self._items.pop(key, None)
            return default
        return value"""
    body = f'''"""Clock-injected expiring cache."""

class ExpiringCache:
    def __init__(self, clock):
        self._clock = clock
        self._items = {{}}

    def set(self, key, value, ttl):
        self._items[key] = (value, self._clock() + ttl)

{old}
'''
    visible = (
        "from src.cache import ExpiringCache\n"
        "now=[10]; cache=ExpiringCache(lambda:now[0])\n"
        f"cache.set('zero',0,{ttl}); assert cache.get('zero','miss')==0\n"
        f"now[0]+={ttl}; assert cache.get('zero','miss')=='miss'\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.cache import ExpiringCache\n"
        "now=[0]; c=ExpiringCache(lambda:now[0]); c.set('x','',2)\n"
        "assert c.get('x','d')==''\n"
        "now[0]=1; assert c.get('x')==''\n"
        "now[0]=2; assert c.get('x','gone')=='gone' and 'x' not in c._items\n"
        "assert c.get('missing',7)==7\n"
    )
    return RepoTask(task_id, "ttl_cache", split, issue,
                    _files(issue, "cache", body, visible,
                           extras={"src/clock.py": "def monotonic_seconds():\n    return 0\n"}),
                    hidden, (Patch("src/cache.py", old, new),), 4)


def _retry(seed: int, task_id: str, split: str) -> RepoTask:
    attempts = 2 + seed % 4
    delay = 1 + seed % 3
    issue = (
        "Fix `retry_schedule(max_attempts, base_delay)` in `src/retry.py`. It returns one "
        "delay per attempt: the first attempt is immediate (`0`), then delays double from "
        "`base_delay`. `max_attempts` counts total attempts, not retries. Require at least one "
        "attempt and a non-negative base delay."
    )
    old = """def retry_schedule(max_attempts, base_delay):
    if max_attempts < 0 or base_delay < 0:
        raise ValueError("invalid retry policy")
    return [base_delay * (2 ** index) for index in range(max_attempts + 1)]"""
    new = """def retry_schedule(max_attempts, base_delay):
    if max_attempts < 1 or base_delay < 0:
        raise ValueError("invalid retry policy")
    return [0] + [base_delay * (2 ** index) for index in range(max_attempts - 1)]"""
    body = f'''"""Pure retry timing policy."""

{old}
'''
    expected = [0] + [delay * (2 ** index) for index in range(attempts - 1)]
    visible = (
        "from src.retry import retry_schedule\n"
        f"assert retry_schedule({attempts},{delay})=={expected!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.retry import retry_schedule\n"
        "assert retry_schedule(1,5)==[0]\n"
        "assert retry_schedule(4,0)==[0,0,0,0]\n"
        "for a,d in [(0,1),(1,-1)]:\n"
        " try:\n  retry_schedule(a,d)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
    )
    return RepoTask(task_id, "retry_schedule", split, issue,
                    _files(issue, "retry", body, visible,
                           extras={"src/worker.py": "from .retry import retry_schedule\n"}),
                    hidden, (Patch("src/retry.py", old, new),), 3)


BUILDERS = {
    "segmented_rollup": _window,
    "specificity_router": _routing,
    "stable_merge": _merge,
    "weighted_quota": _quota,
    "label_intervals": _intervals,
    "collision_index": _slugs,
    "recursive_overlay": _overlay,
    "dependency_order": _topology,
    "ttl_cache": _cache,
    "retry_schedule": _retry,
}

TRAIN_FAMILIES = tuple(list(BUILDERS)[:6])
TRANSFER_FAMILIES = tuple(list(BUILDERS)[6:])


def make_tasks(families: list[str] | tuple[str, ...], tasks_per_family: int,
               seed: int, split: str) -> list[RepoTask]:
    rng = random.Random(seed)
    result = []
    for family in families:
        if family not in BUILDERS:
            raise KeyError(family)
        for index in range(tasks_per_family):
            item_seed = rng.randrange(1_000_000_000)
            task_id = f"repo-{split}-{family}-s{seed}-{index:03d}"
            result.append(BUILDERS[family](item_seed, task_id, split))
    return result


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))


class RepoEnv:
    """Constrained real filesystem plus visible and hidden subprocess tests."""

    def __init__(self, task: RepoTask):
        self.task = task
        self._tmp = tempfile.TemporaryDirectory(prefix="repo_scb_")
        self.root = Path(self._tmp.name)
        for rel, text in task.files.items():
            path = self.root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.tool_calls = 0
        self.patch_calls = 0
        self.test_calls = 0

    def close(self) -> None:
        self._tmp.cleanup()

    def tree(self) -> str:
        return "\n".join(
            str(path.relative_to(self.root))
            for path in sorted(self.root.rglob("*")) if path.is_file()
        )

    def read(self, rel: str) -> str:
        try:
            path = self._safe_path(rel, allow_visible_tests=True)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not path.is_file():
            return f"ERROR: no such file: {rel}"
        return path.read_text(encoding="utf-8")[:16000]

    def search(self, query: str) -> str:
        if not query or len(query) > 200:
            return "ERROR: query must contain 1-200 characters"
        hits = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(self.root).as_posix()
            if not (rel.startswith("src/") or rel == "README.md"):
                continue
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if query in line:
                    hits.append(f"{rel}:{line_no}:{line}")
        return "\n".join(hits[:100]) or "NO_MATCHES"

    def patch(self, rel: str, old: str, new: str) -> str:
        try:
            path = self._safe_path(rel, allow_visible_tests=False)
        except ValueError as exc:
            return f"ERROR: {exc}"
        if not path.is_file():
            return f"ERROR: no such file: {rel}"
        if not old or len(old) > 8000 or len(new) > 8000:
            return "ERROR: patch old must be non-empty and old/new <=8000 characters"
        text = path.read_text(encoding="utf-8")
        count = text.count(old)
        if count != 1:
            return f"ERROR: old text matched {count} times; expected exactly 1"
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        self.patch_calls += 1
        return f"PATCH_OK: {rel}"

    def run_visible(self) -> str:
        self.test_calls += 1
        result = self._run([
            sys.executable, "-c",
            "exec(compile(open('tests/test_visible.py').read(), "
            "'tests/test_visible.py', 'exec'))",
        ])
        return self._format_result(result)

    def visible_pass(self) -> bool:
        return self.run_visible().startswith("PASS")

    def hidden_pass(self) -> bool:
        return self._run([sys.executable, "-c", self.task.hidden_test]).returncode == 0

    def initial_visible_fails(self) -> bool:
        return not self.visible_pass()

    def apply_oracle(self) -> None:
        for patch in self.task.oracle_patches:
            result = self.patch(patch.path, patch.old, patch.new)
            if not result.startswith("PATCH_OK"):
                raise AssertionError(result)

    def workspace_digest(self) -> str:
        digest = hashlib.sha256()
        for path in sorted((self.root / "src").rglob("*.py")):
            digest.update(path.relative_to(self.root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()

    def source_snapshot(self) -> dict[str, str]:
        return {
            path.relative_to(self.root).as_posix(): path.read_text(encoding="utf-8")
            for path in sorted((self.root / "src").rglob("*.py"))
        }

    def _safe_path(self, rel: str, *, allow_visible_tests: bool) -> Path:
        candidate = (self.root / rel).resolve()
        root = self.root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError("path escapes repository")
        relative = candidate.relative_to(root).as_posix()
        allowed = relative.startswith("src/") or relative == "README.md"
        if allow_visible_tests:
            allowed = allowed or relative == "tests/test_visible.py"
        if not allowed:
            raise ValueError("path is outside the allowed repository surface")
        return candidate

    def _run(self, command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            command, cwd=self.root, text=True, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, timeout=6, check=False, preexec_fn=_limit_child,
        )

    @staticmethod
    def _format_result(result: subprocess.CompletedProcess) -> str:
        status = "PASS" if result.returncode == 0 else f"FAIL(exit={result.returncode})"
        output = (result.stdout + "\n" + result.stderr).strip()[-6000:]
        return f"{status}\n{output}".strip()


def manifest_digest(tasks: list[RepoTask]) -> str:
    payload = json.dumps([task.public_manifest() for task in tasks], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
