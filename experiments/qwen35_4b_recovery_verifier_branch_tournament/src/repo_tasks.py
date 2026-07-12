"""Fresh procedural repositories for verifier-conditioned repair.

Every fixture contains two independent defects.  ``partial_patches`` repair one
defect but are required to remain visible-test failures; ``oracle_patches``
repair both.  Hidden executables and both patch sets remain host-side and are
never included in a model-facing manifest.
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
    partial_patches: tuple[Patch, ...]
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
        "README.md": f"# Juniper recovery fixture\n\n{issue}\n",
        "src/__init__.py": "",
        f"src/{module}.py": body,
        "src/constants.py": (
            '"""Unrelated defaults; do not assume every constant is relevant."""\n'
            "DEFAULT_LIMIT = 64\nDEFAULT_MODE = 'safe'\n"
        ),
        "src/models.py": (
            '"""Small unrelated helpers shared by some generated packages."""\n\n'
            "def shallow_copy(value):\n    return dict(value)\n"
        ),
        "tests/test_visible.py": visible,
    }
    result.update(extras or {})
    return result


def _make_task(
    *, task_id: str, family: str, split: str, issue: str, module: str,
    prelude: str, signature: str, old: str, partial: str, new: str,
    visible: str, hidden: str, difficulty: int = 3, extras=None,
) -> RepoTask:
    body = f'{prelude}\n\ndef {signature}:\n{old}\n'
    path = f"src/{module}.py"
    return RepoTask(
        task_id=task_id,
        family=family,
        split=split,
        issue=issue,
        files=_files(issue, module, body, visible, extras=extras),
        hidden_test=hidden,
        oracle_patches=(Patch(path, old, new),),
        partial_patches=(Patch(path, old, partial),),
        difficulty=difficulty,
    )


def _layered_settings(seed: int, task_id: str, split: str) -> RepoTask:
    first_port = 5000 + seed % 1000
    second_port = first_port + 7
    issue = (
        "Fix `merge_layers(layers)` in `src/layers.py`. Later `None` values delete a key. "
        "When both old and new values are dictionaries, shallow-merge the nested mapping "
        "instead of replacing it. Do not mutate any input layer."
    )
    old = """    result = {}
    for layer in layers:
        result.update(layer)
    return result"""
    partial = """    result = {}
    for layer in layers:
        for key, value in layer.items():
            if value is None:
                result.pop(key, None)
            else:
                result[key] = value
    return result"""
    new = """    result = {}
    for layer in layers:
        for key, value in layer.items():
            if value is None:
                result.pop(key, None)
            elif isinstance(value, dict) and isinstance(result.get(key), dict):
                merged = dict(result[key])
                merged.update(value)
                result[key] = merged
            else:
                result[key] = value
    return result"""
    layers = [
        {"db": {"host": "alpha", "port": first_port}, "mode": "fast"},
        {"db": {"port": second_port}, "mode": None},
    ]
    expected = {"db": {"host": "alpha", "port": second_port}}
    visible = (
        "from src.layers import merge_layers\n"
        f"layers={layers!r}\n"
        f"assert merge_layers(layers)=={expected!r}\n"
        f"assert layers=={layers!r}\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.layers import merge_layers\n"
        "x=[{'a':{'x':1},'drop':3},{'a':{'y':2},'drop':None}]\n"
        "assert merge_layers(x)=={'a':{'x':1,'y':2}}\n"
        "assert merge_layers([])=={}\n"
        "assert x==[{'a':{'x':1},'drop':3},{'a':{'y':2},'drop':None}]\n"
    )
    return _make_task(task_id=task_id, family="layered_settings", split=split,
                      issue=issue, module="layers", prelude='"""Layered settings."""',
                      signature="merge_layers(layers)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _retry_backoff(seed: int, task_id: str, split: str) -> RepoTask:
    base = 2 + seed % 4
    factor = 2 + seed % 2
    cap = base * factor * factor
    attempts = 5
    expected = [min(cap, base * (factor ** index)) for index in range(attempts)]
    issue = (
        "Fix `retry_delays(base, factor, cap, attempts)` in `src/backoff.py`. The first "
        "delay is `base`, later delays multiply by `factor`, and every value is capped. "
        "Reject base<=0, factor<1, cap<base, or attempts<0 with ValueError."
    )
    old = """    return [min(cap, base * (factor ** (index + 1))) for index in range(attempts)]"""
    partial = """    return [min(cap, base * (factor ** index)) for index in range(attempts)]"""
    new = """    if base <= 0 or factor < 1 or cap < base or attempts < 0:
        raise ValueError("invalid retry policy")
    return [min(cap, base * (factor ** index)) for index in range(attempts)]"""
    visible = (
        "from src.backoff import retry_delays\n"
        f"assert retry_delays({base},{factor},{cap},{attempts})=={expected!r}\n"
        "for args in [(0,2,5,2),(2,0,5,2),(5,2,4,2),(2,2,8,-1)]:\n"
        " try:\n  retry_delays(*args)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.backoff import retry_delays\n"
        "assert retry_delays(3,1,3,4)==[3,3,3,3]\n"
        "assert retry_delays(1,3,9,0)==[]\n"
        "assert retry_delays(1,3,9,5)==[1,3,9,9,9]\n"
        "try:\n retry_delays(0,2,8,2)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return _make_task(task_id=task_id, family="retry_backoff", split=split,
                      issue=issue, module="backoff", prelude='"""Retry timing."""',
                      signature="retry_delays(base, factor, cap, attempts)", old=old,
                      partial=partial, new=new, visible=visible, hidden=hidden)


def _alias_registry(seed: int, task_id: str, split: str) -> RepoTask:
    owner = f"Service {seed % 17}"
    issue = (
        "Fix `build_alias_map(records)` in `src/aliases.py`. Include each primary name and "
        "every alias, normalize arbitrary whitespace and case, and keep the first owner when "
        "normalized aliases collide."
    )
    old = """    result = {}
    for record in records:
        for alias in record.get("aliases", []):
            result[alias] = record["name"]
    return result"""
    partial = """    result = {}
    for record in records:
        for alias in record.get("aliases", []):
            key = " ".join(alias.split()).casefold()
            result[key] = record["name"]
    return result"""
    new = """    result = {}
    for record in records:
        for alias in [record["name"], *record.get("aliases", [])]:
            key = " ".join(alias.split()).casefold()
            result.setdefault(key, record["name"])
    return result"""
    records = [
        {"name": owner, "aliases": [" MAIN   API", "edge"]},
        {"name": "Fallback", "aliases": ["main api", "backup"]},
    ]
    visible = (
        "from src.aliases import build_alias_map\n"
        f"records={records!r}\n"
        "out=build_alias_map(records)\n"
        f"assert out['main api']=={owner!r}\n"
        f"assert out[{owner.casefold()!r}]=={owner!r}\n"
        "assert out['backup']=='Fallback'\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.aliases import build_alias_map\n"
        "rows=[{'name':'First','aliases':[' X ']},{'name':'Second','aliases':['x']}]\n"
        "assert build_alias_map(rows)=={'first':'First','x':'First','second':'Second'}\n"
        "assert build_alias_map([])=={}\n"
    )
    return _make_task(task_id=task_id, family="alias_registry", split=split,
                      issue=issue, module="aliases", prelude='"""Alias indexing."""',
                      signature="build_alias_map(records)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _window_unique(seed: int, task_id: str, split: str) -> RepoTask:
    window = 2 + seed % 3
    values = ["a", "b", "a"] + [f"x{i}" for i in range(window)] + ["a"]
    issue = (
        "Fix `dedupe_recent(values, window)` in `src/recent.py`. Suppress a value only if "
        "its previous occurrence is within the preceding `window` input positions. A value "
        "at exactly that distance is still recent. Preserve order and reject window<0."
    )
    old = """    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result"""
    partial = """    if window < 0:
        raise ValueError("window must be nonnegative")
    last = {}
    result = []
    for index, value in enumerate(values):
        if value not in last or index - last[value] >= window:
            result.append(value)
        last[value] = index
    return result"""
    new = """    if window < 0:
        raise ValueError("window must be nonnegative")
    last = {}
    result = []
    for index, value in enumerate(values):
        if value not in last or index - last[value] > window:
            result.append(value)
        last[value] = index
    return result"""
    visible = (
        "from src.recent import dedupe_recent\n"
        f"assert dedupe_recent(['a','b','a'],2)==['a','b']\n"
        f"assert dedupe_recent(['a','b','a'],1)==['a','b','a']\n"
        f"assert dedupe_recent({values!r},{window})==['a','b']+{[f'x{i}' for i in range(window)]!r}+['a']\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.recent import dedupe_recent\n"
        "assert dedupe_recent(['x','x'],0)==['x','x']\n"
        "assert dedupe_recent(['x','y','x'],2)==['x','y']\n"
        "assert dedupe_recent([],3)==[]\n"
        "try:\n dedupe_recent(['x'],-1)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return _make_task(task_id=task_id, family="window_unique", split=split,
                      issue=issue, module="recent", prelude='"""Windowed de-duplication."""',
                      signature="dedupe_recent(values, window)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _priority_intervals(seed: int, task_id: str, split: str) -> RepoTask:
    offset = seed % 5
    issue = (
        "Fix `paint(spans)` in `src/paint.py`. Each span is `(start,end,label,priority)` "
        "with an exclusive end. At each integer point choose the highest priority; equal "
        "priority ties keep the earliest input span."
    )
    old = """    result = {}
    for start, end, label, priority in spans:
        for point in range(start, end):
            result[point] = label
    return result"""
    partial = """    result = {}
    priorities = {}
    for start, end, label, priority in spans:
        for point in range(start, end):
            if point not in priorities or priority >= priorities[point]:
                priorities[point] = priority
                result[point] = label
    return result"""
    new = """    result = {}
    priorities = {}
    for start, end, label, priority in spans:
        for point in range(start, end):
            if point not in priorities or priority > priorities[point]:
                priorities[point] = priority
                result[point] = label
    return result"""
    spans = [(offset, offset + 4, "first", 2),
             (offset + 1, offset + 3, "lower", 1),
             (offset + 2, offset + 5, "tie", 2)]
    expected = {offset: "first", offset + 1: "first", offset + 2: "first",
                offset + 3: "first", offset + 4: "tie"}
    visible = (
        "from src.paint import paint\n"
        f"assert paint({spans!r})=={expected!r}\n"
        "assert paint([(0,2,'low',1),(1,2,'high',9)])[1]=='high'\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.paint import paint\n"
        "assert paint([(0,1,'a',3),(0,1,'b',3)])=={0:'a'}\n"
        "assert paint([(2,2,'x',5)])=={}\n"
        "assert paint([])=={}\n"
    )
    return _make_task(task_id=task_id, family="priority_intervals", split=split,
                      issue=issue, module="paint", prelude='"""Priority span painting."""',
                      signature="paint(spans)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _stable_topology(seed: int, task_id: str, split: str) -> RepoTask:
    suffix = str(seed % 13)
    issue = (
        "Fix `dependency_waves(dependencies)` in `src/waves.py`. Return stable waves of "
        "currently-ready jobs, retaining mapping order within each wave. Reject unknown "
        "dependencies and cycles with ValueError."
    )
    old = """    return [[name] for name in sorted(dependencies)]"""
    partial = """    remaining = {name: set(needs) for name, needs in dependencies.items()}
    result = []
    done = set()
    while remaining:
        ready = [name for name, needs in remaining.items() if needs <= done]
        if not ready:
            return result
        result.append(ready)
        done.update(ready)
        for name in ready:
            remaining.pop(name)
    return result"""
    new = """    names = set(dependencies)
    if any(set(needs) - names for needs in dependencies.values()):
        raise ValueError("unknown dependency")
    remaining = {name: set(needs) for name, needs in dependencies.items()}
    result = []
    done = set()
    while remaining:
        ready = [name for name, needs in remaining.items() if needs <= done]
        if not ready:
            raise ValueError("dependency cycle")
        result.append(ready)
        done.update(ready)
        for name in ready:
            remaining.pop(name)
    return result"""
    deps = {f"build{suffix}": [], f"lint{suffix}": [],
            f"test{suffix}": [f"build{suffix}"],
            f"ship{suffix}": [f"test{suffix}", f"lint{suffix}"]}
    expected = [[f"build{suffix}", f"lint{suffix}"], [f"test{suffix}"], [f"ship{suffix}"]]
    visible = (
        "from src.waves import dependency_waves\n"
        f"assert dependency_waves({deps!r})=={expected!r}\n"
        "try:\n dependency_waves({'a':['b'],'b':['a']})\n raise AssertionError('missing cycle')\n"
        "except ValueError:\n pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.waves import dependency_waves\n"
        "assert dependency_waves({'b':[],'a':[]})==[['b','a']]\n"
        "try:\n dependency_waves({'a':['missing']})\n raise AssertionError('missing unknown')\n"
        "except ValueError:\n pass\n"
        "assert dependency_waves({})==[]\n"
    )
    return _make_task(task_id=task_id, family="stable_topology", split=split,
                      issue=issue, module="waves", prelude='"""Stable dependency waves."""',
                      signature="dependency_waves(dependencies)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _lease_cache(seed: int, task_id: str, split: str) -> RepoTask:
    now = 100 + seed % 20
    issue = (
        "Fix `select_live(entries, now, limit)` in `src/leases.py`. Expiry is exclusive: "
        "entries with expires_at<=now are dead. Return at most `limit` keys by descending "
        "last_access, retaining input order for ties; reject negative limits."
    )
    old = """    live = [item for item in entries if item["expires_at"] >= now]
    live.sort(key=lambda item: item["last_access"])
    return [item["key"] for item in live[:limit]]"""
    partial = """    if limit < 0:
        raise ValueError("limit must be nonnegative")
    live = [item for item in entries if item["expires_at"] > now]
    live.sort(key=lambda item: item["last_access"])
    return [item["key"] for item in live[:limit]]"""
    new = """    if limit < 0:
        raise ValueError("limit must be nonnegative")
    live = [item for item in entries if item["expires_at"] > now]
    live.sort(key=lambda item: -item["last_access"])
    return [item["key"] for item in live[:limit]]"""
    entries = [
        {"key": "expired", "expires_at": now, "last_access": 999},
        {"key": "fresh", "expires_at": now + 2, "last_access": 8},
        {"key": "older", "expires_at": now + 5, "last_access": 3},
    ]
    visible = (
        "from src.leases import select_live\n"
        f"entries={entries!r}\n"
        f"assert select_live(entries,{now},2)==['fresh','older']\n"
        "try:\n select_live(entries,0,-1)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.leases import select_live\n"
        "x=[{'key':'a','expires_at':5,'last_access':2},{'key':'b','expires_at':5,'last_access':2}]\n"
        "assert select_live(x,4,2)==['a','b']\n"
        "y=[{'key':'old','expires_at':9,'last_access':1},{'key':'new','expires_at':9,'last_access':7}]\n"
        "assert select_live(y,4,2)==['new','old']\n"
        "assert select_live(x,5,2)==[]\n"
        "assert select_live(x,4,0)==[]\n"
    )
    return _make_task(task_id=task_id, family="lease_cache", split=split,
                      issue=issue, module="leases", prelude='"""Lease selection."""',
                      signature="select_live(entries, now, limit)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _quorum_value(seed: int, task_id: str, split: str) -> RepoTask:
    quorum = 2 + seed % 2
    issue = (
        "Fix `choose_quorum(values, quorum)` in `src/quorum.py`. Return None unless some "
        "value appears at least `quorum` times. Among equally frequent qualifying values, "
        "return the one whose first occurrence is earliest. Reject quorum<1."
    )
    old = """    if not values:
        return None
    return max(set(values), key=values.count)"""
    partial = """    if quorum < 1:
        raise ValueError("quorum must be positive")
    if not values:
        return None
    winner = max(sorted(set(values)), key=values.count)
    return winner if values.count(winner) >= quorum else None"""
    new = """    if quorum < 1:
        raise ValueError("quorum must be positive")
    counts = {}
    order = []
    for value in values:
        if value not in counts:
            counts[value] = 0
            order.append(value)
        counts[value] += 1
    if not counts:
        return None
    best = max(counts.values())
    if best < quorum:
        return None
    return next(value for value in order if counts[value] == best)"""
    values = ["zeta", "alpha", "zeta", "alpha"]
    visible = (
        "from src.quorum import choose_quorum\n"
        f"assert choose_quorum({values!r},2)=='zeta'\n"
        f"assert choose_quorum(['x','y'],{quorum}) is None\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.quorum import choose_quorum\n"
        "assert choose_quorum(['b','a','a'],2)=='a'\n"
        "assert choose_quorum(['z','a','z','a'],2)=='z'\n"
        "assert choose_quorum([],1) is None\n"
        "try:\n choose_quorum(['x'],0)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return _make_task(task_id=task_id, family="quorum_value", split=split,
                      issue=issue, module="quorum", prelude='"""Quorum resolution."""',
                      signature="choose_quorum(values, quorum)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _pattern_router(seed: int, task_id: str, split: str) -> RepoTask:
    segment = f"admin{seed % 7}"
    issue = (
        "Fix `best_route(path, routes, default)` in `src/patterns.py`. Routes use shell "
        "wildcards. Among matches choose the longest literal prefix before the first wildcard, "
        "then fewer wildcard characters; remaining ties keep input order."
    )
    old = """    for pattern, target in routes:
        if fnmatchcase(path, pattern):
            return target
    return default"""
    partial = """    matches = [(pattern, target) for pattern, target in routes if fnmatchcase(path, pattern)]
    if not matches:
        return default
    return max(matches, key=lambda item: len(item[0]))[1]"""
    new = """    best = None
    best_score = None
    for pattern, target in routes:
        if not fnmatchcase(path, pattern):
            continue
        wildcard_at = min([index for index, char in enumerate(pattern) if char in "*?["] or [len(pattern)])
        score = (wildcard_at, -sum(pattern.count(char) for char in "*?[") )
        if best_score is None or score > best_score:
            best = target
            best_score = score
    return best if best_score is not None else default"""
    path = f"api/{segment}/reports/daily"
    routes = [("api/*/reports/daily", "wild-long"),
              (f"api/{segment}/*", "specific"),
              ("other/*", "other")]
    visible = (
        "from src.patterns import best_route\n"
        f"routes={routes!r}\n"
        f"assert best_route({path!r},routes,'none')=='specific'\n"
        "assert best_route('missing',routes,'none')=='none'\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.patterns import best_route\n"
        "r=[('a/*','first'),('a/?','second')]\n"
        "assert best_route('a/x',r,'none')=='first'\n"
        "specific=[('a/*/verylong','broad'),('a/foo*','specific')]\n"
        "assert best_route('a/foo/verylong',specific,'none')=='specific'\n"
        "assert best_route('z',[],'none')=='none'\n"
    )
    return _make_task(task_id=task_id, family="pattern_router", split=split,
                      issue=issue, module="patterns",
                      prelude='"""Wildcard route selection."""\n\nfrom fnmatch import fnmatchcase',
                      signature="best_route(path, routes, default=None)", old=old,
                      partial=partial, new=new, visible=visible, hidden=hidden)


def _rate_buckets(seed: int, task_id: str, split: str) -> RepoTask:
    width = 4 + seed % 4
    issue = (
        "Fix `admit(events, limit, width)` in `src/rates.py`. Events are `(time,key)` in "
        "nondecreasing time. Apply an independent sliding window per key. Events exactly "
        "`width` old have expired; return one boolean per event and reject invalid limits."
    )
    old = """    recent = []
    result = []
    for timestamp, key in events:
        recent = [seen for seen in recent if seen > timestamp - width]
        allowed = len(recent) < limit
        if allowed:
            recent.append(timestamp)
        result.append(allowed)
    return result"""
    partial = """    if limit < 1 or width < 1:
        raise ValueError("invalid rate policy")
    recent = {}
    result = []
    for timestamp, key in events:
        bucket = [seen for seen in recent.get(key, []) if seen >= timestamp - width]
        allowed = len(bucket) < limit
        if allowed:
            bucket.append(timestamp)
        recent[key] = bucket
        result.append(allowed)
    return result"""
    new = """    if limit < 1 or width < 1:
        raise ValueError("invalid rate policy")
    recent = {}
    result = []
    for timestamp, key in events:
        bucket = [seen for seen in recent.get(key, []) if seen > timestamp - width]
        allowed = len(bucket) < limit
        if allowed:
            bucket.append(timestamp)
        recent[key] = bucket
        result.append(allowed)
    return result"""
    events = [(0, "a"), (1, "b"), (width, "a"), (width + 1, "a")]
    visible = (
        "from src.rates import admit\n"
        f"assert admit({events!r},1,{width})==[True,True,True,False]\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.rates import admit\n"
        "assert admit([(0,'x'),(0,'x'),(1,'y')],1,5)==[True,False,True]\n"
        "assert admit([(0,'x'),(5,'x')],1,5)==[True,True]\n"
        "for args in [([],0,1),([],1,0)]:\n"
        " try:\n  admit(*args)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
    )
    return _make_task(task_id=task_id, family="rate_buckets", split=split,
                      issue=issue, module="rates", prelude='"""Per-key rate windows."""',
                      signature="admit(events, limit, width)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _deadline_queue(seed: int, task_id: str, split: str) -> RepoTask:
    now = 40 + seed % 20
    limit = 2 + seed % 2
    issue = (
        "Fix `pop_ready(items, now, limit)` in `src/deadlines.py`. Keep only items with "
        "`ready_at<=now`, order by descending priority then ascending ready_at, retain "
        "input order for exact ties, and return at most limit names. Reject negative limit "
        "and do not mutate the input."
    )
    old = """    ready = [item for item in items if item["ready_at"] < now]
    ready.sort(key=lambda item: item["priority"])
    return [item["name"] for item in ready[:limit]]"""
    partial = """    ready = [item for item in items if item["ready_at"] <= now]
    ready.sort(key=lambda item: (-item["priority"], item["ready_at"]))
    return [item["name"] for item in ready[:limit]]"""
    new = """    if limit < 0:
        raise ValueError("limit must be nonnegative")
    ready = [item for item in items if item["ready_at"] <= now]
    ready.sort(key=lambda item: (-item["priority"], item["ready_at"]))
    return [item["name"] for item in ready[:limit]]"""
    items = [
        {"name": "late", "ready_at": now + 1, "priority": 99},
        {"name": "edge", "ready_at": now, "priority": 3},
        {"name": "early", "ready_at": now - 2, "priority": 3},
        {"name": "low", "ready_at": now - 9, "priority": 1},
    ]
    expected = ["early", "edge", "low"][:limit]
    visible = (
        "from src.deadlines import pop_ready\n"
        f"items={items!r}\n"
        f"assert pop_ready(items,{now},{limit})=={expected!r}\n"
        f"assert items=={items!r}\n"
        "try:\n pop_ready(items,0,-1)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.deadlines import pop_ready\n"
        "ties=[{'name':'a','ready_at':1,'priority':2},{'name':'b','ready_at':1,'priority':2}]\n"
        "assert pop_ready(ties,1,2)==['a','b']\n"
        "assert pop_ready(ties,0,2)==[]\n"
        "assert pop_ready(ties,1,0)==[]\n"
        "assert ties==[{'name':'a','ready_at':1,'priority':2},{'name':'b','ready_at':1,'priority':2}]\n"
        "try:\n pop_ready(ties,1,-1)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return _make_task(task_id=task_id, family="deadline_queue", split=split,
                      issue=issue, module="deadlines", prelude='"""Deadline queue."""',
                      signature="pop_ready(items, now, limit)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _interval_coalescer(seed: int, task_id: str, split: str) -> RepoTask:
    gap = 1 + seed % 3
    issue = (
        "Fix `coalesce(spans, gap)` in `src/spans.py`. Spans are `(label,start,end)` with "
        "an exclusive end. Coalesce separately per label when the next start is at most "
        "`gap` after the current end; return a dict of label to merged `(start,end)` lists. "
        "Reject negative gaps or end<start."
    )
    old = """    merged = []
    for label, start, end in spans:
        if merged and start <= merged[-1][1] + gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return {"all": merged}"""
    partial = """    result = {}
    for label, start, end in spans:
        bucket = result.setdefault(label, [])
        if bucket and start < bucket[-1][1] + gap:
            bucket[-1] = (bucket[-1][0], max(bucket[-1][1], end))
        else:
            bucket.append((start, end))
    return result"""
    new = """    if gap < 0:
        raise ValueError("gap must be nonnegative")
    result = {}
    for label, start, end in spans:
        if end < start:
            raise ValueError("span end precedes start")
        bucket = result.setdefault(label, [])
        if bucket and start <= bucket[-1][1] + gap:
            bucket[-1] = (bucket[-1][0], max(bucket[-1][1], end))
        else:
            bucket.append((start, end))
    return result"""
    spans = [("a", 0, 2), ("b", 0, 1), ("a", 2 + gap, 5 + gap)]
    expected = {"a": [(0, 5 + gap)], "b": [(0, 1)]}
    visible = (
        "from src.spans import coalesce\n"
        f"assert coalesce({spans!r},{gap})=={expected!r}\n"
        "for spans,gap in [([('x',3,2)],0),([], -1)]:\n"
        " try:\n  coalesce(spans,gap)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.spans import coalesce\n"
        "assert coalesce([('x',0,1),('x',3,4)],1)=={'x':[(0,1),(3,4)]}\n"
        "assert coalesce([('x',0,2),('x',1,5)],0)=={'x':[(0,5)]}\n"
        "assert coalesce([],0)=={}\n"
        "for spans,gap in [([('x',2,1)],0),([], -1)]:\n"
        " try:\n  coalesce(spans,gap)\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
    )
    return _make_task(task_id=task_id, family="interval_coalescer", split=split,
                      issue=issue, module="spans", prelude='"""Labeled span coalescing."""',
                      signature="coalesce(spans, gap)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _atomic_reservations(seed: int, task_id: str, split: str) -> RepoTask:
    cpu = 3 + seed % 4
    issue = (
        "Fix `admit(requests, capacities)` in `src/reservations.py`. Each request maps "
        "resource names to nonnegative amounts. Accept it only when every named resource "
        "exists and has enough remaining capacity, then subtract all amounts atomically. "
        "Return `(decisions, remaining)` and do not mutate either input."
    )
    old = """    decisions = []
    for request in requests:
        accepted = False
        for name, amount in request.items():
            if capacities.get(name, 0) >= amount:
                capacities[name] -= amount
                accepted = True
        decisions.append(accepted)
    return decisions, capacities"""
    partial = """    decisions = []
    for request in requests:
        accepted = all(capacities.get(name, 0) >= amount for name, amount in request.items())
        if accepted:
            for name, amount in request.items():
                capacities[name] -= amount
        decisions.append(accepted)
    return decisions, capacities"""
    new = """    remaining = dict(capacities)
    decisions = []
    for request in requests:
        if any(amount < 0 for amount in request.values()):
            raise ValueError("negative reservation")
        accepted = all(name in remaining and remaining[name] >= amount
                       for name, amount in request.items())
        if accepted:
            for name, amount in request.items():
                remaining[name] -= amount
        decisions.append(accepted)
    return decisions, remaining"""
    capacities = {"cpu": cpu, "gpu": 1}
    requests = [{"cpu": cpu, "gpu": 2}, {"cpu": cpu}, {"missing": 0}]
    visible = (
        "from src.reservations import admit\n"
        f"caps={capacities!r}\n"
        f"assert admit({requests!r},caps)==([False,True,False],{{'cpu':0,'gpu':1}})\n"
        f"assert caps=={capacities!r}\n"
        "try:\n admit([{'cpu':-1}],caps)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.reservations import admit\n"
        "assert admit([],{'x':2})==([],{ 'x':2})\n"
        "assert admit([{}, {'x':1}],{'x':1})==([True,True],{'x':0})\n"
        "req=[{'x':2},{'x':1}]\n"
        "caps={'x':2}\n"
        "assert admit(req,caps)==([True,False],{'x':0})\n"
        "assert caps=={'x':2}\n"
        "assert req==[{'x':2},{'x':1}]\n"
    )
    return _make_task(task_id=task_id, family="atomic_reservations", split=split,
                      issue=issue, module="reservations",
                      prelude='"""Atomic resource reservations."""',
                      signature="admit(requests, capacities)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


def _fallback_resolver(seed: int, task_id: str, split: str) -> RepoTask:
    suffix = str(seed % 19)
    issue = (
        "Fix `resolve(name, aliases, available)` in `src/fallbacks.py`. Follow aliases "
        "until an available name is reached. Return None for a missing terminal, return an "
        "already-available input unchanged, and raise ValueError on alias cycles."
    )
    old = """    target = aliases.get(name, name)
    return target if target in available else None"""
    partial = """    seen = set()
    current = name
    while current not in seen:
        if current in available:
            return current
        seen.add(current)
        if current not in aliases:
            return None
        current = aliases[current]
    return None"""
    new = """    seen = set()
    current = name
    while True:
        if current in available:
            return current
        if current in seen:
            raise ValueError("alias cycle")
        seen.add(current)
        if current not in aliases:
            return None
        current = aliases[current]"""
    aliases = {f"primary{suffix}": f"backup{suffix}",
               f"backup{suffix}": f"live{suffix}"}
    available = {f"live{suffix}", "direct"}
    visible = (
        "from src.fallbacks import resolve\n"
        f"assert resolve('primary{suffix}',{aliases!r},{available!r})=='live{suffix}'\n"
        f"assert resolve('direct',{aliases!r},{available!r})=='direct'\n"
        "try:\n resolve('a',{'a':'b','b':'a'},set())\n raise AssertionError('missing cycle')\n"
        "except ValueError:\n pass\n"
        "print('VISIBLE_OK')\n"
    )
    hidden = (
        "from src.fallbacks import resolve\n"
        "assert resolve('x',{'x':'missing'},set()) is None\n"
        "assert resolve('x',{},set()) is None\n"
        "assert resolve('x',{'x':'y','y':'z'},{'z'})=='z'\n"
        "try:\n resolve('x',{'x':'y','y':'x'},set())\n raise AssertionError('missing cycle')\n"
        "except ValueError:\n pass\n"
    )
    return _make_task(task_id=task_id, family="fallback_resolver", split=split,
                      issue=issue, module="fallbacks", prelude='"""Fallback aliases."""',
                      signature="resolve(name, aliases, available)", old=old, partial=partial,
                      new=new, visible=visible, hidden=hidden)


BUILDERS = {
    "layered_settings": _layered_settings,
    "retry_backoff": _retry_backoff,
    "alias_registry": _alias_registry,
    "window_unique": _window_unique,
    "priority_intervals": _priority_intervals,
    "stable_topology": _stable_topology,
    "lease_cache": _lease_cache,
    "quorum_value": _quorum_value,
    "pattern_router": _pattern_router,
    "rate_buckets": _rate_buckets,
    "deadline_queue": _deadline_queue,
    "interval_coalescer": _interval_coalescer,
    "atomic_reservations": _atomic_reservations,
    "fallback_resolver": _fallback_resolver,
}

TRAIN_FAMILIES = tuple(list(BUILDERS)[:6])
TRANSFER_FAMILIES = tuple(list(BUILDERS)[6:10])
PROSPECTIVE_FAMILIES = tuple(list(BUILDERS)[10:])


def make_tasks(
    families: list[str] | tuple[str, ...], tasks_per_family: int, seed: int, split: str
) -> list[RepoTask]:
    rng = random.Random(seed)
    result = []
    for family in families:
        if family not in BUILDERS:
            raise KeyError(family)
        for index in range(tasks_per_family):
            item_seed = rng.randrange(1_000_000_000)
            task_id = f"recovery-{split}-{family}-s{seed}-{index:03d}"
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
        self._tmp = tempfile.TemporaryDirectory(prefix="repo_vcrb_")
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
            for path in sorted(self.root.rglob("*"))
            if path.is_file()
            and "__pycache__" not in path.parts
            and (path.suffix in (".py", ".md"))
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
            if not (
                (rel.startswith("src/") and path.suffix == ".py")
                or rel == "README.md"
            ):
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

    def apply_oracle(self) -> None:
        for patch in self.task.oracle_patches:
            result = self.patch(patch.path, patch.old, patch.new)
            if not result.startswith("PATCH_OK"):
                raise AssertionError(result)

    def apply_partial(self) -> None:
        for patch in self.task.partial_patches:
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
