"""Fresh counterfactual repositories for active specification acquisition.

Each inferred dyad has byte-identical issue text, source, tree/path names, and
non-discriminating files.  Exactly one designated public evidence file differs
between the two members and specifies opposed edge-case behavior.  Hidden
executables, branch labels, and oracle/partial patches remain host-side.
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


EVIDENCE_CHANNELS = ("tests", "docs", "callsite")
PATH_SKINS = {
    "bank": {
        "tests": (
            "tests/spec_examples.py",
            "tests/compat_examples.py",
            "tests/edge_examples.py",
        ),
        "docs": (
            "docs/behavior.md",
            "docs/compatibility.md",
            "docs/edge_cases.md",
        ),
        "callsite": (
            "src/client.py",
            "src/adapter.py",
            "src/consumer.py",
        ),
    },
    "qualification": {
        "tests": (
            "tests/contract_checks.py",
            "tests/acceptance_cases.py",
            "tests/public_examples.py",
        ),
        "docs": (
            "docs/contract.md",
            "docs/semantics.md",
            "docs/reference.md",
        ),
        "callsite": (
            "src/integration.py",
            "src/downstream.py",
            "src/gateway.py",
        ),
    },
    "transfer": {
        "tests": (
            "tests/interface_cases.py",
            "tests/compatibility_cases.py",
            "tests/edge_contracts.py",
        ),
        "docs": (
            "docs/interface.md",
            "docs/compatibility_contract.md",
            "docs/public_policy.md",
        ),
        "callsite": (
            "src/plugin.py",
            "src/service.py",
            "src/bridge.py",
        ),
    },
}
TRAIN_FAMILIES = (
    "spec_lookup",
    "spec_tags",
    "spec_duplicates",
    "spec_score",
)
TRANSFER_FAMILIES = (
    "spec_switch",
    "spec_update",
)
ALL_FAMILIES = TRAIN_FAMILIES + TRANSFER_FAMILIES


@dataclass(frozen=True)
class Patch:
    path: str
    old: str
    new: str


@dataclass(frozen=True)
class Blueprint:
    module: str
    function: str
    signature: str
    issue: str
    old: str
    implementations: tuple[str, str]
    normal_test: str
    direct_edge_tests: tuple[str, str]
    docs_edge_test: str
    clients: tuple[str, str]
    hidden_tests: tuple[str, str]
    policy_descriptions: tuple[str, str]


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
    pair_id: str
    branch: int
    evidence_channel: str
    evidence_path: str
    evidence_path_regime: str
    acquisition_query_skin: str
    acquisition_query: str
    evidence_marker: str
    explicit_contract: bool

    def public_manifest(self) -> dict:
        return {
            "task_id": self.task_id,
            "family": self.family,
            "split": self.split,
            "difficulty": self.difficulty,
            "pair_id": self.pair_id,
            "evidence_channel": self.evidence_channel,
            "evidence_path": self.evidence_path,
            "evidence_path_regime": self.evidence_path_regime,
            "acquisition_query_skin": self.acquisition_query_skin,
            "explicit_contract": self.explicit_contract,
            "issue_sha256": hashlib.sha256(self.issue.encode()).hexdigest(),
            "file_sha256": {
                path: hashlib.sha256(text.encode()).hexdigest()
                for path, text in sorted(self.files.items())
            },
        }


def content_digest(task: RepoTask) -> str:
    payload = {"issue": task.issue, "files": task.files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def pair_static_digest(task: RepoTask) -> str:
    """Fingerprint every public byte except the designated discriminator."""
    files = dict(task.files)
    files[task.evidence_path] = "<COUNTERFACTUAL_DISCRIMINATOR>"
    payload = {"issue": task.issue, "files": files}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _lookup(seed: int) -> Blueprint:
    key = f"known_{seed % 997}"
    missing = f"missing_{(seed // 7) % 991}"
    module = "lookup"
    function = "fetch_value"
    old = """    raise NotImplementedError("implement repository contract")"""
    a = """    if key not in values:
        raise KeyError(key)
    return values[key]"""
    b = """    return values.get(key)"""
    normal = f"values={{{key!r}:17}}\nassert {function}(values,{key!r})==17\n"
    direct_a = (
        f"try:\n {function}({{}},{missing!r})\n raise AssertionError('missing KeyError')\n"
        "except KeyError:\n pass\n"
    )
    direct_b = f"assert {function}({{}},{missing!r}) is None\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        "if 'EDGE: a missing key raises KeyError' in mode:\n"
        f" try:\n  {function}({{}},{missing!r})\n  raise AssertionError('missing KeyError')\n"
        " except KeyError:\n  pass\n"
        "elif 'EDGE: a missing key returns None' in mode:\n"
        f" assert {function}({{}},{missing!r}) is None\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    try:\n        {function}({{}},{missing!r})\n"
        "    except KeyError:\n        return True\n    return False\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({{}},{missing!r}) is None\n"
    )
    hidden_a = normal + direct_a + f"assert {function}({{'x':0}},'x')==0\n"
    hidden_b = normal + direct_b + f"assert {function}({{'x':0}},'x')==0\n"
    return Blueprint(
        module, function, "fetch_value(values, key)",
        "Implement `fetch_value(values, key)`. Present keys return their mapped value. "
        "Match this repository's established behavior for missing keys.",
        old, (a, b), normal, (direct_a, direct_b), docs,
        (client_a, client_b), (hidden_a, hidden_b),
        ("a missing key raises KeyError", "a missing key returns None"),
    )


def _tags(seed: int) -> Blueprint:
    blank = " " * (1 + seed % 3)
    module = "tags"
    function = "clean_tags"
    old = """    raise NotImplementedError("implement repository contract")"""
    a = """    result = []
    for tag in tags:
        normalized = tag.strip().casefold()
        if not normalized:
            raise ValueError("blank tag")
        result.append(normalized)
    return result"""
    b = """    result = []
    for tag in tags:
        normalized = tag.strip().casefold()
        if normalized:
            result.append(normalized)
    return result"""
    normal = f"assert {function}([' Alpha ','BETA'])==['alpha','beta']\n"
    direct_a = (
        f"try:\n {function}([{blank!r}])\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    direct_b = f"assert {function}(['x',{blank!r},' Y '])==['x','y']\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        "if 'EDGE: blank tags raise ValueError' in mode:\n"
        f" try:\n  {function}([{blank!r}])\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
        "elif 'EDGE: blank tags are skipped' in mode:\n"
        f" assert {function}(['x',{blank!r}])==['x']\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    try:\n        {function}([{blank!r}])\n"
        "    except ValueError:\n        return True\n    return False\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}(['x',{blank!r}]) == ['x']\n"
    )
    return Blueprint(
        module, function, "clean_tags(tags)",
        "Implement `clean_tags(tags)`. Strip and case-fold ordinary tags while preserving "
        "order. Match this repository's established behavior for blank tags.",
        old, (a, b), normal, (direct_a, direct_b), docs,
        (client_a, client_b), (normal + direct_a, normal + direct_b),
        ("blank tags raise ValueError", "blank tags are skipped"),
    )


def _duplicates(seed: int) -> Blueprint:
    key = f"k{seed % 101}"
    module = "pairs"
    function = "index_pairs"
    old = """    raise NotImplementedError("implement repository contract")"""
    a = """    result = {}
    for key, value in pairs:
        result.setdefault(key, value)
    return result"""
    b = """    result = {}
    for key, value in pairs:
        result[key] = value
    return result"""
    normal = f"assert {function}([('a',1),('b',2)])=={{'a':1,'b':2}}\n"
    direct_a = f"assert {function}([({key!r},1),({key!r},2)])[{key!r}]==1\n"
    direct_b = f"assert {function}([({key!r},1),({key!r},2)])[{key!r}]==2\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        f"value={function}([({key!r},1),({key!r},2)])[{key!r}]\n"
        "if 'EDGE: duplicate keys keep the first value' in mode:\n assert value==1\n"
        "elif 'EDGE: duplicate keys keep the last value' in mode:\n assert value==2\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}([({key!r},1),({key!r},2)])[{key!r}] == 1\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}([({key!r},1),({key!r},2)])[{key!r}] == 2\n"
    )
    return Blueprint(
        module, function, "index_pairs(pairs)",
        "Implement `index_pairs(pairs)`. Return a mapping for ordinary key/value pairs. "
        "Match this repository's established behavior when a key occurs twice.",
        old, (a, b), normal, (direct_a, direct_b), docs,
        (client_a, client_b), (normal + direct_a, normal + direct_b),
        ("duplicate keys keep the first value", "duplicate keys keep the last value"),
    )


def _score(seed: int) -> Blueprint:
    low = 2 + seed % 4
    high = low + 5 + seed % 3
    outside = high + 4
    module = "scores"
    function = "bound_score"
    old = """    raise NotImplementedError("implement repository contract")"""
    a = """    if low > high:
        raise ValueError("invalid bounds")
    return min(high, max(low, value))"""
    b = """    if low > high:
        raise ValueError("invalid bounds")
    return value if low <= value <= high else None"""
    normal = f"assert {function}({low + 1},{low},{high})=={low + 1}\n"
    direct_a = f"assert {function}({outside},{low},{high})=={high}\n"
    direct_b = f"assert {function}({outside},{low},{high}) is None\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        f"value={function}({outside},{low},{high})\n"
        f"if 'EDGE: outside scores are clamped to the nearest bound' in mode:\n assert value=={high}\n"
        "elif 'EDGE: outside scores return None' in mode:\n assert value is None\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({outside},{low},{high}) == {high}\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({outside},{low},{high}) is None\n"
    )
    bounds = (
        f"try:\n {function}(1,3,2)\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    return Blueprint(
        module, function, "bound_score(value, low, high)",
        "Implement `bound_score(value, low, high)`. In-range values are returned unchanged "
        "and low>high raises ValueError. Match established behavior outside the range.",
        old, (a, b), normal + bounds, (direct_a, direct_b), docs,
        (client_a, client_b), (normal + bounds + direct_a, normal + bounds + direct_b),
        ("outside scores are clamped to the nearest bound", "outside scores return None"),
    )


def _switch(seed: int) -> Blueprint:
    unknown = f"maybe_{seed % 997}"
    module = "switches"
    function = "parse_switch"
    old = """    raise NotImplementedError("implement repository contract")"""
    common = """    normalized = value.strip().casefold()
    if normalized in {"yes", "on", "true"}:
        return True
    if normalized in {"no", "off", "false"}:
        return False"""
    a = common + """
    raise ValueError("unknown switch")"""
    b = common + """
    return False"""
    normal = f"assert {function}(' YES ') is True\nassert {function}('off') is False\n"
    direct_a = (
        f"try:\n {function}({unknown!r})\n raise AssertionError('missing ValueError')\n"
        "except ValueError:\n pass\n"
    )
    direct_b = f"assert {function}({unknown!r}) is False\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        "if 'EDGE: unknown switches raise ValueError' in mode:\n"
        f" try:\n  {function}({unknown!r})\n  raise AssertionError('missing ValueError')\n"
        " except ValueError:\n  pass\n"
        "elif 'EDGE: unknown switches default to False' in mode:\n"
        f" assert {function}({unknown!r}) is False\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    try:\n        {function}({unknown!r})\n"
        "    except ValueError:\n        return True\n    return False\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({unknown!r}) is False\n"
    )
    return Blueprint(
        module, function, "parse_switch(value)",
        "Implement `parse_switch(value)`. Accept common true and false spellings after "
        "trimming and case-folding. Match established behavior for unknown spellings.",
        old, (a, b), normal, (direct_a, direct_b), docs,
        (client_a, client_b), (normal + direct_a, normal + direct_b),
        ("unknown switches raise ValueError", "unknown switches default to False"),
    )


def _update(seed: int) -> Blueprint:
    key = f"slot_{seed % 997}"
    module = "updates"
    function = "set_entry"
    old = """    raise NotImplementedError("implement repository contract")"""
    a = """    result = dict(values)
    result[key] = value
    return result"""
    b = """    if key in values:
        return None
    result = dict(values)
    result[key] = value
    return result"""
    normal = f"base={{'other':1}}\nassert {function}(base,{key!r},7)=={{'other':1,{key!r}:7}}\nassert base=={{'other':1}}\n"
    direct_a = f"assert {function}({{{key!r}:1}},{key!r},2)=={{{key!r}:2}}\n"
    direct_b = f"assert {function}({{{key!r}:1}},{key!r},2) is None\n"
    docs = (
        "mode=open('docs/behavior.md').read()\n"
        f"value={function}({{{key!r}:1}},{key!r},2)\n"
        f"if 'EDGE: existing entries are overwritten' in mode:\n assert value=={{{key!r}:2}}\n"
        "elif 'EDGE: existing entries are rejected with None' in mode:\n assert value is None\n"
        "else:\n raise AssertionError('unknown public contract')\n"
    )
    client_a = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({{{key!r}:1}},{key!r},2) == {{{key!r}:2}}\n"
    )
    client_b = (
        f"from src.{module} import {function}\n\ndef edge_contract_ok():\n"
        f"    return {function}({{{key!r}:1}},{key!r},2) is None\n"
    )
    return Blueprint(
        module, function, "set_entry(values, key, value)",
        "Implement `set_entry(values, key, value)`. New keys are added to a copied mapping "
        "without mutating the input. Match established behavior for an existing key.",
        old, (a, b), normal, (direct_a, direct_b), docs,
        (client_a, client_b), (normal + direct_a, normal + direct_b),
        ("existing entries are overwritten", "existing entries are rejected with None"),
    )


BUILDERS = {
    "spec_lookup": _lookup,
    "spec_tags": _tags,
    "spec_duplicates": _duplicates,
    "spec_score": _score,
    "spec_switch": _switch,
    "spec_update": _update,
}


def _visible_test(
    blueprint: Blueprint,
    branch: int,
    channel: str,
    evidence_path: str,
) -> str:
    prefix = f"from src.{blueprint.module} import {blueprint.function}\n"
    if channel == "tests":
        edge = blueprint.direct_edge_tests[branch]
    elif channel == "docs":
        edge = blueprint.docs_edge_test.replace("docs/behavior.md", evidence_path)
    elif channel == "callsite":
        module = evidence_path.removeprefix("src/").removesuffix(".py").replace("/", ".")
        edge = f"from src.{module} import edge_contract_ok\nassert edge_contract_ok()\n"
    else:
        raise ValueError(channel)
    return prefix + blueprint.normal_test + edge + "print('VISIBLE_OK')\n"


def _files(
    issue: str,
    blueprint: Blueprint,
    branch: int,
    channel: str,
    reference: str,
    evidence_path: str,
) -> tuple[dict[str, str], str]:
    marker = (
        f"CONTRACT {reference} {blueprint.function} [{blueprint.signature}]: "
        f"{blueprint.policy_descriptions[branch]}"
    )
    docs = (
        f"# Compatibility behavior\n\n{marker}.\nEDGE: {blueprint.policy_descriptions[branch]}.\n"
        if channel == "docs"
        else "# Maintenance note\n\nOrdinary inputs follow the issue description.\n"
    )
    client = (
        f"# {marker}\n" + blueprint.clients[branch]
        if channel == "callsite"
        else "def format_result(value):\n    return f'result={value!r}'\n"
    )
    body = (
        f'"""Implementation target for compatibility reference {reference}."""\n\n'
        f"def {blueprint.signature}:\n{blueprint.old}\n"
    )
    visible = _visible_test(blueprint, branch, channel, evidence_path)
    files = {
        "README.md": f"# Cedar compatibility task\n\n{issue}\n",
        "src/__init__.py": "",
        f"src/{blueprint.module}.py": body,
        "src/client.py": "def format_result(value):\n    return f'result={value!r}'\n",
        "src/constants.py": (
            '"""Unrelated defaults; not every constant defines behavior."""\n'
            f"REFERENCE = {reference!r}\nDEFAULT_MODE = 'stable'\n"
        ),
        "src/models.py": "def copy_mapping(value):\n    return dict(value)\n",
        "docs/behavior.md": "# Maintenance note\n\nOrdinary behavior follows the issue.\n",
        "docs/notes.md": "# Notes\n\nKeep ordinary behavior deterministic.\n",
        "tests/test_visible.py": (
            f"exec(compile(open({evidence_path!r}).read(), {evidence_path!r}, 'exec'))\n"
            if channel == "tests"
            else visible
        ),
        "tests/test_smoke.py": "# Unrelated smoke placeholder.\n",
    }
    if channel == "tests":
        files[evidence_path] = f"# {marker}\n" + visible
    elif channel == "docs":
        files[evidence_path] = docs
    else:
        files[evidence_path] = client
    return files, marker


def make_task(
    family: str,
    seed: int,
    split: str,
    pair_index: int,
    branch: int,
    channel: str,
    evidence_path: str,
    evidence_path_regime: str,
    acquisition_query_skin: str,
    *,
    explicit_contract: bool,
) -> RepoTask:
    if family not in BUILDERS:
        raise KeyError(family)
    if branch not in (0, 1):
        raise ValueError(branch)
    if channel not in EVIDENCE_CHANNELS:
        raise ValueError(channel)
    blueprint = BUILDERS[family](seed)
    reference = f"R{seed % 1_000_000:06d}"
    issue = f"{blueprint.issue} Compatibility reference: {reference}."
    if explicit_contract:
        issue += f" Explicit edge policy: {blueprint.policy_descriptions[branch]}."
    files, evidence_marker = _files(
        issue, blueprint, branch, channel, reference, evidence_path
    )
    acquisition_query = {
        "reference": reference,
        "symbol": blueprint.function,
        "signature": blueprint.signature,
    }.get(acquisition_query_skin)
    if acquisition_query is None:
        raise ValueError(f"unknown acquisition query skin: {acquisition_query_skin}")
    path = f"src/{blueprint.module}.py"
    pair_kind = "explicit" if explicit_contract else "inferred"
    pair_id = f"{split}-{family}-{pair_kind}-s{seed}-p{pair_index:03d}"
    task_id = f"spec-{pair_id}-b{branch}"
    oracle = blueprint.implementations[branch]
    partial = blueprint.implementations[1 - branch]
    return RepoTask(
        task_id=task_id,
        family=family,
        split=split,
        issue=issue,
        files=files,
        hidden_test=(
            f"from src.{blueprint.module} import {blueprint.function}\n"
            + blueprint.hidden_tests[branch]
        ),
        oracle_patches=(Patch(path, blueprint.old, oracle),),
        partial_patches=(Patch(path, blueprint.old, partial),),
        difficulty=4,
        pair_id=pair_id,
        branch=branch,
        evidence_channel=channel,
        evidence_path=evidence_path,
        evidence_path_regime=evidence_path_regime,
        acquisition_query_skin=acquisition_query_skin,
        acquisition_query=acquisition_query,
        evidence_marker=evidence_marker,
        explicit_contract=explicit_contract,
    )


def make_pairs(
    families: list[str] | tuple[str, ...],
    pairs_per_family: int,
    seed: int,
    split: str,
    *,
    explicit_contract: bool = False,
    path_regime: str | None = None,
) -> list[RepoTask]:
    if path_regime is None:
        path_regime = (
            "bank" if "bank" in split
            else "transfer" if split.startswith("transfer_")
            else "qualification"
        )
    if path_regime not in PATH_SKINS:
        raise ValueError(f"unknown evidence path regime: {path_regime}")
    rng = random.Random(seed)
    tasks: list[RepoTask] = []
    for family_index, family in enumerate(families):
        for pair_index in range(pairs_per_family):
            item_seed = rng.randrange(1_000_000_000)
            channel = EVIDENCE_CHANNELS[(family_index + pair_index + seed) % 3]
            skins = PATH_SKINS[path_regime][channel]
            evidence_path = skins[(family_index * 5 + pair_index + seed) % len(skins)]
            acquisition_query_skin = (
                "signature"
                if path_regime == "transfer"
                else ("reference", "symbol")[(family_index + pair_index + seed) % 2]
            )
            tasks.extend(
                make_task(
                    family,
                    item_seed,
                    split,
                    pair_index,
                    branch,
                    channel,
                    evidence_path,
                    path_regime,
                    acquisition_query_skin,
                    explicit_contract=explicit_contract,
                )
                for branch in (0, 1)
            )
    return tasks


def make_tasks(
    families: list[str] | tuple[str, ...],
    tasks_per_family: int,
    seed: int,
    split: str,
) -> list[RepoTask]:
    if tasks_per_family % 2:
        raise ValueError("counterfactual tasks_per_family must be even")
    return make_pairs(families, tasks_per_family // 2, seed, split)


def _limit_child() -> None:
    resource.setrlimit(resource.RLIMIT_CPU, (4, 4))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (2 * 1024 * 1024, 2 * 1024 * 1024))


class RepoEnv:
    """Constrained real filesystem plus public and host-private executables."""

    def __init__(self, task: RepoTask):
        self.task = task
        self._tmp = tempfile.TemporaryDirectory(prefix="repo_ceac_")
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
            and path.suffix in (".py", ".md")
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
            if not path.is_file() or path.suffix not in (".py", ".md"):
                continue
            rel = path.relative_to(self.root).as_posix()
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
        return self._format_result(self._visible_process())

    def visible_pass(self) -> bool:
        return self._visible_process().returncode == 0

    def hidden_pass(self) -> bool:
        return self._run([sys.executable, "-c", self.task.hidden_test]).returncode == 0

    def score_workspace(self) -> tuple[bool, bool]:
        """Host-only score used for first-patch analysis; never enters a prompt."""
        return self.visible_pass(), self.hidden_pass()

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
        allowed = (
            relative.startswith("src/")
            or relative.startswith("docs/")
            or relative == "README.md"
        )
        if allow_visible_tests:
            allowed = allowed or (
                relative.startswith("tests/") and relative.endswith(".py")
            )
        if not allowed:
            raise ValueError("path is outside the allowed repository surface")
        return candidate

    def _visible_process(self) -> subprocess.CompletedProcess:
        return self._run([
            sys.executable,
            "-c",
            "exec(compile(open('tests/test_visible.py').read(), "
            "'tests/test_visible.py', 'exec'))",
        ])

    def _run(self, command: list[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            cwd=self.root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=6,
            check=False,
            preexec_fn=_limit_child,
        )

    @staticmethod
    def _format_result(result: subprocess.CompletedProcess) -> str:
        status = "PASS" if result.returncode == 0 else f"FAIL(exit={result.returncode})"
        output = (result.stdout + "\n" + result.stderr).strip()[-6000:]
        return f"{status}\n{output}".strip()


def manifest_digest(tasks: list[RepoTask]) -> str:
    payload = json.dumps([task.public_manifest() for task in tasks], sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()
