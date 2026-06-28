from __future__ import annotations

import ast
import doctest
import json
import math
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


LETTERS = list("ABCDEFGH")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def parse_entry_function(prompt: str, entry_point: str) -> ast.FunctionDef:
    tree = ast.parse(prompt)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            return node
    raise ValueError(f"entry function {entry_point!r} not found")


def annotation_kind(node: ast.AST | None) -> str:
    if node is None:
        return "unknown"
    if isinstance(node, ast.Name):
        if node.id in {"int", "float", "str", "bool"}:
            return node.id
        if node.id in {"list", "List", "Sequence"}:
            return "list[int]"
        return "unknown"
    if isinstance(node, ast.Subscript):
        root = annotation_kind(node.value)
        sub = annotation_kind(node.slice)
        if root in {"list[int]", "unknown"} and sub in {"int", "float", "str", "bool"}:
            return f"list[{sub}]"
        if isinstance(node.value, ast.Name) and node.value.id in {"List", "list", "Sequence"}:
            return f"list[{sub if sub != 'unknown' else 'int'}]"
    if isinstance(node, ast.Tuple):
        parts = [annotation_kind(elt) for elt in node.elts]
        if parts and all(part in {"int", "float", "str", "bool"} for part in parts):
            return "tuple[" + ",".join(parts) + "]"
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return "unknown"


def signature_kinds(prompt: str, entry_point: str) -> tuple[list[str], str]:
    fn = parse_entry_function(prompt, entry_point)
    args = [annotation_kind(arg.annotation) for arg in fn.args.args]
    ret = annotation_kind(fn.returns)
    return args, ret


def _rand_scalar(kind: str, rng: random.Random) -> Any:
    if kind == "int":
        return rng.choice([-10, -5, -3, -1, 0, 1, 2, 3, 5, 7, 10, 13])
    if kind == "float":
        return rng.choice([-5.0, -2.5, -1.0, 0.0, 0.5, 1.0, 2.0, 3.5, 8.0])
    if kind == "bool":
        return bool(rng.randint(0, 1))
    if kind == "str":
        bank = [
            "",
            "a",
            "b",
            "ab",
            "abc",
            "aba",
            "hello",
            "world",
            "(()())",
            "()()",
            "123",
            "a b",
        ]
        return rng.choice(bank)
    return rng.choice([0, 1, "", [], True])


def sample_value(kind: str, rng: random.Random) -> Any:
    if kind.startswith("list[") and kind.endswith("]"):
        inner = kind[5:-1]
        length = rng.randint(0, 6)
        return [_rand_scalar(inner, rng) for _ in range(length)]
    if kind.startswith("tuple[") and kind.endswith("]"):
        parts = [part for part in kind[6:-1].split(",") if part]
        return tuple(_rand_scalar(part, rng) for part in parts)
    return _rand_scalar(kind, rng)


def safe_repr_signature(status: str, value: str) -> str:
    return f"{status}:{value}"


def extract_doctest_visible_tests(prompt: str, entry_point: str, limit: int) -> list[dict[str, Any]]:
    tests: list[dict[str, Any]] = []
    parser = doctest.DocTestParser()
    try:
        fn = parse_entry_function(prompt, entry_point)
        text = ast.get_docstring(fn) or prompt
    except Exception:
        text = prompt
    try:
        parsed = parser.get_examples(text)
    except ValueError:
        return []
    for example in parsed:
        source = example.source.strip()
        if not source:
            continue
        try:
            expr = ast.parse(source, mode="eval").body
        except SyntaxError:
            continue
        if not isinstance(expr, ast.Call) or not isinstance(expr.func, ast.Name) or expr.func.id != entry_point:
            continue
        if expr.keywords:
            continue
        try:
            args = [ast.literal_eval(arg) for arg in expr.args]
        except Exception:
            continue
        expected = example.want.strip()
        if not expected:
            continue
        tests.append({"args": args, "expected": safe_repr_signature("OK", expected), "source": source})
        if len(tests) >= limit:
            break
    return tests


def execute_candidate(
    code: str,
    entry_point: str,
    tests: list[list[Any]],
    official_test: str,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    script = f"""
import collections
import functools
import itertools
import json
import math
import re
import string
import typing
from typing import *

{code}

TEST_ARGS = {tests!r}

def _safe_call(args):
    try:
        value = {entry_point}(*args)
        return {{"status": "OK", "value": repr(value)}}
    except BaseException as exc:
        return {{"status": "EXC", "value": type(exc).__name__}}

outputs = [_safe_call(tuple(args)) for args in TEST_ARGS]
official_ok = False
try:
{indent_block(official_test, "    ")}
    check({entry_point})
    official_ok = True
except BaseException:
    official_ok = False

print(json.dumps({{"outputs": outputs, "official_ok": official_ok}}))
"""
    with tempfile.TemporaryDirectory(prefix="heval_exec_") as tmp:
        path = Path(tmp) / "candidate.py"
        path.write_text(script, encoding="utf-8")
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(path)],
                cwd=tmp,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "outputs": [safe_repr_signature("TIMEOUT", "Timeout") for _ in tests],
                "official_ok": False,
                "error": "timeout",
            }
    if result.returncode != 0:
        return {
            "outputs": [safe_repr_signature("EXC", "CompileOrRuntimeError") for _ in tests],
            "official_ok": False,
            "error": result.stderr[-800:],
        }
    try:
        payload = json.loads(result.stdout.strip().splitlines()[-1])
    except Exception:
        return {
            "outputs": [safe_repr_signature("EXC", "BadJSON") for _ in tests],
            "official_ok": False,
            "error": result.stdout[-800:] + result.stderr[-800:],
        }
    outputs = [safe_repr_signature(row["status"], row["value"]) for row in payload["outputs"]]
    return {"outputs": outputs, "official_ok": bool(payload.get("official_ok"))}


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


def default_bodies() -> list[str]:
    return [
        "    return None\n",
        "    return 0\n",
        "    return 1\n",
        "    return -1\n",
        "    return False\n",
        "    return True\n",
        "    return []\n",
        "    return \"\"\n",
    ]


def mutate_solution(solution: str) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []
    replacements = [
        ("+", "-"),
        ("-", "+"),
        ("*", "+"),
        ("//", "%"),
        ("%", "+"),
        ("<=", "<"),
        (">=", ">"),
        ("<", "<="),
        (">", ">="),
        ("==", "!="),
        ("!=", "=="),
        (" and ", " or "),
        (" or ", " and "),
        ("True", "False"),
        ("False", "True"),
    ]
    for old, new in replacements:
        if old in solution:
            variants.append((f"replace_{old.strip() or old}_with_{new.strip() or new}", solution.replace(old, new, 1)))
    for match in list(re.finditer(r"\b-?\d+\b", solution))[:12]:
        value = int(match.group(0))
        for delta in (-1, 1):
            mutated = solution[: match.start()] + str(value + delta) + solution[match.end() :]
            variants.append((f"const_{value}_to_{value + delta}", mutated))
    lines = solution.splitlines(keepends=True)
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("return ") and len(stripped) > len("return "):
            replacement = line[: len(line) - len(line.lstrip())] + "return None\n"
            mutated_lines = list(lines)
            mutated_lines[i] = replacement
            variants.append((f"return_none_line_{i}", "".join(mutated_lines)))
            break
    return variants


def make_candidate_codes(
    prompt: str,
    canonical_solution: str,
    max_candidates: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    raw: list[tuple[str, str]] = [("canonical", prompt + canonical_solution)]
    for name, body in mutate_solution(canonical_solution):
        raw.append((name, prompt + body))
    for body in default_bodies():
        raw.append((f"default_{body.strip().replace(' ', '_')}", prompt + body))

    dedup: dict[str, tuple[str, str]] = {}
    for source, code in raw:
        try:
            ast.parse(code)
        except SyntaxError:
            continue
        dedup.setdefault(code, (source, code))

    items = list(dedup.values())
    rng.shuffle(items)
    canonical_code = prompt + canonical_solution
    chosen = items[:max_candidates]
    if canonical_code not in {code for _, code in chosen}:
        canonical_item = ("canonical", canonical_code)
        if len(chosen) >= max_candidates:
            chosen[-1] = canonical_item
        else:
            chosen.append(canonical_item)
    return [
        {
            "candidate_id": f"cand_{i:02d}",
            "source": source,
            "code": code,
        }
        for i, (source, code) in enumerate(chosen)
    ]


def generate_successful_tests(
    prompt: str,
    solution: str,
    entry_point: str,
    official_test: str,
    arg_kinds: list[str],
    total: int,
    seed: int,
    max_attempts: int = 3000,
) -> tuple[list[list[Any]], list[str]]:
    rng = random.Random(seed)
    tests: list[list[Any]] = []
    seen: set[str] = set()
    attempts = 0
    while len(tests) < total and attempts < max_attempts:
        attempts += 1
        args = [sample_value(kind, rng) for kind in arg_kinds]
        key = repr(args)
        if key in seen:
            continue
        seen.add(key)
        tests.append(args)
    if not tests:
        return [], []
    canonical = execute_candidate(prompt + solution, entry_point, tests, official_test, timeout_s=8.0)
    outputs = canonical["outputs"]
    kept_tests: list[list[Any]] = []
    kept_outputs: list[str] = []
    for args, output in zip(tests, outputs):
        if output.startswith("OK:"):
            kept_tests.append(args)
            kept_outputs.append(output)
        if len(kept_tests) >= total:
            break
    return kept_tests, kept_outputs


def build_record(
    raw: dict[str, Any],
    split: str,
    index: int,
    visible_tests: int,
    probe_tests: int,
    hidden_tests: int,
    max_candidates: int,
    seed: int,
) -> tuple[dict[str, Any] | None, str | None]:
    task_id = raw["task_id"]
    prompt = raw["prompt"]
    solution = raw["canonical_solution"]
    entry_point = raw["entry_point"]
    official_test = raw["test"]
    try:
        arg_kinds, return_kind = signature_kinds(prompt, entry_point)
    except Exception as exc:
        return None, f"signature_parse:{type(exc).__name__}"
    if not arg_kinds:
        return None, "no_args"
    public_visible = extract_doctest_visible_tests(prompt, entry_point, visible_tests)
    if len(public_visible) < visible_tests:
        return None, f"insufficient_public_visible:{len(public_visible)}"
    total_tests = probe_tests + hidden_tests
    tests, expected = generate_successful_tests(
        prompt,
        solution,
        entry_point,
        official_test,
        arg_kinds,
        total_tests,
        seed + index * 17,
    )
    if len(tests) < total_tests:
        return None, f"insufficient_successful_tests:{len(tests)}"

    rng = random.Random(seed + index * 101)
    candidates = make_candidate_codes(prompt, solution, max_candidates, rng)
    if len(candidates) < 4:
        return None, f"too_few_candidates:{len(candidates)}"

    all_outputs: list[dict[str, Any]] = []
    for candidate in candidates:
        result = execute_candidate(
            candidate["code"],
            entry_point,
            [test["args"] for test in public_visible] + tests,
            official_test,
        )
        outputs = result["outputs"]
        all_outputs.append(
            {
                **candidate,
                "outputs": {
                    "visible": outputs[:visible_tests],
                    "probe": outputs[visible_tests : visible_tests + probe_tests],
                    "hidden": outputs[visible_tests + probe_tests :],
                },
                "official_pass": bool(result["official_ok"]),
            }
        )

    expected_visible = [test["expected"] for test in public_visible]
    expected_probe = expected[:probe_tests]
    expected_hidden = expected[probe_tests:]
    for candidate in all_outputs:
        candidate["visible_pass"] = [a == b for a, b in zip(candidate["outputs"]["visible"], expected_visible)]
        candidate["hidden_generated_pass"] = all(a == b for a, b in zip(candidate["outputs"]["hidden"], expected_hidden))
        candidate["hidden_correct"] = bool(candidate["official_pass"] and candidate["hidden_generated_pass"])

    if not any(candidate["hidden_correct"] for candidate in all_outputs):
        return None, "no_hidden_correct_candidate"
    if not any(candidate["source"] == "canonical" and candidate["hidden_correct"] for candidate in all_outputs):
        return None, "canonical_not_hidden_correct"

    record = {
        "record_id": f"{split}_{index:04d}_{task_id.replace('/', '_')}",
        "split": split,
        "task_id": task_id,
        "entry_point": entry_point,
        "prompt": prompt,
        "arg_kinds": arg_kinds,
        "return_kind": return_kind,
        "visible_tests": public_visible,
        "probe_tests": [
            {"args": args, "reference_output_for_audit": output}
            for args, output in zip(tests[:probe_tests], expected_probe)
        ],
        "hidden_tests": [
            {"args": args, "expected": output} for args, output in zip(tests[probe_tests:], expected_hidden)
        ],
        "candidates": all_outputs,
        "official_test_present": bool(official_test.strip()),
    }
    return record, None


def candidate_passes_observations(candidate: dict[str, Any], record: dict[str, Any], used_probe_indices: Iterable[int]) -> bool:
    _ = record
    _ = used_probe_indices
    for passed in candidate["visible_pass"]:
        if not passed:
            return False
    return True


def survivors(record: dict[str, Any], used_probe_indices: Iterable[int]) -> list[dict[str, Any]]:
    return [cand for cand in record["candidates"] if candidate_passes_observations(cand, record, used_probe_indices)]


def selected_candidate(record: dict[str, Any], used_probe_indices: Iterable[int]) -> dict[str, Any] | None:
    live = survivors(record, used_probe_indices)
    if not live:
        return None
    clusters = agreement_clusters(live, used_probe_indices)
    best_key, best_members = sorted(
        clusters.items(),
        key=lambda item: (-len(item[1]), min(int(cand["candidate_id"].split("_")[1]) for cand in item[1]), repr(item[0])),
    )[0]
    _ = best_key
    return best_members[0]


def agreement_signature(candidate: dict[str, Any], used_probe_indices: Iterable[int]) -> tuple[str, ...]:
    return tuple(candidate["outputs"]["probe"][idx] for idx in used_probe_indices)


def agreement_clusters(
    live: list[dict[str, Any]],
    used_probe_indices: Iterable[int],
    extra_probe_index: int | None = None,
) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    clusters: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    used = list(used_probe_indices)
    for cand in live:
        parts = [cand["outputs"]["probe"][idx] for idx in used]
        if extra_probe_index is not None:
            parts.append(cand["outputs"]["probe"][extra_probe_index])
        key = tuple(parts)
        clusters.setdefault(key, []).append(cand)
    return clusters


def output_buckets(record: dict[str, Any], live: list[dict[str, Any]], probe_index: int) -> dict[str, int]:
    buckets: dict[str, int] = {}
    for cand in live:
        key = cand["outputs"]["probe"][probe_index]
        buckets[key] = buckets.get(key, 0) + 1
    return buckets


def bucket_stats(buckets: dict[str, int]) -> dict[str, Any]:
    total = sum(buckets.values())
    if total <= 0:
        return {"unique": 0, "largest": 0, "expected_remaining": 0.0, "entropy": 0.0, "top": "{}"}
    probs = [count / total for count in buckets.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    expected_remaining = sum(count * count for count in buckets.values()) / total
    ordered = sorted(buckets.items(), key=lambda item: (-item[1], item[0]))[:5]
    short = "{" + ", ".join(f"{key[:28]}:{count}" for key, count in ordered) + "}"
    return {
        "unique": len(buckets),
        "largest": max(buckets.values()),
        "expected_remaining": float(expected_remaining),
        "entropy": float(entropy),
        "top": short,
    }


def cluster_stats(clusters: dict[tuple[str, ...], list[dict[str, Any]]]) -> dict[str, Any]:
    buckets = {repr(key): len(value) for key, value in clusters.items()}
    return bucket_stats(buckets)


def rank_probes_by_expected_split(record: dict[str, Any], used_probe_indices: Iterable[int]) -> list[dict[str, Any]]:
    used = set(used_probe_indices)
    live = survivors(record, used)
    rows: list[dict[str, Any]] = []
    for idx, test in enumerate(record["probe_tests"]):
        if idx in used:
            continue
        stats = cluster_stats(agreement_clusters(live, used, extra_probe_index=idx))
        rows.append({"probe_index": idx, "test": test, **stats})
    rows.sort(key=lambda row: (row["expected_remaining"], row["largest"], -row["unique"], row["probe_index"]))
    return rows


def greedy_next_probe(record: dict[str, Any], used_probe_indices: Iterable[int]) -> dict[str, Any] | None:
    ranked = rank_probes_by_expected_split(record, used_probe_indices)
    return ranked[0] if ranked else None


def hidden_equivalent_count(record: dict[str, Any], live: list[dict[str, Any]]) -> int:
    return sum(1 for cand in live if cand["hidden_correct"])


def current_state_metrics(record: dict[str, Any], used_probe_indices: list[int]) -> dict[str, Any]:
    live = survivors(record, used_probe_indices)
    clusters = agreement_clusters(live, used_probe_indices)
    selected = selected_candidate(record, used_probe_indices)
    largest_cluster = max((len(members) for members in clusters.values()), default=0)
    return {
        "candidate_count": len(live),
        "agreement_cluster_count": len(clusters),
        "selected_cluster_size": largest_cluster,
        "selected_cluster_fraction": float(largest_cluster / len(live)) if live else 0.0,
        "selected_candidate_id": selected["candidate_id"] if selected else None,
        "selected_source": selected["source"] if selected else None,
        "selected_hidden_correct": bool(selected and selected["hidden_correct"]),
        "hidden_correct_survivors": hidden_equivalent_count(record, live),
        "target_reachable": any(cand["hidden_correct"] for cand in live),
    }


def test_text(test: dict[str, Any]) -> str:
    if "expected" in test:
        return f"{test['args']!r} -> {test['expected']}"
    return f"{test['args']!r}"
