#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import io
import json
import re
import sys
import tokenize
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.coverage_utils import EXPERIMENT  # noqa: E402
from src.jsonl import load_jsonl, write_json, write_jsonl  # noqa: E402


ARM_FILES = [
    "retrieval_adapt_semantic_top3_records.jsonl",
    "retrieval_adapt_random_top3_records.jsonl",
    "retrieval_adapt_shuffled_top3_records.jsonl",
    "retrieval_copy_rename_top3_records.jsonl",
]

PY_KEYWORDS_AND_SYNTAX = {
    "def",
    "return",
    "if",
    "else",
    "elif",
    "for",
    "while",
    "in",
    "not",
    "and",
    "or",
    "True",
    "False",
    "None",
    "(",
    ")",
    "[",
    "]",
    "{",
    "}",
    ":",
    ",",
    ".",
    "=",
    "+",
    "-",
    "*",
    "/",
    "//",
    "%",
    "**",
    "<",
    ">",
    "<=",
    ">=",
    "==",
    "!=",
}


def line_offsets(text: str) -> list[int]:
    offsets = [0]
    total = 0
    for line in text.splitlines(keepends=True):
        total += len(line)
        offsets.append(total)
    return offsets


def char_offset(offsets: list[int], position: tuple[int, int]) -> int:
    line, col = position
    if line <= 0:
        return col
    if line - 1 >= len(offsets):
        return offsets[-1]
    return offsets[line - 1] + col


def code_tokens(code: str) -> list[dict[str, Any]]:
    offsets = line_offsets(code)
    rows: list[dict[str, Any]] = []
    try:
        stream = io.StringIO(code).readline
        for tok in tokenize.generate_tokens(stream):
            if tok.type in {tokenize.ENCODING, tokenize.ENDMARKER, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT}:
                continue
            start = char_offset(offsets, tok.start)
            end = char_offset(offsets, tok.end)
            if start >= end:
                continue
            rows.append(
                {
                    "text": tok.string,
                    "type": tokenize.tok_name.get(tok.type, str(tok.type)),
                    "start": start,
                    "end": end,
                }
            )
    except tokenize.TokenError:
        return []
    return rows


def compact_text(text: str, limit: int = 160) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def norm_tokens(text: str) -> set[str]:
    return {item.lower() for item in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+|==|!=|<=|>=|//|\*\*|[<>%+*/-]", text)}


def content_tokens(text: str) -> set[str]:
    return {token for token in norm_tokens(text) if token not in {item.lower() for item in PY_KEYWORDS_AND_SYNTAX} and len(token) > 1}


def source_rank(source: str) -> int | None:
    match = re.search(r"_r(\d+)", source or "")
    if not match:
        return None
    return int(match.group(1))


def algorithm_for_source(plan_row: dict[str, Any], source: str, fallback: str = "semantic") -> dict[str, Any] | None:
    rank = source_rank(source)
    key = "semantic"
    if "_random_" in source:
        key = "random"
    elif "_shuffled_" in source:
        key = "shuffled"
    elif "_semantic_" in source:
        key = "semantic"
    rows = plan_row.get(key) or plan_row.get(fallback) or []
    if rank is None or rank >= len(rows):
        rank = 0
    if not rows:
        return None
    return rows[rank]["algorithm"]


def public_tests(record: dict[str, Any]) -> list[str]:
    return [case["assert_src"] for case in record.get("public_cases", [])]


def task_only_prompt(record: dict[str, Any]) -> str:
    return f"""Return only Python code. Do not use markdown.

Target task:
{record['task_text']}

Define a function named `{record['entry_point']}` and any helpers needed.

Public tests:
{chr(10).join(public_tests(record))}
"""


def retrieved_hint_prompt(record: dict[str, Any], algorithm: dict[str, Any] | None, label: str) -> str:
    if algorithm is None:
        return task_only_prompt(record)
    return f"""Return only Python code. Do not use markdown.

Target task:
{record['task_text']}

Define a function named `{record['entry_point']}` and any helpers needed.

Public tests:
{chr(10).join(public_tests(record))}

Privileged hint ({label}):
The following verified algorithm solves a related task. Use its algorithmic idea only if relevant.
Related task: {algorithm.get('task_text', '')}
Related function: {algorithm.get('entry_point', '')}
Related public tests:
{chr(10).join(algorithm.get('public_tests', []))}
Related code:
{algorithm.get('code', '')}
"""


def reference_hint_prompt(record: dict[str, Any]) -> str:
    return f"""Return only Python code. Do not use markdown.

Target task:
{record['task_text']}

Define a function named `{record['entry_point']}` and any helpers needed.

Public tests:
{chr(10).join(public_tests(record))}

Leakage ceiling hint:
The reference solution is shown below. This is not deployable and is used only to measure an upper-bound pressure signal.
Reference code:
{record.get('reference_code', '')}
"""


def branch_text(code: str, toks: list[dict[str, Any]], start: int, end: int, max_tokens: int) -> tuple[str, int, int]:
    end = min(end, start + max_tokens)
    if start >= end:
        return "", 0, 0
    lo = toks[start]["start"]
    hi = toks[end - 1]["end"]
    return code[lo:hi], lo, hi


def is_docstring_or_comment_branch(toks: list[dict[str, Any]], start: int, end: int, text: str) -> bool:
    if start >= end:
        return False
    first = toks[start]
    if first["type"] == "STRING" and (text.strip().startswith(('"""', "'''")) or len(text) > 80):
        return True
    return False


def classify_branch(correct_branch: str, wrong_branch: str, hint_code: str) -> str:
    branch_content = content_tokens(correct_branch + " " + wrong_branch)
    hint_content = content_tokens(hint_code)
    if branch_content and branch_content & hint_content:
        return "hint_overlap"
    return "task_specific"


def token_bucket(text: str, in_diff: bool, candidate_kind: str) -> str:
    if in_diff:
        return f"discriminating_{candidate_kind}"
    if text in PY_KEYWORDS_AND_SYNTAX:
        return "shared_parse_format"
    if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", text) or re.fullmatch(r"\d+", text):
        return "shared_content"
    return "shared_other"


def diff_forks(
    correct_code: str,
    wrong_code: str,
    hint_code: str,
    max_forks: int,
    max_branch_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    ctoks = code_tokens(correct_code)
    wtoks = code_tokens(wrong_code)
    sm = difflib.SequenceMatcher(a=[tok["text"] for tok in ctoks], b=[tok["text"] for tok in wtoks], autojunk=False)
    forks: list[dict[str, Any]] = []
    correct_diff_idxs: set[int] = set()
    wrong_diff_idxs: set[int] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            continue
        correct_diff_idxs.update(range(i1, i2))
        wrong_diff_idxs.update(range(j1, j2))
        if len(forks) >= max_forks or i1 >= i2 or j1 >= j2:
            continue
        cbranch, clo, chi = branch_text(correct_code, ctoks, i1, i2, max_branch_tokens)
        wbranch, wlo, whi = branch_text(wrong_code, wtoks, j1, j2, max_branch_tokens)
        if not cbranch.strip() or not wbranch.strip():
            continue
        if is_docstring_or_comment_branch(ctoks, i1, min(i2, i1 + max_branch_tokens), cbranch):
            continue
        if is_docstring_or_comment_branch(wtoks, j1, min(j2, j1 + max_branch_tokens), wbranch):
            continue
        prefix = correct_code[:clo]
        forks.append(
            {
                "fork_index": len(forks),
                "opcode": tag,
                "correct_token_span": [i1, min(i2, i1 + max_branch_tokens)],
                "wrong_token_span": [j1, min(j2, j1 + max_branch_tokens)],
                "prefix_text": prefix,
                "prefix_tail": compact_text(prefix[-220:]),
                "correct_branch": cbranch,
                "wrong_branch": wbranch,
                "correct_branch_preview": compact_text(cbranch),
                "wrong_branch_preview": compact_text(wbranch),
                "stratum": classify_branch(cbranch, wbranch, hint_code),
                "correct_branch_tokens": [tok["text"] for tok in ctoks[i1 : min(i2, i1 + max_branch_tokens)]],
                "wrong_branch_tokens": [tok["text"] for tok in wtoks[j1 : min(j2, j1 + max_branch_tokens)]],
            }
        )
    correct_map = {
        "tokens": [
            {
                **tok,
                "bucket": token_bucket(tok["text"], idx in correct_diff_idxs, "correct"),
            }
            for idx, tok in enumerate(ctoks)
        ]
    }
    wrong_map = {
        "tokens": [
            {
                **tok,
                "bucket": token_bucket(tok["text"], idx in wrong_diff_idxs, "wrong"),
            }
            for idx, tok in enumerate(wtoks)
        ]
    }
    return forks, correct_map, wrong_map


def minimal_candidate(candidate: dict[str, Any], arm: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "candidate_id": candidate.get("candidate_id"),
        "source": candidate.get("source"),
        "parse_status": candidate.get("parse_status"),
        "visible_all_pass": candidate.get("visible_all_pass"),
        "full_pass": candidate.get("full_pass"),
        "code": candidate.get("code", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--max-bad-per-correct", type=int, default=8)
    parser.add_argument("--max-forks-per-pair", type=int, default=8)
    parser.add_argument("--max-branch-tokens", type=int, default=8)
    args = parser.parse_args()

    plan_rows = {row["record"]["record_id"]: row for row in load_jsonl(args.data_dir / "retrieval_plan.jsonl")}
    by_record: dict[str, dict[str, Any]] = {}
    for filename in ARM_FILES:
        arm = filename.replace("_records.jsonl", "")
        for row in load_jsonl(args.data_dir / filename):
            bundle = by_record.setdefault(row["record_id"], {"record": {key: value for key, value in row.items() if key != "candidates"}, "correct": [], "wrong": []})
            for candidate in row.get("candidates", []):
                mini = minimal_candidate(candidate, arm)
                if candidate.get("full_pass") and arm == "retrieval_adapt_semantic_top3":
                    bundle["correct"].append(mini)
                elif candidate.get("visible_all_pass") and not candidate.get("full_pass"):
                    bundle["wrong"].append(mini)

    pairs: list[dict[str, Any]] = []
    for record_id, bundle in sorted(by_record.items(), key=lambda item: int(item[1]["record"]["task_id"])):
        if not bundle["correct"] or not bundle["wrong"]:
            continue
        plan_row = plan_rows.get(record_id)
        if plan_row is None:
            continue
        record = bundle["record"]
        for correct_index, correct in enumerate(bundle["correct"]):
            correct_alg = algorithm_for_source(plan_row, correct.get("source", ""), fallback="semantic")
            shuffled_alg = (plan_row.get("shuffled") or [{}])[0].get("algorithm") if plan_row.get("shuffled") else None
            hint_code = correct_alg.get("code", "") if correct_alg else ""
            for wrong_index, wrong in enumerate(bundle["wrong"][: args.max_bad_per_correct]):
                forks, correct_map, wrong_map = diff_forks(
                    correct["code"],
                    wrong["code"],
                    hint_code,
                    max_forks=args.max_forks_per_pair,
                    max_branch_tokens=args.max_branch_tokens,
                )
                if not forks:
                    continue
                pair_id = f"pair_{len(pairs):04d}"
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "record_id": record_id,
                        "task_id": record["task_id"],
                        "task_text": record["task_text"],
                        "entry_point": record["entry_point"],
                        "public_tests": public_tests(record),
                        "reference_code": record.get("reference_code", ""),
                        "correct": correct,
                        "wrong": wrong,
                        "correct_algorithm": correct_alg,
                        "shuffled_algorithm": shuffled_alg,
                        "prompts": {
                            "student_no_hint": task_only_prompt(record),
                            "weak_retrieved": retrieved_hint_prompt(record, correct_alg, "retrieved_algorithm_for_correct_candidate"),
                            "shuffled_retrieved": retrieved_hint_prompt(record, shuffled_alg, "shuffled_retrieval_control"),
                            "full_reference": reference_hint_prompt(record),
                        },
                        "forks": forks,
                        "token_maps": {
                            "correct": correct_map,
                            "wrong": wrong_map,
                        },
                        "metadata": {
                            "correct_index": correct_index,
                            "wrong_index": wrong_index,
                            "max_forks_per_pair": args.max_forks_per_pair,
                            "max_branch_tokens": args.max_branch_tokens,
                        },
                    }
                )

    fork_strata: dict[str, int] = {}
    for pair in pairs:
        for fork in pair["forks"]:
            fork_strata[fork["stratum"]] = fork_strata.get(fork["stratum"], 0) + 1
    summary = {
        "experiment": EXPERIMENT,
        "matched_pairs": len(pairs),
        "tasks": sorted({pair["task_id"] for pair in pairs}),
        "forks": sum(len(pair["forks"]) for pair in pairs),
        "fork_strata": fork_strata,
        "source_arms": ARM_FILES,
        "path": str(args.out),
    }
    write_jsonl(args.out, pairs)
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
