#!/usr/bin/env python3
"""Contamination fixtures + audits for the WHY scale-ladder curriculum.

Identical audit machinery to the why-comment install cell (this cell SCALES that
curriculum), carried in-cell so the scale-ladder is standalone. The WHY scale
curriculum must look like NOTHING in HumanEval / MBPP even as it grows to tens of
thousands of rows: its executable CODE is drawn from >=50 synthetic function
families with contamination-clean identifiers, and each meaningful line carries a
trailing ``#WHY:`` comment giving the causal reason that line is correct. The eval
grader IGNORES comments, so contamination is judged on the code and the prose
exactly as for the sibling cells. Two independent audits enforce the firewall:

1. BANNED FUNCTION NAMES (the primary, always-on gate). Every function name
   defined by either benchmark — HumanEval ``entry_point`` plus every ``def`` in
   its prompt/canonical solution, and every ``def`` in every MBPP solution — is
   collected into a committed in-cell fixture
   (``data/contamination/banned_function_names.json``). No banned name may appear
   as a whole word anywhere in the corpus (code identifiers, spec prose, OR the
   ``#WHY:`` rationale). Python language keywords and the small set of builtins the
   generator emits are WHITELISTED (a benchmark helper literally named ``sum`` must
   not forbid the Python ``sum`` builtin — contamination means reuse of
   benchmark-SPECIFIC identifiers, not of the Python language). The committed
   fixture makes the gate standalone; a present-only verification aid re-derives it
   from the HF cache and asserts equality.

2. N-GRAM CODE OVERLAP (a present-only verification aid). When the HF cache is
   available, the benchmark solution token stream is cut into 7-grams and the
   corpus's code 7-grams must not intersect it at all on any gram that carries a
   DISTINCTIVE (non-idiom) token. Seven tokens is long enough that ubiquitous
   idioms (``for i in range``) never collide distinctively, so a non-empty
   distinctive intersection means a function reproduced benchmark code. Absent the
   cache the aid is skipped with a recorded note (the banned-name gate and the
   generator's synthetic construction already keep the corpus clean).

This module is byte-for-byte the why-comment cell's contamination audit with only
this docstring adapted; the audit logic, whitelist, and n-gram predicate are
identical so the two corpora are cleaned by the same rules.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
FIXTURE = EXP / "data" / "contamination" / "banned_function_names.json"

# Python keywords + the exact builtins the generator emits. These are language
# tokens, never contamination, and are excluded from the whole-word audit.
PY_KEYWORDS = frozenset(
    "False None True and as assert async await break class continue def del elif "
    "else except finally for from global if import in is lambda nonlocal not or "
    "pass raise return try while with yield".split()
)
PY_BUILTINS_EMITTED = frozenset(
    {
        "abs",
        "bool",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "max",
        "min",
        "print",
        "range",
        "str",
        "sum",
    }
)
LANGUAGE_WHITELIST = PY_KEYWORDS | PY_BUILTINS_EMITTED

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Code tokenizer: identifiers/numbers plus single non-space operator/punct chars.
_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+\.\d+|\d+|[^\sA-Za-z0-9_]")

DEF_RE = re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)")
NGRAM_N = 7

# Generic single-letter loop/parameter variables carry no problem identity;
# an n-gram built only from Python keywords, emitted builtins, operators,
# digits and these is a UNIVERSAL syntax idiom (``for i in range(0,``), never
# contamination. Contamination requires a shared span with a DISTINCTIVE token.
GENERIC_VARS = frozenset("i j k t u m n a b c".split())
_STRUCTURAL = {token.lower() for token in LANGUAGE_WHITELIST} | GENERIC_VARS


def is_distinctive_token(token: str) -> bool:
    lowered = token.lower()
    if lowered in _STRUCTURAL:
        return False
    return bool(token) and (token[0].isalpha() or token[0] == "_") and len(token) >= 2


def gram_has_distinctive(gram: tuple[str, ...]) -> bool:
    return any(is_distinctive_token(token) for token in gram)


# --------------------------------------------------------------------- fixture
def build_names_from_cache() -> set[str]:
    """Re-derive the benchmark function-name set from the HF cache.

    Deterministic and independent of any repo state. Raises if the datasets
    library or the cached datasets are unavailable (callers treat that as
    "skip the present-only aid").
    """
    from datasets import load_dataset  # noqa: PLC0415 (present-only)

    names: set[str] = set()
    human_eval = load_dataset("openai/openai_humaneval", split="test")
    for row in human_eval:
        names.add(row["entry_point"])
        names.update(DEF_RE.findall(row["prompt"] + "\n" + row["canonical_solution"]))
    mbpp = load_dataset("google-research-datasets/mbpp", "full")["test"]
    for row in mbpp:
        names.update(DEF_RE.findall(row.get("code", "")))
    return {name for name in names if name}


def build_code_tokens_from_cache() -> list[list[str]]:
    """Token streams of every benchmark solution (present-only)."""
    from datasets import load_dataset  # noqa: PLC0415

    streams: list[list[str]] = []
    human_eval = load_dataset("openai/openai_humaneval", split="test")
    for row in human_eval:
        streams.append(_TOKEN_RE.findall(row["prompt"] + "\n" + row["canonical_solution"]))
    mbpp = load_dataset("google-research-datasets/mbpp", "full")["test"]
    for row in mbpp:
        code = row.get("code", "")
        if code:
            streams.append(_TOKEN_RE.findall(code))
    return streams


def _sha256_hex(text: str) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_fixture(names: set[str]) -> dict:
    ordered = sorted(names)
    payload = {
        "schema_version": 1,
        "note": (
            "Benchmark function names to keep OUT of the why-scale corpus. "
            "HumanEval entry_points + prompt/canonical_solution defs, and MBPP "
            "(full test) solution defs. Whole-word audit; Python keywords and "
            "emitted builtins are whitelisted in contamination.LANGUAGE_WHITELIST."
        ),
        "source": "openai/openai_humaneval[test] + google-research-datasets/mbpp[full,test]",
        "function_name_count": len(ordered),
        "function_names": ordered,
        "function_names_sha256": _sha256_hex("\n".join(ordered)),
    }
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def load_fixture() -> dict:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    ordered = payload["function_names"]
    if payload.get("function_names_sha256") != _sha256_hex("\n".join(ordered)):
        raise ValueError("banned_function_names.json integrity check failed")
    return payload


def banned_names() -> frozenset[str]:
    """The committed banned set, lowercased, minus the language whitelist."""
    ordered = load_fixture()["function_names"]
    return frozenset(name.lower() for name in ordered) - {
        token.lower() for token in LANGUAGE_WHITELIST
    }


# ----------------------------------------------------------------------- audit
def whole_word_hits(text: str, banned: frozenset[str]) -> set[str]:
    """Whole-word banned-token hits in ``text`` (language tokens excluded)."""
    whitelist = {token.lower() for token in LANGUAGE_WHITELIST}
    tokens = {match.lower() for match in _WORD_RE.findall(text)}
    return (tokens & banned) - whitelist


def code_ngrams(code: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    tokens = _TOKEN_RE.findall(code)
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def benchmark_ngrams(streams: list[list[str]], n: int = NGRAM_N) -> set[tuple[str, ...]]:
    grams: set[tuple[str, ...]] = set()
    for tokens in streams:
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            grams.add(tuple(tokens[i : i + n]))
    return grams


def distinctive_overlap(
    corpus_grams: set[tuple[str, ...]], bench_grams: set[tuple[str, ...]]
) -> set[tuple[str, ...]]:
    """Shared n-grams that carry a distinctive (non-idiom) token — the real
    contamination signal. Pure syntax idioms shared by any two Python programs
    are excluded."""
    return {gram for gram in (corpus_grams & bench_grams) if gram_has_distinctive(gram)}


def main() -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--build", action="store_true", help="rebuild the committed banned-name fixture from the HF cache"
    )
    args = parser.parse_args()
    if args.build:
        payload = write_fixture(build_names_from_cache())
        print(json.dumps({k: v for k, v in payload.items() if k != "function_names"}, indent=2))
        return 0
    payload = load_fixture()
    print(
        json.dumps(
            {
                "function_name_count": payload["function_name_count"],
                "banned_after_whitelist": len(banned_names()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
