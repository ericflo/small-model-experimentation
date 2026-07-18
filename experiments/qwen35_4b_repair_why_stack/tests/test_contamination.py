"""Contamination fixture integrity + the whole-word banned-name audit, plus the
UNION-level re-audit that confirms stacking the two clean parents stays clean.

The committed banned-function-name fixture is self-consistent (sha), the language
whitelist is honoured, and the audit fires on a benchmark name yet never on a
Python builtin. The COMBINED corpus is re-audited: zero whole-word banned-name
hits over all 1008 rows, and (present-only, HF-cache) zero distinctive shared
7-grams between the union's executable CODE (docstrings + comments stripped, the
parents' code-only definition) and the benchmark solution code.
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402

CORPUS = EXP / "data" / "sft_repair_why_stack.jsonl"

_TRIPLE = re.compile(r'"""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'')


def union_rows() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_text(row: dict) -> str:
    parts = [m.get("content", "") for m in row.get("messages", []) if isinstance(m.get("content"), str)]
    for key in ("think", "answer"):
        if isinstance(row.get(key), str):
            parts.append(row[key])
    return "\n".join(parts)


def code_only(answer: str) -> str:
    """Executable code only: drop triple-quoted docstrings and inline # comments
    (the parents' 'code-only n-grams; docstring/rationale prose excluded')."""
    stripped = _TRIPLE.sub("", answer)
    return "\n".join(line.split("#", 1)[0] if "#" in line else line for line in stripped.splitlines())


class TestFixtureIntegrity(unittest.TestCase):
    def test_fixture_loads_and_is_sha_consistent(self):
        payload = contam.load_fixture()
        self.assertEqual(payload["function_name_count"], len(payload["function_names"]))
        self.assertGreater(payload["function_name_count"], 500)

    def test_language_tokens_are_whitelisted_out_of_the_banned_set(self):
        banned = contam.banned_names()
        for token in contam.LANGUAGE_WHITELIST:
            self.assertNotIn(token.lower(), banned)
        # "sum" is BOTH a benchmark def name and a Python builtin -> whitelisted.
        self.assertIn("sum", {n.lower() for n in contam.load_fixture()["function_names"]})
        self.assertNotIn("sum", banned)


class TestWholeWordAudit(unittest.TestCase):
    def test_audit_fires_on_a_benchmark_name(self):
        banned = contam.banned_names()
        sample = sorted(banned)[0]
        text = f"here we use {sample} in a sentence"
        self.assertIn(sample, contam.whole_word_hits(text, banned))

    def test_audit_ignores_python_builtins_and_keywords(self):
        banned = contam.banned_names()
        text = "for i in range(len(seq)): acc = acc + sum(seq) if seq else int(0)"
        self.assertEqual(contam.whole_word_hits(text, banned), set())

    def test_audit_is_whole_word_only(self):
        banned = contam.banned_names()
        sample = sorted(banned)[0]
        text = f"my_{sample}_variable = 1"  # one identifier token -> no whole-word hit
        self.assertNotIn(sample, contam.whole_word_hits(text, banned))


class TestDistinctiveOverlapPredicate(unittest.TestCase):
    def test_structural_only_gram_has_no_distinctive_token(self):
        gram = ("for", "i", "in", "range", "(", "0", ",")
        self.assertFalse(contam.gram_has_distinctive(gram))

    def test_gram_with_real_identifier_is_distinctive(self):
        gram = ("acc", "=", "acc", "+", "seq", "[", "i")
        self.assertTrue(contam.gram_has_distinctive(gram))


class TestUnionContamination(unittest.TestCase):
    def test_union_has_the_expected_1008_rows_by_kind(self):
        rows = union_rows()
        self.assertEqual(len(rows), 1008)
        kinds: dict[str, int] = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        self.assertEqual(kinds, {"self_repair": 504, "why_comment": 504})

    def test_union_banned_name_audit_is_zero(self):
        banned = contam.banned_names()
        hits: dict[str, list[str]] = {}
        for row in union_rows():
            found = contam.whole_word_hits(row_text(row), banned)
            if found:
                hits[row.get("task_id", "?")] = sorted(found)
        self.assertEqual(hits, {}, msg=f"banned-name hits on the union: {hits}")

    def test_union_distinctive_ngram_overlap_is_zero_present_only(self):
        try:
            bench = contam.benchmark_ngrams(contam.build_code_tokens_from_cache())
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        corpus_grams: set[tuple[str, ...]] = set()
        for row in union_rows():
            corpus_grams |= contam.code_ngrams(code_only(row.get("answer", "")))
        distinctive = contam.distinctive_overlap(corpus_grams, bench)
        self.assertEqual(distinctive, set(), msg=f"distinctive shared 7-grams on the union: {sorted(distinctive)[:3]}")


class TestCacheRederivationAid(unittest.TestCase):
    def test_committed_fixture_equals_cache_when_present(self):
        try:
            names = contam.build_names_from_cache()
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        committed = set(contam.load_fixture()["function_names"])
        self.assertEqual(names, committed)


if __name__ == "__main__":
    unittest.main()
