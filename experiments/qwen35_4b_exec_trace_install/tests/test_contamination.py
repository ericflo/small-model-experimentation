"""Contamination fixture integrity + the whole-word banned-name audit.

The committed banned-function-name fixture is self-consistent (sha), the
language whitelist is honoured, and the audit fires on a benchmark name yet
never on a Python builtin. When the HF cache is present a verification aid
re-derives the fixture and asserts equality (superset).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402


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
        text = "for i in range(len(arr)): total += sum(arr) if arr else int(0)"
        self.assertEqual(contam.whole_word_hits(text, banned), set())

    def test_audit_is_whole_word_only(self):
        banned = contam.banned_names()
        # a banned name embedded inside a longer identifier must NOT match
        sample = sorted(banned)[0]
        text = f"my_{sample}_variable = 1"  # substring, not a whole word token... unless underscores split
        hits = contam.whole_word_hits(text, banned)
        # the tokenizer treats my_<sample>_variable as ONE identifier -> no hit
        self.assertNotIn(sample, hits)


class TestDistinctiveOverlapPredicate(unittest.TestCase):
    def test_structural_only_gram_has_no_distinctive_token(self):
        gram = ("for", "i", "in", "range", "(", "0", ",")
        self.assertFalse(contam.gram_has_distinctive(gram))

    def test_gram_with_real_identifier_is_distinctive(self):
        gram = ("tally", "+", "=", "arr", "[", "0", "]")
        self.assertTrue(contam.gram_has_distinctive(gram))


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
