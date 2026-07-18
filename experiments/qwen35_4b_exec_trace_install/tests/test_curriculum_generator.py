"""The exec-trace curriculum's fail-closed truth audit.

Every trace is independently re-derived by REAL CPython (``verify_by_execution``)
and byte-matches the primary interpreter; every final answer is correct by
re-execution; the safety/termination caps abort runaway or unsafe programs; the
difficulty mix is the frozen constant schedule; the banned-vocabulary audit and
the benchmark n-gram overlap are zero; every row is unique.
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import contamination as contam  # noqa: E402
import gen_exec_trace_curriculum as gen  # noqa: E402

CORPUS = EXP / "data" / "sft_exec_trace.jsonl"
RECEIPT = EXP / "data" / "curriculum_receipt.json"


def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestTripleVerification(unittest.TestCase):
    def test_committed_corpus_reexecutes_and_matches(self):
        # Independent real-CPython re-execution of every shipped row: think +
        # answer byte-match, banned vocab zero, uniqueness, receipt sha.
        summary = gen.verify_public_corpus(CORPUS, RECEIPT)
        self.assertEqual(summary["rows"], gen.DEFAULT_ROWS)
        self.assertEqual(summary["kinds"], {"exec_trace": gen.DEFAULT_ROWS})

    def test_primary_and_exec_oracle_agree_on_a_fresh_build(self):
        # Build a small curriculum and confirm the two independent code paths
        # (primary interpreter vs real-CPython settrace) byte-match per program.
        import random

        rng = random.Random(gen.CONSTRUCTION_SEED + 7)
        checked = 0
        for tier in ("short", "medium", "long"):
            made = 0
            while made < 4:
                builder = gen.Builder(rng)
                funcs, body = builder.build(tier)
                source_lines, lineno_to_text = gen.render_program(funcs, body)
                try:
                    steps, output, env = gen.trace_program(funcs, body)
                    v_steps, v_output, v_env = gen.verify_by_execution(source_lines, lineno_to_text)
                except Exception:  # noqa: BLE001 (discarded programs are fine)
                    continue
                self.assertEqual(
                    json.dumps(steps, sort_keys=True), json.dumps(v_steps, sort_keys=True),
                    msg="primary trace disagrees with real-CPython re-execution",
                )
                self.assertEqual(output, v_output)
                self.assertEqual(json.dumps(env, sort_keys=True), json.dumps(v_env, sort_keys=True))
                made += 1
                checked += 1
        self.assertEqual(checked, 12)

    def test_tampered_trace_is_caught_by_reexecution(self):
        rows = load_corpus()
        row = dict(rows[0])
        row["think"] = row["think"].replace("Step 1", "Step 1 (tampered)")
        code = gen.extract_code(row["messages"][0]["content"])
        source_lines = code.split("\n")
        lineno_to_text = {i + 1: line.strip() for i, line in enumerate(source_lines)}
        steps, _output, _env = gen.verify_by_execution(source_lines, lineno_to_text)
        self.assertNotEqual(gen.render_think(steps), row["think"])

    def test_tampered_answer_is_caught_by_reexecution(self):
        rows = load_corpus()
        row = rows[0]
        code = gen.extract_code(row["messages"][0]["content"])
        source_lines = code.split("\n")
        lineno_to_text = {i + 1: line.strip() for i, line in enumerate(source_lines)}
        _steps, output, _env = gen.verify_by_execution(source_lines, lineno_to_text)
        self.assertEqual(f"FINAL: {output}", row["answer"])
        self.assertNotEqual(f"FINAL: {output}X", row["answer"])


class TestKnownProgram(unittest.TestCase):
    def test_a_concrete_program_traces_exactly(self):
        # x = 3; total = 0; for i in range(0, 3): total += i; print(total)
        E = gen.E
        S = gen.S
        body = [
            S("assign", ("total", E("lit", 0))),
            S("for", ("i", E("lit", 0), E("lit", 3)),
              body=[S("aug", ("total", "+", E("name", "i")))]),
            S("print", (E("name", "total"),)),
        ]
        source_lines, lineno_to_text = gen.render_program([], body)
        steps, output, env = gen.trace_program([], body)
        v_steps, v_output, v_env = gen.verify_by_execution(source_lines, lineno_to_text)
        self.assertEqual(steps, v_steps)
        self.assertEqual(output, "3")  # 0+1+2
        self.assertEqual(env, {"total": 3, "i": 2})
        # i=0 leaves total unchanged (0+0) -> that += step is suppressed in BOTH.
        set_steps = [s for s in steps if s["kind"] == "set"]
        self.assertIn({"stmt": "total = 0", "kind": "set", "updates": [["total", "0"]]}, steps)
        self.assertTrue(any(s["stmt"] == "for i in range(0, 3):" for s in set_steps))


class TestSafetyAndTermination(unittest.TestCase):
    def test_step_cap_aborts_runaway_loop(self):
        # while 0 < 1: total += 1  (no counter increment -> infinite; must abort)
        E = gen.E
        S = gen.S
        body = [
            S("assign", ("total", E("lit", 0))),
            S("while", (E("cmp", "<", E("lit", 0), E("lit", 1)),),
              body=[S("aug", ("total", "+", E("lit", 1)))]),
            S("print", (E("name", "total"),)),
        ]
        source_lines, lineno_to_text = gen.render_program([], body)
        with self.assertRaises(gen.StepCapExceeded):
            gen.verify_by_execution(source_lines, lineno_to_text)

    def test_restricted_namespace_blocks_imports_and_io(self):
        for source in ("import os\nprint(1)\n", "open('x')\nprint(1)\n", "print(__import__('os'))\n"):
            with self.assertRaises(Exception):  # noqa: B017 (NameError/ImportError both fine)
                gen.verify_by_execution(source.split("\n"), {})


class TestDifficultyMixAndContamination(unittest.TestCase):
    def test_frozen_constant_difficulty_mix(self):
        rows = load_corpus()
        tiers = {t: 0 for t in ("short", "medium", "long")}
        for row in rows:
            tiers[row["tier"]] += 1
        self.assertEqual(tiers, {"short": 80, "medium": 160, "long": 160})
        self.assertEqual(dict(gen.TIER_SCHEDULE), tiers)

    def test_step_counts_are_ordered_by_tier(self):
        rows = load_corpus()
        by_tier = {"short": [], "medium": [], "long": []}
        for row in rows:
            by_tier[row["tier"]].append(row["n_steps"])
        mean = {t: sum(v) / len(v) for t, v in by_tier.items()}
        self.assertLess(mean["short"], mean["medium"])
        self.assertLess(mean["medium"], mean["long"])
        self.assertGreaterEqual(min(by_tier["short"]), 2)

    def test_banned_vocabulary_is_zero(self):
        banned = contam.banned_names()
        rows = load_corpus()
        for i, row in enumerate(rows):
            blob = row["messages"][0]["content"] + "\n" + row["think"] + "\n" + row["answer"]
            self.assertEqual(contam.whole_word_hits(blob, banned), set(), msg=f"row {i} banned hit")

    def test_identifier_pools_are_contamination_clean(self):
        banned = contam.banned_names()
        pools = (
            gen.SCALAR_NAMES + gen.LIST_NAMES + gen.DICT_NAMES + gen.STR_NAMES
            + gen.FUNC_NAMES + gen.LOOP_VARS + gen.WORD_POOL
        )
        for name in pools:
            self.assertNotIn(name.lower(), banned, msg=f"identifier {name!r} collides with a benchmark name")

    def test_rows_are_unique(self):
        rows = load_corpus()
        prompts = {row["messages"][0]["content"] for row in rows}
        task_ids = {row["task_id"] for row in rows}
        codes = {gen.extract_code(row["messages"][0]["content"]) for row in rows}
        self.assertEqual(len(prompts), len(rows))
        self.assertEqual(len(task_ids), len(rows))
        self.assertEqual(len(codes), len(rows))

    def test_benchmark_ngram_overlap_zero_when_cache_present(self):
        try:
            streams = contam.build_code_tokens_from_cache()
        except Exception as exc:  # noqa: BLE001 (present-only aid)
            self.skipTest(f"HF cache unavailable: {type(exc).__name__}")
        bench = contam.benchmark_ngrams(streams)
        corpus = set()
        for row in load_corpus():
            corpus |= contam.code_ngrams(gen.extract_code(row["messages"][0]["content"]))
        self.assertEqual(contam.distinctive_overlap(corpus, bench), set())


class TestGenerationAndDeterminism(unittest.TestCase):
    def test_small_build_validates_including_per_row_reverify(self):
        rows = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 4), ("medium", 4), ("long", 4)))
        summary = gen.validate_generated(rows, expected_rows=12)
        self.assertEqual(summary["rows"], 12)
        self.assertEqual(summary["unique_codes"], 12)

    def test_construction_is_deterministic(self):
        a = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        b = gen.generate_curriculum(gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)))
        dump = lambda rows: [gen.public_row(r) for r in rows]  # noqa: E731
        self.assertEqual(json.dumps(dump(a), sort_keys=True), json.dumps(dump(b), sort_keys=True))

    def test_retention_flag_blends_code_completion_rows(self):
        rows = gen.generate_curriculum(
            gen.CONSTRUCTION_SEED, (("short", 3), ("medium", 3), ("long", 3)), mix_retention=5
        )
        kinds = {}
        for row in rows:
            kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1
        self.assertEqual(kinds.get("code_retention"), 5)
        self.assertEqual(kinds.get("exec_trace"), 9)
        # retention rows are code-WRITE (answer is code, prompt asks to complete)
        retention = [r for r in rows if r["kind"] == "code_retention"]
        self.assertTrue(all("Complete this Python program" in r["messages"][0]["content"] for r in retention))
        self.assertTrue(all(not r["answer"].startswith("FINAL:") for r in retention))


if __name__ == "__main__":
    unittest.main()
