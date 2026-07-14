#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

from gen_curriculum import (  # noqa: E402
    DEFAULT_MIX,
    FAST_MIX,
    SMOKE_MIX,
    generate_curriculum,
    public_row,
    validate_generated,
)


class CurriculumTests(unittest.TestCase):
    def test_smoke_covers_every_skill_and_passes_truth_gates(self) -> None:
        rows = generate_curriculum(SMOKE_MIX, 77001)
        summary = validate_generated(rows)
        self.assertEqual(summary["rows"], 26)
        self.assertEqual(len(summary["kinds"]), 13)
        self.assertTrue(all(row["_audit"]["truth_valid"] for row in rows))

    def test_induction_is_identifiable_and_behaviorally_depth_two(self) -> None:
        rows = generate_curriculum("induct=20", 88103)
        for row in rows:
            audit = row["_audit"]
            self.assertEqual(audit["behavioral_min_depth"], 2)
            self.assertTrue(audit["query_identifiable"])
            self.assertTrue(audit["has_dead_end"])
            self.assertIn(row["answer"].removeprefix("ANSWER: "), row["think"])

    def test_generation_is_byte_deterministic(self) -> None:
        first = generate_curriculum(SMOKE_MIX, 42017)
        second = generate_curriculum(SMOKE_MIX, 42017)
        encode = lambda rows: "".join(
            json.dumps(public_row(row), ensure_ascii=False) + "\n" for row in rows
        )
        self.assertEqual(encode(first), encode(second))

    def test_generation_is_deterministic_across_hash_seeds(self) -> None:
        script = EXP / "scripts" / "gen_curriculum.py"
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / "first.jsonl", Path(directory) / "second.jsonl"]
            for hash_seed, path in zip(("1", "987654"), paths):
                subprocess.run(
                    [sys.executable, str(script), "--smoke", "--seed", "42017", "--out", str(path)],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    env={**os.environ, "PYTHONHASHSEED": hash_seed},
                )
            self.assertEqual(paths[0].read_bytes(), paths[1].read_bytes())

    def test_default_mix_validates(self) -> None:
        rows = generate_curriculum(DEFAULT_MIX, 77001)
        summary = validate_generated(rows)
        self.assertEqual(summary["rows"], 2300)
        self.assertEqual(summary["kinds"]["u_induct"], 300)

    def test_fast_mix_is_frozen_at_800_rows(self) -> None:
        rows = generate_curriculum(FAST_MIX, 77001)
        self.assertEqual(validate_generated(rows)["rows"], 800)


if __name__ == "__main__":
    unittest.main()
