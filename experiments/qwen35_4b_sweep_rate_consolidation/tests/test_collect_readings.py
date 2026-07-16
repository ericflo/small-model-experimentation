import importlib.util
import json
import re
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
FORENSICS_ANALYZER = (
    REPO
    / "experiments"
    / "qwen35_4b_menders_sirens_tier_forensics"
    / "scripts"
    / "analyze_constants.py"
)


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


COLLECT = load_module("consolidation_collect", "collect_readings.py")

FAMILIES_BLOCK_RE = re.compile(r"FAMILIES = \((?:\n    \"[a-z]+\",)+\n\)\n")

# Pinned expectations for the six all-time goal-gate readings.
EXPECTED = {
    78150: {"strict_wins": 8, "ties": ["menders", "rites"], "losses": [], "pass": False},
    78154: {"strict_wins": 10, "ties": [], "losses": [], "pass": True},
    78155: {"strict_wins": 9, "ties": ["menders"], "losses": [], "pass": False},
    78156: {"strict_wins": 8, "ties": ["menders", "warren"], "losses": [], "pass": False},
    78157: {"strict_wins": 10, "ties": [], "losses": [], "pass": True},
    78159: {"strict_wins": 9, "ties": ["menders"], "losses": [], "pass": False},
}

PINNED_SHAS = {
    78150: "a927fc838ca8b1eaa3083d6034ba09ad0659c21a2a13b22c525487cf95a6fb43",
    78154: "6b1a43869f013e24a048a45a04e5603b45fe59488912194eb3e76a43679255fa",
    78155: "482260548d936f6ddd51401328861fd99a67be044f917ffee917348e41b3123b",
    78156: "604b755497a104b3f0337a1c25a36b6996c4c5ccd01ae9ed9e0e9041747fd19a",
    78157: "0ac2c412cc09375446cc1fcee594aedf96bcd7ef9cd6a2214d6b30cf195e0fa3",
    78159: "c83586f0bf1e98cf0e01ebf3918f3d28c98ae8bee7d8f9361dcbcaaf83da8b4d",
}


def spec_for(seed: int) -> dict:
    return next(s for s in COLLECT.READINGS if s["seed"] == seed)


class FamiliesByteIdentityTests(unittest.TestCase):
    def _families_block(self, path: Path) -> str:
        source = path.read_text(encoding="utf-8")
        match = FAMILIES_BLOCK_RE.search(source)
        self.assertIsNotNone(match, f"no FAMILIES literal found in {path}")
        return match.group(0)

    def test_families_byte_identical_to_forensics_analyzer(self) -> None:
        reference = self._families_block(FORENSICS_ANALYZER)
        for script in ("collect_readings.py", "analyze_sweep_rate.py"):
            self.assertEqual(
                self._families_block(EXP / "scripts" / script),
                reference,
                f"{script} FAMILIES literal drifted from the forensics analyzer",
            )

    def test_families_tuple_value(self) -> None:
        self.assertEqual(
            COLLECT.FAMILIES,
            (
                "chronicle",
                "lockpick",
                "menders",
                "mirage",
                "rites",
                "siftstack",
                "sirens",
                "stockade",
                "toolsmith",
                "warren",
            ),
        )


class PinnedReadingTests(unittest.TestCase):
    def test_readings_cover_exactly_the_six_seeds(self) -> None:
        self.assertEqual(
            sorted(s["seed"] for s in COLLECT.READINGS), sorted(EXPECTED)
        )

    def test_pinned_shas(self) -> None:
        for spec in COLLECT.READINGS:
            self.assertEqual(spec["sha256"], PINNED_SHAS[spec["seed"]])

    def test_all_six_readings_match_pinned_expectations(self) -> None:
        for spec in COLLECT.READINGS:
            payload = COLLECT.load_pinned_summary(spec)
            row = COLLECT.compute_reading(spec, payload)
            expected = EXPECTED[spec["seed"]]
            self.assertEqual(row["strict_wins"], expected["strict_wins"], spec["seed"])
            self.assertEqual(row["ties"], expected["ties"], spec["seed"])
            self.assertEqual(row["losses"], expected["losses"], spec["seed"])
            self.assertEqual(row["goal_gate_pass"], expected["pass"], spec["seed"])
            self.assertEqual(
                row["blockers"], sorted(expected["ties"] + expected["losses"])
            )

    def test_seed_78155_warren_won_by_pinned_margin(self) -> None:
        spec = spec_for(78155)
        row = COLLECT.compute_reading(spec, COLLECT.load_pinned_summary(spec))
        self.assertIn("warren", row["wins"])
        self.assertAlmostEqual(row["per_family_delta"]["warren"], 0.2667, places=4)

    def test_seed_78154_cross_checks_against_recorded_goal_gate(self) -> None:
        spec = spec_for(78154)
        row = COLLECT.compute_reading(spec, COLLECT.load_pinned_summary(spec))
        self.assertEqual(row["recorded_goal_gate_cross_check"], "agrees")

    def test_committed_table_matches_fresh_rederivation(self) -> None:
        committed = (EXP / "runs" / "readings_table.json").read_text(encoding="utf-8")
        self.assertEqual(COLLECT.serialize(COLLECT.build_table()), committed)


class ProvenanceDriftTests(unittest.TestCase):
    def _sandbox(self, seed: int) -> tuple[Path, Path, dict]:
        """Copy the cell's data dir into a temp EXP root for tamper tests."""
        scratch = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, scratch, ignore_errors=True)
        exp = scratch / "exp"
        (exp / "data").mkdir(parents=True)
        shutil.copytree(EXP / "data" / "source_summaries", exp / "data" / "source_summaries")
        repo = scratch / "repo"
        repo.mkdir()
        return exp, repo, spec_for(seed)

    def test_missing_local_copy_fails_closed(self) -> None:
        exp, repo, spec = self._sandbox(78150)
        (exp / spec["local"]).unlink()
        with self.assertRaises(SystemExit):
            COLLECT.load_pinned_summary(spec, exp=exp, repo=repo)

    def test_tampered_local_copy_fails_closed(self) -> None:
        exp, repo, spec = self._sandbox(78150)
        target = exp / spec["local"]
        payload = json.loads(target.read_text(encoding="utf-8"))
        payload["scores"]["hygiene_explore"]["per_family"]["menders"] = 1.0
        target.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaises(SystemExit):
            COLLECT.load_pinned_summary(spec, exp=exp, repo=repo)

    def test_drifted_original_fails_closed_even_with_good_local_copy(self) -> None:
        exp, repo, spec = self._sandbox(78150)
        original = repo / spec["original"]
        original.parent.mkdir(parents=True)
        original.write_text("{\"drifted\": true}\n", encoding="utf-8")
        with self.assertRaises(SystemExit):
            COLLECT.load_pinned_summary(spec, exp=exp, repo=repo)

    def test_absent_original_is_allowed_standalone(self) -> None:
        exp, repo, spec = self._sandbox(78150)
        payload = COLLECT.load_pinned_summary(spec, exp=exp, repo=repo)
        self.assertFalse(payload["__original_present__"])
        row = COLLECT.compute_reading(spec, payload)
        self.assertFalse(row["original_present"])
        self.assertEqual(row["strict_wins"], 8)

    def test_disagreeing_recorded_goal_gate_aborts(self) -> None:
        spec = spec_for(78154)
        payload = COLLECT.load_pinned_summary(spec)
        payload["goal_gate"]["per_arm"][spec["arm"]]["strict_wins"] = 9
        with self.assertRaises(SystemExit):
            COLLECT.compute_reading(spec, payload)

    def test_wrong_event_identity_aborts(self) -> None:
        spec = spec_for(78157)
        payload = COLLECT.load_pinned_summary(spec)
        payload["think_budget"] = 2048
        with self.assertRaises(SystemExit):
            COLLECT.compute_reading(spec, payload)


if __name__ == "__main__":
    unittest.main()
