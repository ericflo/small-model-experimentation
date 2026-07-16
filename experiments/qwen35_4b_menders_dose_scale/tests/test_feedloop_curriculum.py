import json
import random
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
REFERENCE_EXP = ROOT / "experiments" / "qwen35_4b_feedback_loop_state_chain_install"
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
sys.path.insert(0, str(EXP / "scripts"))

import gen_curriculum as original  # noqa: E402
import gen_feedloop_curriculum as feedloop  # noqa: E402


APPLY = {
    "troughline": feedloop._trough_apply,
    "trinketcord": feedloop._cord_apply,
    "crankwheel": feedloop._wheel_apply,
    "sigilslate": feedloop._slate_apply,
    "barrowyoke": feedloop._yoke_apply,
    "balesled": feedloop._sled_apply,
    "millround": feedloop._round_apply,
    "skeinreel": feedloop._reel_apply,
}
DESCRIBE = {
    "troughline": feedloop._trough_describe,
    "trinketcord": feedloop._cord_describe,
    "crankwheel": feedloop._wheel_describe,
    "sigilslate": feedloop._slate_describe,
    "barrowyoke": feedloop._yoke_describe,
    "balesled": feedloop._sled_describe,
    "millround": feedloop._round_describe,
    "skeinreel": feedloop._reel_describe,
}


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def run_ops(apply_fn, ops, start):
    state = start
    for op in ops:
        state = apply_fn(tuple(op), state)
    return state


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.SMOKE_MIX, 12345)
        summary = feedloop.validate_generated(rows)
        feedloop.check_banned_vocabulary(rows)
        self.assertEqual(summary["rows"], 16)
        self.assertEqual(set(summary["kinds"]), {"u_feedloop"})
        self.assertEqual(
            set(summary["surfaces"]), set(feedloop.FEEDLOOP_FORMALISMS)
        )

    def test_eight_formalisms_split_reused_and_new(self) -> None:
        self.assertEqual(len(feedloop.FEEDLOOP_FORMALISMS), 8)
        self.assertEqual(
            set(feedloop.FEEDLOOP_FORMALISMS),
            set(feedloop.REUSED_FORMALISMS) | set(feedloop.NEW_FORMALISMS),
        )
        self.assertEqual(
            feedloop.REUSED_FORMALISMS,
            ("troughline", "trinketcord", "crankwheel", "sigilslate"),
        )
        self.assertEqual(
            feedloop.NEW_FORMALISMS,
            ("barrowyoke", "balesled", "millround", "skeinreel"),
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.SMOKE_MIX, 54321)
        poisoned = json.loads(json.dumps(feedloop.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " the warren gate"
        with self.assertRaises(ValueError):
            feedloop.check_banned_vocabulary([poisoned])

    def test_description_noun_ban_actually_fires(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.SMOKE_MIX, 54321)
        for leak in ("a rerun of the steps", "the protocol", "one flag"):
            poisoned = json.loads(json.dumps(feedloop.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError):
                feedloop.check_banned_vocabulary([poisoned])

    def test_statechain_surface_ban_actually_fires(self) -> None:
        """The statechain cells' surfaces are banned in THIS cell."""
        rows = feedloop.generate_curriculum(feedloop.SMOKE_MIX, 54321)
        for leak in (
            "the brewvat murmured",
            "a courierloft chimed",
            "the peatstove crackled",
            "a muletrack call",
            "three drams in",
            "the mule stood",
            "one clod of peat",
            "unhitch at the post",
        ):
            poisoned = json.loads(json.dumps(feedloop.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError):
                feedloop.check_banned_vocabulary([poisoned])

    def test_retained_feedloop_surfaces_are_not_banned(self) -> None:
        banned = set(feedloop.BANNED_PROMPT_TOKENS)
        self.assertFalse(set(feedloop.INHERITED_SURFACE_TOKENS) & banned)
        self.assertFalse(set(feedloop.FRESH_SURFACE_TOKENS) & banned)
        self.assertFalse(
            set(feedloop.FRESH_SURFACE_TOKENS)
            & set(feedloop.INHERITED_SURFACE_TOKENS)
        )
        # The reused formalism names stay usable.
        for retained in ("troughline", "trinketcord", "crankwheel", "sigilslate"):
            self.assertIn(retained, feedloop.INHERITED_SURFACE_TOKENS)
            self.assertNotIn(retained, banned)
        # The statechain formalisms are banned, never inherited.
        for dead in ("brewvat", "courierloft", "peatstove", "muletrack"):
            self.assertIn(dead, banned)
            self.assertNotIn(dead, feedloop.INHERITED_SURFACE_TOKENS)

    def test_fresh_vocabulary_disjoint_from_predecessor_pools(self) -> None:
        original_tokens = {
            item for pool in original.SURFACE_POOLS.values() for item in pool
        }
        fresh = set(feedloop.FRESH_SURFACE_TOKENS)
        self.assertFalse(fresh & original_tokens)
        # The reused feedloop pools are inherited, never claimed fresh.
        for reused in ("brann", "plome", "drasp", "morv", "trough", "sigil"):
            self.assertIn(reused, feedloop.INHERITED_SURFACE_TOKENS)
            self.assertNotIn(reused, fresh)

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.ARM_MIX, 77150)
        regenerated = "".join(
            json.dumps(feedloop.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_feedloop_scale.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_corpus_balance_bounds(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.ARM_MIX, 77150)
        balance = feedloop.check_corpus_balance(rows)
        self.assertEqual(
            balance["feedloop_formalisms"],
            {formalism: 100 for formalism in feedloop.FEEDLOOP_FORMALISMS},
        )
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(kinds, {"u_feedloop": 800})
        # Every row kept the >=2 round-1 ambiguity invariant.
        ambiguity = balance["feedloop_round1_candidate_counts"]
        self.assertTrue(all(int(key) >= 2 for key in ambiguity))

    def test_holdout_mix_yields_five_rows_per_formalism(self) -> None:
        rows = feedloop.generate_curriculum(feedloop.HOLDOUT_MIX, 88037)
        summary = feedloop.validate_generated(rows)
        self.assertEqual(summary["rows"], 40)
        self.assertEqual(
            summary["surfaces"],
            {formalism: 5 for formalism in feedloop.FEEDLOOP_FORMALISMS},
        )

    def test_zero_row_overlap_with_reference_and_statechain_sources(self) -> None:
        """Reused formalism instances are FRESH, not copies."""
        local = {
            message_bytes(row)
            for row in feedloop.generate_curriculum(feedloop.ARM_MIX, 77150)
        } | {
            message_bytes(row)
            for row in feedloop.generate_curriculum(feedloop.HOLDOUT_MIX, 88037)
        }
        for exp_dir, name in (
            (REFERENCE_EXP, "sft_feedloop_state.jsonl"),
            (REFERENCE_EXP, "local_tasks_seed88026.jsonl"),
            (STATECHAIN_EXP, "sft_statechain_only.jsonl"),
        ):
            reference_rows = [
                json.loads(line)
                for line in (exp_dir / "data" / name)
                .read_text(encoding="utf-8")
                .splitlines()
                if line
            ]
            reference = {message_bytes(row) for row in reference_rows}
            self.assertFalse(local & reference, name)


class EpisodeRederivationTests(unittest.TestCase):
    """Re-execute every formalism's episodes from the audit spec."""

    def rederive(self, formalism: str, seed: int, count: int = 6) -> None:
        rng = random.Random(seed)
        apply_fn = APPLY[formalism]
        describe = DESCRIBE[formalism]
        for _ in range(count):
            row = feedloop.feedloop_lesson(rng, formalism)
            audit = row["_audit"]
            spec = audit["spec"]
            written = [tuple(op) for op in spec["written"]]
            bug_at, true_op = spec["true_fix"][0], tuple(spec["true_fix"][1])
            wrong_at, wrong_op = spec["wrong_fix"][0], tuple(spec["wrong_fix"][1])
            start_a, start_b = spec["start_a"], spec["start_b"]
            # The written sequence lands on the recorded failing outcome.
            self.assertEqual(run_ops(apply_fn, written, start_a), spec["finished_a"])
            self.assertNotEqual(spec["finished_a"], spec["wanted_a"])
            # The true fix lands on the wanted outcome in BOTH trials.
            patched = list(written)
            patched[bug_at] = true_op
            self.assertEqual(run_ops(apply_fn, patched, start_a), spec["wanted_a"])
            self.assertEqual(run_ops(apply_fn, patched, start_b), spec["wanted_b"])
            # The wrong attempt squares with trial one but fails trial two,
            # landing exactly on the recorded second-trial evidence.
            attempted = list(written)
            attempted[wrong_at] = wrong_op
            self.assertEqual(run_ops(apply_fn, attempted, start_a), spec["wanted_a"])
            self.assertEqual(
                run_ops(apply_fn, attempted, start_b),
                spec["finished_b_after_wrong"],
            )
            self.assertNotEqual(spec["finished_b_after_wrong"], spec["wanted_b"])
            # Ambiguity and uniqueness invariants.
            self.assertGreaterEqual(audit["candidates_after_round1"], 2)
            self.assertTrue(audit["unique_after_round2"])
            self.assertTrue(audit["wrong_in_round1"])
            # The answer names the true fix.
            self.assertEqual(
                answer_of(row), f"STEP {bug_at + 1}: {describe(true_op)}"
            )
            # Legality clauses appear verbatim in the rendered prompt and the
            # extended-grammar audit found only out-of-bound alternatives.
            prompt = prompt_of(row)
            self.assertTrue(audit["legality_clauses"])
            for clause in audit["legality_clauses"]:
                self.assertIn(clause, prompt)
            extended = audit["extended_uniqueness_audit"]
            self.assertTrue(extended["out_of_bound_only"])
            self.assertTrue(extended["legality_clauses_documented"])
            self.assertEqual(
                extended["amounts_probed_to"], feedloop.EXTENDED_AMOUNT_BOUND
            )
            self.assertGreaterEqual(extended["round2_survivors_extended"], 1)
            # The wrong attempt is described in the prompt (the loop lesson).
            self.assertIn(audit["wrong_attempt"], prompt)

    def test_troughline(self) -> None:
        self.rederive("troughline", 101)

    def test_trinketcord(self) -> None:
        self.rederive("trinketcord", 202)

    def test_crankwheel(self) -> None:
        self.rederive("crankwheel", 303)

    def test_sigilslate(self) -> None:
        self.rederive("sigilslate", 404)

    def test_barrowyoke(self) -> None:
        self.rederive("barrowyoke", 505)

    def test_balesled(self) -> None:
        self.rederive("balesled", 606)

    def test_millround(self) -> None:
        self.rederive("millround", 707)

    def test_skeinreel(self) -> None:
        self.rederive("skeinreel", 808)


class NewFormalismGrammarTests(unittest.TestCase):
    def test_extended_grammars_cover_bounded_grammars(self) -> None:
        rng = random.Random(99)
        for formalism in feedloop.FEEDLOOP_FORMALISMS:
            machine = feedloop._feedloop_machine(rng, formalism)
            bounded = set(machine["grammar"])
            extended = set(machine["extended_grammar"])
            self.assertTrue(bounded <= extended, formalism)
            self.assertGreater(len(extended), len(bounded), formalism)
            for clause in machine["legality_clauses"]:
                self.assertIn(clause, machine["rules"], formalism)

    def test_container_dimension_is_probed_for_named_container_machines(self) -> None:
        """Minor-3 fix: troughline/barrowyoke extended grammars probe the
        container dimension over the FULL pool, with a tolerant probe apply
        under which no phantom-container op can reproduce a wanted state."""
        rng = random.Random(299)
        for formalism, pool, container_of in (
            ("troughline", feedloop.TROUGHS, lambda op: op[2] if op[0] in ("ladle", "tipover") else op[1]),
            ("barrowyoke", feedloop.BARROWS, lambda op: op[2] if op[0] in ("heave", "swaploads") else op[1]),
        ):
            machine = feedloop._feedloop_machine(rng, formalism)
            instance = set(machine["vocabulary"])
            phantoms = set(pool) - instance
            self.assertEqual(len(phantoms), 3, formalism)
            extended_containers = {
                container_of(op) for op in machine["extended_grammar"]
            }
            self.assertTrue(
                phantoms <= extended_containers,
                f"{formalism} extended grammar never probes phantom containers",
            )
            # The probe apply agrees with the bounded apply on bounded ops...
            probe = machine["extended_apply"]
            state = machine["start_a"]
            for op in machine["grammar"]:
                self.assertEqual(
                    probe(op, state), machine["apply"](op, state), (formalism, op)
                )
            # ...and any op touching a phantom container leaves its key
            # behind, so it can never equal a wanted state over the instance
            # containers alone.
            phantom = sorted(phantoms)[0]
            for op in machine["extended_grammar"]:
                touched = [part for part in op[1:] if isinstance(part, str)]
                if phantom not in touched:
                    continue
                result = probe(op, state)
                self.assertIn(phantom, result, (formalism, op))
                self.assertNotEqual(set(result), set(state), (formalism, op))

    def test_probe_scope_recorded_and_validated_per_formalism(self) -> None:
        self.assertEqual(
            set(feedloop.EXTENDED_PROBE_SCOPE), set(feedloop.FEEDLOOP_FORMALISMS)
        )
        for formalism in ("troughline", "barrowyoke"):
            self.assertIn("container", feedloop.EXTENDED_PROBE_SCOPE[formalism])
        # sigilslate's slot indices are declared structural, not overclaimed.
        self.assertIn("NOT probed", feedloop.EXTENDED_PROBE_SCOPE["sigilslate"])
        rng = random.Random(399)
        row = feedloop.feedloop_lesson(rng, "troughline")
        row["task_id"] = "mds_feedloop_00000"
        extended = row["_audit"]["extended_uniqueness_audit"]
        self.assertEqual(
            extended["probe_scope"], feedloop.EXTENDED_PROBE_SCOPE["troughline"]
        )
        self.assertTrue(extended["container_names_probed_over_full_pool"])
        feedloop.validate_generated([row])
        # A row claiming a foreign probe scope must be rejected.
        broken = json.loads(json.dumps(feedloop.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["extended_uniqueness_audit"]["probe_scope"] = (
            feedloop.EXTENDED_PROBE_SCOPE["skeinreel"]
        )
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])
        # A named-container row claiming no container probe must be rejected.
        broken = json.loads(json.dumps(feedloop.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["extended_uniqueness_audit"][
            "container_names_probed_over_full_pool"
        ] = False
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_new_formalism_parameter_bounds(self) -> None:
        rng = random.Random(199)
        machine = feedloop._feedloop_machine(rng, "barrowyoke")
        heaves = [op[1] for op in machine["grammar"] if op[0] == "heave"]
        self.assertEqual(sorted(set(heaves)), [1, 2, 3, 4])
        machine = feedloop._feedloop_machine(rng, "millround")
        turns = [op[1] for op in machine["grammar"] if op[0] == "turnround"]
        self.assertEqual(sorted(set(turns)), [1, 2, 3, 4])
        machine = feedloop._feedloop_machine(rng, "skeinreel")
        winds = [op[1] for op in machine["grammar"] if op[0] == "windon"]
        self.assertEqual(sorted(set(winds)), [1, 2, 3, 4, 5])
        machine = feedloop._feedloop_machine(rng, "balesled")
        lashes = {op[1] for op in machine["grammar"] if op[0] == "lash"}
        self.assertEqual(len(lashes), 4)
        self.assertTrue(lashes <= set(feedloop.BALES))

    def test_new_machine_semantics(self) -> None:
        # barrowyoke: heave adds, swap trades, dump zeros.
        state = {"askel": 3, "brumm": 5, "grond": 1}
        state = feedloop._yoke_apply(("heave", 4, "askel"), state)
        self.assertEqual(state["askel"], 7)
        state = feedloop._yoke_apply(("swaploads", "askel", "grond"), state)
        self.assertEqual((state["askel"], state["grond"]), (1, 7))
        state = feedloop._yoke_apply(("dumpout", "brumm"), state)
        self.assertEqual(state["brumm"], 0)
        # balesled: lash back, shove front, uncouple front, walk around.
        sled = ("kimm", "jelve")
        self.assertEqual(feedloop._sled_apply(("lash", "lorsk"), sled), ("kimm", "jelve", "lorsk"))
        self.assertEqual(feedloop._sled_apply(("shove", "lorsk"), sled), ("lorsk", "kimm", "jelve"))
        self.assertEqual(feedloop._sled_apply(("uncouple",), sled), ("jelve",))
        self.assertEqual(feedloop._sled_apply(("walkround",), sled), ("jelve", "kimm"))
        # millround: mod-6 ring, creel adds position+1, unwind resets.
        self.assertEqual(feedloop._round_apply(("turnround", 4), (4, 0)), (2, 0))
        self.assertEqual(feedloop._round_apply(("emptyvane",), (3, 1)), (3, 5))
        self.assertEqual(feedloop._round_apply(("unwind",), (5, 2)), (0, 2))
        # skeinreel: wind adds, letout halves down, crosslay toggles.
        self.assertEqual(feedloop._reel_apply(("windon", 5), (2, "murled")), (7, "murled"))
        self.assertEqual(feedloop._reel_apply(("letout",), (7, "ferren")), (3, "ferren"))
        self.assertEqual(feedloop._reel_apply(("crosslay",), (7, "ferren")), (7, "murled"))


class ValidatorRejectionTests(unittest.TestCase):
    def make_row(self) -> dict:
        rng = random.Random(4242)
        row = feedloop.feedloop_lesson(rng, "barrowyoke")
        row["task_id"] = "mds_feedloop_00000"
        return row

    def test_validator_accepts_a_clean_row(self) -> None:
        feedloop.validate_generated([self.make_row()])

    def test_validator_rejects_single_candidate_round1(self) -> None:
        row = self.make_row()
        broken = dict(row)
        broken["_audit"] = dict(row["_audit"])
        broken["_audit"]["candidates_after_round1"] = 1
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_validator_rejects_clause_missing_from_prompt(self) -> None:
        row = self.make_row()
        broken = json.loads(json.dumps(feedloop.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["legality_clauses"] = ["a clause not in the prompt"]
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_validator_rejects_in_bound_extended_survivor(self) -> None:
        row = self.make_row()
        broken = json.loads(json.dumps(feedloop.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["extended_uniqueness_audit"]["out_of_bound_only"] = False
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_validator_rejects_missing_wrong_attempt(self) -> None:
        row = self.make_row()
        broken = dict(row)
        broken["_audit"] = dict(row["_audit"])
        broken["_audit"]["wrong_in_round1"] = False
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_validator_rejects_unknown_kind_and_surface(self) -> None:
        row = self.make_row()
        broken = dict(row)
        broken["kind"] = "u_statechain"
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])
        broken = dict(row)
        broken["surface"] = "brewvat"
        with self.assertRaises(ValueError):
            feedloop.validate_generated([broken])

    def test_validator_rejects_duplicates(self) -> None:
        row = self.make_row()
        with self.assertRaises(ValueError):
            feedloop.validate_generated([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
