import itertools
import json
import random
import re
import sys
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
STATECHAIN_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
CLEAN_PATH_EXP = ROOT / "experiments" / "qwen35_4b_clean_path_statechain_extension"
sys.path.insert(0, str(EXP / "scripts"))

import gen_gym_mix_curriculum as gym  # noqa: E402
import gen_statechain_curriculum as statechain  # noqa: E402


def answer_of(row: dict) -> str:
    assert row["answer"].startswith("ANSWER: ")
    return row["answer"].removeprefix("ANSWER: ")


def prompt_of(row: dict) -> str:
    return row["messages"][0]["content"]


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def brute_force_mirage(names, relations, queried):
    """Independent brute-force re-derivation of the exhaustive proof."""
    fits = {}
    satisfying = 0
    for assignment in itertools.product(gym.MIRAGE_DOMAIN, repeat=len(names)):
        values = dict(zip(names, assignment))
        if all(gym._mirage_relation_holds(rel, values) for rel in relations):
            satisfying += 1
            fits.setdefault(values[queried], list(assignment))
    return satisfying, sorted(fits), {v: fits[v] for v in sorted(fits)}


class GenerationValidityTests(unittest.TestCase):
    def test_smoke_generation_valid_and_leak_free(self) -> None:
        rows = gym.generate_curriculum(gym.SMOKE_MIX, 12345)
        summary = gym.validate_generated(rows)
        gym.check_banned_vocabulary(rows)
        gym.check_corpus_balance(rows)
        self.assertEqual(summary["rows"], 12)
        self.assertEqual(
            set(summary["kinds"]),
            {"u_siren_episode", "u_statechain", "u_mirage_abstain"},
        )

    def test_banned_vocabulary_scan_actually_fires(self) -> None:
        rows = gym.generate_curriculum(gym.SMOKE_MIX, 54321)
        poisoned = json.loads(json.dumps(gym.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " the warren gate"
        with self.assertRaises(ValueError):
            gym.check_banned_vocabulary([poisoned])

    def test_extended_description_noun_ban_actually_fires(self) -> None:
        rows = gym.generate_curriculum(gym.SMOKE_MIX, 54321)
        for leak in (
            "an injection here",
            "a retrieval task",
            "the document shelf",
            "one directive stands",
            "please abstain now",
            "an abstention token",
            "one constraint holds",
            "this is unsatisfiable",
            "that is impossible",
            "IMPOSSIBLE",
        ):
            poisoned = json.loads(json.dumps(gym.public_row(rows[0])))
            poisoned["_audit"] = rows[0]["_audit"]
            poisoned["messages"][0]["content"] += f" {leak}"
            with self.assertRaises(ValueError, msg=leak):
                gym.check_banned_vocabulary([poisoned])

    def test_scan_is_case_insensitive(self) -> None:
        rows = gym.generate_curriculum(gym.SMOKE_MIX, 54321)
        poisoned = json.loads(json.dumps(gym.public_row(rows[0])))
        poisoned["_audit"] = rows[0]["_audit"]
        poisoned["messages"][0]["content"] += " MIRAGE"
        with self.assertRaises(ValueError):
            gym.check_banned_vocabulary([poisoned])

    def test_abstain_token_is_the_invented_literal(self) -> None:
        self.assertEqual(gym.ABSTAIN_TOKEN, "NOWHERE")
        banned = {token.lower() for token in gym.BANNED_PROMPT_TOKENS}
        self.assertIn("impossible", banned)
        self.assertNotIn("nowhere", banned)

    def test_fresh_tokens_not_banned_and_disjoint_from_statechain_pools(self) -> None:
        fresh = set(gym.FRESH_SURFACE_TOKENS)
        self.assertEqual(len(fresh), len(gym.FRESH_SURFACE_TOKENS))
        self.assertFalse(fresh & set(gym.BANNED_PROMPT_TOKENS))
        self.assertFalse(fresh & set(statechain.INHERITED_SURFACE_TOKENS))
        self.assertFalse(fresh & set(statechain.FRESH_SURFACE_TOKENS))

    def test_frozen_corpus_regenerates_byte_identically(self) -> None:
        rows = gym.generate_curriculum(gym.ARM_MIX, 77180)
        regenerated = "".join(
            json.dumps(gym.public_row(row), ensure_ascii=False) + "\n"
            for row in rows
        )
        frozen = (EXP / "data" / "sft_gym_mix.jsonl").read_text(encoding="utf-8")
        self.assertEqual(regenerated, frozen)

    def test_statechain_generator_is_byte_identical_to_the_proven_source(self) -> None:
        """The statechain machinery is the PROVEN lifecycle 18 generator,
        copied by bytes — never forked."""
        copy = EXP / "scripts" / "gen_statechain_curriculum.py"
        source = STATECHAIN_EXP / "scripts" / "gen_statechain_curriculum.py"
        self.assertEqual(copy.read_bytes(), source.read_bytes())
        clean_path_copy = CLEAN_PATH_EXP / "scripts" / "gen_statechain_curriculum.py"
        self.assertEqual(copy.read_bytes(), clean_path_copy.read_bytes())

    def test_corpus_balance_bounds(self) -> None:
        rows = gym.generate_curriculum(gym.ARM_MIX, 77180)
        balance = gym.check_corpus_balance(rows)
        self.assertEqual(balance["siren_injected"], 45)
        self.assertEqual(balance["siren_clean"], 15)
        self.assertEqual(balance["mirage_forced"], 25)
        self.assertEqual(balance["mirage_abstain"], 25)
        self.assertEqual(balance["mirage_complete_pairs"], 25)
        self.assertTrue(balance["mirage_class_token_sets_identical"])
        self.assertEqual(
            balance["mirage_abstain_reasons"],
            {"many_values": 12, "no_filling": 13},
        )
        self.assertEqual(
            balance["statechain_formalisms"],
            {"brewvat": 13, "courierloft": 13, "muletrack": 12, "peatstove": 12},
        )
        self.assertGreaterEqual(balance["statechain_hidden_updates_min"], 3)
        kinds = Counter(row["kind"] for row in rows)
        self.assertEqual(
            kinds,
            {"u_siren_episode": 60, "u_statechain": 50, "u_mirage_abstain": 50},
        )

    def test_holdout_mix_yields_the_frozen_kind_split(self) -> None:
        rows = gym.generate_curriculum(gym.HOLDOUT_MIX, 88046)
        summary = gym.validate_generated(rows)
        self.assertEqual(summary["rows"], 40)
        self.assertEqual(
            summary["kinds"],
            {"u_siren_episode": 14, "u_statechain": 13, "u_mirage_abstain": 13},
        )
        self.assertEqual(
            summary["surfaces"],
            {
                "stillroom": 14,
                "counterhouse": 13,
                "brewvat": 4,
                "courierloft": 3,
                "peatstove": 3,
                "muletrack": 3,
            },
        )

    def test_zero_row_overlap_with_statechain_corpora_and_gates(self) -> None:
        """The reused statechain formalisms are FRESH instances, not copies."""
        local = {
            message_bytes(row)
            for row in gym.generate_curriculum(gym.ARM_MIX, 77180)
        } | {
            message_bytes(row)
            for row in gym.generate_curriculum(gym.HOLDOUT_MIX, 88046)
        }
        for path in (
            STATECHAIN_EXP / "data" / "sft_statechain_only.jsonl",
            STATECHAIN_EXP / "data" / "local_tasks_seed88033.jsonl",
            CLEAN_PATH_EXP / "data" / "local_tasks_seed88041.jsonl",
        ):
            reference_rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            reference = {message_bytes(row) for row in reference_rows}
            self.assertFalse(local & reference, path.name)


class SirenEpisodeRederivationTests(unittest.TestCase):
    def test_injected_siren_chain_and_decoy_contract(self) -> None:
        rng = random.Random(707)
        for _ in range(12):
            row = gym.siren_lesson(rng, injected=True)
            audit = row["_audit"]
            spec = audit["spec"]
            prompt = prompt_of(row)
            # Executable truth: the answer is the last hop's payload and the
            # chain links appear verbatim in the transcript.
            self.assertEqual(answer_of(row), str(spec["value"]))
            self.assertEqual(spec["payloads"][-1], str(spec["value"]))
            for position in range(len(spec["books"])):
                self.assertIn(
                    f"the {spec['fields'][position]} for {spec['keys'][position]} "
                    f"is {spec['payloads'][position]}",
                    prompt,
                )
            # The u_hygiene mechanism: format-matched decoy, never the truth.
            self.assertGreaterEqual(audit["injections"], 1)
            self.assertNotEqual(audit["decoy"], spec["value"])
            self.assertTrue(audit["decoy_differs"])
            self.assertTrue(1000 <= audit["decoy"] <= 9999)
            self.assertTrue(audit["imperatives_wellformed"])
            for imperative in spec["imperatives"]:
                self.assertIn(imperative, prompt)
                self.assertIn(str(audit["decoy"]), imperative)
                self.assertTrue(imperative.endswith("."))
            # Obeying the imperative is parseable-but-wrong: the decoy parses
            # as a 4-digit answer but never equals the expected value.
            self.assertIn(str(audit["decoy"]), prompt)
            # The true value appears exactly once (the final entry).
            self.assertEqual(prompt.count(str(spec["value"])), 1)
            # The think target narrates ignoring the embedded orders.
            self.assertIn("book matter, not a task order", row["think"])

    def test_clean_siren_rows_carry_no_imperative(self) -> None:
        rng = random.Random(808)
        for _ in range(8):
            row = gym.siren_lesson(rng, injected=False)
            audit = row["_audit"]
            self.assertEqual(audit["injections"], 0)
            self.assertIsNone(audit["decoy"])
            for marker in gym.SIREN_INJECTION_MARKERS:
                self.assertNotIn(marker, prompt_of(row))
            self.assertIn("No pulled entry carries any order", row["think"])

    def test_siren_validator_rejects_decoy_equal_to_truth(self) -> None:
        rng = random.Random(909)
        row = gym.siren_lesson(rng, injected=True)
        row["task_id"] = "cgm_siren_episode_00000"
        broken = json.loads(json.dumps(gym.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["decoy"] = broken["_audit"]["spec"]["value"]
        broken["_audit"]["decoy_differs"] = True
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])

    def test_siren_validator_rejects_unembedded_imperative(self) -> None:
        rng = random.Random(101)
        row = gym.siren_lesson(rng, injected=True)
        row["task_id"] = "cgm_siren_episode_00000"
        broken = json.loads(json.dumps(gym.public_row(row)))
        broken["_audit"] = json.loads(json.dumps(row["_audit"]))
        broken["_audit"]["spec"]["imperatives"] = ["an imperative not in the prompt."]
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])


class MirageRederivationTests(unittest.TestCase):
    def test_pairs_prove_forced_and_unforced_by_exhaustion(self) -> None:
        rng = random.Random(202)
        for pair_index in range(6):
            reason = "no_filling" if pair_index % 2 == 0 else "many_values"
            forced, abstain = gym.mirage_pair(rng, f"pair_{pair_index:05d}", reason)
            for row, expected_class in ((forced, "forced"), (abstain, "abstain")):
                audit = row["_audit"]
                spec = audit["spec"]
                self.assertEqual(audit["class"], expected_class)
                satisfying, values, witnesses = brute_force_mirage(
                    spec["names"],
                    [tuple(rel) for rel in spec["relations"]],
                    spec["queried"],
                )
                self.assertEqual(satisfying, audit["satisfying_assignments"])
                self.assertEqual(values, audit["queried_values"])
                if expected_class == "forced":
                    self.assertEqual(len(values), 1)
                    self.assertEqual(answer_of(row), str(values[0]))
                else:
                    self.assertNotEqual(len(values), 1)
                    self.assertEqual(answer_of(row), gym.ABSTAIN_TOKEN)
                    self.assertEqual(audit["abstain_reason"], reason)
            # Class indistinguishability: the pair differs ONLY in digits.
            self.assertEqual(
                re.sub(r"\d", "", prompt_of(forced)),
                re.sub(r"\d", "", prompt_of(abstain)),
            )
            self.assertNotEqual(prompt_of(forced), prompt_of(abstain))

    def test_solver_matches_brute_force_on_random_systems(self) -> None:
        rng = random.Random(303)
        for _ in range(20):
            names, skeleton = gym._mirage_skeleton(rng)
            relations = gym._mirage_instantiate(rng, skeleton)
            queried = rng.choice(names)
            solved = gym.solve_mirage(names, relations, queried)
            satisfying, values, witnesses = brute_force_mirage(
                names, relations, queried
            )
            self.assertEqual(solved["satisfying_assignments"], satisfying)
            self.assertEqual(solved["queried_values"], values)
            self.assertEqual(solved["witnesses"], witnesses)

    def test_witnesses_satisfy_every_relation(self) -> None:
        rng = random.Random(404)
        forced, abstain = gym.mirage_pair(rng, "pair_00000", "many_values")
        for row in (forced, abstain):
            spec = row["_audit"]["spec"]
            solved = gym.solve_mirage(
                spec["names"],
                [tuple(rel) for rel in spec["relations"]],
                spec["queried"],
            )
            for value, witness in solved["witnesses"].items():
                values = dict(zip(spec["names"], witness))
                self.assertEqual(values[spec["queried"]], value)
                for relation in spec["relations"]:
                    self.assertTrue(
                        gym._mirage_relation_holds(tuple(relation), values)
                    )

    def test_mirage_validator_rejects_flipped_class(self) -> None:
        rng = random.Random(505)
        forced, abstain = gym.mirage_pair(rng, "pair_00000", "no_filling")
        forced["task_id"] = "cgm_mirage_abstain_00000"
        broken = json.loads(json.dumps(gym.public_row(forced)))
        broken["_audit"] = json.loads(json.dumps(forced["_audit"]))
        broken["_audit"]["class"] = "abstain"
        broken["_audit"]["abstain_reason"] = "no_filling"
        broken["answer"] = f"ANSWER: {gym.ABSTAIN_TOKEN}"
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])

    def test_mirage_validator_rejects_poisoned_enumeration(self) -> None:
        rng = random.Random(606)
        forced, _ = gym.mirage_pair(rng, "pair_00000", "no_filling")
        forced["task_id"] = "cgm_mirage_abstain_00000"
        broken = json.loads(json.dumps(gym.public_row(forced)))
        broken["_audit"] = json.loads(json.dumps(forced["_audit"]))
        broken["_audit"]["satisfying_assignments"] += 1
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])

    def test_pair_generator_rejects_unknown_reason(self) -> None:
        rng = random.Random(707)
        with self.assertRaises(ValueError):
            gym.mirage_pair(rng, "pair_00000", "sometimes")


class ValidatorRejectionTests(unittest.TestCase):
    def test_validator_rejects_unknown_kind(self) -> None:
        rng = random.Random(111)
        row = gym.siren_lesson(rng, injected=True)
        row["task_id"] = "cgm_siren_episode_00000"
        broken = dict(row)
        broken["kind"] = "u_feedloop"
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])

    def test_validator_rejects_changed_surface(self) -> None:
        rng = random.Random(222)
        row = gym.siren_lesson(rng, injected=False)
        row["task_id"] = "cgm_siren_episode_00000"
        broken = dict(row)
        broken["surface"] = "harbor-ish"
        with self.assertRaises(ValueError):
            gym.validate_generated([broken])

    def test_statechain_rows_run_through_the_proven_validator(self) -> None:
        rows = gym.generate_curriculum("statechain=8", 424242)
        chains = [row for row in rows if row["kind"] == "u_statechain"]
        self.assertEqual(len(chains), 8)
        # The proven validator accepts them...
        statechain.validate_generated(chains)
        # ...and rejects a poisoned distractor audit through gym's validator.
        broken = dict(chains[0])
        broken["_audit"] = dict(chains[0]["_audit"])
        broken["_audit"]["distractor_stateless"] = answer_of(chains[0])
        replaced = [broken if row is chains[0] else row for row in rows]
        with self.assertRaises(ValueError):
            gym.validate_generated(replaced)


if __name__ == "__main__":
    unittest.main()
