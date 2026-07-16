"""Unit tests for the preregistered confirmation readings (synthetic tables).

Everything here runs on synthetic score tables: no model, no gateway, no
files. Covers the forensics-style strict-win/goal-gate counting per seed
(exact ties are never strict wins), the ORDERED TOTAL PARTITION verdict
at its preregistered boundaries AND over the full 3^3 x 2^3 = 216
independent (aggregate outcome x gate outcome) enumeration (exactly one
verdict fires per combination; the CONFIRMED region is exactly
{aggregate strictly wins all three} INTERSECT {gate passes >= 2}), the
exclusion of the discovery seed 78154 from the verdict (reported
alongside, never counted), the fragility margins and blocking families,
the fail-closed implementation-signature equality against the discovery
block, the budget-integrity scoping (paired_comparison_valid flips on
any over-budget arm at any seed while scores stay recorded), the readout
schema, and the MAJOR-1 provenance layers: authenticate_ledger (the
readout requires the complete canonical ledger — missing, incomplete,
crashed, out-of-order, tampered, or legacy-without-receipt-pins all
refuse) and require_summary_consistency (a receipt whose
scores/budget/implementation diverge from the sealed summary refuses —
a swapped receipt can no longer flip the verdict undetectably).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import check_benchmark as cb  # noqa: E402


SEED_1, SEED_2, SEED_3 = cb.SEED_ORDER


def per_family(value: float, **overrides: float) -> dict[str, float]:
    values = {family: value for family in cb.FAMILIES}
    for family, override in overrides.items():
        if family not in values:
            raise AssertionError(f"unknown family: {family}")
        values[family] = override
    return values


def base_arm() -> dict:
    # Mirrors the discovery shape: aggregate 0.08, four nonzero families.
    return {
        "aggregate": 0.08,
        "per_family": per_family(
            0.0, chronicle=0.1, sirens=0.4, toolsmith=0.2, warren=0.1
        ),
    }


def passing_treated_arm() -> dict:
    # Strictly above base on every family, including thin menders/warren
    # margins like the discovery event's 0.0167 and 0.05.
    return {
        "aggregate": 0.3663,
        "per_family": per_family(
            0.3,
            chronicle=0.2,
            menders=0.0167,
            sirens=0.6,
            toolsmith=0.8,
            warren=0.15,
        ),
    }


def tying_treated_arm() -> dict:
    # Exact ties with base on menders and warren: eight strict wins, no
    # pass, but the aggregate still strictly beats base.
    treated = passing_treated_arm()
    treated["per_family"]["menders"] = 0.0
    treated["per_family"]["warren"] = 0.1
    treated["aggregate"] = 0.34
    return treated


def losing_treated_arm() -> dict:
    # An aggregate strictly below base.
    return {"aggregate": 0.05, "per_family": per_family(0.0, chronicle=0.05)}


def seed_pair(treated: dict) -> dict:
    return {"base": base_arm(), cb.TREATED_ARM: treated}


def synthetic_scores(
    seed_1: dict | None = None,
    seed_2: dict | None = None,
    seed_3: dict | None = None,
) -> dict[int, dict]:
    return {
        SEED_1: seed_pair(seed_1 or passing_treated_arm()),
        SEED_2: seed_pair(seed_2 or passing_treated_arm()),
        SEED_3: seed_pair(seed_3 or passing_treated_arm()),
    }


def synthetic_discovery_scores() -> dict[str, dict]:
    return {
        cb.DISCOVERY["base_arm"]: base_arm(),
        cb.DISCOVERY["treated_arm"]: passing_treated_arm(),
    }


def synthetic_implementation() -> dict:
    return {
        "runner_sha256": "a" * 64,
        "source_inventory_sha256": "b" * 64,
        "source_file_count": 56,
    }


def synthetic_discovery(implementation: dict | None = None) -> dict:
    return {
        "scores": synthetic_discovery_scores(),
        "benchmark_implementation": implementation or synthetic_implementation(),
    }


def synthetic_budget(**overrides: dict) -> dict[int, dict]:
    budget = {
        seed: {
            label: {"within_budget": True, "wall_seconds": 100.0 + index}
            for index, label in enumerate(cb.MODEL_ORDER)
        }
        for seed in cb.SEED_ORDER
    }
    for key, override in overrides.items():
        seed_text, _, label = key.partition("__")
        seed = int(seed_text)
        if seed not in budget or label not in budget[seed]:
            raise AssertionError(f"unknown arm: {key}")
        budget[seed][label].update(override)
    return budget


def synthetic_receipts() -> dict[int, dict]:
    return {
        seed: {
            label: {
                "path": f"runs/benchmark/x_{seed}/{label}.json",
                "sha256": "0" * 64,
            }
            for label in cb.MODEL_ORDER
        }
        for seed in cb.SEED_ORDER
    }


class TestGoalGateCounting(unittest.TestCase):
    def test_all_ten_strict_wins_pass(self) -> None:
        row = cb.goal_gate_row(
            base_arm()["per_family"], passing_treated_arm()["per_family"]
        )
        self.assertEqual(row["strict_wins"], 10)
        self.assertEqual(row["wins"], list(cb.FAMILIES))
        self.assertEqual(row["losses"], [])
        self.assertEqual(row["ties"], [])
        self.assertTrue(row["goal_gate_pass"])

    def test_an_exact_tie_is_not_a_strict_win(self) -> None:
        row = cb.goal_gate_row(
            base_arm()["per_family"], tying_treated_arm()["per_family"]
        )
        self.assertEqual(row["strict_wins"], 8)
        self.assertEqual(row["ties"], ["menders", "warren"])
        self.assertEqual(row["losses"], [])
        self.assertFalse(row["goal_gate_pass"])

    def test_losses_and_ties_recorded_separately(self) -> None:
        treated = passing_treated_arm()
        treated["per_family"]["sirens"] = 0.3  # below base's 0.4
        row = cb.goal_gate_row(base_arm()["per_family"], treated["per_family"])
        self.assertEqual(row["strict_wins"], 9)
        self.assertEqual(row["losses"], ["sirens"])
        self.assertEqual(row["ties"], [])
        self.assertFalse(row["goal_gate_pass"])

    def test_nine_wins_never_pass(self) -> None:
        treated = passing_treated_arm()
        treated["per_family"]["menders"] = 0.0
        row = cb.goal_gate_row(base_arm()["per_family"], treated["per_family"])
        self.assertEqual(row["strict_wins"], 9)
        self.assertFalse(row["goal_gate_pass"])


class TestPerSeedReading(unittest.TestCase):
    def test_per_seed_shape_and_contents(self) -> None:
        reading = cb.per_seed_reading(synthetic_scores(seed_3=tying_treated_arm()))
        self.assertEqual(set(reading), {str(seed) for seed in cb.SEED_ORDER})
        first = reading[str(SEED_1)]
        self.assertEqual(first["aggregates"]["base"], 0.08)
        self.assertEqual(first["aggregates"][cb.TREATED_ARM], 0.3663)
        self.assertAlmostEqual(first["aggregate_margin"], 0.2863)
        self.assertTrue(first["treated_beats_base_aggregate"])
        self.assertTrue(first["goal_gate"]["goal_gate_pass"])
        self.assertEqual(set(first["per_family"]), set(cb.MODEL_ORDER))
        self.assertEqual(set(first["per_family"]["base"]), set(cb.FAMILIES))
        third = reading[str(SEED_3)]
        self.assertFalse(third["goal_gate"]["goal_gate_pass"])
        self.assertTrue(third["treated_beats_base_aggregate"])


class TestConfirmationVerdict(unittest.TestCase):
    def verdict(self, scores: dict[int, dict]) -> dict:
        return cb.confirmation_verdict(scores, synthetic_discovery_scores())

    def test_three_gate_passes_confirm(self) -> None:
        reading = self.verdict(synthetic_scores())
        self.assertEqual(reading["verdict"], "CONFIRMED")
        self.assertTrue(reading["aggregate_wins_all_three_seeds"])
        self.assertEqual(reading["goal_gate_pass_count"], 3)
        self.assertTrue(reading["goal_gate_majority"])

    def test_two_of_three_gate_passes_confirm(self) -> None:
        # THE preregistered boundary: 2/3 goal-gate passes with all three
        # aggregate wins is still CONFIRMED.
        reading = self.verdict(synthetic_scores(seed_2=tying_treated_arm()))
        self.assertEqual(reading["verdict"], "CONFIRMED")
        self.assertEqual(reading["goal_gate_pass_count"], 2)

    def test_one_of_three_gate_passes_is_aggregate_only(self) -> None:
        reading = self.verdict(
            synthetic_scores(
                seed_1=tying_treated_arm(), seed_2=tying_treated_arm()
            )
        )
        self.assertEqual(reading["verdict"], "AGGREGATE_ONLY")
        self.assertTrue(reading["aggregate_wins_all_three_seeds"])
        self.assertEqual(reading["goal_gate_pass_count"], 1)
        self.assertFalse(reading["goal_gate_majority"])

    def test_zero_gate_passes_is_aggregate_only(self) -> None:
        reading = self.verdict(
            synthetic_scores(
                seed_1=tying_treated_arm(),
                seed_2=tying_treated_arm(),
                seed_3=tying_treated_arm(),
            )
        )
        self.assertEqual(reading["verdict"], "AGGREGATE_ONLY")
        self.assertEqual(reading["goal_gate_pass_count"], 0)

    def test_aggregate_loss_on_one_seed_is_not_replicated(self) -> None:
        # Even with 2/3 goal-gate passes, one aggregate loss breaks the
        # all-three requirement.
        reading = self.verdict(synthetic_scores(seed_3=losing_treated_arm()))
        self.assertEqual(reading["verdict"], "NOT_REPLICATED")
        self.assertFalse(reading["aggregate_wins_all_three_seeds"])
        self.assertEqual(reading["goal_gate_pass_count"], 2)

    def test_aggregate_exact_tie_fails_strict(self) -> None:
        # An exact aggregate tie on one seed is NOT a strict win: even a
        # 10/10 goal-gate pass on that seed cannot rescue CONFIRMED.
        tied = passing_treated_arm()
        tied["aggregate"] = base_arm()["aggregate"]
        reading = self.verdict(synthetic_scores(seed_2=tied))
        self.assertEqual(reading["verdict"], "NOT_REPLICATED")
        self.assertFalse(reading["aggregate_strict_wins"][str(SEED_2)])
        self.assertTrue(reading["goal_gate_passes"][str(SEED_2)])

    def test_partition_is_total_and_ordered_over_all_216_combinations(self) -> None:
        # HONEST enumeration: aggregate outcome (win/tie/loss vs base's
        # 0.08) and goal-gate outcome (pass/fail) vary INDEPENDENTLY per
        # seed — all 3^3 x 2^3 = 216 combinations. Exactly one verdict
        # fires for each, and the CONFIRMED set is exactly
        # {aggregate strictly wins all 3} INTERSECT {gate passes >= 2}.
        import itertools  # noqa: PLC0415

        base_aggregate = base_arm()["aggregate"]
        aggregate_values = {
            "win": base_aggregate + 0.1,
            "tie": base_aggregate,
            "loss": base_aggregate - 0.03,
        }

        def treated(aggregate_outcome: str, gate_outcome: str) -> dict:
            arm = passing_treated_arm() if gate_outcome == "pass" else tying_treated_arm()
            arm["aggregate"] = aggregate_values[aggregate_outcome]
            return arm

        combinations = list(
            itertools.product(
                itertools.product(("win", "tie", "loss"), repeat=3),
                itertools.product(("pass", "fail"), repeat=3),
            )
        )
        self.assertEqual(len(combinations), 216)
        confirmed_set = set()
        for aggregate_outcomes, gate_outcomes in combinations:
            scores = synthetic_scores(
                seed_1=treated(aggregate_outcomes[0], gate_outcomes[0]),
                seed_2=treated(aggregate_outcomes[1], gate_outcomes[1]),
                seed_3=treated(aggregate_outcomes[2], gate_outcomes[2]),
            )
            reading = self.verdict(scores)
            # Totality: the verdict is always exactly one of the three.
            self.assertIn(reading["verdict"], cb.VERDICTS)
            self.assertEqual(
                sum(reading["verdict"] == option for option in cb.VERDICTS), 1
            )
            # The synthetic gate outcomes must be realized faithfully.
            realized_passes = [
                reading["goal_gate_passes"][str(seed)] for seed in cb.SEED_ORDER
            ]
            self.assertEqual(
                realized_passes, [outcome == "pass" for outcome in gate_outcomes]
            )
            aggregate_all = all(
                outcome == "win" for outcome in aggregate_outcomes
            )
            passes = sum(outcome == "pass" for outcome in gate_outcomes)
            if aggregate_all and passes >= 2:
                expected = "CONFIRMED"
            elif aggregate_all:
                expected = "AGGREGATE_ONLY"
            else:
                expected = "NOT_REPLICATED"
            self.assertEqual(
                reading["verdict"], expected, (aggregate_outcomes, gate_outcomes)
            )
            if reading["verdict"] == "CONFIRMED":
                confirmed_set.add((aggregate_outcomes, gate_outcomes))
        # The CONFIRMED region is exactly the preregistered intersection:
        # 1 aggregate combination x 4 gate combinations with >= 2 passes.
        self.assertEqual(
            confirmed_set,
            {
                (("win", "win", "win"), gates)
                for gates in itertools.product(("pass", "fail"), repeat=3)
                if sum(gate == "pass" for gate in gates) >= 2
            },
        )
        self.assertEqual(len(confirmed_set), 4)

    def test_discovery_seed_is_reported_but_never_counted(self) -> None:
        # A NOT_REPLICATED table stays NOT_REPLICATED no matter what the
        # discovery block says, and vice versa: the discovery scores feed
        # only the report.
        failing = synthetic_scores(
            seed_1=losing_treated_arm(),
            seed_2=losing_treated_arm(),
            seed_3=losing_treated_arm(),
        )
        for discovery_treated in (passing_treated_arm(), losing_treated_arm()):
            reading = cb.confirmation_verdict(
                failing,
                {
                    cb.DISCOVERY["base_arm"]: base_arm(),
                    cb.DISCOVERY["treated_arm"]: discovery_treated,
                },
            )
            self.assertEqual(reading["verdict"], "NOT_REPLICATED")
        confirmed = cb.confirmation_verdict(
            synthetic_scores(),
            {
                cb.DISCOVERY["base_arm"]: base_arm(),
                cb.DISCOVERY["treated_arm"]: losing_treated_arm(),
            },
        )
        self.assertEqual(confirmed["verdict"], "CONFIRMED")

    def test_discovery_block_contents(self) -> None:
        reading = self.verdict(synthetic_scores())
        discovery = reading["discovery"]
        self.assertEqual(discovery["seed"], 78154)
        self.assertIs(discovery["counted_in_verdict"], False)
        self.assertEqual(discovery["summary_sha256"], cb.DISCOVERY["summary_sha256"])
        self.assertEqual(
            discovery["treated_arm_label"], cb.DISCOVERY["treated_arm"]
        )
        self.assertTrue(discovery["goal_gate"]["goal_gate_pass"])
        self.assertAlmostEqual(discovery["fragility_margins"]["menders"], 0.0167)
        self.assertAlmostEqual(discovery["fragility_margins"]["warren"], 0.05)
        self.assertNotIn(str(discovery["seed"]), reading["aggregate_strict_wins"])
        self.assertNotIn(str(discovery["seed"]), reading["goal_gate_passes"])

    def test_rule_text_is_carried(self) -> None:
        reading = self.verdict(synthetic_scores())
        self.assertEqual(reading["rule"], cb.VERDICT_RULE)
        self.assertIn("NEVER counted", reading["rule"])


class TestFragility(unittest.TestCase):
    def test_margins_and_blocking_families(self) -> None:
        reading = cb.fragility(synthetic_scores(seed_2=tying_treated_arm()))
        self.assertEqual(reading["families"], ["menders", "warren"])
        first = reading["per_seed"][str(SEED_1)]
        self.assertAlmostEqual(first["menders_margin"], 0.0167)
        self.assertAlmostEqual(first["warren_margin"], 0.05)
        self.assertTrue(first["goal_gate_pass"])
        self.assertEqual(first["blocking_families"], [])
        second = reading["per_seed"][str(SEED_2)]
        self.assertAlmostEqual(second["menders_margin"], 0.0)
        self.assertAlmostEqual(second["warren_margin"], 0.0)
        self.assertFalse(second["goal_gate_pass"])
        self.assertEqual(second["blocking_families"], ["menders", "warren"])

    def test_blocking_families_include_losses(self) -> None:
        treated = passing_treated_arm()
        treated["per_family"]["sirens"] = 0.3
        treated["per_family"]["warren"] = 0.1
        reading = cb.fragility(synthetic_scores(seed_1=treated))
        first = reading["per_seed"][str(SEED_1)]
        self.assertEqual(first["blocking_families"], ["sirens", "warren"])
        self.assertAlmostEqual(first["warren_margin"], 0.0)

    def test_negative_margins_preserved(self) -> None:
        treated = passing_treated_arm()
        treated["per_family"]["warren"] = 0.05
        reading = cb.fragility(synthetic_scores(seed_3=treated))
        third = reading["per_seed"][str(SEED_3)]
        self.assertAlmostEqual(third["warren_margin"], -0.05)
        self.assertEqual(third["blocking_families"], ["warren"])


class TestBudgetIntegrity(unittest.TestCase):
    def test_all_within_budget_is_valid(self) -> None:
        reading = cb.budget_integrity(synthetic_budget())
        self.assertTrue(reading["all_within_budget"])
        self.assertTrue(reading["paired_comparison_valid"])
        self.assertIsNone(reading["reason"])
        self.assertEqual(
            set(reading["per_seed"]), {str(seed) for seed in cb.SEED_ORDER}
        )
        first = reading["per_seed"][str(SEED_1)]
        self.assertEqual(first["per_arm"]["base"]["wall_seconds"], 100.0)
        self.assertTrue(first["paired_comparison_valid"])

    def test_any_over_budget_arm_invalidates_the_comparison(self) -> None:
        reading = cb.budget_integrity(
            synthetic_budget(**{f"{SEED_2}__hygiene_explore": {"within_budget": False}})
        )
        self.assertFalse(reading["all_within_budget"])
        self.assertFalse(reading["paired_comparison_valid"])
        self.assertIn("hygiene_explore", reading["reason"])
        self.assertIn(str(SEED_2), reading["reason"])
        second = reading["per_seed"][str(SEED_2)]
        self.assertFalse(second["paired_comparison_valid"])
        self.assertIn("scores recorded", second["reason"])
        # The other seeds stay budget-valid; the flag never drops a record.
        self.assertTrue(reading["per_seed"][str(SEED_1)]["paired_comparison_valid"])
        self.assertIs(
            second["per_arm"]["hygiene_explore"]["within_budget"], False
        )

    def test_every_over_budget_arm_is_named(self) -> None:
        reading = cb.budget_integrity(
            synthetic_budget(
                **{
                    f"{SEED_1}__base": {"within_budget": False},
                    f"{SEED_3}__hygiene_explore": {"within_budget": False},
                }
            )
        )
        self.assertFalse(reading["paired_comparison_valid"])
        self.assertIn(f"seed {SEED_1}: base", reading["reason"])
        self.assertIn(f"seed {SEED_3}: hygiene_explore", reading["reason"])

    def test_wall_seconds_carried_verbatim(self) -> None:
        budget = synthetic_budget(
            **{f"{SEED_1}__hygiene_explore": {"wall_seconds": 5400.5}}
        )
        reading = cb.budget_integrity(budget)
        self.assertEqual(
            reading["per_seed"][str(SEED_1)]["per_arm"]["hygiene_explore"][
                "wall_seconds"
            ],
            5400.5,
        )


class TestImplementationEquality(unittest.TestCase):
    def test_equality_guard_is_exact(self) -> None:
        cb.require_implementation_equality(
            synthetic_implementation(), synthetic_implementation()
        )
        for drift in (
            {"runner_sha256": "c" * 64},
            {"source_inventory_sha256": "d" * 64},
            {"source_file_count": 57},
        ):
            with self.assertRaises(ValueError, msg=str(drift)):
                cb.require_implementation_equality(
                    {**synthetic_implementation(), **drift},
                    synthetic_implementation(),
                )

    def test_valid_implementation_predicate(self) -> None:
        self.assertTrue(cb._valid_implementation(synthetic_implementation()))
        self.assertTrue(cb._valid_implementation(cb.DISCOVERY_IMPLEMENTATION))
        for bad in (
            None,
            {},
            {**synthetic_implementation(), "extra": 1},
            {**synthetic_implementation(), "runner_sha256": "zz"},
            {**synthetic_implementation(), "source_file_count": 0},
            {**synthetic_implementation(), "source_file_count": True},
            {**synthetic_implementation(), "source_file_count": "56"},
        ):
            self.assertFalse(cb._valid_implementation(bad), str(bad))


class TestReadoutSchema(unittest.TestCase):
    def build(
        self,
        scores: dict[int, dict] | None = None,
        budget: dict[int, dict] | None = None,
        implementation: dict | None = None,
        discovery: dict | None = None,
    ) -> dict:
        return cb.build_readout(
            scores or synthetic_scores(),
            budget or synthetic_budget(),
            implementation or synthetic_implementation(),
            discovery or synthetic_discovery(),
            synthetic_receipts(),
            "f" * 64,
        )

    def setUp(self) -> None:
        self.readout = self.build()

    def test_top_level_schema(self) -> None:
        self.assertEqual(
            set(self.readout),
            {
                "schema_version", "experiment_id", "stage", "name", "tier",
                "think_budget", "seeds", "benchmark_data_read", "promoted",
                "outcome", "verdict", "paired_comparison_valid",
                "design_receipt_sha256", "provenance", "discovery_reference",
                "benchmark_implementation", "receipts", "scores", "budget",
                "readings",
            },
        )
        provenance = self.readout["provenance"]
        self.assertEqual(provenance["ledger"], "runs/benchmark_events.jsonl")
        self.assertIs(provenance["receipt_sha256s_pinned_in_closed_records"], True)
        self.assertIs(provenance["ledger_complete_sequence_required"], True)
        self.assertEqual(self.readout["schema_version"], 1)
        self.assertEqual(self.readout["stage"], "goal_gate_confirmation_readout")
        self.assertEqual(self.readout["tier"], "medium")
        self.assertEqual(self.readout["think_budget"], 1024)
        self.assertEqual(self.readout["seeds"], [78155, 78156, 78157])

    def test_measurement_intake_never_promotes(self) -> None:
        self.assertIsNone(self.readout["promoted"])
        self.assertIs(self.readout["benchmark_data_read"], False)
        self.assertEqual(self.readout["outcome"], "CONFIRMATION_READ_COMPLETE")

    def test_readings_are_exactly_the_four_preregistered(self) -> None:
        self.assertEqual(
            set(self.readout["readings"]),
            {
                "per_seed", "confirmation_verdict", "fragility",
                "budget_integrity",
            },
        )
        self.assertEqual(self.readout["verdict"], "CONFIRMED")
        self.assertEqual(
            self.readout["readings"]["confirmation_verdict"]["verdict"],
            "CONFIRMED",
        )

    def test_verdict_hoisted_from_the_reading(self) -> None:
        readout = self.build(scores=synthetic_scores(seed_1=losing_treated_arm()))
        self.assertEqual(readout["verdict"], "NOT_REPLICATED")
        self.assertEqual(
            readout["readings"]["confirmation_verdict"]["verdict"],
            "NOT_REPLICATED",
        )

    def test_within_budget_event_is_a_valid_paired_comparison(self) -> None:
        self.assertIs(self.readout["paired_comparison_valid"], True)
        self.assertIsNone(self.readout["readings"]["budget_integrity"]["reason"])

    def test_over_budget_arm_scopes_but_keeps_scores(self) -> None:
        readout = self.build(
            budget=synthetic_budget(
                **{f"{SEED_3}__hygiene_explore": {"within_budget": False}}
            )
        )
        self.assertIs(readout["paired_comparison_valid"], False)
        integrity = readout["readings"]["budget_integrity"]
        self.assertIs(integrity["paired_comparison_valid"], False)
        self.assertIn("hygiene_explore", integrity["reason"])
        # Scores, the verdict, and the other readings stay fully recorded.
        self.assertEqual(readout["verdict"], "CONFIRMED")
        self.assertEqual(
            set(readout["scores"]), {str(seed) for seed in cb.SEED_ORDER}
        )

    def test_scores_and_budget_carried_verbatim(self) -> None:
        scores = synthetic_scores()
        budget = synthetic_budget()
        readout = self.build(scores=scores, budget=budget)
        for seed in cb.SEED_ORDER:
            self.assertEqual(readout["scores"][str(seed)], scores[seed])
            self.assertEqual(readout["budget"][str(seed)], budget[seed])
        self.assertEqual(readout["discovery_reference"], cb.DISCOVERY)
        self.assertEqual(
            set(readout["receipts"][str(SEED_1)]), set(cb.MODEL_ORDER)
        )

    def test_implementation_mismatch_aborts_the_whole_readout(self) -> None:
        with self.assertRaises(ValueError):
            self.build(
                implementation={
                    **synthetic_implementation(),
                    "runner_sha256": "e" * 64,
                }
            )
        with self.assertRaises(ValueError):
            self.build(
                discovery=synthetic_discovery(
                    {**synthetic_implementation(), "source_file_count": 57}
                )
            )

    def test_matching_signature_is_surfaced_on_both_sides(self) -> None:
        block = self.readout["benchmark_implementation"]
        self.assertEqual(block["signature"], synthetic_implementation())
        self.assertEqual(block["discovery"], synthetic_implementation())
        self.assertIs(
            block["identical_across_all_six_receipts_and_discovery"], True
        )


def opened(seed: int) -> dict:
    return cb.opened_record(seed)


def closed(seed: int) -> dict:
    return {
        "name": cb.FROZEN_NAME,
        "phase": "closed",
        "tier": cb.FROZEN_TIER,
        "think_budget": cb.FROZEN_THINK_BUDGET,
        "seed": seed,
        "summary": str(cb.EVENT_DIRS[seed] / "summary.json"),
        "summary_sha256": "0" * 64,
        "receipts": {"base": "1" * 64, "hygiene_explore": "2" * 64},
    }


def complete_ledger() -> list[dict]:
    rows = []
    for seed in cb.SEED_ORDER:
        rows.append(opened(seed))
        rows.append(closed(seed))
    return rows


class TestLedgerAnchoring(unittest.TestCase):
    def test_complete_ledger_returns_the_three_closed_records(self) -> None:
        records = cb.authenticate_ledger(complete_ledger())
        self.assertEqual(set(records), set(cb.SEED_ORDER))
        for seed in cb.SEED_ORDER:
            self.assertEqual(records[seed], closed(seed))

    def test_missing_or_empty_ledger_refuses(self) -> None:
        # MAJOR-1 verified consequence: with no ledger, six forged receipts
        # must never produce a readout.
        with self.assertRaises(ValueError):
            cb.authenticate_ledger([])

    def test_incomplete_ledger_refuses(self) -> None:
        rows = complete_ledger()
        for cut in (1, 2, 3, 4, 5):
            with self.assertRaises(ValueError, msg=str(cut)):
                cb.authenticate_ledger(rows[:cut])

    def test_crashed_trailing_opened_record_refuses(self) -> None:
        rows = complete_ledger()[:4] + [opened(cb.SEED_ORDER[2])]
        with self.assertRaises(ValueError):
            cb.authenticate_ledger(rows)

    def test_out_of_order_seeds_refuse(self) -> None:
        rows = complete_ledger()
        rows[0], rows[2] = rows[2], rows[0]
        rows[1], rows[3] = rows[3], rows[1]
        with self.assertRaises(ValueError):
            cb.authenticate_ledger(rows)

    def test_trailing_extra_rows_refuse(self) -> None:
        with self.assertRaises(ValueError):
            cb.authenticate_ledger(complete_ledger() + [opened(cb.SEED_ORDER[0])])

    def test_tampered_closed_record_refuses(self) -> None:
        for drift in (
            {"summary_sha256": "zz"},
            {"receipts": {"base": "1" * 64}},
            {"receipts": {"base": "1" * 64, "hygiene_explore": "bad"}},
            {"tier": "quick"},
            {"think_budget": 4096},
        ):
            rows = complete_ledger()
            rows[1] = {**rows[1], **drift}
            with self.assertRaises(ValueError, msg=str(drift)):
                cb.authenticate_ledger(rows)

    def test_legacy_closed_record_without_receipts_refuses(self) -> None:
        rows = complete_ledger()
        rows[3] = {
            key: value for key, value in rows[3].items() if key != "receipts"
        }
        with self.assertRaises(ValueError):
            cb.authenticate_ledger(rows)


def synthetic_event(
    label: str, scores: dict, budget: dict, implementation: dict
) -> dict:
    return {
        "aggregate": scores[label]["aggregate"],
        "per_family": dict(scores[label]["per_family"]),
        "within_budget": budget[label]["within_budget"],
        "wall_seconds": budget[label]["wall_seconds"],
        "benchmark_runner_sha256": implementation["runner_sha256"],
        "benchmark_source_inventory_sha256": implementation["source_inventory_sha256"],
        "benchmark_source_file_count": implementation["source_file_count"],
    }


class TestSummaryReceiptConsistency(unittest.TestCase):
    def setUp(self) -> None:
        self.seed = SEED_1
        self.scores = seed_pair(passing_treated_arm())
        self.budget = {
            label: {"within_budget": True, "wall_seconds": 100.0 + index}
            for index, label in enumerate(cb.MODEL_ORDER)
        }
        self.implementation = synthetic_implementation()
        self.events = {
            label: synthetic_event(
                label, self.scores, self.budget, self.implementation
            )
            for label in cb.MODEL_ORDER
        }
        self.summary = {
            "schema_version": 1,
            "name": cb.FROZEN_NAME,
            "tier": cb.FROZEN_TIER,
            "think_budget": cb.FROZEN_THINK_BUDGET,
            "seed": self.seed,
            "model_order": list(cb.MODEL_ORDER),
            "promoted": None,
            "benchmark_data_read": False,
            "gateway_sha256": cb.GATEWAY_SHA256,
            "discovery_summary_sha256": cb.DISCOVERY["summary_sha256"],
            "scores": {
                label: {
                    "aggregate": self.scores[label]["aggregate"],
                    "per_family": dict(self.scores[label]["per_family"]),
                }
                for label in cb.MODEL_ORDER
            },
            "budget": self.budget,
            "benchmark_implementation": dict(self.implementation),
        }

    def test_consistent_summary_passes(self) -> None:
        cb.require_summary_consistency(self.seed, self.summary, self.events)

    def test_swapped_receipt_scores_refuse(self) -> None:
        # MAJOR-1 verified consequence: swapping one receipt flipped the
        # verdict undetectably; now the receipt must equal the sealed
        # summary's recorded block.
        events = {label: dict(event) for label, event in self.events.items()}
        events["hygiene_explore"]["aggregate"] = 0.9999
        with self.assertRaises(ValueError):
            cb.require_summary_consistency(self.seed, self.summary, events)

    def test_swapped_family_score_refuses(self) -> None:
        events = {label: dict(event) for label, event in self.events.items()}
        events["hygiene_explore"]["per_family"] = {
            **events["hygiene_explore"]["per_family"], "menders": 1.0,
        }
        with self.assertRaises(ValueError):
            cb.require_summary_consistency(self.seed, self.summary, events)

    def test_swapped_budget_refuses(self) -> None:
        events = {label: dict(event) for label, event in self.events.items()}
        events["base"]["within_budget"] = False
        with self.assertRaises(ValueError):
            cb.require_summary_consistency(self.seed, self.summary, events)

    def test_swapped_implementation_refuses(self) -> None:
        events = {label: dict(event) for label, event in self.events.items()}
        events["base"]["benchmark_runner_sha256"] = "e" * 64
        with self.assertRaises(ValueError):
            cb.require_summary_consistency(self.seed, self.summary, events)

    def test_wrong_seed_refuses(self) -> None:
        with self.assertRaises(ValueError):
            cb.require_summary_consistency(SEED_2, self.summary, self.events)

    def test_tampered_summary_pins_refuse(self) -> None:
        for drift in (
            {"gateway_sha256": "f" * 64},
            {"discovery_summary_sha256": "f" * 64},
            {"promoted": "hygiene_explore"},
            {"benchmark_data_read": True},
            {"model_order": ["hygiene_explore", "base"]},
        ):
            summary = {**self.summary, **drift}
            with self.assertRaises(ValueError, msg=str(drift)):
                cb.require_summary_consistency(self.seed, summary, self.events)


if __name__ == "__main__":
    unittest.main()
