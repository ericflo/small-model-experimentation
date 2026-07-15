import importlib.util
import sys
import unittest
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, EXP / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CHECK = load_module("fresh_check_local", "check_local.py")

KINDS = sorted(CHECK.EXPECTED_KINDS)


def synthetic_payload(per_arm_correct: dict[str, dict[str, int]],
                      cap_contacts: dict[str, int] | None = None,
                      route_answer: dict[str, str] | None = None) -> dict:
    """Build a receipt: per arm, per kind, the first N of 8 rows are correct."""
    caps = cap_contacts or {}
    route_answers = route_answer or {}
    rows = []
    for label in CHECK.ARMS:
        remaining_caps = caps.get(label, 0)
        for kind in KINDS:
            correct_n = per_arm_correct[label].get(kind, 6)
            for index in range(CHECK.PER_KIND):
                correct = index < correct_n
                parsed = "x"
                if kind == "u_route" and label in route_answers:
                    parsed = route_answers[label]
                    correct = False
                cap = False
                if remaining_caps > 0 and kind == "u_state":
                    cap = True
                    remaining_caps -= 1
                rows.append({
                    "adapter": label,
                    "task_id": f"{kind}_{index}",
                    "kind": kind,
                    "parsed": parsed,
                    "correct": bool(correct),
                    "cap_contact": bool(cap),
                })
    return {
        "seed": CHECK.SEED,
        "rows_per_arm": CHECK.ROWS,
        "labels": list(CHECK.ARMS),
        "rows": rows,
    }


def uniform(n: int) -> dict[str, int]:
    return {kind: n for kind in KINDS}


class PromotionGateTests(unittest.TestCase):
    def test_single_passing_candidate_promotes(self) -> None:
        payload = synthetic_payload({
            "replay_after_close_parent": uniform(6),
            "replay_repeat": uniform(6),
            "designed_fresh": uniform(7),
            "budget_commit": uniform(5),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "designed_fresh")
        self.assertEqual(result["eligible"], ["designed_fresh"])

    def test_tie_with_control_fails(self) -> None:
        payload = synthetic_payload({
            "replay_after_close_parent": uniform(6),
            "replay_repeat": uniform(7),
            "designed_fresh": uniform(7),
            "budget_commit": uniform(6),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])

    def test_two_passing_candidates_tiebreak_prefers_budget(self) -> None:
        payload = synthetic_payload({
            "replay_after_close_parent": uniform(5),
            "replay_repeat": uniform(5),
            "designed_fresh": uniform(7),
            "budget_commit": uniform(7),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(sorted(result["eligible"]), ["budget_commit", "designed_fresh"])
        self.assertEqual(result["promoted"], "budget_commit")

    def test_cap_contacts_break_second_level_ties(self) -> None:
        payload = synthetic_payload(
            {
                "replay_after_close_parent": uniform(5),
                "replay_repeat": uniform(5),
                "designed_fresh": uniform(7),
                "budget_commit": uniform(7),
            },
            cap_contacts={"budget_commit": 3},
        )
        result = CHECK.evaluate_promotion(payload)
        self.assertEqual(result["promoted"], "designed_fresh")

    def test_budget_answer_on_route_counts_as_abstention(self) -> None:
        payload = synthetic_payload(
            {
                "replay_after_close_parent": uniform(5),
                "replay_repeat": uniform(5),
                "designed_fresh": uniform(7),
                "budget_commit": uniform(7),
            },
            route_answer={"budget_commit": "BUDGET"},
        )
        result = CHECK.evaluate_promotion(payload)
        gate = result["gates"]["budget_commit"]
        self.assertEqual(gate["route_abstentions"], 8)
        self.assertFalse(gate["checks"]["route_abstentions_at_most_4_of_8"])
        self.assertEqual(result["promoted"], "designed_fresh")

    def test_parse_floor_enforced(self) -> None:
        payload = synthetic_payload({
            "replay_after_close_parent": uniform(5),
            "replay_repeat": uniform(5),
            "designed_fresh": uniform(7),
            "budget_commit": uniform(7),
        })
        for row in payload["rows"]:
            if row["adapter"] == "designed_fresh" and row["kind"] in ("u_state", "u_order"):
                row["parsed"] = None
                row["correct"] = False
        result = CHECK.evaluate_promotion(payload)
        self.assertNotIn("designed_fresh", result["eligible"])
        self.assertEqual(result["promoted"], "budget_commit")

    def test_target_kind_floor_enforced(self) -> None:
        correct = uniform(7)
        correct["u_induct"] = 3
        payload = synthetic_payload({
            "replay_after_close_parent": uniform(5),
            "replay_repeat": uniform(5),
            "designed_fresh": correct,
            "budget_commit": uniform(5),
        })
        result = CHECK.evaluate_promotion(payload)
        self.assertIsNone(result["promoted"])


if __name__ == "__main__":
    unittest.main()
