import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
DATA = EXP / "data"
SEEDS = (88022, 88023, 88024, 88025)
ROWS_PER_SCREEN = 104
RETENTION_PER_KIND = 8
RETENTION_KINDS = {
    "u_abstain",
    "u_count",
    "u_execute",
    "u_induct",
    "u_optimize",
    "u_order",
    "u_probe",
    "u_repair",
    "u_route",
    "u_select",
    "u_state",
    "u_trace",
    "u_verify",
}


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def canonical_message(row: dict) -> str:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


class ScreenUniquenessTests(unittest.TestCase):
    """The four frozen screens must be internally and mutually duplicate-free.

    Fail closed: if the screens have not been generated yet, these tests
    fail rather than skip.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = {}
        cls.inputs = {}
        for seed in SEEDS:
            tasks_path = DATA / f"local_tasks_seed{seed}.jsonl"
            input_path = DATA / f"local_input_seed{seed}.jsonl"
            assert tasks_path.is_file(), f"missing frozen screen: {tasks_path}"
            assert input_path.is_file(), f"missing runner input: {input_path}"
            cls.tasks[seed] = load_jsonl(tasks_path)
            cls.inputs[seed] = load_jsonl(input_path)

    def test_each_screen_has_104_rows_with_fresh_prefixed_unique_ids(self) -> None:
        for seed in SEEDS:
            rows = self.tasks[seed]
            self.assertEqual(len(rows), ROWS_PER_SCREEN)
            task_ids = [row["task_id"] for row in rows]
            self.assertEqual(len(set(task_ids)), ROWS_PER_SCREEN)
            for task_id in task_ids:
                self.assertTrue(task_id.startswith(f"ret{seed}_"), task_id)

    def test_each_screen_is_kind_balanced_eight_per_skill(self) -> None:
        for seed in SEEDS:
            kinds = Counter(row["kind"] for row in self.tasks[seed])
            self.assertEqual(set(kinds), RETENTION_KINDS)
            self.assertEqual(
                set(kinds.values()), {RETENTION_PER_KIND}, f"screen {seed}"
            )

    def test_no_duplicate_prompts_within_any_screen(self) -> None:
        for seed in SEEDS:
            messages = {canonical_message(row) for row in self.tasks[seed]}
            self.assertEqual(len(messages), ROWS_PER_SCREEN, f"screen {seed}")

    def test_no_duplicate_prompts_across_the_four_screens(self) -> None:
        combined: set[str] = set()
        for seed in SEEDS:
            screen_messages = {canonical_message(row) for row in self.tasks[seed]}
            self.assertFalse(
                combined & screen_messages,
                f"screen {seed} repeats an earlier screen's prompt",
            )
            combined |= screen_messages
        self.assertEqual(len(combined), ROWS_PER_SCREEN * len(SEEDS))

    def test_no_duplicate_task_ids_across_the_four_screens(self) -> None:
        combined: set[str] = set()
        for seed in SEEDS:
            ids = {row["task_id"] for row in self.tasks[seed]}
            self.assertFalse(combined & ids)
            combined |= ids
        self.assertEqual(len(combined), ROWS_PER_SCREEN * len(SEEDS))

    def test_runner_inputs_are_oracle_free_and_aligned(self) -> None:
        for seed in SEEDS:
            tasks = self.tasks[seed]
            inputs = self.inputs[seed]
            self.assertEqual(len(inputs), ROWS_PER_SCREEN)
            for task, model_row in zip(tasks, inputs):
                self.assertEqual(set(model_row), {"id", "messages", "meta"})
                self.assertEqual(model_row["id"], task["task_id"])
                self.assertEqual(model_row["messages"], task["messages"])
                self.assertEqual(
                    set(model_row["meta"]), {"kind", "surface", "seed", "instrument"}
                )
                self.assertEqual(model_row["meta"]["seed"], seed)
                self.assertEqual(model_row["meta"]["instrument"], "retention")
                rendered = json.dumps(model_row, sort_keys=True)
                self.assertNotIn('"answer"', rendered)
                self.assertNotIn('"think"', rendered)

    def test_design_receipt_pins_match_the_frozen_files(self) -> None:
        receipt_path = DATA / "local_design_receipt.json"
        self.assertTrue(receipt_path.is_file(), "missing frozen design receipt")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.assertEqual(receipt["seeds"], list(SEEDS))
        self.assertEqual(receipt["rows_per_screen"], ROWS_PER_SCREEN)
        self.assertTrue(receipt["calibration_cell"])
        self.assertTrue(receipt["trains_nothing"])
        freshness = receipt["freshness"]
        self.assertEqual(
            freshness["unique_local_messages"], ROWS_PER_SCREEN * len(SEEDS)
        )
        self.assertEqual(
            set(freshness["cross_screen_overlap"].values()), {0}
        )
        for seed in SEEDS:
            screen = receipt["screens"][str(seed)]
            for key, path in (
                ("source", DATA / f"local_tasks_seed{seed}.jsonl"),
                ("runner_input", DATA / f"local_input_seed{seed}.jsonl"),
            ):
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(screen[key]["sha256"], digest, f"{seed} {key}")


if __name__ == "__main__":
    unittest.main()
