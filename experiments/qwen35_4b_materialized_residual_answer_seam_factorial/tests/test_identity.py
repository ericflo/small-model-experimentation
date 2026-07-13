from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

from identity import (  # noqa: E402
    TASK_NAMESPACE,
    canonical_sha256,
    namespaced_task_id,
    public_instance_fingerprint,
)

RUNNER_SPEC = importlib.util.spec_from_file_location(
    "factorial_seed_runner", SRC / "vllm_runner.py"
)
assert RUNNER_SPEC and RUNNER_SPEC.loader
seed_runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = seed_runner
RUNNER_SPEC.loader.exec_module(seed_runner)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class IdentityTests(unittest.TestCase):
    def test_task_ids_are_exact_and_splits_are_disjoint(self) -> None:
        calibration = read_jsonl(EXP / "data/procedural/calibration_public.jsonl")
        mechanics = read_jsonl(EXP / "data/procedural/mechanics_public.jsonl")
        self.assertEqual(
            [row["task_id"] for row in calibration],
            [
                namespaced_task_id(TASK_NAMESPACE, "calibration", index)
                for index in range(48)
            ],
        )
        self.assertEqual(
            [row["task_id"] for row in mechanics],
            [
                namespaced_task_id(TASK_NAMESPACE, "mechanics", index)
                for index in range(24)
            ],
        )
        self.assertFalse(
            {row["task_id"] for row in calibration}
            & {row["task_id"] for row in mechanics}
        )

    def test_72_public_instances_are_unique_and_parent_disjoint(self) -> None:
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        fingerprints = summary["public_instance_fingerprints"]
        self.assertEqual(len(fingerprints), 72)
        self.assertEqual(len(set(fingerprints)), 72)
        self.assertEqual(summary["parent_public_overlap"], 0)
        rows = [
            *read_jsonl(EXP / "data/procedural/calibration_public.jsonl"),
            *read_jsonl(EXP / "data/procedural/mechanics_public.jsonl"),
        ]
        # Calibration-public is a request table rather than the task table, so
        # the construction receipt is the canonical 72-fingerprint inventory.
        self.assertEqual(len(rows), 72)

    def test_request_ids_are_canonical_and_families_do_not_collide(self) -> None:
        files = sorted((EXP / "runs/prepared").glob("*_requests.jsonl"))
        rows_by_name = {path.stem: read_jsonl(path) for path in files}
        all_rows = [row for rows in rows_by_name.values() for row in rows]
        self.assertTrue(
            all(
                row["id"] == canonical_sha256(row["meta"]["seed_key"])
                for row in all_rows
            )
        )
        suffix_orders = [
            [row["id"] for row in rows_by_name[f"suffix_{name}_requests"]]
            for name in ("materialized", "name_only", "shuffled")
        ]
        self.assertEqual(suffix_orders[0], suffix_orders[1])
        self.assertEqual(suffix_orders[0], suffix_orders[2])
        representatives = [
            *rows_by_name["calibration_requests"],
            *rows_by_name["transport_requests"],
            *rows_by_name["suffix_materialized_requests"],
            *rows_by_name["direct_requests"],
        ]
        self.assertEqual(len({row["id"] for row in representatives}), 2952)

    def test_paired_seed_domains_are_unique_and_cross_domain_disjoint(self) -> None:
        representatives = []
        for name in (
            "calibration",
            "transport",
            "suffix_materialized",
            "direct",
        ):
            representatives.extend(
                read_jsonl(EXP / f"runs/prepared/{name}_requests.jsonl")
            )
        seeds = []
        for row in representatives:
            run_seed = (
                2026072801
                if row["meta"]["family"] == "calibration"
                else 2026072803
                if row["meta"]["family"] == "direct"
                else 2026072802
            )
            seeds.extend(
                (
                    seed_runner._stable_seed(run_seed, row["id"], -1, "thought"),
                    seed_runner._stable_seed(run_seed, row["id"], 0, "answer"),
                )
            )
        self.assertEqual(len(seeds), 2 * 2952)
        self.assertEqual(len(set(seeds)), len(seeds))

    def test_construction_receipt_records_no_forbidden_reads_or_calls(self) -> None:
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        for key in (
            "hidden_files_read",
            "qualification_files_read",
            "confirmation_files_read",
            "benchmark_files_read",
        ):
            self.assertEqual(summary[key], [])
        self.assertFalse(summary["model_loaded"])
        self.assertEqual(summary["model_calls"], 0)
        self.assertEqual(summary["sampled_model_outputs"], 0)
        self.assertEqual(summary["request_freshness"]["parent_request_id_overlap"], 0)
        self.assertEqual(summary["request_freshness"]["parent_seed_key_overlap"], 0)
        self.assertEqual(summary["request_freshness"]["parent_user_prompt_overlap"], 0)


if __name__ == "__main__":
    unittest.main()
