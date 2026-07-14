from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from identity import (  # noqa: E402
    TASK_NAMESPACE,
    TRANSPORT_REQUEST_NAMESPACE,
    canonical_sha256,
    namespaced_task_id,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


class ConstructionTests(unittest.TestCase):
    def test_task_ids_splits_and_parent_freshness(self) -> None:
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
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        fingerprints = summary["public_instance_fingerprints"]
        self.assertEqual(len(fingerprints), 72)
        self.assertEqual(len(set(fingerprints)), 72)
        self.assertEqual(summary["excluded_parent_public_fingerprints"], 72)
        self.assertEqual(summary["parent_public_overlap"], 0)

    def test_calibration_aliases_and_strata_are_balanced_per_arity(self) -> None:
        rows = read_jsonl(EXP / "data/procedural/calibration_public.jsonl")
        expected_strata = {"single": 8, "double": 8, "triple": 4, "quad": 4}
        for arity in (2, 3):
            subset = [row for row in rows if row["arity"] == arity]
            self.assertEqual(len(subset), 24)
            self.assertEqual(
                {name: sum(row["stratum"] == name for row in subset) for name in expected_strata},
                expected_strata,
            )
            for position in range(arity):
                self.assertEqual(
                    sorted(row["expected_aliases"][position] for row in subset),
                    list("ABCDEFGHIJKLMNOPQRSTUVWX"),
                )

    def test_request_ids_suffix_pairing_and_transport_namespace(self) -> None:
        rows_by_name = {
            path.stem: read_jsonl(path)
            for path in sorted((EXP / "runs/prepared").glob("*_requests.jsonl"))
        }
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
        self.assertTrue(
            all(
                row["meta"]["seed_key"][:2]
                == [TRANSPORT_REQUEST_NAMESPACE, "transport"]
                for row in rows_by_name["transport_requests"]
            )
        )

    def test_receipt_records_derived_seed_and_forbidden_read_firewalls(self) -> None:
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        freshness = summary["request_freshness"]
        self.assertEqual(freshness["derived_runner_seeds"], 5904)
        self.assertEqual(freshness["parent_derived_runner_seed_overlap"], 0)
        self.assertEqual(freshness["transport_request_id_overlap_current_families"], 0)
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

    def test_hidden_gold_is_ciphertext_only_and_key_is_ignored(self) -> None:
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        receipt = summary["hidden_ciphertext"]
        ciphertext = ROOT / receipt["ciphertext_path"]
        key = ROOT / receipt["local_key_path"]
        self.assertTrue(ciphertext.is_file())
        self.assertTrue(key.is_file())
        self.assertFalse((EXP / "data/procedural/mechanics_gold.jsonl").exists())
        self.assertEqual(hashlib.sha256(ciphertext.read_bytes()).hexdigest(), receipt["ciphertext_sha256"])
        self.assertEqual(hashlib.sha256(key.read_bytes()).hexdigest(), receipt["local_key_sha256"])
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(key.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)


if __name__ == "__main__":
    unittest.main()
