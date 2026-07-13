from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data_pipeline import build_datasets, read_jsonl
from src.substrate import (
    FAMILIES,
    generate_counterfactual_pair,
    generate_example,
    trajectory_targets,
    verify_example,
)


class SubstrateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(ROOT / "configs" / "smoke.yaml")
        cls.architecture = cls.config["architecture"]
        cls.substrate = cls.config["substrate"]

    def make(self, seed: int, family: str, depth: int) -> dict:
        return generate_example(
            seed=seed,
            split="test",
            family=family,
            template="ledger",
            depth=depth,
            node_count=self.substrate["node_count"],
            checksum_modulus=self.substrate["checksum_modulus"],
            num_choices=self.substrate["num_choices"],
            state_token=self.architecture["state_token"],
            state_slots=self.architecture["state_slots"],
            max_attempts=self.substrate["max_generation_attempts"],
        )

    def test_generation_is_deterministic_and_exact_for_every_family(self) -> None:
        for index, family in enumerate(FAMILIES):
            with self.subTest(family=family):
                first = self.make(100 + index, family, 7)
                second = self.make(100 + index, family, 7)
                self.assertEqual(first, second)
                verify_example(first, self.architecture["state_token"], self.architecture["state_slots"])
                self.assertEqual(len(first["trajectory"]), 8)
                self.assertEqual(len({tuple(sorted(x.items())) for x in first["trajectory"]}), 8)
                query_values = [state[first["query_kind"]] for state in first["trajectory"]]
                self.assertNotIn(query_values[-1], query_values[:-1])

    def test_dataset_archives_are_deterministic_across_python_hash_seeds(self) -> None:
        code = """
import json
import sys
from pathlib import Path
from src.config import load_config
from src.data_pipeline import build_datasets
root = Path.cwd()
manifest = build_datasets(load_config(root / "configs" / "smoke.yaml"), sys.argv[1])
print(json.dumps(
    {name: receipt["sha256"] for name, receipt in manifest["files"].items()},
    sort_keys=True,
    separators=(",", ":"),
))
"""
        with tempfile.TemporaryDirectory() as directory:
            outputs = []
            for index, hash_seed in enumerate(("1", "987654")):
                environment = os.environ.copy()
                environment["PYTHONHASHSEED"] = hash_seed
                environment["PYTHONPATH"] = str(ROOT)
                outputs.append(
                    subprocess.check_output(
                        [sys.executable, "-c", code, str(Path(directory) / str(index))],
                        cwd=ROOT,
                        env=environment,
                        text=True,
                    )
                )
        self.assertEqual(outputs[0], outputs[1])
        self.assertIn("train", json.loads(outputs[0]))

    def test_query_is_causally_after_every_state_slot(self) -> None:
        row = self.make(201, "phase_branch", 4)
        prompt = row["prompt"]
        self.assertEqual(prompt.count(self.architecture["state_token"]), self.architecture["state_slots"])
        self.assertLess(prompt.rindex(self.architecture["state_token"]), prompt.index("Query:"))

    def test_targets_halt_after_requested_depth(self) -> None:
        row = self.make(202, "checksum_branch", 2)
        targets = trajectory_targets(row, 6)
        first_transition_node = row["trajectory"][1]["node"]
        self.assertEqual(
            targets["node"][0], row["table_order"].index(first_transition_node)
        )
        for name in ("node", "phase", "checksum"):
            self.assertEqual(targets[name][1:], [targets[name][-1]] * 5)

    def test_counterfactual_pair_shares_world_and_has_distinct_consequence(self) -> None:
        first, second = generate_counterfactual_pair(
            seed=203,
            split="cf",
            family="phase_branch",
            template="prose",
            depth=6,
            node_count=self.substrate["node_count"],
            checksum_modulus=self.substrate["checksum_modulus"],
            num_choices=self.substrate["num_choices"],
            state_token=self.architecture["state_token"],
            state_slots=self.architecture["state_slots"],
            max_attempts=self.substrate["max_generation_attempts"],
            query_kind="checksum",
        )
        self.assertEqual(first["pair_id"], second["pair_id"])
        self.assertEqual(first["world"], second["world"])
        self.assertEqual(first["table_order"], second["table_order"])
        self.assertEqual(first["choices"], second["choices"])
        self.assertEqual(first["query_kind"], "checksum")
        self.assertEqual(second["query_kind"], "checksum")
        self.assertNotEqual(first["initial"], second["initial"])
        self.assertEqual(first["initial"]["node"], second["initial"]["node"])
        self.assertNotEqual(first["initial"]["phase"], second["initial"]["phase"])
        self.assertNotEqual(first["initial"]["checksum"], second["initial"]["checksum"])
        first_value = first["choices"][first["correct_choice"]]
        second_value = second["choices"][second["correct_choice"]]
        self.assertNotEqual(first_value, second_value)
        self.assertIn(second_value, first["choices"])
        self.assertIn(first_value, second["choices"])
        for row in (first, second):
            query_values = [state[row["query_kind"]] for state in row["trajectory"]]
            self.assertNotIn(query_values[-1], query_values[:-1])

    def test_smoke_dataset_manifest_and_cross_split_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_datasets(self.config, directory)
            self.assertEqual(manifest["cross_split_structural_duplicates"], 0)
            self.assertEqual(manifest["benchmark_files_read"], 0)
            self.assertEqual(manifest["files"]["train"]["rows"], 24)
            self.assertEqual(manifest["files"]["pilot_depth"]["rows"], 8)
            self.assertEqual(manifest["files"]["pilot_joint"]["rows"], 8)
            self.assertEqual(manifest["files"]["pilot_counterfactual"]["rows"], 4)
            self.assertEqual(manifest["files"]["pilot_validation"]["rows"], 8)
            self.assertEqual(len(manifest["source_contract_sha256"]), 64)
            for receipt in manifest["files"].values():
                self.assertEqual(receipt["query_kinds"].get("node"), receipt["rows"] // 2)
                self.assertEqual(
                    receipt["query_kinds"].get("checksum"), receipt["rows"] // 2
                )
                for cell in receipt["query_kind_grid"].values():
                    self.assertEqual(cell.get("node"), cell.get("checksum"))
            rows = read_jsonl(Path(directory) / "train.jsonl.gz")
            self.assertEqual(len(rows), 24)

    def test_compressed_dataset_hashes_are_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = build_datasets(self.config, first_dir)
            second = build_datasets(self.config, second_dir)
            self.assertEqual(first, second)
            self.assertEqual(
                {name: receipt["sha256"] for name, receipt in first["files"].items()},
                {name: receipt["sha256"] for name, receipt in second["files"].items()},
            )


if __name__ == "__main__":
    unittest.main()
