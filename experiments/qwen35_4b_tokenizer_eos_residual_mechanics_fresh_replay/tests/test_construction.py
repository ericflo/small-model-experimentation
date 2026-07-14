from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

from identity import (  # noqa: E402
    PARENT_COLLISION_MANIFEST,
    TASK_NAMESPACE,
    TRANSPORT_REQUEST_NAMESPACE,
    canonical_sha256,
    namespaced_task_id,
)


CONSTRUCT_SPEC = importlib.util.spec_from_file_location(
    "fresh_replay_construct_test", EXP / "scripts/construct.py"
)
if CONSTRUCT_SPEC is None or CONSTRUCT_SPEC.loader is None:
    raise RuntimeError("cannot load construction module")
CONSTRUCT = importlib.util.module_from_spec(CONSTRUCT_SPEC)
CONSTRUCT_SPEC.loader.exec_module(CONSTRUCT)


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
        functions = summary["common_function_fingerprints"]
        self.assertEqual(len(functions), 72)
        self.assertEqual(len(set(functions)), 72)
        self.assertEqual(summary["excluded_parent_function_fingerprints"], 72)
        self.assertEqual(summary["parent_function_fingerprint_overlap"], 0)

    def test_parent_boundary_is_one_authenticated_hash_only_manifest(self) -> None:
        manifest_path = ROOT / PARENT_COLLISION_MANIFEST["path"]
        manifest = json.loads(manifest_path.read_text())
        self.assertEqual(
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            PARENT_COLLISION_MANIFEST["sha256"],
        )
        self.assertEqual(len(manifest["administrative_sources"]), 8)
        self.assertEqual(len(manifest["common_function_fingerprints"]), 72)
        self.assertEqual(len(manifest["public_instance_fingerprints"]), 72)
        self.assertEqual(len(manifest["request_ids"]), 2952)
        self.assertEqual(len(manifest["seed_key_sha256s"]), 2952)
        self.assertEqual(len(manifest["derived_runner_seeds"]), 5904)
        self.assertEqual(len(manifest["prompt_sha256s"]), 1824)
        self.assertEqual(len(manifest["prompt_token_sequence_sha256s"]), 3648)
        for name in (
            "benchmark_files_read",
            "hidden_files_read",
            "parent_raw_sampled_bundles_read",
        ):
            self.assertEqual(manifest[name], [])
        summary = json.loads((EXP / "runs/construction/summary.json").read_text())
        self.assertEqual(
            summary["parent_read_receipt"],
            {
                PARENT_COLLISION_MANIFEST["path"]: {
                    "sha256": PARENT_COLLISION_MANIFEST["sha256"],
                    "purpose": "authenticated_hash_only_parent_collision_domains",
                }
            },
        )

    def test_parent_collision_receipt_repair_changes_only_administrative_bindings(self) -> None:
        repair_path = EXP / "runs/construction/parent_collision_receipt_repair.json"
        repair = json.loads(repair_path.read_text())
        self.assertEqual(
            repair["decision"], "PARENT_COLLISION_RECEIPT_REPAIR_PASS"
        )
        self.assertTrue(repair["collision_domains_exactly_unchanged"])
        self.assertEqual(repair["new_manifest_sha256"], PARENT_COLLISION_MANIFEST["sha256"])
        for field in (
            "hidden_files_read",
            "local_key_files_read",
            "benchmark_files_read",
            "parent_raw_sampled_bundles_read",
        ):
            self.assertEqual(repair[field], [])
        self.assertFalse(repair["model_loaded"])
        self.assertEqual(repair["model_calls"], 0)
        self.assertEqual(repair["sampled_model_outputs_read"], 0)
        for relative, entry in repair["receipt_migrations"].items():
            self.assertEqual(
                hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
                entry["new_sha256"],
            )

        old_payload = subprocess.run(
            [
                "git",
                "show",
                (
                    repair["old_git_object"]
                    + ":"
                    + PARENT_COLLISION_MANIFEST["path"]
                ),
            ],
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        self.assertEqual(
            hashlib.sha256(old_payload).hexdigest(), repair["old_manifest_sha256"]
        )
        old_manifest = json.loads(old_payload)
        new_manifest = json.loads(
            (ROOT / PARENT_COLLISION_MANIFEST["path"]).read_text()
        )
        self.assertEqual(
            {key: value for key, value in old_manifest.items() if key != "administrative_sources"},
            {key: value for key, value in new_manifest.items() if key != "administrative_sources"},
        )
        exporter_source = (
            EXP / "scripts/export_parent_collision_manifest.py"
        ).read_text()
        self.assertNotIn("parent.parent_inventory(", exporter_source)
        self.assertIn("install_repository_read_firewall()", exporter_source)

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
        self.assertEqual(freshness["parent_task_id_overlap"], 0)
        self.assertEqual(freshness["parent_user_prompt_overlap"], 0)
        self.assertEqual(
            freshness["parent_rendered_prompt_token_sequence_overlap"], 0
        )
        self.assertEqual(freshness["unique_user_prompts"], 1824)
        self.assertEqual(
            freshness["unique_rendered_prompt_token_sequences"], 3648
        )
        self.assertTrue(freshness["tokenizer_loaded"])
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
        self.assertEqual(len(receipt["local_key_sha256"]), 64)
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", str(key.relative_to(ROOT))],
            cwd=ROOT,
            check=False,
        )
        self.assertEqual(ignored.returncode, 0)

    def test_completed_rerun_authenticates_without_opening_key(self) -> None:
        key = CONSTRUCT.HIDDEN_KEY
        original_open = Path.open

        def audited_open(path: Path, *args: object, **kwargs: object):
            mode = str(args[0] if args else kwargs.get("mode", "r"))
            if Path(path) == key and "r" in mode:
                raise AssertionError("completed construction attempted to reread key")
            return original_open(path, *args, **kwargs)

        with mock.patch.object(Path, "open", audited_open):
            summary = CONSTRUCT.validate_completed_construction()
        self.assertIsNotNone(summary)
        self.assertEqual(summary["decision"], "CONSTRUCTION_PASS")


if __name__ == "__main__":
    unittest.main()
