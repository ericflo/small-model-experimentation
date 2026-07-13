from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import identity  # noqa: E402


class IdentityTests(unittest.TestCase):
    def test_fresh_seed_domain_is_exact_unique_and_parent_disjoint(self) -> None:
        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        seeds = [int(value) for value in config["seeds"].values()]
        self.assertEqual(set(seeds), set(range(2026072700, 2026072710)))
        self.assertEqual(len(seeds), len(set(seeds)))
        self.assertFalse(set(seeds) & set(range(2026072600, 2026072610)))

    def test_complete_parent_lineage_authenticates(self) -> None:
        observed = identity.verify_parent_lineage(ROOT)
        self.assertEqual(observed, dict(sorted(identity.PARENT_LINEAGE.items())))
        self.assertEqual(len(observed), 13)

    def test_parent_lineage_rejects_hash_drift_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            target = root / "target"
            target.write_text("value")
            link = root / "link"
            link.symlink_to(target)
            with self.assertRaisesRegex(RuntimeError, "symlink"):
                identity._verified_regular_file(
                    root, "link", identity.file_sha256(target)
                )
            with self.assertRaisesRegex(RuntimeError, "hash drift"):
                identity._verified_regular_file(root, "target", "0" * 64)

    def test_task_and_request_namespaces_are_operational(self) -> None:
        task_id = identity.namespaced_task_id(
            identity.TASK_NAMESPACE, "mechanics", 7
        )
        self.assertEqual(
            task_id,
            "materialized-residual-fresh-replication-v1/mechanics/00007",
        )
        key = identity.request_seed_key(
            identity.REQUEST_NAMESPACE, "suffix", task_id, "reverse"
        )
        self.assertEqual(key[0], identity.REQUEST_NAMESPACE)
        self.assertEqual(len(identity.request_id(key)), 64)
        with self.assertRaises(ValueError):
            identity.request_seed_key("parent-v1", "suffix", task_id, "reverse")

    def test_public_instance_omits_only_administrative_identity(self) -> None:
        task = {
            "task_id": "one",
            "depth": 3,
            "viability_live_alias": "A",
            "visible": [{"input": [1], "output": [2]}],
            "unlabeled_probe_inputs": [[3]],
        }
        administrative = copy.deepcopy(task)
        administrative["task_id"] = "two"
        administrative["viability_live_alias"] = "B"
        self.assertEqual(
            identity.public_instance_fingerprint(task),
            identity.public_instance_fingerprint(administrative),
        )
        substantive = copy.deepcopy(task)
        substantive["visible"][0]["output"] = [9]
        self.assertNotEqual(
            identity.public_instance_fingerprint(task),
            identity.public_instance_fingerprint(substantive),
        )


if __name__ == "__main__":
    unittest.main()
