from __future__ import annotations

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_pipeline import (  # noqa: E402
    canonical_rows_receipt,
    validate_parent_data_parity,
)


class ParentDataParityTests(unittest.TestCase):
    def test_canonical_receipt_covers_ids_content_and_order(self) -> None:
        rows = [
            {"id": "one", "prompt": "alpha", "nested": {"b": 2, "a": 1}},
            {"id": "two", "prompt": "beta", "value": 3},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    # Deliberately noncanonical serialization; receipt semantics
                    # are parsed row content rather than gzip/container details.
                    handle.write(json.dumps(row) + "\n")
            baseline = canonical_rows_receipt(path)
            self.assertEqual(baseline["rows"], 2)

            rows[0]["id"] = "changed"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
            self.assertNotEqual(
                canonical_rows_receipt(path)["canonical_rows_sha256"],
                baseline["canonical_rows_sha256"],
            )

            with gzip.open(path, "wt", encoding="utf-8") as handle:
                for row in reversed(rows):
                    handle.write(json.dumps(row) + "\n")
            reordered = canonical_rows_receipt(path)
            self.assertNotEqual(
                reordered["canonical_rows_sha256"],
                canonical_rows_receipt_from_rows(rows),
            )

    def test_consumer_recomputes_rows_and_rejects_copied_pass_metadata(self) -> None:
        rows = [{"id": "one", "prompt": "parent row"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "train.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[0]) + "\n")
            expected = {"train": canonical_rows_receipt(path)}
            import hashlib

            contract_digest = hashlib.sha256(
                json.dumps(expected, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest()
            config = {
                "parent_experiment": {
                    "data_manifest_sha256": "a" * 64,
                },
                "parent_data_contract": {
                    "parent_experiment_id": "parent",
                    "canonicalization": "sorted_compact_json_plus_lf",
                    "splits": expected,
                },
            }
            manifest = {
                "files": {"train": {"path": path.name}},
                "parent_data_parity": {
                    "status": "PARENT_DATA_PARITY_PASS",
                    "parent_experiment_id": "parent",
                    "canonicalization": "sorted_compact_json_plus_lf",
                    "frozen_contract_sha256": contract_digest,
                    "frozen_contract_match": True,
                    "parent_artifacts_available": False,
                    "direct_parent_artifact_match": None,
                    "parent_manifest_sha256": None,
                    "splits": expected,
                },
            }
            validate_parent_data_parity(config, root, manifest)

            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps({"id": "changed", "prompt": "other"}) + "\n")
            with self.assertRaisesRegex(RuntimeError, "current prepared rows differ"):
                validate_parent_data_parity(config, root, manifest)

            # Restoring rows cannot rescue a parity receipt with the wrong
            # frozen parent identity/digest metadata.
            with gzip.open(path, "wt", encoding="utf-8") as handle:
                handle.write(json.dumps(rows[0]) + "\n")
            manifest["parent_data_parity"]["parent_experiment_id"] = "other"
            with self.assertRaisesRegex(RuntimeError, "parity metadata mismatch"):
                validate_parent_data_parity(config, root, manifest)


def canonical_rows_receipt_from_rows(rows: list[dict]) -> str:
    """Test helper matching the registered canonical byte stream."""
    import hashlib

    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


if __name__ == "__main__":
    unittest.main()
