from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from io_utils import validate_policy_cache_provenance  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CacheProvenanceTests(unittest.TestCase):
    def test_binds_all_policy_paths_and_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("frozen\n", encoding="utf-8")
            paths = {}
            models = {}
            for policy in ("quick", "deep", "soup"):
                path = root / policy
                path.mkdir()
                (path / "config.json").write_text(
                    json.dumps({"policy": policy}), encoding="utf-8"
                )
                (path / "merge_receipt.json").write_text(
                    json.dumps({"policy": policy}), encoding="utf-8"
                )
                paths[policy] = path
                models[policy] = {
                    "path": str(path),
                    "config_sha256": _sha(path / "config.json"),
                    "merge_receipt_sha256": _sha(path / "merge_receipt.json"),
                }
            config = {
                "model": {
                    "quick_teacher": str(paths["quick"]),
                    "deep_teacher": str(paths["deep"]),
                    "student_checkpoint": str(paths["soup"]),
                },
                "mopd": {"top_k": 50},
            }
            metadata = {
                "stage": "matched_all_policy_topk_cache",
                "config_sha256": _sha(config_path),
                "top_k": 50,
                "models": models,
            }
            validate_policy_cache_provenance(metadata, config, config_path)
            metadata["models"]["deep"]["path"] = str(paths["quick"])
            with self.assertRaisesRegex(ValueError, "models.deep.path"):
                validate_policy_cache_provenance(metadata, config, config_path)

    def test_rejects_stage_config_and_topk_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "config.yaml"
            config_path.write_text("frozen\n", encoding="utf-8")
            paths = {}
            models = {}
            for policy in ("quick", "deep", "soup"):
                path = root / policy
                path.mkdir()
                for filename in ("config.json", "merge_receipt.json"):
                    (path / filename).write_text(policy, encoding="utf-8")
                paths[policy] = path
                models[policy] = {
                    "path": str(path),
                    "config_sha256": _sha(path / "config.json"),
                    "merge_receipt_sha256": _sha(path / "merge_receipt.json"),
                }
            config = {
                "model": {
                    "quick_teacher": str(paths["quick"]),
                    "deep_teacher": str(paths["deep"]),
                    "student_checkpoint": str(paths["soup"]),
                },
                "mopd": {"top_k": 50},
            }
            metadata = {
                "stage": "wrong",
                "config_sha256": "0" * 64,
                "top_k": 49,
                "models": models,
            }
            with self.assertRaisesRegex(ValueError, "config_sha256.*stage.*top_k"):
                validate_policy_cache_provenance(metadata, config, config_path)


if __name__ == "__main__":
    unittest.main()
