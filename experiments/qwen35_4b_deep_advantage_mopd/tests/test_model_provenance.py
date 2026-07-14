from __future__ import annotations

import contextlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import authorize_benchmark  # noqa: E402
import model_provenance  # noqa: E402
from io_utils import sha256_file  # noqa: E402


def _fixture(root: Path, *, tokenizer_marker: str = "source") -> dict:
    model = root / "model"
    model.mkdir(parents=True)
    contents = {
        "chat_template.jinja": "{{ messages }}\n",
        "config.json": '{"model_type":"qwen3_5"}\n',
        "generation_config.json": '{"max_length":16384}\n',
        "tokenizer.json": '{"model":{"type":"BPE"}}\n',
        "tokenizer_config.json": json.dumps({"profile": tokenizer_marker}) + "\n",
    }
    for name, value in contents.items():
        (model / name).write_text(value, encoding="utf-8")
    weight = model / "model.safetensors"
    weight.write_bytes(b"synthetic-weight")
    inference = [
        {"path": name, "sha256": sha256_file(model / name)}
        for name in model_provenance.INFERENCE_FILES
    ]
    receipt = {
        "method": "synthetic",
        "weight_files": [
            {"name": weight.name, "sha256": sha256_file(weight)}
        ],
        "inference_files": inference,
    }
    receipt_path = model / "merge_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    profile = {
        name: sha256_file(model / name)
        for name in model_provenance.LOAD_PROFILES["source"]
    }
    return {
        "model": model,
        "weight": weight,
        "receipt": receipt_path,
        "profile": profile,
    }


@contextlib.contextmanager
def _profiles(fixture: dict):
    with mock.patch.dict(
        model_provenance.LOAD_PROFILES,
        {"source": fixture["profile"], "local": fixture["profile"]},
        clear=True,
    ):
        yield


class ModelProvenanceTests(unittest.TestCase):
    def test_valid_source_local_and_legacy_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _fixture(Path(temporary))
            with _profiles(fixture):
                source = model_provenance.validate_model_checkpoint(
                    fixture["model"],
                    profile="source",
                    require_recorded_inference_inventory=True,
                )
                local = model_provenance.validate_model_checkpoint(
                    fixture["model"], profile="local"
                )
                self.assertEqual(
                    source["model_inference_inventory_sha256"],
                    local["model_inference_inventory_sha256"],
                )
                receipt = json.loads(fixture["receipt"].read_text())
                receipt.pop("inference_files")
                fixture["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
                model_provenance.validate_model_checkpoint(
                    fixture["model"], profile="local"
                )
                with self.assertRaisesRegex(ValueError, "lacks its inference"):
                    model_provenance.validate_model_checkpoint(
                        fixture["model"],
                        profile="source",
                        require_recorded_inference_inventory=True,
                    )

    def test_load_file_mutation_missing_and_extra_fail_closed(self):
        cases = {
            "weight_mutated": lambda row: row["weight"].write_bytes(b"changed"),
            "weight_missing": lambda row: row["weight"].unlink(),
            "config_mutated": lambda row: (row["model"] / "config.json").write_text(
                "{}\n", encoding="utf-8"
            ),
            "tokenizer_mutated": lambda row: (
                row["model"] / "tokenizer.json"
            ).write_text("{}\n", encoding="utf-8"),
            "tokenizer_config_missing": lambda row: (
                row["model"] / "tokenizer_config.json"
            ).unlink(),
            "generation_missing": lambda row: (
                row["model"] / "generation_config.json"
            ).unlink(),
            "chat_missing": lambda row: (
                row["model"] / "chat_template.jinja"
            ).unlink(),
            "extra_weight": lambda row: (
                row["model"] / "extra.safetensors"
            ).write_bytes(b"extra"),
            "extra_config": lambda row: (
                row["model"] / "special_tokens_map.json"
            ).write_text("{}\n", encoding="utf-8"),
            "nested_extra": lambda row: (
                row["model"] / "assets"
            ).mkdir(),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for index, (name, mutate) in enumerate(cases.items()):
                with self.subTest(name=name):
                    fixture = _fixture(root / str(index))
                    with _profiles(fixture):
                        mutate(fixture)
                        with self.assertRaises(ValueError):
                            model_provenance.validate_model_checkpoint(
                                fixture["model"], profile="source"
                            )

    def test_symlink_leaf_directory_ancestor_and_nonregular_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            leaf = _fixture(root / "leaf")
            with _profiles(leaf):
                target = root / "external-tokenizer.json"
                target.write_bytes((leaf["model"] / "tokenizer.json").read_bytes())
                (leaf["model"] / "tokenizer.json").unlink()
                (leaf["model"] / "tokenizer.json").symlink_to(target)
                with self.assertRaisesRegex(ValueError, "unsafe entry"):
                    model_provenance.validate_model_checkpoint(
                        leaf["model"], profile="source"
                    )

            directory = _fixture(root / "directory")
            with _profiles(directory):
                outside = root / "outside"
                outside.mkdir()
                (directory["model"] / "assets").symlink_to(
                    outside, target_is_directory=True
                )
                with self.assertRaisesRegex(ValueError, "unsafe entry"):
                    model_provenance.validate_model_checkpoint(
                        directory["model"], profile="source"
                    )

            ancestor = _fixture(root / "real")
            alias = root / "alias"
            alias.symlink_to(ancestor["model"], target_is_directory=True)
            with _profiles(ancestor), self.assertRaisesRegex(ValueError, "symlinked"):
                model_provenance.validate_model_checkpoint(alias, profile="source")

            special = _fixture(root / "special")
            with _profiles(special):
                (special["model"] / "tokenizer.json").unlink()
                os.mkfifo(special["model"] / "tokenizer.json")
                with self.assertRaisesRegex(ValueError, "unsafe entry"):
                    model_provenance.validate_model_checkpoint(
                        special["model"], profile="source"
                    )

    def test_source_receipt_path_hash_and_checkpoint_hashes_are_exact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixtures = {
                name: _fixture(root / name) for name in ("quick", "deep", "soup")
            }
            config = {
                "model": {
                    "id": model_provenance.MODEL_ID,
                    "revision": model_provenance.MODEL_REVISION,
                    "quick_teacher": str(fixtures["quick"]["model"]),
                    "deep_teacher": str(fixtures["deep"]["model"]),
                    "student_checkpoint": str(fixtures["soup"]["model"]),
                }
            }
            payload = {
                "schema_version": 1,
                "model": model_provenance.MODEL_ID,
                "revision": model_provenance.MODEL_REVISION,
            }
            for name, fixture in fixtures.items():
                payload[name] = {
                    "path": str(fixture["model"].resolve()),
                    "merge_receipt_sha256": sha256_file(fixture["receipt"]),
                    "model_sha256": sha256_file(fixture["weight"]),
                }
            receipt = root / "checkpoint_receipts.json"
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with _profiles(fixtures["quick"]):
                result = model_provenance.validate_source_checkpoint_receipts(
                    config,
                    receipt_path=receipt,
                    expected_receipt_sha256=sha256_file(receipt),
                    verify_source_commit=False,
                )
                self.assertEqual(set(result), {"quick", "deep", "soup"})
                stale = json.loads(receipt.read_text())
                stale["quick"]["path"] = str(root / "wrong")
                receipt.write_text(json.dumps(stale), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "receipt changed"):
                    model_provenance.validate_source_checkpoint_receipts(
                        config,
                        receipt_path=receipt,
                        expected_receipt_sha256=payload["quick"]["model_sha256"],
                        verify_source_commit=False,
                    )

    def test_source_receipt_commit_must_be_an_ancestor(self):
        with tempfile.TemporaryDirectory() as temporary:
            receipt = Path(temporary) / "checkpoint_receipts.json"
            receipt.write_text("{}\n", encoding="utf-8")
            with mock.patch.object(
                model_provenance.subprocess,
                "run",
                return_value=mock.Mock(returncode=1),
            ), self.assertRaisesRegex(ValueError, "commit is not an ancestor"):
                model_provenance.validate_source_checkpoint_receipts(
                    {},
                    receipt_path=receipt,
                    expected_receipt_sha256=sha256_file(receipt),
                )

    def test_round_merge_semantics_reject_stale_base_and_adapter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _fixture(root / "fixture")
            base = root / "base"
            adapter = root / "adapter"
            base.mkdir()
            adapter.mkdir()
            (base / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
            adapter_config = adapter / "adapter_config.json"
            adapter_weight = adapter / "adapter_model.safetensors"
            adapter_config.write_text("{}\n", encoding="utf-8")
            adapter_weight.write_bytes(b"adapter")
            receipt = json.loads(fixture["receipt"].read_text())
            receipt.update(
                {
                    "method": "explicit_composite_lora_merge",
                    "base_model": str(base.resolve()),
                    "adapter": str(adapter.resolve()),
                    "adapter_config_sha256": sha256_file(adapter_config),
                    "adapter_weights_sha256": sha256_file(adapter_weight),
                    "applied_lora_modules": 1,
                    "nonzero_lora_modules": 1,
                }
            )
            fixture["receipt"].write_text(json.dumps(receipt), encoding="utf-8")
            with _profiles(fixture):
                authorize_benchmark._audit_merge_receipt(
                    fixture["model"], base, adapter
                )
                for field in ("base_model", "adapter"):
                    changed = json.loads(fixture["receipt"].read_text())
                    changed[field] = str(root / "wrong")
                    fixture["receipt"].write_text(
                        json.dumps(changed), encoding="utf-8"
                    )
                    with self.assertRaisesRegex(ValueError, "semantically stale"):
                        authorize_benchmark._audit_merge_receipt(
                            fixture["model"], base, adapter
                        )
                    changed[field] = str((base if field == "base_model" else adapter).resolve())
                    fixture["receipt"].write_text(
                        json.dumps(changed), encoding="utf-8"
                    )

    def test_arm_map_requires_exact_schema_and_soup_alias(self):
        row = {
            "model": "/model",
            "model_merge_receipt_sha256": "a" * 64,
            "model_config_sha256": "b" * 64,
            "model_inference_inventory_sha256": "c" * 64,
            "decode": "greedy",
        }
        arms = {name: dict(row) for name in model_provenance.CONFIRMATION_ARM_NAMES}
        arms["soup"]["model"] = "/soup"
        arms["soup_best8"] = {**arms["soup"], "decode": "sample8"}
        model_provenance.validate_confirmation_arm_map(arms)
        for mutation in (
            {name: value for name, value in arms.items() if name != "deep"},
            {**arms, "deep": {**arms["deep"], "extra": "x"}},
            {**arms, "soup_best8": {**arms["soup_best8"], "model": "/other"}},
        ):
            with self.assertRaises(ValueError):
                model_provenance.validate_confirmation_arm_map(mutation)


if __name__ == "__main__":
    unittest.main()
