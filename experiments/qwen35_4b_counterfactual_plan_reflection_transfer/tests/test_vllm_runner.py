from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import yaml


RUNNER_PATH = Path(__file__).resolve().parents[1] / "src" / "vllm_runner.py"
sys.path.insert(0, str(RUNNER_PATH.parent))
import checkpoint_lineage  # noqa: E402
import merge_replay  # noqa: E402
import tensor_merge  # noqa: E402

SPEC = importlib.util.spec_from_file_location("template_vllm_runner", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class _FakeCudagraphMode:
    def __init__(self, name: str, decode: str, mixed: str, has_full: bool):
        self.name = name
        self._decode = decode
        self._mixed = mixed
        self._has_full = has_full

    def decode_mode(self) -> SimpleNamespace:
        return SimpleNamespace(name=self._decode)

    def mixed_mode(self) -> SimpleNamespace:
        return SimpleNamespace(name=self._mixed)

    def has_full_cudagraphs(self) -> bool:
        return self._has_full


def _compilation_config(
    sizes: tuple[int, ...] = (1, 2, 4, 8, 15),
    *,
    maximum: int = 15,
    mode: str = "FULL_DECODE_ONLY",
    decode: str = "FULL",
    mixed: str = "NONE",
    has_full: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        cudagraph_capture_sizes=list(sizes),
        max_cudagraph_capture_size=maximum,
        cudagraph_mode=_FakeCudagraphMode(mode, decode, mixed, has_full),
    )


class _FakeLoadWindowGuard:
    def __init__(self, roots: object, *, expected_content: object):
        self.roots = roots
        self.expected_content = expected_content
        self.receipt = {"synthetic": "immutable-load-window"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        return False

    def verify(self) -> dict[str, str]:
        return self.receipt

    def bind_authenticated_content(self, before_load: object, after_load: object) -> None:
        if before_load != self.expected_content or after_load != self.expected_content:
            raise RuntimeError("synthetic content commitment changed")


class EngineConfigCaptureGeometryTests(unittest.TestCase):
    @staticmethod
    def live_cache(num_blocks: int = 1100) -> dict:
        concurrency = num_blocks / 11
        return {
            "num_gpu_blocks": num_blocks,
            "block_size": 528,
            "kv_cache_size_tokens": int(concurrency * 4096),
            "kv_cache_max_concurrency": concurrency,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "mamba_block_size": 4096,
        }

    def test_live_hybrid_capacity_geometry_and_invocation_preflight(self) -> None:
        config = runner.EngineConfig(
            max_model_len=4096,
            max_num_seqs=64,
            max_num_batched_tokens=16384,
            cudagraph_capture_sizes=(1, 2, 4, 8, 16, 32, 64),
        )
        cache = self.live_cache()
        shape = runner._validate_live_cache_geometry(cache, config)
        self.assertEqual(shape["blocks_per_max_request"], 11)
        live = {
            "live_model": {"max_model_len": 4096, "dtype": "torch.bfloat16"},
            "live_scheduler": {
                "max_num_seqs": 64,
                "max_num_batched_tokens": 16384,
                "async_scheduling": False,
            },
            "live_parallel": {
                "world_size": 1,
                "tensor_parallel_size": 1,
                "data_parallel_size": 1,
            },
            "live_cache": cache,
            "cache_shape": shape,
        }
        receipt = runner._capacity_preflight(
            live=live,
            config=config,
            prompt_lengths=[500] * 144,
            sampling=runner.SamplingConfig(
                thinking="budget", thinking_budget=1024, answer_max_tokens=128, n=16
            ),
            close_tokens=2,
        )
        self.assertEqual(receipt["decision"], "LIVE_KV_CAPACITY_PASS")
        self.assertEqual(receipt["invocation"]["active_sequences"], 64)
        self.assertGreater(receipt["invocation"]["remaining_cache_blocks"], 0)

    def test_live_capacity_rejects_changed_geometry_and_overcommit(self) -> None:
        config = runner.EngineConfig(max_model_len=4096, max_num_seqs=64)
        changed = self.live_cache()
        changed["block_size"] = 512
        with self.assertRaisesRegex(RuntimeError, "geometry changed"):
            runner._validate_live_cache_geometry(changed, config)

        cache = self.live_cache(704)
        shape = runner._validate_live_cache_geometry(cache, config)
        live = {"live_cache": cache, "cache_shape": shape}
        with self.assertRaisesRegex(RuntimeError, "cannot fit"):
            runner._capacity_preflight(
                live=live,
                config=config,
                prompt_lengths=[3000] * 64,
                sampling=runner.SamplingConfig(
                    thinking="budget", thinking_budget=1024, answer_max_tokens=128, n=1
                ),
                close_tokens=2,
            )

    def test_merged_model_override_is_existing_and_mutually_exclusive(self) -> None:
        runner.EngineConfig(model_override=RUNNER_PATH.parent).validate()
        with self.assertRaisesRegex(ValueError, "existing merged-checkpoint"):
            runner.EngineConfig(model_override=RUNNER_PATH.parent / "missing").validate()
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            runner.EngineConfig(
                model_override=RUNNER_PATH.parent,
                adapter=RUNNER_PATH.parent,
            ).validate()
        with self.assertRaisesRegex(ValueError, "runtime LoRA adapters are forbidden"):
            runner.EngineConfig(adapter=RUNNER_PATH.parent).validate()

    def test_dummy_merged_override_and_self_issued_lineage_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "weights.safetensors").write_bytes(b"weights")
            (root / "merge_receipt.json").write_text(json.dumps({"schema_version": 3}))
            with self.assertRaisesRegex(ValueError, "embedded source lineage"):
                runner._validate_model_override(root)

    @staticmethod
    def _synthetic_config() -> dict:
        text = {key: 0 for key in checkpoint_lineage.PINNED_TEXT_CONFIG_KEYS}
        text["model_type"] = "synthetic_qwen_text"
        vision = {key: 0 for key in checkpoint_lineage.PINNED_VISION_CONFIG_KEYS}
        vision["model_type"] = "synthetic_qwen"
        return {
            "architectures": ["SyntheticQwen"],
            "image_token_id": 4,
            "model_type": "synthetic_qwen",
            "text_config": text,
            "tie_word_embeddings": True,
            "video_token_id": 5,
            "vision_config": vision,
            "vision_end_token_id": 3,
            "vision_start_token_id": 2,
        }

    @staticmethod
    def _write_safetensors(path: Path, tensors: dict[str, dict], *, sparse: bool = False) -> None:
        cursor = 0
        header = {}
        for key, value in tensors.items():
            nbytes = value["nbytes"]
            header[key] = {
                "dtype": value["dtype"],
                "shape": value["shape"],
                "data_offsets": [cursor, cursor + nbytes],
            }
            cursor += nbytes
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
        encoded += b" " * ((-len(encoded)) % 8)
        prefix = struct.pack("<Q", len(encoded)) + encoded
        if sparse:
            with path.open("wb") as handle:
                handle.write(prefix)
                handle.seek(len(prefix) + cursor - 1)
                handle.write(b"\0")
        else:
            path.write_bytes(prefix + b"\0" * cursor)

    def _synthetic_expected(self, tensors: dict[str, dict]) -> dict:
        config = self._synthetic_config()
        return {
            "model_type": config["model_type"],
            "architecture": config["architectures"][0],
            "config_structure_sha256": checkpoint_lineage.canonical_sha256(
                checkpoint_lineage.config_structure(config)
            ),
            "tensor_count": len(tensors),
            "tensor_bytes": sum(value["nbytes"] for value in tensors.values()),
            "tensor_inventory_sha256": checkpoint_lineage.canonical_sha256(tensors),
            "dtype_counts": {
                dtype: sum(value["dtype"] == dtype for value in tensors.values())
                for dtype in sorted({value["dtype"] for value in tensors.values()})
            },
        }

    def test_checkpoint_inventory_derives_metadata_and_rejects_index_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._synthetic_config()
            for name, value in {"config.json": config}.items():
                (root / name).write_text(json.dumps(value))
            shard = root / "model-00001-of-00001.safetensors"
            tensors = {
                "layer.weight": {"dtype": "F32", "shape": [2, 2], "nbytes": 16}
            }
            self._write_safetensors(shard, tensors)
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {"layer.weight": shard.name},
                    }
                )
            )
            expected = self._synthetic_expected(tensors)
            inventory = checkpoint_lineage.merged_checkpoint_inventory(
                root, expected=expected
            )
            self.assertEqual(inventory["tensor_count"], 1)
            self.assertGreaterEqual(
                inventory["allocated_shard_bytes"], inventory["logical_shard_bytes"]
            )
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 999},
                        "weight_map": {"layer.weight": shard.name},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "tensor inventory"):
                checkpoint_lineage.merged_checkpoint_inventory(root, expected=expected)
            index_path.write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 16},
                        "weight_map": {"forged.weight": shard.name},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "tensor placement|tensor keys"):
                checkpoint_lineage.merged_checkpoint_inventory(root, expected=expected)

    def test_tensor_level_merge_preserves_f32_and_exact_two_shard_contract(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            output = root / "merged"
            base.mkdir()
            config = self._synthetic_config()
            (base / "config.json").write_text(json.dumps(config))
            shard_names = [
                "model.safetensors-00001-of-00002.safetensors",
                "model.safetensors-00002-of-00002.safetensors",
            ]
            adapted = "model.language_model.layers.0.mlp.up_proj.weight"
            unchanged = "model.language_model.layers.0.mixer.A_log"
            storage = {
                str(base / shard_names[0]): {
                    adapted: torch.tensor(
                        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
                    )
                },
                str(base / shard_names[1]): {
                    unchanged: torch.tensor([5.0, 6.0], dtype=torch.float32)
                },
            }
            for shard_name in shard_names:
                (base / shard_name).write_bytes(b"synthetic handle")
            weight_map = {adapted: shard_names[0], unchanged: shard_names[1]}
            index = {"metadata": {"total_size": 16}, "weight_map": weight_map}
            index_path = base / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index))
            adapter_path = root / "adapter.safetensors"
            adapter_path.write_bytes(b"synthetic handle")
            key_a = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
            key_b = "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"
            storage[str(adapter_path)] = {
                    key_a: torch.tensor([[1.0, -1.0]], dtype=torch.float32),
                    key_b: torch.tensor([[2.0], [3.0]], dtype=torch.float32),
                }

            class FakeSafeOpen:
                def __init__(self, path: str, **_kwargs):
                    self.path = path

                def __enter__(self):
                    return self

                def __exit__(self, *_args):
                    return False

                def keys(self):
                    return storage[self.path].keys()

                def get_tensor(self, key: str):
                    return storage[self.path][key]

                def metadata(self):
                    return {"format": "pt"}

            def fake_save_file(tensors, path: str, metadata=None):
                self.assertEqual(metadata, {"format": "pt"})
                storage[path] = {key: value.clone() for key, value in tensors.items()}
                Path(path).write_bytes(b"synthetic serialized shard")

            fake_safetensors = types.ModuleType("safetensors")
            fake_safetensors.safe_open = FakeSafeOpen
            fake_safetensors_torch = types.ModuleType("safetensors.torch")
            fake_safetensors_torch.save_file = fake_save_file
            contract = {
                "implementation": "tensor_level_safetensors",
                "shard_policy": "preserve_exact_pinned_source_index",
                "expected_shards": shard_names,
                "unchanged_tensor_policy": "exact_value_shape_and_source_dtype",
                "adapted_tensor_math": (
                    "base_dtype(base.float32 + (B.float32 @ A.float32) * alpha/rank)"
                ),
                "local_trust_remote_code": False,
                "physical_allocation": "allocated_bytes_gte_logical_bytes",
            }
            with mock.patch.dict(
                sys.modules,
                {
                    "safetensors": fake_safetensors,
                    "safetensors.torch": fake_safetensors_torch,
                },
            ):
                result = tensor_merge.write_tensor_level_merge(
                    base_root=base,
                    base_index=index,
                    adapter_path=adapter_path,
                    output=output,
                    recipe={
                        "target_modules": ["up_proj"],
                        "lora_alpha": 1,
                        "lora_rank": 1,
                    },
                    contract=contract,
                )
            self.assertEqual(result["adapted_module_count"], 1)
            self.assertEqual(result["unchanged_tensor_count"], 1)
            self.assertEqual(
                checkpoint_lineage.sha256_file(output / "model.safetensors.index.json"),
                checkpoint_lineage.sha256_file(index_path),
            )
            expected_adapted = (
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16).float()
                + torch.tensor([[2.0], [3.0]]) @ torch.tensor([[1.0, -1.0]])
            ).to(torch.bfloat16)
            self.assertTrue(
                torch.equal(storage[str(output / shard_names[0])][adapted], expected_adapted)
            )
            preserved = storage[str(output / shard_names[1])][unchanged]
            self.assertEqual(preserved.dtype, torch.float32)
            self.assertTrue(torch.equal(preserved, torch.tensor([5.0, 6.0])))

            bad_contract = {**contract, "expected_shards": [shard_names[0]]}
            with mock.patch.dict(
                sys.modules,
                {
                    "safetensors": fake_safetensors,
                    "safetensors.torch": fake_safetensors_torch,
                },
            ), self.assertRaisesRegex(ValueError, "frozen two-shard policy"):
                tensor_merge.write_tensor_level_merge(
                    base_root=base,
                    base_index=index,
                    adapter_path=adapter_path,
                    output=root / "bad-merged",
                    recipe={
                        "target_modules": ["up_proj"],
                        "lora_alpha": 1,
                        "lora_rank": 1,
                    },
                    contract=bad_contract,
                )

    def test_exact_config_shard_surface_and_executable_injection_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self._synthetic_config()
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            shard_names = [
                "model.safetensors-00001-of-00002.safetensors",
                "model.safetensors-00002-of-00002.safetensors",
            ]
            tensors = {
                "first": {"dtype": "BF16", "shape": [2], "nbytes": 4},
                "second": {"dtype": "F32", "shape": [1], "nbytes": 4},
            }
            self._write_safetensors(root / shard_names[0], {"first": tensors["first"]})
            self._write_safetensors(root / shard_names[1], {"second": tensors["second"]})
            index = {
                "metadata": {"total_size": 8},
                "weight_map": {"first": shard_names[0], "second": shard_names[1]},
            }
            index_path = root / "model.safetensors.index.json"
            index_path.write_text(json.dumps(index))
            expected = self._synthetic_expected(tensors)
            expected.update(
                base_config_sha256=checkpoint_lineage.sha256_file(config_path),
                base_index_sha256=checkpoint_lineage.sha256_file(index_path),
                source_shard_sha256={name: "synthetic" for name in shard_names},
            )
            inventory = checkpoint_lineage.merged_checkpoint_inventory(
                root, expected=expected
            )
            self.assertEqual(inventory["dtype_counts"], {"BF16": 1, "F32": 1})
            self.assertEqual(inventory["shard_count"], 2)

            injected = {**config, "auto_map": {"AutoConfig": "unrelated.Config"}}
            config_path.write_text(json.dumps(injected))
            with self.assertRaisesRegex(ValueError, "exact pinned Qwen3.5-4B config"):
                checkpoint_lineage.merged_checkpoint_inventory(root, expected=expected)
            config_path.write_text(json.dumps(config))
            (root / "configuration_unrelated.py").write_text("raise RuntimeError\n")
            with self.assertRaisesRegex(ValueError, "unexpected runtime file"):
                checkpoint_lineage.merged_checkpoint_inventory(root, expected=expected)

    def test_real_safetensors_mixed_dtype_serialization_when_available(self) -> None:
        if importlib.util.find_spec("safetensors") is None:
            self.skipTest("real safetensors regression runs in the pinned experiment environment")
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            output = root / "merged"
            base.mkdir()
            (base / "config.json").write_text(json.dumps(self._synthetic_config()))
            shards = [
                "model.safetensors-00001-of-00002.safetensors",
                "model.safetensors-00002-of-00002.safetensors",
            ]
            adapted = "model.language_model.layers.0.mlp.up_proj.weight"
            unchanged = "model.language_model.layers.0.mixer.A_log"
            same_shard_unchanged = "model.language_model.embed_tokens.weight"
            base_adapted = torch.tensor(
                [[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16
            )
            base_unchanged = torch.tensor([5.0, 6.0], dtype=torch.float32)
            base_same_shard = torch.tensor([7.0, 8.0], dtype=torch.bfloat16)
            save_file(
                {adapted: base_adapted, same_shard_unchanged: base_same_shard},
                str(base / shards[0]),
            )
            save_file({unchanged: base_unchanged}, str(base / shards[1]))
            index = {
                "metadata": {"total_size": 20},
                "weight_map": {
                    adapted: shards[0],
                    same_shard_unchanged: shards[0],
                    unchanged: shards[1],
                },
            }
            (base / "model.safetensors.index.json").write_text(json.dumps(index))
            adapter_path = root / "adapter.safetensors"
            key_a = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
            key_b = "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"
            tensor_a = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
            tensor_b = torch.tensor([[2.0], [3.0]], dtype=torch.float32)
            save_file({key_a: tensor_a, key_b: tensor_b}, str(adapter_path))
            contract = {
                "implementation": "tensor_level_safetensors",
                "shard_policy": "preserve_exact_pinned_source_index",
                "expected_shards": shards,
                "unchanged_tensor_policy": "exact_value_shape_and_source_dtype",
                "adapted_tensor_math": (
                    "base_dtype(base.float32 + (B.float32 @ A.float32) * alpha/rank)"
                ),
                "local_trust_remote_code": False,
                "physical_allocation": "allocated_bytes_gte_logical_bytes",
            }
            tensor_merge.write_tensor_level_merge(
                base_root=base,
                base_index=index,
                adapter_path=adapter_path,
                output=output,
                recipe={"target_modules": ["up_proj"], "lora_alpha": 1, "lora_rank": 1},
                contract=contract,
            )
            self.assertEqual(
                json.loads((output / "model.safetensors.index.json").read_text()), index
            )
            with safe_open(str(output / shards[0]), framework="pt") as merged:
                expected = (
                    base_adapted.float() + tensor_b.float() @ tensor_a.float()
                ).to(torch.bfloat16)
                self.assertTrue(torch.equal(merged.get_tensor(adapted), expected))
                self.assertTrue(
                    torch.equal(merged.get_tensor(same_shard_unchanged), base_same_shard)
                )
            with safe_open(str(output / shards[1]), framework="pt") as merged:
                observed = merged.get_tensor(unchanged)
                self.assertEqual(observed.dtype, torch.float32)
                self.assertTrue(torch.equal(observed, base_unchanged))

    def test_partially_punched_safetensors_payload_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "punched.safetensors"
            self._write_safetensors(
                path,
                {"payload": {"dtype": "U8", "shape": [10_000_000], "nbytes": 10_000_000}},
            )
            with path.open("rb") as handle:
                header_bytes = struct.unpack("<Q", handle.read(8))[0]
            data_start = 8 + header_bytes
            punch_offset = ((data_start + 8191) // 4096) * 4096
            punched = subprocess.run(
                [
                    "fallocate",
                    "--punch-hole",
                    "--keep-size",
                    "--offset",
                    str(punch_offset),
                    "--length",
                    "4096",
                    str(path),
                ],
                capture_output=True,
                text=True,
            )
            if punched.returncode != 0:
                self.skipTest(f"filesystem cannot punch a payload hole: {punched.stderr}")
            with self.assertRaisesRegex(ValueError, "sparse or physically incomplete"):
                checkpoint_lineage._read_safetensors_header(path)

    def test_sparse_logical_shard_and_non_qwen_config_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in {"config.json": self._synthetic_config()}.items():
                (root / name).write_text(json.dumps(value))
            shard = root / "model-00001-of-00001.safetensors"
            tensors = {
                "huge.weight": {
                    "dtype": "U8",
                    "shape": [5_000_000_000],
                    "nbytes": 5_000_000_000,
                }
            }
            self._write_safetensors(shard, tensors, sparse=True)
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "metadata": {"total_size": 5_000_000_000},
                        "weight_map": {"huge.weight": shard.name},
                    }
                )
            )
            with self.assertRaisesRegex(ValueError, "sparse or physically incomplete"):
                checkpoint_lineage.merged_checkpoint_inventory(
                    root, expected=self._synthetic_expected(tensors)
                )
            with self.assertRaisesRegex(ValueError, "exact pinned Qwen3.5-4B config"):
                checkpoint_lineage.merged_checkpoint_inventory(root)

    def test_exact_merge_replay_checks_adapted_and_unchanged_tensors(self) -> None:
        import torch

        adapted = "model.language_model.layers.0.mlp.up_proj.weight"
        unchanged = "model.language_model.embed_tokens.weight"
        key_a = "base_model.model.model.layers.0.mlp.up_proj.lora_A.weight"
        key_b = "base_model.model.model.layers.0.mlp.up_proj.lora_B.weight"
        base = {
            adapted: torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.bfloat16),
            unchanged: torch.tensor([[5.0, 6.0]], dtype=torch.bfloat16),
        }
        adapter = {
            key_a: torch.tensor([[1.0, -1.0]], dtype=torch.float32),
            key_b: torch.tensor([[2.0], [3.0]], dtype=torch.float32),
        }
        merged = {
            adapted: (
                base[adapted].float() + adapter[key_b] @ adapter[key_a]
            ).to(torch.bfloat16),
            unchanged: base[unchanged].clone(),
        }
        result = merge_replay.verify_tensor_equations(
            base=base,
            merged=merged,
            adapter=adapter,
            target_modules={"up_proj"},
            scale=1.0,
        )
        self.assertEqual(result["adapted_module_count"], 1)
        self.assertEqual(result["unchanged_tensor_count"], 1)
        wrong = {**merged, adapted: merged[adapted].clone()}
        wrong[adapted][0, 0] += 1
        with self.assertRaisesRegex(ValueError, "base plus LoRA"):
            merge_replay.verify_tensor_equations(
                base=base,
                merged=wrong,
                adapter=adapter,
                target_modules={"up_proj"},
                scale=1.0,
            )
        wrong = {**merged, unchanged: merged[unchanged] + 1}
        with self.assertRaisesRegex(ValueError, "unmodified merged tensor"):
            merge_replay.verify_tensor_equations(
                base=base,
                merged=wrong,
                adapter=adapter,
                target_modules={"up_proj"},
                scale=1.0,
            )
        signed_zero = {**merged, unchanged: merged[unchanged].clone()}
        signed_zero[unchanged][0, 0] = -0.0
        base_zero = {**base, unchanged: base[unchanged].clone()}
        base_zero[unchanged][0, 0] = 0.0
        with self.assertRaisesRegex(ValueError, "unmodified merged tensor"):
            merge_replay.verify_tensor_equations(
                base=base_zero,
                merged=signed_zero,
                adapter=adapter,
                target_modules={"up_proj"},
                scale=1.0,
            )

    def test_explicit_capture_list_requires_strict_positive_tied_geometry(self) -> None:
        runner.EngineConfig(
            max_num_seqs=19,
            cudagraph_capture_sizes=(1, 2, 4, 8, 16, 19),
        ).validate()

        invalid = (
            ((), "positive integers"),
            ((0, 19), "positive integers"),
            ((True, 19), "positive integers"),
            ((1, 4, 2, 19), "strictly increasing"),
            ((1, 2, 2, 19), "strictly increasing"),
            ((1, 2, 4, 16), "largest cudagraph capture size"),
        )
        for sizes, message in invalid:
            with self.subTest(sizes=sizes), self.assertRaisesRegex(
                ValueError, message
            ):
                runner.EngineConfig(
                    max_num_seqs=19,
                    cudagraph_capture_sizes=sizes,
                ).validate()

        with self.assertRaisesRegex(ValueError, "incompatible with enforce_eager"):
            runner.EngineConfig(
                max_num_seqs=19,
                cudagraph_capture_sizes=(1, 2, 4, 8, 16, 19),
                enforce_eager=True,
            ).validate()

    def test_frozen_default_geometry_is_capacity_fitted_and_tied(self) -> None:
        config = yaml.safe_load(
            (RUNNER_PATH.parents[1] / "configs" / "default.yaml").read_text()
        )
        engine = config["evaluation"]["engine"]
        self.assertEqual(engine["max_num_seqs"], 15)
        self.assertEqual(engine["cudagraph_capture_sizes"], [1, 2, 4, 8, 15])
        self.assertEqual(engine["cudagraph_capture_sizes"][-1], engine["max_num_seqs"])


class PackageInventoryTests(unittest.TestCase):
    def test_atomic_json_writer_returns_exact_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "rows.jsonl"
            digest = runner._write_json_atomic(
                output, [{"id": "one"}, {"id": "two"}], jsonl=True
            )

            self.assertEqual(digest, runner._sha256_file(output))
            self.assertEqual(len(output.read_text().splitlines()), 2)

    def test_vendored_duplicate_cannot_override_real_distribution(self) -> None:
        real = SimpleNamespace(
            metadata={"Name": "packaging"},
            version="26.2",
        )
        vendored = SimpleNamespace(
            metadata={"Name": "packaging"},
            version="26.0",
        )
        with mock.patch.object(
            runner.importlib.metadata,
            "distributions",
            return_value=[real, vendored],
        ), mock.patch.object(
            runner.importlib.metadata,
            "version",
            return_value="26.2",
        ) as resolve:
            inventory = runner._installed_packages()

        self.assertEqual(inventory, {"packaging": "26.2"})
        resolve.assert_called_once_with("packaging")


class TerminationSemanticsTests(unittest.TestCase):
    def test_model_and_tokenizer_eos_ids_remain_distinct(self) -> None:
        runner._validate_termination_ids(248044, 248046)
        for model_eos, tokenizer_eos in ((248046, 248046), (248044, 248044)):
            with self.subTest(
                model_eos=model_eos, tokenizer_eos=tokenizer_eos
            ), self.assertRaisesRegex(RuntimeError, "termination IDs changed"):
                runner._validate_termination_ids(model_eos, tokenizer_eos)

    def test_sampling_ignores_tokenizer_eos_and_stops_on_model_eos(self) -> None:
        captured: dict[str, object] = {}

        class FakeSamplingParams:
            def __init__(self, **kwargs: object):
                captured.update(kwargs)

        fake_vllm = types.ModuleType("vllm")
        fake_vllm.SamplingParams = FakeSamplingParams
        instance = object.__new__(runner.VLLMRunner)
        instance.hf_eos_id = 248044
        with mock.patch.dict(sys.modules, {"vllm": fake_vllm}):
            instance._params(
                runner.SamplingConfig(thinking="off", greedy=True),
                max_tokens=8,
                seed=17,
                n=1,
            )

        self.assertIs(captured["ignore_eos"], True)
        self.assertEqual(captured["stop_token_ids"], [248044])

    def test_trimming_preserves_tokenizer_eos_and_removes_model_eos(self) -> None:
        instance = object.__new__(runner.VLLMRunner)
        instance.hf_eos_id = 248044
        self.assertEqual(
            instance._trim_hf_eos([1, 248046, 2, 248044, 3]),
            [1, 248046, 2],
        )


class CliRewriteTests(unittest.TestCase):
    def test_no_site_runner_preserves_the_invoked_venv_bin_directory(self) -> None:
        self.assertEqual(
            runner._PYTHON_BIN,
            str(Path(sys.executable).parent),
        )
        self.assertNotEqual(runner._PYTHON_BIN, str(Path(sys.executable).resolve().parent))

    def test_runner_contains_no_adaptive_exec_path(self) -> None:
        source = RUNNER_PATH.read_text()
        self.assertNotIn("os.exec", source)
        self.assertNotIn("_rewrite_max_num_seqs_argv", source)

    def test_cli_disables_long_option_abbreviation(self) -> None:
        for abbreviated in ("--max-num-seq", "--cudagraph-capture-siz"):
            with self.subTest(argument=abbreviated), contextlib.redirect_stderr(
                io.StringIO()
            ), self.assertRaises(SystemExit) as raised:
                runner._parse_args(
                    ["--smoke", "1", "--output", "out.jsonl", abbreviated, "15"]
                )
            self.assertEqual(raised.exception.code, 2)


class MambaReexecGuardTests(unittest.TestCase):
    def test_obsolete_adaptive_geometry_state_is_rejected_before_import(self) -> None:
        for name in (
            "QWEN_RUNNER_MAMBA_REEXEC",
            "QWEN_RUNNER_MAMBA_REEXEC_CUDAGRAPH",
        ):
            with self.subTest(name=name), mock.patch.dict(
                os.environ, {name: "synthetic"}, clear=False
            ), self.assertRaisesRegex(RuntimeError, "adaptive Mamba geometry is forbidden"):
                runner.VLLMRunner(
                    runner.EngineConfig(
                        max_num_seqs=15,
                        cudagraph_capture_sizes=(1, 2, 4, 8, 15),
                    )
                )


class ResolvedCudagraphTests(unittest.TestCase):
    def test_supported_full_decode_modes_are_accepted(self) -> None:
        requested = (1, 2, 4, 8, 15)
        modes = (
            ("FULL", "FULL", "FULL"),
            ("FULL_DECODE_ONLY", "FULL", "NONE"),
            ("FULL_AND_PIECEWISE", "FULL", "PIECEWISE"),
        )
        for mode, decode, mixed in modes:
            with self.subTest(mode=mode):
                resolved = runner._resolved_cudagraph_metadata(
                    _compilation_config(mode=mode, decode=decode, mixed=mixed)
                )
                runner._validate_explicit_cudagraph_resolution(requested, resolved)

    def test_none_piecewise_only_and_truncated_resolutions_are_rejected(self) -> None:
        requested = (1, 2, 4, 8, 15)
        rejected = (
            _compilation_config(
                mode="NONE", decode="NONE", mixed="NONE", has_full=False
            ),
            _compilation_config(
                mode="PIECEWISE",
                decode="PIECEWISE",
                mixed="PIECEWISE",
                has_full=False,
            ),
            _compilation_config((1, 2, 4, 8), maximum=8),
            _compilation_config(maximum=8),
        )
        for compilation_config in rejected:
            resolved = runner._resolved_cudagraph_metadata(compilation_config)
            with self.subTest(resolved=resolved), self.assertRaisesRegex(
                RuntimeError, "did not honor"
            ):
                runner._validate_explicit_cudagraph_resolution(requested, resolved)

    def test_frozen_geometry_is_forwarded_without_adaptation(self) -> None:
        effective = (1, 2, 4, 8, 15)
        captured_engine_args: dict[str, object] = {}

        class FakeTokenizer:
            eos_token_id = 248046
            eos_token = "<|im_end|>"

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                values = {
                    "<|endoftext|>": [248044],
                    "<|im_end|>": [248046],
                    "<think>": [248068],
                    "</think>": [248069],
                    "</think>\n\n": [248069, 198],
                    "<|im_start|>assistant\n<think>\n": [1, 2, 3],
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n": [1, 2, 4],
                }
                return values[text]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
                return FakeTokenizer()

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044))

        class FakeLLM:
            def __init__(self, **kwargs: object):
                captured_engine_args.update(kwargs)
                cache_blocks = 1100
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        compilation_config=_compilation_config(),
                        cache_config=SimpleNamespace(
                            num_gpu_blocks=cache_blocks,
                            block_size=528,
                            kv_cache_size_tokens=int((cache_blocks / 11) * 4096),
                            kv_cache_max_concurrency=cache_blocks / 11,
                            enable_prefix_caching=False,
                            mamba_cache_mode="none",
                            mamba_block_size=4096,
                        ),
                        scheduler_config=SimpleNamespace(
                            max_num_seqs=15,
                            max_num_batched_tokens=32768,
                            async_scheduling=False,
                        ),
                        model_config=SimpleNamespace(
                            max_model_len=4096,
                            dtype="torch.bfloat16",
                        ),
                        parallel_config=SimpleNamespace(
                            world_size=1,
                            tensor_parallel_size=1,
                            data_parallel_size=1,
                        ),
                    ),
                    engine_core=SimpleNamespace(shutdown=lambda: None),
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.Qwen2Tokenizer = FakeAutoTokenizer
        fake_transformers.AutoConfig = FakeAutoConfig
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        base_path = Path("/synthetic/base")
        tokenizer_path = Path("/synthetic/tokenizer")
        tokenizer_commitment = {"synthetic": "tokenizer"}

        with mock.patch.dict(
            sys.modules,
            {"transformers": fake_transformers, "vllm": fake_vllm},
        ), mock.patch(
            "merge_replay.authenticate_base_snapshot",
            return_value=(base_path, {}, {}),
        ), mock.patch(
            "merge_replay.base_snapshot_commitment",
            return_value={"synthetic": "base"},
        ), mock.patch(
            "tokenizer_lineage.authenticate_tokenizer_snapshot",
            return_value=tokenizer_commitment,
        ), mock.patch(
            "tokenizer_lineage.ensure_closed_tokenizer_view",
            return_value=(tokenizer_path, tokenizer_commitment),
        ), mock.patch(
            "tokenizer_lineage.authenticate_closed_tokenizer_view",
            return_value=tokenizer_commitment,
        ), mock.patch(
            "load_window_guard.LoadWindowGuard", _FakeLoadWindowGuard
        ), mock.patch(
            "runtime_contract.bind_active_cuda_identity",
            return_value={"synthetic": "gpu"},
        ):
            instance = runner.VLLMRunner(
                runner.EngineConfig(
                    max_model_len=4096,
                    max_num_seqs=15,
                    cudagraph_capture_sizes=effective,
                )
            )

        self.assertEqual(captured_engine_args["max_num_seqs"], 15)
        self.assertIs(captured_engine_args["trust_remote_code"], False)
        self.assertEqual(captured_engine_args["cudagraph_capture_sizes"], list(effective))
        self.assertEqual(captured_engine_args["max_cudagraph_capture_size"], 15)
        self.assertNotIn("requested_max_num_seqs", instance.engine_args)
        self.assertNotIn("effective_max_num_seqs", instance.engine_args)
        self.assertEqual(
            instance.resolved_cudagraph["cudagraph_capture_sizes"], list(effective)
        )
        instance.close()

    def test_engine_load_byte_change_shuts_down_before_generation(self) -> None:
        shutdown = {"called": False}

        class FakeTokenizer:
            eos_token_id = 248046
            eos_token = "<|im_end|>"

            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                return {
                    "<|endoftext|>": [248044],
                    "<|im_end|>": [248046],
                    "<think>": [248068],
                    "</think>": [248069],
                    "</think>\n\n": [248069, 198],
                    "<|im_start|>assistant\n<think>\n": [1, 2, 3],
                    "<|im_start|>assistant\n<think>\n\n</think>\n\n": [1, 2, 4],
                }[text]

        class FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> FakeTokenizer:
                return FakeTokenizer()

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> SimpleNamespace:
                return SimpleNamespace(text_config=SimpleNamespace(eos_token_id=248044))

        class FakeLLM:
            def __init__(self, **_kwargs: object):
                cache_blocks = 1100
                self.llm_engine = SimpleNamespace(
                    vllm_config=SimpleNamespace(
                        compilation_config=_compilation_config(),
                        cache_config=SimpleNamespace(
                            num_gpu_blocks=cache_blocks,
                            block_size=528,
                            kv_cache_size_tokens=int((cache_blocks / 11) * 4096),
                            kv_cache_max_concurrency=cache_blocks / 11,
                            enable_prefix_caching=False,
                            mamba_cache_mode="none",
                            mamba_block_size=4096,
                        ),
                        scheduler_config=SimpleNamespace(
                            max_num_seqs=15,
                            max_num_batched_tokens=32768,
                            async_scheduling=False,
                        ),
                        model_config=SimpleNamespace(
                            max_model_len=4096,
                            dtype="torch.bfloat16",
                        ),
                        parallel_config=SimpleNamespace(
                            world_size=1,
                            tensor_parallel_size=1,
                            data_parallel_size=1,
                        ),
                    ),
                    engine_core=SimpleNamespace(
                        shutdown=lambda: shutdown.__setitem__("called", True)
                    ),
                )

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.Qwen2Tokenizer = FakeAutoTokenizer
        fake_transformers.AutoConfig = FakeAutoConfig
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        base_path = Path("/synthetic/base")
        tokenizer_path = Path("/synthetic/tokenizer")
        tokenizer_commitment = {"synthetic": "tokenizer"}
        with mock.patch.dict(
            sys.modules,
            {"transformers": fake_transformers, "vllm": fake_vllm},
        ), mock.patch(
            "merge_replay.authenticate_base_snapshot",
            return_value=(base_path, {}, {}),
        ), mock.patch(
            "merge_replay.base_snapshot_commitment",
            side_effect=[
                {"version": 1},
                {"version": 1},
                {"version": 1},
                {"version": 1},
                {"version": 2},
            ],
        ), mock.patch(
            "tokenizer_lineage.authenticate_tokenizer_snapshot",
            return_value=tokenizer_commitment,
        ), mock.patch(
            "tokenizer_lineage.ensure_closed_tokenizer_view",
            return_value=(tokenizer_path, tokenizer_commitment),
        ), mock.patch(
            "tokenizer_lineage.authenticate_closed_tokenizer_view",
            return_value=tokenizer_commitment,
        ), mock.patch(
            "load_window_guard.LoadWindowGuard", _FakeLoadWindowGuard
        ), self.assertRaisesRegex(RuntimeError, "synthetic content commitment changed"):
            runner.VLLMRunner(
                runner.EngineConfig(
                    max_model_len=4096,
                    max_num_seqs=15,
                    cudagraph_capture_sizes=(1, 2, 4, 8, 15),
                )
            )
        self.assertTrue(shutdown["called"])


if __name__ == "__main__":
    unittest.main()
