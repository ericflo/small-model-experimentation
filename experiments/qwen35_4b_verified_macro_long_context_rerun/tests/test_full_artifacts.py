"""Model-free tests for Amendment 8 full-run durability."""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
SRC = EXP / "src"
sys.path.insert(0, str(SRC))

import full_artifacts as full_store  # noqa: E402
import model_harness as harness  # noqa: E402


RUN_SPEC = importlib.util.spec_from_file_location(
    "verified_macro_full_run_test", EXP / "scripts" / "run.py"
)
assert RUN_SPEC is not None and RUN_SPEC.loader is not None
run = importlib.util.module_from_spec(RUN_SPEC)
sys.modules[RUN_SPEC.name] = run
RUN_SPEC.loader.exec_module(run)


ANALYZE_SPEC = importlib.util.spec_from_file_location(
    "verified_macro_full_analyze_test", EXP / "scripts" / "analyze.py"
)
assert ANALYZE_SPEC is not None and ANALYZE_SPEC.loader is not None
analyze = importlib.util.module_from_spec(ANALYZE_SPEC)
sys.modules[ANALYZE_SPEC.name] = analyze
ANALYZE_SPEC.loader.exec_module(analyze)


def full_tasks() -> list[dict[str, str]]:
    return [
        *({"id": f"no_{index:03d}", "split": "no_reuse"} for index in range(40)),
        *({"id": f"reuse_{index:03d}", "split": "reuse"} for index in range(80)),
    ]


def runtime_metadata(
    *, git_commit: str = "commit-a", git_dirty: bool = False
) -> dict[str, object]:
    return {
        "python": "3.12.0",
        "python_executable": "/venv/bin/python",
        "platform": "Linux-test",
        "packages": {"torch": "2.11.0", "vllm": "0.24.0"},
        "environment_lock": {"sha256": "f" * 64},
        "uv": "uv 0.8.0",
        "cuda_toolkit": "CUDA 12.9",
        "gpu": "RTX 6000 Ada, 550.127, 49140",
        "vllm_enable_v1_multiprocessing": "0",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def summary(
    *,
    requests: int,
    completions: int,
    sampled_tokens: int = 0,
    budget: int = 32768,
    answer_max_tokens: int = 512,
    git_commit: str = "commit-a",
    git_dirty: bool = False,
) -> dict[str, object]:
    return {
        "schema_version": 3,
        "model": "Qwen/Qwen3.5-4B",
        "model_revision": "revision",
        "runner_sha256": "a" * 64,
        "adapter": None,
        "sampling": {
            "thinking": "budget",
            "thinking_budget": budget,
            "n": completions // requests,
            "max_tokens": answer_max_tokens,
            "answer_max_tokens": answer_max_tokens,
        },
        "resolved_sampling": {"temperature": 0.6},
        "engine": {"max_model_len": 65536},
        "engine_args": {"async_scheduling": False},
        "runtime": runtime_metadata(git_commit=git_commit, git_dirty=git_dirty),
        "rng_isolation": {"engine_seed": 0},
        "termination": {
            "hf_model_eos_token_id": 9999,
            "vllm_tokenizer_eos_ignored": 9998,
        },
        "think_token_ids": {
            "close": 2,
            "forced_close_sequence": [2, 3],
        },
        "counts": {
            "requests": requests,
            "completions": completions,
            "unique_input_prompt_tokens": requests * 5,
            "stage1_logical_prompt_tokens": completions * 5,
            "stage2_logical_prompt_tokens": 0,
            "logical_model_input_tokens": completions * 5,
            "sampled_tokens": sampled_tokens,
            "injected_tokens": 0,
        },
        "timing": {
            "model_load_seconds": 1.0,
            "generation_seconds": 2.0,
            "sampled_tokens_per_second": sampled_tokens / 2.0,
        },
    }


def preflight_fixture(
    task_ids: list[str],
    *,
    arm: str,
    budget: int,
    answer_max_tokens: int = 512,
    max_model_len: int = 65536,
) -> dict[str, object]:
    reserve = budget + 2 + answer_max_tokens
    records = [
        {
            "id": f"{task_id}::{arm}",
            "input_record_sha256": hashlib.sha256(
                f"input:{task_id}:{arm}".encode()
            ).hexdigest(),
            "rendered_prompt_sha256": hashlib.sha256(
                f"prompt:{task_id}:{arm}".encode()
            ).hexdigest(),
            "prompt_tokens": 5,
            "prompt_plus_reserve_tokens": 5 + reserve,
        }
        for task_id in task_ids
    ]
    return {
        "schema_version": 1,
        "pass": True,
        "max_model_len": max_model_len,
        "generation_reserve_tokens": reserve,
        "n_records": len(task_ids),
        "min_prompt_tokens": 5,
        "max_prompt_tokens": 5,
        "max_prompt_plus_reserve_tokens": 5 + reserve,
        "records": records,
    }


def rows_fixture(
    task_ids: list[str],
    *,
    arm: str,
    k: int,
    preflight: dict[str, object],
) -> list[dict[str, object]]:
    prompts = {str(row["id"]): row for row in preflight["records"]}
    rows: list[dict[str, object]] = []
    for task_id in task_ids:
        record_id = f"{task_id}::{arm}"
        prompt = prompts[record_id]
        outputs = []
        for sample_index in range(k):
            outputs.append(
                {
                    "sample_index": sample_index,
                    "token_ids": [2],
                    "stage1_token_ids": [2],
                    "stage2_token_ids": [],
                    "injected_token_ids": [],
                    "n_stage1_prompt_tokens": 5,
                    "n_stage2_prompt_tokens": 0,
                    "n_sampled_tokens": 1,
                    "n_injected_tokens": 0,
                    "n_completion_tokens": 1,
                    "n_thinking_tokens": 0,
                    "n_answer_tokens": 0,
                    "n_terminal_tokens_trimmed": 0,
                    "forced_close": False,
                    "seed_stage2": None,
                    "text": "PROGRAM: ADD1",
                    "finish_reason": "stop",
                    "stage1_finish_reason": "stop",
                    "truncated": False,
                    "thinking_closed": True,
                }
            )
        rows.append(
            {
                "id": record_id,
                "meta": {"task_id": task_id, "arm": arm},
                "prompt_sha256": prompt["rendered_prompt_sha256"],
                "n_prompt_tokens": 5,
                "outputs": outputs,
            }
        )
    return rows


def write_complete_shard(
    root: Path,
    *,
    plan_hash: str,
    budget: int = 32768,
    arm: str = "base",
    task_ids: tuple[str, ...] = ("t0", "t1"),
    k: int = 2,
) -> Path:
    shard_dir = full_store.shard_directory(root, budget=budget, arm=arm, shard_index=0)
    shard_dir.mkdir(parents=True)
    task_id_list = list(task_ids)
    preflight = preflight_fixture(task_id_list, arm=arm, budget=budget)
    rows = rows_fixture(task_id_list, arm=arm, k=k, preflight=preflight)
    (shard_dir / "preflight.json").write_text(
        json.dumps(preflight, sort_keys=True) + "\n", encoding="utf-8"
    )
    (shard_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (shard_dir / "runner.meta.json").write_text(
        json.dumps(
            summary(
                requests=len(task_ids),
                completions=len(task_ids) * k,
                sampled_tokens=len(task_ids) * k,
                budget=budget,
            ),
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt = full_store.make_receipt(
        shard_dir,
        shard_plan_sha256=plan_hash,
        budget=budget,
        arm=arm,
        shard_index=0,
        task_ids=task_ids,
        k=k,
    )
    (shard_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return shard_dir


def write_protocol_shard(
    root: Path,
    *,
    runner_sha256: str,
    plan_hash: str,
    config: dict[str, object],
    budget: int,
    arm: str,
    shard_index: int,
    task_ids: list[str],
    k: int,
) -> tuple[Path, dict[str, object]]:
    shard_dir = full_store.shard_directory(
        root, budget=budget, arm=arm, shard_index=shard_index
    )
    shard_dir.mkdir(parents=True)
    inference = config["inference"]
    seeds = config["seeds"]
    preflight = preflight_fixture(
        task_ids,
        arm=arm,
        budget=budget,
        answer_max_tokens=int(inference["answer_max_tokens"]),
        max_model_len=int(inference["max_model_len"]),
    )
    rows = rows_fixture(task_ids, arm=arm, k=k, preflight=preflight)
    sampling = harness.SamplingConfig(
        thinking="budget",
        thinking_budget=budget,
        n=k,
        max_tokens=int(inference["answer_max_tokens"]),
        answer_max_tokens=int(inference["answer_max_tokens"]),
        temperature=float(inference["temperature"]),
        top_p=float(inference["top_p"]),
        top_k=int(inference["top_k"]),
        run_seed=int(seeds["vllm_solver"]),
    )
    engine = harness.EngineConfig(
        max_model_len=int(inference["max_model_len"]),
        max_num_seqs=int(inference["max_num_seqs"]),
        max_num_batched_tokens=int(inference["max_num_batched_tokens"]),
    )
    completions = len(task_ids) * k
    metadata = {
        "schema_version": 3,
        "model": harness.REQUIRED_MODEL_ID,
        "model_revision": harness.MODEL_REVISION,
        "runner_sha256": runner_sha256,
        "adapter": None,
        "sampling": dataclasses.asdict(sampling),
        "resolved_sampling": sampling.resolved_sampling(),
        "engine": dataclasses.asdict(engine),
        "engine_args": {"async_scheduling": False},
        "runtime": runtime_metadata(),
        "rng_isolation": {"engine_seed": 0},
        "termination": {
            "hf_model_eos_token_id": 9999,
            "vllm_tokenizer_eos_ignored": 9998,
        },
        "think_token_ids": {
            "close": 2,
            "forced_close_sequence": [2, 3],
        },
        "counts": {
            "requests": len(task_ids),
            "completions": completions,
            "unique_input_prompt_tokens": len(task_ids) * 5,
            "stage1_logical_prompt_tokens": completions * 5,
            "stage2_logical_prompt_tokens": 0,
            "logical_model_input_tokens": completions * 5,
            "sampled_tokens": completions,
            "injected_tokens": 0,
        },
        "timing": {
            "model_load_seconds": 1.0,
            "generation_seconds": 1.0,
            "sampled_tokens_per_second": 0.0,
        },
    }
    (shard_dir / "preflight.json").write_text(
        json.dumps(preflight, sort_keys=True) + "\n", encoding="utf-8"
    )
    (shard_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    (shard_dir / "runner.meta.json").write_text(
        json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = full_store.make_receipt(
        shard_dir,
        shard_plan_sha256=plan_hash,
        budget=budget,
        arm=arm,
        shard_index=shard_index,
        task_ids=task_ids,
        k=k,
    )
    (shard_dir / "receipt.json").write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )
    return shard_dir, receipt


def mutate_payload_and_rehash(
    shard_dir: Path,
    filename: str,
    mutate: object,
) -> None:
    path = shard_dir / filename
    if filename == "rows.jsonl":
        payload = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutate(payload)
        path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in payload),
            encoding="utf-8",
        )
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        mutate(payload)
        path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    receipt_path = shard_dir / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["files"][filename] = full_store.file_integrity(path)
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
    )


class ShardPlanTests(unittest.TestCase):
    def test_plan_is_balanced_nested_and_exactly_144_completions(self) -> None:
        plan = full_store.build_shard_plan(full_tasks(), ["base", "mined"])
        self.assertEqual(
            plan["triplets"][0], ["no_000", "reuse_000", "reuse_001"]
        )
        self.assertEqual(plan["arms"]["base"]["shard_count"], 20)
        self.assertEqual(plan["arms"]["mined"]["shard_count"], 10)
        self.assertTrue(
            all(row["completions"] == 144 for row in plan["arms"]["base"]["shards"])
        )
        self.assertTrue(
            all(row["completions"] == 144 for row in plan["arms"]["mined"]["shards"])
        )
        self.assertEqual(
            plan["arms"]["mined"]["shards"][0]["task_ids"],
            plan["arms"]["base"]["shards"][0]["task_ids"]
            + plan["arms"]["base"]["shards"][1]["task_ids"],
        )
        for arm in ("base", "mined"):
            for shard in plan["arms"][arm]["shards"]:
                splits = [
                    "no_reuse" if task_id.startswith("no_") else "reuse"
                    for task_id in shard["task_ids"]
                ]
                self.assertEqual(splits.count("reuse"), 2 * splits.count("no_reuse"))

    def test_plan_rejects_count_and_arm_drift(self) -> None:
        with self.assertRaisesRegex(full_store.FullArtifactError, "40 no_reuse"):
            full_store.build_shard_plan(full_tasks()[1:], ["base", "mined"])
        with self.assertRaisesRegex(full_store.FullArtifactError, "base must be the first"):
            full_store.build_shard_plan(full_tasks(), ["mined", "base"])


class ReceiptTests(unittest.TestCase):
    def test_receipt_binds_files_order_prompts_and_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_complete_shard(root, plan_hash="b" * 64)
            receipt = full_store.validate_shard_directory(
                shard,
                root=root,
                shard_plan_sha256="b" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0", "t1"),
                k=2,
            )
            self.assertEqual(receipt["ordered_record_ids"], ["t0::base", "t1::base"])
            self.assertEqual(
                receipt["protocol_identity"]["runtime"]["environment_lock"],
                {"sha256": "f" * 64},
            )
            self.assertEqual(receipt["provenance"]["runtime"]["git_commit"], "commit-a")
            self.assertEqual(
                full_store.receipt_protocol_identity(receipt),
                receipt["protocol_identity"],
            )
            self.assertEqual(set(receipt["files"]), set(full_store.PAYLOAD_FILES))
            catalog_entry = full_store.catalog_shard_entry(root, shard, receipt)
            self.assertEqual(
                full_store.require_catalog_shard_entry(
                    root, shard, receipt, catalog_entry
                ),
                catalog_entry,
            )
            drifted_entry = copy.deepcopy(catalog_entry)
            drifted_entry["receipt"]["bytes"] += 1
            with self.assertRaisesRegex(full_store.FullArtifactError, "catalog shard entry drift"):
                full_store.require_catalog_shard_entry(
                    root, shard, receipt, drifted_entry
                )

            with (shard / "rows.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(" \n")
            with self.assertRaisesRegex(full_store.FullArtifactError, "hash/size mismatch"):
                full_store.validate_shard_directory(
                    shard,
                    root=root,
                    shard_plan_sha256="b" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0", "t1"),
                    k=2,
                )

    def test_validator_accepts_answer_restart_without_forced_reasoning_close(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = full_store.shard_directory(
                root, budget=32768, arm="base", shard_index=0
            )
            shard.mkdir(parents=True)
            preflight = preflight_fixture(["t0"], arm="base", budget=32768)
            rows = rows_fixture(["t0"], arm="base", k=1, preflight=preflight)
            rows[0]["outputs"][0] = {
                "sample_index": 0,
                "token_ids": [10, 11, 2, 3, 20],
                "stage1_token_ids": [10, 11, 2],
                "retained_thinking_token_ids": [10, 11],
                "stage2_token_ids": [20],
                "injected_token_ids": [2, 3],
                "n_stage1_prompt_tokens": 5,
                "n_stage2_prompt_tokens": 9,
                "n_sampled_tokens": 4,
                "n_injected_tokens": 2,
                "n_completion_tokens": 5,
                "n_thinking_tokens": 2,
                "n_answer_tokens": 1,
                "n_terminal_tokens_trimmed": 0,
                "forced_close": False,
                "seed_stage2": 123,
                "text": "PROGRAM: ADD1",
                "finish_reason": "stop",
                "stage1_finish_reason": "length",
                "truncated": False,
                "thinking_closed": True,
            }
            metadata = summary(
                requests=1,
                completions=1,
                sampled_tokens=4,
            )
            metadata["counts"].update(
                {
                    "stage2_logical_prompt_tokens": 9,
                    "logical_model_input_tokens": 14,
                    "injected_tokens": 2,
                }
            )
            (shard / "preflight.json").write_text(
                json.dumps(preflight, sort_keys=True) + "\n", encoding="utf-8"
            )
            (shard / "rows.jsonl").write_text(
                json.dumps(rows[0], sort_keys=True) + "\n", encoding="utf-8"
            )
            (shard / "runner.meta.json").write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
            receipt = full_store.make_receipt(
                shard,
                shard_plan_sha256="8" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0",),
                k=1,
            )
            (shard / "receipt.json").write_text(
                json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8"
            )
            validated = full_store.validate_shard_directory(
                shard,
                root=root,
                shard_plan_sha256="8" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0",),
                k=1,
            )
            self.assertEqual(validated["status"], "complete")

    def test_cache_protocol_allows_git_audit_drift_but_rejects_package_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shard = write_complete_shard(root, plan_hash="7" * 64)
            original_receipt = json.loads(
                (shard / "receipt.json").read_text(encoding="utf-8")
            )
            expected = full_store.receipt_protocol_identity(original_receipt)

            metadata_path = shard / "runner.meta.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metadata["runtime"]["git_commit"] = "commit-after-checkpoint"
            metadata["runtime"]["git_dirty"] = True
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
            git_drift_receipt = full_store.make_receipt(
                shard,
                shard_plan_sha256="7" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0", "t1"),
                k=2,
            )
            (shard / "receipt.json").write_text(
                json.dumps(git_drift_receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            validated = full_store.validate_shard_directory(
                shard,
                root=root,
                shard_plan_sha256="7" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0", "t1"),
                k=2,
                expected_protocol_identity=expected,
            )
            self.assertNotEqual(
                validated["provenance_sha256"],
                original_receipt["provenance_sha256"],
            )
            self.assertEqual(
                validated["protocol_identity_sha256"],
                original_receipt["protocol_identity_sha256"],
            )

            metadata["runtime"]["packages"]["vllm"] = "0.24.1-drift"
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
            package_drift_receipt = full_store.make_receipt(
                shard,
                shard_plan_sha256="7" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0", "t1"),
                k=2,
            )
            (shard / "receipt.json").write_text(
                json.dumps(package_drift_receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                full_store.FullArtifactError, "protocol identity drift"
            ):
                full_store.validate_shard_directory(
                    shard,
                    root=root,
                    shard_plan_sha256="7" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0", "t1"),
                    k=2,
                    expected_protocol_identity=expected,
                )
            metadata["runtime"]["packages"]["vllm"] = "0.24.0"
            metadata["runtime"]["gpu"] = "different GPU/driver/memory"
            metadata_path.write_text(
                json.dumps(metadata, sort_keys=True) + "\n", encoding="utf-8"
            )
            gpu_drift_receipt = full_store.make_receipt(
                shard,
                shard_plan_sha256="7" * 64,
                budget=32768,
                arm="base",
                shard_index=0,
                task_ids=("t0", "t1"),
                k=2,
            )
            (shard / "receipt.json").write_text(
                json.dumps(gpu_drift_receipt, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                full_store.FullArtifactError, "protocol identity drift"
            ):
                full_store.validate_shard_directory(
                    shard,
                    root=root,
                    shard_plan_sha256="7" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0", "t1"),
                    k=2,
                    expected_protocol_identity=expected,
                )

    def test_receipt_rejects_escape_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as other:
            root = Path(directory)
            shard = write_complete_shard(root, plan_hash="c" * 64)
            with self.assertRaisesRegex(full_store.FullArtifactError, "escapes canonical root"):
                full_store.validate_shard_directory(
                    shard,
                    root=Path(other),
                    shard_plan_sha256="c" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0", "t1"),
                    k=2,
                )
            (shard / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(full_store.FullArtifactError, "unexpected files"):
                full_store.validate_shard_directory(
                    shard,
                    root=root,
                    shard_plan_sha256="c" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0", "t1"),
                    k=2,
                )

    def test_validator_recomputes_preflight_rows_and_summary_after_rehash(self) -> None:
        cases = (
            (
                "preflight.json",
                lambda payload: payload.__setitem__(
                    "max_prompt_plus_reserve_tokens",
                    payload["max_prompt_plus_reserve_tokens"] + 1,
                ),
                "max prompt-plus-reserve mismatch",
            ),
            (
                "preflight.json",
                lambda payload: payload["records"][0].__setitem__(
                    "prompt_plus_reserve_tokens",
                    payload["records"][0]["prompt_plus_reserve_tokens"] + 1,
                ),
                "per-record reserve mismatch",
            ),
            (
                "rows.jsonl",
                lambda payload: payload[0].__setitem__("prompt_sha256", "0" * 64),
                "rendered prompt hash mismatch",
            ),
            (
                "rows.jsonl",
                lambda payload: (
                    payload[0]["outputs"][0].__setitem__("sample_index", 1),
                    payload[0]["outputs"][1].__setitem__("sample_index", 0),
                ),
                "sample indices are not exact and ordered",
            ),
            (
                "runner.meta.json",
                lambda payload: payload["counts"].__setitem__(
                    "sampled_tokens", payload["counts"]["sampled_tokens"] + 1
                ),
                "summary count sampled_tokens mismatch",
            ),
        )
        for filename, mutation, error in cases:
            with self.subTest(filename=filename, error=error):
                with tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    shard = write_complete_shard(root, plan_hash="9" * 64)
                    mutate_payload_and_rehash(shard, filename, mutation)
                    with self.assertRaisesRegex(full_store.FullArtifactError, error):
                        full_store.validate_shard_directory(
                            shard,
                            root=root,
                            shard_plan_sha256="9" * 64,
                            budget=32768,
                            arm="base",
                            shard_index=0,
                            task_ids=("t0", "t1"),
                            k=2,
                        )
    def test_summary_aggregation_requires_one_identity_and_sums_counts(self) -> None:
        first = summary(requests=6, completions=144, sampled_tokens=1000)
        second = summary(
            requests=6,
            completions=144,
            sampled_tokens=2000,
            git_commit="commit-b",
            git_dirty=True,
        )
        self.assertNotEqual(
            full_store.runner_provenance(first),
            full_store.runner_provenance(second),
        )
        self.assertEqual(
            full_store.protocol_identity(first),
            full_store.protocol_identity(second),
        )
        self.assertEqual(
            set(full_store.protocol_identity(first)["runtime"]),
            set(full_store.PROTOCOL_RUNTIME_KEYS),
        )
        first_binding = full_store.protocol_binding(first)
        second_binding = full_store.protocol_binding(second)
        self.assertNotEqual(
            first_binding["provenance_sha256"],
            second_binding["provenance_sha256"],
        )
        self.assertEqual(
            first_binding["protocol_identity_sha256"],
            second_binding["protocol_identity_sha256"],
        )
        combined = full_store.aggregate_runner_summaries([first, second])
        self.assertEqual(combined["counts"]["requests"], 12)
        self.assertEqual(combined["counts"]["completions"], 288)
        self.assertEqual(combined["counts"]["sampled_tokens"], 3000)
        self.assertEqual(combined["timing"]["generation_seconds"], 4.0)
        drifted = copy.deepcopy(second)
        drifted["runtime"]["packages"]["vllm"] = "different"
        with self.assertRaisesRegex(full_store.FullArtifactError, "protocol identity differs"):
            full_store.aggregate_runner_summaries([first, drifted])
        with self.assertRaisesRegex(full_store.FullArtifactError, "cache protocol identity drift"):
            full_store.require_protocol_identity(
                drifted,
                full_store.protocol_identity(first),
                where="cache",
            )

    def test_analyzer_verifies_catalog_selection_and_every_expected_receipt(self) -> None:
        config = copy.deepcopy(run.load_config())
        arms = ["base", "mined"]
        config["inference"]["arms"] = arms
        task_rows = full_tasks()
        tasks = {row["id"]: row for row in task_rows}
        libraries = {arm: {} for arm in arms}
        plan = full_store.build_shard_plan(task_rows, arms)
        plan_hash = full_store.plan_sha256(plan)
        budget = 32768
        with tempfile.TemporaryDirectory() as directory:
            fixture_exp = Path(directory) / "fixture_experiment"
            analysis_dir = fixture_exp / "analysis"
            source_dir = fixture_exp / "src"
            root = Path(directory) / "external"
            analysis_dir.mkdir(parents=True)
            source_dir.mkdir(parents=True)
            runner_bytes = (EXP / "src" / "vllm_runner.py").read_bytes()
            (source_dir / "vllm_runner.py").write_bytes(runner_bytes)
            runner_hash = hashlib.sha256(runner_bytes).hexdigest()
            config["full_run"]["external_root"] = str(root)
            (analysis_dir / "full_shard_plan.json").write_text(
                json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            selection_arms: dict[str, object] = {}
            completed_entries: list[dict[str, object]] = []
            for arm in arms:
                shard_audits = []
                for spec in plan["arms"][arm]["shards"]:
                    shard_dir, receipt = write_protocol_shard(
                        root,
                        runner_sha256=runner_hash,
                        plan_hash=plan_hash,
                        config=config,
                        budget=budget,
                        arm=arm,
                        shard_index=int(spec["shard_index"]),
                        task_ids=list(spec["task_ids"]),
                        k=int(spec["k"]),
                    )
                    shard_audits.append(
                        {"shard_index": spec["shard_index"], "status": "complete"}
                    )
                    completed_entries.append(
                        full_store.catalog_shard_entry(root, shard_dir, receipt)
                    )
                selection_arms[arm] = {
                    "status": "complete",
                    "complete": True,
                    "adequate": True,
                    "shards": shard_audits,
                }
            selection = {
                "schema_version": 1,
                "run": "full",
                "pass": True,
                "selected_thinking_budget": budget,
                "starting_thinking_budget": budget,
                "shard_plan_sha256": plan_hash,
                "canonical_external_root": str(root.resolve()),
                "passed_smoke_selection": {
                    "path": "analysis/smoke_budget_selection.json",
                    "sha256": "pending",
                },
                "tiers": [
                    {
                        "budget": budget,
                        "status": "selectable",
                        "complete": True,
                        "adequate": True,
                        "rejecting_arm": None,
                        "arms": selection_arms,
                    }
                ],
            }
            smoke_path = analysis_dir / "smoke_budget_selection.json"
            smoke_path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "pass": True,
                        "selected_thinking_budget": budget,
                    },
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
            (analysis_dir / "interface_gate.json").write_text(
                '{"pass":true}\n', encoding="utf-8"
            )
            (analysis_dir / "smoke_verdict.json").write_text(
                '{"smoke_gate":{"pass":true}}\n', encoding="utf-8"
            )
            smoke_hash = full_store.file_integrity(smoke_path)["sha256"]
            selection["passed_smoke_selection"]["sha256"] = smoke_hash
            selection_path = analysis_dir / "full_budget_selection.json"
            selection_path.write_text(
                json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            completed_entries.sort(
                key=lambda row: (arms.index(str(row["arm"])), int(row["shard_index"]))
            )
            catalog = {
                "schema_version": full_store.FULL_CATALOG_SCHEMA_VERSION,
                "experiment_id": fixture_exp.name,
                "status": "selected",
                "canonical_external_root": str(root.resolve()),
                "protocol_binding": {"binding": "fixed"},
                "shard_plan": {
                    "path": "analysis/full_shard_plan.json",
                    **full_store.file_integrity(analysis_dir / "full_shard_plan.json"),
                    "content_sha256": plan_hash,
                },
                "budget_selection": {
                    "path": "analysis/full_budget_selection.json",
                    **full_store.file_integrity(selection_path),
                },
                "starting_thinking_budget": budget,
                "passed_smoke_selection": {
                    "path": "analysis/smoke_budget_selection.json",
                    "sha256": smoke_hash,
                },
                "selected_tier": {
                    "thinking_budget": budget,
                    "relative_path": f"think_{budget}",
                    "logical_promotion_only": True,
                    "repository_raw_copy": None,
                },
                "arm_order": arms,
                "completed_shards": completed_entries,
                "temporary_shards": [],
                "selected_shards": completed_entries,
            }
            (analysis_dir / "full_artifact_catalog.json").write_text(
                json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            with (
                mock.patch.object(analyze, "_expected_full_arm_order", return_value=arms),
                mock.patch.object(
                    analyze.full_store,
                    "build_full_binding",
                    return_value={"binding": "fixed"},
                ),
            ):
                verified = analyze._verify_full_artifact_catalog(
                    exp=fixture_exp,
                    config=config,
                    tasks=tasks,
                    libraries=libraries,
                )
            self.assertEqual(verified["selected_budget"], budget)
            self.assertEqual(len(verified["selected_entries"]), 30)
            base_rows, base_summary = analyze._load_full_arm_artifacts(verified, "base")
            mined_rows, mined_summary = analyze._load_full_arm_artifacts(verified, "mined")
            self.assertEqual(len(base_rows), 120)
            self.assertEqual(base_summary["counts"]["completions"], 2880)
            self.assertEqual(len(mined_rows), 120)
            self.assertEqual(mined_summary["counts"]["completions"], 1440)

            with selection_path.open("a", encoding="utf-8") as handle:
                handle.write(" \n")
            with (
                mock.patch.object(analyze, "_expected_full_arm_order", return_value=arms),
                mock.patch.object(
                    analyze.full_store,
                    "build_full_binding",
                    return_value={"binding": "fixed"},
                ),
                self.assertRaisesRegex(ValueError, "selection reference drift"),
            ):
                analyze._verify_full_artifact_catalog(
                    exp=fixture_exp,
                    config=config,
                    tasks=tasks,
                    libraries=libraries,
                )


class FullRunnerTests(unittest.TestCase):
    def test_atomic_generator_commits_only_four_file_final_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            final = full_store.shard_directory(
                root, budget=32768, arm="base", shard_index=0
            )
            record = {"id": "t0::base", "meta": {"task_id": "t0", "arm": "base"}}
            preflight = preflight_fixture(["t0"], arm="base", budget=32768)
            rows = rows_fixture(
                ["t0"], arm="base", k=2, preflight=preflight
            )
            batch = types.SimpleNamespace(
                rows=rows,
                summary=summary(requests=1, completions=2, sampled_tokens=2),
            )
            fake_harness = types.SimpleNamespace(
                generate_vllm_batch=mock.Mock(return_value=batch)
            )
            with (
                mock.patch.object(run, "_preflight_records", return_value=preflight),
                mock.patch.object(run, "_validate_runner_artifact", return_value=True),
                mock.patch.object(
                    run,
                    "_current_full_protocol_identity",
                    return_value=full_store.protocol_identity(batch.summary),
                ),
            ):
                receipt = run._generate_atomic_full_shard(
                    runner=object(),
                    harness=fake_harness,
                    config={},
                    root=root,
                    shard_dir=final,
                    shard_plan_sha256="d" * 64,
                    budget=32768,
                    arm="base",
                    shard_index=0,
                    task_ids=("t0",),
                    k=2,
                    records=(record,),
                    sampling=object(),
                    expected_protocol_identity=full_store.protocol_identity(batch.summary),
                )
            self.assertTrue(final.is_dir())
            self.assertEqual(
                {path.name for path in final.iterdir()},
                set(full_store.PAYLOAD_FILES) | {full_store.RECEIPT_FILE},
            )
            self.assertEqual(receipt["status"], "complete")
            self.assertFalse(any(final.parent.glob(".shard_000.tmp-*")))

    def test_irreversible_bounds_fire_at_first_impossible_count(self) -> None:
        config = run.load_config()
        zero = {
            "samples": 2880,
            "cap_contacts": 0,
            "unresolved_cap_contacts": 0,
            "periodic_loop_contacts": 0,
            "answer_limit_contacts": 0,
            "stage2_truncations": 0,
        }
        for arm, field, below, at in (
            ("base", "unresolved_cap_contacts", 143, 144),
            ("base", "answer_limit_contacts", 143, 144),
            ("base", "periodic_loop_contacts", 720, 721),
            ("mined", "unresolved_cap_contacts", 71, 72),
            ("mined", "answer_limit_contacts", 71, 72),
            ("mined", "periodic_loop_contacts", 360, 361),
        ):
            with self.subTest(arm=arm, field=field):
                counts = dict(zero)
                counts[field] = below
                self.assertIsNone(run._full_early_fail_reason(counts, arm=arm, config=config))
                counts[field] = at
                self.assertIn(field, run._full_early_fail_reason(counts, arm=arm, config=config))

    def test_full_stage_stops_after_first_irreversible_shard_and_skips_remainder(self) -> None:
        config = copy.deepcopy(run.load_config())
        tasks = full_tasks()

        def records(**kwargs: object) -> list[dict[str, object]]:
            arm = str(kwargs["arm"])
            return [
                {"id": f"{task['id']}::{arm}"}
                for task in kwargs["tasks"]  # type: ignore[union-attr]
            ]

        termination = {
            "samples": 144,
            "cap_contacts": 144,
            "unresolved_cap_contacts": 144,
            "periodic_loop_contacts": 0,
            "answer_limit_contacts": 0,
            "stage2_truncations": 0,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "external"
            analysis_dir = Path(directory) / "analysis"
            analysis_dir.mkdir()
            (analysis_dir / "smoke_budget_selection.json").write_text(
                "{}\n", encoding="utf-8"
            )
            config["full_run"]["external_root"] = str(root)
            libraries = {arm: {} for arm in full_store.NON_QWEN_ARMS}
            with (
                mock.patch.object(run, "ANALYSIS", analysis_dir),
                mock.patch.object(run, "_solver_records", side_effect=records),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    return_value=types.SimpleNamespace(thinking_budget=61440),
                ),
                mock.patch.object(run, "_validate_cached_full_shard", return_value={"status": "complete"}) as validate,
                mock.patch.object(run, "_current_full_protocol_identity", return_value={"protocol": "fixed"}),
                mock.patch.object(run, "_read_jsonl", return_value=[{"sealed": True}]),
                mock.patch.object(run, "_termination_metrics", return_value=termination),
                mock.patch.object(run.full_store, "build_full_binding", return_value={"binding": "fixed"}),
                mock.patch.object(
                    run.full_store,
                    "catalog_shard_entry",
                    return_value={"receipt": "bound"},
                ),
                mock.patch.object(run, "_write_full_artifact_catalog") as catalog,
            ):
                with self.assertRaisesRegex(ValueError, "setup-inconclusive"):
                    run._run_full_scientific_stage(
                        runner=object(),
                        harness=object(),
                        domain=object(),
                        config=config,
                        tasks=tasks,
                        libraries=libraries,
                        demonstrations=[],
                        starting_budget=61440,
                    )
            self.assertEqual(validate.call_count, 100)
            self.assertEqual(catalog.call_count, 3)
            selection = json.loads(
                (analysis_dir / "full_budget_selection.json").read_text(encoding="utf-8")
            )
            base = selection["tiers"][0]["arms"]["base"]
            self.assertEqual(base["status"], "irreversibly_rejected")
            self.assertEqual(base["termination"]["unresolved_cap_contacts"], 144)
            self.assertEqual(base["shards"][0]["status"], "complete")
            self.assertTrue(all(row["status"] == "skipped" for row in base["shards"][1:]))
            self.assertTrue(
                all(
                    selection["tiers"][0]["arms"][arm]["status"] == "skipped"
                    for arm in full_store.NON_QWEN_ARMS[1:]
                )
            )


class CompactAnalysisTests(unittest.TestCase):
    def test_compact_full_task_keeps_hash_selected_program_and_grades_only(self) -> None:
        candidate = {
            "sample_index": 0,
            "completion_sha256": "e" * 64,
            "program": ["ADD1"],
            "expanded_program": ["ADD1"],
            "surface_depth": 1,
            "expanded_depth": 1,
            "visible_correct": 8,
            "visible_total": 8,
            "visible_pass": True,
            "hidden_correct": 8,
            "hidden_total": 8,
            "hidden_pass": True,
            "probe_correct": 8,
            "probe_total": 8,
            "probe_pass": True,
            "macro_used": False,
            "macro_tokens": [],
            "sampled_tokens": 10,
            "thinking_tokens": 5,
            "answer_tokens": 5,
            "injected_tokens": 0,
            "forced_close": False,
            "unresolved_cap_contact": False,
            "periodic_loop": False,
            "answer_limit_contact": False,
            "parsed": True,
            "valid": True,
            "selected": True,
        }
        task = {
            "task_id": "t0",
            "split": "reuse",
            "arm": "base",
            "library_id": "base-id",
            "target_min_depth": 5,
            "n_samples": 1,
            "unique_prompt_tokens": 100,
            "candidates": [candidate],
            "oracle_hidden_pass": True,
            "abstained": False,
        }
        compact = analyze._compact_full_task(task)
        serialized = json.dumps(compact)
        self.assertNotIn("candidates", compact)
        self.assertNotIn("raw_text", serialized)
        self.assertNotIn("token_ids", serialized)
        self.assertEqual(compact["selected"]["program"], ["ADD1"])
        self.assertEqual(compact["selected"]["completion_sha256"], "e" * 64)
        self.assertTrue(compact["selected"]["hidden_pass"])


if __name__ == "__main__":
    unittest.main()
