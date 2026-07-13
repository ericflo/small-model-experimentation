from __future__ import annotations

import importlib.util
import inspect
import json
import math
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import yaml


EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import mechanics as core  # noqa: E402
from task_data import (  # noqa: E402
    ALIASES,
    CONCRETE_OPERATIONS,
    operation_from_record,
)

SPEC = importlib.util.spec_from_file_location(
    "materialized_residual_mechanics_runner", EXP / "scripts" / "run_mechanics.py"
)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def load_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def write_canonical(path: Path, value: object) -> None:
    path.write_bytes(runner.json_bytes(value))


def synthetic_prepare_receipt() -> dict:
    construction = runner.read_json(runner.CONSTRUCTION_MANIFEST)
    published = construction["tokenizer"]
    invocations = {}
    for name in runner.INVOCATIONS:
        source_name = "suffix_echo_mechanics_live" if name == "suffix_echo" else name
        invocations[name] = deepcopy(published["conditions"][source_name])
    return {
        "tokenizer": {
            "plain_alias_token_ids": {
                alias: published["alias_token_ids"][alias]["plain"][0]
                for alias in ALIASES
            },
            "forced_close_token_ids": published["forced_close_token_ids"],
            "invocations": invocations,
        }
    }


def valid_preflight(lock_path: Path) -> tuple[dict, dict, dict, Path]:
    config = load_config()
    receipt = synthetic_prepare_receipt()
    receipt_path = lock_path.parent / "preoutcome.json"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    parent_preflight = (
        ROOT
        / runner.PARENT_LINEAGE["attempt_2_live_preflight"]["path"]
    )
    preflight = deepcopy(runner.read_json(parent_preflight))
    preflight["schema_version"] = 2
    preflight["implementation_lock_sha256"] = runner.sha256_file(lock_path)
    preflight["prepare_receipt_sha256"] = runner.sha256_file(receipt_path)
    cache = preflight["live_cache"]
    blocks_per_sequence = runner._validate_group_aware_cache_geometry(
        cache, runner._engine_config(config)
    )["blocks_per_max_request"]
    rows: dict[str, dict] = {}
    for name in runner.INVOCATIONS:
        token_row = receipt["tokenizer"]["invocations"][name]
        sampling = runner._sampling(
            name, config, receipt["tokenizer"]["plain_alias_token_ids"]
        )
        reserve = (
            int(sampling.thinking_budget)
            + len(receipt["tokenizer"]["forced_close_token_ids"])
            + sampling.answer_max_tokens
            if sampling.thinking == "budget"
            else sampling.max_tokens
        )
        active = min(runner.EXPECTED_COUNTS[name], runner._engine_config(config).max_num_seqs)
        required = active * blocks_per_sequence
        rows[name] = {
            "requests": runner.EXPECTED_COUNTS[name],
            "prompt_tokens_min": token_row["prompt_tokens_min"],
            "prompt_tokens_max": token_row["prompt_tokens_max"],
            "reserve_tokens": reserve,
            "max_prompt_plus_reserve": token_row["prompt_tokens_max"] + reserve,
            "active_sequences": active,
            "reserved_blocks_per_sequence": blocks_per_sequence,
            "required_cache_blocks": required,
            "remaining_cache_blocks": cache["num_gpu_blocks"] - required,
        }
    preflight["invocations"] = rows
    return config, receipt, preflight, receipt_path


class _HashTokenizer:
    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, enable_thinking
    ) -> str:
        assert not tokenize and add_generation_prompt
        return messages[0]["content"] + f"\nTHINKING={enable_thinking}"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return list(bytes.fromhex(runner.sha256_bytes(text.encode())))


class RequestConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public = runner.read_jsonl(runner.PUBLIC_PATH)
        cls.audit = runner.read_jsonl(runner.AUDIT_PATH)
        runner._validate_public_inputs(cls.public, cls.audit)
        cls.requests = runner._build_requests(cls.public, cls.audit)

    def test_exact_nine_invocation_inventory(self) -> None:
        self.assertEqual(tuple(self.requests), runner.INVOCATIONS)
        self.assertEqual(
            {name: len(rows) for name, rows in self.requests.items()},
            runner.EXPECTED_COUNTS,
        )
        self.assertEqual(sum(map(len, self.requests.values())), 1984)

    def test_parent_scientific_scoring_and_prompt_protocol_are_byte_identical(self) -> None:
        parent = ROOT / "experiments" / "qwen35_4b_materialized_residual_sibling_search"
        self.assertEqual(
            runner.sha256_file(EXP / "src" / "mechanics.py"),
            runner.sha256_file(parent / "src" / "mechanics.py"),
        )
        self.assertEqual(
            runner.sha256_file(EXP / "src" / "protocol.py"),
            runner.sha256_file(parent / "src" / "protocol.py"),
        )
        parent_config = yaml.safe_load((parent / "configs" / "default.yaml").read_text())
        fresh_config = load_config()
        for section in ("model", "data", "aliases", "generation", "mechanics"):
            self.assertEqual(fresh_config[section], parent_config[section])

    def test_paired_causal_arms_share_ids_order_and_seed_keys(self) -> None:
        for arms in (runner.SUFFIX_ARMS, runner.BINARY_ARMS):
            reference = self.requests[arms[0]]
            ids = [row["id"] for row in reference]
            seed_keys = [row["meta"]["seed_key"] for row in reference]
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(len(value) == 64 for value in ids))
            for arm in arms[1:]:
                self.assertEqual([row["id"] for row in self.requests[arm]], ids)
                self.assertEqual(
                    [row["meta"]["seed_key"] for row in self.requests[arm]],
                    seed_keys,
                )

    def test_suffix_rows_are_exactly_public_live_rows(self) -> None:
        live = core.public_live_map(self.audit)
        observed: dict[str, set[str]] = {task_id: set() for task_id in live}
        for row in self.requests["suffix_materialized"]:
            observed[row["meta"]["task_id"]].add(row["meta"]["candidate_alias"])
            self.assertIsNone(row["meta"]["supplied_suffix"])
        self.assertEqual(observed, live)
        self.assertTrue(
            all(row["meta"]["supplied_suffix"] for row in self.requests["suffix_echo"])
        )

    def test_binary_rows_cover_every_task_candidate_once(self) -> None:
        pairs = [
            (row["meta"]["task_id"], row["meta"]["candidate_alias"])
            for row in self.requests["viability_materialized"]
        ]
        self.assertEqual(len(pairs), 24 * 24)
        self.assertEqual(len(set(pairs)), 24 * 24)
        for task in self.public:
            aliases = [alias for task_id, alias in pairs if task_id == task["task_id"]]
            self.assertEqual(tuple(aliases), ALIASES)

    def test_seed_domains_do_not_collide(self) -> None:
        domains = {
            name: {row["id"] for row in rows}
            for name, rows in self.requests.items()
            if name in {"suffix_materialized", "direct", "viability_materialized", "listwise"}
        }
        names = tuple(domains)
        for index, left in enumerate(names):
            for right in names[index + 1 :]:
                self.assertTrue(domains[left].isdisjoint(domains[right]))

    def test_actual_parent_request_identity_and_terminal_prompt_gate(self) -> None:
        receipt = runner._request_freshness_receipt(
            self.requests, _HashTokenizer(), load_config()
        )
        self.assertEqual(
            receipt["required_zero_intersections"],
            {
                "task_ids": 0,
                "request_ids": 0,
                "canonical_seed_keys": 0,
                "derived_stage1_seeds": 0,
                "derived_stage2_seeds": 0,
                "derived_any_stage_seeds": 0,
                "all_parent_user_prompts": 0,
                "terminal_user_prompts": 0,
                "terminal_rendered_prompt_token_ids": 0,
            },
        )
        self.assertEqual(receipt["fresh_unique_request_ids"], 676)

    def test_parent_request_seed_prompt_and_derived_seed_mutations_fail(self) -> None:
        parent = runner._authenticated_parent_requests(
            runner.verify_parent_lineage(ROOT)
        )
        mutated = deepcopy(self.requests)
        mutated["direct"][0]["id"] = parent["direct"][0]["id"]
        with self.assertRaisesRegex(RuntimeError, "derivation|freshness gate"):
            runner._request_freshness_receipt(
                mutated, _HashTokenizer(), load_config(), parent_requests=parent
            )

        mutated = deepcopy(self.requests)
        mutated["direct"][0]["meta"]["seed_key"] = deepcopy(
            parent["direct"][0]["meta"]["seed_key"]
        )
        with self.assertRaisesRegex(RuntimeError, "derivation|freshness gate"):
            runner._request_freshness_receipt(
                mutated, _HashTokenizer(), load_config(), parent_requests=parent
            )

        mutated = deepcopy(self.requests)
        mutated["direct"][0]["messages"] = deepcopy(
            parent["suffix_materialized"][0]["messages"]
        )
        with self.assertRaisesRegex(RuntimeError, "freshness gate"):
            runner._request_freshness_receipt(
                mutated, _HashTokenizer(), load_config(), parent_requests=parent
            )

        original_seed = runner.runner_stable_seed
        fresh_id = self.requests["direct"][0]["id"]
        parent_id = parent["listwise"][0]["id"]

        def cross_stage_seed(run_seed, record_id, sample_index, stage):
            if run_seed == 2026072702 and record_id == fresh_id and stage == "stage1":
                return 1
            if run_seed == 2026072602 and record_id == parent_id and stage == "stage2":
                return 1
            return original_seed(run_seed, record_id, sample_index, stage)

        with mock.patch.object(
            runner, "runner_stable_seed", side_effect=cross_stage_seed
        ):
            with self.assertRaisesRegex(RuntimeError, "freshness gate"):
                runner._request_freshness_receipt(
                    self.requests,
                    _HashTokenizer(),
                    load_config(),
                    parent_requests=parent,
                )

    def test_arbitrary_and_cross_family_request_id_mutations_fail(self) -> None:
        mutated = deepcopy(self.requests)
        mutated["direct"][0]["id"] = "f" * 64
        with self.assertRaisesRegex(RuntimeError, "derivation"):
            runner._request_freshness_receipt(
                mutated, _HashTokenizer(), load_config()
            )
        mutated = deepcopy(self.requests)
        mutated["direct"][0]["id"] = mutated["listwise"][0]["id"]
        with self.assertRaisesRegex(RuntimeError, "derivation"):
            runner._request_freshness_receipt(
                mutated, _HashTokenizer(), load_config()
            )

    def test_sampling_and_engine_settings_are_exact(self) -> None:
        config = load_config()
        token_ids = {alias: 32 + index for index, alias in enumerate(ALIASES)}
        suffix = runner._sampling("suffix_materialized", config, token_ids)
        direct = runner._sampling("direct", config, token_ids)
        binary = runner._sampling("viability_materialized", config, token_ids)
        listwise = runner._sampling("listwise", config, token_ids)
        for sampling in (suffix, direct, binary, listwise):
            self.assertEqual(sampling.n, 1)
            self.assertEqual(sampling.temperature, 0.6)
            self.assertEqual(sampling.top_p, 0.95)
            self.assertEqual(sampling.top_k, 20)
            self.assertFalse(sampling.greedy)
            self.assertEqual(sampling.run_seed, 2026072702)
        self.assertEqual((suffix.thinking_budget, suffix.answer_max_tokens), (512, 64))
        self.assertEqual((direct.thinking_budget, direct.answer_max_tokens), (1024, 96))
        self.assertEqual((binary.thinking, binary.max_tokens, binary.logprobs), ("off", 1, 2))
        self.assertEqual(listwise.logprobs, 24)
        self.assertEqual(listwise.logprob_token_ids, tuple(range(32, 56)))
        expected_engine = runner._expected_engine_args(config)
        self.assertEqual(expected_engine["max_logprobs"], 24)
        self.assertEqual(expected_engine["logprobs_mode"], "raw_logprobs")
        self.assertEqual(expected_engine["max_num_seqs"], 64)
        self.assertEqual(
            expected_engine["cudagraph_capture_sizes"], [1, 2, 4, 8, 16, 32, 64]
        )
        self.assertFalse(expected_engine["enable_prefix_caching"])
        self.assertEqual(expected_engine["mamba_cache_mode"], "none")

    def test_prepare_allowlist_excludes_hidden_and_future_data(self) -> None:
        relative = "\n".join(runner.PREPARE_SOURCE_FILES).lower()
        self.assertNotIn("mechanics_gold", relative)
        self.assertNotIn("qualification_", relative)
        self.assertNotIn("confirmation_", relative)
        self.assertNotIn("benchmarks/", relative)
        self.assertEqual(runner.PUBLIC_PATH.name, "mechanics_public.jsonl")
        self.assertEqual(runner.AUDIT_PATH.name, "mechanics_audit.jsonl")

    def test_pending_or_duplicate_review_verdict_blocks_preparation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            review = Path(value) / "review.md"
            for text in (
                "pending",
                runner.IMPLEMENTATION_REVIEW_VERDICT
                + "\n"
                + runner.IMPLEMENTATION_REVIEW_AUTHORIZATION
                + "\n"
                + runner.IMPLEMENTATION_REVIEW_VERDICT,
            ):
                review.write_text(text + "\n")
                with mock.patch.object(
                    runner, "IMPLEMENTATION_REVIEW", review
                ), self.assertRaisesRegex(RuntimeError, "does not authorize"):
                    runner.build_prepared()
                with mock.patch.object(
                    runner, "IMPLEMENTATION_REVIEW", review
                ), mock.patch.object(
                    runner, "_validate_current_scientific_environment"
                ), mock.patch.object(
                    runner, "write_frozen"
                ) as write, self.assertRaisesRegex(
                    RuntimeError, "does not authorize"
                ):
                    runner.prepare()
                write.assert_not_called()

    def test_v2_requires_exact_v1_prepared_payload_table(self) -> None:
        with mock.patch.object(
            runner, "_validate_v2_review_authorization"
        ), mock.patch.object(
            runner, "_prepared_file_table", return_value={}
        ), self.assertRaisesRegex(RuntimeError, "payload table changed"):
            runner.build_prepared()

    def test_lock_incident_and_split_frozen_commit_mapping_are_exact(self) -> None:
        incident = runner.build_lock_attempt_1_incident()
        self.assertEqual(incident["decision"], "FAILED_CLOSED_BEFORE_LOCK_CREATION")
        self.assertEqual(incident["model_calls"], 0)
        self.assertFalse(incident["implementation_lock_created"])
        self.assertEqual(
            runner.sha256_file(runner.PRIOR_PREOUTCOME_RECEIPT),
            runner.PRIOR_PREOUTCOME_SHA256,
        )
        for relative in runner.CONSTRUCTION_OUTPUT_FROZEN_FILES:
            self.assertEqual(
                runner._frozen_commit_for(relative),
                runner.PUBLISHED_CONSTRUCTION_COMMIT,
            )
        self.assertEqual(
            runner._frozen_commit_for(str(runner.EXP_REL / "src/identity.py")),
            runner.DESIGN_COMMIT,
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            later_raw = Path(value) / "raw"
            later_raw.mkdir()
            with mock.patch.object(runner, "RAW", later_raw):
                self.assertEqual(
                    runner.build_lock_attempt_1_incident(), incident
                )
                with self.assertRaisesRegex(RuntimeError, "before later artifacts"):
                    runner.build_lock_attempt_1_incident(require_pristine=True)

    def test_dangling_symlinks_cannot_be_replaced_by_frozen_writes_or_pristine_gate(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            for name in (
                "lock_attempt_1_incident.json",
                "preoutcome_receipt_v2.json",
                "implementation_lock.json",
            ):
                path = root / name
                path.symlink_to(root / "missing-target")
                with self.subTest(name=name), self.assertRaisesRegex(
                    RuntimeError, "path is a symlink"
                ):
                    runner.write_frozen(path, {"status": "FORBIDDEN"})
                self.assertTrue(path.is_symlink())
            dangling_lock = root / "implementation_lock_dangling.json"
            dangling_lock.symlink_to(root / "missing-lock")
            with mock.patch.object(
                runner, "IMPLEMENTATION_LOCK", dangling_lock
            ), self.assertRaisesRegex(RuntimeError, "before later artifacts"):
                runner.build_lock_attempt_1_incident(require_pristine=True)


class EnvironmentAuthenticationTests(unittest.TestCase):
    def test_group_aware_hybrid_cache_geometry_uses_vllm_floor(self) -> None:
        engine = runner._engine_config(load_config())
        cache = {
            "num_gpu_blocks": 2042,
            "block_size": 528,
            "kv_cache_size_tokens": 760366,
            "kv_cache_max_concurrency": 185.63636363636363,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "mamba_block_size": 4096,
        }
        geometry = runner._validate_group_aware_cache_geometry(cache, engine)
        self.assertEqual(
            geometry,
            {
                "blocks_per_max_request": 11,
                "attention_blocks_at_max": 8,
                "mamba_blocks_at_max": 3,
                "mamba_group_count": 3,
            },
        )
        self.assertEqual(
            int(cache["kv_cache_max_concurrency"] * engine.max_model_len),
            cache["kv_cache_size_tokens"],
        )
        self.assertFalse(
            math.isclose(
                cache["kv_cache_max_concurrency"],
                cache["kv_cache_size_tokens"] / engine.max_model_len,
                rel_tol=1e-12,
                abs_tol=1e-12,
            )
        )
        for key, value in (
            ("kv_cache_size_tokens", 760367),
            ("kv_cache_max_concurrency", 185.5),
            ("block_size", 512),
            ("mamba_block_size", 2048),
            ("num_gpu_blocks", 63),
        ):
            corrupted = dict(cache)
            corrupted[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(
                RuntimeError, "cache geometry"
            ):
                runner._validate_group_aware_cache_geometry(corrupted, engine)

    def test_703_block_boundary_rejects_64_active_sequences(self) -> None:
        engine = runner._engine_config(load_config())
        concurrency = 703 / 11
        cache = {
            "num_gpu_blocks": 703,
            "block_size": 528,
            "kv_cache_size_tokens": int(concurrency * engine.max_model_len),
            "kv_cache_max_concurrency": concurrency,
            "enable_prefix_caching": False,
            "mamba_cache_mode": "none",
            "mamba_block_size": 4096,
        }
        old_token_proxy = (
            engine.max_num_seqs * math.ceil(824 / cache["block_size"])
            * cache["block_size"]
        )
        self.assertLessEqual(old_token_proxy, cache["kv_cache_size_tokens"])
        self.assertGreater(engine.max_num_seqs * 11, cache["num_gpu_blocks"])
        with self.assertRaisesRegex(RuntimeError, "cache geometry"):
            runner._validate_group_aware_cache_geometry(cache, engine)

    def test_recorded_preflight_rejects_each_persisted_block_field(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            lock_path = Path(value) / "lock.json"
            lock_path.write_text("{}\n")
            config, receipt, preflight, receipt_path = valid_preflight(lock_path)
            with mock.patch.object(runner, "PREOUTCOME_RECEIPT", receipt_path):
                runner._validate_recorded_live_preflight(
                    preflight, config, receipt, lock_path
                )
            for field in (
                "reserved_blocks_per_sequence",
                "required_cache_blocks",
                "remaining_cache_blocks",
            ):
                corrupted = deepcopy(preflight)
                corrupted["invocations"]["viability_materialized"][field] += 1
                with self.subTest(field=field), self.assertRaisesRegex(
                    RuntimeError, "invocation geometry"
                ):
                    with mock.patch.object(runner, "PREOUTCOME_RECEIPT", receipt_path):
                        runner._validate_recorded_live_preflight(
                            corrupted, config, receipt, lock_path
                        )

    def test_preflight_is_validated_before_pass_receipt_is_written(self) -> None:
        source = inspect.getsource(runner._live_preflight)
        self.assertLess(
            source.index("_validate_recorded_live_preflight"),
            source.index('path = RAW / "live_preflight.json"'),
        )

    def test_failed_live_validation_writes_no_pass_receipt(self) -> None:
        config = load_config()
        receipt = synthetic_prepare_receipt()
        prior = runner.read_json(
            ROOT / runner.PARENT_LINEAGE["attempt_2_live_preflight"]["path"]
        )
        engine = runner._engine_config(config)
        cache = SimpleNamespace(**prior["live_cache"])
        vllm_config = SimpleNamespace(
            cache_config=cache,
            scheduler_config=SimpleNamespace(**prior["live_scheduler"]),
            model_config=SimpleNamespace(**prior["live_model"]),
            parallel_config=SimpleNamespace(**prior["live_parallel"]),
        )
        fake = SimpleNamespace(
            config=engine,
            engine_args=runner._expected_engine_args(config),
            resolved_logprobs_mode="raw_logprobs",
            resolved_cudagraph=prior["resolved_cudagraph"],
            close_ids=receipt["tokenizer"]["forced_close_token_ids"],
            llm=SimpleNamespace(
                llm_engine=SimpleNamespace(vllm_config=vllm_config)
            ),
            runtime_metadata=lambda: prior["runtime"],
        )
        fake.prepare = lambda rows, *_args: [
            SimpleNamespace(prompt_token_ids=[1]) for _row in rows
        ]
        prepared = {
            name: [{} for _index in range(runner.EXPECTED_COUNTS[name])]
            for name in runner.INVOCATIONS
        }
        expected_token_hashes = [
            receipt["tokenizer"]["invocations"][name][
                "prompt_token_ids_sha256"
            ]
            for name in runner.INVOCATIONS
        ]
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            lock_path = Path(value) / "lock.json"
            lock_path.write_text("{}\n")
            receipt_path = Path(value) / "preoutcome.json"
            receipt_path.write_text(json.dumps(receipt) + "\n")
            with mock.patch.object(
                runner, "canonical_sha256", side_effect=expected_token_hashes
            ), mock.patch.object(
                runner,
                "_validate_recorded_live_preflight",
                side_effect=RuntimeError("synthetic validation failure"),
            ), mock.patch.object(
                runner, "write_exclusive_durable"
            ) as write, mock.patch.object(runner, "PREOUTCOME_RECEIPT", receipt_path):
                with self.assertRaisesRegex(RuntimeError, "synthetic validation"):
                    runner._live_preflight(
                        fake, config, prepared, receipt, lock_path
                    )
                write.assert_not_called()

    def test_fresh_boundary_authenticates_terminal_parent_lineage(self) -> None:
        lineage = runner.verify_parent_lineage(ROOT)
        self.assertEqual(lineage, runner.PARENT_LINEAGE)
        self.assertEqual(runner.RAW.name, "raw")
        self.assertEqual(runner.SCORED.name, "scored")
        self.assertEqual(runner.SUMMARY.name, "summary.json")
        self.assertEqual(runner.IMPLEMENTATION_LOCK.name, "implementation_lock.json")
        terminal = lineage["attempt_2_terminal_started"]
        self.assertEqual(
            runner.sha256_file(ROOT / terminal["path"]), terminal["sha256"]
        )

    def test_bootstrap_rejects_duplicate_stage_and_lock_spellings(self) -> None:
        invalid = (
            ["runner", "--stage", "prepare", "--stage=run"],
            ["runner", "--lock=a", "--lock", "b"],
            ["runner", "--stage"],
            ["runner", "--lock="],
        )
        for argv in invalid:
            with mock.patch.object(sys, "argv", argv), self.assertRaises(RuntimeError):
                runner._bootstrap_cli_options()
        with mock.patch.object(
            sys,
            "argv",
            ["runner", "--stage=analyze", "--lock", "receipt.json"],
        ):
            self.assertEqual(
                runner._bootstrap_cli_options(),
                {"--stage": "analyze", "--lock": "receipt.json"},
            )

    def test_lock_path_rejects_final_and_component_symlinks_before_resolve(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            real = root / "real"
            real.mkdir()
            lock = real / "lock.json"
            lock.write_text("{}\n")
            final_link = root / "lock-link.json"
            final_link.symlink_to(lock)
            directory_link = root / "dir-link"
            directory_link.symlink_to(real, target_is_directory=True)
            for path in (final_link, directory_link / "lock.json"):
                with self.subTest(path=path), self.assertRaisesRegex(
                    RuntimeError, "contains a symlink"
                ):
                    runner._bootstrap_safe_lock_path(str(path))
                with self.subTest(path=path), self.assertRaisesRegex(
                    RuntimeError, "contains a symlink"
                ):
                    runner.verify_implementation_lock(path)

    def test_base_interpreter_analyze_fails_before_artifact_work(self) -> None:
        result = subprocess.run(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(EXP / "scripts" / "run_mechanics.py"),
                "--stage",
                "analyze",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("pinned .venv-vllm interpreter", result.stderr)

    def test_lock_parser_covers_exact_and_direct_url_distributions(self) -> None:
        versions = runner._locked_environment_versions(load_config())
        self.assertEqual(versions["numpy"], "2.3.5")
        self.assertEqual(versions["torch"], "2.11.0+cu129")
        self.assertEqual(versions["transformers"], "5.13.0")
        self.assertEqual(versions["vllm"], "0.24.0+cu129")
        self.assertGreater(len(versions), 100)

    def test_missing_wrong_and_normalized_collision_packages_fail(self) -> None:
        config = load_config()
        packages = runner._locked_environment_versions(config)
        runner._validate_package_inventory(packages, config, source="fixture")
        missing = dict(packages)
        missing.pop("torch")
        wrong = dict(packages)
        wrong["transformers"] = "0.0.0"
        collision = {**packages, "flashinfer_python": packages["flashinfer-python"]}
        for value in (missing, wrong, collision):
            with self.subTest(value=len(value)), self.assertRaises(RuntimeError):
                runner._validate_package_inventory(value, config, source="fixture")

    def test_invocation_runtime_must_exactly_match_preflight(self) -> None:
        config = load_config()
        runtime = {
            "python": "3.12.0",
            "python_executable": str(ROOT / ".venv-vllm" / "bin" / "python"),
            "platform": "fixture-platform",
            "packages": runner._locked_environment_versions(config),
            "environment_lock": {
                "path": str(ROOT / "requirements-vllm.lock.txt"),
                "sha256": runner.sha256_file(ROOT / "requirements-vllm.lock.txt"),
            },
            "uv": "uv fixture",
            "cuda_toolkit": "cuda fixture",
            "gpu": "fixture GPU, driver, memory",
            "vllm_enable_v1_multiprocessing": "0",
            "git_commit": "a" * 40,
            "git_dirty": True,
        }
        preflight = {"runtime": deepcopy(runtime)}
        runner._validate_invocation_runtime(
            runtime, preflight, config, source="fixture"
        )
        expected_live_transition = deepcopy(runtime)
        expected_live_transition["git_commit"] = "b" * 40
        expected_live_transition["git_dirty"] = not runtime["git_dirty"]
        runner._validate_invocation_runtime(
            expected_live_transition,
            preflight,
            config,
            source="expected live-artifact dirty transition",
        )
        self.assertEqual(
            runner._runtime_projection(expected_live_transition),
            runner._runtime_projection(runtime),
        )
        changed = deepcopy(runtime)
        changed["gpu"] = "different GPU or driver"
        with self.assertRaisesRegex(RuntimeError, "live preflight runtime"):
            runner._validate_invocation_runtime(
                changed, preflight, config, source="fixture"
            )


class SurfaceControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.public = runner.read_jsonl(runner.PUBLIC_PATH)
        cls.audit = runner.read_jsonl(runner.AUDIT_PATH)
        cls.scores, cls.folds = core.build_surface_control(cls.public, cls.audit)

    def test_hand_computed_exact_relation_features(self) -> None:
        visible = []
        for index in range(8):
            values = [index, index + 1, index + 2]
            visible.append({"input": values, "output": list(reversed(values))})
        task = {
            "task_id": "feature-fixture",
            "depth": 3,
            "viability_live_alias": "A",
            "visible": visible,
            "unlabeled_probe_inputs": [[1, 2, 3]],
        }
        features = core.surface_features(task, CONCRETE_OPERATIONS[0])
        expected = np.asarray(
            [1.0, *([0.0] * 23), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        )
        np.testing.assert_array_equal(features, expected)

    def test_aligned_l1_is_per_row_l1_sum_not_coordinate_mae(self) -> None:
        visible = [
            {"input": [1, 2, 4], "output": [1, 2, 4]}
            for _ in range(8)
        ]
        task = {
            "task_id": "l1-fixture",
            "depth": 3,
            "viability_live_alias": "A",
            "visible": visible,
            "unlabeled_probe_inputs": [[1, 2, 3]],
        }
        features = core.surface_features(task, CONCRETE_OPERATIONS[0])
        self.assertEqual(features[-2], 6.0)

    def test_looto_geometry_convergence_and_finite_coverage(self) -> None:
        self.assertEqual(len(self.scores), 24 * 24)
        self.assertEqual(len(self.folds), 24)
        self.assertTrue(all(math.isfinite(row["score"]) for row in self.scores))
        self.assertTrue(
            all(
                fold["training_rows"] == 23 * 24
                and fold["held_rows"] == 24
                and fold["iterations"] < 10000
                for fold in self.folds
            )
        )
        for task in self.public:
            rows = [row for row in self.scores if row["task_id"] == task["task_id"]]
            self.assertEqual([row["candidate_alias"] for row in rows], list(ALIASES))

    def test_scaler_is_training_fold_only(self) -> None:
        held_id = self.public[0]["task_id"]
        train_features = []
        for task in self.public[1:]:
            for candidate in CONCRETE_OPERATIONS:
                train_features.append(core.surface_features(task, candidate))
        expected_mean = np.stack(train_features).mean(axis=0)
        fold = next(row for row in self.folds if row["held_task_id"] == held_id)
        np.testing.assert_allclose(fold["mean"], expected_mean, rtol=0.0, atol=0.0)

    def test_solver_is_row_order_invariant(self) -> None:
        live = core.public_live_map(self.audit)
        features = []
        labels = []
        for task in self.public[1:]:
            for alias, candidate in zip(ALIASES, CONCRETE_OPERATIONS, strict=True):
                features.append(core.surface_features(task, candidate))
                labels.append(float(alias in live[task["task_id"]]))
        x = np.stack(features)
        y = np.asarray(labels)
        forward = core.fit_balanced_ridge(x, y)
        reverse = core.fit_balanced_ridge(x[::-1], y[::-1])
        np.testing.assert_allclose(
            forward["coefficient"], reverse["coefficient"], rtol=0.0, atol=1e-12
        )
        self.assertAlmostEqual(forward["intercept"], reverse["intercept"], 12)

    def test_random_control_is_seeded_deterministic_and_realized(self) -> None:
        first = core.build_random_control(self.public, seed=2026072705)
        second = core.build_random_control(self.public, seed=2026072705)
        changed = core.build_random_control(self.public, seed=2026072706)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 576)


def ranking_output(values: dict[int | str, float]) -> dict:
    return {
        "n_sampled_tokens": 1,
        "seed_stage2": None,
        "stage2_logprobs": None,
        "stage1_logprobs": [
            {key: {"logprob": value, "rank": None} for key, value in values.items()}
        ],
    }


class _FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs) -> str:
        return "rendered"

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        self.last_encoded = (text, add_special_tokens)
        return [101, 102, 103]

    def decode(self, _token_ids: list[int], *, skip_special_tokens: bool) -> str:
        self.last_decode_flag = skip_special_tokens
        return "decoded"


def authenticated_generation_fixture() -> tuple[dict, list[dict], dict, list[dict], dict]:
    config = load_config()
    token_ids = {alias: 32 + index for index, alias in enumerate(ALIASES)}
    sampling = runner._sampling("direct", config, token_ids)
    request_id = "a" * 64
    meta = {"task_id": "fixture", "condition": "direct", "seed_key": ["fixture"]}
    prepared = [{"id": request_id, "messages": [{"role": "user", "content": "x"}], "meta": meta}]
    parent_seed = runner.runner_stable_seed(sampling.run_seed, request_id, -1, "stage1")
    stage2_seed = runner.runner_stable_seed(sampling.run_seed, request_id, 0, "stage2")
    output = {
        "sample_index": 0,
        "stage1_parent_seed": parent_seed,
        "seed_stage1": parent_seed,
        "seed_stage2": stage2_seed,
        "text": "decoded",
        "token_ids": [10, 11, 248069, 271, 12],
        "stage1_token_ids": [10, 11],
        "retained_thinking_token_ids": [10, 11],
        "injected_token_ids": [248069, 271],
        "stage2_token_ids": [12],
        "n_thinking_tokens": 2,
        "n_answer_tokens": 1,
        "n_sampled_tokens": 3,
        "n_injected_tokens": 2,
        "n_completion_tokens": 5,
        "n_terminal_tokens_trimmed": 0,
        "n_stage1_prompt_tokens": 3,
        "n_stage2_prompt_tokens": 7,
        "thinking_closed": True,
        "forced_close": True,
        "finish_reason": "stop",
        "stop_reason": None,
        "stage1_finish_reason": "length",
        "stage1_stop_reason": None,
        "truncated": False,
        "stage1_cumulative_logprob": None,
        "stage2_cumulative_logprob": None,
        "sampled_cumulative_logprob": None,
        "stage1_logprobs": None,
        "stage2_logprobs": None,
    }
    rows = [
        {
            "id": request_id,
            "meta": meta,
            "prompt_sha256": runner.sha256_bytes(b"rendered"),
            "n_prompt_tokens": 3,
            "prompt_channel": "thinking",
            "prompt_logprobs": None,
            "outputs": [output],
        }
    ]
    runtime = {
        "python": "3.12.0",
        "python_executable": str(ROOT / ".venv-vllm" / "bin" / "python"),
        "platform": "fixture-platform",
        "packages": runner._locked_environment_versions(config),
        "environment_lock": {
            "path": str(ROOT / "requirements-vllm.lock.txt"),
            "sha256": runner.sha256_file(ROOT / "requirements-vllm.lock.txt"),
        },
        "uv": "uv fixture",
        "cuda_toolkit": "cuda fixture",
        "gpu": "fixture GPU, driver, memory",
        "vllm_enable_v1_multiprocessing": "0",
        "git_commit": "b" * 40,
        "git_dirty": True,
    }
    resolved_cudagraph = {"fixture": "authenticated-by-preflight"}
    metadata = {
        "schema_version": 4,
        "model": runner.MODEL_ID,
        "model_revision": runner.MODEL_REVISION,
        "runner_sha256": runner.sha256_file(EXP / "src" / "vllm_runner.py"),
        "engine": runner._normalized(
            runner.dataclasses.asdict(runner._engine_config(config))
        ),
        "engine_args": runner._expected_engine_args(config),
        "resolved_cudagraph": resolved_cudagraph,
        "resolved_logprobs_mode": "raw_logprobs",
        "sampling": runner.dataclasses.asdict(sampling),
        "resolved_sampling": sampling.resolved_sampling(),
        "adapter": None,
        "think_token_ids": {
            "open": 248068,
            "close": 248069,
            "forced_close_sequence": [248069, 271],
            "thinking_prompt_suffix": [248045, 74455, 198, 248068, 198],
            "no_thinking_prompt_suffix": [248045, 74455, 198, 248068, 271, 248069, 271],
        },
        "termination": {
            "hf_model_eos_token_id": 248044,
            "vllm_tokenizer_eos_ignored": 248046,
        },
        "rng_isolation": {
            "engine_seed": 0,
            "caller_global_rng_state_restored": True,
        },
        "counts": {
            "requests": 1,
            "completions": 1,
            "unique_input_prompt_tokens": 3,
            "stage1_logical_prompt_tokens": 3,
            "stage2_logical_prompt_tokens": 7,
            "logical_model_input_tokens": 10,
            "sampled_tokens": 3,
            "injected_tokens": 2,
        },
        "runtime": runtime,
    }
    receipt = {
        "tokenizer": {
            "plain_alias_token_ids": token_ids,
            "think_open_token_ids": [248068],
            "think_close_token_ids": [248069],
            "forced_close_token_ids": [248069, 271],
            "thinking_prompt_suffix_ids": [248045, 74455, 198, 248068, 198],
            "no_thinking_prompt_suffix_ids": [248045, 74455, 198, 248068, 271, 248069, 271],
        }
    }
    preflight = {"runtime": deepcopy(runtime), "resolved_cudagraph": resolved_cudagraph}
    return config, rows, metadata, prepared, {"receipt": receipt, "preflight": preflight}


class InvocationAuthenticationTests(unittest.TestCase):
    def authenticate(self, rows: list[dict], metadata: dict, prepared: list[dict], context: dict) -> None:
        with mock.patch.object(
            runner, "EXPECTED_COUNTS", {**runner.EXPECTED_COUNTS, "direct": 1}
        ), mock.patch(
            "transformers.AutoTokenizer.from_pretrained",
            return_value=_FakeTokenizer(),
        ):
            runner._authenticate_invocation(
                "direct",
                rows,
                metadata,
                prepared,
                load_config(),
                context["receipt"],
                context["preflight"],
            )

    def test_valid_forced_continuation_authenticates(self) -> None:
        _config, rows, metadata, prepared, context = authenticated_generation_fixture()
        self.authenticate(rows, metadata, prepared, context)

    def test_corruptions_fail_closed(self) -> None:
        mutations = []
        _config, rows, metadata, prepared, context = authenticated_generation_fixture()
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        changed[0][0]["prompt_sha256"] = "0" * 64
        mutations.append(changed)
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        output = changed[0][0]["outputs"][0]
        output["retained_thinking_token_ids"] = [99, 11]
        output["token_ids"] = [99, 11, 248069, 271, 12]
        mutations.append(changed)
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        output = changed[0][0]["outputs"][0]
        output["stage1_token_ids"] = [10, 11, 248069]
        output["n_sampled_tokens"] = 4
        output["forced_close"] = False
        output["stage1_finish_reason"] = "stop"
        changed[1]["counts"]["sampled_tokens"] = 4
        mutations.append(changed)
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        changed[0][0]["outputs"][0]["truncated"] = True
        mutations.append(changed)
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        changed[1]["counts"]["sampled_tokens"] = 4
        mutations.append(changed)
        changed = (deepcopy(rows), deepcopy(metadata), deepcopy(context))
        changed[1]["runtime"]["gpu"] = "different driver"
        mutations.append(changed)
        for raw_rows, meta, ctx in mutations:
            with self.subTest(meta=meta), self.assertRaises(RuntimeError):
                self.authenticate(raw_rows, meta, prepared, ctx)


class RawLogprobTests(unittest.TestCase):
    def test_binary_orientation_and_float32_rule(self) -> None:
        output = ranking_output({32: -2.1250001, 33: -1.25})
        score_a, values_a = core.binary_rank_score(
            output, live_alias="A", token_ids={"A": 32, "B": 33}
        )
        score_b, _ = core.binary_rank_score(
            output, live_alias="B", token_ids={"A": 32, "B": 33}
        )
        expected = float(np.float32(values_a["A"]) - np.float32(values_a["B"]))
        self.assertEqual(score_a, expected)
        self.assertEqual(score_b, -expected)

    def test_listwise_extracts_all_plain_alias_ids(self) -> None:
        token_ids = {alias: 32 + index for index, alias in enumerate(ALIASES)}
        output = ranking_output({token_id: -float(index) for index, token_id in enumerate(token_ids.values())})
        scores, raw = core.listwise_rank_scores(output, token_ids=token_ids)
        self.assertEqual(tuple(scores), ALIASES)
        self.assertEqual(len(raw), 24)
        self.assertTrue(all(math.isfinite(value) for value in scores.values()))

    def test_missing_duplicate_nonfinite_and_multiposition_fail_closed(self) -> None:
        invalid = [
            ranking_output({32: -1.0}),
            ranking_output({32: -1.0, "32": -1.1, 33: -2.0}),
            ranking_output({32: math.nan, 33: -2.0}),
            ranking_output({32: math.inf, 33: -2.0}),
        ]
        multiple = ranking_output({32: -1.0, 33: -2.0})
        multiple["stage1_logprobs"].append({32: {"logprob": -1.0}})
        invalid.append(multiple)
        second_stage = ranking_output({32: -1.0, 33: -2.0})
        second_stage["seed_stage2"] = 3
        invalid.append(second_stage)
        for output in invalid:
            with self.subTest(output=output), self.assertRaises(ValueError):
                core.binary_rank_score(
                    output, live_alias="A", token_ids={"A": 32, "B": 33}
                )


class RankingMetricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.audit = runner.read_jsonl(runner.AUDIT_PATH)

    def test_task_metrics_are_order_invariant_and_complete(self) -> None:
        rows = [
            {
                "task_id": audit["task_id"],
                "candidate_alias": alias,
                "score": float(-index),
            }
            for audit in self.audit
            for index, alias in enumerate(ALIASES)
        ]
        forward = core.ranking_metrics(rows, self.audit)
        reverse = core.ranking_metrics(list(reversed(rows)), self.audit)
        self.assertEqual(forward, reverse)
        self.assertEqual(forward["score_rows"], 576)
        self.assertEqual(len(forward["task_rows"]), 24)

    def test_duplicate_and_incomplete_scores_fail(self) -> None:
        rows = [
            {
                "task_id": audit["task_id"],
                "candidate_alias": alias,
                "score": 0.0,
            }
            for audit in self.audit
            for alias in ALIASES
        ]
        with self.assertRaises(ValueError):
            core.ranking_metrics([*rows, rows[0]], self.audit)
        with self.assertRaises(ValueError):
            core.ranking_metrics(rows[:-1], self.audit)


def a_metrics_fixture() -> dict[str, dict]:
    base = {
        "rows": 52,
        "parse_successes": 47,
        "answer_cap_contacts": 2,
        "raw_visible_successes": 0,
        "task_any_live_successes": 0,
        "successful_parameterized_rows": 0,
        "successful_parameter_free_rows": 0,
    }
    metrics = {
        name: deepcopy(base)
        for name in (
            "suffix_materialized",
            "suffix_name_only",
            "suffix_shuffled",
            "suffix_echo",
        )
    }
    metrics["direct"] = {
        **deepcopy(base),
        "rows": 24,
        "parse_successes": 22,
        "answer_cap_contacts": 1,
    }
    metrics["suffix_echo"]["raw_visible_successes"] = 47
    metrics["suffix_materialized"].update(
        task_any_live_successes=9,
        successful_parameterized_rows=1,
        successful_parameter_free_rows=1,
    )
    metrics["suffix_name_only"]["task_any_live_successes"] = 5
    metrics["suffix_shuffled"]["task_any_live_successes"] = 5
    return metrics


def b_ranker(recall: float, *, materialized: bool = False) -> dict:
    return {
        "score_rows": 576,
        "finite_score_rows": 576,
        "mean_recall_at_4": recall,
        "mean_hit_at_4": 2.0 / 3.0 if materialized else 0.5,
        "hit_tasks": 16 if materialized else 12,
        "retrieved_live_operation_count": 10 if materialized else 8,
    }


def b_metrics_fixture() -> dict[str, dict]:
    return {
        "viability_materialized": b_ranker(0.60, materialized=True),
        "viability_name_only": b_ranker(0.50),
        "viability_shuffled": b_ranker(0.50),
        "listwise": b_ranker(0.50),
        "surface": b_ranker(0.55),
        "random": b_ranker(0.45),
    }


class GateBoundaryTests(unittest.TestCase):
    def test_top4_failure_never_vetoes_primary_all24_authorization(self) -> None:
        pass_a = "MATERIALIZED_SUFFIX_INTERFACE_PASS"
        fail_a = "NO_ACTIONABLE_MATERIALIZED_RESIDUAL"
        pass_b = "CHEAP_SIBLING_RANKING_PASS"
        fail_b = "CHEAP_SIBLING_RANKING_FAIL"
        self.assertEqual(
            core.mechanics_authorization(pass_a, pass_b),
            {
                "qualification_authorized": True,
                "top4_secondary_authorized": True,
            },
        )
        self.assertEqual(
            core.mechanics_authorization(pass_a, fail_b),
            {
                "qualification_authorized": True,
                "top4_secondary_authorized": False,
            },
        )
        for b_decision in (pass_b, fail_b):
            self.assertEqual(
                core.mechanics_authorization(fail_a, b_decision),
                {
                    "qualification_authorized": False,
                    "top4_secondary_authorized": False,
                },
            )

    def test_mechanics_a_exact_registered_boundaries_pass(self) -> None:
        result = core.decide_mechanics_a(a_metrics_fixture(), load_config()["mechanics"])
        self.assertEqual(result["decision"], "MATERIALIZED_SUFFIX_INTERFACE_PASS")
        self.assertEqual(
            result["integer_boundaries"],
            {
                "suffix_parse_successes_min": 47,
                "suffix_answer_cap_contacts_max": 2,
                "direct_parse_successes_min": 22,
                "direct_answer_cap_contacts_max": 1,
                "echo_visible_successes_min": 47,
                "materialized_task_successes_min": 9,
                "task_gain_over_name_min": 4,
                "task_gain_over_shuffled_min": 4,
            },
        )

    def test_mechanics_a_interface_failures_have_precedence(self) -> None:
        mutations = []
        changed = a_metrics_fixture()
        changed["suffix_materialized"]["parse_successes"] = 46
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_name_only"]["answer_cap_contacts"] = 3
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["direct"]["parse_successes"] = 21
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["direct"]["answer_cap_contacts"] = 2
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_echo"]["raw_visible_successes"] = 46
        mutations.append(changed)
        for metrics in mutations:
            with self.subTest(metrics=metrics):
                result = core.decide_mechanics_a(metrics, load_config()["mechanics"])
                self.assertEqual(result["decision"], "MECHANICS_INTERFACE_INVALID")

    def test_mechanics_a_each_scientific_boundary_fails_one_below(self) -> None:
        mutations = []
        changed = a_metrics_fixture()
        changed["suffix_materialized"]["task_any_live_successes"] = 8
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_name_only"]["task_any_live_successes"] = 6
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_shuffled"]["task_any_live_successes"] = 6
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_materialized"]["successful_parameterized_rows"] = 0
        mutations.append(changed)
        changed = a_metrics_fixture()
        changed["suffix_materialized"]["successful_parameter_free_rows"] = 0
        mutations.append(changed)
        for metrics in mutations:
            with self.subTest(metrics=metrics):
                result = core.decide_mechanics_a(metrics, load_config()["mechanics"])
                self.assertEqual(
                    result["decision"], "NO_ACTIONABLE_MATERIALIZED_RESIDUAL"
                )

    def test_mechanics_b_exact_float_boundaries_pass(self) -> None:
        result = core.decide_mechanics_b(
            b_metrics_fixture(),
            load_config()["mechanics"],
            registered_top4_static_context_fit=True,
        )
        self.assertEqual(result["decision"], "CHEAP_SIBLING_RANKING_PASS")
        self.assertEqual(result["authenticated_model_score_rows"], 2304)
        self.assertEqual(result["authenticated_requested_raw_logprob_values"], 4032)

    def test_mechanics_b_each_gate_fails_below_boundary(self) -> None:
        mutations: list[tuple[dict[str, dict], bool]] = []
        changed = b_metrics_fixture()
        changed["viability_materialized"]["mean_recall_at_4"] = 0.399999
        mutations.append((changed, True))
        changed = b_metrics_fixture()
        changed["viability_materialized"]["mean_hit_at_4"] = 0.649999
        mutations.append((changed, True))
        for comparator, allowed in (
            ("viability_name_only", 0.500001),
            ("viability_shuffled", 0.500001),
            ("listwise", 0.500001),
            ("surface", 0.550001),
            ("random", 0.450001),
        ):
            changed = b_metrics_fixture()
            changed[comparator]["mean_recall_at_4"] = allowed
            mutations.append((changed, True))
        changed = b_metrics_fixture()
        changed["viability_materialized"]["hit_tasks"] = 15
        mutations.append((changed, True))
        changed = b_metrics_fixture()
        changed["viability_materialized"]["retrieved_live_operation_count"] = 9
        mutations.append((changed, True))
        mutations.append((b_metrics_fixture(), False))
        for metrics, context_fit in mutations:
            with self.subTest(metrics=metrics, context_fit=context_fit):
                result = core.decide_mechanics_b(
                    metrics,
                    load_config()["mechanics"],
                    registered_top4_static_context_fit=context_fit,
                )
                self.assertEqual(result["decision"], "CHEAP_SIBLING_RANKING_FAIL")


class GenerationScoringTests(unittest.TestCase):
    def test_budget_branch_is_derived_from_stage_one_evidence(self) -> None:
        natural = runner._budget_branch_contract([10, 11, 248069, 12], "stop")
        self.assertTrue(natural["natural"])
        self.assertFalse(natural["forced_close"])
        self.assertEqual(natural["retained_thinking_token_ids"], [10, 11])
        unfinished_close = runner._budget_branch_contract(
            [10, 11, 248069, 12], "length"
        )
        self.assertFalse(unfinished_close["natural"])
        self.assertFalse(unfinished_close["forced_close"])
        self.assertEqual(unfinished_close["retained_thinking_token_ids"], [10, 11])
        forced = runner._budget_branch_contract([10, 11], "length")
        self.assertFalse(forced["natural"])
        self.assertTrue(forced["forced_close"])
        self.assertEqual(forced["retained_thinking_token_ids"], [10, 11])

    def test_public_echo_witness_executes_and_echoes_exactly(self) -> None:
        public = runner.read_jsonl(runner.PUBLIC_PATH)
        audit = runner.read_jsonl(runner.AUDIT_PATH)
        public_by_id = {row["task_id"]: row for row in public}
        witness = audit[0]["public_live"][0]
        candidate = operation_from_record(witness["operation"])
        suffix = tuple(
            operation_from_record(row) for row in witness["first_fitting_suffix"]
        )
        text = "PROGRAM: " + " | ".join(core.operation_alias(row) for row in suffix)
        raw = {
            "id": core.record_id("test", audit[0]["task_id"], candidate),
            "meta": {
                "task_id": audit[0]["task_id"],
                "condition": "suffix_echo",
                "candidate": runner.operation_record(candidate),
                "supplied_suffix": [runner.operation_record(row) for row in suffix],
            },
            "outputs": [
                {
                    "text": text,
                    "n_answer_tokens": 8,
                    "finish_reason": "stop",
                    "stage2_finish_reason": "stop",
                    "seed_stage1": 1,
                    "seed_stage2": 2,
                    "n_sampled_tokens": 8,
                    "n_stage1_prompt_tokens": 100,
                    "n_stage2_prompt_tokens": 120,
                }
            ],
        }
        scored, metrics = core.score_generation_arm(
            public_by_id, [raw], answer_cap=64, direct=False
        )
        self.assertTrue(scored[0]["raw_visible_pass"])
        self.assertTrue(scored[0]["exact_echo"])
        self.assertEqual(metrics["raw_visible_successes"], 1)

    def test_answer_cap_contact_is_conservative(self) -> None:
        self.assertTrue(core.answer_cap_contact({"n_answer_tokens": 64}, 64))
        self.assertFalse(
            core.answer_cap_contact(
                {"n_answer_tokens": 63, "finish_reason": "stop"}, 64
            )
        )
        self.assertTrue(
            core.answer_cap_contact(
                {"n_answer_tokens": 1, "stage2_finish_reason": "length"}, 64
            )
        )


class TransactionTests(unittest.TestCase):
    def test_all_sixteen_transaction_existence_states_fail_or_classify_exactly(self) -> None:
        names = ("started", "bundle", "generated", "complete")
        accepted = {
            (False, False, False, False): "PENDING",
            (True, True, False, False): "DURABLE_BUNDLE",
            (True, True, True, False): "GENERATED",
            (True, True, True, True): "COMPLETE",
        }
        for mask in range(16):
            signature = tuple(bool(mask & (1 << index)) for index in range(4))
            exists = dict(zip(names, signature, strict=True))
            with self.subTest(signature=signature):
                if signature in accepted:
                    self.assertEqual(
                        runner._classify_transaction(exists), accepted[signature]
                    )
                else:
                    with self.assertRaisesRegex(RuntimeError, "terminal"):
                        runner._classify_transaction(exists)

    def test_committed_lock_byte_check_ignores_git_status_assumptions(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            path = Path(value) / "lock.json"
            path.write_bytes(b"locked\n")
            with mock.patch.object(runner, "_git_bytes", return_value=b"locked\n"):
                self.assertEqual(
                    runner._verify_committed_file_bytes(path, "lock.json", "HEAD"),
                    b"locked\n",
                )
            with mock.patch.object(runner, "_git_bytes", return_value=b"different\n"):
                with self.assertRaisesRegex(RuntimeError, "working bytes differ"):
                    runner._verify_committed_file_bytes(path, "lock.json", "HEAD")

    def test_durable_writer_orders_directory_file_directory_fsync(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            path = Path(value) / "receipt.json"
            trace: list[str] = []
            real_fsync = runner.os.fsync

            def file_fsync(descriptor: int) -> None:
                trace.append("file_fsync")
                real_fsync(descriptor)

            with mock.patch.object(
                runner, "_fsync_directory", side_effect=lambda _path: trace.append("dir_fsync")
            ), mock.patch.object(runner.os, "fsync", side_effect=file_fsync):
                runner.write_exclusive_durable(path, {"status": "DURABLE"})
            self.assertEqual(trace, ["dir_fsync", "file_fsync", "dir_fsync"])
            self.assertEqual(runner.read_canonical_json(path), {"status": "DURABLE"})
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                runner.write_exclusive_durable(path, {"status": "SECOND"})

    def test_full_write_interrupted_before_file_fsync_can_be_redurable(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            path = Path(value) / "bundle.generated.json"
            with mock.patch.object(runner, "_fsync_directory"), mock.patch.object(
                runner.os, "fsync", side_effect=OSError("synthetic fsync crash")
            ):
                with self.assertRaisesRegex(OSError, "fsync crash"):
                    runner.write_exclusive_durable(path, {"status": "FULL_BYTES"})
            self.assertEqual(
                runner.read_canonical_json(path), {"status": "FULL_BYTES"}
            )
            runner._redurable_existing_file(path)

    def test_failure_at_each_publish_boundary_never_skips_a_state(self) -> None:
        suffixes = {
            "started": ".started.json",
            "bundle": ".generated.json",
            "generated": ".generated.receipt.json",
            "complete": ".complete.json",
        }
        expected = {
            "started": (False, False, False, False),
            "bundle": (True, False, False, False),
            "generated": (True, True, False, False),
            "complete": (True, True, True, False),
        }
        for boundary, suffix in suffixes.items():
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as value:
                root = Path(value)
                raw = root / "raw"
                prepared = root / "prepared"
                raw.mkdir()
                prepared.mkdir()
                write_canonical(raw / "live_preflight.json", {})
                (prepared / "suffix_materialized_requests.jsonl").write_text("")
                lock = root / "lock.json"
                lock.write_text("{}\n")
                real_write = runner.write_exclusive_durable

                def fail_at(path: Path, payload: object) -> None:
                    if path.name.endswith(suffix):
                        raise OSError(f"synthetic {boundary} crash")
                    real_write(path, payload)

                fake = SimpleNamespace(
                    generate=lambda _rows, _sampling: ([{"id": "row"}], {"m": 1})
                )
                with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                    runner, "PREPARED", prepared
                ), mock.patch.object(
                    runner, "write_exclusive_durable", side_effect=fail_at
                ), mock.patch.object(
                    runner, "_authenticate_invocation"
                ), mock.patch.object(
                    runner, "_completion_receipt", return_value={"status": "COMPLETE"}
                ):
                    with self.assertRaisesRegex(OSError, f"{boundary} crash"):
                        runner._generate(
                            "suffix_materialized",
                            fake,
                            [],
                            load_config(),
                            {"tokenizer": {"plain_alias_token_ids": {}}},
                            {},
                            lock,
                        )
                    paths = runner._artifact_paths("suffix_materialized")
                    observed = tuple(
                        paths[name].exists()
                        for name in ("started", "bundle", "generated", "complete")
                    )
                    self.assertEqual(observed, expected[boundary])

    def test_run_lock_rejects_concurrent_mutating_analysis(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            with mock.patch.object(runner, "RAW", raw), runner._run_lock():
                with self.assertRaisesRegex(RuntimeError, "another mechanics process"):
                    with runner._run_lock():
                        self.fail("nested mechanics lock unexpectedly succeeded")

    def test_run_live_recovers_all_generated_invocations_without_constructing_runner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            raw.mkdir()
            lock = root / "lock.json"
            lock.write_text("{}\n")
            for name in runner.INVOCATIONS:
                paths = {
                    key: raw / path.name
                    for key, path in runner._artifact_paths(name).items()
                }
                write_canonical(paths["started"], {"invocation": name, "status": "STARTED"})
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": name,
                        "raw_rows": [],
                        "metadata": {},
                    },
                )
                write_canonical(
                    paths["generated"], {"invocation": name, "status": "GENERATED"}
                )
            prepared = {name: [] for name in runner.INVOCATIONS}
            prepare_receipt = {"tokenizer": {"plain_alias_token_ids": {}}}
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "verify_implementation_lock"
            ), mock.patch.object(
                runner, "_validate_current_scientific_environment"
            ), mock.patch.object(
                runner, "_load_prepared", return_value=(prepare_receipt, prepared)
            ), mock.patch.object(
                runner, "_load_recorded_live_preflight", return_value={}
            ), mock.patch.object(
                runner,
                "_started_receipt",
                side_effect=lambda name, _lock: {"invocation": name, "status": "STARTED"},
            ), mock.patch.object(
                runner,
                "_generated_receipt",
                side_effect=lambda name, _lock: {"invocation": name, "status": "GENERATED"},
            ), mock.patch.object(
                runner,
                "_completion_receipt",
                side_effect=lambda name, _lock: {"invocation": name, "status": "COMPLETE"},
            ), mock.patch.object(
                runner, "_authenticate_invocation"
            ), mock.patch.object(
                runner, "_analyze_locked", return_value={"decision": "OFFLINE_RECOVERY"}
            ), mock.patch.object(
                runner, "VLLMRunner", side_effect=AssertionError("runner constructed")
            ):
                result = runner.run_live(lock)
            self.assertEqual(result, {"decision": "OFFLINE_RECOVERY"})
            for name in runner.INVOCATIONS:
                self.assertTrue((raw / f"{name}.complete.json").is_file())

    def test_existing_authentication_receipt_with_missing_invocation_blocks_runner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            raw.mkdir()
            write_canonical(raw / "authentication_receipt.json", {})
            lock = root / "lock.json"
            lock.write_text("{}\n")
            prepared = {name: [] for name in runner.INVOCATIONS}
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "verify_implementation_lock"
            ), mock.patch.object(
                runner, "_validate_current_scientific_environment"
            ), mock.patch.object(
                runner, "_load_prepared", return_value=({}, prepared)
            ), mock.patch.object(
                runner, "VLLMRunner", side_effect=AssertionError("runner constructed")
            ):
                with self.assertRaisesRegex(RuntimeError, "incomplete invocation"):
                    runner.run_live(lock)

    def test_started_only_is_ambiguous_and_never_resampled(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            with mock.patch.object(runner, "RAW", raw):
                path = runner._artifact_paths("direct")["started"]
                path.parent.mkdir(parents=True)
                path.write_text("{}\n")
                with self.assertRaisesRegex(RuntimeError, "ambiguous STARTED"):
                    runner._load_completed(
                        "direct", [], {}, {}, None, Path(value) / "lock.json"
                    )

    def test_durable_generated_bundle_finalizes_without_resampling(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            lock = Path(value) / "lock.json"
            lock.write_text("{}\n")
            with mock.patch.object(runner, "RAW", raw):
                paths = runner._artifact_paths("direct")
                paths["started"].parent.mkdir(parents=True)
                write_canonical(paths["started"], {"status": "STARTED"})
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": "direct",
                        "raw_rows": [{"id": "row"}],
                        "metadata": {"model": "authenticated"},
                    },
                )
                write_canonical(paths["generated"], {"status": "GENERATED"})
                completion = {"status": "COMPLETE"}
                with mock.patch.object(
                    runner,
                    "_started_receipt",
                    return_value={"status": "STARTED"},
                ), mock.patch.object(
                    runner,
                    "_generated_receipt",
                    return_value={"status": "GENERATED"},
                ), mock.patch.object(
                    runner, "_completion_receipt", return_value=completion
                ), mock.patch.object(
                    runner, "_authenticate_invocation"
                ) as authenticate:
                    loaded = runner._load_completed(
                        "direct", [], {}, {}, {}, lock
                    )
                self.assertEqual(loaded, ([{"id": "row"}], {"model": "authenticated"}))
                self.assertEqual(json.loads(paths["complete"].read_text()), completion)
                authenticate.assert_called_once()

    def test_authentication_failure_preserves_generated_bytes_and_restart_uses_no_runner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            prepared = root / "prepared"
            prepared.mkdir()
            (prepared / "suffix_materialized_requests.jsonl").write_text("")
            raw.mkdir()
            (raw / "live_preflight.json").write_text("{}\n")
            lock = root / "lock.json"
            lock.write_text("{}\n")
            fake_runner = SimpleNamespace(
                generate=lambda _rows, _sampling: ([{"id": "durable"}], {"m": 1})
            )
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "PREPARED", prepared
            ), mock.patch.object(
                runner,
                "_authenticate_invocation",
                side_effect=RuntimeError("synthetic authentication failure"),
            ):
                with self.assertRaisesRegex(RuntimeError, "authentication failure"):
                    runner._generate(
                        "suffix_materialized",
                        fake_runner,
                        [],
                        load_config(),
                        {"tokenizer": {"plain_alias_token_ids": {}}},
                        {},
                        lock,
                    )
            paths = {
                key: raw / path.name
                for key, path in runner._artifact_paths("suffix_materialized").items()
            }
            self.assertTrue(paths["started"].is_file())
            self.assertTrue(paths["bundle"].is_file())
            self.assertTrue(paths["generated"].is_file())
            self.assertFalse(paths["complete"].exists())
            self.assertEqual(
                runner.read_json(paths["bundle"])["raw_rows"], [{"id": "durable"}]
            )
            completion = {"status": "COMPLETE"}
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "PREPARED", prepared
            ), mock.patch.object(
                runner, "_authenticate_invocation"
            ) as authenticate, mock.patch.object(
                runner, "_completion_receipt", return_value=completion
            ):
                loaded = runner._load_completed(
                    "suffix_materialized", [], load_config(), {}, {}, lock
                )
            self.assertEqual(loaded, ([{"id": "durable"}], {"m": 1}))
            authenticate.assert_called_once()
            self.assertEqual(runner.read_json(paths["complete"]), completion)

    def test_malformed_bundle_without_generated_receipt_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            with mock.patch.object(runner, "RAW", raw):
                paths = runner._artifact_paths("direct")
                raw.mkdir()
                paths["started"].write_text("{}\n")
                paths["bundle"].write_text("{}\n")
                with mock.patch.object(
                    runner, "_started_receipt", return_value={}
                ), self.assertRaisesRegex(RuntimeError, "bundle schema"):
                    runner._load_completed(
                        "direct", [], {}, {}, {}, Path(value) / "lock.json"
                    )

    def test_noncanonical_or_duplicate_key_bundle_is_never_promoted(self) -> None:
        payloads = (
            b'{"schema_version":1,"invocation":"direct","raw_rows":[],"metadata":{}}\n',
            b'{"schema_version":1,"schema_version":1,"invocation":"direct","raw_rows":[],"metadata":{}}\n',
        )
        for payload in payloads:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory(
                dir=ROOT
            ) as value:
                raw = Path(value) / "raw"
                raw.mkdir()
                with mock.patch.object(runner, "RAW", raw):
                    paths = runner._artifact_paths("direct")
                    write_canonical(paths["started"], {})
                    paths["bundle"].write_bytes(payload)
                    with mock.patch.object(
                        runner, "_started_receipt", return_value={}
                    ), self.assertRaisesRegex(
                        RuntimeError, "noncanonical|duplicate key"
                    ):
                        runner._load_completed(
                            "direct", [], {}, {}, {}, Path(value) / "lock.json"
                        )
                    self.assertFalse(paths["generated"].exists())
                    self.assertFalse(paths["complete"].exists())

    def test_intact_bundle_reconstructs_generated_receipt_without_runner(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            prepared = root / "prepared"
            raw.mkdir()
            prepared.mkdir()
            lock = root / "lock.json"
            lock.write_text("{}\n")
            (raw / "live_preflight.json").write_text("{}\n")
            (prepared / "suffix_materialized_requests.jsonl").write_text("")
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "PREPARED", prepared
            ):
                paths = runner._artifact_paths("suffix_materialized")
                write_canonical(
                    paths["started"],
                    runner._started_receipt("suffix_materialized", lock),
                )
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": "suffix_materialized",
                        "raw_rows": [{"id": "durable"}],
                        "metadata": {"m": 1},
                    },
                )
                completion = {"status": "COMPLETE"}
                with mock.patch.object(
                    runner, "_authenticate_invocation"
                ) as authenticate, mock.patch.object(
                    runner, "_completion_receipt", return_value=completion
                ):
                    loaded = runner._load_completed(
                        "suffix_materialized", [], load_config(), {}, {}, lock
                    )
                self.assertEqual(loaded, ([{"id": "durable"}], {"m": 1}))
                self.assertTrue(paths["generated"].is_file())
                self.assertEqual(runner.read_json(paths["generated"])["status"], "GENERATED")
                self.assertEqual(runner.read_json(paths["complete"]), completion)
                authenticate.assert_called_once()

    def test_generated_receipt_hash_mismatch_is_terminal(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            prepared = root / "prepared"
            raw.mkdir()
            prepared.mkdir()
            lock = root / "lock.json"
            lock.write_text("{}\n")
            (raw / "live_preflight.json").write_text("{}\n")
            (prepared / "direct_requests.jsonl").write_text("")
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "PREPARED", prepared
            ):
                paths = runner._artifact_paths("direct")
                started = runner._started_receipt("direct", lock)
                write_canonical(paths["started"], started)
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": "direct",
                        "raw_rows": [],
                        "metadata": {},
                    },
                )
                write_canonical(paths["generated"], {"bundle_sha256": "wrong"})
                with self.assertRaisesRegex(RuntimeError, "generated receipt changed"):
                    runner._load_completed(
                        "direct", [], {}, {}, {}, lock
                    )

    def test_corrupt_complete_receipt_fails_after_generated_authentication(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            raw.mkdir()
            lock = Path(value) / "lock.json"
            lock.write_text("{}\n")
            with mock.patch.object(runner, "RAW", raw):
                paths = runner._artifact_paths("suffix_materialized")
                write_canonical(paths["started"], {"status": "STARTED"})
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": "suffix_materialized",
                        "raw_rows": [],
                        "metadata": {},
                    },
                )
                write_canonical(paths["generated"], {"status": "GENERATED"})
                write_canonical(paths["complete"], {"status": "CORRUPT"})
                with mock.patch.object(
                    runner, "_started_receipt", return_value={"status": "STARTED"}
                ), mock.patch.object(
                    runner, "_generated_receipt", return_value={"status": "GENERATED"}
                ), mock.patch.object(
                    runner, "_completion_receipt", return_value={"status": "COMPLETE"}
                ), mock.patch.object(
                    runner, "_authenticate_invocation"
                ) as authenticate, self.assertRaisesRegex(
                    RuntimeError, "completion receipt changed"
                ):
                    runner._load_completed(
                        "suffix_materialized", [], {}, {}, {}, lock
                    )
                authenticate.assert_called_once()

    def test_completion_receipt_hash_chains_to_previous_invocation(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            root = Path(value)
            raw = root / "raw"
            prepared = root / "prepared"
            raw.mkdir()
            prepared.mkdir()
            lock = root / "lock.json"
            lock.write_text("{}\n")
            write_canonical(raw / "live_preflight.json", {})
            (prepared / "suffix_name_only_requests.jsonl").write_text("")
            with mock.patch.object(runner, "RAW", raw), mock.patch.object(
                runner, "PREPARED", prepared
            ):
                previous = runner._artifact_paths("suffix_materialized")["complete"]
                write_canonical(previous, {"version": 1})
                paths = runner._artifact_paths("suffix_name_only")
                write_canonical(paths["started"], {})
                write_canonical(
                    paths["bundle"],
                    {
                        "schema_version": 1,
                        "invocation": "suffix_name_only",
                        "raw_rows": [],
                        "metadata": {},
                    },
                )
                write_canonical(paths["generated"], {})
                first = runner._completion_receipt("suffix_name_only", lock)
                self.assertEqual(
                    first["previous_chain_sha256"], runner.sha256_file(previous)
                )
                write_canonical(previous, {"version": 2})
                second = runner._completion_receipt("suffix_name_only", lock)
                self.assertNotEqual(
                    first["previous_chain_sha256"], second["previous_chain_sha256"]
                )

    def test_unknown_or_symlink_raw_inventory_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as value:
            raw = Path(value) / "raw"
            raw.mkdir()
            (raw / "unknown.json").write_text("{}\n")
            with mock.patch.object(runner, "RAW", raw), self.assertRaises(RuntimeError):
                runner._validate_raw_inventory()
            (raw / "unknown.json").unlink()
            (raw / "live_preflight.json").symlink_to(Path(value) / "missing")
            with mock.patch.object(runner, "RAW", raw), self.assertRaises(RuntimeError):
                runner._validate_raw_inventory()


if __name__ == "__main__":
    unittest.main()
