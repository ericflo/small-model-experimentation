"""Model-free tests for the experiment orchestrator."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from unittest import mock
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verified_macro_run", EXP / "scripts" / "run.py")
assert SPEC is not None and SPEC.loader is not None
run = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run
SPEC.loader.exec_module(run)

import model_harness as harness  # noqa: E402
import macro_domain as domain  # noqa: E402
import vllm_runner as local_vllm  # noqa: E402


class ProposalConstructionTests(unittest.TestCase):
    def test_supported_proposals_are_ranked_and_behaviorally_deduplicated(self) -> None:
        a = ("ADD1", "MUL2")
        alias_a = ("MUL2", "ADD1")
        b = ("NEG", "REV")
        programs = [
            {"program": [*a, *b, *alias_a]},
            {"program": [*a, *b, *alias_a]},
            {"program": [*a, *b]},
        ]

        def proposal(name: str, expansion: tuple[str, ...]) -> types.SimpleNamespace:
            return types.SimpleNamespace(name=name, expansion=expansion)

        parsed = [
            types.SimpleNamespace(
                record_id="proposal",
                sample_index=0,
                parse_error=None,
                proposals=(proposal("A", a), proposal("ALIAS_A", alias_a), proposal("B", b)),
            ),
            types.SimpleNamespace(
                record_id="proposal",
                sample_index=1,
                parse_error=None,
                proposals=(proposal("A", a), proposal("B", b)),
            ),
        ]

        def verify(expansion: tuple[str, ...]) -> types.SimpleNamespace:
            signature = "same-a" if expansion in {a, alias_a} else "b"
            return types.SimpleNamespace(
                valid=True,
                exact=True,
                nondegenerate=True,
                signature=signature,
            )

        ranked, names, audit = run._proposal_candidates(
            parsed,
            programs,
            min_support=2,
            verify_expansion=verify,
        )
        self.assertEqual(ranked, [a, b])
        self.assertEqual(names[a], "A")
        self.assertEqual(audit["unique_supported_exact_expansions"], 3)
        self.assertEqual(audit["unique_supported_candidates"], 2)
        self.assertEqual(len(audit["dropped_behavioral_aliases"]), 1)


class ArtifactTests(unittest.TestCase):
    def test_freeze_accepts_identical_content_and_rejects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            run._freeze_json(path, {"a": [1, 2]})
            original = path.read_bytes()
            run._freeze_json(path, {"a": [1, 2]})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(run.FrozenArtifactError):
                run._freeze_json(path, {"a": [1, 3]})

    def test_config_keeps_vllm_and_matched_k_contract(self) -> None:
        config = run.load_config()
        run._validate_config(config)
        self.assertEqual(config["inference"]["backend"], "vllm")
        self.assertEqual(config["inference"]["base_max_k"], 24)
        self.assertEqual(config["inference"]["macro_k"], 12)
        self.assertEqual(config["inference"]["smoke_thinking_budget"], 768)
        self.assertEqual(config["inference"]["interface_thinking"], "off")

    def test_interface_probe_is_train_only_and_uses_no_think_sampling(self) -> None:
        config = run.load_config()
        libraries = run._read_json(EXP / "data" / "libraries.json")["libraries"]
        records = run._interface_gate_records(
            harness=harness,
            domain=domain,
            library=libraries["designed_ceiling"],
        )
        self.assertEqual(len(records), 4)
        self.assertTrue(all(record["id"].startswith("interface-v3-") for record in records))
        self.assertTrue(all(record["meta"]["eval_data_used"] is False for record in records))
        sampling = run._interface_sampling(harness, config, n=4)
        self.assertEqual(sampling.thinking, "off")
        self.assertIsNone(sampling.thinking_budget)
        self.assertEqual(sampling.answer_max_tokens, 128)

    def test_cached_runner_artifact_is_bound_to_prompt_sampling_and_runner(self) -> None:
        config = run.load_config()
        sampling = run._solver_sampling(harness, config, run="smoke", n=1)
        record = {"id": "task::base", "messages": [{"role": "user", "content": "x"}], "meta": {"arm": "base"}}
        rendered_hash = "a" * 64
        row = {
            "id": record["id"],
            "meta": record["meta"],
            "prompt_sha256": rendered_hash,
            "n_prompt_tokens": 10,
            "outputs": [
                {
                    "sample_index": 0,
                    "text": "PROGRAM: ADD1",
                    "token_ids": [1],
                    "n_stage1_prompt_tokens": 10,
                    "n_stage2_prompt_tokens": 0,
                    "n_sampled_tokens": 1,
                    "n_injected_tokens": 0,
                    "n_completion_tokens": 1,
                    "n_thinking_tokens": 0,
                    "n_answer_tokens": 1,
                    "n_terminal_tokens_trimmed": 0,
                }
            ],
        }
        accounting = harness.extract_token_accounting([row])
        summary = {
            "schema_version": local_vllm.RUNNER_SCHEMA_VERSION,
            "model": harness.REQUIRED_MODEL_ID,
            "model_revision": harness.MODEL_REVISION,
            "runner_sha256": run._sha256_file(EXP / "src" / "vllm_runner.py"),
            "engine": run._expected_engine_metadata(harness, config),
            "sampling": __import__("json").loads(
                run._canonical_bytes(__import__("dataclasses").asdict(sampling)).decode()
            ),
            "resolved_sampling": sampling.resolved_sampling(),
            "adapter": None,
            "counts": {
                "requests": accounting.requests,
                "completions": accounting.completions,
                "unique_input_prompt_tokens": accounting.unique_input_prompt_tokens,
                "stage1_logical_prompt_tokens": accounting.stage1_logical_prompt_tokens,
                "stage2_logical_prompt_tokens": accounting.stage2_logical_prompt_tokens,
                "logical_model_input_tokens": accounting.logical_model_input_tokens,
                "sampled_tokens": accounting.sampled_tokens,
                "injected_tokens": accounting.injected_tokens,
            },
        }
        preflight = {
            "schema_version": run.SCHEMA_VERSION,
            "pass": True,
            "n_records": 1,
            "records": [
                {
                    "id": record["id"],
                    "input_record_sha256": run._sha256_value(record),
                    "rendered_prompt_sha256": rendered_hash,
                    "prompt_tokens": 10,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "base.jsonl"
            run._atomic_jsonl(path, [row])
            run._atomic_json(path.with_suffix(".meta.json"), summary)
            preflight_path = path.with_suffix(".preflight.json")
            run._atomic_json(preflight_path, preflight)
            self.assertTrue(
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=[record],
                    sampling=sampling,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )
            )
            drifted = run._solver_sampling(harness, config, run="smoke", n=1)
            object.__setattr__(drifted, "run_seed", drifted.run_seed + 1)
            with self.assertRaisesRegex(ValueError, "sampling/seed mismatch"):
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=[record],
                    sampling=drifted,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )

    def test_smoke_gate_is_always_recomputed(self) -> None:
        with mock.patch.object(
            run,
            "_invoke_analyzer",
            return_value={"run": "smoke", "smoke_gate": {"pass": True}},
        ) as invoke:
            run._require_smoke_gate()
        invoke.assert_called_once_with("smoke")

    def test_terminal_interface_failure_forbids_smoke_and_full_reruns(self) -> None:
        config = run.load_config()
        for stage in ("smoke", "full"):
            with self.subTest(stage=stage), self.assertRaisesRegex(
                RuntimeError, "experiment stopped after the preregistered interface-v3 gate"
            ):
                run.run_model_stage(stage, config)


if __name__ == "__main__":
    unittest.main()
