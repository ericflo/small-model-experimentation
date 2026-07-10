"""Model-free tests for the experiment orchestrator."""

from __future__ import annotations

import copy
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


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

    def test_config_keeps_vllm_long_context_and_matched_compute_contract(self) -> None:
        config = run.load_config()
        run._validate_config(config)
        self.assertEqual(config["inference"]["backend"], "vllm")
        self.assertEqual(config["inference"]["thinking"], "budget")
        self.assertEqual(
            run._budget_ladder(config),
            [16384, 32768, 49152, 61440],
        )
        self.assertEqual(config["inference"]["answer_max_tokens"], 512)
        self.assertEqual(config["inference"]["max_model_len"], 65536)
        self.assertEqual(config["inference"]["max_num_seqs"], 64)
        self.assertEqual(config["inference"]["max_num_batched_tokens"], 32768)
        self.assertEqual(config["inference"]["base_max_k"], 24)
        self.assertEqual(config["inference"]["macro_k"], 12)
        self.assertEqual(config["inference"]["scientific_probe_k"], 4)
        self.assertEqual(config["inference"]["calibration_k"], 16)
        self.assertEqual(config["inference"]["interface_k"], 4)
        self.assertEqual(config["context_envelope"]["proposal_max_prompt_tokens"], 3478)
        self.assertEqual(config["context_envelope"]["forced_close_tokens"], 2)
        self.assertEqual(config["context_envelope"]["minimum_headroom_tokens"], 104)
        self.assertEqual(config["decision"]["budget_p99_thinking_fraction"], 0.80)
        self.assertEqual(config["decision"]["loop_override_min_budget"], 16384)

    def test_cpu_context_envelope_freezes_proposal_and_104_token_guard(self) -> None:
        config = run.load_config()
        _, _, proposal_view, _ = run._load_prepared()
        record = run._macro_proposal_record(
            harness=harness,
            domain=domain,
            config=config,
            proposal_view=proposal_view,
        )
        audit = run._context_envelope_regression(
            config=config,
            proposal_record=record,
            observed_prompt_tokens=3478,
        )
        self.assertEqual(audit["proposal_max_prompt_tokens"], 3478)
        self.assertEqual(audit["observed_prompt_tokens"], 3478)
        self.assertTrue(audit["prompt_token_count_verified"])
        self.assertEqual(audit["generation_reserve_tokens"], 61954)
        self.assertEqual(audit["max_prompt_plus_reserve_tokens"], 65432)
        self.assertEqual(audit["headroom_tokens"], 104)
        self.assertTrue(audit["pass"])

        drifted_record = copy.deepcopy(record)
        drifted_record["messages"][1]["content"] += "\nDRIFT"
        with self.assertRaisesRegex(ValueError, "proposal record drifted"):
            run._context_envelope_regression(
                config=config,
                proposal_record=drifted_record,
            )

        insufficient = copy.deepcopy(config)
        insufficient["context_envelope"]["minimum_headroom_tokens"] = 105
        with self.assertRaisesRegex(ValueError, "headroom fell below"):
            run._context_envelope_regression(
                config=insufficient,
                proposal_record=record,
                observed_prompt_tokens=3478,
            )

        with self.assertRaisesRegex(ValueError, "prompt drifted"):
            run._context_envelope_regression(
                config=config,
                proposal_record=record,
                observed_prompt_tokens=3479,
            )

    def test_config_rejects_amendment_critical_protocol_drift(self) -> None:
        config = run.load_config()
        cases = (
            ("inference", "max_num_batched_tokens", 16384, "batch-token ceiling"),
            ("decision", "smoke_matched_k", 11, "smoke matched K"),
            ("decision", "budget_max_cap_contact", 0.10, "cap-contact ceiling"),
            (
                "decision",
                "budget_max_answer_truncation",
                0.10,
                "answer-truncation ceiling",
            ),
        )
        for section, key, value, message in cases:
            with self.subTest(field=f"{section}.{key}"):
                drifted = copy.deepcopy(config)
                drifted[section][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    run._validate_config(drifted)

    def test_calibration_and_heldout_interface_sets_use_budget_sampling(self) -> None:
        config = run.load_config()
        libraries = run._read_json(EXP / "data" / "libraries.json")["libraries"]
        library = libraries["designed_ceiling"]
        calibration = run._calibration_records(domain=domain, library=library)
        heldout = run._interface_gate_records(
            harness=harness,
            domain=domain,
            library=library,
        )
        self.assertEqual(len(calibration), 4)
        self.assertEqual(len(heldout), 16)
        self.assertEqual(len(calibration) * config["inference"]["calibration_k"], 64)
        self.assertEqual(len(heldout) * config["inference"]["interface_k"], 64)
        self.assertTrue(all(record["id"].startswith("budget-calibration-") for record in calibration))
        self.assertTrue(
            all(record["id"].startswith("interface-long-context-heldout-") for record in heldout)
        )
        self.assertTrue(
            all(record["meta"]["eval_data_used"] is False for record in calibration + heldout)
        )
        self.assertTrue(
            set(record["id"] for record in calibration).isdisjoint(
                record["id"] for record in heldout
            )
        )

        calibration_sampling = run._interface_sampling(
            harness,
            config,
            budget=16384,
            n=config["inference"]["calibration_k"],
            calibration=True,
        )
        heldout_sampling = run._interface_sampling(
            harness,
            config,
            budget=16384,
            n=config["inference"]["interface_k"],
        )
        for sampling, expected_n in ((calibration_sampling, 16), (heldout_sampling, 4)):
            with self.subTest(expected_n=expected_n):
                self.assertEqual(sampling.thinking, "budget")
                self.assertEqual(sampling.thinking_budget, 16384)
                self.assertEqual(sampling.n, expected_n)
                self.assertEqual(sampling.max_tokens, 512)
                self.assertEqual(sampling.answer_max_tokens, 512)
        self.assertEqual(calibration_sampling.run_seed, config["seeds"]["vllm_calibration"])
        self.assertEqual(heldout_sampling.run_seed, config["seeds"]["vllm_solver"])

    def test_frozen_parent_artifacts_verify_exact_provenance(self) -> None:
        config = run.load_config()
        manifest = run._read_json(EXP / "data" / "dataset_manifest.json")
        self.assertEqual(manifest["config_sha256"], run.PARENT_CONFIG_SHA256)
        self.assertEqual(manifest["artifact_sha256"], run.PARENT_ARTIFACT_SHA256)
        run._verify_frozen_data(config)

        with mock.patch.object(run, "_sha256_file", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "frozen artifact hash mismatch"):
                run._verify_frozen_data(config)

    def test_termination_metrics_detect_cap_contact_and_require_headroom(self) -> None:
        config = run.load_config()

        def output(
            *, thinking: int, answer: int, forced: bool = False, truncated: bool = False
        ) -> dict[str, object]:
            return {
                "sample_index": 0,
                "n_thinking_tokens": thinking,
                "n_answer_tokens": answer,
                "forced_close": forced,
                "stage1_finish_reason": "length" if forced else "stop",
                "truncated": truncated,
            }

        adequate = run._termination_metrics(
            [{"id": "probe", "outputs": [output(thinking=512, answer=32)]}],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=True,
        )
        self.assertTrue(adequate["adequate"])
        self.assertEqual(adequate["cap_contacts"], 0)
        self.assertEqual(adequate["answer_truncations"], 0)
        self.assertTrue(adequate["thinking_headroom_pass"])
        self.assertTrue(adequate["answer_headroom_pass"])
        self.assertFalse(adequate["selection_uses_output_content"])

        censored = run._termination_metrics(
            [
                {
                    "id": "probe",
                    "outputs": [
                        output(thinking=16384, answer=512, forced=True, truncated=True)
                    ],
                }
            ],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=True,
        )
        self.assertFalse(censored["adequate"])
        self.assertEqual(censored["cap_contact_rate"], 1.0)
        self.assertEqual(censored["answer_truncation_rate"], 1.0)
        self.assertFalse(censored["thinking_headroom_pass"])
        self.assertFalse(censored["answer_headroom_pass"])

    def test_amendment9_reasoning_boundary_and_answer_restart_cases(self) -> None:
        config = run.load_config()
        budget = 16384

        def output(
            sample_index: int,
            *,
            thinking: int,
            answer: int = 32,
            forced: bool,
            stage1_finish: str,
        ) -> dict[str, object]:
            return {
                "sample_index": sample_index,
                "n_thinking_tokens": thinking,
                "n_answer_tokens": answer,
                "forced_close": forced,
                "stage1_finish_reason": stage1_finish,
                "finish_reason": "stop",
                "truncated": False,
            }

        rows = [
            {
                "id": "amendment9",
                "outputs": [
                    # (a) Literal forced close after using the full reasoning budget.
                    output(0, thinking=budget, forced=True, stage1_finish="length"),
                    # (b) Early EOS without </think>: forced intervention, not raw length.
                    output(1, thinking=10, forced=True, stage1_finish="stop"),
                    # (c) </think> occupied the final stage-one slot.
                    output(2, thinking=budget - 1, forced=False, stage1_finish="length"),
                    # (d) Earlier natural close; discarded partial answer was restarted.
                    output(3, thinking=100, forced=False, stage1_finish="length"),
                    # (e) Same restart, but the fresh semantic answer reaches its allowance.
                    output(
                        4,
                        thinking=100,
                        answer=512,
                        forced=False,
                        stage1_finish="length",
                    ),
                ],
            }
        ]
        metrics = run._termination_metrics(
            rows,
            budget=budget,
            answer_cap=512,
            config=config,
            require_headroom=False,
        )

        self.assertEqual(metrics["cap_contacts"], 3)
        self.assertEqual(metrics["forced_interventions"], 2)
        self.assertEqual(metrics["reasoning_boundary_contacts"], 2)
        self.assertEqual(metrics["stage1_length_finishes"], 4)
        self.assertEqual(metrics["answer_restarts_after_natural_close"], 2)
        self.assertEqual(metrics["answer_limit_contacts"], 1)
        contacts = {row["sample_index"] for row in metrics["cap_contact_samples"]}
        restarts = {
            row["sample_index"]
            for row in metrics["answer_restart_after_natural_close_samples"]
        }
        self.assertEqual(contacts, {0, 1, 2})
        self.assertEqual(restarts, {3, 4})
        self.assertEqual(
            {row["sample_index"] for row in metrics["answer_limit_contact_samples"]},
            {4},
        )

    def test_periodic_loop_override_distinguishes_repetition_from_unresolved_censoring(
        self,
    ) -> None:
        config = run.load_config()

        def forced_close(retained_ids: list[int]) -> dict[str, object]:
            return {
                "sample_index": 0,
                "n_thinking_tokens": 16384,
                "n_answer_tokens": 32,
                "forced_close": True,
                "stage1_finish_reason": "length",
                "truncated": False,
                "retained_thinking_token_ids": retained_ids,
            }

        repeated_tail = [101, 202, 303, 404] * 2048
        periodic = run._termination_metrics(
            [{"id": "periodic", "outputs": [forced_close(repeated_tail)]}],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=False,
        )
        self.assertEqual(periodic["cap_contacts"], 1)
        self.assertEqual(periodic["periodic_loop_contacts"], 1)
        self.assertEqual(periodic["unresolved_cap_contacts"], 0)
        self.assertEqual(periodic["periodic_loop_samples"][0]["period_tokens"], 4)
        self.assertEqual(periodic["periodic_loop_samples"][0]["tail_tokens"], 8192)

        nonperiodic_tail = list(range(8192))
        unresolved = run._termination_metrics(
            [{"id": "nonperiodic", "outputs": [forced_close(nonperiodic_tail)]}],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=False,
        )
        self.assertEqual(unresolved["cap_contacts"], 1)
        self.assertEqual(unresolved["periodic_loop_contacts"], 0)
        self.assertEqual(unresolved["unresolved_cap_contacts"], 1)

    def test_natural_stop_equal_to_answer_ceiling_is_limit_contact(self) -> None:
        config = run.load_config()

        def natural_stop(answer_tokens: int) -> dict[str, object]:
            return {
                "sample_index": 0,
                "n_thinking_tokens": 512,
                "n_answer_tokens": answer_tokens,
                "forced_close": False,
                "stage1_finish_reason": "stop",
                "truncated": False,
                "finish_reason": "stop",
            }

        at_limit = run._termination_metrics(
            [{"id": "at-limit", "outputs": [natural_stop(512)]}],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=False,
        )
        below_limit = run._termination_metrics(
            [{"id": "below-limit", "outputs": [natural_stop(511)]}],
            budget=16384,
            answer_cap=512,
            config=config,
            require_headroom=False,
        )

        self.assertFalse(at_limit["adequate"])
        self.assertEqual(at_limit["answer_truncations"], 1)
        self.assertEqual(at_limit["answer_limit_contacts"], 1)
        self.assertEqual(at_limit["stage2_truncations"], 0)
        self.assertTrue(below_limit["adequate"])
        self.assertEqual(below_limit["answer_truncations"], 0)
        self.assertEqual(below_limit["answer_limit_contacts"], 0)
        self.assertEqual(below_limit["stage2_truncations"], 0)

    def test_budget_escalation_audits_sampled_token_prefixes(self) -> None:
        lower = [
            {
                "id": "probe",
                "outputs": [{"sample_index": 0, "stage1_token_ids": [1, 2]}],
            }
        ]
        higher = [
            {
                "id": "probe",
                "outputs": [{"sample_index": 0, "stage1_token_ids": [1, 2, 3]}],
            }
        ]
        self.assertEqual(
            run._stage1_prefix_audit(lower, higher),
            {"samples": 1, "pass": True, "failures": []},
        )

        changed = [
            {
                "id": "probe",
                "outputs": [{"sample_index": 0, "stage1_token_ids": [1, 9, 3]}],
            }
        ]
        audit = run._stage1_prefix_audit(lower, changed)
        self.assertFalse(audit["pass"])
        self.assertEqual(audit["failures"], [{"id": "probe", "sample_index": 0}])

    def test_scientific_stage_reuses_cached_arm_and_skips_rejected_rung_remainder(
        self,
    ) -> None:
        config = run.load_config()
        validation_calls: dict[tuple[str, str], int] = {}
        generated: list[tuple[str, int]] = []

        def validate(path: Path, **_: object) -> bool:
            key = (path.parent.name, path.stem)
            validation_calls[key] = validation_calls.get(key, 0) + 1
            if key == ("think_16384", "base"):
                return True
            return validation_calls[key] > 1

        def rows(path: Path) -> list[dict[str, object]]:
            return [{"termination_fixture": (path.parent.name, path.stem)}]

        def termination(
            rows: list[dict[str, object]], **_: object
        ) -> dict[str, object]:
            return {
                "adequate": rows[0]["termination_fixture"]
                != ("think_16384", "base")
            }

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                records: list[dict[str, object]],
                sampling: types.SimpleNamespace,
            ) -> object:
                del runner
                generated.append((str(records[0]["id"]), sampling.thinking_budget))
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # This models the interrupted designed-arm preflight in the live
            # experiment.  The rejected rung must neither read nor rewrite it.
            interrupted = (
                root
                / "external"
                / "smoke_tiers"
                / "think_16384"
                / "designed_ceiling.preflight.json"
            )
            run._atomic_json(interrupted, {"interrupted": True})
            original_preflight = interrupted.read_bytes()

            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=root / "external"),
                mock.patch.object(run, "_validate_cached_arm", side_effect=validate),
                mock.patch.object(
                    run,
                    "_validate_cached_scientific_arm",
                    side_effect=lambda **kwargs: validate(
                        run.scientific_store.bundle_paths(
                            kwargs["root"], kwargs["relative_prefix"]
                        ).rows
                    ),
                ),
                mock.patch.object(
                    run,
                    "_solver_records",
                    side_effect=lambda **kwargs: [{"id": kwargs["arm"]}],
                ),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    side_effect=lambda _harness, _config, *, budget, n: types.SimpleNamespace(
                        thinking_budget=budget, n=n
                    ),
                ),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog") as catalog_write,
                mock.patch.object(run, "_read_jsonl", side_effect=rows),
                mock.patch.object(run, "_termination_metrics", side_effect=termination),
                mock.patch.object(run, "_stage1_prefix_audit", return_value={"pass": True}),
                mock.patch.object(
                    run, "_scientific_artifact_hashes", return_value={"rows": "frozen"}
                ),
                mock.patch.object(run, "_run_smoke_budget_probe") as probe,
            ):
                selected = run._run_scientific_stage(
                    run="smoke",
                    runner=object(),
                    harness=FakeHarness,
                    domain=object(),
                    config=config,
                    tasks=[],
                    libraries={"base": {}, "designed_ceiling": {}},
                    demonstrations=[],
                    starting_budget=16384,
                )

            self.assertEqual(selected, 32768)
            self.assertEqual(
                generated,
                [("base", 32768), ("designed_ceiling", 32768)],
            )
            self.assertEqual(interrupted.read_bytes(), original_preflight)
            probe.assert_not_called()
            self.assertFalse((root / "runs" / "smoke").exists())
            self.assertEqual(catalog_write.call_args.kwargs["selected_budget"], 32768)
            selection = run._read_json(root / "analysis" / "smoke_budget_selection.json")
            self.assertEqual(selection["selected_thinking_budget"], 32768)
            rejected, selected_tier = selection["tiers"]
            self.assertEqual(rejected["rejecting_arm"], "base")
            self.assertFalse(rejected["complete"])
            self.assertEqual(rejected["arms"]["base"]["status"], "complete")
            self.assertEqual(rejected["arms"]["designed_ceiling"]["status"], "skipped")
            self.assertTrue(selected_tier["complete"])
            self.assertTrue(selected_tier["adequate"])
            self.assertEqual(selected_tier["tier_mode"], "complete_k12_matrix")
            self.assertIsNone(selected_tier["scientific_probe"])
            self.assertTrue(selection["selection_uses_output_content"] is False)

    def test_smoke_uses_probes_only_after_rejected_32k_matrix(self) -> None:
        config = run.load_config()
        validation_calls: dict[tuple[str, str], int] = {}
        generated: list[tuple[str, int, int]] = []
        probe_budgets: list[int] = []

        def validate(path: Path, **_: object) -> bool:
            key = (path.parent.name, path.stem)
            validation_calls[key] = validation_calls.get(key, 0) + 1
            return validation_calls[key] > 1

        def rows(path: Path) -> list[dict[str, object]]:
            return [{"termination_fixture": (path.parent.name, path.stem)}]

        def termination(
            rows: list[dict[str, object]], **_: object
        ) -> dict[str, object]:
            return {
                "adequate": rows[0]["termination_fixture"]
                != ("think_32768", "base")
            }

        def probe(**kwargs: object) -> dict[str, object]:
            budget = int(kwargs["budget"])
            probe_budgets.append(budget)
            adequate = budget == 61440
            return {
                "status": "complete",
                "budget": budget,
                "k": 4,
                "termination": {"adequate": adequate},
                "artifacts": {"rows": f"probe-{budget}"},
                "eligible_for_promotion": False,
                "eligible_for_scoring": False,
                "eligible_for_prefix_pooling": False,
                "selection_uses_output_content": False,
            }

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                records: list[dict[str, object]],
                sampling: types.SimpleNamespace,
            ) -> object:
                del runner
                generated.append(
                    (
                        str(records[0]["id"]),
                        int(sampling.thinking_budget),
                        int(sampling.n),
                    )
                )
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=root / "external"),
                mock.patch.object(run, "_validate_cached_arm", side_effect=validate),
                mock.patch.object(
                    run,
                    "_validate_cached_scientific_arm",
                    side_effect=lambda **kwargs: validate(
                        run.scientific_store.bundle_paths(
                            kwargs["root"], kwargs["relative_prefix"]
                        ).rows
                    ),
                ),
                mock.patch.object(
                    run,
                    "_solver_records",
                    side_effect=lambda **kwargs: [{"id": kwargs["arm"]}],
                ),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    side_effect=lambda _harness, _config, *, budget, n: types.SimpleNamespace(
                        thinking_budget=budget, n=n
                    ),
                ),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog") as catalog_write,
                mock.patch.object(run, "_read_jsonl", side_effect=rows),
                mock.patch.object(run, "_termination_metrics", side_effect=termination),
                mock.patch.object(
                    run, "_stage1_prefix_audit", return_value={"pass": True}
                ) as prefix,
                mock.patch.object(
                    run, "_scientific_artifact_hashes", return_value={"rows": "matrix"}
                ),
                mock.patch.object(run, "_run_smoke_budget_probe", side_effect=probe),
            ):
                selected = run._run_scientific_stage(
                    run="smoke",
                    runner=object(),
                    harness=FakeHarness,
                    domain=object(),
                    config=config,
                    tasks=[],
                    libraries={"base": {}, "designed_ceiling": {}},
                    demonstrations=[],
                    starting_budget=32768,
                )

            self.assertEqual(selected, 61440)
            self.assertEqual(probe_budgets, [49152, 61440])
            self.assertEqual(
                generated,
                [
                    ("base", 32768, 12),
                    ("base", 61440, 12),
                    ("designed_ceiling", 61440, 12),
                ],
            )
            prefix.assert_called_once()
            self.assertFalse((root / "runs" / "smoke").exists())
            self.assertEqual(catalog_write.call_args.kwargs["selected_budget"], 61440)
            selection = run._read_json(root / "analysis" / "smoke_budget_selection.json")
            self.assertEqual(selection["selected_thinking_budget"], 61440)
            rejected_32, probe_only_49, selected_61 = selection["tiers"]
            self.assertEqual(rejected_32["tier_mode"], "complete_k12_matrix")
            self.assertEqual(probe_only_49["status"], "probe_only_rejected")
            self.assertEqual(probe_only_49["tier_mode"], "termination_probe_only")
            self.assertEqual(probe_only_49["scientific_probe"]["k"], 4)
            self.assertEqual(
                probe_only_49["scientific_probe"]["artifacts"],
                {"rows": "probe-49152"},
            )
            self.assertTrue(
                all(
                    arm["status"] == "skipped"
                    for arm in probe_only_49["arms"].values()
                )
            )
            self.assertTrue(selected_61["complete"])
            self.assertTrue(selected_61["adequate"])
            self.assertTrue(selected_61["scientific_probe"]["termination"]["adequate"])
            self.assertTrue(
                selection["probes_excluded_from_promotion_scoring_and_prefix_pooling"]
            )

    def test_smoke_final_inadequate_probe_is_setup_inconclusive(self) -> None:
        config = run.load_config()
        validation_calls = 0
        generated: list[tuple[str, int]] = []
        probe_budgets: list[int] = []

        def validate(_path: Path, **_: object) -> bool:
            nonlocal validation_calls
            validation_calls += 1
            return validation_calls % 2 == 0

        def probe(**kwargs: object) -> dict[str, object]:
            budget = int(kwargs["budget"])
            probe_budgets.append(budget)
            return {
                "status": "complete",
                "budget": budget,
                "k": 4,
                "termination": {"adequate": False},
                "artifacts": {"rows": f"probe-{budget}"},
                "selection_uses_output_content": False,
            }

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                records: list[dict[str, object]],
                sampling: types.SimpleNamespace,
            ) -> object:
                del runner
                generated.append((str(records[0]["id"]), int(sampling.thinking_budget)))
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=root / "external"),
                mock.patch.object(run, "_validate_cached_arm", return_value=True),
                mock.patch.object(
                    run, "_validate_cached_scientific_arm", side_effect=[False, True]
                ),
                mock.patch.object(
                    run,
                    "_solver_records",
                    side_effect=lambda **kwargs: [{"id": kwargs["arm"]}],
                ),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    side_effect=lambda _harness, _config, *, budget, n: types.SimpleNamespace(
                        thinking_budget=budget, n=n
                    ),
                ),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog") as catalog_write,
                mock.patch.object(
                    run,
                    "_read_jsonl",
                    return_value=[{"termination_fixture": "inadequate"}],
                ),
                mock.patch.object(
                    run,
                    "_termination_metrics",
                    return_value={"adequate": False},
                ),
                mock.patch.object(
                    run, "_scientific_artifact_hashes", return_value={"rows": "matrix"}
                ),
                mock.patch.object(run, "_run_smoke_budget_probe", side_effect=probe),
            ):
                with self.assertRaisesRegex(ValueError, "setup-inconclusive"):
                    run._run_scientific_stage(
                        run="smoke",
                        runner=object(),
                        harness=FakeHarness,
                        domain=object(),
                        config=config,
                        tasks=[],
                        libraries={"base": {}, "designed_ceiling": {}},
                        demonstrations=[],
                        starting_budget=32768,
                    )

            self.assertEqual(generated, [("base", 32768)])
            self.assertEqual(probe_budgets, [49152, 61440])
            self.assertTrue(catalog_write.called)
            self.assertFalse((root / "runs" / "smoke").exists())
            selection = run._read_json(root / "analysis" / "smoke_budget_selection.json")
            self.assertFalse(selection["pass"])
            self.assertEqual(
                [tier["status"] for tier in selection["tiers"]],
                ["rejected", "probe_only_rejected", "probe_only_rejected"],
            )
            self.assertFalse(selection["tiers"][-1]["scientific_probe"]["termination"]["adequate"])

    def test_scientific_probe_is_content_blind_and_exactly_cache_validated(self) -> None:
        config = run.load_config()
        records = [
            {
                "id": f"task-{index}::base",
                "meta": {"task_id": f"task-{index}", "arm": "base"},
            }
            for index in range(12)
        ]
        rows = [
            {
                "id": "task-0::base",
                "outputs": [
                    {
                        "text": "THIS ANSWER CONTENT MUST NOT AFFECT BUDGET SELECTION",
                        "n_thinking_tokens": 10,
                    }
                ],
            }
        ]
        generated: list[int] = []

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                input_records: list[dict[str, object]],
                sampling: types.SimpleNamespace,
            ) -> object:
                del runner, input_records
                generated.append(int(sampling.n))
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=root / "external"),
                mock.patch.object(run, "_solver_records", return_value=records),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    return_value=types.SimpleNamespace(thinking_budget=49152, n=4),
                ),
                mock.patch.object(
                    run,
                    "_validate_cached_scientific_arm",
                    side_effect=[False, True, True],
                ) as validate,
                mock.patch.object(run, "_validate_cached_arm", return_value=True),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog"),
                mock.patch.object(run, "_read_jsonl", return_value=rows),
                mock.patch.object(
                    run,
                    "_termination_metrics",
                    return_value={
                        "adequate": True,
                        "selection_uses_output_content": False,
                    },
                ) as termination,
                mock.patch.object(
                    run,
                    "_scientific_artifact_hashes",
                    return_value={"rows": "probe", "meta": "bound", "preflight": "bound"},
                ),
                mock.patch.object(run, "_stage1_prefix_audit") as prefix,
                mock.patch.object(run, "_invoke_analyzer") as analyze,
            ):
                first = run._run_smoke_budget_probe(
                    runner=object(),
                    harness=FakeHarness,
                    domain=object(),
                    config=config,
                    tasks=[],
                    library={},
                    demonstrations=[],
                    budget=49152,
                )
                second = run._run_smoke_budget_probe(
                    runner=object(),
                    harness=FakeHarness,
                    domain=object(),
                    config=config,
                    tasks=[],
                    library={},
                    demonstrations=[],
                    budget=49152,
                )

            self.assertEqual(generated, [4])
            self.assertEqual(validate.call_count, 3)
            expected_path = (
                root
                / "external"
                / "smoke_budget_probes"
                / "think_49152"
                / "base.jsonl"
            )
            self.assertTrue(
                all(
                    run.scientific_store.bundle_paths(
                        call.kwargs["root"], call.kwargs["relative_prefix"]
                    ).rows
                    == expected_path
                    for call in validate.call_args_list
                )
            )
            self.assertTrue(
                all(call.kwargs["expected_n"] == 4 for call in validate.call_args_list)
            )
            self.assertEqual(termination.call_count, 2)
            self.assertTrue(all(call.args[0] is rows for call in termination.call_args_list))
            self.assertFalse(first["selection_uses_output_content"])
            self.assertFalse(first["eligible_for_promotion"])
            self.assertFalse(first["eligible_for_scoring"])
            self.assertFalse(first["eligible_for_prefix_pooling"])
            self.assertEqual(first, second)
            prefix.assert_not_called()
            self.assertFalse((root / "runs" / "smoke").exists())
            analyze.assert_not_called()

    def test_scientific_stage_final_rung_fails_after_first_inadequate_arm(self) -> None:
        config = run.load_config()
        validation_calls: dict[str, int] = {}
        generated: list[str] = []

        def validate(path: Path, **_: object) -> bool:
            validation_calls[path.stem] = validation_calls.get(path.stem, 0) + 1
            return validation_calls[path.stem] > 1

        class FakeHarness:
            @staticmethod
            def generate_vllm_batch(
                runner: object,
                records: list[dict[str, object]],
                sampling: object,
            ) -> object:
                del runner, sampling
                generated.append(str(records[0]["id"]))
                return object()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=root / "external"),
                mock.patch.object(run, "_validate_cached_arm", side_effect=validate),
                mock.patch.object(
                    run,
                    "_validate_cached_scientific_arm",
                    side_effect=lambda **kwargs: validate(
                        run.scientific_store.bundle_paths(
                            kwargs["root"], kwargs["relative_prefix"]
                        ).rows
                    ),
                ),
                mock.patch.object(
                    run,
                    "_solver_records",
                    side_effect=lambda **kwargs: [{"id": kwargs["arm"]}],
                ),
                mock.patch.object(
                    run,
                    "_solver_sampling",
                    side_effect=lambda *_args, **kwargs: types.SimpleNamespace(
                        thinking_budget=kwargs["budget"]
                    ),
                ),
                mock.patch.object(run, "_preflight_records", return_value={"pass": True}),
                mock.patch.object(run.scientific_store, "write_preflight_only"),
                mock.patch.object(run, "_write_batch"),
                mock.patch.object(run, "_commit_scientific_receipt"),
                mock.patch.object(run, "_write_scientific_catalog") as catalog_write,
                mock.patch.object(
                    run,
                    "_read_jsonl",
                    return_value=[{"termination_fixture": "inadequate"}],
                ),
                mock.patch.object(
                    run,
                    "_termination_metrics",
                    return_value={"adequate": False},
                ),
                mock.patch.object(
                    run, "_scientific_artifact_hashes", return_value={"rows": "frozen"}
                ),
            ):
                with self.assertRaisesRegex(ValueError, "setup-inconclusive"):
                    run._run_scientific_stage(
                        run="smoke",
                        runner=object(),
                        harness=FakeHarness,
                        domain=object(),
                        config=config,
                        tasks=[],
                        libraries={"base": {}, "designed_ceiling": {}},
                        demonstrations=[],
                        starting_budget=61440,
                    )

            self.assertEqual(generated, ["base"])
            self.assertTrue(catalog_write.called)
            self.assertFalse((root / "runs" / "smoke").exists())
            selection = run._read_json(root / "analysis" / "smoke_budget_selection.json")
            self.assertFalse(selection["pass"])
            self.assertEqual(len(selection["tiers"]), 1)
            tier = selection["tiers"][0]
            self.assertEqual(tier["budget"], 61440)
            self.assertEqual(tier["rejecting_arm"], "base")
            self.assertEqual(tier["arms"]["designed_ceiling"]["status"], "skipped")

    @staticmethod
    def _fixture_protocol_binding() -> dict[str, object]:
        core: dict[str, object] = {
            "schema_version": 1,
            "experiment_id": run.scientific_store.EXPERIMENT_ID,
            "files": [],
            "smoke_libraries": {
                "base": {"library_id": "base", "content_sha256": "b" * 64},
                "designed_ceiling": {
                    "library_id": "designed",
                    "content_sha256": "d" * 64,
                },
            },
            "library_scope": "fixture",
        }
        return {**core, "binding_sha256": run._sha256_value(core)}

    def test_model_free_migration_is_staged_idempotent_and_removes_only_after_verify(self) -> None:
        config = run.load_config()
        record = {"id": "task::base"}
        budget = 32768
        reserve = budget + 2 + 512
        sampling = types.SimpleNamespace(
            thinking_budget=budget, answer_max_tokens=512, n=12
        )
        preflight = {
            "schema_version": 1,
            "pass": True,
            "max_model_len": 65536,
            "generation_reserve_tokens": reserve,
            "n_records": 1,
            "min_prompt_tokens": 5,
            "max_prompt_tokens": 5,
            "max_prompt_plus_reserve_tokens": reserve + 5,
            "records": [
                {
                    "id": "task::base",
                    "input_record_sha256": run._sha256_value(record),
                    "rendered_prompt_sha256": "a" * 64,
                    "prompt_tokens": 5,
                    "prompt_plus_reserve_tokens": reserve + 5,
                }
            ],
        }
        spec = {
            "probe": False,
            "budget": budget,
            "arm": "base",
            "k": 12,
            "records": [record],
            "sampling": sampling,
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            local = (
                root
                / "runs"
                / "smoke_tiers"
                / "think_32768"
                / "base.preflight.json"
            )
            run._atomic_json(local, preflight)
            external = root / "external"
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=external),
                mock.patch.object(
                    run,
                    "_scientific_protocol_binding",
                    return_value=self._fixture_protocol_binding(),
                ),
                mock.patch.object(
                    run,
                    "_load_prepared",
                    return_value=({}, {"libraries": {"base": {}}}, [], []),
                ),
                mock.patch.object(run, "_stage_tasks", return_value=[]),
                mock.patch.object(run, "_migration_bundle_spec", return_value=spec),
            ):
                first = run.migrate_scientific_artifacts(config, remove_local=False)
                self.assertEqual(first["status"], "installed_and_verified")
                self.assertTrue(external.is_dir())
                self.assertTrue(local.is_file())
                self.assertTrue(
                    (root / "analysis" / "scientific_smoke_artifact_catalog.json").is_file()
                )
                second = run.migrate_scientific_artifacts(config, remove_local=True)
                self.assertEqual(second["status"], "already_installed_and_verified")
                self.assertTrue(second["local_removed"])
                self.assertFalse((root / "runs" / "smoke_tiers").exists())
                self.assertTrue(external.is_dir())

    def test_migration_rejects_temporary_guard_without_installing_it(self) -> None:
        config = run.load_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            guard = (
                root
                / "runs"
                / "smoke_tiers"
                / "think_49152"
                / "base.preflight.json"
            )
            run._atomic_json(guard, {"intentional_guard": True, "pass": False})
            external = root / "external"
            spec = {
                "probe": False,
                "budget": 49152,
                "arm": "base",
                "k": 12,
                "records": [{"id": "task::base"}],
                "sampling": types.SimpleNamespace(
                    thinking_budget=49152, answer_max_tokens=512, n=12
                ),
            }
            with (
                mock.patch.object(run, "RUNS", root / "runs"),
                mock.patch.object(run, "ANALYSIS", root / "analysis"),
                mock.patch.object(run, "_scientific_external_root", return_value=external),
                mock.patch.object(
                    run,
                    "_scientific_protocol_binding",
                    return_value=self._fixture_protocol_binding(),
                ),
                mock.patch.object(
                    run,
                    "_load_prepared",
                    return_value=({}, {"libraries": {"base": {}}}, [], []),
                ),
                mock.patch.object(run, "_stage_tasks", return_value=[]),
                mock.patch.object(run, "_migration_bundle_spec", return_value=spec),
            ):
                with self.assertRaisesRegex(ValueError, "guard or non-passing"):
                    run.migrate_scientific_artifacts(config)
            self.assertFalse(external.exists())
            self.assertTrue(guard.is_file())

    def test_fresh_clone_prepare_does_not_touch_external_scientific_root(self) -> None:
        with mock.patch.object(
            run,
            "_scientific_external_root",
            side_effect=AssertionError("prepare touched external scientific storage"),
        ):
            self.assertEqual(run.main(["--prepare"]), 0)

    def test_cached_runner_artifact_is_bound_to_prompt_sampling_and_runner(self) -> None:
        config = run.load_config()
        sampling = run._solver_sampling(harness, config, budget=16384, n=1)
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
            "think_token_ids": {"forced_close_sequence": [248069, 271]},
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
            "max_model_len": 65536,
            "generation_reserve_tokens": 16898,
            "n_records": 1,
            "min_prompt_tokens": 10,
            "max_prompt_tokens": 10,
            "max_prompt_plus_reserve_tokens": 16908,
            "records": [
                {
                    "id": record["id"],
                    "input_record_sha256": run._sha256_value(record),
                    "rendered_prompt_sha256": rendered_hash,
                    "prompt_tokens": 10,
                    "prompt_plus_reserve_tokens": 16908,
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
            bad_reserve = copy.deepcopy(preflight)
            bad_reserve["generation_reserve_tokens"] -= 1
            run._atomic_json(preflight_path, bad_reserve)
            with self.assertRaisesRegex(ValueError, "generation reserve mismatch"):
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=[record],
                    sampling=sampling,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )
            bad_per_record = copy.deepcopy(preflight)
            bad_per_record["records"][0]["prompt_plus_reserve_tokens"] -= 1
            run._atomic_json(preflight_path, bad_per_record)
            with self.assertRaisesRegex(ValueError, "prompt-plus-reserve drift"):
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=[record],
                    sampling=sampling,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )
            run._atomic_json(preflight_path, preflight)
            drifted = run._solver_sampling(harness, config, budget=16384, n=1)
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

    def test_cached_runner_artifact_rejects_preflight_and_row_reordering(self) -> None:
        config = run.load_config()
        sampling = run._solver_sampling(harness, config, budget=16384, n=1)
        records = [
            {
                "id": f"task-{suffix}::base",
                "messages": [{"role": "user", "content": suffix}],
                "meta": {"arm": "base", "task_id": f"task-{suffix}"},
            }
            for suffix in ("a", "b")
        ]
        rendered_hashes = ["a" * 64, "b" * 64]
        rows = [
            {
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
            for record, rendered_hash in zip(records, rendered_hashes, strict=True)
        ]
        accounting = harness.extract_token_accounting(rows)
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
            "think_token_ids": {"forced_close_sequence": [248069, 271]},
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
            "max_model_len": 65536,
            "generation_reserve_tokens": 16898,
            "n_records": len(records),
            "min_prompt_tokens": 10,
            "max_prompt_tokens": 10,
            "max_prompt_plus_reserve_tokens": 16908,
            "records": [
                {
                    "id": record["id"],
                    "input_record_sha256": run._sha256_value(record),
                    "rendered_prompt_sha256": rendered_hash,
                    "prompt_tokens": 10,
                    "prompt_plus_reserve_tokens": 16908,
                }
                for record, rendered_hash in zip(records, rendered_hashes, strict=True)
            ],
        }

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "base.jsonl"
            preflight_path = path.with_suffix(".preflight.json")
            run._atomic_jsonl(path, rows)
            run._atomic_json(path.with_suffix(".meta.json"), summary)
            run._atomic_json(preflight_path, preflight)
            self.assertTrue(
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=records,
                    sampling=sampling,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )
            )

            reordered_preflight = {**preflight, "records": list(reversed(preflight["records"]))}
            run._atomic_json(preflight_path, reordered_preflight)
            with self.assertRaisesRegex(ValueError, "record id order mismatch"):
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=records,
                    sampling=sampling,
                    harness=harness,
                    config=config,
                    expected_n=1,
                )

            run._atomic_json(preflight_path, preflight)
            run._atomic_jsonl(path, list(reversed(rows)))
            with self.assertRaisesRegex(ValueError, "row id order mismatch"):
                run._validate_runner_artifact(
                    path,
                    preflight_path=preflight_path,
                    records=records,
                    sampling=sampling,
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

if __name__ == "__main__":
    unittest.main()
