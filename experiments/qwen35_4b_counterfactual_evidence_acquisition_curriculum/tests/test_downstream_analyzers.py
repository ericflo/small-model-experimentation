from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import yaml


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"
sys.path.insert(0, str(SCRIPTS))


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_script("downstream_common")
calibration = load_script("analyze_calibration")
transfer = load_script("analyze_transfer")
transfer_feasibility = load_script("analyze_transfer_feasibility")
retention = load_script("analyze_retention")
menagerie = load_script("analyze_menagerie")


def base_config() -> dict:
    return yaml.safe_load((EXP / "configs" / "default.yaml").read_text())


def synthetic_aggregate(
    rate: float,
    *,
    scenario: str = "ambiguous_source",
    success: float = 0.8,
    channels: tuple[str, ...] = ("tests", "docs", "callsite"),
    query_skin: str = "signature",
) -> dict:
    families = ("spec_switch", "spec_update")
    dyads = []
    for index in range(16):
        dyads.append({
            "pair_id": f"pair-{index:02d}",
            "scenario": scenario,
            "family": families[index % len(families)],
            "evidence_channel": channels[index % len(channels)],
            "evidence_path_regime": "transfer",
            "acquisition_query_skin": query_skin,
            "explicit_contract": False,
            "paired_preverifier_success": index < round(rate * 16),
        })

    def dimension(key: str) -> dict:
        result = {}
        for name in sorted({row[key] for row in dyads}):
            rows = [row for row in dyads if row[key] == name]
            result[name] = {
                "paired_preverifier_success": sum(
                    row["paired_preverifier_success"] for row in rows
                ) / len(rows)
            }
        return result

    return {
        "paired_preverifier_success": rate,
        "success": success,
        "first_patch_full_correct": 0.9,
        "unnecessary_evidence_before_first_patch": 0.1,
        "invalid_action_rate_per_turn": 0.0,
        "unusable_answer_cap_hit_rate_per_turn": 0.0,
        "verified_given_success": 0.95,
        "commit_given_verified": 0.95,
        "mean_non_source_inspects_before_first_patch": 1.0,
        "mean_sampled_tokens": 700.0,
        "mean_logical_model_tokens": 1600.0,
        "per_family": dimension("family"),
        "per_channel": dimension("evidence_channel"),
        "per_query_skin": dimension("acquisition_query_skin"),
        "per_scenario": {
            "rejected_patch": {
                "changed_patch_within_two": 1.0,
                "valid_changed_patch_within_two": 1.0,
            },
            "failed_test": {"changed_patch_within_two": 1.0},
        },
        "dyads": dyads,
    }


def behavior_payload(
    rate: float,
    *,
    arm: str,
    model_hash: str,
    block: str = "transfer_dev",
    scenario_set: str = "acquisition",
    scenario: str = "ambiguous_source",
    scaffold: bool = False,
    success: float = 0.8,
) -> dict:
    return {
        "arm": arm,
        "model_weight_sha256": model_hash,
        "block": block,
        "contract": "inferred",
        "scenario_set": scenario_set,
        "mode": "deep",
        "scaffold": scaffold,
        "history_policy": common.HISTORY_POLICY,
        "think_budget": 512,
        "answer_max_tokens": 1024,
        "task_manifest_sha256": "1" * 64,
        "task_content_manifest_sha256": "2" * 64,
        "pair_static_manifest_sha256": "3" * 64,
        "composed_mapping_manifest": [{"registered": True}],
        "aggregate": synthetic_aggregate(rate, scenario=scenario, success=success),
    }


class CalibrationAnalyzerTests(unittest.TestCase):
    def test_feasibility_and_candidate_gate_use_paired_capability_metrics(self) -> None:
        cfg = base_config()
        start = behavior_payload(
            0.30,
            arm="start",
            model_hash="a" * 64,
            block="trained_calibration",
        )
        explicit = behavior_payload(
            0.40,
            arm="explicit_redundant",
            model_hash="b" * 64,
            block="trained_calibration",
        )
        shuffled = behavior_payload(
            0.35,
            arm="shuffled_binding",
            model_hash="d" * 64,
            block="trained_calibration",
        )
        feasibility = calibration.analyze(
            cfg,
            start=start,
            explicit_control=explicit,
            shuffled_control=shuffled,
        )
        self.assertTrue(feasibility["gate"]["passed"])

        candidate = behavior_payload(
            0.70,
            arm="candidate",
            model_hash="c" * 64,
            block="trained_calibration",
        )
        candidate_explicit = behavior_payload(
            0.70,
            arm="candidate",
            model_hash="c" * 64,
            block="explicit_retention",
        )
        candidate_explicit["answer_max_tokens"] = 1024
        locality = {
            "gate": {"passed": True},
            "median_non_target_centered_logit_drift": 0.05,
            "mean_entropy_delta": 0.0,
            "mean_varentropy_delta": 0.01,
        }
        result = calibration.analyze(
            cfg,
            start=start,
            explicit_control=explicit,
            shuffled_control=shuffled,
            candidate=candidate,
            candidate_explicit=candidate_explicit,
            locality=locality,
        )
        self.assertTrue(result["gate"]["passed"])
        self.assertAlmostEqual(result["candidate_deltas"]["start"], 0.40)

    def test_feasibility_fails_when_control_leaves_no_registered_headroom(self) -> None:
        cfg = base_config()
        start = behavior_payload(
            0.95,
            arm="start",
            model_hash="a" * 64,
            block="trained_calibration",
        )
        result = calibration.analyze(
            cfg,
            start=start,
            explicit_control=behavior_payload(
                0.95,
                arm="explicit_redundant",
                model_hash="b" * 64,
                block="trained_calibration",
            ),
            shuffled_control=behavior_payload(
                0.95,
                arm="shuffled_binding",
                model_hash="d" * 64,
                block="trained_calibration",
            ),
        )
        self.assertFalse(result["gate"]["passed"])
        self.assertFalse(result["checks"]["delta_vs_start_attainable"])


class TransferAnalyzerTests(unittest.TestCase):
    def _inputs(self) -> tuple[dict, dict]:
        cfg = base_config()
        candidate_hash = "c" * 64
        start_hash = "a" * 64
        candidate = behavior_payload(0.75, arm="candidate", model_hash=candidate_hash)
        start = behavior_payload(0.50, arm="start", model_hash=start_hash)
        incumbent = behavior_payload(0.50, arm="incumbent", model_hash="e" * 64)
        explicit = behavior_payload(
            0.50, arm="explicit_redundant", model_hash="b" * 64
        )
        shuffled = behavior_payload(
            0.50, arm="shuffled_binding", model_hash="d" * 64
        )
        control_search = behavior_payload(
            0.50,
            arm="candidate",
            model_hash=candidate_hash,
            scenario_set="random",
            scenario="nondiscriminating_search_injected",
        )
        injected = behavior_payload(
            0.80,
            arm="candidate",
            model_hash=candidate_hash,
            scenario_set="injected",
            scenario="evidence_injected",
        )
        normal_candidate = behavior_payload(
            0.70,
            arm="candidate",
            model_hash=candidate_hash,
            scenario_set="normal",
            scenario="normal",
            success=0.80,
        )
        normal_start = behavior_payload(
            0.60,
            arm="start",
            model_hash=start_hash,
            scenario_set="normal",
            scenario="normal",
            success=0.70,
        )
        recovery_candidate = behavior_payload(
            0.70,
            arm="candidate",
            model_hash=candidate_hash,
            scenario_set="recovery",
            scenario="rejected_patch",
            success=0.80,
        )
        recovery_start = behavior_payload(
            0.60,
            arm="start",
            model_hash=start_hash,
            scenario_set="recovery",
            scenario="rejected_patch",
            success=0.70,
        )
        scaffold = behavior_payload(
            0.75,
            arm="candidate",
            model_hash=candidate_hash,
            scenario_set="recovery",
            scenario="rejected_patch",
            scaffold=True,
            success=0.90,
        )
        explicit_retention = behavior_payload(
            0.70,
            arm="candidate",
            model_hash=candidate_hash,
            block="explicit_retention",
        )
        inputs = {
            "block": "transfer_dev",
            "candidate": candidate,
            "start": start,
            "incumbent": incumbent,
            "explicit_control": explicit,
            "shuffled_control": shuffled,
            "nondiscriminating_search": control_search,
            "candidate_injected": injected,
            "candidate_normal": normal_candidate,
            "start_normal": normal_start,
            "candidate_recovery": recovery_candidate,
            "start_recovery": recovery_start,
            "candidate_recovery_scaffold": scaffold,
            "candidate_explicit": explicit_retention,
            "sample_match_start": {
                "dual_overmatch_paired_preverifier_success": 0.55
            },
            "sample_match_incumbent": {
                "dual_overmatch_paired_preverifier_success": 0.60
            },
        }
        return cfg, inputs

    def test_transfer_uses_dyad_bootstrap_and_stronger_matched_sample_control(self) -> None:
        cfg, inputs = self._inputs()
        result = transfer.analyze(cfg, **inputs)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["paired_bootstrap_vs_start"]["unit"], "counterfactual_dyad"
        )
        self.assertEqual(
            result["sample_more_comparator"]["selected_stronger_arm"], "incumbent"
        )
        self.assertFalse(result["sample_more_comparator"]["full_pool_used_for_gate"])
        self.assertEqual(set(result["channel_rates"]), {"tests", "docs", "callsite"})
        self.assertEqual(set(result["query_skin_rates"]), {"signature"})

    def test_transfer_fails_closed_on_required_query_skin(self) -> None:
        cfg, inputs = self._inputs()
        inputs["candidate"]["aggregate"]["per_query_skin"]["signature"][
            "paired_preverifier_success"
        ] = 0.49
        result = transfer.analyze(cfg, **inputs)
        self.assertFalse(result["checks"]["all_transfer_query_skins"])
        self.assertFalse(result["gate"]["passed"])

    def test_rejected_recovery_requires_a_valid_changed_patch(self) -> None:
        cfg, inputs = self._inputs()
        rejected = inputs["candidate_recovery"]["aggregate"]["per_scenario"][
            "rejected_patch"
        ]
        rejected["changed_patch_within_two"] = 1.0
        rejected["valid_changed_patch_within_two"] = 0.0
        result = transfer.analyze(cfg, **inputs)
        self.assertEqual(result["recovery"]["rejected_transition"], 0.0)
        self.assertFalse(result["checks"]["rejected_transition"])

    def test_dual_match_prefix_cannot_peek_at_later_success(self) -> None:
        costs = [
            {
                "trajectory": 0,
                "sampled_tokens": 6,
                "logical_model_tokens": 6,
                "workspace_success": False,
                "preverifier_member_success": False,
            },
            {
                "trajectory": 1,
                "sampled_tokens": 4,
                "logical_model_tokens": 4,
                "workspace_success": False,
                "preverifier_member_success": False,
            },
            {
                "trajectory": 2,
                "sampled_tokens": 100,
                "logical_model_tokens": 100,
                "workspace_success": True,
                "preverifier_member_success": True,
            },
        ]
        prefix = transfer._recompute_dual_prefix(
            costs, target_sampled=10, target_logical=10
        )
        self.assertEqual(prefix["trajectories"], 2)
        self.assertFalse(prefix["workspace_success"])
        self.assertFalse(prefix["preverifier_member_success"])

    def test_transfer_feasibility_stops_unreachable_baseline_and_sample_margins(self) -> None:
        cfg = base_config()
        baseline = transfer_feasibility.baseline_headroom_checks(
            cfg,
            start=0.99,
            incumbent=0.50,
            explicit_control=0.50,
            shuffled_control=0.50,
            start_normal=0.50,
        )
        self.assertFalse(baseline["delta_vs_start_attainable"])
        comparators = transfer_feasibility.comparator_headroom_checks(
            cfg,
            nondiscriminating_search=0.50,
            sample_start=None,
            sample_incumbent=0.50,
        )
        self.assertFalse(comparators["sample_pool_compute_attainable"])
        self.assertFalse(
            comparators["delta_vs_stronger_sample_more_attainable"]
        )


def retention_payload(
    families: tuple[str, ...],
    *,
    candidate: bool,
    scenario: str,
    model_hash: str,
    family_penalty: str | None = None,
) -> dict:
    per_family = {family: {"success": 0.8} for family in families}
    if family_penalty is not None and candidate:
        per_family[family_penalty]["success"] = 0.5
    cases = []
    scenarios = ("normal",) if scenario == "normal" else (
        "rejected_patch",
        "failed_test",
    )
    for family in families:
        for scenario_name in scenarios:
            cases.append({
                "scenario": scenario_name,
                "family": family,
                "success": True,
                "verified": True,
                "commit": True,
                "rejected_transition": scenario_name == "rejected_patch",
                "failed_transition": scenario_name == "failed_test",
            })
    success = sum(row["success"] for row in per_family.values()) / len(per_family)
    return {
        "task_manifest_sha256": "9" * 64,
        "answer_max_tokens": 1024,
        "model_weight_sha256": model_hash,
        "analyzer_turn_rates": {
            "invalid_action_rate_per_turn": 0.0,
            "unusable_answer_cap_hit_rate_per_turn": 0.0,
        },
        "aggregate": {
            "success": success,
            "verified": 0.95,
            "commit": 0.95,
            "per_family": per_family,
            "cases": cases,
        },
    }


class RetentionAnalyzerTests(unittest.TestCase):
    def test_transition_rates_are_conditioned_on_the_relevant_scenario(self) -> None:
        cfg = base_config()
        candidate_hash = "c" * 64
        broad_families = tuple(cfg["families"]["legacy_broad_recovery"])
        transaction_families = tuple(cfg["families"]["legacy_transaction_transfer"])

        def bundle(families: tuple[str, ...]) -> dict[str, dict]:
            return {
                "candidate_normal": retention_payload(
                    families, candidate=True, scenario="normal", model_hash=candidate_hash
                ),
                "start_normal": retention_payload(
                    families, candidate=False, scenario="normal", model_hash="a" * 64
                ),
                "candidate_recovery": retention_payload(
                    families, candidate=True, scenario="recovery", model_hash=candidate_hash
                ),
                "start_recovery": retention_payload(
                    families, candidate=False, scenario="recovery", model_hash="a" * 64
                ),
            }

        result = retention.analyze(
            cfg,
            broad=bundle(broad_families),
            transaction=bundle(transaction_families),
        )
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(
            result["substrates"]["broad"]["transition_metrics"],
            {"rejected": 1.0, "failed": 1.0},
        )
        self.assertEqual(
            result["substrates"]["broad"]["normal_verified_given_success"], 1.0
        )
        self.assertEqual(
            result["substrates"]["broad"]["normal_commit_given_verified"], 1.0
        )


def menagerie_event(
    tier: str,
    seed: int,
    *,
    incumbent: float,
    candidate: float,
    candidate_hash: str = "c" * 64,
) -> dict:
    def arm(weight_hash: str, score: float) -> dict:
        incumbent_arm = weight_hash == "e" * 64
        return {
            "model_path": "/models/incumbent" if incumbent_arm else "/models/candidate",
            "model_weight_sha256": weight_hash,
            "model_config_sha256": ("1" if incumbent_arm else "2") * 64,
            "generation_config_sha256": ("0" if incumbent_arm else "f") * 64,
            "merge_receipt_sha256": ("3" if incumbent_arm else "4") * 64,
            "tokenizer_files": {
                name: ("8" if incumbent_arm else "9") * 64
                for name in common.harness.TOKENIZER_FILE_NAMES
            },
            "tokenizer_manifest_sha256": ("5" if incumbent_arm else "6") * 64,
            "tokenizer_compatibility_sha256": "7" * 64,
            "aggregate": score,
            "per_family": {"family_a": score, "family_b": score},
            "within_budget": True,
            "wall_seconds": 1.0,
        }

    return {
        "schema_version": 1,
        "tier": tier,
        "seed": seed,
        "arms": {
            "incumbent": arm("e" * 64, incumbent),
            "candidate": arm(candidate_hash, candidate),
        },
        "firewall_storage": "aggregate_and_per_family_only",
        "delta": candidate - incumbent,
        "provenance": {
            "config_path": "/experiment/configs/default.yaml",
            "config_sha256": "a" * 64,
            "bench_path": "/experiment/scripts/bench.py",
            "bench_sha256": "b" * 64,
            "analyzer_path": "/experiment/scripts/analyze_menagerie.py",
            "analyzer_sha256": "c" * 64,
            "design_lock_path": "/experiment/runs/preregistration_receipt.json",
            "design_lock_sha256": "d" * 64,
            "design_commit": "1" * 40,
            "authorization_path": "/experiment/analysis/whitebox_authorization.json",
            "authorization_sha256": "e" * 64,
            "public_menagerie_git_tree": {
                "repository_path": str(Path("benchmarks") / "menagerie"),
                "git_tree_oid": "2" * 40,
            },
        },
    }


class MenagerieAnalyzerTests(unittest.TestCase):
    @staticmethod
    def _authorization(locality_path: Path) -> dict:
        return {
            "gate": {"passed": True},
            "all_whitebox_gates_passed": True,
            "menagerie_authorized": True,
            "candidate_model_weight_sha256": "c" * 64,
            "incumbent_model_weight_sha256": "e" * 64,
            "selected_answer_max_tokens": 1024,
            "gate_receipts": {
                "locality_candidate_vs_anchor": {
                    "path": str(locality_path.resolve()),
                    "sha256": common.sha256_file(locality_path),
                }
            },
            "authorization_path": "/authorization",
            "authorization_sha256": "0" * 64,
        }

    def test_aggregate_only_events_reduce_under_frozen_tier_gate(self) -> None:
        cfg = dict(base_config()["menagerie"])
        cfg["_anchor_weight_sha256"] = "e" * 64
        cfg["_anchor_generation_config_sha256"] = "0" * 64
        cfg["_start_generation_config_sha256"] = "f" * 64
        cfg["_anchor_tokenizer_manifest_sha256"] = "5" * 64
        cfg["_start_tokenizer_manifest_sha256"] = "6" * 64
        cfg["_tokenizer_compatibility_sha256"] = "7" * 64
        with tempfile.TemporaryDirectory() as temporary:
            locality_path = Path(temporary) / "locality.json"
            locality_path.write_text(json.dumps({
                "before_model_config_sha256": "1" * 64,
                "after_model_config_sha256": "2" * 64,
                "before_model_generation_config_sha256": "0" * 64,
                "after_model_generation_config_sha256": "f" * 64,
                "before_merge_receipt_sha256": "3" * 64,
                "after_merge_receipt_sha256": "4" * 64,
                "before_tokenizer_manifest_sha256": "5" * 64,
                "after_tokenizer_manifest_sha256": "6" * 64,
                "before_tokenizer_compatibility_sha256": "7" * 64,
                "after_tokenizer_compatibility_sha256": "7" * 64,
            }))
            events = [
                menagerie.validate_event(
                    menagerie_event("quick", 71311, incumbent=0.50, candidate=0.53),
                    cfg,
                    source="quick",
                ),
                menagerie.validate_event(
                    menagerie_event("medium", 71312, incumbent=0.50, candidate=0.49),
                    cfg,
                    source="medium",
                ),
            ]
            result = menagerie.analyze(
                cfg, events, self._authorization(locality_path)
            )
            self.assertTrue(result["gate"]["passed"])
            unexpected = menagerie.validate_event(
                menagerie_event("quick", 99999, incumbent=0.50, candidate=0.53),
                cfg,
                source="unexpected",
            )
            with self.assertRaisesRegex(SystemExit, "unexpected"):
                menagerie.analyze(
                    cfg, [*events, unexpected], self._authorization(locality_path)
                )

    def test_firewall_rejects_task_level_detail_even_with_marker(self) -> None:
        cfg = dict(base_config()["menagerie"])
        cfg["_anchor_weight_sha256"] = "e" * 64
        cfg["_anchor_generation_config_sha256"] = "0" * 64
        cfg["_start_generation_config_sha256"] = "f" * 64
        cfg["_anchor_tokenizer_manifest_sha256"] = "5" * 64
        cfg["_start_tokenizer_manifest_sha256"] = "6" * 64
        cfg["_tokenizer_compatibility_sha256"] = "7" * 64
        event = menagerie_event("quick", 71311, incumbent=0.50, candidate=0.53)
        event["tasks"] = [{"prompt": "must never be consumed"}]
        with self.assertRaisesRegex(SystemExit, "detail key"):
            menagerie.validate_event(event, cfg, source="contaminated")

    def test_analysis_rejects_an_alternate_config_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            alternate = Path(temporary) / "default.yaml"
            alternate.write_text(
                (EXP / "configs" / "default.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SystemExit, "exact frozen default config"):
                menagerie.load_registered_config(alternate)

    def test_event_requires_matching_completed_seed_reservation(self) -> None:
        cfg = dict(base_config()["menagerie"])
        cfg["_anchor_weight_sha256"] = "e" * 64
        cfg["_anchor_generation_config_sha256"] = "0" * 64
        cfg["_start_generation_config_sha256"] = "f" * 64
        cfg["_anchor_tokenizer_manifest_sha256"] = "5" * 64
        cfg["_start_tokenizer_manifest_sha256"] = "6" * 64
        cfg["_tokenizer_compatibility_sha256"] = "7" * 64
        raw = menagerie_event("quick", 71311, incumbent=0.50, candidate=0.53)
        event = menagerie.validate_event(raw, cfg, source="synthetic")
        provenance = raw["provenance"]
        with tempfile.TemporaryDirectory() as temporary:
            original_exp = menagerie.EXP
            try:
                menagerie.EXP = Path(temporary)
                with self.assertRaisesRegex(SystemExit, "unreserved"):
                    menagerie.validate_completed_reservation(
                        raw, event, provenance, source="preseeded"
                    )
                path = menagerie.reservation_path("quick", 71311)
                path.parent.mkdir(parents=True)
                reservation = {
                    "schema_version": 1,
                    "tier": "quick",
                    "seed": 71311,
                    "status": "aggregate_event_recorded",
                    "provenance": provenance,
                    "checkpoint_fingerprints": (
                        menagerie.checkpoint_fingerprints_from_event(event)
                    ),
                    "event_sha256": menagerie.canonical_json_sha256(raw),
                }
                path.write_text(json.dumps(reservation), encoding="utf-8")
                observed = menagerie.validate_completed_reservation(
                    raw, event, provenance, source="registered"
                )
                self.assertEqual(observed["event_sha256"], reservation["event_sha256"])
                reservation["event_sha256"] = "0" * 64
                path.write_text(json.dumps(reservation), encoding="utf-8")
                with self.assertRaisesRegex(SystemExit, "no matching completed"):
                    menagerie.validate_completed_reservation(
                        raw, event, provenance, source="tampered"
                    )
            finally:
                menagerie.EXP = original_exp

    def test_public_benchmark_identity_uses_git_tree_metadata_and_rejects_drift(self) -> None:
        original = menagerie._git_output
        calls = []
        public_tree = str(Path("benchmarks") / "menagerie")

        def stable(*args: str) -> str:
            calls.append(args)
            if args[:2] == ("status", "--short"):
                return ""
            return "a" * 40

        try:
            menagerie._git_output = stable
            identity = menagerie.public_menagerie_git_tree_identity("1" * 40)
            self.assertEqual(identity["git_tree_oid"], "a" * 40)
            self.assertEqual(
                calls,
                [
                    ("status", "--short", "--", public_tree),
                    ("rev-parse", "1" * 40 + ":" + public_tree),
                    ("rev-parse", "HEAD:" + public_tree),
                ],
            )

            def drifted(*args: str) -> str:
                if args[:2] == ("status", "--short"):
                    return ""
                return "a" * 40 if args[1].startswith("1" * 40) else "b" * 40

            menagerie._git_output = drifted
            with self.assertRaisesRegex(SystemExit, "changed after the design lock"):
                menagerie.public_menagerie_git_tree_identity("1" * 40)
        finally:
            menagerie._git_output = original

    def test_checkpoint_fingerprint_rehashes_every_live_surface(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = Path(temporary) / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            (model / "generation_config.json").write_text("{}\n", encoding="utf-8")
            (model / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
            (model / "chat_template.jinja").write_text(
                "{{ messages }}\n", encoding="utf-8"
            )
            (model / "tokenizer.json").write_text(
                '{"version":"1.0"}\n', encoding="utf-8"
            )
            (model / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
            before = menagerie.checkpoint_fingerprint(model)
            self.assertEqual(set(before), menagerie.CHECKPOINT_FINGERPRINT_KEYS)
            (model / "merge_receipt.json").write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            after = menagerie.checkpoint_fingerprint(model)
            self.assertNotEqual(
                before["merge_receipt_sha256"], after["merge_receipt_sha256"]
            )

    def test_authorization_rehashes_every_whitebox_gate_receipt(self) -> None:
        cfg = base_config()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "analysis").mkdir()
            (root / "scripts").mkdir()
            (root / "configs").mkdir()
            script_names = {
                "run.py",
                "audit_locality.py",
                "analyze_calibration.py",
                "analyze_transfer.py",
                "analyze_retention.py",
            }
            for name in script_names:
                (root / "scripts" / name).write_text(f"# {name}\n")
            config_path = root / "configs" / "default.yaml"
            config_path.write_text("schema_version: 1\n")
            config_hash = common.sha256_file(config_path)
            design_lock = root / "runs" / "preregistration_receipt.json"
            design_lock.parent.mkdir()
            design_lock.write_text('{"status":"locked"}\n')
            candidate_hash = "c" * 64
            incumbent_hash = cfg["model"]["anchor_weight_sha256"]
            cfg["artifacts"]["root"] = "large_artifacts/test_authorization"
            training_receipts = {}
            serial = {}
            steps = {}
            merged_hashes = {}
            for arm in ("evidence_binding", "explicit_redundant", "shuffled_binding"):
                path = (
                    root / cfg["artifacts"]["root"] / "adapters" / arm
                    / "training_receipt.json"
                )
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "arm": arm,
                    "max_steps": 36,
                    "optimizer_steps": 36,
                    "serial_forward_tokens_per_epoch": 1000,
                }))
                training_receipts[arm] = {
                    "path": str(path.resolve()),
                    "sha256": common.sha256_file(path),
                }
                serial[arm] = 1000
                steps[arm] = {"planned": 36, "actual": 36}
                merged = root / cfg["artifacts"]["root"] / "merged" / arm
                merged.mkdir(parents=True)
                (merged / "model.safetensors").write_bytes(arm.encode())
                (merged / "config.json").write_text("{}\n")
                (merged / "generation_config.json").write_text("{}\n")
                (merged / "merge_receipt.json").write_text("{}\n")
                (merged / "chat_template.jinja").write_text("{{ messages }}\n")
                (merged / "tokenizer.json").write_text('{"version":"1.0"}\n')
                (merged / "tokenizer_config.json").write_text("{}\n")
                merged_hashes[arm] = menagerie.checkpoint_fingerprint(merged)[
                    "model_weight_sha256"
                ]
            candidate_hash = merged_hashes["evidence_binding"]
            gate_payloads = {
                "training_compute_gate": {
                    "schema_version": 1,
                    "stage": "training_compute",
                    "gate": {
                        "passed": True,
                        "verdict": "TRAINING_COMPUTE_MATCHED",
                    },
                    "issuer_sha256": common.sha256_file(
                        root / "scripts" / "run.py"
                    ),
                    "config_sha256": config_hash,
                    "design_lock_sha256": common.sha256_file(design_lock),
                    "candidate_model_weight_sha256": candidate_hash,
                    "training_receipts": training_receipts,
                    "serial_forward_tokens_per_epoch": serial,
                    "optimizer_steps": steps,
                    "merged_model_weight_sha256": merged_hashes,
                    "max_to_min_ratio": 1.0,
                    "registered_ratio_max": float(
                        cfg["training"]["serial_token_compute_ratio_max"]
                    ),
                    "menagerie_authorized": False,
                },
                "locality_candidate_vs_anchor": {
                    "gate": {"passed": True},
                    "after_model_weight_sha256": candidate_hash,
                    "before_model_weight_sha256": incumbent_hash,
                    "auditor_sha256": common.sha256_file(
                        root / "scripts" / "audit_locality.py"
                    ),
                },
                "calibration_gate": {
                    "gate": {"passed": True},
                    "stage": "trained_calibration",
                    "candidate_model_weight_sha256": candidate_hash,
                    "answer_max_tokens": 1024,
                    "analyzer_sha256": common.sha256_file(
                        root / "scripts" / "analyze_calibration.py"
                    ),
                    "config_sha256": config_hash,
                },
                "transfer_dev_gate": {
                    "gate": {"passed": True},
                    "stage": "transfer",
                    "block": "transfer_dev",
                    "candidate_model_weight_sha256": candidate_hash,
                    "answer_max_tokens": 1024,
                    "analyzer_sha256": common.sha256_file(
                        root / "scripts" / "analyze_transfer.py"
                    ),
                    "config_sha256": config_hash,
                },
                "transfer_confirm_gate": {
                    "gate": {"passed": True},
                    "stage": "transfer",
                    "block": "transfer_confirm",
                    "candidate_model_weight_sha256": candidate_hash,
                    "answer_max_tokens": 1024,
                    "analyzer_sha256": common.sha256_file(
                        root / "scripts" / "analyze_transfer.py"
                    ),
                    "config_sha256": config_hash,
                },
                "retention_gate": {
                    "gate": {"passed": True},
                    "stage": "legacy_retention",
                    "candidate_model_weight_sha256": candidate_hash,
                    "answer_max_tokens": 1024,
                    "analyzer_sha256": common.sha256_file(
                        root / "scripts" / "analyze_retention.py"
                    ),
                    "config_sha256": config_hash,
                },
            }
            gate_receipts = {}
            for name, payload in gate_payloads.items():
                gate_path = root / "analysis" / f"{name}.json"
                gate_path.write_text(json.dumps(payload))
                gate_receipts[name] = {
                    "path": str(gate_path.resolve()),
                    "sha256": common.sha256_file(gate_path),
                }
            authorization_path = root / "analysis" / "whitebox_authorization.json"
            authorization = {
                "schema_version": 1,
                "stage": "whitebox_authorization",
                "gate": {"passed": True, "verdict": "WHITEBOX_PASS"},
                "all_whitebox_gates_passed": True,
                "menagerie_authorized": True,
                "candidate_model_weight_sha256": candidate_hash,
                "incumbent_model_weight_sha256": incumbent_hash,
                "selected_answer_max_tokens": 1024,
                "gate_receipts": gate_receipts,
            }
            authorization_path.write_text(json.dumps(authorization))
            original_exp, original_root = menagerie.EXP, menagerie.ROOT
            try:
                menagerie.EXP, menagerie.ROOT = root, root
                observed = menagerie.validate_authorization(authorization_path, cfg)
                self.assertEqual(
                    observed["authorization_sha256"],
                    common.sha256_file(authorization_path),
                )
                self.assertEqual(
                    observed["gate_receipts"]["retention_gate"]["sha256"],
                    gate_receipts["retention_gate"]["sha256"],
                )
            finally:
                menagerie.EXP, menagerie.ROOT = original_exp, original_root


class BehaviorReceiptProvenanceTests(unittest.TestCase):
    def test_registered_task_grid_rejects_tasks_per_family_override(self) -> None:
        cfg = base_config()
        config_text = (EXP / "configs" / "default.yaml").read_text()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text(config_text)
            cfg["__analysis_config_path__"] = str(config_path.resolve())
            model = root / "model"
            model.mkdir()
            (model / "model.safetensors").write_bytes(b"weights")
            (model / "config.json").write_text("{}")
            (model / "generation_config.json").write_text("{}")
            (model / "merge_receipt.json").write_text("{}")
            (model / "chat_template.jinja").write_text("{{ messages }}\n")
            (model / "tokenizer.json").write_text('{"version":"1.0"}\n')
            (model / "tokenizer_config.json").write_text("{}\n")
            tokenizer_provenance = common.harness.tokenizer_provenance(model)
            cfg["model"]["start_tokenizer_manifest_sha256"] = (
                tokenizer_provenance["tokenizer_manifest_sha256"]
            )
            cfg["model"]["tokenizer_compatibility_sha256"] = (
                tokenizer_provenance["tokenizer_compatibility_sha256"]
            )
            tasks = common._registered_behavior_tasks(
                cfg, "trained_calibration", "inferred"
            )
            manifests = common._registered_behavior_manifests(tasks)
            cases = []
            by_pair: dict[str, list[dict]] = defaultdict(list)
            trajectories = []
            for task in tasks:
                case = {
                    "case_id": f"{task.task_id}:ambiguous_source",
                    "task_id": task.task_id,
                    "pair_id": task.pair_id,
                    "branch": task.branch,
                    "scenario": "ambiguous_source",
                    "family": task.family,
                    "evidence_channel": task.evidence_channel,
                    "acquisition_query_skin": task.acquisition_query_skin,
                    "success": False,
                }
                cases.append(case)
                by_pair[task.pair_id].append(case)
                trajectories.append({
                    "task_id": task.task_id,
                    "scenario": "ambiguous_source",
                    "trajectory": 0,
                    "turns": 1,
                    "sampled_tokens": 1,
                    "logical_model_input_tokens": 1,
                })
            dyads = []
            for pair_id, members in sorted(by_pair.items()):
                first = members[0]
                dyads.append({
                    "pair_id": pair_id,
                    "scenario": "ambiguous_source",
                    "family": first["family"],
                    "evidence_channel": first["evidence_channel"],
                    "evidence_path_regime": next(
                        task.evidence_path_regime for task in tasks if task.pair_id == pair_id
                    ),
                    "acquisition_query_skin": first["acquisition_query_skin"],
                    "explicit_contract": False,
                    "paired_preverifier_success": False,
                })

            def dimension(key: str) -> dict:
                return {
                    name: {"paired_preverifier_success": 0.0}
                    for name in sorted({row[key] for row in dyads})
                }

            model_path = str(model.resolve())
            answer_cap = 1024
            per_call = cfg["evaluation"]["think_budget"] + answer_cap
            payload = {
                "schema_version": 1,
                "arm": "start",
                "model": model_path,
                "model_weight_sha256": common.sha256_file(model / "model.safetensors"),
                "model_config_sha256": common.sha256_file(model / "config.json"),
                "model_generation_config_sha256": common.sha256_file(
                    model / "generation_config.json"
                ),
                "merge_receipt_sha256": common.sha256_file(model / "merge_receipt.json"),
                **tokenizer_provenance,
                "config_sha256": common.sha256_file(config_path),
                "evaluator_sha256": common.sha256_file(EXP / "scripts" / "eval_repo_agent.py"),
                "repo_agent_sha256": common.sha256_file(EXP / "src" / "repo_agent.py"),
                "task_generator_sha256": common.sha256_file(EXP / "src" / "repo_tasks.py"),
                "block": "trained_calibration",
                "contract": "inferred",
                "scenario_set": "acquisition",
                "mode": "deep",
                "scaffold": False,
                "history_policy": common.HISTORY_POLICY,
                "think_budget": cfg["evaluation"]["think_budget"],
                "answer_max_tokens": answer_cap,
                "tasks_per_family": cfg["evaluation"]["blocks"]["trained_calibration"][
                    "tasks_per_family"
                ],
                "reserved_sampled_tokens_per_case": cfg["evaluation"]["controlled"][
                    "deep_turns"
                ] * per_call,
                "deep_reserved_sampled_tokens_per_case": cfg["evaluation"]["controlled"][
                    "deep_turns"
                ] * per_call,
                **manifests,
                "runner_summaries": [{
                    "runner_sha256": common.sha256_file(EXP / "src" / "vllm_runner.py"),
                    "model": model_path,
                    "engine": {
                        **cfg["engine"],
                        "adapter": None,
                        "model_override": model_path,
                    },
                    "sampling": {
                        "thinking": "budget",
                        "thinking_budget": cfg["evaluation"]["think_budget"],
                        "answer_max_tokens": answer_cap,
                        "greedy": True,
                        "run_seed": cfg["evaluation"]["deep_run_seed"],
                    },
                    "counts": {
                        "requests": len(trajectories),
                        "sampled_tokens": len(trajectories),
                        "logical_model_input_tokens": len(trajectories),
                    },
                }],
                "aggregate": {
                    "n_cases": len(cases),
                    "n_dyads": len(dyads),
                    "paired_preverifier_success": 0.0,
                    "success": 0.0,
                    "first_patch_full_correct": 0.0,
                    "unnecessary_evidence_before_first_patch": 0.0,
                    "invalid_action_rate_per_turn": 0.0,
                    "answer_cap_hit_rate_per_turn": 0.0,
                    "unusable_answer_cap_hit_rate_per_turn": 0.0,
                    "verified_given_success": 0.0,
                    "commit_given_verified": 0.0,
                    "mean_non_source_inspects_before_first_patch": 0.0,
                    "per_family": dimension("family"),
                    "per_channel": dimension("evidence_channel"),
                    "per_query_skin": dimension("acquisition_query_skin"),
                    "cases": cases,
                    "dyads": dyads,
                },
                "trajectories": trajectories,
            }
            receipt = root / "receipt.json"
            receipt.write_text(json.dumps(payload))
            common.validate_behavior_receipt(
                receipt,
                cfg,
                block="trained_calibration",
                contract="inferred",
                scenario_set="acquisition",
                mode="deep",
                arm="start",
                scaffold=False,
            )
            chat_template = model / "chat_template.jinja"
            original_template = chat_template.read_text()
            chat_template.write_text("{{ messages }}\nchanged\n")
            with self.assertRaisesRegex(SystemExit, "tokenizer provenance drift"):
                common.validate_behavior_receipt(
                    receipt,
                    cfg,
                    block="trained_calibration",
                    contract="inferred",
                    scenario_set="acquisition",
                    mode="deep",
                    arm="start",
                    scaffold=False,
                )
            chat_template.write_text(original_template)
            payload["tasks_per_family"] = 2
            receipt.write_text(json.dumps(payload))
            with self.assertRaisesRegex(SystemExit, "tasks-per-family override"):
                common.validate_behavior_receipt(
                    receipt,
                    cfg,
                    block="trained_calibration",
                    contract="inferred",
                    scenario_set="acquisition",
                    mode="deep",
                    arm="start",
                    scaffold=False,
                )


if __name__ == "__main__":
    unittest.main()
