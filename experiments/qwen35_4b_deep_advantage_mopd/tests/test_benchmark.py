from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
SCRIPT = EXP / "scripts" / "analyze_benchmark.py"
BENCH = EXP / "scripts" / "bench.py"
AUTHORIZER = EXP / "scripts" / "authorize_benchmark.py"
CONFIRMATION_ANALYZER = EXP / "scripts" / "analyze_confirmation.py"
CONFIRMATION_EVALUATOR = EXP / "scripts" / "eval_policy.py"
CONTROL_REMATCH = EXP / "src" / "control_rematch.py"
GATEWAY = REPO / "scripts" / "run_benchmark_aggregate.py"
MENAGERIE = REPO / "benchmarks" / "menagerie" / "run.py"
CONFIG = EXP / "configs" / "default.yaml"
PREREGISTRATION = EXP / "runs" / "preregistration_receipt.json"
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

from bench import (  # noqa: E402
    PUBLIC_FAMILY_KEYS,
    benchmark_source_inventory,
    model_provenance,
)
import authorize_benchmark  # noqa: E402
from authorize_benchmark import _audit_integration, _audit_merge_receipt  # noqa: E402
from io_utils import (  # noqa: E402
    confirmation_evaluator_source_inventory,
    sha256_file,
)


class BenchmarkAnalysisTests(unittest.TestCase):
    def _model(self, root: Path) -> dict[str, str]:
        model = root / "model"
        model.mkdir()
        weights = model / "model.safetensors"
        weights.write_bytes(b"synthetic-test-weights")
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "tokenizer.json").write_text('{"synthetic":true}\n', encoding="utf-8")
        (model / "merge_receipt.json").write_text(
            json.dumps(
                {
                    "weight_files": [
                        {"name": weights.name, "sha256": sha256_file(weights)}
                    ]
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return model_provenance(model)

    def _run(
        self,
        *,
        inject_score_failure: bool,
        inject_provenance_failure: bool = False,
        inject_budget_failure: bool = False,
        inject_confirmation_artifact_failure: bool = False,
    ) -> subprocess.CompletedProcess:
        root = Path(self.temporary.name) / f"case_{self.case_index}"
        self.case_index += 1
        root.mkdir()
        provenance = self._model(root)
        manifest = root / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        raw_confirmation = root / "atom_rows.jsonl.gz"
        raw_confirmation.write_bytes(b"synthetic-raw-confirmation")
        confirmation = root / "confirmation.json"
        confirmation.write_text(
            json.dumps(
                {
                    "stage": "two_block_same_prefix_advantage_confirmation",
                    "config_sha256": sha256_file(CONFIG),
                    "manifest": str(manifest.resolve()),
                    "manifest_sha256": sha256_file(manifest),
                    "gate": {"passed": True},
                    "downstream_authorization": "benchmark_cli",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        integration = root / "integration.json"
        integration.write_text('{"gate":{"passed":true}}\n', encoding="utf-8")
        controls = root / "controls.json"
        controls.write_text('{"gate":{"passed":true}}\n', encoding="utf-8")
        first = 56201
        tiers = {
            "quick": range(first, first + 3),
            "medium": range(first + 3, first + 11),
        }
        bindings = []
        for tier, seeds in tiers.items():
            for seed in seeds:
                for label in ("primary", "soup", "visible"):
                    bindings.append(
                        {"tier": tier, "seed": seed, "label": label, **provenance}
                    )
        authorization = root / "authorization.json"
        source_inventory = benchmark_source_inventory(MENAGERIE.parent)
        integration_receipts = [
            {"path": str(integration.resolve()), "sha256": sha256_file(integration)}
        ]
        controls_receipt = {
            "path": str(controls.resolve()),
            "sha256": sha256_file(controls),
        }
        confirmation_artifacts = [
            {"path": str(manifest.resolve()), "sha256": sha256_file(manifest)},
            {
                "path": str(raw_confirmation.resolve()),
                "sha256": sha256_file(raw_confirmation),
            },
        ]
        evidence_artifacts = sorted(
            [*integration_receipts, controls_receipt, *confirmation_artifacts],
            key=lambda row: row["path"],
        )
        authorization.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "stage": "benchmark_aggregate_authorization",
                    "config_sha256": sha256_file(CONFIG),
                    "preregistration_sha256": sha256_file(PREREGISTRATION),
                    "confirmation_sha256": sha256_file(confirmation),
                    "confirmation_manifest_sha256": sha256_file(manifest),
                    "integration_receipts": integration_receipts,
                    "controls_receipt": controls_receipt,
                    "confirmation_artifacts": confirmation_artifacts,
                    "evidence_artifacts": evidence_artifacts,
                    "aggregate_gateway_sha256": sha256_file(GATEWAY),
                    "benchmark_runner_sha256": sha256_file(MENAGERIE),
                    "benchmark_source_inventory_sha256": source_inventory["sha256"],
                    "benchmark_source_file_count": source_inventory["file_count"],
                    "bench_sha256": sha256_file(BENCH),
                    "analyzer_sha256": sha256_file(SCRIPT),
                    "confirmation_analyzer_sha256": sha256_file(
                        CONFIRMATION_ANALYZER
                    ),
                    "confirmation_evaluator_sha256": sha256_file(
                        CONFIRMATION_EVALUATOR
                    ),
                    "confirmation_evaluator_source_inventory_sha256": (
                        confirmation_evaluator_source_inventory()["sha256"]
                    ),
                    "confirmation_evaluator_source_file_count": (
                        confirmation_evaluator_source_inventory()["file_count"]
                    ),
                    "control_rematch_sha256": sha256_file(CONTROL_REMATCH),
                    "authorizer_sha256": sha256_file(AUTHORIZER),
                    "backend": "qwen_vllm",
                    "events": bindings,
                    "gate": {"passed": True},
                    "downstream_authorization": "aggregate_only_benchmark_cli",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        authorization_sha256 = sha256_file(authorization)
        if inject_confirmation_artifact_failure:
            raw_confirmation.unlink()
        events = []
        for binding in bindings:
            score = {"primary": 0.6, "soup": 0.5, "visible": 0.4}[
                binding["label"]
            ]
            if (
                inject_score_failure
                and binding["tier"] == "quick"
                and binding["seed"] == first
                and binding["label"] == "primary"
            ):
                score = 0.3
            event_authorization = authorization_sha256
            if inject_provenance_failure and not events:
                event_authorization = "0" * 64
            path = root / (
                f"{binding['tier']}-{binding['seed']}-{binding['label']}.json"
            )
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "stage": "aggregate_only_menagerie_event",
                        "config": str(CONFIG.resolve()),
                        "config_sha256": sha256_file(CONFIG),
                        "authorization": str(authorization.resolve()),
                        "authorization_sha256": event_authorization,
                        "confirmation_sha256": sha256_file(confirmation),
                        "tier": binding["tier"],
                        "seed": binding["seed"],
                        "label": binding["label"],
                        "backend": "qwen_vllm",
                        "model": binding["model"],
                        "model_merge_receipt_sha256": binding[
                            "model_merge_receipt_sha256"
                        ],
                        "model_weight_inventory_sha256": binding[
                            "model_weight_inventory_sha256"
                        ],
                        "model_config_sha256": binding["model_config_sha256"],
                        "model_inference_inventory_sha256": binding[
                            "model_inference_inventory_sha256"
                        ],
                        "aggregate_gateway_sha256": sha256_file(GATEWAY),
                        "benchmark_runner_sha256": sha256_file(MENAGERIE),
                        "benchmark_source_inventory_sha256": source_inventory[
                            "sha256"
                        ],
                        "benchmark_source_file_count": source_inventory["file_count"],
                        "aggregate": score,
                        "per_family": {
                            family: score for family in PUBLIC_FAMILY_KEYS
                        },
                        "within_budget": not (
                            inject_budget_failure and not events
                        ),
                        "wall_seconds": 1.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            events.append(path)
        command = [
            sys.executable,
            str(SCRIPT),
            "--authorization",
            str(authorization),
            "--confirmation",
            str(confirmation),
        ]
        for path in events:
            command.extend(("--event", str(path)))
        command.extend(("--out", str(root / "analysis.json")))
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.case_index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def test_model_provenance_binds_nonweight_inference_files(self):
        root = Path(self.temporary.name) / "model_inventory"
        root.mkdir()
        before = self._model(root)
        tokenizer = root / "model" / "tokenizer.json"
        tokenizer.write_text('{"synthetic":false}\n', encoding="utf-8")
        after = model_provenance(root / "model")
        self.assertEqual(
            before["model_weight_inventory_sha256"],
            after["model_weight_inventory_sha256"],
        )
        self.assertNotEqual(
            before["model_inference_inventory_sha256"],
            after["model_inference_inventory_sha256"],
        )

    def test_round_merge_audit_hashes_exact_intermediate_weights(self):
        root = Path(self.temporary.name) / "round_merge"
        base, adapter, merged = root / "base", root / "adapter", root / "merged"
        for path in (base, adapter, merged):
            path.mkdir(parents=True)
        (base / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
        adapter_config = adapter / "adapter_config.json"
        adapter_weights = adapter / "adapter_model.safetensors"
        adapter_config.write_text("{}\n", encoding="utf-8")
        adapter_weights.write_bytes(b"adapter")
        model_weights = merged / "model.safetensors"
        model_weights.write_bytes(b"before")
        receipt = merged / "merge_receipt.json"
        receipt.write_text(
            json.dumps(
                {
                    "method": "explicit_composite_lora_merge",
                    "base_model": str(base.resolve()),
                    "adapter": str(adapter.resolve()),
                    "adapter_config_sha256": sha256_file(adapter_config),
                    "adapter_weights_sha256": sha256_file(adapter_weights),
                    "applied_lora_modules": 1,
                    "nonzero_lora_modules": 1,
                    "weight_files": [
                        {
                            "name": model_weights.name,
                            "sha256": sha256_file(model_weights),
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        self.assertEqual(
            _audit_merge_receipt(merged, base, adapter), sha256_file(receipt)
        )
        model_weights.write_bytes(b"after")
        with self.assertRaises(ValueError):
            _audit_merge_receipt(merged, base, adapter)

    def test_replicate_audit_reuses_only_primary_round_zero_online_data(self):
        root = Path(self.temporary.name) / "shared_round_zero"
        experiment = root / "experiment"
        artifacts = root / "artifacts"
        soup = root / "soup"
        config_path = experiment / "configs" / "default.yaml"
        receipt_path = experiment / "runs" / "integration" / "seed_43.json"
        config_path.parent.mkdir(parents=True)
        receipt_path.parent.mkdir(parents=True)
        soup.mkdir(parents=True)
        config_path.write_text("synthetic: true\n", encoding="utf-8")
        (soup / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
        config = {
            "model": {
                "artifacts_root": str(artifacts),
                "student_checkpoint": str(soup),
            },
            "seeds": {"integration_training": [42, 43, 44]},
            "mopd": {"rounds": 4, "updates_per_round": 20},
        }

        rounds = []
        expected_target_caches = []
        for round_index, data_seed in enumerate((42, 43, 43, 43)):
            data_root = (
                artifacts
                / "online"
                / "primary"
                / f"seed_{data_seed}"
                / f"round_{round_index}"
            )
            data_root.mkdir(parents=True)
            manifest = data_root / "training_round.json"
            target_cache = data_root / "all_policy_targets.pt"
            manifest.write_text(
                json.dumps({"round": round_index, "data_seed": data_seed}) + "\n",
                encoding="utf-8",
            )
            target_cache.write_bytes(f"cache-{round_index}".encode())
            expected_target_caches.append(target_cache.resolve())

            adapter = (
                artifacts
                / "adapters"
                / "primary"
                / "seed_43"
                / f"round_{round_index}"
            )
            merged = (
                artifacts
                / "merged"
                / "primary"
                / "seed_43"
                / f"round_{round_index}"
            )
            adapter.mkdir(parents=True)
            merged.mkdir(parents=True)
            training_receipt = adapter / "training_receipt.json"
            merge_receipt = merged / "merge_receipt.json"
            training_receipt.write_text("{}\n", encoding="utf-8")
            merge_receipt.write_text("{}\n", encoding="utf-8")
            round_gate = {"passed": True, "completed_all_updates": True}
            rounds.append(
                {
                    "round": round_index,
                    "round_manifest": str(manifest.resolve()),
                    "round_manifest_sha256": sha256_file(manifest),
                    "target_cache": str(target_cache.resolve()),
                    "target_cache_sha256": sha256_file(target_cache),
                    "training_receipt": str(training_receipt.resolve()),
                    "training_receipt_sha256": sha256_file(training_receipt),
                    "round_gate": round_gate,
                    "merged": str(merged.resolve()),
                    "merge_receipt_sha256": sha256_file(merge_receipt),
                }
            )

        primary = artifacts / "merged" / "primary" / "seed_43" / "round_3"
        receipt_path.write_text(
            json.dumps(
                {
                    "stage": "four_round_deep_advantage_routed_mopd",
                    "config_sha256": sha256_file(config_path),
                    "seed": 43,
                    "rounds": rounds,
                    "completed_rounds": 4,
                    "gate": {"passed": True},
                    "final_model": str(primary.resolve()),
                }
            )
            + "\n",
            encoding="utf-8",
        )

        with (
            mock.patch.object(authorize_benchmark, "EXP", experiment),
            mock.patch.object(
                authorize_benchmark,
                "resolve_repo_path",
                side_effect=lambda value: Path(value).resolve(),
            ),
            mock.patch.object(
                authorize_benchmark,
                "_audit_mopd_training_receipt",
            ) as audit_training,
            mock.patch.object(
                authorize_benchmark,
                "_audit_merge_receipt",
                side_effect=lambda merged, _base, _adapter: sha256_file(
                    merged / "merge_receipt.json"
                ),
            ),
        ):
            artifact = _audit_integration(config, config_path, 43, primary)

        self.assertEqual(artifact["path"], str(receipt_path.resolve()))
        self.assertEqual(audit_training.call_count, 4)
        self.assertEqual(
            [call.kwargs["target_cache"] for call in audit_training.call_args_list],
            expected_target_caches,
        )

    def test_all_paired_events_must_be_positive(self):
        passed = self._run(inject_score_failure=False)
        self.assertEqual(passed.returncode, 0, passed.stderr + passed.stdout)
        failed = self._run(inject_score_failure=True)
        self.assertEqual(failed.returncode, 4, failed.stderr + failed.stdout)

    def test_event_provenance_mismatch_fails_closed(self):
        failed = self._run(
            inject_score_failure=False, inject_provenance_failure=True
        )
        self.assertEqual(failed.returncode, 4, failed.stderr + failed.stdout)

    def test_every_event_must_be_within_budget(self):
        failed = self._run(
            inject_score_failure=False,
            inject_budget_failure=True,
        )
        self.assertEqual(failed.returncode, 4, failed.stderr + failed.stdout)

    def test_confirmation_raw_mutation_stales_benchmark_authorization(self):
        failed = self._run(
            inject_score_failure=False,
            inject_confirmation_artifact_failure=True,
        )
        self.assertNotEqual(failed.returncode, 0, failed.stderr + failed.stdout)


if __name__ == "__main__":
    unittest.main()
