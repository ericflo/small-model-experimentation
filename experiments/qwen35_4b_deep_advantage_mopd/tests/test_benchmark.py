from __future__ import annotations

import contextlib
import io
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
CONTROL_RECEIPTS = EXP / "src" / "control_receipts.py"
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
import analyze_benchmark  # noqa: E402
import authorize_benchmark  # noqa: E402
import bench  # noqa: E402
from authorize_benchmark import _audit_integration, _audit_merge_receipt  # noqa: E402
from io_utils import (  # noqa: E402
    canonical_hash,
    confirmation_evaluator_source_inventory,
    sha256_file,
)


class BenchmarkAnalysisTests(unittest.TestCase):
    def test_authorizer_code_provenance_executes_without_manifest_scope(self):
        provenance = authorize_benchmark._code_provenance()
        self.assertEqual(
            provenance["authorizer_sha256"], sha256_file(AUTHORIZER)
        )
        self.assertEqual(
            provenance["benchmark_runner_sha256"], sha256_file(MENAGERIE)
        )
        self.assertEqual(
            provenance["confirmation_evaluator_sha256"],
            sha256_file(CONFIRMATION_EVALUATOR),
        )

    def test_authorization_seal_is_no_clobber_and_exact_rerun_is_read_only(self):
        experiment = Path(self.temporary.name) / "publication-experiment"
        output = experiment / "analysis" / "benchmark_authorization.json"
        payload = {
            "stage": "authorization",
            "gate": {"passed": True},
            "nested": {"zero": 0, "two": 2},
        }
        with mock.patch.object(authorize_benchmark, "EXP", experiment):
            authorize_benchmark._publish_authorization_no_clobber(output, payload)
            original_bytes = output.read_bytes()
            original_stat = output.stat()

            authorize_benchmark._publish_authorization_no_clobber(output, payload)
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(output.stat().st_mtime_ns, original_stat.st_mtime_ns)

            type_mutations = (
                {**payload, "nested": {"zero": False, "two": 2}},
                {**payload, "nested": {"zero": 0, "two": 2.0}},
            )
            for mutation in type_mutations:
                output.write_text(
                    json.dumps(mutation, sort_keys=True) + "\n", encoding="utf-8"
                )
                mutation_bytes = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    authorize_benchmark._publish_authorization_no_clobber(
                        output, payload
                    )
                self.assertEqual(output.read_bytes(), mutation_bytes)

            stale_bytes = b'{"winner":true}\n'
            output.write_bytes(stale_bytes)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                authorize_benchmark._publish_authorization_no_clobber(output, payload)
            self.assertEqual(output.read_bytes(), stale_bytes)

    def test_authorization_seal_race_never_overwrites_winner(self):
        experiment = Path(self.temporary.name) / "race-experiment"
        output = experiment / "analysis" / "benchmark_authorization.json"

        def lose_race(_source, target):
            Path(target).write_bytes(b"winner")
            raise FileExistsError

        with mock.patch.object(
            authorize_benchmark, "EXP", experiment
        ), mock.patch.object(
            authorize_benchmark.os, "link", side_effect=lose_race
        ), self.assertRaisesRegex(ValueError, "lost a race"):
            authorize_benchmark._publish_authorization_no_clobber(
                output, {"ours": True}
            )
        self.assertEqual(output.read_bytes(), b"winner")

    def test_authorization_seal_rejects_symlinked_canonical_root_before_io(self):
        root = Path(self.temporary.name) / "canonical-symlink"
        experiment = root / "experiment"
        experiment.mkdir(parents=True)
        real_analysis = root / "real-analysis"
        real_analysis.mkdir()
        (experiment / "analysis").symlink_to(real_analysis, target_is_directory=True)
        output = real_analysis / "benchmark_authorization.json"

        with mock.patch.object(
            authorize_benchmark, "EXP", experiment
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
            authorize_benchmark.tempfile, "NamedTemporaryFile"
        ) as temporary_file, mock.patch.object(
            authorize_benchmark.os, "link"
        ) as link, self.assertRaisesRegex(ValueError, "symlinked existing"):
            authorize_benchmark._publish_authorization_no_clobber(
                output, {"ours": True}
            )

        mkdir.assert_not_called()
        temporary_file.assert_not_called()
        link.assert_not_called()
        self.assertFalse(output.exists())

    def test_authorization_seal_rejects_symlinked_output_ancestor_before_io(self):
        experiment = Path(self.temporary.name) / "ancestor-symlink"
        analysis = experiment / "analysis"
        real_parent = analysis / "real-parent"
        real_parent.mkdir(parents=True)
        alias = analysis / "alias"
        alias.symlink_to(real_parent, target_is_directory=True)
        output = alias / "benchmark_authorization.json"

        with mock.patch.object(
            authorize_benchmark, "EXP", experiment
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
            authorize_benchmark.tempfile, "NamedTemporaryFile"
        ) as temporary_file, mock.patch.object(
            authorize_benchmark.os, "link"
        ) as link, self.assertRaisesRegex(ValueError, "symlinked existing"):
            authorize_benchmark._publish_authorization_no_clobber(
                output, {"ours": True}
            )

        mkdir.assert_not_called()
        temporary_file.assert_not_called()
        link.assert_not_called()
        self.assertFalse((real_parent / output.name).exists())

    def test_authorization_seal_rejects_parent_outside_analysis_root(self):
        root = Path(self.temporary.name) / "outside-root"
        experiment = root / "experiment"
        (experiment / "analysis").mkdir(parents=True)
        outside = root / "outside"
        outside.mkdir()
        output = outside / "benchmark_authorization.json"

        with mock.patch.object(
            authorize_benchmark, "EXP", experiment
        ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
            authorize_benchmark.tempfile, "NamedTemporaryFile"
        ) as temporary_file, mock.patch.object(
            authorize_benchmark.os, "link"
        ) as link, self.assertRaisesRegex(ValueError, "outside the experiment"):
            authorize_benchmark._publish_authorization_no_clobber(
                output, {"ours": True}
            )

        mkdir.assert_not_called()
        temporary_file.assert_not_called()
        link.assert_not_called()
        self.assertFalse(output.exists())

    def test_event_path_is_exactly_bound_to_tier_seed_and_label(self):
        experiment = Path(self.temporary.name) / "event-path-experiment"
        wrong = (
            experiment
            / "runs"
            / "benchmark"
            / "quick"
            / "seed_56201"
            / "soup.json"
        )
        with mock.patch.object(bench, "EXP", experiment), self.assertRaisesRegex(
            ValueError, "not the exact canonical path"
        ):
            bench._benchmark_event_publication_start(
                wrong, tier="quick", seed=56201, label="primary"
            )

    def test_event_path_rejects_symlinked_canonical_root_before_io(self):
        root = Path(self.temporary.name) / "event-root-symlink"
        experiment = root / "experiment"
        runs = experiment / "runs"
        runs.mkdir(parents=True)
        real_benchmark = root / "real-benchmark"
        real_benchmark.mkdir()
        (runs / "benchmark").symlink_to(
            real_benchmark, target_is_directory=True
        )
        output = (
            runs
            / "benchmark"
            / "quick"
            / "seed_56201"
            / "primary.json"
        )
        with mock.patch.object(bench, "EXP", experiment), self.assertRaisesRegex(
            ValueError, "symlinked existing"
        ):
            bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )

    def test_event_path_rejects_symlinked_ancestor_before_io(self):
        root = Path(self.temporary.name) / "event-ancestor-symlink"
        experiment = root / "experiment"
        benchmark = experiment / "runs" / "benchmark"
        benchmark.mkdir(parents=True)
        real_quick = root / "real-quick"
        real_quick.mkdir()
        (benchmark / "quick").symlink_to(real_quick, target_is_directory=True)
        output = benchmark / "quick" / "seed_56201" / "primary.json"
        with mock.patch.object(bench, "EXP", experiment), self.assertRaisesRegex(
            ValueError, "symlinked existing"
        ):
            bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )

    def test_event_path_rejects_symlinked_leaf_before_io(self):
        root = Path(self.temporary.name) / "event-leaf-symlink"
        experiment = root / "experiment"
        output = (
            experiment
            / "runs"
            / "benchmark"
            / "quick"
            / "seed_56201"
            / "primary.json"
        )
        output.parent.mkdir(parents=True)
        target = root / "target.json"
        target.write_text("{}\n", encoding="utf-8")
        output.symlink_to(target)
        with mock.patch.object(bench, "EXP", experiment), self.assertRaisesRegex(
            ValueError, "symlinked existing"
        ):
            bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )

    def test_event_seal_is_no_clobber_and_exact_rerun_is_read_only(self):
        experiment = Path(self.temporary.name) / "event-seal-experiment"
        output = (
            experiment
            / "runs"
            / "benchmark"
            / "quick"
            / "seed_56201"
            / "primary.json"
        )
        payload = {"stage": "aggregate_only_menagerie_event", "score": 0.75}
        with mock.patch.object(bench, "EXP", experiment):
            output, existed = bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )
            self.assertFalse(existed)
            self.assertTrue(
                bench._publish_benchmark_event_no_clobber(
                    output,
                    payload,
                    tier="quick",
                    seed=56201,
                    label="primary",
                    existed_at_start=existed,
                )
            )
            original_bytes = output.read_bytes()
            original_mtime = output.stat().st_mtime_ns

            output, existed = bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )
            self.assertTrue(existed)
            self.assertFalse(
                bench._publish_benchmark_event_no_clobber(
                    output,
                    payload,
                    tier="quick",
                    seed=56201,
                    label="primary",
                    existed_at_start=existed,
                )
            )
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(output.stat().st_mtime_ns, original_mtime)

            stale_bytes = b'{"stale":true}\n'
            output.write_bytes(stale_bytes)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite stale"):
                bench._publish_benchmark_event_no_clobber(
                    output,
                    payload,
                    tier="quick",
                    seed=56201,
                    label="primary",
                    existed_at_start=True,
                )
            self.assertEqual(output.read_bytes(), stale_bytes)

    def test_event_seal_race_never_overwrites_winner(self):
        experiment = Path(self.temporary.name) / "event-race-experiment"
        output = (
            experiment
            / "runs"
            / "benchmark"
            / "quick"
            / "seed_56201"
            / "primary.json"
        )

        def lose_race(_source, target):
            Path(target).write_bytes(b"winner")
            raise FileExistsError

        with mock.patch.object(bench, "EXP", experiment):
            output, existed = bench._benchmark_event_publication_start(
                output, tier="quick", seed=56201, label="primary"
            )
            with mock.patch.object(
                bench.os, "link", side_effect=lose_race
            ), self.assertRaisesRegex(ValueError, "lost a race"):
                bench._publish_benchmark_event_no_clobber(
                    output,
                    {"ours": True},
                    tier="quick",
                    seed=56201,
                    label="primary",
                    existed_at_start=existed,
                )
        self.assertEqual(output.read_bytes(), b"winner")

    def test_event_seal_exact_rerun_comparison_preserves_json_types(self):
        cases = (
            ({"nested": [{"value": 0}]}, {"nested": [{"value": False}]}),
            ({"nested": [{"value": 2}]}, {"nested": [{"value": 2.0}]}),
        )
        for index, (sealed, mutated) in enumerate(cases):
            with self.subTest(index=index):
                experiment = (
                    Path(self.temporary.name) / f"event-json-types-{index}"
                )
                output = (
                    experiment
                    / "runs"
                    / "benchmark"
                    / "quick"
                    / "seed_56201"
                    / "primary.json"
                )
                with mock.patch.object(bench, "EXP", experiment):
                    output, existed = bench._benchmark_event_publication_start(
                        output, tier="quick", seed=56201, label="primary"
                    )
                    bench._publish_benchmark_event_no_clobber(
                        output,
                        sealed,
                        tier="quick",
                        seed=56201,
                        label="primary",
                        existed_at_start=existed,
                    )
                    original_bytes = output.read_bytes()
                    with self.assertRaisesRegex(
                        ValueError, "refusing to overwrite stale"
                    ):
                        bench._publish_benchmark_event_no_clobber(
                            output,
                            mutated,
                            tier="quick",
                            seed=56201,
                            label="primary",
                            existed_at_start=True,
                        )
                    self.assertEqual(output.read_bytes(), original_bytes)

    def test_analysis_seal_is_no_clobber_and_exact_rerun_is_read_only(self):
        experiment = Path(self.temporary.name) / "analysis-seal-experiment"
        output = experiment / "analysis" / "benchmark.json"
        result = {"stage": "aggregate_only_menagerie_confirmation", "gate": True}
        with mock.patch.object(analyze_benchmark, "EXP", experiment):
            output, existed = analyze_benchmark._analysis_publication_start(output)
            self.assertTrue(
                analyze_benchmark._publish_analysis_no_clobber(
                    output, result, existed_at_start=existed
                )
            )
            original_bytes = output.read_bytes()
            original_mtime = output.stat().st_mtime_ns

            output, existed = analyze_benchmark._analysis_publication_start(output)
            self.assertFalse(
                analyze_benchmark._publish_analysis_no_clobber(
                    output, result, existed_at_start=existed
                )
            )
            self.assertEqual(output.read_bytes(), original_bytes)
            self.assertEqual(output.stat().st_mtime_ns, original_mtime)

            stale_bytes = b'{"stale":true}\n'
            output.write_bytes(stale_bytes)
            with self.assertRaisesRegex(ValueError, "refusing to overwrite stale"):
                analyze_benchmark._publish_analysis_no_clobber(
                    output, result, existed_at_start=True
                )
            self.assertEqual(output.read_bytes(), stale_bytes)

    def test_analysis_seal_race_never_overwrites_winner(self):
        experiment = Path(self.temporary.name) / "analysis-race-experiment"
        output = experiment / "analysis" / "benchmark.json"

        def lose_race(_source, target):
            Path(target).write_bytes(b"winner")
            raise FileExistsError

        with mock.patch.object(analyze_benchmark, "EXP", experiment):
            output, existed = analyze_benchmark._analysis_publication_start(output)
            with mock.patch.object(
                bench.os, "link", side_effect=lose_race
            ), self.assertRaisesRegex(ValueError, "lost a race"):
                analyze_benchmark._publish_analysis_no_clobber(
                    output, {"ours": True}, existed_at_start=existed
                )
        self.assertEqual(output.read_bytes(), b"winner")

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
        weight_inventory = [
            {"name": weights.name, "sha256": sha256_file(weights)}
        ]
        inference_inventory = [
            {
                "path": path.relative_to(model).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(model.iterdir())
        ]
        return {
            "model": str(model.resolve()),
            "model_merge_receipt_sha256": sha256_file(
                model / "merge_receipt.json"
            ),
            "model_weight_inventory_sha256": canonical_hash(weight_inventory),
            "model_config_sha256": sha256_file(model / "config.json"),
            "model_inference_inventory_sha256": canonical_hash(
                inference_inventory
            ),
        }

    def _run(
        self,
        *,
        inject_score_failure: bool,
        inject_provenance_failure: bool = False,
        inject_budget_failure: bool = False,
        inject_confirmation_artifact_failure: bool = False,
        inject_wrong_event_path: bool = False,
        inject_extra_benchmark_entry: str | None = None,
        inject_extra_confirmation_entry: str | None = None,
        inject_event_mutation_after_snapshot: bool = False,
        inject_post_authorization_benchmark_extra: bool = False,
        inject_post_authorization_confirmation_extra: str | None = None,
    ) -> subprocess.CompletedProcess:
        root = Path(self.temporary.name) / f"case_{self.case_index}"
        self.case_index += 1
        root.mkdir()
        experiment = root / "experiment"
        experiment.mkdir()
        provenance = self._model(root)
        manifest = root / "manifest.json"
        manifest.write_text("{}\n", encoding="utf-8")
        raw_confirmation = root / "atom_rows.jsonl.gz"
        raw_confirmation.write_bytes(b"synthetic-raw-confirmation")
        confirmation_campaign = root / "confirmation-campaign"
        confirmation_raw_root = root / "confirmation-raw"
        confirmation_campaign.mkdir()
        confirmation_raw_root.mkdir()
        admission_path = confirmation_campaign / "ADMISSION.json"
        admission_path.write_text(
            '{"blocks":[17],"arms":{"arm":{"model":"m"}}}\n',
            encoding="utf-8",
        )
        confirmation = root / "confirmation.json"
        confirmation.write_text(
            json.dumps(
                {
                    "stage": "two_block_same_prefix_advantage_confirmation",
                    "config_sha256": sha256_file(CONFIG),
                    "manifest": str(manifest.resolve()),
                    "manifest_sha256": sha256_file(manifest),
                    "confirmation_admission": {
                        "path": str(admission_path.resolve()),
                        "sha256": sha256_file(admission_path),
                    },
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
                    "control_receipts_sha256": sha256_file(CONTROL_RECEIPTS),
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
        for binding_index, binding in enumerate(bindings):
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
            path = (
                experiment
                / "runs"
                / "benchmark"
                / binding["tier"]
                / f"seed_{binding['seed']}"
                / f"{binding['label']}.json"
            )
            if inject_wrong_event_path and binding_index == 0:
                path = root / "wrong-event-path.json"
            path.parent.mkdir(parents=True, exist_ok=True)
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

        benchmark_root = experiment / "runs" / "benchmark"

        def add_benchmark_extra(kind: str) -> Path:
            if kind == "file":
                extra = benchmark_root / "retry-result.json"
                extra.write_text("{}\n", encoding="utf-8")
            elif kind == "directory":
                extra = benchmark_root / "retry"
                extra.mkdir()
            elif kind == "symlink":
                target = root / "retry-target.json"
                target.write_text("{}\n", encoding="utf-8")
                extra = benchmark_root / "retry-result.json"
                extra.symlink_to(target)
            else:
                raise AssertionError(f"unknown benchmark extra kind: {kind}")
            return extra

        if inject_extra_benchmark_entry is not None:
            add_benchmark_extra(inject_extra_benchmark_entry)

        def add_confirmation_extra(kind: str) -> Path:
            if kind == "raw_file":
                extra = confirmation_raw_root / "rogue-extra.json"
                extra.write_text("{}\n", encoding="utf-8")
            elif kind == "raw_directory":
                extra = confirmation_raw_root / "rogue-extra"
                extra.mkdir()
            elif kind == "raw_symlink":
                target = root / "rogue-confirmation-target.json"
                target.write_text("{}\n", encoding="utf-8")
                extra = confirmation_raw_root / "rogue-extra.json"
                extra.symlink_to(target)
            elif kind == "score_file":
                extra = confirmation_campaign / "rogue-score.json"
                extra.write_text("{}\n", encoding="utf-8")
            else:
                raise AssertionError(f"unknown confirmation extra kind: {kind}")
            return extra

        if inject_extra_confirmation_entry is not None:
            add_confirmation_extra(inject_extra_confirmation_entry)

        original_authorization = analyze_benchmark._authorization
        original_event_snapshot = analyze_benchmark._load_event_snapshot
        event_snapshot_mutated = False

        def authorization_then_mutate(*args, **kwargs):
            result = original_authorization(*args, **kwargs)
            if inject_post_authorization_benchmark_extra:
                add_benchmark_extra("file")
            if inject_post_authorization_confirmation_extra is not None:
                add_confirmation_extra(
                    inject_post_authorization_confirmation_extra
                )
            return result

        def snapshot_then_mutate(path):
            nonlocal event_snapshot_mutated
            result = original_event_snapshot(path)
            if inject_event_mutation_after_snapshot and not event_snapshot_mutated:
                Path(path).write_bytes(Path(path).read_bytes() + b" \n")
                event_snapshot_mutated = True
            return result

        def validate_synthetic_confirmation_campaign(
            observed_admission,
            *,
            raw_root,
            terminal,
            require_manifest,
        ):
            if (
                Path(observed_admission) != admission_path
                or Path(raw_root) != confirmation_raw_root
                or terminal is not True
                or require_manifest is not True
            ):
                raise ValueError("terminal confirmation validation arguments changed")
            visible_extras = {
                path.name for path in confirmation_campaign.iterdir()
            } - {"ADMISSION.json"}
            if list(confirmation_raw_root.iterdir()) or visible_extras:
                raise ValueError(
                    "confirmation campaign contains unregistered entries"
                )
            return {"block_0/arm": "COMMITTED"}

        command = [
            "--authorization",
            str(authorization),
            "--confirmation",
            str(confirmation),
        ]
        for path in events:
            command.extend(("--event", str(path)))
        command.extend(
            ("--out", str(experiment / "analysis" / "benchmark.json"))
        )
        with (
            mock.patch.object(bench, "EXP", experiment),
            mock.patch.object(analyze_benchmark, "EXP", experiment),
            mock.patch.object(
                analyze_benchmark,
                "model_provenance",
                return_value=provenance,
            ),
            mock.patch.object(
                analyze_benchmark,
                "_authorization",
                side_effect=authorization_then_mutate,
            ),
            mock.patch.object(
                analyze_benchmark,
                "_load_event_snapshot",
                side_effect=snapshot_then_mutate,
            ),
            mock.patch.object(
                analyze_benchmark,
                "configured_confirmation_raw_root",
                return_value=confirmation_raw_root,
            ),
            mock.patch.object(
                analyze_benchmark,
                "validate_confirmation_campaign_tree",
                side_effect=validate_synthetic_confirmation_campaign,
            ),
        ):
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    returncode = analyze_benchmark.main(command)
            except (OSError, TypeError, ValueError) as exc:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout=stdout.getvalue(),
                    stderr=f"{type(exc).__name__}: {exc}",
                )
        return subprocess.CompletedProcess(
            command, returncode, stdout=stdout.getvalue(), stderr=""
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.case_index = 0

    def tearDown(self):
        self.temporary.cleanup()

    def test_model_provenance_rejects_incomplete_checkpoint_shapes(self):
        root = Path(self.temporary.name) / "model_inventory"
        root.mkdir()
        self._model(root)
        with self.assertRaisesRegex(ValueError, "tokenizer configuration"):
            model_provenance(root / "model")

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
        with mock.patch.object(
            authorize_benchmark,
            "validate_model_checkpoint",
            return_value={"model_merge_receipt_sha256": sha256_file(receipt)},
        ):
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

    def test_analysis_rejects_correct_event_content_at_wrong_path(self):
        failed = self._run(
            inject_score_failure=False,
            inject_wrong_event_path=True,
        )
        self.assertNotEqual(failed.returncode, 0, failed.stderr + failed.stdout)
        self.assertIn("not an exact canonical path", failed.stderr)

    def test_analysis_rejects_unregistered_benchmark_tree_entries(self):
        for kind in ("file", "directory", "symlink"):
            with self.subTest(kind=kind):
                failed = self._run(
                    inject_score_failure=False,
                    inject_extra_benchmark_entry=kind,
                )
                self.assertNotEqual(
                    failed.returncode, 0, failed.stderr + failed.stdout
                )
                self.assertIn("benchmark campaign", failed.stderr)

    def test_analysis_rejects_benchmark_extra_added_after_authorization(self):
        failed = self._run(
            inject_score_failure=False,
            inject_post_authorization_benchmark_extra=True,
        )
        self.assertNotEqual(failed.returncode, 0, failed.stderr + failed.stdout)
        self.assertIn("benchmark campaign", failed.stderr)

    def test_analysis_rejects_event_mutated_after_snapshot(self):
        failed = self._run(
            inject_score_failure=False,
            inject_event_mutation_after_snapshot=True,
        )
        self.assertNotEqual(failed.returncode, 0, failed.stderr + failed.stdout)
        self.assertIn("event changed during analysis", failed.stderr)

    def test_analysis_rejects_confirmation_extra_added_after_authorization(self):
        for kind in ("raw_file", "raw_symlink", "score_file"):
            with self.subTest(kind=kind):
                failed = self._run(
                    inject_score_failure=False,
                    inject_post_authorization_confirmation_extra=kind,
                )
                self.assertNotEqual(
                    failed.returncode, 0, failed.stderr + failed.stdout
                )
                self.assertIn("confirmation campaign", failed.stderr)

    def test_analysis_rejects_preexisting_confirmation_campaign_extra(self):
        for kind in ("raw_file", "raw_directory", "raw_symlink", "score_file"):
            with self.subTest(kind=kind):
                failed = self._run(
                    inject_score_failure=False,
                    inject_extra_confirmation_entry=kind,
                )
                self.assertNotEqual(
                    failed.returncode, 0, failed.stderr + failed.stdout
                )
                self.assertIn("confirmation campaign", failed.stderr)


if __name__ == "__main__":
    unittest.main()
