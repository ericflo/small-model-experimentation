from __future__ import annotations

import ast
import contextlib
import hashlib
import io
import importlib
import importlib.util
import json
import os
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class StaticContractTests(unittest.TestCase):
    @staticmethod
    def load_gpu_runner():
        """Import source-only runner tests without requiring the GPU venv."""

        try:
            return importlib.import_module("src.gpu_runner")
        except ModuleNotFoundError as exc:
            if exc.name != "transformers":
                raise

        transformers = types.ModuleType("transformers")
        transformers.__version__ = "5.13.0"

        class ModelLoader:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise AssertionError("test loader must be patched")

        class TokenizerLoader:
            @staticmethod
            def from_pretrained(*_args, **_kwargs):
                raise AssertionError("test loader must be patched")

        transformers.AutoModelForCausalLM = ModelLoader
        transformers.AutoTokenizer = TokenizerLoader
        utils = types.ModuleType("transformers.utils")
        hub = types.ModuleType("transformers.utils.hub")
        hub.cached_file = lambda *_args, **_kwargs: None

        def extract_commit_hash(path, _commit_hash):
            candidate = Path(path)
            if candidate.parent.parent.name == "snapshots":
                return candidate.parent.name
            return None

        hub.extract_commit_hash = extract_commit_hash
        import_utils = types.ModuleType("transformers.utils.import_utils")
        import_utils.is_causal_conv1d_available = lambda: False
        import_utils.is_flash_linear_attention_available = lambda: False
        stubs = {
            "transformers": transformers,
            "transformers.utils": utils,
            "transformers.utils.hub": hub,
            "transformers.utils.import_utils": import_utils,
        }
        with mock.patch.dict(sys.modules, stubs):
            return importlib.import_module("src.gpu_runner")

    @staticmethod
    def load_cli():
        spec = importlib.util.spec_from_file_location(
            "capacity_adjudication_cli", ROOT / "scripts" / "run.py"
        )
        if spec is None or spec.loader is None:
            raise RuntimeError("capacity-adjudication CLI cannot be imported")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        @contextlib.contextmanager
        def authorized_source_snapshot():
            yield module.LOADED_SOURCE_CONTRACT_SHA256

        module.authorized_source_execution_snapshot = authorized_source_snapshot
        return module

    @staticmethod
    def registered_cli_cells(experiment: Path, config: dict) -> list[dict]:
        seed = 7411
        steps = int(config["training"]["train_steps"])
        large = (experiment / config["paths"]["large_artifacts_dir"]).resolve()
        cells: list[dict] = [
            {
                "name": "cpu-smoke",
                "argv": ["--stage", "cpu-smoke"],
                "output": (experiment / "runs" / "cpu_smoke" / "receipt.json").resolve(),
                "authorizations": {"none"},
            },
            {
                "name": "design-boundary",
                "argv": ["--stage", "design-boundary"],
                "output": (experiment / config["paths"]["design_receipt"]).resolve(),
                "authorizations": {"none"},
            },
            {
                "name": "prepare-data",
                "argv": ["--stage", "prepare-data"],
                "output": (experiment / config["paths"]["data_dir"]).resolve(),
                "authorizations": {"none"},
            },
            {
                "name": "prepare-init",
                "argv": ["--stage", "prepare-init", "--seed", str(seed)],
                "output": large / f"initialization_seed{seed}.pt",
                "authorizations": {"none"},
            },
        ]
        lora_miss = {"lora_joint"}
        for stage, prefix in (
            ("model-smoke", "g0"),
            ("positive-control", "positive_control"),
        ):
            for capacity in ("lora", "fullrank"):
                cells.append(
                    {
                        "name": f"{stage}:{capacity}",
                        "argv": [
                            "--stage", stage, "--capacity", capacity,
                            "--seed", str(seed),
                        ],
                        "output": (
                            experiment / "runs" / "setup"
                            / f"{prefix}_{capacity}_seed{seed}.json"
                        ).resolve(),
                        "authorizations": (
                            {"none"} if capacity == "lora" else lora_miss
                        ),
                    }
                )
        training_authorizations = {
            ("lora", "joint"): {"none"},
            ("lora", "state_only"): lora_miss,
            ("fullrank", "joint"): lora_miss,
            ("fullrank", "state_only"): {"stage_b", "fullrank_joint"},
        }
        for capacity in ("lora", "fullrank"):
            for objective in ("joint", "state_only"):
                base = f"{capacity}_{objective}_seed{seed}"
                cells.append(
                    {
                        "name": f"train:{capacity}:{objective}",
                        "argv": [
                            "--stage", "train", "--capacity", capacity,
                            "--objective", objective, "--seed", str(seed),
                        ],
                        "output": (large / base).resolve(),
                        "authorizations": training_authorizations[(capacity, objective)],
                    }
                )
                for eval_set in (
                    ("trigger", "contrast") if objective == "joint" else ("trigger",)
                ):
                    cells.append(
                        {
                            "name": f"evaluate-state:{capacity}:{objective}:{eval_set}",
                            "argv": [
                                "--stage", "evaluate-state", "--capacity", capacity,
                                "--objective", objective, "--eval-set", eval_set,
                                "--seed", str(seed), "--checkpoint",
                                str(large / base / f"checkpoint_{steps:06d}"),
                            ],
                            "output": (
                                experiment / "runs" / f"{base}_{eval_set}"
                            ).resolve(),
                            "authorizations": (
                                {"stage_b"}
                                if eval_set == "contrast"
                                else training_authorizations[(capacity, objective)]
                            ),
                        }
                    )
        analysis_outputs = {
            "lora_joint": ("lora_joint_trigger.json", {"none"}),
            "lora_control": ("lora_control.json", lora_miss),
            "stage_b_seal": ("stage_b_seal.json", lora_miss),
            "fullrank_joint": ("fullrank_joint.json", {"stage_b"}),
            "fullrank_control": (
                "summary.json", {"stage_b", "fullrank_joint"}
            ),
        }
        for phase, (filename, authorizations) in analysis_outputs.items():
            cells.append(
                {
                    "name": f"analyze:{phase}",
                    "argv": ["--stage", "analyze", "--phase", phase],
                    "output": (experiment / "analysis" / filename).resolve(),
                    "authorizations": authorizations,
                }
            )
        return cells

    @staticmethod
    def fill_canonical_cli_inputs(module, args, config: dict) -> None:
        if args.seed is None:
            return
        large = (module.ROOT / config["paths"]["large_artifacts_dir"]).resolve()
        args.initialization_bundle = str(large / f"initialization_seed{args.seed}.pt")
        args.model_smoke_receipt = str(
            module.ROOT / "runs" / "setup"
            / f"g0_{args.capacity}_seed{args.seed}.json"
        )
        args.positive_control_receipt = str(
            module.ROOT / "runs" / "setup"
            / f"positive_control_{args.capacity}_seed{args.seed}.json"
        )

    @staticmethod
    def runtime_paths() -> list[Path]:
        return sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py")])

    def test_only_qwen35_4b_model_identifier_appears_in_runtime_or_configs(self) -> None:
        model_pattern = re.compile(r"Qwen/[A-Za-z0-9_.-]+")
        identifiers: set[str] = set()
        for path in [*self.runtime_paths(), *ROOT.glob("configs/*.yaml")]:
            identifiers.update(model_pattern.findall(path.read_text(encoding="utf-8")))
        self.assertEqual(identifiers, {"Qwen/Qwen3.5-4B"})

    def test_runtime_never_imports_benchmarks_or_vllm(self) -> None:
        for path in self.runtime_paths():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                with self.subTest(path=path.name, names=names):
                    self.assertFalse(
                        any(name == "benchmarks" or name.startswith("benchmarks.") for name in names)
                    )
                    self.assertFalse(any(name == "vllm" or name.startswith("vllm.") for name in names))

        pipeline = (ROOT / "src" / "data_pipeline.py").read_text(encoding="utf-8")
        self.assertIn('"benchmark_files_read": 0', pipeline)
        self.assertNotIn("validate_parent_data_parity", pipeline)

    def test_every_python_runtime_source_compiles(self) -> None:
        for path in self.runtime_paths():
            with self.subTest(path=path.name):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_cli_exposes_only_registered_capacity_objective_phase_and_eval_axes(self) -> None:
        module = self.load_cli()

        design = module.parse_args(["--stage", "design-boundary"])
        self.assertEqual(design.stage, "design-boundary")
        for phase in (
            "lora_joint",
            "lora_control",
            "stage_b_seal",
            "fullrank_joint",
            "fullrank_control",
        ):
            args = module.parse_args(["--stage", "analyze", "--phase", phase])
            self.assertEqual(args.phase, phase)
        for capacity in ("lora", "fullrank"):
            for objective in ("joint", "state_only"):
                args = module.parse_args(
                    ["--stage", "train", "--capacity", capacity, "--objective", objective]
                )
                self.assertEqual((args.capacity, args.objective), (capacity, objective))
        for eval_set in ("trigger", "contrast"):
            args = module.parse_args(["--stage", "evaluate-state", "--eval-set", eval_set])
            self.assertEqual(args.eval_set, eval_set)

        source = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")
        for option in (
            "--initialization-bundle",
            "--model-smoke-receipt",
            "--positive-control-receipt",
            "--authorization-receipt",
            "--checkpoint",
        ):
            self.assertIn(option, source)
        self.assertIn("--seed is required", source)

    def test_cli_rejects_loaded_modules_outside_authorized_source_snapshot(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"

        @contextlib.contextmanager
        def mismatched_snapshot():
            yield "f" * 64

        with tempfile.TemporaryDirectory() as directory:
            experiment = Path(directory) / "repo" / "experiments" / ROOT.name
            experiment.mkdir(parents=True)
            with mock.patch.object(module, "ROOT", experiment), mock.patch.object(
                module, "authorized_source_execution_snapshot", mismatched_snapshot
            ), mock.patch.object(module, "cpu_smoke") as target:
                with self.assertRaisesRegex(RuntimeError, "loaded CLI modules"):
                    module.main(["--config", str(config_path), "--stage", "cpu-smoke"])
            target.assert_not_called()

    def test_no_authorization_preauthorization_cli_paths_reach_their_stage(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            for stage, target_name, extra in (
                ("cpu-smoke", "cpu_smoke", []),
                ("design-boundary", "freeze_design", []),
                ("prepare-data", "build_datasets", []),
                (
                    "prepare-init", "prepare_initialization_bundle",
                    ["--seed", "7411"],
                ),
            ):
                with self.subTest(stage=stage), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(
                    module, target_name, return_value={"status": "TEST_PASS"}
                ) as target, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        module.main(
                            ["--config", str(config_path), "--stage", stage, *extra]
                        ),
                        0,
                    )
                    target.assert_called_once()
            for smoke_args in (["--smoke"], ["--smoke", "--stage", "cpu-smoke"]):
                with self.subTest(smoke_args=smoke_args), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(
                    module, "cpu_smoke", return_value={"status": "TEST_PASS"}
                ) as target, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        module.main(
                            ["--config", str(config_path), *smoke_args]
                        ),
                        0,
                    )
                    target.assert_called_once()

    def test_no_authorization_lora_joint_cli_paths_pass_the_input_gate(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        from src import analysis

        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            checkpoint = (
                experiment
                / "../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication"
                / "lora_joint_seed7411"
                / "checkpoint_000002"
            ).resolve()
            gpu_cases = (
                ["--stage", "model-smoke", "--seed", "7411"],
                ["--stage", "positive-control", "--seed", "7411"],
                ["--stage", "train", "--seed", "7411"],
                [
                    "--stage", "evaluate-state", "--seed", "7411",
                    "--checkpoint", str(checkpoint),
                ],
            )
            for stage_args in gpu_cases:
                with self.subTest(stage=stage_args[1]), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(
                    module, "_gpu_stage", return_value=0
                ) as gpu_stage, contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        module.main(["--config", str(config_path), *stage_args]),
                        0,
                    )
                    called_args = gpu_stage.call_args.args[0]
                    self.assertIsNone(called_args.authorization_receipt)

            with mock.patch.object(
                module, "ROOT", experiment
            ), mock.patch.object(
                analysis, "analyze_phase", return_value={"status": "TEST_PASS"}
            ) as analyze, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    module.main(
                        [
                            "--config", str(config_path), "--stage", "analyze",
                            "--phase", "lora_joint",
                        ]
                    ),
                    0,
                )
                self.assertIsNone(analyze.call_args.args[4])

            with mock.patch.object(
                module, "ROOT", experiment
            ), mock.patch.object(module, "_gpu_stage") as gpu_stage:
                with self.assertRaisesRegex(SystemExit, "must be one of: none"):
                    module.main(
                        [
                            "--config", str(config_path), "--stage", "train",
                            "--seed", "7411", "--capacity", "lora",
                            "--objective", "joint", "--authorization-receipt",
                            str(experiment / "analysis" / "lora_joint_trigger.json"),
                        ]
                    )
                gpu_stage.assert_not_called()

    def test_preauthorization_cli_paths_reject_any_ignored_receipt(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            ignored = experiment / "analysis" / "stage_b_seal.json"
            for stage, target_name, extra in (
                ("cpu-smoke", "cpu_smoke", []),
                ("design-boundary", "freeze_design", []),
                ("prepare-data", "build_datasets", []),
                ("prepare-init", "prepare_initialization_bundle", ["--seed", "7411"]),
            ):
                with self.subTest(stage=stage), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(module, target_name) as target:
                    with self.assertRaisesRegex(SystemExit, "must be one of: none"):
                        module.main(
                            [
                                "--config", str(config_path), "--stage", stage,
                                "--capacity", "fullrank", "--objective", "state_only",
                                "--authorization-receipt", str(ignored), *extra,
                            ]
                        )
                    target.assert_not_called()

    def test_contrast_cli_accepts_only_joint_objective_for_both_capacities(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            large = (
                experiment
                / "../../large_artifacts/qwen35_4b_state_formation_capacity_adjudication"
            ).resolve()
            authorization = experiment / "analysis" / "stage_b_seal.json"
            for capacity in ("lora", "fullrank"):
                joint_checkpoint = (
                    large
                    / f"{capacity}_joint_seed7411"
                    / "checkpoint_000002"
                )
                with self.subTest(capacity=capacity, objective="joint"), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(module, "_gpu_stage", return_value=0) as gpu_stage:
                    self.assertEqual(
                        module.main(
                            [
                                "--config", str(config_path), "--stage", "evaluate-state",
                                "--eval-set", "contrast", "--capacity", capacity,
                                "--objective", "joint", "--seed", "7411",
                                "--checkpoint", str(joint_checkpoint),
                                "--authorization-receipt", str(authorization),
                            ]
                        ),
                        0,
                    )
                    gpu_stage.assert_called_once()

                state_only_checkpoint = (
                    large
                    / f"{capacity}_state_only_seed7411"
                    / "checkpoint_000002"
                )
                with self.subTest(
                    capacity=capacity, objective="state_only"
                ), mock.patch.object(
                    module, "ROOT", experiment
                ), mock.patch.object(module, "_gpu_stage") as gpu_stage:
                    with self.assertRaisesRegex(
                        SystemExit, "contrast evaluation requires --objective joint"
                    ):
                        module.main(
                            [
                                "--config", str(config_path), "--stage", "evaluate-state",
                                "--eval-set", "contrast", "--capacity", capacity,
                                "--objective", "state_only", "--seed", "7411",
                                "--checkpoint", str(state_only_checkpoint),
                                "--authorization-receipt", str(authorization),
                            ]
                        )
                    gpu_stage.assert_not_called()

    def test_complete_registered_cli_authorization_matrix_is_exact(self) -> None:
        module = self.load_cli()
        config = module.load_config(ROOT / "configs" / "smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            candidates = {
                "none": None,
                "lora_joint": experiment / "analysis" / "lora_joint_trigger.json",
                "stage_b": experiment / "analysis" / "stage_b_seal.json",
                "fullrank_joint": experiment / "analysis" / "fullrank_joint.json",
                "junk": experiment / "analysis" / "junk.json",
            }
            with mock.patch.object(module, "ROOT", experiment):
                cells = self.registered_cli_cells(experiment, config)
                self.assertEqual(len(cells), 23)
                for cell in cells:
                    accepted = set()
                    for candidate, path in candidates.items():
                        args = module.parse_args(cell["argv"])
                        self.fill_canonical_cli_inputs(module, args, config)
                        args.authorization_receipt = (
                            str(path) if path is not None else None
                        )
                        with self.subTest(cell=cell["name"], candidate=candidate):
                            try:
                                module._require_canonical_inputs(args, config)
                            except SystemExit:
                                pass
                            else:
                                accepted.add(candidate)
                    self.assertEqual(
                        accepted,
                        cell["authorizations"],
                        f"authorization matrix drifted for {cell['name']}",
                    )

    def test_all_registered_cli_outputs_are_exact_and_only_setup_collides(self) -> None:
        module = self.load_cli()
        config = module.load_config(ROOT / "configs" / "smoke.yaml")
        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            outputs: dict[Path, list[str]] = {}
            with mock.patch.object(module, "ROOT", experiment):
                cells = self.registered_cli_cells(experiment, config)
                for cell in cells:
                    args = module.parse_args(cell["argv"])
                    actual = module._canonical_output(args, config)
                    with self.subTest(cell=cell["name"]):
                        self.assertEqual(actual, cell["output"])
                    outputs.setdefault(actual, []).append(cell["name"])

            self.assertEqual(len(cells), 23)
            self.assertEqual(len(outputs), 23)
            with mock.patch.object(module, "ROOT", experiment):
                for stage in ("model-smoke", "positive-control"):
                    for capacity in ("lora", "fullrank"):
                        alias = module.parse_args(
                            [
                                "--stage", stage, "--capacity", capacity,
                                "--objective", "state_only", "--seed", "7411",
                            ]
                        )
                        alias_name = f"{stage}:{capacity}:state_only"
                        alias_output = module._canonical_output(alias, config)
                        registered_name = f"{stage}:{capacity}"
                        self.assertEqual(
                            alias_output,
                            next(
                                cell["output"]
                                for cell in cells
                                if cell["name"] == registered_name
                            ),
                        )
                        outputs[alias_output].append(alias_name)
            collisions = {
                tuple(sorted(names))
                for names in outputs.values()
                if len(names) > 1
            }
            self.assertEqual(
                collisions,
                {
                    (
                        "model-smoke:fullrank",
                        "model-smoke:fullrank:state_only",
                    ),
                    ("model-smoke:lora", "model-smoke:lora:state_only"),
                    (
                        "positive-control:fullrank",
                        "positive-control:fullrank:state_only",
                    ),
                    (
                        "positive-control:lora",
                        "positive-control:lora:state_only",
                    ),
                },
            )

    def test_invalid_cells_fail_before_canonical_output_construction(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        config = module.load_config(config_path)
        direct_cases = (
            (
                module.parse_args(["--stage", "analyze"]),
                "registered analysis phase",
            ),
            (
                module.parse_args(
                    [
                        "--stage", "evaluate-state", "--objective", "state_only",
                        "--eval-set", "contrast",
                    ]
                ),
                "requires --objective joint",
            ),
        )
        unregistered = module.parse_args(
            ["--stage", "analyze", "--phase", "lora_joint"]
        )
        unregistered.phase = "unregistered_phase"
        direct_cases += ((unregistered, "registered analysis phase"),)
        for args, message in direct_cases:
            with self.subTest(direct=message), self.assertRaisesRegex(
                SystemExit, message
            ):
                module._canonical_output(args, config)

        main_cases = (
            (["--stage", "analyze"], "registered analysis phase"),
            (
                [
                    "--stage", "evaluate-state", "--objective", "state_only",
                    "--eval-set", "contrast",
                ],
                "requires --objective joint",
            ),
            (
                ["--stage", "cpu-smoke", "--phase", "lora_joint"],
                "only valid for analyze",
            ),
            (
                ["--stage", "cpu-smoke", "--eval-set", "contrast"],
                "only valid for evaluate-state",
            ),
            (
                ["--smoke", "--stage", "train"],
                "cannot override a non-cpu-smoke",
            ),
        )
        for argv, message in main_cases:
            with self.subTest(argv=argv), mock.patch.object(
                module, "_canonical_output"
            ) as canonical_output:
                with self.assertRaisesRegex(SystemExit, message):
                    module.main(["--config", str(config_path), *argv])
                canonical_output.assert_not_called()
        self.assertNotIn(
            "missing_phase",
            (ROOT / "scripts" / "run.py").read_text(encoding="utf-8"),
        )

    def test_noncanonical_explicit_output_never_dispatches_any_stage_family(self) -> None:
        module = self.load_cli()
        config_path = ROOT / "configs" / "smoke.yaml"
        config = module.load_config(config_path)
        from src import analysis

        with tempfile.TemporaryDirectory() as directory:
            experiment = (
                Path(directory)
                / "repo"
                / "experiments"
                / "qwen35_4b_state_formation_capacity_adjudication"
            )
            experiment.mkdir(parents=True)
            by_name = {
                cell["name"]: cell
                for cell in self.registered_cli_cells(experiment, config)
            }
            family_cells = (
                "cpu-smoke",
                "design-boundary",
                "prepare-data",
                "prepare-init",
                "model-smoke:lora",
                "positive-control:lora",
                "train:lora:joint",
                "evaluate-state:lora:joint:trigger",
                "analyze:lora_joint",
            )
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(module, "ROOT", experiment))
                dispatches = (
                    stack.enter_context(mock.patch.object(module, "cpu_smoke")),
                    stack.enter_context(mock.patch.object(module, "freeze_design")),
                    stack.enter_context(mock.patch.object(module, "build_datasets")),
                    stack.enter_context(
                        mock.patch.object(module, "prepare_initialization_bundle")
                    ),
                    stack.enter_context(mock.patch.object(module, "_gpu_stage")),
                    stack.enter_context(mock.patch.object(analysis, "analyze_phase")),
                )
                for name in family_cells:
                    for dispatch in dispatches:
                        dispatch.reset_mock()
                    with self.subTest(cell=name), self.assertRaisesRegex(
                        SystemExit, "--output must be the canonical path"
                    ):
                        module.main(
                            [
                                "--config", str(config_path), *by_name[name]["argv"],
                                "--output", str(experiment / "noncanonical-output"),
                            ]
                        )
                    self.assertFalse(any(dispatch.called for dispatch in dispatches))

    def test_common_adaptation_backend_and_disabled_evaluation_are_wired(self) -> None:
        adaptation = (ROOT / "src" / "adaptation.py").read_text(encoding="utf-8")
        model = (ROOT / "src" / "state_loop_model.py").read_text(encoding="utf-8")
        runner = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        self.assertIn("class AdaptationBank", adaptation)
        self.assertIn('capacity not in {"lora", "fullrank"}', adaptation)
        self.assertIn("torch.ops.aten.native_dropout.default", adaptation)
        self.assertIn("microbatch_dropout_seed", runner)
        self.assertIn("load_initialization_bundle", runner)
        self.assertIn("adaptation_gradient_clip", runner)
        self.assertIn("common_gradient_clip", runner)
        self.assertIn('for mode in ("intact", "disabled")', runner)
        self.assertIn("wrapper.adaptation.suspended()", runner)
        self.assertIn("AdaptationBank", model)
        self.assertIn('compute_answer=(objective == "joint")', runner)
        self.assertIn("state-only objective computed a prohibited answer graph", runner)

    def test_executable_surface_has_no_state_bag_arm(self) -> None:
        executable = self.runtime_paths()
        for path in executable:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertIsNone(re.search(r"\bbag\b", source, flags=re.IGNORECASE))

    def test_unregistered_predecessor_control_surfaces_are_absent(self) -> None:
        for path in self.runtime_paths():
            source = path.read_text(encoding="utf-8")
            for token in (
                "generate_counterfactual_pair",
                "counterfactual",
                "PILOT_PROMOTION_READY",
                "pilot_promotion",
                "checkpoint_phase",
            ):
                with self.subTest(path=path.name, token=token):
                    self.assertFalse(
                        token in source,
                        f"unregistered predecessor token {token!r} appears in {path.name}",
                    )

    def test_phase_authorizations_and_sealed_contrast_are_fail_closed(self) -> None:
        runner = (ROOT / "src" / "gpu_runner.py").read_text(encoding="utf-8")
        analysis = (ROOT / "src" / "analysis.py").read_text(encoding="utf-8")
        boundary = (ROOT / "src" / "design_boundary.py").read_text(encoding="utf-8")
        for status in (
            "DESIGN_FROZEN",
            "LORA_JOINT_MISS_CONTROLS_REQUIRED",
            "STAGE_B_CONTRAST_AUTHORIZED",
            "FULLRANK_STATE_ONLY_REQUIRED",
            "DIRECT_FULLSHAPE_RECIPE_RESCUE",
        ):
            self.assertIn(status, runner + analysis + boundary)
        self.assertIn("sealed contrast evaluation requires", runner)
        self.assertIn("refusing to resume or overwrite", runner)
        self.assertIn("receipt_identity_sha256", analysis)
        self.assertIn("checkpoint_metadata_sha256", analysis)
        self.assertNotIn("SYNTHETIC_TEST_GUARD", analysis)

    def test_contrast_evaluator_seals_checkpoint_before_access_or_decompression(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "evaluate_state"
        )
        body = ast.get_source_segment(source, function) or ""
        checkpoint_index = body.find("_load_checkpoint(")
        setup_barrier_index = body.find("_setup_barrier(")
        training_barrier_index = body.find("_reached_training_barrier(")
        output_index = body.find("_require_new_output(")
        lineage_index = body.find("checkpoint_lineage = {", checkpoint_index)
        record_index = body.find("record_contrast_access(")
        marker_index = body.find("ensure_attempt_output(", record_index)
        started_index = body.find("start_contrast_access(", marker_index)
        validation_index = body.find("validated_data_rows(", started_index)
        self.assertGreaterEqual(checkpoint_index, 0)
        self.assertGreater(setup_barrier_index, 0)
        self.assertGreater(training_barrier_index, setup_barrier_index)
        self.assertGreater(checkpoint_index, training_barrier_index)
        self.assertGreater(output_index, checkpoint_index)
        self.assertGreater(lineage_index, checkpoint_index)
        self.assertGreater(record_index, lineage_index)
        self.assertGreaterEqual(record_index, 0)
        self.assertGreater(marker_index, record_index)
        self.assertGreater(started_index, marker_index)
        self.assertGreater(validation_index, started_index)
        canonical_checkpoint = body.find("checkpoint = canonical_paths.checkpoint_dir")
        canonical_output = body.find("output_dir = expected_output")
        self.assertGreater(canonical_checkpoint, body.find("if checkpoint.absolute()"))
        self.assertGreater(canonical_output, body.find("if output_dir.absolute()"))
        self.assertLess(canonical_checkpoint, checkpoint_index)
        self.assertLess(canonical_output, checkpoint_index)
        lineage_source = body[lineage_index:record_index]
        for field in (
            '"path"',
            '"metadata_sha256"',
            '"checkpoint_identity_sha256"',
        ):
            self.assertIn(field, lineage_source)
        record_call = body[record_index:marker_index]
        self.assertIn("checkpoint_lineage=checkpoint_lineage", record_call)
        self.assertIn("_contrast_authorization_details(", body)
        helper = source[
            source.index("def _contrast_authorization_details("):
            source.index("\ndef _setup_receipt_path(")
        ]
        self.assertIn("STAGE_B_CONTRAST_BRANCH", helper)
        self.assertIn('"stage_b_seal.json"', helper)
        self.assertIn('objective != "joint"', body)

    def test_result_training_preflight_precedes_content_model_and_output(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        function = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "train"
        )
        body = ast.get_source_segment(source, function) or ""
        ordered = (
            "_authorization_details_for(",
            "content_splits=set()",
            "_setup_barrier(",
            "_prior_training_barriers(",
            "canonical_training_cell_paths(",
            "recover_published_training_completion(",
            "training_launch_preflight(",
            "prepare_training_attempt(",
            "ensure_attempt_output(",
            "start_training_attempt(",
            "validate_training_attempt_history(",
            "validated_data_rows(",
            "_build_new(",
        )
        positions = [body.find(fragment) for fragment in ordered]
        self.assertTrue(all(position >= 0 for position in positions), positions)
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("_authorization_chain_contains", source)
        self.assertIn("validate_branch_authorization(", source)
        canonical_output = body.find("output_dir = canonical_paths.external_dir")
        self.assertGreater(canonical_output, body.find("if output_dir.absolute()"))
        self.assertLess(canonical_output, positions[5])
        publication = body.find('_write_new_json_pair(output_dir / "run.json"')
        injected_crash = body.find('"terminal_receipts_published"', publication)
        completion = body.find("complete_training_attempt(", injected_crash)
        self.assertGreater(publication, positions[-1])
        self.assertGreater(injected_crash, publication)
        self.assertGreater(completion, injected_crash)

    def test_g0_binds_two_step_gradients_calls_clips_and_destructive_reload(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        functions = {
            node.name: ast.get_source_segment(source, node) or ""
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        smoke = functions["_model_smoke_attempt"]
        gradients = functions["_gradient_receipt"]
        self.assertIn("for probe_step in (1, 2):", smoke)
        self.assertIn('probe_step == 2', smoke)
        self.assertIn('"all_required_tensors_finite_nonzero"', gradients)
        self.assertIn(
            'required = ("adaptation", "initializer", "step", "sufficiency", "damping")',
            gradients,
        )
        self.assertNotIn('"aggregate"', gradients.split("required =", 1)[1])
        self.assertIn('k1_adaptation_calls != 0', smoke)
        self.assertIn(
            'expected_calls = 3 * int(config["architecture"]["adaptation"][capacity]["expected_targets"])',
            smoke,
        )
        self.assertIn(
            'expected_worst_calls = (worst_k - 1) * int(',
            smoke,
        )
        self.assertIn('"call_manifest_sha256"', (ROOT / "src" / "adaptation.py").read_text(encoding="utf-8"))
        self.assertIn("adaptation_gradient_clip", smoke)
        self.assertIn("common_gradient_clip", smoke)
        self.assertIn('"destructive_adaptation_digest_changed": True', smoke)
        self.assertIn('"destructive_common_digest_changed": True', smoke)
        self.assertIn("wrapper.load_delta_state_dict", smoke)
        self.assertIn("wrapper.load_extra_state_dict", smoke)

    def test_g0_executes_a_live_joint_backward_probe_over_every_trainable_group(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        smoke = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "_model_smoke_attempt"
            ),
        ) or ""
        joint_start = smoke.index("joint_dropout_seed =")
        joint_end = smoke.index("optimizer_receipt =", joint_start)
        joint = smoke[joint_start:joint_end]
        self.assertIn("compute_answer=True", joint)
        self.assertIn('_objective_loss(joint_output, config, "joint")', joint)
        self.assertIn("joint_output.answer_loss is None", joint)
        self.assertIn("torch.isfinite(joint_output.answer_loss)", joint)
        self.assertIn("torch.isfinite(joint_loss)", joint)
        self.assertIn("joint_loss.backward()", joint)
        self.assertIn("joint_gradients = _gradient_receipt(wrapper)", joint)
        for group in (
            "adaptation",
            "initializer",
            "step",
            "sufficiency",
            "damping",
            "aggregate_exempt",
        ):
            self.assertIn(f'"{group}"', joint)
        self.assertIn('joint_gradients["base_gradient_tensors"] != 0', joint)
        self.assertIn('joint_dropout_probe["calls"] != expected_calls', joint)
        self.assertIn('joint_dropout_probe["cycles"] != 3', joint)
        self.assertIn('joint_dropout_probe["cycle_order_identical"]', joint)
        self.assertIn('joint_dropout_probe["each_cycle_exact_target_set"]', joint)
        self.assertIn("adaptation_gradient_clip", joint)
        self.assertIn("common_gradient_clip", joint)
        self.assertIn("optimizer.step()", joint)
        self.assertIn('"elapsed_seconds": joint_elapsed', joint)
        self.assertIn('"peak_allocated_gib": torch.cuda.max_memory_allocated()', joint)
        self.assertIn('"answer_loss": float(joint_output.answer_loss.detach().cpu())', joint)
        self.assertIn('"all_joint_trainable_groups_finite_nonzero"', joint)

    def test_model_smoke_never_shadows_its_canonical_output_path(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        smoke = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "model_smoke"
        )
        shadowing_stores = [
            node
            for node in ast.walk(smoke)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and node.id == "output"
        ]
        self.assertEqual(shadowing_stores, [])

    def test_pinned_snapshot_receipt_binds_every_weight_and_tokenizer_file(self) -> None:
        gpu_runner = self.load_gpu_runner()

        revision = gpu_runner.MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            snapshot = Path(directory) / "snapshots" / revision
            snapshot.mkdir(parents=True)
            shards = (
                "model.safetensors-00001-of-00002.safetensors",
                "model.safetensors-00002-of-00002.safetensors",
            )
            for filename in gpu_runner.PINNED_SNAPSHOT_FILES:
                payload = b"{}"
                if filename == "model.safetensors.index.json":
                    payload = json.dumps(
                        {"weight_map": {"a": shards[0], "b": shards[1]}}
                    ).encode("utf-8")
                (snapshot / filename).write_bytes(payload)
            for index, shard in enumerate(shards):
                (snapshot / shard).write_bytes(f"shard-{index}".encode("ascii"))

            with mock.patch.object(
                gpu_runner,
                "cached_file",
                side_effect=lambda _model, filename, **_kwargs: str(snapshot / filename),
            ):
                receipt = gpu_runner._pinned_snapshot_receipt()

            self.assertEqual(receipt["requested_revision"], revision)
            self.assertEqual(receipt["resolved_revision"], revision)
            self.assertEqual(receipt["snapshot_layout"], f"snapshots/{revision}")
            self.assertEqual(
                [entry["filename"] for entry in receipt["files"]],
                sorted((*gpu_runner.PINNED_SNAPSHOT_FILES, *shards)),
            )
            self.assertTrue(all(entry["resolved_revision"] == revision for entry in receipt["files"]))
            self.assertRegex(receipt["files_sha256"], r"^[0-9a-f]{64}$")

            wrong = Path(directory) / "snapshots" / ("f" * 40)
            wrong.mkdir(parents=True)
            for filename in gpu_runner.PINNED_SNAPSHOT_FILES:
                (wrong / filename).write_bytes((snapshot / filename).read_bytes())
            with mock.patch.object(
                gpu_runner,
                "cached_file",
                side_effect=lambda _model, filename, **_kwargs: str(wrong / filename),
            ):
                with self.assertRaisesRegex(RuntimeError, "outside the pinned model revision"):
                    gpu_runner._pinned_snapshot_receipt()

    def test_pinned_snapshot_receipt_rejects_malformed_index_and_mixed_paths(self) -> None:
        gpu_runner = self.load_gpu_runner()

        revision = gpu_runner.MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / revision
            snapshot.mkdir(parents=True)
            for filename in gpu_runner.PINNED_SNAPSHOT_FILES:
                (snapshot / filename).write_bytes(b"{}")

            def cached_from(base: Path):
                return lambda _model, filename, **_kwargs: str(base / filename)

            with mock.patch.object(gpu_runner, "cached_file", side_effect=cached_from(snapshot)):
                with self.assertRaisesRegex(RuntimeError, "has no weight map"):
                    gpu_runner._pinned_snapshot_receipt()

            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": "../escape.safetensors"}}),
                encoding="utf-8",
            )
            with mock.patch.object(gpu_runner, "cached_file", side_effect=cached_from(snapshot)):
                with self.assertRaisesRegex(RuntimeError, "invalid shard path"):
                    gpu_runner._pinned_snapshot_receipt()

            shard = "model.safetensors-00001-of-00001.safetensors"
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"a": shard}}), encoding="utf-8"
            )
            with mock.patch.object(gpu_runner, "cached_file", side_effect=cached_from(snapshot)):
                with self.assertRaisesRegex(RuntimeError, "not readable"):
                    gpu_runner._pinned_snapshot_receipt()

            (snapshot / shard).write_bytes(b"weights")
            second_snapshot = root / "other" / "snapshots" / revision
            second_snapshot.mkdir(parents=True)
            (second_snapshot / shard).write_bytes(b"weights")

            def mixed(_model, filename, **_kwargs):
                base = second_snapshot if filename == shard else snapshot
                return str(base / filename)

            with mock.patch.object(gpu_runner, "cached_file", side_effect=mixed):
                with self.assertRaisesRegex(RuntimeError, "mixed snapshot roots"):
                    gpu_runner._pinned_snapshot_receipt()

            wrong_basename = snapshot / "wrong-name.json"
            wrong_basename.write_bytes(b"{}")
            with mock.patch.object(
                gpu_runner,
                "cached_file",
                side_effect=lambda _model, filename, **_kwargs: str(wrong_basename),
            ):
                with self.assertRaisesRegex(RuntimeError, "basename changed"):
                    gpu_runner._pinned_snapshot_receipt()

    def test_pinned_snapshot_receipt_accepts_only_canonical_blob_symlinks(self) -> None:
        gpu_runner = self.load_gpu_runner()

        revision = gpu_runner.MODEL_REVISION
        with tempfile.TemporaryDirectory() as directory:
            model_cache = Path(directory) / "models--Qwen--Qwen3.5-4B"
            snapshot = model_cache / "snapshots" / revision
            blobs = model_cache / "blobs"
            snapshot.mkdir(parents=True)
            blobs.mkdir()
            shards = (
                "model.safetensors-00001-of-00002.safetensors",
                "model.safetensors-00002-of-00002.safetensors",
            )

            blob_paths: dict[str, Path] = {}

            def install(filename: str, payload: bytes) -> None:
                blob_name = hashlib.sha256(payload).hexdigest()
                blob = blobs / blob_name
                blob.write_bytes(payload)
                (snapshot / filename).symlink_to(f"../../blobs/{blob_name}")
                blob_paths[filename] = blob

            for filename in gpu_runner.PINNED_SNAPSHOT_FILES:
                payload = b"{}"
                if filename == "model.safetensors.index.json":
                    payload = json.dumps(
                        {"weight_map": {"a": shards[0], "b": shards[1]}}
                    ).encode("utf-8")
                install(filename, payload)
            for index, shard in enumerate(shards):
                install(shard, f"shard-{index}".encode("ascii"))

            cached = lambda _model, filename, **_kwargs: str(snapshot / filename)
            with mock.patch.object(gpu_runner, "cached_file", side_effect=cached):
                receipt = gpu_runner._pinned_snapshot_receipt()
            self.assertEqual(
                [entry["filename"] for entry in receipt["files"]],
                sorted((*gpu_runner.PINNED_SNAPSHOT_FILES, *shards)),
            )

            chat = snapshot / "chat_template.jinja"
            chat.unlink()
            chat.symlink_to("../../../outside")
            with mock.patch.object(
                gpu_runner, "cached_file", side_effect=cached
            ), self.assertRaisesRegex(RuntimeError, "content-addressed blob"):
                gpu_runner._pinned_snapshot_receipt()
            chat.unlink()
            chat.symlink_to(
                f"../../blobs/{blob_paths['chat_template.jinja'].name}"
            )

            alias = Path(directory) / "aliased-blob"
            os.link(blob_paths["chat_template.jinja"], alias)
            try:
                with mock.patch.object(
                    gpu_runner, "cached_file", side_effect=cached
                ), self.assertRaisesRegex(RuntimeError, "alias"):
                    gpu_runner._pinned_snapshot_receipt()
            finally:
                alias.unlink()

    def test_load_base_accepts_missing_runtime_hash_only_after_snapshot_proof(self) -> None:
        gpu_runner = self.load_gpu_runner()
        from src.config import load_config

        config = load_config(ROOT / "configs" / "default.yaml")
        tokenizer = mock.Mock()
        model = mock.Mock()
        model.config = SimpleNamespace(_commit_hash=None, use_cache=True)
        model.cuda.return_value = model
        proof = {
            "model_id": gpu_runner.MODEL_ID,
            "requested_revision": gpu_runner.MODEL_REVISION,
            "resolved_revision": gpu_runner.MODEL_REVISION,
            "snapshot_layout": f"snapshots/{gpu_runner.MODEL_REVISION}",
            "files": [],
            "files_sha256": "a" * 64,
        }
        with (
            mock.patch.object(gpu_runner, "_pinned_snapshot_receipt", return_value=proof),
            mock.patch.object(
                gpu_runner.AutoTokenizer, "from_pretrained", return_value=tokenizer
            ) as tokenizer_loader,
            mock.patch.object(
                gpu_runner.AutoModelForCausalLM, "from_pretrained", return_value=model
            ) as model_loader,
            mock.patch.object(
                gpu_runner,
                "_validate_tokenizer",
                return_value={"state_token_id": 1, "answer_token_ids": [2, 3, 4, 5]},
            ),
        ):
            _, loaded, receipt = gpu_runner._load_base(config)
        self.assertIs(loaded, model)
        self.assertIsNone(receipt["runtime_model_config_commit_hash"])
        self.assertEqual(receipt["pinned_snapshot"], proof)
        model.requires_grad_.assert_called_once_with(False)
        self.assertIs(tokenizer_loader.call_args.kwargs["local_files_only"], True)
        self.assertIs(model_loader.call_args.kwargs["local_files_only"], True)
        self.assertEqual(tokenizer_loader.call_args.kwargs["revision"], gpu_runner.MODEL_REVISION)
        self.assertEqual(model_loader.call_args.kwargs["revision"], gpu_runner.MODEL_REVISION)
        self.assertIs(model_loader.call_args.kwargs["use_safetensors"], True)

        with (
            mock.patch.object(
                gpu_runner, "_pinned_snapshot_receipt", side_effect=RuntimeError("proof failed")
            ),
            mock.patch.object(gpu_runner.AutoTokenizer, "from_pretrained") as tokenizer_loader,
            mock.patch.object(gpu_runner.AutoModelForCausalLM, "from_pretrained") as model_loader,
        ):
            with self.assertRaisesRegex(RuntimeError, "proof failed"):
                gpu_runner._load_base(config)
        tokenizer_loader.assert_not_called()
        model_loader.assert_not_called()

        model.config._commit_hash = gpu_runner.MODEL_REVISION
        model.reset_mock()
        with (
            mock.patch.object(gpu_runner, "_pinned_snapshot_receipt", return_value=proof),
            mock.patch.object(gpu_runner.AutoTokenizer, "from_pretrained", return_value=tokenizer),
            mock.patch.object(gpu_runner.AutoModelForCausalLM, "from_pretrained", return_value=model),
            mock.patch.object(
                gpu_runner,
                "_validate_tokenizer",
                return_value={"state_token_id": 1, "answer_token_ids": [2, 3, 4, 5]},
            ),
        ):
            _, loaded, receipt = gpu_runner._load_base(config)
        self.assertIs(loaded, model)
        self.assertEqual(
            receipt["runtime_model_config_commit_hash"], gpu_runner.MODEL_REVISION
        )

        model.config._commit_hash = "f" * 40
        with (
            mock.patch.object(gpu_runner, "_pinned_snapshot_receipt", return_value=proof),
            mock.patch.object(gpu_runner.AutoTokenizer, "from_pretrained", return_value=tokenizer),
            mock.patch.object(gpu_runner.AutoModelForCausalLM, "from_pretrained", return_value=model),
            mock.patch.object(
                gpu_runner,
                "_validate_tokenizer",
                return_value={"state_token_id": 1, "answer_token_ids": [2, 3, 4, 5]},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "loaded model revision differs"):
                gpu_runner._load_base(config)

    def test_checkpoint_initialization_lineage_rejects_symlinked_ancestor(self) -> None:
        gpu_runner = self.load_gpu_runner()

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            real = repo / "real-bundles"
            real.mkdir()
            bundle = real / "initialization.pt"
            bundle.write_bytes(b"synthetic")
            alias = repo / "bundles"
            alias.symlink_to(real, target_is_directory=True)
            with mock.patch.object(gpu_runner, "REPO_ROOT", repo):
                with self.assertRaisesRegex(RuntimeError, "not canonical"):
                    gpu_runner._resolve_repo_path("bundles/initialization.pt")
                with self.assertRaisesRegex(RuntimeError, "not canonical"):
                    gpu_runner._repo_relative(alias / "initialization.pt")
                self.assertEqual(
                    gpu_runner._resolve_repo_path("real-bundles/initialization.pt"),
                    bundle,
                )

    def test_positive_control_uses_fresh_grid_without_opening_result_rows(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        control = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "positive_control"
            ),
        ) or ""
        self.assertIn("generate_control_rows(config, manifest)", control)
        self.assertIn("produce_oracle_analysis_receipt(rows)", control)
        self.assertIn("content_splits=set()", control)
        self.assertNotIn("read_jsonl(", control)
        self.assertNotIn("validated_data_rows(", control)

    def test_pre_authorization_model_paths_open_only_registered_noncontrast_rows(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        functions = {
            node.name: ast.get_source_segment(source, node) or ""
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        self.assertIn('content_splits=set()', functions["_model_smoke_attempt"])
        self.assertIn('validated_data_rows(', functions["_model_smoke_attempt"])
        self.assertIn('("train",)', functions["_model_smoke_attempt"])
        self.assertIn(
            'seed=int(config["training"]["g0_control"]["worst_depth_seed"])',
            functions["_model_smoke_attempt"],
        )
        self.assertIn(
            'worst["structural_fingerprint"] in result_fingerprints',
            functions["_model_smoke_attempt"],
        )
        self.assertIn(
            '"cross_result_structural_overlap": 0',
            functions["_model_smoke_attempt"],
        )
        for split in ("validation", "depth_extrapolation", "joint_holdout"):
            self.assertNotIn(
                f'read_jsonl(data_dir / "{split}.jsonl.gz")',
                functions["_model_smoke_attempt"],
            )
        self.assertIn("content_splits=set()", functions["positive_control"])
        self.assertNotIn("validated_data_rows(", functions["positive_control"])
        self.assertIn("validated_data_rows(", functions["train"])
        self.assertIn('("train",)', functions["train"])
        for name in ("model_smoke", "positive_control", "train"):
            self.assertNotIn("contrast_depth", functions[name])
            self.assertNotIn("contrast_joint", functions[name])
            self.assertNotIn("contrast_validation", functions[name])

    def test_training_order_digest_is_objective_invariant(self) -> None:
        path = ROOT / "src" / "gpu_runner.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        train = ast.get_source_segment(
            source,
            next(
                node
                for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == "train"
            ),
        ) or ""
        event_start = train.index("event = {")
        event_end = train.index("encoded_event =", event_start)
        order_event = train[event_start:event_end]
        for field in ('"microbatch_index"', '"id"', '"k"'):
            self.assertIn(field, order_event)
        self.assertNotIn('"prompt_tokens"', order_event)
        self.assertNotIn("layer_token_applications", order_event)
        schedule_start = train.index("schedule_event = {")
        schedule_end = train.index("dropout_digest.update(", schedule_start)
        self.assertIn('"prompt_tokens"', train[schedule_start:schedule_end])
        self.assertIn("layer_token_applications += compute", train)

    def test_contrast_ledger_uses_stable_lock_and_atomic_fsynced_replaces(self) -> None:
        path = ROOT / "src" / "data_pipeline.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        functions = {
            node.name: ast.get_source_segment(source, node) or ""
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
        }
        record = functions["record_contrast_access"]
        atomic = functions["_atomic_write_json"]
        build = functions["build_datasets"]
        self.assertIn('lock_path = path.with_name(f"{path.name}.lock")', record)
        self.assertIn("with locked_regular(lock_path):", record)
        self.assertGreaterEqual(
            record.count("durable_atomic_write_json(path, payload, replace=True)"), 2
        )
        self.assertNotIn(".seek(", record)
        self.assertNotIn(".truncate(", record)
        self.assertNotIn('path.open("r+"', record)

        self.assertIn("publish_new_bytes(path.parent, path.name, encoded, mode=0o644)", atomic)
        self.assertIn("_atomic_write_json(manifest_path, manifest)", build)
        self.assertIn("_atomic_write_json(ledger_path, ledger)", build)
        writer = functions["_write_jsonl_gz"]
        self.assertIn("publish_new_file(", writer)
        self.assertIn("write_compressed", writer)
        self.assertIn("read_stable_bytes(path.parent, path)", writer)
        self.assertNotIn("os.O_EXCL", writer)

    def test_producers_and_archivers_share_one_execution_generation_lock(self) -> None:
        run = (ROOT / "scripts" / "run.py").read_text(encoding="utf-8")
        failed = (ROOT / "scripts" / "archive_failed_attempt.py").read_text(
            encoding="utf-8"
        )
        invalidated = (
            ROOT / "scripts" / "archive_invalidated_setup.py"
        ).read_text(encoding="utf-8")
        lock_name = 'ROOT / "runs" / "run.lock"'
        self.assertIn(lock_name, run)
        self.assertIn(lock_name, failed)
        self.assertIn(lock_name, invalidated)
        self.assertIn("locked_regular(EXECUTION_GENERATION_LOCK)", run)
        self.assertIn("with locked_regular(execution_lock):", failed)
        self.assertIn("with locked_regular(execution_generation_lock):", invalidated)


if __name__ == "__main__":
    unittest.main()
