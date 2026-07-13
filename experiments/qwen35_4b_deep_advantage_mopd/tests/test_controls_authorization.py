from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))
sys.path.insert(0, str(EXP / "scripts"))

import authorize_controls  # noqa: E402
import run as run_script  # noqa: E402
from control_code_inventory import control_code_inventory  # noqa: E402


class ControlsAuthorizationTests(unittest.TestCase):
    def test_control_code_inventory_covers_all_executed_control_surfaces(self):
        inventory = control_code_inventory()
        names = {Path(row["path"]).name for row in inventory["files"]}
        self.assertTrue(
            {
                "run.py",
                "authorize_controls.py",
                "authorize_benchmark.py",
                "train_mopd_round.py",
                "train_offpolicy_round.py",
                "build_control_overlay.py",
                "control_rematch.py",
                "control_receipts.py",
                "control_code_inventory.py",
            }.issubset(names)
        )
        self.assertEqual(inventory["file_count"], len(inventory["files"]))

    def test_authorizer_audits_every_integration_then_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.yaml"
            config_path.write_text("synthetic\n", encoding="utf-8")
            model = root / "model"
            model.mkdir()
            (model / "merge_receipt.json").write_text("{}\n", encoding="utf-8")
            config = {"seeds": {"integration_training": [42, 43, 44]}}
            with mock.patch.object(
                authorize_controls,
                "_audit_preregistration",
                return_value={"design_commit": "frozen-design"},
            ), mock.patch.object(
                authorize_controls,
                "_integration_model",
                side_effect=lambda _config, seed: root / f"primary-{seed}",
            ), mock.patch.object(
                authorize_controls,
                "_audit_integration",
                side_effect=lambda _config, _path, seed, _model: {
                    "path": f"seed-{seed}",
                    "sha256": str(seed),
                },
            ) as integration, mock.patch.object(
                authorize_controls,
                "_audit_controls",
                return_value=(
                    {"path": "controls", "sha256": "controls-sha"},
                    {"non_advantage_route": model},
                ),
            ) as controls:
                result = authorize_controls._build_authorization(config, config_path)
            self.assertTrue(result["gate"]["passed"])
            self.assertEqual(
                result["control_code_inventory_before_sha256"],
                result["control_code_inventory"]["sha256"],
            )
            self.assertEqual(
                result["control_code_inventory_after_sha256"],
                result["control_code_inventory"]["sha256"],
            )
            self.assertEqual(integration.call_count, 3)
            controls.assert_called_once_with(config, config_path)

    def test_semantic_control_failure_precedes_every_confirmation_eval(self):
        config = {"seeds": {"integration_training": [42, 43, 44]}}

        def require_gate(path: Path):
            if path.name == "controls_authorization.json":
                raise SystemExit("semantic controls authorization failed")
            return {"gate": {"passed": True}}

        with mock.patch.object(
            run_script, "_require_gate", side_effect=require_gate
        ), mock.patch.object(
            run_script, "_run", return_value=4
        ) as runner, mock.patch.object(
            run_script, "_paths"
        ) as paths, mock.patch.object(
            run_script, "_build_parameter_controls"
        ) as parameter_controls:
            with self.assertRaisesRegex(SystemExit, "semantic controls"):
                run_script._confirm(config, Path("config.yaml"))
        runner.assert_called_once()
        self.assertIn("authorize_controls.py", " ".join(runner.call_args.args[0]))
        paths.assert_not_called()
        parameter_controls.assert_not_called()

    def test_fresh_authorizer_exit_four_cannot_reuse_an_existing_receipt(self):
        with mock.patch.object(
            run_script, "_run", return_value=4
        ) as runner, mock.patch.object(
            run_script, "_require_gate"
        ) as gate:
            with self.assertRaisesRegex(SystemExit, "fresh semantic controls"):
                run_script._authorize_confirmation_inputs({}, Path("config.yaml"))
        self.assertEqual(runner.call_args.kwargs["allowed"], (0,))
        gate.assert_not_called()

    def test_controls_authorizer_propagates_semantic_audit_failure(self):
        config = {"seeds": {"integration_training": [42, 43, 44]}}
        with mock.patch.object(
            authorize_controls,
            "_audit_preregistration",
            return_value={"design_commit": "frozen-design"},
        ), mock.patch.object(
            authorize_controls, "_integration_model", return_value=Path("model")
        ), mock.patch.object(
            authorize_controls,
            "_audit_integration",
            return_value={"path": "integration", "sha256": "sha"},
        ), mock.patch.object(
            authorize_controls,
            "_audit_controls",
            side_effect=ValueError("target initial pressure mismatch"),
        ):
            with self.assertRaisesRegex(ValueError, "target initial pressure"):
                authorize_controls._build_authorization(config, Path("config.yaml"))

    def test_authorizer_detects_code_change_during_or_after_audit(self):
        config = {"seeds": {"integration_training": []}}
        inventory = {
            "files": [{"path": "a.py", "sha256": "a" * 64}],
            "file_count": 1,
            "sha256": "b" * 64,
        }
        changed = {
            **inventory,
            "sha256": "c" * 64,
        }
        with mock.patch.object(
            authorize_controls,
            "control_code_inventory",
            side_effect=[inventory, changed],
        ), mock.patch.object(
            authorize_controls, "sha256_file", return_value="hash"
        ), mock.patch.object(
            authorize_controls,
            "_audit_preregistration",
            return_value={"design_commit": "design"},
        ), mock.patch.object(
            authorize_controls, "_audit_controls", return_value=({}, {})
        ):
            with self.assertRaisesRegex(ValueError, "changed during"):
                authorize_controls._build_authorization(config, Path("config.yaml"))

        with mock.patch.object(
            authorize_controls,
            "control_code_inventory",
            side_effect=[inventory, inventory, changed],
        ), mock.patch.object(
            authorize_controls, "sha256_file", return_value="hash"
        ), mock.patch.object(
            authorize_controls,
            "_audit_preregistration",
            return_value={"design_commit": "design"},
        ), mock.patch.object(
            authorize_controls, "_audit_controls", return_value=({}, {})
        ):
            with self.assertRaisesRegex(ValueError, "changed after"):
                authorize_controls._build_authorization(config, Path("config.yaml"))

    def test_authorization_publication_is_no_clobber_and_exact_rerun_is_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            output = experiment / "analysis" / "controls_authorization.json"
            payload = {
                "stage": "authorization",
                "gate": {"passed": True},
                "nested": {"zero": 0, "two": 2},
            }
            with mock.patch.object(authorize_controls, "EXP", experiment):
                authorize_controls._publish_no_clobber(output, payload)
                before = output.read_bytes()
                before_stat = output.stat()
                authorize_controls._publish_no_clobber(output, payload)
                self.assertEqual(output.read_bytes(), before)
                self.assertEqual(output.stat().st_mtime_ns, before_stat.st_mtime_ns)

                type_mutations = (
                    {**payload, "nested": {"zero": False, "two": 2}},
                    {**payload, "nested": {"zero": 0, "two": 2.0}},
                )
                for mutation in type_mutations:
                    output.write_text(
                        json.dumps(mutation, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    mutation_bytes = output.read_bytes()
                    with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                        authorize_controls._publish_no_clobber(output, payload)
                    self.assertEqual(output.read_bytes(), mutation_bytes)

                tampered = json.loads(output.read_text(encoding="utf-8"))
                tampered["gate"] = {"passed": False}
                output.write_text(json.dumps(tampered), encoding="utf-8")
                tampered_bytes = output.read_bytes()
                with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
                    authorize_controls._publish_no_clobber(output, payload)
                self.assertEqual(output.read_bytes(), tampered_bytes)

    def test_authorization_publication_race_never_overwrites_winner(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            output = experiment / "analysis" / "controls_authorization.json"

            def lose_race(_source, target):
                Path(target).write_bytes(b"winner")
                raise FileExistsError

            with mock.patch.object(
                authorize_controls, "EXP", experiment
            ), mock.patch.object(
                authorize_controls.os, "link", side_effect=lose_race
            ), self.assertRaisesRegex(ValueError, "lost a race"):
                authorize_controls._publish_no_clobber(output, {"ours": True})
            self.assertEqual(output.read_bytes(), b"winner")

    def test_authorization_publication_rejects_symlinked_canonical_root_before_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            experiment.mkdir()
            real_analysis = root / "real-analysis"
            real_analysis.mkdir()
            (experiment / "analysis").symlink_to(
                real_analysis, target_is_directory=True
            )
            output = real_analysis / "controls_authorization.json"

            with mock.patch.object(
                authorize_controls, "EXP", experiment
            ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
                authorize_controls.tempfile, "NamedTemporaryFile"
            ) as temporary_file, mock.patch.object(
                authorize_controls.os, "link"
            ) as link, self.assertRaisesRegex(ValueError, "symlinked existing"):
                authorize_controls._publish_no_clobber(output, {"ours": True})

            mkdir.assert_not_called()
            temporary_file.assert_not_called()
            link.assert_not_called()
            self.assertFalse(output.exists())

    def test_authorization_publication_rejects_symlinked_output_ancestor_before_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "experiment"
            analysis = experiment / "analysis"
            real_parent = analysis / "real-parent"
            real_parent.mkdir(parents=True)
            alias = analysis / "alias"
            alias.symlink_to(real_parent, target_is_directory=True)
            output = alias / "controls_authorization.json"

            with mock.patch.object(
                authorize_controls, "EXP", experiment
            ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
                authorize_controls.tempfile, "NamedTemporaryFile"
            ) as temporary_file, mock.patch.object(
                authorize_controls.os, "link"
            ) as link, self.assertRaisesRegex(ValueError, "symlinked existing"):
                authorize_controls._publish_no_clobber(output, {"ours": True})

            mkdir.assert_not_called()
            temporary_file.assert_not_called()
            link.assert_not_called()
            self.assertFalse((real_parent / output.name).exists())

    def test_authorization_publication_rejects_parent_outside_analysis_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment = root / "experiment"
            (experiment / "analysis").mkdir(parents=True)
            outside = root / "outside"
            outside.mkdir()
            output = outside / "controls_authorization.json"

            with mock.patch.object(
                authorize_controls, "EXP", experiment
            ), mock.patch.object(Path, "mkdir") as mkdir, mock.patch.object(
                authorize_controls.tempfile, "NamedTemporaryFile"
            ) as temporary_file, mock.patch.object(
                authorize_controls.os, "link"
            ) as link, self.assertRaisesRegex(ValueError, "outside the experiment"):
                authorize_controls._publish_no_clobber(output, {"ours": True})

            mkdir.assert_not_called()
            temporary_file.assert_not_called()
            link.assert_not_called()
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
