from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
REPO = EXP.parents[1]
GATEWAY_PATH = REPO / "scripts" / "run_benchmark_aggregate.py"
SPEC = importlib.util.spec_from_file_location("benchmark_aggregate_gateway", GATEWAY_PATH)
assert SPEC is not None and SPEC.loader is not None
gateway = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gateway)


SENTINEL = "DO_NOT_EXPOSE_PRIVATE_BENCHMARK_ITEM_7f20"


class AggregateFirewallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.runner = self.root / "run.py"
        self.runner.write_text("# synthetic runner identity\n", encoding="utf-8")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "merge_receipt.json").write_text("{}\n", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _assert_private_stdio(self, kwargs: dict) -> None:
        self.assertEqual(kwargs.get("stdin"), subprocess.DEVNULL)
        self.assertEqual(kwargs.get("stdout"), subprocess.DEVNULL)
        self.assertEqual(kwargs.get("stderr"), subprocess.DEVNULL)
        self.assertNotIn("capture_output", kwargs)

    def test_item_and_transcript_sentinels_do_not_cross_gateway(self):
        out = self.root / "aggregate.json"
        private_paths = []
        commands = []

        def synthetic_run(command, **kwargs):
            self._assert_private_stdio(kwargs)
            commands.append(command)
            raw = Path(command[command.index("--out") + 1])
            private_paths.append(raw)
            raw.write_text(
                json.dumps(
                    {
                        "aggregate": 0.75,
                        "per_family": {
                            family: {
                                "score": 0.75,
                                "detail": SENTINEL if family == "chronicle" else None,
                            }
                            for family in gateway.PUBLIC_FAMILY_KEYS
                        },
                        "within_budget": True,
                        "items": [{"prompt": SENTINEL, "transcript": SENTINEL}],
                        "debug_artifacts": SENTINEL,
                    }
                ),
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=0)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(gateway.subprocess, "run", side_effect=synthetic_run):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = gateway.run_event(
                    tier="quick",
                    think_budget=1024,
                    seed=123,
                    model=self.model,
                    out=out,
                    runner=self.runner,
                    python=Path("/synthetic/python"),
                )
        persisted = out.read_text(encoding="utf-8")
        self.assertEqual(set(result), gateway.OUTPUT_KEYS)
        self.assertEqual(result["think_budget"], 1024)
        self.assertEqual(commands[0][commands[0].index("--think-budget") + 1], "1024")
        self.assertNotIn(SENTINEL, persisted)
        self.assertNotIn(SENTINEL, stdout.getvalue())
        self.assertNotIn(SENTINEL, stderr.getvalue())
        self.assertEqual(
            json.loads(persisted)["per_family"],
            {family: 0.75 for family in gateway.PUBLIC_FAMILY_KEYS},
        )
        self.assertEqual(len(private_paths), 1)
        self.assertFalse(private_paths[0].exists())

    def test_private_text_cannot_cross_as_a_family_key(self):
        out = self.root / "aggregate.json"
        private_paths = []

        def synthetic_run(command, **kwargs):
            self._assert_private_stdio(kwargs)
            raw = Path(command[command.index("--out") + 1])
            private_paths.append(raw)
            families = {family: 0.5 for family in gateway.PUBLIC_FAMILY_KEYS}
            families.pop("chronicle")
            families[SENTINEL] = 0.5
            raw.write_text(
                json.dumps(
                    {
                        "aggregate": 0.5,
                        "per_family": families,
                        "within_budget": True,
                    }
                ),
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=0)

        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch.object(gateway.subprocess, "run", side_effect=synthetic_run):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(gateway.AggregateFailure) as raised:
                    gateway.run_event(
                        tier="quick",
                        seed=789,
                        model=self.model,
                        out=out,
                        runner=self.runner,
                        python=Path("/synthetic/python"),
                    )
        self.assertFalse(out.exists())
        self.assertNotIn(SENTINEL, str(raised.exception))
        self.assertNotIn(SENTINEL, stdout.getvalue())
        self.assertNotIn(SENTINEL, stderr.getvalue())
        self.assertEqual(len(private_paths), 1)
        self.assertFalse(private_paths[0].exists())

    def test_false_budget_gate_is_rejected(self):
        out = self.root / "aggregate.json"

        def synthetic_run(command, **kwargs):
            self._assert_private_stdio(kwargs)
            raw = Path(command[command.index("--out") + 1])
            raw.write_text(
                json.dumps(
                    {
                        "aggregate": 0.5,
                        "per_family": {
                            family: 0.5 for family in gateway.PUBLIC_FAMILY_KEYS
                        },
                        "within_budget": False,
                    }
                ),
                encoding="utf-8",
            )
            return types.SimpleNamespace(returncode=0)

        with mock.patch.object(gateway.subprocess, "run", side_effect=synthetic_run):
            with self.assertRaises(gateway.AggregateFailure):
                gateway.run_event(
                    tier="medium",
                    seed=790,
                    model=self.model,
                    out=out,
                    runner=self.runner,
                    python=Path("/synthetic/python"),
                )
        self.assertFalse(out.exists())

    def test_runner_failure_has_no_raw_output_channel(self):
        out = self.root / "aggregate.json"
        private_paths = []

        def synthetic_failure(command, **kwargs):
            self._assert_private_stdio(kwargs)
            raw = Path(command[command.index("--out") + 1])
            private_paths.append(raw)
            raw.write_text(SENTINEL, encoding="utf-8")
            return types.SimpleNamespace(
                returncode=17, stdout=SENTINEL, stderr=SENTINEL
            )

        with mock.patch.object(gateway.subprocess, "run", side_effect=synthetic_failure):
            with self.assertRaises(gateway.RunnerFailure) as raised:
                gateway.run_event(
                    tier="medium",
                    seed=456,
                    model=self.model,
                    out=out,
                    runner=self.runner,
                    python=Path("/synthetic/python"),
                )
        self.assertFalse(out.exists())
        self.assertNotIn(SENTINEL, str(raised.exception))
        self.assertIn("stdout/stderr suppressed", str(raised.exception))
        self.assertEqual(len(private_paths), 1)
        self.assertFalse(private_paths[0].exists())

    def test_source_inventory_binds_imported_suite_files(self):
        before = gateway.benchmark_source_inventory(self.root)
        dependency = self.root / "families" / "public_family" / "prompt.template"
        dependency.parent.mkdir(parents=True)
        dependency.write_text("# synthetic benchmark dependency\n", encoding="utf-8")
        after = gateway.benchmark_source_inventory(self.root)
        self.assertEqual(after["file_count"], before["file_count"] + 1)
        self.assertNotEqual(after["sha256"], before["sha256"])


if __name__ == "__main__":
    unittest.main()
