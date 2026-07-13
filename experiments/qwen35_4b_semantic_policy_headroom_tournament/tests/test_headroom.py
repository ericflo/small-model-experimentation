from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "src"))

import repo_tasks  # noqa: E402


def load_script(name: str):
    path = EXP / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"headroom_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class HeadroomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())

    def test_exact_model_and_no_benchmark_authorization(self):
        self.assertEqual(self.cfg["model"]["id"], "Qwen/Qwen3.5-4B")
        self.assertEqual(
            self.cfg["model"]["weight_sha256"],
            "1cf5fbca317808d6d00225f5cd533c82c7e1602b2b2e5e2da8f4307b01941ba3",
        )
        self.assertFalse(self.cfg["benchmark"]["menagerie_authorized"])

    def test_family_axis_registration_and_contract_contrast(self):
        self.assertEqual(
            tuple(self.cfg["families"]["all_headroom"]), repo_tasks.HEADROOM_FAMILIES
        )
        self.assertEqual(
            tuple(self.cfg["families"]["explicit_controls"]),
            repo_tasks.HEADROOM_EXPLICIT_FAMILIES,
        )
        for axis, families in repo_tasks.HEADROOM_AXES.items():
            self.assertEqual(tuple(self.cfg["families"][f"inferred_{axis}"]), families)
        tasks = repo_tasks.make_tasks(repo_tasks.HEADROOM_FAMILIES, 1, 88101, "unit")
        for task in tasks:
            if "_explicit_" in task.family:
                self.assertIn("must raise", task.issue)
            else:
                self.assertNotIn("must raise", task.issue)
                self.assertIn("input contract", task.issue)

    def test_all_tasks_have_exact_near_correct_lifecycle(self):
        tasks = repo_tasks.make_tasks(repo_tasks.HEADROOM_FAMILIES, 2, 88102, "unit")
        for task in tasks:
            for state, expected in (
                ("initial", (False, False)),
                ("partial", (False, False)),
                ("oracle", (True, True)),
            ):
                env = repo_tasks.RepoEnv(task)
                try:
                    if state == "partial":
                        env.apply_partial()
                    elif state == "oracle":
                        env.apply_oracle()
                    self.assertEqual(
                        (env.visible_pass(), env.hidden_pass()), expected,
                        (task.family, state),
                    )
                finally:
                    env.close()

    def test_blocks_are_unique_and_content_disjoint(self):
        families = repo_tasks.HEADROOM_FAMILIES
        blocks = {
            name: repo_tasks.make_tasks(
                families,
                int(self.cfg["evaluation"]["blocks"][name]["tasks_per_family"]),
                int(self.cfg["evaluation"]["blocks"][name]["seed"]),
                name,
            )
            for name in ("headroom_a", "headroom_b")
        }
        digests = {
            name: [repo_tasks.content_digest(task) for task in tasks]
            for name, tasks in blocks.items()
        }
        self.assertEqual(len(digests["headroom_a"]), 36)
        self.assertEqual(len(digests["headroom_b"]), 36)
        self.assertEqual(len(set(digests["headroom_a"])), 36)
        self.assertEqual(len(set(digests["headroom_b"])), 36)
        self.assertFalse(set(digests["headroom_a"]) & set(digests["headroom_b"]))

    def test_manifest_is_hash_seed_stable(self):
        snippet = (
            "import sys;"
            f"sys.path.insert(0,{str(EXP / 'src')!r});"
            "import repo_tasks;"
            "t=repo_tasks.make_tasks(repo_tasks.HEADROOM_FAMILIES,1,88200,'stable');"
            "print(repo_tasks.manifest_digest(t))"
        )
        values = []
        for seed in ("0", "99991"):
            values.append(subprocess.check_output(
                [sys.executable, "-c", snippet],
                text=True,
                env={**os.environ, "PYTHONHASHSEED": seed, "PYTHONDONTWRITEBYTECODE": "1"},
            ).strip())
        self.assertEqual(values[0], values[1])

    def test_qualification_band_is_nontrivial_and_children_are_deterministic(self):
        gates = self.cfg["qualification"]
        self.assertGreater(float(gates["inferred_axis_failed_test_success_min"]), 0.0)
        self.assertLess(float(gates["inferred_axis_failed_test_success_max"]), 1.0)
        runner = load_script("run")
        with patch.object(
            runner.subprocess, "run", return_value=SimpleNamespace(returncode=0)
        ) as mocked:
            runner.command(["synthetic"])
        self.assertEqual(mocked.call_args.kwargs["env"]["PYTHONHASHSEED"], "0")
        self.assertEqual(mocked.call_args.kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")


if __name__ == "__main__":
    unittest.main()
