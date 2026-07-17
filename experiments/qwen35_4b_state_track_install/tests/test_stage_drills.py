"""Stage-prerequisite drills: refusals are one-line SystemExits, never
raw git tracebacks.

``run.py``'s ``require_pushed_checkpoint`` guards every staged gate. Drilled
in a scratch git repo (clean, pushed, branch main): a prerequisite missing
from HEAD must refuse with the one-line ``stage prerequisite is not
committed at HEAD: <path>`` message (the git cat-file probe is caught, not
propagated); a dirty tree refuses with the clean-checkpoint one-liner; a
committed prerequisite passes.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPTS = EXP / "scripts"


def load_harness():
    spec = importlib.util.spec_from_file_location(
        "state_track_stage_drill_harness", SCRIPTS / "run.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HARNESS = load_harness()


def git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        env={
            "PATH": "/usr/bin:/bin",
            "GIT_AUTHOR_NAME": "drill",
            "GIT_AUTHOR_EMAIL": "drill@example.invalid",
            "GIT_COMMITTER_NAME": "drill",
            "GIT_COMMITTER_EMAIL": "drill@example.invalid",
            "HOME": str(cwd),
        },
    )


class TestRequirePushedCheckpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        base = Path(cls._tmp.name)
        origin = base / "origin.git"
        origin.mkdir()
        git(origin, "init", "--bare", "--initial-branch=main", ".")
        cls.repo = base / "repo"
        cls.repo.mkdir()
        git(cls.repo, "init", "--initial-branch=main", ".")
        (cls.repo / "committed.txt").write_text("frozen\n", encoding="utf-8")
        git(cls.repo, "add", "committed.txt")
        git(cls.repo, "commit", "-m", "drill checkpoint")
        git(cls.repo, "remote", "add", "origin", str(origin))
        git(cls.repo, "push", "origin", "main")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def setUp(self):
        self.real_root = HARNESS.ROOT
        self.addCleanup(setattr, HARNESS, "ROOT", self.real_root)
        HARNESS.ROOT = self.repo

    def test_committed_pushed_prerequisite_passes(self):
        HARNESS.require_pushed_checkpoint("committed.txt")

    def test_missing_from_head_refuses_with_the_one_line_message(self):
        with self.assertRaises(SystemExit) as caught:
            HARNESS.require_pushed_checkpoint("reports/never_committed.md")
        self.assertEqual(
            str(caught.exception),
            "stage prerequisite is not committed at HEAD: "
            "reports/never_committed.md",
        )

    def test_untracked_worktree_file_not_at_head_refuses_cleanly(self):
        # A file that exists on disk but was never committed: the worktree
        # must first be clean, so create it, observe the dirty refusal,
        # then commit WITHOUT pushing to observe the pushed refusal.
        drifting = self.repo / "drifting.txt"
        drifting.write_text("uncommitted\n", encoding="utf-8")
        try:
            with self.assertRaises(SystemExit) as caught:
                HARNESS.require_pushed_checkpoint("drifting.txt")
            self.assertEqual(
                str(caught.exception),
                "stage requires a clean pushed main checkpoint",
            )
        finally:
            drifting.unlink()

    def test_dirty_tree_refuses_with_the_one_line_message(self):
        scratch = self.repo / "dirty.txt"
        scratch.write_text("dirty\n", encoding="utf-8")
        try:
            with self.assertRaises(SystemExit) as caught:
                HARNESS.require_pushed_checkpoint("committed.txt")
            self.assertEqual(
                str(caught.exception),
                "stage requires a clean pushed main checkpoint",
            )
        finally:
            scratch.unlink()

    def test_refusals_never_leak_a_called_process_error(self):
        # The cat-file probe is check=False: SystemExit is the only refusal
        # shape for a missing prerequisite.
        try:
            HARNESS.require_pushed_checkpoint("also/never_committed.json")
        except SystemExit as error:
            self.assertIn("not committed at HEAD", str(error))
        except subprocess.CalledProcessError:  # pragma: no cover
            self.fail("missing prerequisite leaked a raw CalledProcessError")


if __name__ == "__main__":
    unittest.main()
