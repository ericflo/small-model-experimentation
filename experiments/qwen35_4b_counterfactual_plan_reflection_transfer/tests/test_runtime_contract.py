from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

import runtime_contract as R  # noqa: E402
import tokenizer_lineage as T  # noqa: E402


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


class RuntimeContractTests(unittest.TestCase):
    def test_execution_requires_clean_detached_root_cwd(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "repo"
            root.mkdir()
            _git(root, "init", "-q")
            _git(root, "config", "user.email", "test@example.invalid")
            _git(root, "config", "user.name", "Test")
            (root / "tracked.txt").write_text("v1\n")
            _git(root, "add", "tracked.txt")
            _git(root, "commit", "-qm", "initial")
            commit = _git(root, "rev-parse", "HEAD")

            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "clean detached"):
                    R.require_detached_execution_worktree(root)

            _git(root, "checkout", "--detach", "-q", commit)
            with mock.patch("pathlib.Path.cwd", return_value=root):
                self.assertEqual(
                    R.require_detached_execution_worktree(root),
                    {
                        "repo_root": str(root.resolve()),
                        "git_commit": commit,
                        "head_mode": "detached",
                        "cwd": str(root.resolve()),
                    },
                )

            (root / "tracked.txt").write_text("dirty\n")
            with mock.patch("pathlib.Path.cwd", return_value=root):
                with self.assertRaisesRegex(ValueError, "clean detached"):
                    R.require_detached_execution_worktree(root)
            with mock.patch("pathlib.Path.cwd", return_value=root.parent):
                with self.assertRaisesRegex(ValueError, "worktree root as cwd"):
                    R.require_detached_execution_worktree(root)

    def test_tokenizer_authentication_is_exact_and_mutation_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            files = {
                "chat_template.jinja": b"template\n",
                "merges.txt": b"a b\n",
                "tokenizer.json": b"{}\n",
                "tokenizer_config.json": b"{}\n",
                "vocab.json": b"{}\n",
            }
            for name, payload in files.items():
                (root / name).write_bytes(payload)
            pin = {
                "schema_version": 1,
                "model_id": "Qwen/Qwen3.5-4B",
                "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                "files": {
                    name: {
                        "sha256": __import__("hashlib").sha256(payload).hexdigest(),
                        "size": len(payload),
                    }
                    for name, payload in files.items()
                },
            }
            with mock.patch.object(T, "load_pinned_tokenizer", return_value=pin):
                receipt = T.authenticate_tokenizer_snapshot(root)
                self.assertEqual(receipt["files"], pin["files"])
                self.assertEqual(len(receipt["files_sha256"]), 64)
                (root / "vocab.json").write_bytes(b"tampered\n")
                with self.assertRaisesRegex(ValueError, "differs from exact revision"):
                    T.authenticate_tokenizer_snapshot(root)

    def test_pinned_tokenizer_file_set_is_exact(self) -> None:
        pin = json.loads((EXP / "configs" / "pinned_tokenizer_structure.json").read_text())
        self.assertEqual(
            set(pin["files"]),
            {
                "chat_template.jinja",
                "merges.txt",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            },
        )
        self.assertEqual(T.load_pinned_tokenizer(), pin)


if __name__ == "__main__":
    unittest.main()
