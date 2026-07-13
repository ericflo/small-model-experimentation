from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import build_knowledgebase as builder


class CatalogIgnoreInvarianceTests(unittest.TestCase):
    def _write(self, path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def _snapshot(self, root: Path) -> dict[str, bytes]:
        selected = [root / "experiments" / "example" / "metadata.yaml"]
        selected.extend(
            path for path in (root / "knowledge").rglob("*") if path.is_file()
        )
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in sorted(selected)
        }

    def _run_to_fixpoint(self, root: Path) -> dict[str, bytes]:
        previous: dict[str, bytes] | None = None
        for _ in range(6):
            with contextlib.redirect_stdout(io.StringIO()):
                builder.main()
            current = self._snapshot(root)
            if current == previous:
                return current
            previous = current
        self.fail("catalog fixture did not reach a generated-file fixpoint")

    def test_ignored_artifact_cannot_change_any_generated_catalog_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(
                ["git", "init", "--quiet"], cwd=root, check=True
            )
            experiments = root / "experiments"
            experiment = experiments / "example"
            knowledge = root / "knowledge"
            programs = root / "research_programs"
            claims = knowledge / "claims"

            self._write(
                experiment / "README.md",
                "# Example Experiment\n\n"
                "This sufficiently long summary makes the temporary catalog "
                "record deterministic and human-readable.\n",
            )
            self._write(
                experiment / "reports" / "report.md",
                "# Report\n\nA stable result report for the catalog fixture.\n",
            )
            self._write(
                experiment / "scripts" / "run.py",
                "print('catalog fixture')\n",
            )
            self._write(
                experiment / "data" / "generated" / ".gitignore",
                "*.jsonl.gz\nmanifest.json\n",
            )
            self._write(
                experiment / ".gitignore",
                "reports/final_report.md\n"
                "scripts/local_only_smoke.py\n"
                "local_cache/\n",
            )
            self._write(programs / "registry.yaml", "programs:\n")
            self._write(
                claims / "claim_ledger.json",
                json.dumps({"claims": []}) + "\n",
            )
            self._write(
                knowledge / "future_experiment_queue.json",
                json.dumps({"candidate_programs": [], "proposals": []}) + "\n",
            )

            patched = {
                "ROOT": root,
                "EXPERIMENTS": experiments,
                "KNOWLEDGE": knowledge,
                "PROGRAMS": programs,
                "PROGRAM_REGISTRY": programs / "registry.yaml",
                "CLAIMS": claims,
                "CLAIM_LEDGER": claims / "claim_ledger.json",
                "FUTURE_QUEUE": knowledge / "future_experiment_queue.json",
                "PROGRAMS_CACHE": [],
            }
            with mock.patch.multiple(builder, **patched):
                baseline = self._run_to_fixpoint(root)
                sentinel = (
                    experiment
                    / "data"
                    / "generated"
                    / "local_only_rows.jsonl.gz"
                )
                sentinel.write_bytes(b"ignored external artifact\n")
                ignored_report = experiment / "reports" / "final_report.md"
                self._write(
                    ignored_report,
                    "# Wrong Local Report\n\nThis must never become the primary report.\n",
                )
                ignored_script = experiment / "scripts" / "local_only_smoke.py"
                self._write(ignored_script, "# smoke signal from ignored code\n")
                ignored_cache = experiment / "local_cache" / "payload.bin"
                ignored_cache.parent.mkdir(parents=True)
                ignored_cache.write_bytes(b"ignored cache\n")
                visible_files = builder.git_visible_experiment_files()
                for ignored_path in (
                    sentinel,
                    ignored_report,
                    ignored_script,
                    ignored_cache,
                ):
                    self.assertNotIn(ignored_path, visible_files)

                with contextlib.redirect_stdout(io.StringIO()):
                    builder.main()
                self.assertEqual(baseline, self._snapshot(root))


if __name__ == "__main__":
    unittest.main()
