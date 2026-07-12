from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
SCRIPT = EXP / "scripts" / "analyze_benchmark.py"
CONFIG = EXP / "configs" / "default.yaml"
sys.path.insert(0, str(EXP / "src"))

from io_utils import sha256_file  # noqa: E402


class BenchmarkAnalysisTests(unittest.TestCase):
    def _run(self, *, inject_failure: bool) -> subprocess.CompletedProcess:
        root = Path(self.temporary.name)
        events = []
        first = 56201
        tiers = {"quick": range(first, first + 3), "medium": range(first + 3, first + 11)}
        for tier, seeds in tiers.items():
            for seed in seeds:
                for label, score in (("primary", 0.6), ("soup", 0.5), ("visible", 0.4)):
                    if inject_failure and tier == "quick" and seed == first and label == "primary":
                        score = 0.3
                    path = root / f"{tier}-{seed}-{label}.json"
                    path.write_text(
                        json.dumps(
                            {
                                "stage": "aggregate_only_menagerie_event",
                                "config_sha256": sha256_file(CONFIG),
                                "tier": tier,
                                "seed": seed,
                                "label": label,
                                "aggregate": score,
                            }
                        ),
                        encoding="utf-8",
                    )
                    events.append(path)
        command = [sys.executable, str(SCRIPT)]
        for path in events:
            command.extend(("--event", str(path)))
        command.extend(("--out", str(root / "analysis.json")))
        return subprocess.run(command, text=True, capture_output=True, check=False)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temporary.cleanup()

    def test_all_paired_events_must_be_positive(self):
        passed = self._run(inject_failure=False)
        self.assertEqual(passed.returncode, 0, passed.stderr + passed.stdout)
        failed = self._run(inject_failure=True)
        self.assertEqual(failed.returncode, 4, failed.stderr + failed.stdout)


if __name__ == "__main__":
    unittest.main()
