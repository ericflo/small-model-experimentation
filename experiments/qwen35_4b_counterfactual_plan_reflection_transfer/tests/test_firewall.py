from __future__ import annotations

import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "src"))

from firewall import install_benchmark_firewall  # noqa: E402


class FirewallTests(unittest.TestCase):
    def test_nonexistent_probe_under_forbidden_root_is_denied_before_open(self) -> None:
        install_benchmark_firewall(EXP.parents[1])
        forbidden_probe = EXP.parents[1] / "benchmarks" / "__firewall_probe_nonexistent__"
        with self.assertRaisesRegex(PermissionError, "benchmark read firewall"):
            forbidden_probe.open()


if __name__ == "__main__":
    unittest.main()
