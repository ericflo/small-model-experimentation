from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


EXP = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


F = _load("build_data_test_families", EXP / "src" / "families.py")
_prior = {
    name: sys.modules.get(name)
    for name in ("families", "experiment_common", "oracle_data")
}
sys.modules["families"] = F
try:
    C = _load("build_data_test_common", EXP / "scripts" / "experiment_common.py")
    sys.modules["experiment_common"] = C
    O = _load("build_data_test_oracle", EXP / "scripts" / "oracle_data.py")
    sys.modules["oracle_data"] = O
    B = _load("build_data_under_test", EXP / "scripts" / "build_data.py")
finally:
    for _name, _module in _prior.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


class SplitSpecificationTests(unittest.TestCase):
    def test_full_specs_have_dedicated_depth_five_development_split(self) -> None:
        cfg = C.load_config()["task"]

        self.assertEqual(
            B.split_specs(cfg, False),
            [
                ("calibration", 4, 48, 8101),
                ("development", 5, 12, 9001),
                ("primary", 5, 60, 9101),
            ],
        )

    def test_smoke_uses_two_fresh_tasks_in_each_partition(self) -> None:
        cfg = C.load_config()["task"]
        specs = B.split_specs(cfg, True)

        self.assertEqual([name for name, _depth, _count, _seed in specs], ["calibration", "development", "primary"])
        self.assertEqual([count for _name, _depth, count, _seed in specs], [2, 2, 2])
        self.assertEqual([depth for _name, depth, _count, _seed in specs], [4, 5, 5])
        self.assertEqual(len({seed for _name, _depth, _count, seed in specs}), 3)


if __name__ == "__main__":
    unittest.main()

