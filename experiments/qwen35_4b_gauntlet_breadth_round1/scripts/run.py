#!/usr/bin/env python3
"""Orchestrator for gauntlet round 1.

Smoke (CPU-only): config parse + all 12 family selftests.
Full pipeline stages are explicit (single-tenant GPU — run one at a time):

  1. .venv-vllm/bin/python scripts/harvest.py --stage both        (~2-4 h GPU)
  2. python3 scripts/build_sft.py                                  (CPU)
  3. .venv/bin/python scripts/train_think.py --out <adapter dir>   (~30-40 min GPU)
  4. .venv-vllm/bin/python scripts/eval_gym.py --tag base          (once)
     .venv-vllm/bin/python scripts/eval_gym.py --tag round1 --adapter <dir>
  5. python3 scripts/bench.py --seed <fresh> --tier quick --arms base adapter \
         --adapter <dir>
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true",
                        help="CPU-only scaffold check: config + family selftests")
    args = parser.parse_args()

    if args.smoke:
        import yaml

        config = yaml.safe_load((EXP / "configs" / "default.yaml").read_text())
        assert config["round"] >= 1 and "harvest" in config and "train" in config
        print("config parse: ok")
        result = subprocess.run(
            [sys.executable, str(EXP / "scripts" / "selftest_gym.py")], text=True
        )
        if result.returncode != 0:
            return result.returncode
        print("smoke scaffold passed: qwen35_4b_gauntlet_breadth_round1")
        return 0
    parser.error("run the pipeline stages individually; see --help")
    return 2


if __name__ == "__main__":
    sys.exit(main())
