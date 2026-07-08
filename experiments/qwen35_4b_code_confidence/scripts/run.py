#!/usr/bin/env python3
"""Run harness: full pipeline = eval_code_conf.py (generate + execute + confidence signals) then analyze.py."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny end-to-end run (4 problems, k=2), no analysis")
    parser.add_argument("--n", type=int, default=260)
    parser.add_argument("--k", type=int, default=8)
    args = parser.parse_args()

    if args.smoke:
        return subprocess.call([sys.executable, str(SCRIPTS / "eval_code_conf.py"), "--n", "4", "--k", "2"])
    rc = subprocess.call([sys.executable, str(SCRIPTS / "eval_code_conf.py"), "--n", str(args.n), "--k", str(args.k)])
    if rc != 0:
        return rc
    return subprocess.call([sys.executable, str(SCRIPTS / "analyze.py")])


if __name__ == "__main__":
    sys.exit(main())
