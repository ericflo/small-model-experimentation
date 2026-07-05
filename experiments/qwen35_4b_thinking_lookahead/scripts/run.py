#!/usr/bin/env python3
"""Starter run harness for this experiment."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="check the scaffold without running the full experiment")
    args = parser.parse_args()

    if args.smoke:
        print("smoke scaffold passed: qwen35_4b_thinking_lookahead")
        return 0
    parser.error("implement the full experiment run before using this command")
    return 2


if __name__ == "__main__":
    sys.exit(main())
