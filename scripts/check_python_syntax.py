#!/usr/bin/env python3
"""Compile repository maintenance scripts without writing cache files into the repo."""

from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    scripts = sorted((ROOT / "scripts").glob("*.py"))
    with tempfile.TemporaryDirectory() as tempdir:
        temp = Path(tempdir)
        for script in scripts:
            py_compile.compile(str(script), cfile=str(temp / f"{script.name}c"), doraise=True)
    print("python syntax check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
