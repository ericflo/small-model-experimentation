#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_gen import build_dataset  # noqa: E402


def main() -> None:
    summary = build_dataset(ROOT / "data")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
