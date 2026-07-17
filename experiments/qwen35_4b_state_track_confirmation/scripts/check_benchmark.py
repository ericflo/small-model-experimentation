#!/usr/bin/env python3
"""Recompute (or verify) the confirmation readout from the ledger-pinned inputs.

Thin CLI over ``run_benchmark.render_readout`` — the single source of
truth for every frozen constant and every authentication layer lives in
``run_benchmark.py``; this tool never consumes a seed and never runs the
gateway. The readout is a pure function of the twelve ledger-pinned
gateway receipts, the six sealed per-seed summaries, and the sha-pinned
prior-event summary, so:

- with ``--out`` it writes the readout (refusing to overwrite);
- without ``--out`` it recomputes the readout and requires the published
  file to match BYTE-IDENTICALLY (verify mode, used by ``run.py --smoke``).

Every input is authenticated through the complete six-seed write-ahead
ledger; unanchored receipt files can never reach the verdict. The
benchmark suite directory is never read.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def _load_run_benchmark():
    spec = importlib.util.spec_from_file_location(
        "state_track_confirmation_run_benchmark", SCRIPTS / "run_benchmark.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, help="write the readout (refuses overwrite)")
    args = parser.parse_args()
    bench = _load_run_benchmark()
    try:
        value = bench.render_readout()
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.out is not None:
        if args.out.exists():
            parser.error("refusing to overwrite confirmation readout")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(value)
        target = args.out
    else:
        if not bench.READOUT.is_file() or bench.READOUT.read_bytes() != value:
            parser.error("published confirmation readout is absent or changed")
        target = bench.READOUT
    payload = json.loads(value.decode("utf-8"))
    print(
        json.dumps(
            {
                "out": str(target),
                "sha256": hashlib.sha256(value).hexdigest(),
                "outcome": payload["outcome"],
                "verdict": payload["verdict"],
                "frozen_claim": payload["frozen_claim"],
            },
            indent=1,
            sort_keys=True,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
