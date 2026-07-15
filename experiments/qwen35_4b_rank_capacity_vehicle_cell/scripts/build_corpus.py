#!/usr/bin/env python3
"""Freeze the single axis-curriculum treatment corpus and its manifest.

The corpus is the 160-row designed axis mix (40 rows per stuck-axis kind)
rendered by gen_axis_curriculum.py at the frozen construction seed. Everything
is a pure function of that seed; `--check` regenerates and byte-compares. The
treatment file itself may already exist frozen (it is a preregistered pin), so
generation mode accepts byte-identical existing targets and refuses mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_curriculum as axis  # noqa: E402

CONSTRUCTION_SEED = 77117
CORPUS_PATH = EXP / "data" / "sft_axis160.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
EXPECTED_KINDS = {"u_tracefix": 40, "u_explore": 40, "u_hygiene": 40, "u_protocol": 40}


def build() -> tuple[dict[Path, bytes], dict]:
    rows = axis.generate_curriculum(axis.ARM_MIX, CONSTRUCTION_SEED)
    summary = axis.validate_generated(rows)
    axis.check_banned_vocabulary(rows)
    balance = axis.check_corpus_balance(rows)
    if len(rows) != 160 or summary["kinds"] != EXPECTED_KINDS:
        raise ValueError(f"axis corpus mix changed: {summary['kinds']}")

    corpus = "".join(
        json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "generator": "scripts/gen_axis_curriculum.py",
        "banned_vocabulary_checked": True,
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": len(rows),
            "mix": axis.ARM_MIX,
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "sha256": hashlib.sha256(corpus).hexdigest(),
        },
        "balance": balance,
    }
    return {CORPUS_PATH: corpus}, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs, manifest = build()
    targets = dict(outputs)
    targets[MANIFEST_PATH] = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        for path, expected in targets.items():
            if not path.is_file() or path.read_bytes() != expected:
                raise SystemExit(f"frozen corpus artifact is absent or changed: {path}")
    else:
        conflicts = [
            path
            for path, expected in targets.items()
            if path.exists() and path.read_bytes() != expected
        ]
        if conflicts:
            parser.error(f"refusing to overwrite a differing frozen corpus artifact: {conflicts[0]}")
        for path, value in targets.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "corpus_sha256": manifest["corpus"]["sha256"],
        "manifest_sha256": hashlib.sha256(targets[MANIFEST_PATH]).hexdigest(),
        "kinds": manifest["corpus"]["kinds"],
        "balance": manifest["balance"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
