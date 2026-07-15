#!/usr/bin/env python3
"""Freeze the de-stacked hygiene+explore treatment corpus and inherit the replay blend.

Two corpora feed this experiment:

- TREATMENT (fresh): 80 rows from scripts/gen_axis_v2.py at the frozen
  construction seed 77119 with the frozen TREATMENT_MIX (u_hygiene 40,
  u_explore 40) - only the two lessons with replicated installs; the closed
  trace-repair lessons are deliberately absent. Generation validates the
  schema and truth audits, scans the banned vocabulary, and enforces the
  corpus-level anti-shortcut balance before a byte is written; the manifest
  freezes the kind counts, hashes, and balance.
- REPLAY (inherited): data/sft_blend.jsonl copied byte-identically from the
  clean-parent predecessor against its frozen hash pin.

``--check`` regenerates everything and byte-compares; generation mode accepts
byte-identical existing targets and refuses mismatches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_axis_v2 as axis  # noqa: E402

DONOR_EXP = ROOT / "experiments" / "qwen35_4b_universal_fresh_surface_budget_commit_target_match"
DONOR_REPLAY = DONOR_EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
CONSTRUCTION_SEED = 77119
TREATMENT_MIX = "hygiene=40,explore=40"
CORPUS_PATH = EXP / "data" / "sft_hygiene_explore.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
EXPECTED_KINDS = {
    "u_hygiene": 40,
    "u_explore": 40,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build() -> tuple[dict[Path, bytes], dict]:
    if not DONOR_REPLAY.is_file():
        raise ValueError("donor replay corpus provenance is absent")
    replay = DONOR_REPLAY.read_bytes()
    if sha256_bytes(replay) != REPLAY_SHA256:
        raise ValueError(f"frozen donor replay corpus changed: {DONOR_REPLAY}")
    if len(replay.decode("utf-8").splitlines()) != REPLAY_ROWS:
        raise ValueError("frozen donor replay corpus row count changed")

    rows = axis.generate_curriculum(TREATMENT_MIX, CONSTRUCTION_SEED)
    summary = axis.validate_generated(rows)
    axis.check_banned_vocabulary(rows)
    balance = axis.check_corpus_balance(rows)
    if summary["rows"] != 80 or summary["kinds"] != EXPECTED_KINDS:
        raise ValueError(f"de-stacked corpus mix changed: {summary['kinds']}")
    corpus = "".join(
        json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")

    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "stage": "hygiene_explore_corpus_freeze",
        "generator": "scripts/gen_axis_v2.py",
        "generator_sha256": sha256_bytes((EXP / "scripts" / "gen_axis_v2.py").read_bytes()),
        "construction_seed": CONSTRUCTION_SEED,
        "mix": TREATMENT_MIX,
        "banned_vocabulary_checked": True,
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": summary["rows"],
            "sha256": sha256_bytes(corpus),
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "max_estimated_think_tokens": summary["max_estimated_think_tokens"],
        },
        "balance": balance,
        "replay": {
            "path": REPLAY_PATH.relative_to(EXP).as_posix(),
            "rows": REPLAY_ROWS,
            "sha256": REPLAY_SHA256,
            "inheritance": {
                "from_experiment": DONOR_EXP.name,
                "byte_identical": True,
            },
        },
        "model_loaded": False,
        "benchmark_data_read": False,
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    return {
        CORPUS_PATH: corpus,
        MANIFEST_PATH: manifest_bytes,
        REPLAY_PATH: replay,
    }, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    targets, manifest = build()
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
            parser.error(
                f"refusing to overwrite a differing frozen corpus artifact: {conflicts[0]}"
            )
        for path, value in targets.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "construction_seed": CONSTRUCTION_SEED,
        "corpus_sha256": manifest["corpus"]["sha256"],
        "manifest_sha256": sha256_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
        ),
        "kinds": manifest["corpus"]["kinds"],
        "balance": manifest["balance"],
        "replay_sha256": REPLAY_SHA256,
        "replay_inherited_from": DONOR_EXP.name,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
