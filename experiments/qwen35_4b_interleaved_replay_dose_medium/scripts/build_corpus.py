#!/usr/bin/env python3
"""Authenticate and inherit the frozen two-lesson treatment corpus byte-identically.

This experiment changes exactly ONE design variable relative to its donor
(qwen35_4b_hygiene_explore_destack_medium): the parent lineage. The corpora
are therefore INHERITED, not rebuilt:

- TREATMENT (inherited): data/sft_hygiene_explore.jsonl copied byte-identically
  from the donor against its frozen hash pin, then re-derived from the copied
  generator (scripts/gen_axis_v2.py, byte-identical) at the frozen construction
  seed 77119 with the frozen TREATMENT_MIX (u_hygiene 40, u_explore 40) and
  required to reproduce the inherited bytes exactly. Schema and truth audits,
  the banned-vocabulary scan, and the corpus-level anti-shortcut balance are
  re-validated on the re-derived rows before a byte is written.
- REPLAY (inherited): data/sft_blend.jsonl copied byte-identically from the
  donor against its frozen hash pin.
- MANIFEST (inherited): data/corpus_manifest.json copied byte-identically from
  the donor against its frozen hash pin; its recorded corpus/replay hashes and
  generator hash are re-authenticated against the copied artifacts.

``--check`` re-authenticates everything and byte-compares; generation mode
accepts byte-identical existing targets and refuses mismatches.
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

DONOR_EXP = ROOT / "experiments" / "qwen35_4b_hygiene_explore_destack_medium"
DONOR_TREATMENT = DONOR_EXP / "data" / "sft_hygiene_explore.jsonl"
DONOR_REPLAY = DONOR_EXP / "data" / "sft_blend.jsonl"
DONOR_MANIFEST = DONOR_EXP / "data" / "corpus_manifest.json"
TREATMENT_SHA256 = "8b3e97919c62cbb0893add281dc1d3ae881aa0138d0d1721043fec26b0c22cf1"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
MANIFEST_SHA256 = "cbc9ae6d132d09b9bac2eb43010c2f3eb051993493d6ea659950ac07f0a1e903"
TREATMENT_ROWS = 80
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


def load_donor(path: Path, expected_sha256: str) -> bytes:
    if not path.is_file():
        raise ValueError(f"donor corpus provenance is absent: {path}")
    raw = path.read_bytes()
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"frozen donor corpus artifact changed: {path}")
    return raw


def build() -> tuple[dict[Path, bytes], dict]:
    treatment = load_donor(DONOR_TREATMENT, TREATMENT_SHA256)
    replay = load_donor(DONOR_REPLAY, REPLAY_SHA256)
    manifest_bytes = load_donor(DONOR_MANIFEST, MANIFEST_SHA256)
    if len(treatment.decode("utf-8").splitlines()) != TREATMENT_ROWS:
        raise ValueError("frozen donor treatment corpus row count changed")
    if len(replay.decode("utf-8").splitlines()) != REPLAY_ROWS:
        raise ValueError("frozen donor replay corpus row count changed")

    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("donor corpus manifest is not a JSON object")
    generator_sha256 = sha256_bytes((EXP / "scripts" / "gen_axis_v2.py").read_bytes())
    if (
        manifest.get("experiment_id") != DONOR_EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("mix") != TREATMENT_MIX
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("generator") != "scripts/gen_axis_v2.py"
        or manifest.get("generator_sha256") != generator_sha256
        or manifest.get("corpus", {}).get("rows") != TREATMENT_ROWS
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("kinds") != EXPECTED_KINDS
        or manifest.get("replay", {}).get("rows") != REPLAY_ROWS
        or manifest.get("replay", {}).get("sha256") != REPLAY_SHA256
        or manifest.get("model_loaded") is not False
        or manifest.get("benchmark_data_read") is not False
    ):
        raise ValueError("donor corpus manifest violates the frozen inheritance contract")

    # Re-derive the inherited treatment corpus from the byte-identical copied
    # generator at the frozen construction seed: the inheritance is proven by
    # exact byte reproduction, not by hash trust alone.
    rows = axis.generate_curriculum(TREATMENT_MIX, CONSTRUCTION_SEED)
    summary = axis.validate_generated(rows)
    axis.check_banned_vocabulary(rows)
    balance = axis.check_corpus_balance(rows)
    if summary["rows"] != TREATMENT_ROWS or summary["kinds"] != EXPECTED_KINDS:
        raise ValueError(f"inherited corpus mix changed: {summary['kinds']}")
    rederived = "".join(
        json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    if rederived != treatment:
        raise ValueError(
            "re-derivation from the copied generator at the frozen seed does "
            "not reproduce the inherited treatment corpus bytes"
        )
    if balance != manifest.get("balance"):
        raise ValueError("re-derived corpus balance disagrees with the donor manifest")

    return {
        CORPUS_PATH: treatment,
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
                raise SystemExit(f"inherited corpus artifact is absent or changed: {path}")
    else:
        conflicts = [
            path
            for path, expected in targets.items()
            if path.exists() and path.read_bytes() != expected
        ]
        if conflicts:
            parser.error(
                f"refusing to overwrite a differing inherited corpus artifact: {conflicts[0]}"
            )
        for path, value in targets.items():
            if path.exists():
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    print(json.dumps({
        "construction_seed": CONSTRUCTION_SEED,
        "corpus_sha256": TREATMENT_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "kinds": manifest["corpus"]["kinds"],
        "balance": manifest["balance"],
        "replay_sha256": REPLAY_SHA256,
        "inherited_from": DONOR_EXP.name,
        "rederivation_byte_identical": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
