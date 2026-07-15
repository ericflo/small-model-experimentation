#!/usr/bin/env python3
"""Authenticate the byte-identical inherited axis-curriculum treatment corpus.

This experiment does NOT construct a new treatment corpus: it inherits the
frozen 160-row axis mix (40 rows per stuck-axis kind) byte-identically from
the goal-gap axis experiment, together with that experiment's frozen corpus
manifest. This script is therefore an inheritance authenticator, not a
generator: it verifies the donor files against their frozen hash pins,
re-derives the corpus from the copied generator at the donor's frozen
construction seed to prove the generator lineage is intact, and materializes
(or, under ``--check``, byte-compares) the local copies. Generation mode
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

import gen_axis_curriculum as axis  # noqa: E402

DONOR_EXP = ROOT / "experiments" / "qwen35_4b_goal_gap_axis_curriculum_target_match"
DONOR_CORPUS = DONOR_EXP / "data" / "sft_axis160.jsonl"
DONOR_MANIFEST = DONOR_EXP / "data" / "corpus_manifest.json"
CONSTRUCTION_SEED = 77117
CORPUS_SHA256 = "e7a95d73c619e7c4f20f18ae98ac193e2f57373bd49dc9aede11fd548831686e"
MANIFEST_SHA256 = "dd881023b4437211cf05936d29599e3cd8eee0ded07fd7dba07849c0cd1231e4"
CORPUS_PATH = EXP / "data" / "sft_axis160.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
EXPECTED_KINDS = {"u_tracefix": 40, "u_explore": 40, "u_hygiene": 40, "u_protocol": 40}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build() -> tuple[dict[Path, bytes], dict]:
    if not DONOR_CORPUS.is_file() or not DONOR_MANIFEST.is_file():
        raise ValueError("donor corpus provenance is absent")
    corpus = DONOR_CORPUS.read_bytes()
    manifest_bytes = DONOR_MANIFEST.read_bytes()
    if sha256_bytes(corpus) != CORPUS_SHA256:
        raise ValueError(f"frozen donor corpus changed: {DONOR_CORPUS}")
    if sha256_bytes(manifest_bytes) != MANIFEST_SHA256:
        raise ValueError(f"frozen donor manifest changed: {DONOR_MANIFEST}")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if (
        manifest.get("experiment_id") != DONOR_EXP.name
        or manifest.get("construction_seed") != CONSTRUCTION_SEED
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("rows") != 160
        or manifest.get("corpus", {}).get("sha256") != CORPUS_SHA256
        or manifest.get("corpus", {}).get("kinds") != EXPECTED_KINDS
    ):
        raise ValueError("donor manifest violates the frozen inheritance contract")

    # Re-derive the corpus from the copied generator at the donor's frozen
    # construction seed: proves the inherited bytes and the local generator
    # lineage still agree, without minting any new construction seed.
    rows = axis.generate_curriculum(axis.ARM_MIX, CONSTRUCTION_SEED)
    summary = axis.validate_generated(rows)
    axis.check_banned_vocabulary(rows)
    balance = axis.check_corpus_balance(rows)
    if len(rows) != 160 or summary["kinds"] != EXPECTED_KINDS:
        raise ValueError(f"axis corpus mix changed: {summary['kinds']}")
    regenerated = "".join(
        json.dumps(axis.public_row(row), ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")
    if regenerated != corpus:
        raise ValueError("regenerated corpus diverged from the inherited bytes")
    if manifest.get("balance") != balance:
        raise ValueError("regenerated corpus balance diverged from the donor manifest")

    return {CORPUS_PATH: corpus, MANIFEST_PATH: manifest_bytes}, manifest


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
        "inherited_from": DONOR_EXP.name,
        "corpus_sha256": CORPUS_SHA256,
        "manifest_sha256": MANIFEST_SHA256,
        "construction_seed": CONSTRUCTION_SEED,
        "kinds": manifest["corpus"]["kinds"],
        "balance": manifest["balance"],
        "byte_identical_inheritance": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
