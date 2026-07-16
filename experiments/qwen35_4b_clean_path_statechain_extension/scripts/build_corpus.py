#!/usr/bin/env python3
"""Freeze the BYTE-COPIED statechain treatment corpus and its manifest.

Lifecycle 23 applies the PROVEN statechain converter dose (lifecycle 18,
``qwen35_4b_statechain_only_dose``) to the zero-root composite. The
treatment corpus is deliberately NOT regenerated: fresh instances would
change the treatment, and the byte-identical copy is the controlled
choice — the 160 frozen rows (40 per formalism: brewvat, courierloft,
peatstove, muletrack) are copied from the source cell and held to its
committed sha256 pins. This builder:

- verifies the copied treatment corpus and replay pool are byte-identical
  to the source cell's committed files (fail-closed sha256 pins);
- verifies the source cell's committed corpus manifest (which carries the
  original fresh-surface grep audit and the row-overlap audit over the
  full predecessor inventory) is unchanged, and records it as the
  provenance document for those audits;
- REGENERATES the corpus through the byte-identical copied generator at
  the source construction seed 77,140 and byte-compares — the receipt
  that the copied generator still produces exactly the frozen treatment
  (the same generator later mints the FRESH seed-88041 axis holdout);
- re-runs the generator's structural validation, banned-vocabulary scan,
  and balance audit on the regenerated rows;
- checks the treatment shares zero canonical user messages with the
  replay pool.

Everything is a pure function of the pinned bytes; ``--check``
recomputes and byte-compares the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]
sys.path.insert(0, str(EXP / "scripts"))

import gen_statechain_curriculum as statechain  # noqa: E402

SOURCE_EXP = ROOT / "experiments" / "qwen35_4b_statechain_only_dose"
SOURCE_CONSTRUCTION_SEED = 77140
CORPUS_PATH = EXP / "data" / "sft_statechain_only.jsonl"
MANIFEST_PATH = EXP / "data" / "corpus_manifest.json"
TREATMENT_SHA256 = "ab6c78458eb8a41f42ebee25e79354d42beb558f78bb3d348dd7902ca3a9bad3"
TREATMENT_ROWS = 160
REPLAY_PATH = EXP / "data" / "sft_blend.jsonl"
REPLAY_SHA256 = "25a9595f2e70e4d5cab0a730f0e2613d314843f2a5dfe96187bc30d5d2abf0c2"
REPLAY_ROWS = 2240
EXPECTED_KINDS = {"u_statechain": 160}
EXPECTED_SURFACES = {formalism: 40 for formalism in statechain.STATECHAIN_FORMALISMS}
# The source cell's committed artifacts this copy is held to.
SOURCE_TREATMENT = SOURCE_EXP / "data" / "sft_statechain_only.jsonl"
SOURCE_REPLAY = SOURCE_EXP / "data" / "sft_blend.jsonl"
SOURCE_MANIFEST = SOURCE_EXP / "data" / "corpus_manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "97aa9f504c43e7b124c46f17ae367c862c1f67952ff1c72c197b70f019390467"
)
SOURCE_GENERATOR = SOURCE_EXP / "scripts" / "gen_statechain_curriculum.py"
GENERATOR = EXP / "scripts" / "gen_statechain_curriculum.py"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def message_bytes(row: dict) -> bytes:
    return json.dumps(
        row["messages"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def check_byte_copies() -> dict:
    """The copied treatment/replay/generator must match the source bytes."""
    copies = {
        "treatment": (CORPUS_PATH, SOURCE_TREATMENT, TREATMENT_SHA256),
        "replay": (REPLAY_PATH, SOURCE_REPLAY, REPLAY_SHA256),
        "generator": (GENERATOR, SOURCE_GENERATOR, None),
    }
    receipt = {}
    for name, (copy, source, expected) in copies.items():
        if not copy.is_file() or not source.is_file():
            raise ValueError(f"byte-copy input is absent: {name}")
        copy_bytes = copy.read_bytes()
        if copy_bytes != source.read_bytes():
            raise ValueError(f"copied {name} is not byte-identical to the source cell")
        digest = sha256_bytes(copy_bytes)
        if expected is not None and digest != expected:
            raise ValueError(f"copied {name} violates its frozen sha256 pin")
        receipt[name] = {
            "path": copy.relative_to(ROOT).as_posix(),
            "source": source.relative_to(ROOT).as_posix(),
            "sha256": digest,
            "byte_identical_to_source": True,
        }
    return receipt


def check_source_manifest() -> dict:
    """The source cell's audited corpus manifest is the provenance document."""
    if (
        not SOURCE_MANIFEST.is_file()
        or sha256_file(SOURCE_MANIFEST) != SOURCE_MANIFEST_SHA256
    ):
        raise ValueError("source cell's committed corpus manifest is absent or changed")
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    surface_audit = manifest.get("surface_freshness_audit", {})
    overlap_audit = manifest.get("row_overlap_audit", {})
    if (
        manifest.get("experiment_id") != SOURCE_EXP.name
        or manifest.get("construction_seed") != SOURCE_CONSTRUCTION_SEED
        or manifest.get("banned_vocabulary_checked") is not True
        or manifest.get("corpus", {}).get("sha256") != TREATMENT_SHA256
        or manifest.get("corpus", {}).get("rows") != TREATMENT_ROWS
        or not surface_audit.get("sources")
        or any(
            source.get("hits") != 0
            for source in surface_audit.get("sources", {}).values()
        )
        or not overlap_audit.get("sources")
        or any(
            source.get("overlap") != 0
            for source in overlap_audit.get("sources", {}).values()
        )
    ):
        raise ValueError("source corpus manifest violates the inherited audit contract")
    return manifest


def build() -> dict:
    copies = check_byte_copies()
    source_manifest = check_source_manifest()
    rows = statechain.generate_curriculum(
        statechain.ARM_MIX, SOURCE_CONSTRUCTION_SEED
    )
    regenerated = "".join(
        json.dumps(statechain.public_row(row), ensure_ascii=False) + "\n"
        for row in rows
    ).encode("utf-8")
    if regenerated != CORPUS_PATH.read_bytes():
        raise ValueError(
            "copied generator no longer regenerates the frozen treatment corpus "
            "byte-identically at the source construction seed"
        )
    summary = statechain.validate_generated(rows)
    statechain.check_banned_vocabulary(rows)
    balance = statechain.check_corpus_balance(rows)
    if (
        len(rows) != TREATMENT_ROWS
        or summary["kinds"] != EXPECTED_KINDS
        or summary["surfaces"] != EXPECTED_SURFACES
    ):
        raise ValueError(f"statechain corpus mix changed: {summary}")
    corpus_messages = {message_bytes(row) for row in rows}
    if len(corpus_messages) != TREATMENT_ROWS:
        raise ValueError("treatment corpus collides on canonical user messages")
    replay_rows = [
        json.loads(line)
        for line in REPLAY_PATH.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(replay_rows) != REPLAY_ROWS:
        raise ValueError("replay pool row count changed")
    replay_messages = {
        message_bytes(row) for row in replay_rows if row.get("messages")
    }
    if corpus_messages & replay_messages:
        raise ValueError("treatment corpus rows overlap the replay pool")
    banned = tuple(statechain.BANNED_PROMPT_TOKENS)
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "treatment_provenance": {
            "policy": "byte_identical_copy_of_the_proven_dose",
            "rationale": (
                "the statechain dose is the PROVEN treatment (lifecycle 18: "
                "21/40 axis strict over both controls, rites conversion 0.300 "
                "vs 0.100/0.100 paired); fresh instances would change the "
                "treatment, so the byte-identical copy is the controlled "
                "choice for applying it to the zero-root composite"
            ),
            "source_experiment": SOURCE_EXP.name,
            "source_construction_seed": SOURCE_CONSTRUCTION_SEED,
            "source_manifest": SOURCE_MANIFEST.relative_to(ROOT).as_posix(),
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "inherited_audits": {
                "surface_freshness_audit_sources": len(
                    source_manifest["surface_freshness_audit"]["sources"]
                ),
                "surface_freshness_total_hits": 0,
                "row_overlap_audit_sources": len(
                    source_manifest["row_overlap_audit"]["sources"]
                ),
                "row_overlap_total": 0,
                "documented_in": "the source cell's committed corpus manifest",
            },
            "copies": copies,
            "regenerated_byte_identically_via_copied_generator": True,
        },
        "banned_vocabulary_checked": True,
        "banned_vocabulary": {
            "tokens": len(banned),
            "sha256": sha256_bytes("\n".join(banned).encode("utf-8")),
        },
        "corpus": {
            "path": CORPUS_PATH.relative_to(EXP).as_posix(),
            "rows": TREATMENT_ROWS,
            "mix": statechain.ARM_MIX,
            "kinds": summary["kinds"],
            "surfaces": summary["surfaces"],
            "sha256": TREATMENT_SHA256,
        },
        "replay": {
            "path": REPLAY_PATH.relative_to(EXP).as_posix(),
            "rows": REPLAY_ROWS,
            "sha256": REPLAY_SHA256,
            "byte_identical_to_every_predecessor_copy": True,
        },
        "balance": balance,
        "replay_overlap": {
            "messages_compared": len(replay_messages),
            "overlap": 0,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = build()
    value = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    if args.check:
        if not MANIFEST_PATH.is_file() or MANIFEST_PATH.read_bytes() != value:
            raise SystemExit(
                f"frozen corpus manifest is absent or changed: {MANIFEST_PATH}"
            )
    else:
        if MANIFEST_PATH.exists() and MANIFEST_PATH.read_bytes() != value:
            parser.error("refusing to overwrite a differing frozen corpus manifest")
        if not MANIFEST_PATH.exists():
            MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
            MANIFEST_PATH.write_bytes(value)
    print(json.dumps({
        "corpus_sha256": manifest["corpus"]["sha256"],
        "manifest_sha256": sha256_bytes(value),
        "kinds": manifest["corpus"]["kinds"],
        "surfaces": manifest["corpus"]["surfaces"],
        "byte_copy_verified": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
