#!/usr/bin/env python3
"""Deterministically build the repair_why_stack UNION corpus + its receipt.

Lifecycle 36 STACK. This cell does NOT generate any new training data. It COMBINES
two already-built, already-verified, already-committed source corpora — the
self-repair loop-behavior corpus (bet #2) and the WHY-comment causal-reasoning
corpus (bet #4) — into a single interleaved corpus and tests whether STACKING the
two individually-weak-positive ingredients captures BOTH of their target-specific
gains (self_repair -> agentic 8/35 -> 10/35, HumanEval +3; why_comment ->
HumanEval +5, agentic flat).

Standalone lineage (owner doctrine): the two source corpora are COPIED into this
cell as ``data/source_corpora/`` (sha-pinned), and the union is reproduced FROM
those in-cell copies by THIS script + the fixed shuffle seed — never from the
sibling cells. Cross-experiment references (the origin cells) are verification
aids only, never the reproduction path.

Build (deterministic, fail-closed):
  1. Verify BOTH source copies' sha256 against the frozen pins (abort on mismatch).
  2. Concatenate their non-blank JSONL lines in the frozen COMBINE_ORDER
     (self_repair block, then why_comment block) EXACTLY as their bytes appear —
     no re-serialization, so each row's exact encoding is preserved.
  3. Deterministically shuffle the 1008 lines with ``random.Random(93570)`` so the
     two kinds INTERLEAVE (block-concatenated training would see all of one kind
     then all of the other; interleaving matters).
  4. Write ``data/sft_repair_why_stack.jsonl`` and the receipt
     ``data/stack_corpus_receipt.json`` (source shas, combine order, shuffle seed,
     final sha, row count by kind, contamination result).

The final corpus sha is a pure function of the two committed source shas and the
shuffle seed, so it is stable across rebuilds and machines. ``--verify-corpus``
re-derives it twice in memory, asserts the two rebuilds are byte-identical, and
asserts the committed corpus + the receipt pin + the frozen constant all agree —
without writing anything (used by ``run.py --smoke``).

Usage:
  python scripts/build_corpus.py                 # build corpus + receipt (writes)
  python scripts/build_corpus.py --verify-corpus # fail-closed verify only (no writes)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parents[1]

# --- frozen build identities -------------------------------------------------
# COMBINE_ORDER: concatenate the self_repair block first, then why_comment, THEN
# shuffle. (The order only affects the pre-shuffle list; the seeded shuffle then
# interleaves. It is frozen so the corpus sha is reproducible.)
SOURCES = (
    (
        "self_repair",
        EXP / "data" / "source_corpora" / "sft_self_repair.jsonl",
        "920cb228172677f005bdbc4501f593ce60dc7a9c4f22cbf177f05660ffc392cb",
        "qwen35_4b_self_repair_install",
    ),
    (
        "why_comment",
        EXP / "data" / "source_corpora" / "sft_why_comment.jsonl",
        "040be350678ea0337b8fe0607f783aba9e9071f789471b0ea00f7ce1ebef2962",
        "qwen35_4b_why_comment_install",
    ),
)
SHUFFLE_SEED = 93570
COMBINE_ORDER = tuple(kind for kind, _p, _s, _o in SOURCES)

COMBINED = EXP / "data" / "sft_repair_why_stack.jsonl"
RECEIPT = EXP / "data" / "stack_corpus_receipt.json"
# The frozen union sha (pure function of the two source shas + the shuffle seed).
COMBINED_SHA256 = "2462c93ea2a8dcfbd9413e1c6115ed1456ad438e5dabfdc01e924be6148ddbe5"
EXPECTED_ROWS_BY_KIND = {"self_repair": 504, "why_comment": 504}
EXPECTED_TOTAL = 1008

# Parents' committed distinctive-shared-7-gram counts (verification aids from
# their curriculum_receipt.json). The union's code n-grams are the UNION of the
# two parents' code n-grams, so if each parent shares 0 distinctive 7-grams with
# the benchmark n-grams, the union shares 0 as well. A present-only HF-cache aid
# in tests/test_contamination.py re-verifies this over the combined corpus.
PARENT_DISTINCTIVE_NGRAMS = {"self_repair": 0, "why_comment": 0}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources() -> None:
    """Fail-closed: every in-cell source copy must match its frozen sha pin."""
    for kind, path, pinned, _origin in SOURCES:
        if not path.is_file():
            raise SystemExit(f"source corpus missing: {path}")
        observed = sha256_file(path)
        if observed != pinned:
            raise SystemExit(
                f"source corpus sha mismatch for {kind}: {observed} != pinned {pinned}"
            )


def source_lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_bytes() -> bytes:
    """Deterministically build the union corpus bytes (verifies sources first)."""
    verify_sources()
    lines: list[str] = []
    for _kind, path, _sha, _origin in SOURCES:
        lines.extend(source_lines(path))
    if len(lines) != EXPECTED_TOTAL:
        raise SystemExit(f"expected {EXPECTED_TOTAL} combined rows, got {len(lines)}")
    random.Random(SHUFFLE_SEED).shuffle(lines)
    return ("\n".join(lines) + "\n").encode("utf-8")


def kind_counts(payload: bytes) -> dict[str, int]:
    counts: dict[str, int] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        kind = json.loads(line).get("kind")
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def row_text(row: dict) -> str:
    parts: list[str] = []
    for message in row.get("messages", []):
        content = message.get("content")
        if isinstance(content, str):
            parts.append(content)
    for key in ("think", "answer"):
        value = row.get(key)
        if isinstance(value, str):
            parts.append(value)
    return "\n".join(parts)


def banned_audit(payload: bytes) -> dict:
    """Offline whole-word banned-benchmark-name audit over the union (prompt +
    think + answer of every row). The always-on, cache-independent gate."""
    sys.path.insert(0, str(EXP / "scripts"))
    import contamination as contam  # noqa: PLC0415

    banned = contam.banned_names()
    fixture = contam.load_fixture()
    hits: dict[str, list[str]] = {}
    for line in payload.decode("utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        found = contam.whole_word_hits(row_text(row), banned)
        if found:
            hits[row.get("task_id", "?")] = sorted(found)
    return {
        "banned_function_names": fixture["function_name_count"],
        "banned_after_whitelist": len(banned),
        "banned_hits": len(hits),
        "offending_task_ids": hits,
    }


def build_receipt(payload: bytes) -> dict:
    counts = kind_counts(payload)
    banned = banned_audit(payload)
    if banned["banned_hits"] != 0:
        raise SystemExit(f"UNION contamination: {banned['banned_hits']} banned-name hits")
    return {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "method": "deterministic_union_of_two_committed_source_corpora",
        "combine_order": list(COMBINE_ORDER),
        "shuffle_seed": SHUFFLE_SEED,
        "sources": [
            {
                "kind": kind,
                "path": str(path.relative_to(EXP).as_posix()),
                "sha256": pinned,
                "rows": EXPECTED_ROWS_BY_KIND[kind],
                "origin_experiment": origin,
            }
            for kind, path, pinned, origin in SOURCES
        ],
        "corpus": str(COMBINED.relative_to(EXP).as_posix()),
        "corpus_sha256": sha256_bytes(payload),
        "rows": EXPECTED_TOTAL,
        "kinds": counts,
        "contamination": {
            "banned_function_names": banned["banned_function_names"],
            "banned_after_whitelist": banned["banned_after_whitelist"],
            "banned_hits": banned["banned_hits"],
            "ngram_overlap": {
                "ngram_n": 7,
                "status": "inherited_clean",
                "shared_ngrams_distinctive": 0,
                "self_repair_distinctive_shared_ngrams": PARENT_DISTINCTIVE_NGRAMS["self_repair"],
                "why_comment_distinctive_shared_ngrams": PARENT_DISTINCTIVE_NGRAMS["why_comment"],
                "note": (
                    "No NEW generation: the union's code 7-grams are the UNION of the two "
                    "parents' code 7-grams. Each parent recorded 0 distinctive shared 7-grams "
                    "vs the HumanEval+MBPP benchmark 7-grams (their curriculum_receipt.json), "
                    "so the union shares 0 as well. A present-only HF-cache aid re-verifies "
                    "this over the combined corpus in tests/test_contamination.py."
                ),
            },
        },
    }


def write_all() -> dict:
    payload = build_bytes()
    if sha256_bytes(payload) != COMBINED_SHA256:
        raise SystemExit(
            f"built corpus sha {sha256_bytes(payload)} != frozen pin {COMBINED_SHA256}"
        )
    receipt = build_receipt(payload)
    COMBINED.write_bytes(payload)
    RECEIPT.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return receipt


def verify_corpus() -> None:
    """Fail-closed verification (no writes): determinism + committed + pins agree."""
    verify_sources()
    first = build_bytes()
    second = build_bytes()
    if first != second:
        raise SystemExit("non-deterministic build: two in-memory rebuilds differ")
    observed = sha256_bytes(first)
    if observed != COMBINED_SHA256:
        raise SystemExit(f"rebuild sha {observed} != frozen pin {COMBINED_SHA256}")
    counts = kind_counts(first)
    if counts != EXPECTED_ROWS_BY_KIND:
        raise SystemExit(f"row-by-kind mismatch: {counts} != {EXPECTED_ROWS_BY_KIND}")
    if not COMBINED.is_file():
        raise SystemExit(f"committed corpus missing: {COMBINED}")
    if sha256_file(COMBINED) != COMBINED_SHA256:
        raise SystemExit("committed corpus sha differs from the frozen pin")
    if not RECEIPT.is_file():
        raise SystemExit(f"committed receipt missing: {RECEIPT}")
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    if receipt.get("corpus_sha256") != COMBINED_SHA256:
        raise SystemExit("receipt corpus_sha256 differs from the frozen pin")
    if receipt.get("kinds") != EXPECTED_ROWS_BY_KIND:
        raise SystemExit("receipt kinds differ from the frozen row-by-kind counts")
    banned = banned_audit(first)
    if banned["banned_hits"] != 0:
        raise SystemExit(f"UNION contamination: {banned['banned_hits']} banned-name hits")
    print(
        f"PASS: repair_why_stack union verified — {EXPECTED_TOTAL} rows "
        f"({EXPECTED_ROWS_BY_KIND['self_repair']} self_repair + "
        f"{EXPECTED_ROWS_BY_KIND['why_comment']} why_comment), deterministic sha "
        f"{COMBINED_SHA256}, 0 banned-name hits on the union."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--verify-corpus", action="store_true",
        help="fail-closed verify the committed union (determinism + sha pins + contamination); no writes",
    )
    args = parser.parse_args()
    if args.verify_corpus:
        verify_corpus()
        return 0
    receipt = write_all()
    print(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
