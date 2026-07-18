#!/usr/bin/env python3
"""Build the scale-ladder corpora and the sha-pinned ladder manifest.

The ladder trains the SAME WHY curriculum at four rung sizes to find the PEAK of
the WHY scaling curve before overfit/collapse. Each rung is a deterministic
corpus of the given size from the shared construction seed (94100) — a different
N, the SAME generator — so the rungs are nested draws of one high-diversity
process, not four unrelated datasets.

For each rung this:
  * generates exactly N verified rows (every row truth-audited by real execution;
    see gen_why_scale_curriculum),
  * runs the whole-word banned-name audit (ZERO hits) and the present-only
    code-only distinctive 7-gram overlap aid (ZERO distinctive shared spans),
  * records the diversity report (families, categories, unique-program %, distinct
    normalized WHY templates) and the character-length distribution,
  * writes the corpus to the gitignored large-artifacts tree, and
  * pins its sha256 + row count + audits into ``data/ladder_manifest.json``.

The corpora are large and DETERMINISTICALLY regenerable, so they live under
``large_artifacts/`` (gitignored) and are reproduced by rerunning this script;
the small committed manifest is the standalone contract (generator sha + fixture
sha + per-rung corpus sha), and ``--verify`` regenerates each rung and checks its
sha against the manifest without writing. Run under the repo .venv (CPU only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import contamination as contam  # noqa: E402
import gen_why_scale_curriculum as gen  # noqa: E402

EXP = Path(__file__).resolve().parents[1]
ROOT = EXP.parents[1]

CONSTRUCTION_SEED = gen.CONSTRUCTION_SEED  # 94100
LADDER_SIZES = (2000, 5000, 10000, 20000)
CORPUS_ROOT = ROOT / "large_artifacts" / EXP.name / "corpora"
MANIFEST = EXP / "data" / "ladder_manifest.json"
GENERATOR = EXP / "scripts" / "gen_why_scale_curriculum.py"
FIXTURE = EXP / "data" / "contamination" / "banned_function_names.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_path(rows: int) -> Path:
    return CORPUS_ROOT / f"why_scale_{rows}.jsonl"


def build_rung(rows: int, *, run_ngram: bool, bench_grams: set | None = None) -> tuple[str, dict]:
    """Generate one rung; return (payload, rung_receipt). Pure function of (seed, rows)."""
    corpus = gen.generate_curriculum(CONSTRUCTION_SEED, rows)
    summary = gen.validate_generated(corpus)
    diversity = gen.diversity_report(corpus)
    length_stats = gen._length_stats(corpus)
    payload = "".join(json.dumps(gen.public_row(r), ensure_ascii=False) + "\n" for r in corpus)

    overlap: dict = {"status": "not_run"}
    if run_ngram:
        if bench_grams is None:
            overlap = gen.contamination_ngram_overlap(corpus)
        else:
            corpus_grams: set = set()
            for r in corpus:
                corpus_grams |= gen.code_grams_no_comments(r["_audit"]["clean_code"])
            distinctive = contam.distinctive_overlap(corpus_grams, bench_grams)
            overlap = {
                "status": "checked",
                "ngram_n": contam.NGRAM_N,
                "benchmark_ngrams": len(bench_grams),
                "corpus_ngrams": len(corpus_grams),
                "shared_ngrams_structural_idiom": len(corpus_grams & bench_grams),
                "shared_ngrams_distinctive": len(distinctive),
            }
        if overlap.get("shared_ngrams_distinctive"):
            raise SystemExit(f"rung {rows}: {overlap['shared_ngrams_distinctive']} distinctive shared n-grams")

    receipt = {
        "rows": rows,
        "seed": CONSTRUCTION_SEED,
        "path": str(corpus_path(rows).relative_to(ROOT)),
        "corpus_sha256": gen.sha256_text(payload),
        "unique_programs": summary["unique_programs"],
        "categories": summary["categories"],
        "diversity": {
            "distinct_families": diversity["distinct_families"],
            "distinct_categories": diversity["distinct_categories"],
            "unique_program_pct": diversity["unique_program_pct"],
            "distinct_normalized_why_templates": diversity["distinct_normalized_why_templates"],
        },
        "length_stats": length_stats,
        "contamination": {
            "banned_function_names": len(contam.banned_names()),
            "banned_hits": 0,
            "ngram_overlap": overlap,
        },
    }
    return payload, receipt


def build(sizes=LADDER_SIZES, *, write: bool = True, run_ngram: bool = True) -> dict:
    bench_grams = None
    if run_ngram:
        try:
            bench_grams = contam.benchmark_ngrams(contam.build_code_tokens_from_cache())
        except Exception as exc:  # noqa: BLE001 (cache absent -> skip aid, recorded per rung)
            bench_grams = None
            print(f"[build_ladder] n-gram aid unavailable ({type(exc).__name__}); rungs record skipped", flush=True)
    rungs = []
    for rows in sizes:
        started = time.perf_counter()
        payload, receipt = build_rung(rows, run_ngram=run_ngram and bench_grams is not None, bench_grams=bench_grams)
        if not run_ngram or bench_grams is None:
            receipt["contamination"]["ngram_overlap"] = {"status": "skipped_cache_absent"}
        if write:
            path = corpus_path(rows)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(payload, encoding="utf-8")
        receipt["wall_seconds"] = round(time.perf_counter() - started, 2)
        rungs.append(receipt)
        print(f"[build_ladder] rung {rows}: sha {receipt['corpus_sha256'][:16]} "
              f"families {receipt['diversity']['distinct_families']} "
              f"why_templates {receipt['diversity']['distinct_normalized_why_templates']} "
              f"({receipt['wall_seconds']}s)", flush=True)
    manifest = {
        "schema_version": 1,
        "experiment_id": EXP.name,
        "construction_seed": CONSTRUCTION_SEED,
        "generator": str(GENERATOR.relative_to(ROOT)),
        "generator_sha256": sha256_file(GENERATOR),
        "contamination_fixture_sha256": sha256_file(FIXTURE),
        "ladder_sizes": list(sizes),
        "rungs": rungs,
    }
    return manifest


def verify(sizes=LADDER_SIZES) -> dict:
    """Regenerate each rung and confirm its sha matches the committed manifest."""
    if not MANIFEST.is_file():
        raise SystemExit(f"no ladder manifest to verify: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("generator_sha256") != sha256_file(GENERATOR):
        raise SystemExit("ladder manifest generator_sha256 disagrees with the current generator")
    if manifest.get("contamination_fixture_sha256") != sha256_file(FIXTURE):
        raise SystemExit("ladder manifest contamination_fixture_sha256 disagrees with the current fixture")
    by_rows = {r["rows"]: r for r in manifest["rungs"]}
    checked = []
    for rows in sizes:
        pinned = by_rows.get(rows)
        if pinned is None:
            raise SystemExit(f"rung {rows} absent from the manifest")
        corpus = gen.generate_curriculum(CONSTRUCTION_SEED, rows)
        payload = "".join(json.dumps(gen.public_row(r), ensure_ascii=False) + "\n" for r in corpus)
        observed = gen.sha256_text(payload)
        if observed != pinned["corpus_sha256"]:
            raise SystemExit(f"rung {rows} sha mismatch: {observed} != {pinned['corpus_sha256']}")
        checked.append({"rows": rows, "corpus_sha256": observed})
    return {"verified": checked}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="regenerate + check shas vs the committed manifest (no write)")
    parser.add_argument("--sizes", type=int, nargs="+", default=list(LADDER_SIZES))
    parser.add_argument("--no-ngram", action="store_true", help="skip the present-only n-gram overlap aid")
    parser.add_argument("--no-write", action="store_true", help="build + manifest but do not write corpora")
    parser.add_argument("--out", type=Path, default=MANIFEST)
    args = parser.parse_args()

    if args.verify:
        print(json.dumps(verify(tuple(args.sizes)), indent=2, sort_keys=True))
        return 0

    manifest = build(tuple(args.sizes), write=not args.no_write, run_ngram=not args.no_ngram)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(args.out), "rungs": [r["rows"] for r in manifest["rungs"]]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
