# Qwen3.5-4B Foofah Program Strategy Portfolio

Standalone experiment for searching a small portfolio of executable program-generation strategies for Foofah table transformations.

The experiment uses only the local Foofah benchmark files under `/workspace/large_artifacts/external_sources/foofah_benchmarks`, with cases materialized in `data/cases.jsonl`. Strategy selection is performed on train/dev task families, then the frozen portfolio is evaluated on held-out families.

