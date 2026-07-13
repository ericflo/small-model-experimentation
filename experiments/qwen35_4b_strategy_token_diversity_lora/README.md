# qwen35_4b_strategy_token_diversity_lora

**Status:** finished

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/final_report.md](reports/final_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

Can a small QLoRA adapter with explicit strategy tokens make extra samples on base-missed MBPP tasks behave like a more complementary ensemble, recovering misses at roughly the cost of one hot K32 arm instead of a three-policy union?
