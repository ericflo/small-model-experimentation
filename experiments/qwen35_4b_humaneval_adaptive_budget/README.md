# Qwen3.5-4B HumanEval Adaptive Evidence Budget

**Status:** finished

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/qwen35_4b_humaneval_adaptive_budget_report.md](reports/qwen35_4b_humaneval_adaptive_budget_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

This standalone experiment tests whether Qwen3.5-4B can be posttrained as a STOP/MORE controller for an executable Python verifier on HumanEval tasks. The verifier generates candidate implementations, chooses unlabeled probes by target-independent output-agreement split, and commits the first candidate in the largest output-agreement cluster. The model only decides whether to commit or spend one more executable probe.
