# Qwen3.5-4B Oracle-Distilled Semantic Verifier

**Status:** finished

This top-level README was generated during repository normalization because the imported experiment did not include one.

- Source track: `track-z`
- Primary report: [reports/qwen35_4b_oracle_distilled_semantic_verifier_report.md](reports/qwen35_4b_oracle_distilled_semantic_verifier_report.md)
- Metadata: [metadata.yaml](metadata.yaml)

## How To Read

Start with the primary report, then inspect `data/`, `reports/`, `analysis/`, `src/`, and `scripts/` as available. This folder remains self-contained; do not move its run data into shared directories.

## Summary

Train Qwen3.5-4B as a deployable verifier for Python candidate programs. The training oracle labels visible-test-passing candidates by hidden-test execution. At inference, the model sees only the task, public tests, public-test status, and candidate code; it ranks candidates by the probability that they pass hidden tests.
