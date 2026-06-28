# Experiment Log

## 2026-06-26

- Created standalone experiment package.
- Copied only generic evaluator/model utilities and retrieval scripts into the standalone package.
- Localized the experiment identity to `qwen35_4b_retrieval_adapt_verify_scale`.
- Copied the 80-task MBPP heldout direct K=4 baseline into this package as the scale substrate.
- Added selector scripts for target-independent agreement probes and frozen-Qwen visible-candidate reranking.
- Rebuilt base manifest: 80 records, 56/80 direct coverage, 24 residual misses.
- Verified 364 MBPP train reference algorithms for the retrieval library.
- Planned top-3 semantic, random, and shuffled retrieval for all 24 residual tasks.
- Ran copy/rename retrieval: 1/24 residual pool recovery.
- Ran generated retrieval adaptation arms:
  - semantic top-3: 8/24 residual pool recovery, 24,352 forward tokens;
  - random top-3: 4/24 residual pool recovery, 25,603 forward tokens;
  - shuffled-query top-3: 3/24 residual pool recovery, 26,127 forward tokens.
- Evaluated copy+semantic selectors:
  - residual oracle pool recovery: 8/24;
  - first-visible selected recovery: 7/24, with 7 visible-pass hidden-wrong commits;
  - agreement-consensus selected recovery: 6/24, with 8 visible-pass hidden-wrong commits;
  - frozen-Qwen visible rerank selected recovery: 5/24, with 8 visible-pass hidden-wrong commits.
- Generated final report, machine-readable summary, and four figures under `reports/`.
