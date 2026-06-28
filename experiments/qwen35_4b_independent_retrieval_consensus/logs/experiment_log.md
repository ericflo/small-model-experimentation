# Experiment Log

## 2026-06-26

- Created standalone experiment package.
- Copied base direct K=4 records, verified algorithm library, and generic evaluator/model utilities into this package.
- Localized experiment identity to `qwen35_4b_independent_retrieval_consensus`.
- Planned measurements: independent retrieval gate, independent vs same-neighborhood adaptation pools, disagreement-input consensus selector, and direct sample-more baseline on the same residual tasks.

## 2026-06-26 Retrieval Independence Gate

- Built retrieval plan over 24 base-miss residual tasks and 364 verified library entries.
- Same-neighborhood top-6 mean pairwise code distance: 0.701.
- MMR independent top-6 mean pairwise code distance: 0.803.
- Same-neighborhood top-6 mean pairwise task-token distance: 0.635.
- MMR independent top-6 mean pairwise task-token distance: 0.700.
- Gate passed; proceed to generation.

## 2026-06-26 Adaptation Pools

- Generated independent top-6 adaptation pool: 24 records, 144 model calls, coverage 7/24, pass1 proxy 3/24, forward tokens 51,282.
- Generated same-neighborhood top-6 adaptation pool: 24 records, 144 model calls, coverage 9/24, pass1 proxy 3/24, forward tokens 52,001.
- Early read: independence increased source diversity but did not increase pool coverage; same-neighborhood control is stronger at the coverage level.

## 2026-06-26 Direct Sample-More Baseline

- Generated direct K12 baseline on the same 24 residual tasks.
- Direct K12: 288 model calls, coverage 7/24, pass1 proxy 1/24, forward tokens 74,780.
- Same-neighborhood retrieval-adapt remains the strongest pool coverage before consensus selection.
