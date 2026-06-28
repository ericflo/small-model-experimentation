# Experiment Log

## 2026-06-27

- Created standalone experiment package.
- Copied generic safe Python execution and JSONL utilities into this package.
- Planned no-training code ABI oracle coverage ladder over MBPP test tasks.
- Primary gate: measure whether a verified primitive ABI can express decomposable real-code task slices before any compiler training.
- Implemented `scripts/run_coverage_ladder.py`.
- Ran core ABI rung on 160 MBPP test records:
  - oracle coverage: 20/160 (12.5%)
  - first visible-consistent candidate correct: 17/160 (10.6%)
  - mean candidates/task: 66.9
- Inspected misses and added broad reusable primitives for row-sum sorting, counters, string/list transforms, simple formulas, bit predicates, and regex-style tasks.
- Ran expanded ABI rung:
  - oracle coverage: 92/160 (57.5%)
  - first visible-consistent candidate correct: 67/160 (41.9%)
  - mean candidates/task: 164.2
- Inspected remaining misses and added a final broad utility layer: tuple/list conversions, sequence recurrences, run-length encoding, range sums, divisor/bit counts, pairwise reducers, and small geometry.
- One final-rung run hung because some enumeration-style primitives were tried outside their intended domain. Killed that run and added explicit numeric/domain bounds before rerunning.
- Ran final ABI rung:
  - oracle coverage: 134/160 (83.75%)
  - first visible-consistent candidate correct: 95/160 (59.4%)
  - task-level visible-pass/no-full-winner rate among visible-any tasks: 13/147 (8.8%)
  - candidate-level hidden-wrong rate among visible-consistent candidates: 628/793 (79.2%)
  - mean candidates/task: 224.0
- Generated charts and standalone report at `reports/final_report.md`.
- Gate read: code-ABI oracle coverage is high enough to justify a compiler-training pilot on a frozen ABI, but selection/verification remains unresolved and the result is test-suite coverage rather than proof of semantic equivalence.
