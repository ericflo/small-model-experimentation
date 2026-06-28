# qwen35_4b_substrate_coverage_ladder

## Purpose

This standalone experiment tests the oracle substrate ceiling before any model training. The question is whether a fixed or mined executable substrate can express held-out MBPP tasks that a large direct sampling pool did not solve. Hidden tests are used only for measurement of oracle coverage and false-pass rate.

## Protocol

1. Build the residual task set from local baseline coverage artifacts.
2. Prepare a train-split reference library for retrieval/transplant controls.
3. Run smoke search on two residual tasks.
4. Run pilot search with the manual substrate.
5. Run the full ladder: manual core, manual expanded, retrieved transplant, and combined.
6. Generate figures and a final report.

## Running Notes

- 2026-06-26: Created the experiment package and copied local baseline coverage inputs into `data/`.
- 2026-06-26: Prepared the dataset. The K128-residual set contains 9 MBPP held-out tasks: 16, 26, 31, 39, 43, 44, 48, 60, and 77. The retrieval library contains 374 MBPP train-split reference entries.
- 2026-06-26: Smoke run on two residual tasks passed. Both tasks had hidden-test-correct substrate candidates, and both also had public-test-passing hidden-wrong candidates.
- 2026-06-26: Manual-only pilot v1 solved 8/9 residual tasks. The miss was the top-k-frequency task, where the substrate had a heap ranking candidate but returned heap-array order instead of heap-pop order.
- 2026-06-26: Added the generic heap-pop-order top-k-frequency kernel and reran the manual pilot. Manual expanded coverage reached 9/9 on hidden tests.
- 2026-06-26: Main ladder run completed. Manual core solved 4/9; manual expanded solved 9/9; retrieved train-reference transplants solved 0/9 and added 8 public-test-passing hidden-fail candidates. Combined coverage is 9/9, with 17 visible-pass hidden-fail candidates out of 30 visible-pass candidates.
- 2026-06-26: Gate interpretation corrected after review. The weak expressivity gate cleared, but the meaningful reusable-substrate gate did not: 9/9 coverage came from task-specific manual templates, while the genuine reuse arm solved 0/9.
