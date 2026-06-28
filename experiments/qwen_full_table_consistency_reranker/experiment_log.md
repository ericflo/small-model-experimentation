# Qwen Full-Table Consistency Reranker Experiment Log

## Objective

Test whether multiple row-level model guesses can be converted into task-level consistency by scoring entire candidate output tables against the examples.

The unit of selection is the full held-out output vector, not a natural-language rule, a deterministic program, or one row at a time.

## Initial Plan

1. Create a new standalone experiment directory.
2. Load public text-transformation tasks.
3. Generate multiple candidate outputs per held-out row using Qwen prompt and sampling variants.
4. Build full-table candidates from method outputs, majority vote, and frequency-ranked row-candidate combinations.
5. Measure oracle coverage:
   - row-candidate oracle: each row has the gold answer somewhere in its candidate set.
   - table-candidate oracle: the exact gold table appears in the enumerated table candidates.
6. Train a cross-validated consistency reranker on task-level candidates.
7. Include shuffled-label and heuristic controls.
8. Write CSVs, charts, Markdown report, and HTML report.

## Run Notes

### 2026-06-27 07:19 UTC - Scaffold and no-Qwen smoke

- Created the standalone experiment directory.
- Added a runner that loads public text-transformation tasks, generates row candidates, enumerates full-table candidates, trains cross-validated rerankers, writes CSVs, charts, Markdown, and HTML.
- Ran `smoke_no_qwen` to test the non-model path. The run used no Qwen candidates, so the metrics were intentionally meaningless, but the pipeline produced all expected artifacts.

### 2026-06-27 07:21 UTC - Real-Qwen pilot

- Ran `pilot_qwen_6` on 6 tasks, 4 held-out rows, one sampled row candidate per row, and 128 max table candidates.
- Candidate generation made the exact table reachable on all 6 pilot tasks.
- Direct greedy solved 4/6 tasks; the table oracle solved 6/6.
- The learned reranker solved 3/6 and underperformed direct on this small split, while the heuristic tied direct.
- Fixed candidate bookkeeping so `row_majority` remains represented even when it produces the same output table as another method.

### 2026-06-27 07:31 UTC - Main run

- Ran `main_qwen_table_40` on 40 tasks, 4 train rows, up to 6 held-out rows, two sampled row candidates per row, and 512 max table candidates.
- Generated 1,100 Qwen candidate calls into the run-local cache.
- Main results:
  - `direct_row_greedy`: 72.3% row exact, 50.0% full-task exact.
  - `row_majority`: 72.7% row exact, 50.0% full-task exact.
  - `heuristic`: 72.7% row exact, 50.0% full-task exact.
  - `learned_reranker`: 72.3% row exact, 50.0% full-task exact.
  - `shuffled_label_reranker`: 72.5% row exact, 47.5% full-task exact.
  - `table_oracle`: 82.5% row exact, 62.5% full-task exact.
- Interpretation:
  - The candidate generator creates real headroom: exact full tables are reachable on 25/40 tasks versus 20/40 solved by direct greedy.
  - The learned reranker does not capture that headroom: it solves 20/40 and captures 0/5 reachable-headroom tasks.
  - Candidate-level discrimination is strong (mean AUC 92.9% versus 36.4% for shuffled-label control), but task-level selection remains unsolved.

### 2026-06-27 07:32 UTC - Report hardening

- Added explicit headroom-capture diagnostics.
- Added learned-selection-change diagnostics.
- Regenerated the standalone Markdown and HTML reports with eight charts.
