# Qwen Pairwise Table Judge Experiment Log

## Objective

Test whether a language model can select the more task-consistent full output table when shown concrete alternatives for the same transformation task.

The primary method is pairwise table judging. The experiment includes:

1. A deployable tournament over a non-label shortlist of candidate tables.
2. A diagnostic direct-vs-hidden-correct comparison where the correct table is present but not labeled as such.
3. Controls that remove or corrupt the task context.

The experiment is standalone: all run-local inputs, candidate pools, judgments, analyses, charts, and reports are stored under this directory.

## Initial Plan

1. Create a fresh experiment directory.
2. Copy a fixed candidate-table pool into the run directory.
3. Load the public text-transformation tasks and render the train examples/query rows.
4. Ask Qwen to judge candidate table pairs.
5. Compare deployable tournament, diagnostic direct-vs-correct judging, no-example control, shuffled-example control, and row-shuffled-candidate control.
6. Generate CSVs, charts, Markdown report, and HTML report.

## Run Notes

### 2026-06-27 08:06 UTC - Scaffold

- Created a fresh experiment directory at `/workspace/experiments/qwen_pairwise_table_judge`.
- Created a separate large-artifact root at `/workspace/large_artifacts/qwen_pairwise_table_judge`.
- Symlinked the public PROSE benchmark data under the large-artifact root instead of duplicating it in the experiment directory.
- Implemented `src/qwen_pairwise_table_judge.py`.
- Verified syntax with `python -m py_compile`.

### 2026-06-27 08:07 UTC - No-Qwen Smoke

Command:

```bash
python /workspace/experiments/qwen_pairwise_table_judge/src/qwen_pairwise_table_judge.py \
  --run_name smoke_no_qwen \
  --task_limit 6 \
  --heldout_cap 4 \
  --shortlist 4 \
  --no_qwen
```

Purpose: validate filesystem layout, candidate-pool copying, benchmark loading, CSV output, chart generation, Markdown report generation, and HTML report generation without spending model calls.

Fixes made during smoke:

- Replaced a pandas `Series` boolean fallback with an explicit `None` check.
- Fixed a `mode_summary.mode` attribute collision by indexing `mode_summary["mode"]`.

Smoke metrics are not interpreted because `--no_qwen` defaults uncached judgments to candidate A.

### 2026-06-27 08:08 UTC - Real-Qwen Pilot

Command:

```bash
python /workspace/experiments/qwen_pairwise_table_judge/src/qwen_pairwise_table_judge.py \
  --run_name pilot_qwen_6 \
  --task_limit 6 \
  --heldout_cap 4 \
  --shortlist 4
```

Result:

- 52 real model judgments.
- `table_oracle`: 100.0% full-task exact.
- `direct_row_greedy`: 66.7% full-task exact.
- `pairwise_tournament`: 66.7% full-task exact.
- `row_repair_diagnostic`: 16.7% full-task exact.

Pilot diagnosis:

- Qwen emitted parseable A/B choices.
- The small task sample was mostly saturated, so it was only used to validate mechanics and prompt parsing.

### 2026-06-27 08:09 UTC - Main Run

Command:

```bash
python /workspace/experiments/qwen_pairwise_table_judge/src/qwen_pairwise_table_judge.py \
  --run_name main_qwen_pairwise_40 \
  --task_limit 40 \
  --heldout_cap 6 \
  --shortlist 6
```

Result:

- 40 public text-transformation tasks.
- 336 unique cached model judgments.
- No blank choices in the cached judgment records.
- `table_oracle`: 62.5% full-task exact.
- `direct_row_greedy`: 50.0% full-task exact.
- `pairwise_tournament`: 47.5% full-task exact.
- `row_repair_diagnostic`: 80.0% full-task exact on the 25 tasks with an exact candidate.

Main diagnosis:

- The deployable pairwise tournament is a negative result: it is 2.5 points worse than direct greedy and does not capture the 12.5-point oracle headroom.
- The all-oracle-task diagnostic is misleading if read alone: normal direct-vs-hidden-correct judging picks the hidden-correct table 90.0% of the time, but most of those comparisons are saturated cases where direct and oracle are the same candidate.
- On the five true headroom tasks, Qwen picks candidate A regardless of semantics:
  - direct as A, hidden-correct as B: 0.0% picked hidden-correct.
  - hidden-correct as A, direct as B: 100.0% picked hidden-correct.
  - no examples, shuffled examples, and row-shuffled candidate controls with direct as A: 0.0% picked hidden-correct.
- The useful signal is row-local and diagnostic only: when hidden correct row replacements are explicitly supplied, row repair reaches 80.0% full-task exact. That is not deployable because it uses oracle alternatives.

### 2026-06-27 08:11 UTC - Report Hardening

- Added `analysis/diagnostic_headroom_summary.csv`.
- Added `analysis/tournament_changes.csv`.
- Added `analysis/figures/diagnostic_headroom_pick_oracle.png`.
- Updated the Markdown and HTML report to make the headroom-only position-bias result explicit.
- Verified the report is standalone and contains no references to earlier experiments.

Final artifacts:

- `src/qwen_pairwise_table_judge.py`
- `runs/main_qwen_pairwise_40/pairwise_judgments.csv`
- `runs/main_qwen_pairwise_40/judge_details.csv`
- `runs/main_qwen_pairwise_40/selected_tables.csv`
- `runs/main_qwen_pairwise_40/diagnostic_direct_vs_correct.csv`
- `analysis/summary.csv`
- `analysis/diagnostic_summary.csv`
- `analysis/diagnostic_headroom_summary.csv`
- `analysis/tournament_changes.csv`
- `analysis/figures/*.png`
- `reports/qwen_pairwise_table_judge_report.md`
- `reports/qwen_pairwise_table_judge_report.html`
