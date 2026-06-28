# Experiment Log

## 2026-06-24

- Created a fresh standalone multi-seed reattribution experiment directory.
- Core question: does the executable latent compiler's length-24 performance remain stable across random seeds, and which controlled arm has the best mean behavior?
- Result-bearing arms:
  - `max24_curriculum`
  - `expand_copy`
  - `max24_no_curriculum`
- Matched seed set planned for main: `123,456,789`.
- Large checkpoints are configured under `/workspace/large_artifacts/qwen_compiler_multiseed_reattribution/checkpoints`.
- Planned process:
  - Smoke: one tiny run to validate the forked trainer, seed columns, report generation, and artifact paths.
  - Pilot: reduced two-seed, three-arm grid to validate aggregation and charts before the long run.
  - Main: three arms times three seeds, then final Markdown and HTML reports with variance charts.

### Smoke

- Command: `python src/run_multiseed_suite.py --phase smoke --seeds 123,456,789 --arms max24_curriculum,expand_copy,max24_no_curriculum`
- Completed run: `smoke_max24_curriculum_seed123`
- Outcome: passed.
- Verified:
  - The forked trainer loads Qwen/Qwen3-4B, attaches QLoRA, trains through staged max-24 curriculum, and writes run CSV/JSON outputs.
  - `seed` is present in run metadata.
  - The analysis script writes both Markdown and HTML reports.
  - Large artifact path is configured separately; smoke used `--save_checkpoints 0`, so no large files were produced.

### Pilot

- First pilot attempt caught a runner bug before any result-bearing run: the reduced pilot override supplied three stage-step counts to the one-stage `max24_no_curriculum` arm.
- Fix: preserve each arm's stage structure in the runner; one-stage arms now receive a single reduced pilot budget.
- Corrected pilot command: `python src/run_multiseed_suite.py --phase pilot --seeds 123,456,789 --arms max24_curriculum,expand_copy,max24_no_curriculum`
- Completed runs:
  - `pilot_max24_curriculum_seed123`
  - `pilot_expand_copy_seed123`
  - `pilot_max24_no_curriculum_seed123`
  - `pilot_max24_curriculum_seed456`
  - `pilot_expand_copy_seed456`
  - `pilot_max24_no_curriculum_seed456`
- Outcome: passed after the runner fix.
- Verified:
  - The analyzer aggregates by arm and seed.
  - Diagnostic Markdown/HTML reports are generated.
  - Figures are generated: mean accuracy with seed spread, per-seed heatmap, standard accuracy by seed, length curve, training loss, and training state accuracy.
- Interpretation: pilot rows are diagnostic only and are not result-bearing.

### Main

- Planned command: `python src/run_multiseed_suite.py --phase main --seeds 123,456,789 --arms max24_curriculum,expand_copy,max24_no_curriculum`
- Main configuration: Qwen/Qwen3-4B QLoRA, 512-wide compiler, LoRA rank 8, batch 8, 512 train examples, 64 examples per single-template split, 32 paired examples per paired split.
- Completed command: `python src/run_multiseed_suite.py --phase main --seeds 123,456,789 --arms max24_curriculum,expand_copy,max24_no_curriculum`
- Completed runs:
  - `main_max24_curriculum_seed123`
  - `main_expand_copy_seed123`
  - `main_max24_no_curriculum_seed123`
  - `main_max24_curriculum_seed456`
  - `main_expand_copy_seed456`
  - `main_max24_no_curriculum_seed456`
  - `main_max24_curriculum_seed789`
  - `main_expand_copy_seed789`
  - `main_max24_no_curriculum_seed789`
- Outcome: all nine result-bearing runs completed and the analysis step passed.
- Final standard-L24 executable accuracy, mean +/- seed std:
  - `expand_copy`: 43.2% +/- 40.9, range 0.0% to 81.2%.
  - `max24_curriculum`: 14.1% +/- 11.3, range 1.6% to 23.4%.
  - `max24_no_curriculum`: 0.5% +/- 0.9, range 0.0% to 1.6%.
- Final paired-L24 executable accuracy, mean +/- seed std:
  - `expand_copy`: 60.4% +/- 30.9.
  - `max24_curriculum`: 40.1% +/- 34.7.
  - `max24_no_curriculum`: 24.5% +/- 25.0.
- Key interpretation:
  - Copied expansion has the best mean in this seed set, but the seed spread is very large.
  - No-curriculum training is weak on standard L24 even when some wording splits score much higher.
  - High state-prefix recovery does not reliably translate into exact length-24 program recovery.
  - Single-seed results are not sufficient for attribution in this harness.
- Reports:
  - `reports/qwen_compiler_multiseed_reattribution_report.md`
  - `reports/qwen_compiler_multiseed_reattribution_report.html`
  - `reports/checkpoint_manifest_all.csv`
