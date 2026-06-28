# Qwen Progressive Repair Compiler Experiment Log

## Objective

Train a deployable repair-selection layer for a frozen Qwen-attached numeric
compiler. The learned verifier should choose local executable-program repairs
without test-time access to the true answer or true trajectory.

## Success Criteria

- Keep this experiment in its own directory with source, runs, analysis, reports,
  and checkpoint manifest.
- Store bulky checkpoints under `large_artifacts/qwen_progressive_repair_compiler/`.
- Evaluate base compiler selection, learned verifier selection, pair reranking,
  and oracle trajectory selection.
- Produce CSVs, figures, a markdown report, and an HTML report.
- Run smoke/pilot/main configurations and record outcomes.

## Planned Runs

### Smoke

`smoke_progressive_repair`

- Tiny data and a short two-stage candidate curriculum.
- Purpose: verify checkpoint loading, candidate generation, staged verifier
  training, metric writing, checkpoint writing, and report generation.
- Completed successfully. Metrics are not used for conclusions because the
  top-2/one-edit neighborhood and tiny held-out sets were intentionally too
  small.

### Pilot

`pilot_progressive_repair_s96`

- Moderate data with full two-edit validation.
- Purpose: test whether progressive candidate-space training is stable before
  running the larger configuration.
- Completed successfully.
- Fresh paired length-24 result: base 28.1%, learned verifier 40.6%, pair
  rerank 42.2%, oracle 87.5%.
- Validation result: base 37.5%, learned verifier 45.8%, oracle 85.4%.
- This justified running the larger configuration.

### Main

`main_progressive_repair_s512`

- 512 verifier-training examples, 128 validation examples, and fresh held-out
  standard, paraphrase, and paired splits.
- Candidate curriculum:
  `small:2:1:8:3,medium:3:1:16:4,full:3:2:24:11`.
- Completed successfully in 874.9 seconds.
- Best validation epoch: 14.
- Fresh standard length-24: base 28.5%, learned verifier 47.7%, oracle 90.6%.
- Fresh paraphrase length-24: base 28.5%, learned verifier 44.9%, oracle 86.7%.
- Fresh paired length-24: base 30.3%, learned verifier 48.6%, pair rerank
  53.7%, oracle 88.1%.
- The small and medium curriculum stages did not materially improve full-space
  validation accuracy; most of the gain appeared after full-neighborhood
  training began.

## Running Notes

Initialized the scaffold with self-contained source files, a copied Qwen compiler
checkpoint under the experiment's large-artifact root, and an analyzer that
generates markdown and HTML reports.
