# Qwen3.5-4B Deep-Advantage MOPD Experiment Log

## 2026-07-12 — intake and pre-output design

- Created a new result-bearing directory rather than modifying the completed
  two-teacher experiment.
- Selected the deep-only branch because deep independently passed both
  same-prefix audit contrasts in both predecessor blocks; MOPD remained
  untested only because quick was separately mandatory.
- Reused the exact immutable 40/60 soup by SHA-256 instead of constructing a
  numerically new starting checkpoint. New training artifacts have a separate
  external root.
- Froze two new route blocks, the unchanged strict deep-over-quick-and-student
  rule, five-update locality, four 60-deep/20-soup rounds, three primary seeds,
  and unconditional final comparisons against sources, router, controls, and
  sample-more.
- Added two direct mechanism controls: deep targets on one-to-one matched
  non-deep-selected states, and quick targets on the exact selected states.
  The off-policy continuation control and parameter soups remain.
- Copied the parent harness and procedural gym, then adapted quotas, route gate,
  target-cache inventory, locality mixture, controls, and confirmation without
  generating task-model output.
- Passed 50 isolated tests and all 14 family selftests; verified the exact
  quick, deep, and immutable-soup file hashes. The smoke receipt contains no
  task-model generation.
- Committed the complete frozen design at `1ef1f5ad`, pushed it to shared
  `main`, and wrote `runs/preregistration_receipt.json` with byte hashes for all
  frozen files before any Qwen load.
