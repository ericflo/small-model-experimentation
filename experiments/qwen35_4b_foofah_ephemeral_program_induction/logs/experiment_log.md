# Experiment Log

## 2026-06-27

- Created standalone ephemeral-program-induction experiment package.
- Objective: compare direct Qwen table output with generated executable Python transformers on the same Foofah held-out `TestAnswer` rows.
- Planned iteration:
  - Build Foofah case JSONL.
  - Smoke test strict induction and context-aware program prompts.
  - Select the better prompt for the full 250-case run.
  - Report direct, program, agreement, fallback, and oracle-union metrics with figures.
- Built 250 Foofah cases across 50 families.
- Safe executor smoke passed, including safe-import stripping for generated programs.
- Prefix smoke, 8 cases:
  - strict induction: direct 8/8, program hidden 5/8, visible-pass 5/8, agreement precision 5/5.
  - context prompt: direct 8/8, program hidden 7/8, visible-pass 7/8, agreement precision 7/7.
- Hard spread smoke, 25 cases with stride 10:
  - context prompt: direct 11/25, program hidden 3/25, oracle union 12/25, visible-pass 7/25, agreement precision 2/6.
  - context_v2 prompt: direct 11/25, program hidden 3/25, oracle union 11/25, visible-pass 6/25, agreement precision 3/4.
- Selected `context` prompt for full run because it preserved the only direct-failure recovery in the hard spread. `context_v2` is retained as an iteration record: more conservative agreement, less coverage.
- Full 250-case `context` run complete:
  - Direct JSON generation: 138/250 exact (55.2%), parse 236/250 (94.4%).
  - Visible-verified generated program: 38/250 exact (15.2%).
  - Direct with program fallback on direct parse failure: 139/250 exact (55.6%).
  - Oracle union, direct OR program: 148/250 (59.2%).
  - Program-only recoveries: 10 cases.
  - Visible-pass hidden-wrong: 9/47 (19.1%).
  - Direct/program agreement: 34 cases, 28 correct (82.4% precision).
- Generated final report and figures under `reports/`.
