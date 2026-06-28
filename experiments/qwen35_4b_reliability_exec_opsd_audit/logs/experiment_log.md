# Experiment Log

## 2026-06-26

- Created standalone experiment package.
- Copied local residual candidate pools, retrieval plan, verified library, and generic evaluator/model utilities into this package.
- Localized experiment identity to `qwen35_4b_reliability_exec_opsd_audit`.
- Planned three measurements: MAP likelihood selector, low-temperature semantic adaptation, and execution-grounded locality audit.

## 2026-06-26T22:13:20Z

- Completed semantic retrieval-adapt top-3 at T=0.0 and T=0.1, using the existing T=0.2 pool as baseline comparison.
- T=0.0: 24 records, coverage 7/24, pass1 proxy 4/24, forward tokens 25,057.
- T=0.1: 24 records, coverage 8/24, pass1 proxy 3/24, forward tokens 24,961.
- Both arms wrote records and manifests under `data/`.
- Next step: raw likelihood/MAP candidate scoring to test whether the base model's own probability favors hidden-correct visible-pass adaptations.

## 2026-06-26T22:16:00Z

- Completed raw likelihood/MAP selector scoring.
- Semantic temperature union pool coverage was 8/24. First-visible selected 8/24 hidden-correct with 6 visible-pass hidden-wrong selections. MAP selected 6/24 hidden-correct with 8 hidden-wrong selections.
- Copy+semantic T=0.2 pool coverage was 8/24. First-visible selected 7/24 hidden-correct with 7 hidden-wrong selections. MAP selected 6/24 hidden-correct with 8 hidden-wrong selections.
- Interpretation before final report: raw model likelihood is not a reliable selector for these near-miss retrieval-adapt pools.

## 2026-06-26T22:30:00Z

- Built 59 matched correct-vs-hidden-wrong adaptation pairs across 3 tasks.
- Pair set contains 216 position-matched fork rows: 54 task-specific forks and 162 hint-overlap forks.
- Scored teacher contexts: no hint/student, execution observation with correct output, failing-input-only observation, shuffled execution observation, and full-reference leakage ceiling.
- Stage-1 execution-grounded OPSD gate failed:
  - Student task-specific correct-branch preference: 4.573 nats.
  - Execution observation: 4.636 nats, +0.063 over student.
  - Shuffled execution observation: 4.652 nats, +0.079 over student.
  - Full-reference leakage ceiling: 4.811 nats, +0.237 over student.
- Interpretation before final report: execution feedback with expected output is not producing a task-specific teacher signal beyond the shuffled control on this audit set; do not train this OPSD variant.
