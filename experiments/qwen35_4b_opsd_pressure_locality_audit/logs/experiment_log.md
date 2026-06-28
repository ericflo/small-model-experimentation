# Experiment Log

## 2026-06-26

- Created standalone no-training OPSD pressure-locality audit package.
- Copied generated retrieval-adaptation candidate pools, retrieval plan, verified library, and generic evaluator/model utilities into the package.
- Localized the experiment identity to `qwen35_4b_opsd_pressure_locality_audit`.
- Pre-registered primary gate as same-prefix counterfactual teacher preference at code forks.
- Built 14 hidden-correct versus visible-pass hidden-wrong matched pairs across tasks 35, 44, and 87.
- Extracted 50 executable code forks: 14 task-specific and 36 hint-overlap.
- Ran one-pair model-scoring smoke test, then full scoring over 512 sequences.
- Corrected the gate to include delta over the no-hint student; absolute teacher preference alone was confounded because the no-hint student already strongly preferred the correct task-specific branch.
- Final gate result: fail.
  - Weak retrieved hint task-specific absolute preference: 4.441.
  - No-hint student task-specific preference: 4.449.
  - Weak retrieved hint delta over student: -0.008.
  - Shuffled hint delta over student: -0.028.
  - Full-reference leakage ceiling delta over student: +0.213.
  - Weak retrieved hint-overlap delta over student: +1.357.
- Generated final report, machine-readable summary, and four figures under `reports/`.
