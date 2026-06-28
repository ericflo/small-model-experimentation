# Experiment Log

## 2026-06-27

- Created standalone experiment package.
- Defined the frozen ABI as a general-purpose Python/stdlib-style primitive inventory before running coverage.
- Primary gate: held-out oracle coverage on unseen MBPP test records using only this frozen ABI.
- No training is planned unless held-out coverage is meaningfully high.
- Implemented `scripts/run_independent_abi_gate.py` with the frozen ABI and evaluator.
- Ran calibration, held-out, train, and three-seed held-out sweep.
- Main results:
  - calibration coverage: 60/160 (37.5%)
  - held-out coverage: 23/160 (14.4%)
  - train coverage: 73/374 (19.5%)
  - held-out sweep: 30/160, 29/160, 29/160; mean 18.3%
  - held-out covered depth counts: 20 depth-1, 3 depth-2, 0 depth-3
  - held-out task-level visible-pass/no-full-winner rate: 59.6%
  - held-out candidate-level hidden-wrong rate among visible-consistent candidates: 91.9%
- Decision: no compiler training. The frozen independent ABI does not cover held-out MBPP tasks at a useful rate.
- Generated report and charts at `reports/final_report.md` and `reports/figures/`.
