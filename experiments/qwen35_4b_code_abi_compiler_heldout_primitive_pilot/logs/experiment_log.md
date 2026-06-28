# Experiment Log

## 2026-06-27

- Created standalone experiment package.
- Copied the frozen ABI implementation into this package so it can be rerun independently.
- Primary gate: held-out frozen-ABI oracle coverage before any compiler training.
- Planned metrics: held-out coverage drop, depth-1 selection vs depth-2/3 composition, parse validity, execution accuracy, confusion patterns, and direct Python sampling on the same held-out records.
- Built calibration, held-out, and train target records using the frozen ABI.
- Gate 1 primary result:
  - calibration coverage: 134/160 (83.8%)
  - held-out coverage: 22/160 (13.8%)
  - coverage drop: 70.0 percentage points
  - train coverage: 83/374 (22.2%)
  - compiler train targets after validation split: 66
- Ran a three-seed held-out split sweep over the test suffix excluded from calibration:
  - seed 11: 31/160 (19.4%)
  - seed 17: 29/160 (18.1%)
  - seed 23: 28/160 (17.5%)
  - mean coverage: 18.3%
- Decision: compiler training was gated off. Frozen ABI reuse collapsed on held-out tasks, so a QLoRA compiler pilot would not measure reusable code compilation.
- Generated `reports/final_report.md` and figures under `reports/figures/`.
