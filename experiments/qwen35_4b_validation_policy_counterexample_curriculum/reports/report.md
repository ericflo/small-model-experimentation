# Report: validation-policy counterexample curriculum

## Current status

The design is frozen pending its immutable design-commit receipt. Deterministic
CPU preflight passed; no Qwen training or evaluation generation has run.

## Preflight evidence

- Parent weight: `1cf5fb...41ba3`; C54 locality/Menagerie anchor:
  `c93316...608d5`.
- Frozen prior bank: `9c196d...9315`; prior receipt: `8c2c33...e63e`.
- Candidate/control: 48 task blocks, 336 rows, every seven-transition stratum
  represented 48 times, zero think loss.
- Candidate treatment: 24 replacements at only
  `diagnosis_to_changed_patch`; 312 prior rows unchanged.
- Weighted action mass: 38,248 per operator in both arms; maximum encoded row
  length 1,179 tokens, below the 4,096-token training window.
- Procedural content: bank 24/24 unique, calibration 24/24 unique and disjoint;
  dev 32/32 unique, confirmation 32/32 unique and disjoint.
- Every tested initial/partial workspace fails both executable suites and every
  oracle passes; the bank and its receipt are firewall-clean.

## Outcome boundary

These facts establish a valid, isolated experiment, not a model improvement.
The result section will be written from frozen locality, calibration, transfer,
retention, and conditional Menagerie receipts after the run. Claim and shared
synthesis surfaces remain unchanged before that evidence exists.
