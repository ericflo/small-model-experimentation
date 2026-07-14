# Replay-Anchored Universal Curriculum Continuation Experiment Log

## Scaffold

Created as a result-separated successor after the parent designed-only arm displaced
three benchmark families and lost 0.1385 aggregate to `blend`.

## 2026-07-13 — design freeze and smoke

- Copied the parent's 800-row truth-audited designed corpus and the C53 2,240-row broad
  replay corpus into the experiment; source SHA-256 values match their parents.
- Materialized exact nested arms: candidate = 400 designed + 1,120 shared replay;
  control = the same 1,120 replay + 400 additional replay.
- Both arms have 1,520 rows and 190 optimizer steps. Exact tokenization at max length
  4,096 produced zero skips. Candidate dose is 1,231,404 forward tokens and control is
  1,444,589.
- Frozen local seed 88,003 and aggregate-only quick@1,024 seed 78,133 before training.
- `scripts/run.py --smoke` passed, and all three dose construction tests passed.

## 2026-07-13 — parent factorial closed; candidate launch

- The parent's from-base 800-designed + 2,240-replay arm completed 3,040/3,040 rows
  with zero skips, then failed its frozen local parse (0.846 < 0.90) and cap-contact
  (4 > 2) gates at seed 88,002. Benchmark seed 78,132 remained unconsumed.
- Started the frozen `warm_union` candidate only after the parent negative was durable.
  `replay_refresh` remains gated on candidate local success to avoid unnecessary spend.

## 2026-07-13 — candidate local pass

- `warm_union` completed all 1,520 rows and 190 optimizer steps with zero skips, finite
  loss 0.7727, and adapter SHA-256 `26837fad...8f18`.
- Frozen seed 88,003 passed: accuracy 0.7308, parse 0.9615, cap contacts 1/26, route
  abstentions 0. Induction and state remained 0/2 and are preserved as residuals.
- Candidate training receipt SHA-256 is `450e367c...cbd4`; local gate receipt SHA-256 is
  `7ae1d6ae...fe11`.
- The pass authorized and launched the frozen `replay_refresh` mechanism control.

## 2026-07-13 — mechanism control and explicit merges

- `replay_refresh` completed all 1,520 rows and 190 optimizer steps with zero skips,
  finite loss 0.4365, and adapter SHA-256 `c296c774...d36a`. Its checked-in training
  receipt SHA-256 is `f2a92713...cba0`.
- Explicitly merged the candidate and control into full checkpoints. Candidate merged
  weight SHA-256 is `29baf3ad...22f6`; replay-control merged weight SHA-256 is
  `22c61ceb...bc9e`.
- External merge-receipt SHA-256 values are `35894a31...63cf` (candidate) and
  `f32d7fc6...af1c` (control). The frozen paired aggregate event is now authorized.

## 2026-07-13 — aggregate pilot; candidate negative, replay anchor advances

- Consumed the sole frozen quick@1,024 seed 78,133 event through the trusted
  aggregate-only vLLM gateway. All four arms used explicit merged checkpoints.
- Aggregate scores: base 0.1750, `blend` 0.4410, `replay_refresh` 0.4851, and
  `warm_union` 0.4238.
- `warm_union` was +0.2488 versus base but -0.0172 versus `blend` and -0.0613
  versus replay refresh. It regressed `rites` by 0.125 below base, so the frozen
  all-family, strong-control, and mechanism-control gates all failed.
- `replay_refresh` was +0.3101 versus base and +0.0441 versus `blend`, with eight
  strictly positive families and no negative family. It tied base on `rites` and
  `sirens`, so it is a stronger anchor rather than a universal-feature winner.
- Replay refresh had 17.3% more forward-token exposure despite matched optimizer
  steps. The candidate is rejected, but the whole gap cannot be assigned to content;
  a successor must match both steps and token exposure.
- Closed the experiment as a preserved negative. Any adaptive targeting of the two
  tied axes must use a new experiment, new local seed, and new benchmark seed.
