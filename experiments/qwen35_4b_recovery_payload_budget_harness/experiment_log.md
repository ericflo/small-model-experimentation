# Experiment Log

## 2026-07-12 — intake and preregistration

- Parent λ=.18 result: 58/60 recovery, 0.104 drift, but zero eligible candidates.
- Post-stop evidence fixed the intervention: 24/24 invalids hit the 256-token
  answer cap with closed thinking; 30/30 rejected cases changed within two turns
  and solved, with 20 INSPECT→PATCH paths.
- Froze candidate weights and a 512-thinking/512-answer interface for every arm.
- Froze valid rejected recovery as PATCH or INSPECT→PATCH within two turns;
  invalid-first paths do not qualify.
- Added a third, wholly disjoint 48-context locality block.
- No 512-answer behavior, transfer task, or Menagerie seed has been exposed.
