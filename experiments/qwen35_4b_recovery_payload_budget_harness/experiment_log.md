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

## 2026-07-12 — full run and registered stop

- Fresh locality passed: drift 0.114, entropy Δ −0.0059, varentropy Δ −0.0105.
- Calibration passed at 60/60 candidate versus 58/60 action-only; all candidate
  conditional transitions passed and payload truncation fell to 1/219 turns.
- Transfer dev passed every gate: candidate 57/80, base 47/80, action 53/80,
  sample-more 40/80, scaffold 53/80; ordinary tasks were exactly retained.
- Transfer confirm candidate scored 55/80, tying action-only instead of clearing
  the frozen +3pp contrast. All other checks passed. Recorded
  `TRANSFER_CONFIRM_FAIL` and stopped before Menagerie.
- Post-stop paired audit: candidate/action union was 63/80 on both dev and
  confirm. Confirm had eight exclusive wins per policy, concentrated in
  different family/scenario cells. Queued public-verifier branch selection,
  not another global interpolation dose.
