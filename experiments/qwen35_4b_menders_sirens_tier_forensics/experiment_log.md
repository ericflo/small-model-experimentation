# Menders/Sirens + Tier Forensics Experiment Log

## 2026-07-15 — Sweep, analysis, and closure (no model event)

- Swept 2,278 committed JSON files under `experiments/*/runs/`; 380 raw
  family-score rows; 356 after the frozen cleaning (one post-first-look
  refinement — summary-file exclusion — recorded honestly in the
  preregistration; headline readings unchanged by it).
- Constant check: the goal-gap pilot's "menders = 0 and sirens = 0.500 for
  every arm at every seed at quick/tb1024" has three genuine committed
  counterexamples (base sirens 0.375 at seed 78,131; candidate menders
  0.021 at 78,131; replay_refresh menders 0.125 at 78,133) — the constants
  are item-draw artifacts of the quick instrument, not structural walls.
- Tier adjudication: paired within-event strict-win analysis gives the
  goal gate (10/10 families strictly above base) 9 passes in 94 medium
  arm-events versus 1 in 84 quick arm-events; at medium the base never
  sits at a family ceiling (0/95 events) and sirens leaves its 0.500
  sticking point (exactly-0.5 in 14/95 base events vs 49/82 at quick).
- Blocking families in 9/10 near-misses: menders/sirens/warren at quick;
  menders/rites/warren at medium.
- Caveat preserved: all nine medium passers are gym-trained arms from the
  old line; the contamination-free universal arms have never been measured
  at medium — that measurement is the funded successor.
