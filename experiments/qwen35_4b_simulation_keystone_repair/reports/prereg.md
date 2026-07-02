# Pre-registration: Simulation Keystone Repair

Logged 2026-07-02, before any data. Tests C13's causal claim by INTERVENTION: if multi-step mental
simulation is the keystone primitive under the inverse-capability ladder, then repairing it (SFT on
interpreter-generated forward-simulation traces — unlimited, teacher-free, never an inverse task) should
transfer UP the ladder to untrained capabilities. C13 next_tests #1 and #2, executed.

## Phase 0 — simulator microbenchmark (frozen model; falsification gate)

Task: given a STATED pipeline (names+params+meanings) and ONE input list, write the full state chain
(input -> state after each op -> output). No code. Graded: final-output exact match (primary) +
first-divergence step (diagnostic). Grid: d ∈ {1..5} × k ∈ {0,2}, n=25/cell, verified-depth compositions;
no-think and thinking (budget 512) arms.

- **P-K0a**: simulation decays geometrically: per-op retention ~0.85–0.90 (matching the 0.88 single-step
  recognition), so output-accuracy ≤0.6 by d3 and ≤0.4 by d5 in the better arm.
- **P-K0b**: thinking arm ≤ no-think arm at d≥3 (P12: deliberate simulation is systematically wrong).
- **GATE**: if d4 output-accuracy ≥0.8 (simulation NOT broken in isolation), the keystone framing dies —
  stop, report the (interesting) contradiction with C13's inference.

## Phase 1 — training arms (matched training tokens, QLoRA from base, identical hyperparams)

- **SIM**: SFT on ~1.5k interpreter-generated state-chain traces (given pipeline+input -> emit full chain),
  depths 1–3 ONLY, with 3 primitives fully HELD OUT (rotate_k, dedup_adjacent, running_max).
- **PROD** (control): SFT on prompt->code production pairs (C11/C12 recipe) matched in token count —
  controls for "any substrate SFT helps" (vocabulary/format exposure).
- **BASE**: frozen.

## Phase 2 — retest (all arms, fresh verified tasks, same protocols as C13)

(a) simulation: in-distribution d1–3, LENGTH-GENERALIZATION d4–5, held-out-primitive chains.
(b) the C13 ladder: thinking-2AFC, no-think 2AFC (logit), segmented identification, bare identification
    (pass@4), plan-given transcription (interference check), monolithic greedy@1.

- **P-K1 (manipulation check)**: SIM lifts d3 simulation ≥ +30pp over BASE. If not → branch C
  (simulation unlearnable at QLoRA scale — architectural serial-compute limit), stop.
- **P-K2 (length generalization)**: SIM lifts d4 simulation ≥ +15pp despite never training beyond d3.
  If gains stop at trained depths, "unlimited data" collapses to depth-bounded SFT.
- **P-K3 (keystone transfer)**: the ladder moves in mechanism order — thinking-2AFC 0.50 → ≥0.70
  (simulate-and-compare now works), segmented d2 ≥0.75 (from 0.50), bare identification smallest gain.
- **P-K4 (dissociation — load-bearing)**: PROD moves monolithic solve but NOT the inverse ladder;
  SIM moves the ladder but NOT (necessarily) monolithic. If PROD moves the ladder equally, "keystone"
  is just "substrate SFT" and the causal story fails.
- **P-K5 (no interference)**: transcription stays ≥0.95 in all arms.
- **P-K6 (held-out primitives)**: SIM's simulation gains transfer to held-out primitives attenuated but
  present (composition skill, not transition memorization).

## Decision rules (locked)

- **KEYSTONE CONFIRMED** = P-K1 ∧ (≥2 ladder rungs move per P-K3) ∧ P-K4 dissociation.
- **REFUTED-separable** = P-K1 holds but ladder flat → forward/inverse competences separately
  represented; SFT is task-local; mechanism diagnoses do not license training-transfer predictions.
- **REFUTED-unlearnable** = P-K1 fails → the wall is architectural, not data-coverage.
Each branch maps to a distinct claim update.

## Phase-1 refinement (logged before Phase 1 ran)

PROD control sharpened: instead of C11-style self-harvested solutions, PROD = bare-task prompt (I/O
examples) -> generator REFERENCE code. Both arms are now equally supervised with unlimited ground truth;
the ONLY difference is the supervised CONTENT (state chains vs code). This makes PROD a direct
end-task-training arm: if unlimited direct SFT cracks bare identification by itself, "train the end task"
beats keystone practically (informative); if SIM transfers to the ladder better than PROD at matched
tokens, the keystone claim stands in its strongest form.
