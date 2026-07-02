# Qwen3.5-4B Simulation Keystone Repair Experiment Log

## Design

Intervention test of C13 (broken multi-step mental simulation as the keystone under the inverse-capability
ladder). Pre-registered (reports/prereg.md). Phase 0 = frozen simulator microbenchmark (falsification
gate). Phase 1 = matched-token QLoRA arms: SIM (state chains) vs PROD (reference code; direct end-task
control). Phase 2 = simulation retest (in-dist / length-gen / held-out primitives) + full C13 ladder
(bare / plan-given / segmented / 2AFC no-think / 2AFC thinking) on fresh verified tasks, all three models.

## Phase 0 (gate PASSED; P-K0b refuted)

Frozen, output exact-match by depth (think arm): 0.96 / 0.88 / 0.58 / 0.30 / 0.36; no-think: 0.84 / 0.52 /
0.46 / 0.30 / 0.16. P-K0a CONFIRMED (broken in isolation, d4 ~0.3 << 0.8 kill threshold) -> proceed.
P-K0b REFUTED: thinking HELPS single-pipeline simulation at every depth -- refines C13/P12: deliberate
simulation is length-fragile, not globally wrong; P12's chance-level 2AFC = double-simulation + comparison
load. No clean single-r geometric decay (retention 0.92 -> 0.66 -> 0.52 by step) -- decay accelerates.

## Phase 1 data

SIM 1500 records / 227,938 tokens vs PROD 554 records / 232,447 tokens (matched within 2%). Held-out
primitives excluded from both. PROD sharpened to reference-code supervision (prereg addendum, logged
before Phase 1 ran).

## Phases 1-2

Running (scripts/phase12_chain.sh): train SIM + PROD -> simbench x2 -> ladder x3.
