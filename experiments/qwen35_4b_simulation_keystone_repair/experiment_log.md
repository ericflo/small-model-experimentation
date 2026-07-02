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

## Phase 1-2 results (VERDICT: keystone REFUTED -- separable/format-local branch)

Simulator REPAIRED: SIM 0.92/0.82/0.80/0.84/0.76 by depth (base 0.96/0.88/0.58/0.30/0.36); +54pp at
untrained d4 (P-K2 ok); held-out prims 0.42->0.85 (P-K6 ok); P-K1's d3 letter missed by 8pp (+22 vs +30)
but d4/d5 vastly exceed. LADDER FLAT for SIM: bare 0.08->0.09, segmented 0.14->0.17, afc_nothink
0.75->0.78 (P-K3 fail). P-K4 INVERTED: PROD tripled segmented (0.14->0.41; d3k0 0.20->0.65) -- format-
adjacent transfer -- and degraded plan-given 0.93->0.72. Both adapters crashed afc_think to 0.10-0.15:
verified FORMAT CAPTURE on raw generations (SIM answers the A/B question with ```python blocks).
Locked-rule verdict: REFUTED-separable. Insight: capability is FORMAT-LOCAL in the fixed 4B (claim C14).
Chain runtime ~5.6h total (SIM train 2406s loss 0.021; PROD 898s loss 0.116; 3 ladders ~1-1.5h each).
