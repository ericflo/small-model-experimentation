# Pre-registration: does banking shift the PROPOSAL distribution? (concentration vs expansion)

Logged 2026-07-03, before any data. C17 proved the generation wall is COVERAGE, not selection: sample+filter
recovers 2–5× over single-shot but that *is* sample-more; you cannot beat sample-more by better selection.
The only lever left is shifting the PROPOSAL distribution. C11/C12 showed banking the model's own verified
solutions raises pass@k. The novel, unresolved question this tests: does banking **CONCENTRATE** existing
coverage into the greedy sample (curve shifts left, same asymptote) or **EXPAND** the coverage ceiling
(propose programs it never sampled — actually crossing the wall)?

## Setup

Substrate `list` (canonical). Fresh verified-depth, collapse-rejected tasks, disjoint TRAIN / EVAL splits.

- **Harvest (TRAIN):** depths [1,2,3], tasks [20,35,35]=90. Sample K=40 identification completions
  (think-mode, budget 512, no op-menu — the canonical prompt). Keep execution-verified (hidden-correct)
  programs; cap 12 per task (balance so depth-1 doesn't dominate). Build `{prompt, code}` SFT pairs = the
  model's OWN verified code (no teacher).
- **Bank:** QLoRA-SFT (r32/alpha64, bnb 4-bit, 3 epochs) single-shot prompt→code, thinking off (C11/C12
  protocol).
- **Eval (HELD-OUT):** depths [1,2,3,4], 20/depth (disjoint from TRAIN). ONE consistent no-think harness for
  BOTH base and banked: greedy@1 (deployable single-shot) + coverage@k (k≤16, no-think sampling). Grade each
  vs 8 hidden examples.

## Predictions (locked)

- **P1 (banking lifts single-shot):** banked greedy@1 > base greedy@1 at depths 1–3 (trained), Δ ≥ +0.15 at
  depth 2 (the largest headroom: base first@1 ~0.10, sample+filter ~0.30).
- **P2 (beats sample-more at deployment):** banked greedy@1 (1 sample) ≥ base coverage@16 (16 samples) at
  depth 2 — a single banked sample matches or beats sampling 16× from the base.
- **P3 (concentration vs expansion — the crux):** banking CONCENTRATES more than it EXPANDS. Operationalized:
  Δ(greedy@1) > Δ(coverage@16) at depths 1–3 (single-shot moves more than the ceiling). STRONG-EXPANSION
  alternative (would be the bigger result): banked coverage@16 > base coverage@16 by ≥ +0.10 at depth 2 or 3
  — the ceiling itself rises (new programs proposed).
- **P4 (generalization vs memorization):** the depth-2 lift holds on HELD-OUT eval tasks (never trained),
  not just an artifact — i.e. P1 is measured only on held-out tasks. And extension to UNTRAINED depth 4:
  prediction — banking does NOT lift depth-4 (base coverage ~0; nothing to concentrate; expansion fails
  where the model never proposed), consistent with C17's coverage wall and C14 format-locality.

## Decision mapping

- **CONCENTRATION** (P3 main): banking makes sample-more's shallow-depth gains deployable at k=1 (a real
  deployment win — beats sample-more on compute) but does NOT create new capability; the coverage ceiling is
  unmoved, so "sample more on the banked model" gains little further. This is the likely, honest outcome and
  still valuable (single-shot ≈ old sample+filter).
- **EXPANSION** (P3 strong): banked coverage ceiling rises → banking proposes compositions the base never
  sampled → genuinely crosses the wall. The stronger, more surprising result; would reframe banking as
  capability-creation, not just concentration.
- **MEMORIZATION** (P4 fails): lift appears on train but not held-out ⇒ banking memorizes, consistent with
  C14 format-locality; report as a scoping of C11/C12.

## Controls / honesty

- TRAIN and EVAL task sets are disjoint (held-out generalization is the only thing P1/P2 are scored on).
- Base and banked evaluated in the IDENTICAL no-think harness (isolates the proposal-shift from any
  thinking-mode difference); base no-think numbers are the fair baseline (not C17's think numbers).
- Diversity check: unique-program count per task, base vs banked (guard against C11-style collapse worry).
- Depth-4 is an untrained control for expansion vs memorization.
- No teacher, no external capability: training targets are the fixed 4B's OWN execution-verified outputs.
