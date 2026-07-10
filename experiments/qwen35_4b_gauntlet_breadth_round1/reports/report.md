# Gauntlet round 1: breadth-first agentic expert iteration Report

## Summary

One effective round of breadth-first self-training — 925 verified,
self-generated examples across 9 invented gym families — moved the blackbox
menagerie instrument from 0.140→0.363 (+0.223, quick tier, seed 52004) and
0.152→0.446 (+0.294, seed 52005): the first install in this repository ever
measured, and replicated, on the corpus's held-out benchmark. Gym-internal
mean rose 0.184→0.701, with equally large gains on two never-trained held-out
families (+0.54/+0.61) and on a family that contributed zero training data
(+0.40). At this scope, breadth defeats the substrate-locality law
(C43/C45/C48-scoped).

A second, instrument-level finding matters beyond this experiment: vLLM 0.24
runtime LoRA silently does not apply Qwen3.5-4B PEFT adapters. Every adapter
arm evaluated through vLLM — in any experiment — measures the base model
(see Controls).

## Research Program Fit

Program `agentic_breadth_installation`: every prior locality law was derived
from single-substrate training; this experiment instantiates the untested
variable (breadth: 10 simultaneous format-diverse, verifier-gated families)
and arbitrates on the blackbox instrument, per the program charter and the
iteration-speed doctrine (fast rounds, menagerie quick every round).

## Method

- **Gym**: 12 procedurally generated families (10 trained, 2 held out)
  spanning the ten public menagerie capability axes — atoms (final-line
  `ANSWER:` scoring) and lockstep-batched multi-turn episodes (one-line
  action grammars) with machine-checkable verifiers and oracle/random/
  degenerate selftest floors (`reports/gym_design.md`; firewall statement
  therein — gym content is invented against public axis descriptions only).
- **Loop** (fast profile): ~80-min sharded harvest (K=2, L1–L2,
  family-adaptive think budgets: 4096 where the base model cannot close its
  chains, else 2048) → verified-sample SFT build → ~40-min think-channel
  QLoRA (r32/α64, C48 recipe) → menagerie quick base-vs-treatment on a fresh
  seed.
- **Round-2 recipe deltas** (the ones that produced the install):
  1. atom targets canonicalized to the model's own think chain + the terse
     `ANSWER: <value>` line (its own verified value);
  2. forced-close recovery arm — correct-after-cut samples train
     `truncated_think + </think> + ANSWER: x` with the chain as pure context
     (loss weight 0), putting the deployment-critical post-force-close state
     in-distribution;
  3. per-token loss weights (prompt 0 / think 0.2 / answer & action 1.0):
     round 1 (full-weight, naturally-closed-only) was near-self-distillation
     and installed nothing measurable;
  4. deployment as a merged composite checkpoint / HF-backend adapter (see
     Controls: vLLM runtime LoRA is a no-op).

## Results

Gym-internal (greedy@1, deployed think budget 1024, held-out generation
seeds; base vs round-2):

| family | base | round 2 | delta | parse-fail /100 |
|---|---:|---:|---:|---|
| caravan | 0.020 | 0.810 | +0.790 | 98 → 8 |
| kilnrite | 0.231 | 0.964 | +0.732 | 93 → 0 |
| runeward | 0.030 | 0.960 | +0.930 | 97 → 0 |
| loomfix | 0.073 | 0.665 | +0.593 | 99 → 2 |
| burrowmaze | 0.226 | 0.766 | +0.540 | 73 → 5 |
| ferrier | 0.484 | 0.960 | +0.476 | 61 → 0 |
| foundry_ledger | 0.140 | 0.460 | +0.320 | 86 → 13 |
| gatepost | 0.581 | 0.766 | +0.185 | 38 → 5 |
| glyphgate | 0.161 | 0.266 | +0.105 | 100 → 24 |
| stallwright (0 train examples) | 0.000 | 0.395 | +0.395 | 100 → 13 |
| **brinework (HELD OUT)** | 0.070 | 0.610 | **+0.540** | 93 → 19 |
| **spindle (HELD OUT)** | 0.187 | 0.795 | **+0.608** | 100 → 3 |
| mean | 0.184 | 0.701 | +0.518 | |

Menagerie (paired base-vs-adapter, HF parity backend both arms — deterministic,
no decode noise; fresh seed per event, aggregate scores only):

| event | tier | base | adapter | delta |
|---|---|---:|---:|---:|
| seed 52004 | quick | 0.1396 | 0.3625 | **+0.2229** |
| seed 52005 | quick | 0.1521 | 0.4458 | **+0.2938** |
| seed 52006 | medium | 0.1217 | 0.4453 | **+0.3237** |

Medium (multi-turn episodes + L3/L4 atoms) confirms with EVERY family
positive, including axes flat on quick: mirage +0.600, siftstack +0.700,
chronicle +0.700, toolsmith +0.383, lockpick +0.200, sirens +0.200,
warren +0.200, menders +0.100, rites +0.100, stockade +0.053. The
pre-registered three-leg decision rule (two positive quick seeds with mean
≥ +0.03; medium ≥ +0.02) is met at 8–16× its bars: verdict POSITIVE.

Per-family quick deltas (seed 52004): chronicle +0.750, siftstack +0.625,
toolsmith +0.354 (→1.000), mirage +0.250, lockpick/rites/warren +0.125,
menders 0.000, sirens 0.000, stockade −0.125.

## Controls

- **Same-seed pairing + deterministic backend**: all valid menagerie deltas
  are within the HF parity backend on one seed; vLLM numbers are never mixed
  in. A vLLM base-base-adapter event (seed 52001) measured the same-seed
  decode-nondeterminism spread at ~±0.03 aggregate — and, retrospectively,
  its "adapter" arm was the base model (below).
- **The vLLM LoRA no-op**: two different trained adapters produced 1200/1200
  byte-identical gym-eval generations; an in-process on-vs-off probe (same
  engine, same prompts, greedy) gave token-identical outputs. Mechanism:
  adapter tensors are `base_model.model.model.layers.*`, the served composite
  keeps text layers under `model.language_model.layers.*`; vLLM matches
  nothing silently. The repo's earlier zero-adapter "plumbing test" cannot
  catch this by construction. Codified with the required on-vs-off gate in
  `docs/vllm_inference.md`; merged-composite deployment implemented in
  `scripts/merge_adapter.py`. Menagerie's `--adapter` (vLLM path) is presumed
  equally affected; its `--model-id` guard also rejects local checkpoint
  paths (README documents checkpoint runs) — both flagged for maintainer
  action.
- **Transfer-vs-leakage signature**: menagerie gains concentrate on trained
  axes; axes without training signal stay flat (menders, sirens) or dip
  (stockade — whose gym analogue was harvest-starved). Held-out gym families
  move as much as trained ones. Gym content was authored under the benchmark
  firewall (no menagerie contents ever read).
- **Lucky-guess gate (C28)**: small-answer-domain items require ≥3/K correct
  samples before entering SFT; episode turns require verifier-accepted
  actions (`action_ok`); at most 2 samples/item and 2 rollouts/instance.

## Oracle Versus Deployable Evidence

All headline numbers are deployable: greedy, think-mode, deployed budgets, no
scaffolding, bare model (adapter or merged). Oracle policies validated the
gym instruments only; no oracle output, no benchmark content, and no external
model enters training — provenance is exclusively the model's own
execution-verified samples.

## Interpretation

- The binding deployment constraint at these difficulty levels was the
  truncation cascade at the answer-emission seam: the model consumes any
  budget, is force-closed, then restarts a verbose explanation and never
  emits a parseable answer. Training the model — on its own verified
  outputs — to conclude and to commit from a truncated chain removes that
  constraint generally: across formats, across gym families it never saw,
  and across the blackbox benchmark.
- Breadth + emission-seam supervision installs something substrate-general;
  the C43/C45/C48 locality laws do not extend to this regime. Whether the
  remaining gap (menagerie ceiling; glyphgate/loomfix/stallwright frontiers)
  yields to iterated rounds is the program's next question.
- Methodologically: gradient signal placement beat dose. 925 examples with
  weighted loss moved the blackbox instrument where 849 at full weight moved
  nothing.

## Round 3 addendum — iteration re-saturates

Re-harvesting with the round-2 model opened the frontier at the data level
(stallwright 0/160 → 48/60 correct at L1; glyphgate L1 2/80 → 60/60; 2,276
new examples, all families represented) — but blackbox gains did NOT
compound: quick +0.285/+0.227 (seeds 52007/52008) vs round-2's
+0.223/+0.294; medium +0.264 (seed 52009) vs +0.324; gym-internal +0.019.
The install is a one-time step change from the recipe, stable at
+0.22..+0.32 across five paired events; same-recipe expert iteration
re-saturates (C11's coverage-boundedness, one level up).

## Next Experiments

- Difficulty escalation as the frontier lever (L3–L4 mass, harder gym
  generators, longer horizons) — same-recipe rounds are exhausted.
- Recovery-arm-only ablation to split emission-seam repair from axis
  competence; breadth-vs-matched-dose single-family ablation (is breadth
  causal for the held-out-family transfer?).
- slow/deep-tier confirmation events; vLLM-path restoration (merged
  checkpoints through menagerie once the --model-id guard is resolved).

## Artifact Manifest

`reports/artifact_manifest.yaml` lists the adapter + merged checkpoint under
`large_artifacts/` (regenerable end-to-end from seeds in configs) and the
gzipped harvest/eval row files under `runs/`.
