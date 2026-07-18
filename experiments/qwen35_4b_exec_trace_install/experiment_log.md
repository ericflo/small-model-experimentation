# Qwen35 4B Exec-Trace Install Experiment Log

## 2026-07-17 — model-free construction frozen

Lifecycle 32, the FIRST curriculum bet of the cognitive-core coding program.
Mission: install real coding capability into base Qwen/Qwen3.5-4B via a
designed, contamination-free curriculum, proven by transfer — after the
menagerie proxy was shown NOT to transfer (McNemar p=1.00 vs base).

The bet: install an accurate "mental interpreter" via EXECUTION TRACING (train
`code -> execution-verified running-state trace`), self-generated and
execution-verified (no larger teacher), disjoint from HumanEval/MBPP.

Built and verified (no GPU, no commit):

- `scripts/gen_exec_trace_curriculum.py` (seed 90210) — 400 random terminating
  Python programs + traces, TRIPLE-verified (primary interpreter + real-CPython
  `sys.settrace` re-execution + safety/termination caps). Corpus sha
  `7c5b77ea87438f4fb46b1d6d1b468edb275feee4e915161c9180c109d410e32e`; tiers
  short 80 / medium 160 / long 160; steps 5-47, mean ≈ 23; 400 unique programs.
  A `--mix-retention R` switch (default OFF) is prepared as the forgetting
  guard.
- `scripts/contamination.py` + `data/contamination/banned_function_names.json`
  (668 benchmark function names) — 0 whole-word hits; 0 distinctive shared
  7-grams with benchmark solution code.
- Vendored `scripts/train_think.py` (sha e0eca2a2…) and `scripts/merge_adapter.py`
  (sha cb9af8b4…), byte-identical to the chain trainer/merger.
- `scripts/train_trial.py` — fail-closed base authentication (in-cell
  provenance copy + tree manifest + full weights hash); recipe r32/a64, 1 epoch,
  lr 1e-5, seed 90211 (50 optimizer steps).
- `scripts/measure_transfer.py` — invokes the shared fitness harness for both
  arms x both datasets; frozen INSTALLED_CODING / RETENTION_FAIL / NULL
  consequence.
- `scripts/run.py` — checkpointed `--smoke | --stage train | --stage merge |
  --stage measure`; each GPU stage gated behind a staged adversarial review.
- 51 unit tests green (2 present-only HF-cache aids skip without the cache);
  `run.py --smoke` green; boundary drills refuse.

Grep-fresh note: construction seed 90210 is fully fresh repo-wide; training
seed 90211's only other occurrence is an unrelated menagerie confidence-gym
sampling-seed default (`qwen35_4b_gauntlet_frontier/scripts/gym_confidence.py`),
a different seed context — no training-seed collision in this chain.

GPU stages (train/merge/measure) are pending their staged reviews.

## 2026-07-18 — Transfer measurement complete: NULL (bet #1 closed)

- exec_trace vs base: HumanEval 0.7622->0.7683 (+1 problem), MBPP
  0.5650->0.5500 (-3), agentic 8/35->8/35 (5v5 discordant). Retention
  held (no catastrophic forgetting - the distinct-instruction +
  moderate-dose design worked). No real coding improvement on any
  signal. The frozen fast-rule technically fired INSTALLED_CODING on the
  +1 HumanEval, but that is noise (near ceiling) and the agentic real
  target is exactly flat -> honest verdict NULL.
- LAW: install!=convert extends to coding. Installing a passive
  cognitive primitive (execution-tracing / state-tracking) reshuffles
  WHICH coding tasks are solved, not HOW MANY - the same signature the
  menagerie composites showed. Two independent curriculum families now
  reshuffle-without-raising, suggesting narrow static-SFT skill-install
  has a structural limit for coding capability.
- NEXT (bet #2): target the AGENTIC LOOP directly (plan-act-verify-
  repair / persistence + self-correction), the observed failure mode
  (model does one pass and stops), rather than a passive component; and
  tighten the transfer rule to a meaningful-delta / significance bar.
