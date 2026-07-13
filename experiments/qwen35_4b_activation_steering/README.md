# Qwen3.5-4B Activation Steering: is the latent first-op causally usable?

**Status:** finished

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: C19 found the first op is linearly *decodable*. Is it causally *usable* — can we steer it
  out of the residual stream (training-free) and make the model use it?
- Prior anchors: C19 (latent representation), C17 (selection is free), C18 (banking installs capability).

## Question

Add the decoded "correct first-op" direction back to the residual stream during generation (ActAdd). Does the
model then name / use it?

## Hypothesis

Pre-registered (`reports/prereg.md`): steer_true ≥ baseline + 0.10 (usable); steer_wrong < baseline and
steer_random ≈ baseline (specific); depth-1 sanity ≥ +0.15; identification lift (the prize).

## Setup

- Model: Qwen3.5-4B (only permitted model). Inference only + a forward hook.
- Directions: mean-difference `d_c = mean(acts[first_op==c, L]) − mean(acts[all, L])` from C19's cached
  activations, at the C19 probe-best layer (depth-1 L15, depth-2 L22).
- Steering: forward hook on `model.model.layers[L−1]` adds `coef · d_c` to the residual at all positions.
- Tasks: fresh verified `list`, depth 2 (primary) + depth 1 (sanity), n=150, disjoint from C19.
- Readout: forced-answer first-op naming (fast, baseline parse ≈ 1.0). Conditions
  {baseline, steer_true, steer_wrong, steer_random} × coef {2,4,6,8,12,20}. Secondary: identification pass@1.

## Run

Smoke: `python scripts/steer.py --n 30 --coefs 0 6 12 --depths 1`
Full: `python scripts/steer.py --n 150 --coefs 0 2 4 6 8 12 20 --depths 2 1 && python scripts/analyze.py`

## Results

**INERT — decodability ≠ steerability.** Depth 1 (probe 0.99): steer_true never beats baseline (max +0.03),
degrades at high coef. Depth 2: faint whiff (+0.05, within noise of random, below the +0.10 bar). Null at
earlier layers (8, 12) and on identification (0.03→0.03). All pre-registered predictions refuted. See
`reports/report.md`, `analysis/steering.png`, `runs/steer_results.json`.

## Interpretation

The latent signal C19 found is readable but not writable into behavior via standard ActAdd. Strengthens the
throughline: test-time interventions (selection C17, steering C20) don't move the wall; only weight edits
(banking C18) and tools (C12) do. Limit: a clean negative for mean-difference steering; patching / optimized
vectors untested.

## Knowledgebase Update

- Program evidence updated: `research_programs/structured_execution_and_compilers/evidence.md` (C20)
- Claim ledger updated: C20 added

## Artifacts

- `scripts/steer.py` (ActAdd hook + readouts), `scripts/analyze.py`
- `runs/steer_results.json` (naming sweep, ident arm, earlier-layer supplementary), `analysis/steering.png`
- Directions built from C19's `scratchpad/probe_artifacts/acts.npy` (external) + the probe experiment's `labels.json`
