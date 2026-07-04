# Pre-registration: is the latent first-op direction causally usable?

Logged 2026-07-03, before any steering data. C19 found the composition's first operation is linearly present
in the residual stream far above the model's behavioral access (depth-2 probe 0.42 vs naming 0.13), but
flagged that "linearly decodable ≠ usable for generation." This is the decisive follow-up: if we ADD the
"correct first-op" direction to the residual stream at generation time, does the model actually USE it — a
**training-free** elicitation of the latent signal (the mission's dream: cleverer access, not compute)?

## Method

Reuse C19's cached activations to build steering directions; test on FRESH held-out tasks with a forward hook.

- **Directions (ActAdd / mean-difference):** for each first-op class c, `d_c = mean(acts[first_op==c, L]) −
  mean(acts[all, L])` at probe layer L, computed from C19's cached activations (depth-matched). Steering adds
  `coef · d_c` to the residual stream at the output of decoder layer L−1 (probe index L), at all positions,
  every generation step, via a forward hook on `model.model.layers[L−1]`.
- **Tasks:** fresh verified-depth `list` tasks, depth 2 (primary; probe signal 0.42, behavior ~0.13 — the
  sweet spot) and depth 1 (sanity; probe 0.99), disjoint from C19's set (new seed). Steer at L = the depth's
  C19 best layer (depth-2 L22 → decoder 21; depth-1 L15 → decoder 14).
- **Conditions per task:** `baseline` (no steer), `steer_true` (toward the task's TRUE first-op), `steer_wrong`
  (toward a random WRONG first-op), `steer_random` (a random Gaussian vector scaled to ‖d_true‖).
- **Primary readout — first-op naming** (cheap, targeted): "which op is applied first?" → accuracy per
  condition, swept over coef ∈ {0, 2, 4, 8, 16} (naming is cheap enough to sweep widely).
- **Secondary — identification pass@1** (think, greedy): at the best coef, baseline vs steer_true — does
  steering toward the first op help the model actually SOLVE (the real prize; harder, needs all ops right).

## Predictions (locked)

- **P1 (causal usability):** at the best coef, depth-2 `steer_true` naming ≥ baseline + 0.10. Refuted if
  `steer_true` ≤ baseline + 0.03 at every coef (readable-but-inert).
- **P2 (specificity, not just any perturbation):** `steer_wrong` naming < baseline (steering toward a wrong op
  degrades), and `steer_random` ≈ baseline (within ±0.05). If a random vector helps as much as `steer_true`,
  the effect is nonspecific and P1 is void.
- **P3 (depth-1 sanity):** `steer_true` moves depth-1 naming by ≥ +0.15 at some coef (the direction is clean
  where the probe is 0.99).
- **P4 (the prize — usable for generation):** depth-2 `steer_true` identification pass@1 > baseline (any
  positive lift = the latent signal is usable for GENERATION, not just naming). Prediction: a small positive
  lift, capped because steering fixes only the first of two ops.

## Decision mapping

- **USABLE** (P1 ∧ P2): the latent first-op direction is causally wired to output — training-free activation
  steering elicits it. A genuine "cleverer access" win; opens deployable self-steering (steer toward the
  probe's own prediction) as the next step. If P4 also holds, it is usable for generation, not just naming.
- **INERT** (P1 refuted): the first op is *readable* (C19) but *not causally usable* — a linear decoder sees
  it but the model cannot act on it. Refines C19 (latent-but-inert), and reinforces that only
  proposal-installation (banking/tools, C18/C12), not readout OR steering, crosses the wall. A clean, decisive
  negative.

## Honesty notes

- `steer_true` is an ORACLE (it uses the known answer) — an UPPER BOUND on steering usefulness, a mechanistic
  usability test, not a deployable method. Deployable self-steering (toward the probe's prediction, ~0.42
  accurate at depth 2) is only worth building if the oracle works.
- coef too small → no effect; too large → breaks fluency (naming parse-rate drops). Report parse rate; sweep
  guards against reading a null at the wrong scale.
- The random/wrong controls are essential: a positive `steer_true` result only counts if it is specific.
