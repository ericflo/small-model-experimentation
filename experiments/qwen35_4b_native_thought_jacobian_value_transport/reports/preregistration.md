# Preregistration: Native-Thought Jacobian Value Transport

Frozen before any result-bearing model call. CPU generation, exact enumeration,
unit tests, and repository checks may precede the immutable design commit. A
two-item model plumbing smoke may follow it but cannot tune any rule below.

## 1. Scientific ladder

This experiment separates three claims:

1. **Value-readable:** continuation success is held-out-by-task decodable from a
   natural thought prefix in the replicated J coordinates.
2. **Value-causal:** a scalar high-value coordinate changes fresh continuation
   success, beyond answer-identity and geometry controls.
3. **Deployable capability:** a non-oracle controller improves held-out accuracy
   beyond frozen and matched sampling.

Only claims 1 and 2 are tested here. Claim 3 requires a separate experiment.

## 2. Fixed model, lens, and backend

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Exact replicated context lens SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Frozen layers 4, 5, 6, 7, 8; no layer or coefficient selection.
- Transformers bf16 SDPA, unpadded batch one, full-prefix recomputation,
  `use_cache=False` for generation, labels, and every intervention arm. vLLM is
  intentionally excluded because activation patching requires internals and
  backend mixing would invalidate matched continuations.
- Natural thinking only. Never inject `</think>` or force an answer seam.

## 3. Fresh identifiable task

All 80 tasks are generated locally and fingerprint-disjoint from the direct
Jacobian parent: 16 seam calibration, 32 value fit, 32 causal confirmation.
Each has two list operations, eight visible and eight hidden I/O examples.
Exhaustive enumeration over the concrete two-operation DSL requires all pipelines
matching the visible examples to share one first-operation type.

The final answer is a one-token alias from the frozen lens (`cat=reverse`, etc.).
The prompt exposes the complete one-to-one alias mapping. `negate` remains a
second-operation/distractor but is excluded as a target because CPU review found
its two-step first position algebraically reorderable and therefore not
behaviorally identifiable. The primary exact outcome is correct alias, a narrow
first-operation identification capability—not full pipeline execution.

No file under `benchmarks/` is read or imported.

## 4. Natural prefixes and continuation value

For each task, sample three native-thinking traces at temperature 0.6, top-p
0.95, top-k 20, maximum 160 thought tokens and eight answer tokens. A trace must
naturally emit the think-close token and a parseable answer. At frozen fractions
0.33 and 0.67 of a think span of at least 16 tokens, retain the exact token prefix.

From each prefix, sample three disjoint-seed natural continuations. Define

`V(prefix) = correct continuations / 3`.

Whole-trace outcomes are never copied onto every token. Initial trace seeds,
value continuations, causal donor-selection continuations, and causal evaluation
continuations are disjoint.

## 5. Seam gate

The 16 seam tasks establish that the natural interface has headroom before value
fitting. Across three traces per task, parse rate must be >=0.90, natural close
rate >=0.80, success in [0.05, 0.95], and at least six tasks must contain both a
correct and incorrect trace. Failure freezes `NO_NATURAL_SEAM`.

## 6. J-space value model

At the last token of each natural prefix, read all 24 frozen J coordinates at
layers 4--8 and concatenate them. Standardize using training folds only. Fit an
L2 linear logistic value readout with four GroupKFold splits by task. Task and
checkpoint-fraction means are removed using training-fold statistics.

Primary score is task-macro pairwise AUC: within each mixed-value task, how often
does the out-of-fold scalar rank a higher-V prefix over a lower-V prefix?
Comparators are correct-alias coordinate activity, answer entropy, prefix length,
a full-residual probe, and labels shuffled within task/fraction.

G1 requires at least 16 mixed tasks and 128 scored prefixes, pairwise AUC >=0.65,
an advantage >=0.03 over correct-alias activity, and shuffled AUC within 0.05 of
0.50. The full-residual probe is diagnostic and cannot rescue J failure. Freeze
the all-task value weights, normalization, training high-value 0.90 quantile,
and shuffled-axis weights before causal confirmation.

## 7. Scalar certainty clamp

Let `z_l` be the 24 J coordinates at layer l and `beta_l` the corresponding
slice of the frozen value readout. For recipient score `s` and donor score `s*`,
set fixed clean desired coordinates

`z'_l = z_l + (s* - s) beta_l / sum_j ||beta_j||^2`.

At the recipient's last natural thought-prefix token, clamp these desired
coordinates across layers 4--8 on every cache-free full-prefix recomputation.
This transfers one scalar value coordinate; it does not copy the donor's full J
state or raw activation.

## 8. Control calibration

Before causal outcomes, apply the frozen scalar rule to at least 16 seam prefixes
using the training high-value quantile. Construct `random_a` and `random_b` with
32 fixed continuous starts and the replicated exact bf16 lattice solver. Every
realized layer delta must have relative norm error <=1e-5 and J-span projection
fraction <=0.01. Candidate selection sees geometry only. Any failure freezes
`CONTROL_UNREACHABLE`; thresholds cannot be relaxed.

## 9. Causal donor selection and arms

On 32 fresh tasks, generate prefixes and estimate V using three donor-selection
continuations. Within the same task and checkpoint fraction, select the largest
deterministic high/low gap with low <=1/3 and high >=2/3. Ties use stable IDs.
Require at least 16 eligible task pairs. Evaluation uses two new continuation
seeds per pair, shared across arms.

Arms:

1. unpatched low prefix;
2. scalar certainty J clamp (primary);
3. full-24-J high donor;
4. correct-alias-versus-best-competitor J clamp;
5. within-task-label-shuffled value axis;
6. exact random_a;
7. exact random_b;
8. separately fitted logit-lens value axis;
9. full raw high-donor activation;
10. sparse full-J component of the donor delta;
11. norm-matched non-J donor remainder;
12. value-fit high-minus-low ActAdd;
13. full-J high donor from a different task with a different correct alias.

All non-donor magnitude controls match the realized primary scalar delta per
layer. Every confirmation random row must again pass both numeric thresholds.
Direct alias-coordinate movement, generated tokens, parse, and termination are
diagnostics; exact alias success is primary. Matched additional baseline samples
and total forward/generated tokens are reported but cannot turn this oracle stage
into a capability claim.

## 10. Frozen causal decision

Require at least 16 pairs. `ORACLE_VALUE_CAUSAL` requires:

- scalar certainty success uplift >=0.10 over paired baseline;
- paired 10,000-resample 95% lower bound >0;
- advantage >=0.08 over the worse exact random arm;
- advantage >=0.08 over the shuffled value axis;
- advantage >=0.05 over the matched non-J remainder;
- parse-rate drop <=0.05; and
- all design, seam, G1, numeric, token, position, and causal invariance contracts.

If full-J/correct-alias identity controls work but scalar certainty fails, use
`IDENTITY_NOT_CERTAINTY`. Other labels are `NO_NATURAL_SEAM`,
`VALUE_NOT_DECODABLE`, `CONTROL_UNREACHABLE`, and `DECODED_NOT_CAUSAL`.

## 11. Scope and next branch

Ground-truth labels select high/low donors and score outcomes, so even
`ORACLE_VALUE_CAUSAL` is non-deployable mechanism evidence. Only that label
licenses a separate learned controller. No claim ID is reserved while the
repository claim re-grade remains open.
