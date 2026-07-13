# Preregistration: Jacobian Counterfactual Branching

Frozen before model load or outcome computation.

## Scope and claim boundary

This experiment asks whether the independently replicated early semantic J
space can *create* useful native-thought continuation diversity without knowing
the correct answer. It does not revisit scalar certainty, donor value, or
terminal score tuning.

Only `Qwen/Qwen3.5-4B` revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` is permitted. No teacher, judge,
other checkpoint, benchmark content, training, or adapter is allowed.

## Fresh procedural data

Generate exact-depth-two list transformations with the parent's generator and
new split seed: four mechanics, 24 qualification, and 48 confirmation tasks.
Reject every fingerprint colliding with any direct ancestor. Mechanics labels
exist for generator validation but no mechanics stage may load `first_op`,
`target_pipeline`, hidden examples, or correctness. Qualification and
confirmation are disjoint and may never pool.

## Balanced branch geometry

At each frozen layer 4--8:

1. normalize the first 12 concept columns of the frozen 24-concept J dictionary;
2. subtract their column mean, producing 12 branch deltas whose vector sum is
   exactly zero before bf16;
3. scale all columns by one common factor so RMS branch norm equals the frozen
   replicated donor-clamp median for that layer times alpha; and
4. use identical alias order `cat` through `square` at every layer.

For non-J control, factor the centered J branch matrix with SVD, replace its
left singular vectors by a deterministic orthonormal basis outside the complete
24-coordinate J span, and reconstruct. This must preserve the full 12 x 12 Gram
matrix, each norm, rank, and zero-sum relation—not just mean norm.

Both arms are applied to actual bf16 live residuals. Every realized non-J delta
must have norm error <=1e-5 relative to its paired realized J delta and J-span
projection <=0.01. Exact lattice repair from the independent transport
replication may be reused; no tolerance may be relaxed after model smoke.

## Label-free mechanics and alpha

Generate one 512-token prefix on each of four mechanics tasks. For alpha in
`[0.5,1.0,2.0]`, apply all 12 J and non-J branches to the final prefix token at
layers 4--8, immediately append the fixed close/`First:` slot, and measure each
branch's supplied target alias probability/choice. No correct alias is loaded.

Select the smallest alpha satisfying all:

- J target selection >=60%;
- mean J target probability lift over unpatched slot >=0.15;
- J minus non-J target selection >=35pp;
- all numeric, zero-sum, Gram, rank, cache, and finite controls pass.

No alpha passing is terminal `NO_NATIVE_J_BRANCH_CONTROL`; no correctness-
scored continuation opens. Alpha selection is a mechanism calibration, not a
capability result.

## Cache fork and generation policy

Generate a single native prefix to exactly 512 thought tokens while suppressing
`</think>` and EOS under a frozen fixed-cap policy shared by every arm. Cache
prompt plus the first 511 thought tokens once, expand it to 12 rows, and process
the final prefix token with clean/J/non-J hooks. Continue each row for exactly
512 more tokens under temperature 0.6, top-p 0.95, top-k 20. Branch target-to-
sampling-row assignment rotates deterministically by task so seed is not tied to
alias identity.

At cap 1,024 append exactly `</think>\n\nFirst:` and compute the 12 constrained
alias probabilities. Primary ensemble output is argmax mean probabilities;
majority, max-confidence trace, and minimum-entropy trace are mandatory
deployable diagnostics. Cache-based slot logits must match cache-free full
prefill within 0.05 max absolute alias-logit difference on mechanics; otherwise
use and charge full prefill everywhere.

## Arms

- `j_balanced`: 12 semantic branches.
- `clean_shared`: 12 clean stochastic continuations from the same midpoint.
- `non_j_gram`: 12 Gram/norm-matched generic branches.
- `full_sample_more`: 12 independent 1,024-token traces from the original
  prompt, created once as a master pool.

From the full master pool, determine before grading the largest global K whose
sampled-token resource and the largest K whose total logical model-token
resource do not exceed the J arm. Report both frozen prefixes and K=12 as an
overmatched reference. Pool order/seeds are frozen and never sorted by outcomes.

## Qualification and confirmation gates

On each stage independently:

- all rows, arms, branches, caches, probabilities, and resource receipts finite
  and complete;
- J mean-probability ensemble accuracy in [0.15,0.75];
- J accuracy at least +0.10 over clean-shared, non-J, sampled-token-matched full
  sampling, total-token-matched full sampling, and every stronger registered
  deployable selector on those pools;
- one-sided 95% paired-task bootstrap lower bound >0 for every comparison;
- J oracle answer coverage at least +0.05 over clean-shared and each matched
  full-sampling pool;
- J predictions span >=8 public aliases and successes >=6 target aliases; and
- every model/backend/numeric/cache/compute contract passes.

Qualification failure is terminal `NO_J_BRANCH_CAPABILITY`. Only a passing,
committed, pushed qualification may open confirmation. Confirmation uses the
same alpha, seeds policy, K resource rules, and gates. A two-stage pass is
`J_BRANCH_CAPABILITY_REPLICATED`; it supports a capability-elicitation claim but
not capability installation into weights.

## Statistics and artifacts

The task is the bootstrap and macro unit (10,000 resamples). Store complete
branch token IDs, hashes, alias probabilities, choices, intervention geometry,
post-bf16 controls, cache/resource traces, task predictions, win/loss/tie
tables, and automatic terminal summaries. No correctness may be read until all
arms for that split are complete and hash-locked.
