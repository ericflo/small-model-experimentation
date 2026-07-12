# Preregistration: Commit-Slot Jacobian Value Transport

Frozen before any model call. CPU-only task generation, exact enumeration,
fingerprint comparison, lens hashing, gate arithmetic, unit tests, and repository
validation may precede the immutable design commit. The first model-loading
stage is outcome-blind and may run only after the design boundary is anchored.

## 1. Scientific ladder

This experiment separates four statements:

1. **Constrained seam:** after a fixed thought budget, syntax-only commit plus a
   closed public answer vocabulary exposes a usable semantic decision.
2. **Value-readable:** correct-choice probability is held-out-by-task rankable
   from a scalar readout of the replicated J coordinates at an earlier thought
   position, beyond direct answer identity and ordinary output confidence.
3. **Value-causal:** writing that one scalar J coordinate changes the later
   slot decision beyond exact numeric, shuffled-axis, direct-logit, donor, and
   non-J controls.
4. **Deployable capability:** a label-free controller beats frozen inference and
   matched-compute sampling on untouched contamination-free tasks.

Statements 1--3 are eligible here. Statement 4 requires a new experiment. A
positive constrained seam alone is an elicitation result for this fixed
multiple-choice interface, not installation of a free-form capability.

## 2. Fixed model, lens, and backend

- Only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Replicated 24-concept context lens, SHA-256
  `e373b6e93956fdfc5cb446e9bee8249655707c8258a7868f0653d11f1ffd0213`.
- Frozen J band: layers 4, 5, 6, 7, and 8. No layer search.
- Transformers, bf16, SDPA, unpadded batch one throughout. Native trace and
  close-only generation use KV caching; slot measurement and all later
  activation edits use one exact cache-free full prefill. vLLM is excluded
  because the measurement requires internal activations, and no backend is
  mixed within a comparison.
- Temperature 0.6, top-p 0.95, top-k 20 for sampled native/free-form text.
  Constrained slot choice is deterministic argmax.
- Native think-open ID 248068 and think-close ID 248069 must match the pinned
  tokenizer. The 12 answer aliases must be distinct leading-space single tokens.

Model smoke may record architecture, package/GPU versions, token IDs, lens
ranks, input lengths, cache contracts, fixed-slot tokenization, finite logits,
and whether the constrained choice belongs to the alias set. It may not record
task correctness, trace text, the chosen alias, or any comparative outcome.

## 3. Fresh exact-depth substrate

Generate 96 local procedural tasks: 16 seam selection, 16 untouched seam
confirmation, 32 value fit, and 32 causal confirmation. Each task contains
eight visible and eight hidden examples from the inherited two-operation list
DSL. Only visible examples and the public operation-to-alias map enter the model
prompt.

For every visible set, exhaustive CPU enumeration must establish both:

- every consistent concrete depth-two pipeline has the same first-operation
  type; and
- no concrete depth-one operation fits all visible examples.

`negate` may occur second or as a distractor but is excluded from first-operation
target support because its compositions can be algebraically reorderable. The
11 eligible first-operation types are balanced within one item. All 96 task
fingerprints must be unique and absent from the four direct J/seam parents.
No file under `benchmarks/` is read or imported.

The first attempted seed block collided exactly with a prior procedural split.
CPU smoke detected this before model loading. The entire block was replaced;
only the replacement block in `configs/default.yaml` is registered.

## 4. Commit-slot policy

For one native sampled path and candidate cap `B`:

1. generate from the ordinary thinking prompt until natural close, EOS, or the
   maximum registered cap;
2. if natural close occurs at or before `B`, retain exactly the thought tokens
   before that close;
3. otherwise retain exactly the first `B` thought tokens;
4. EOS or another malformed termination before `B` is a failed row and remains
   in every denominator;
5. append exactly `</think>\n\nFirst:`; and
6. at the next position, take argmax over only the 12 public leading-space alias
   tokens.

The controller provides output syntax and a closed vocabulary, never answer
identity. The full-vocabulary logits are preserved before masking. Report the
unmasked top-is-alias rate, total alias probability mass, and correct-alias
full-vocabulary probability. These diagnostics measure how strongly the mask
coerces the decision; they are not post-hoc rescue gates.

## 5. Mechanism controls at the seam

Four interfaces are kept separate:

1. **real thought slot:** the registered policy above;
2. **no-thought slot:** the same prompt, immediate close, fixed syntax, and alias
   mask, with zero generated thought tokens;
3. **shuffled-thought slot:** the exact same thought-token multiset and length,
   deterministically permuted by a disjoint seed before the same close, syntax,
   and mask; and
4. **close-only free-form:** the exact real thought prefix plus close, followed
   by up to 16 sampled answer tokens without fixed syntax or an alias mask.

The no-thought arm controls the fixed interface but can still perform computation
inside its prefill. The shuffled arm is therefore load-bearing for the claim
that coherent autoregressive thought content, rather than length or token
presence, improves the constrained choice. Shuffling preserves any alias token
mentions, so real-over-shuffled cannot be explained by their bag-of-tokens
presence alone. The close-only arm measures the already observed emission wall;
its failure cannot invalidate a working registered slot, and its success cannot
rescue a failed slot.

For real thoughts, record whether any public alias token, the correct alias, and
the last-mentioned alias occur in the prefix. This is a verbalization diagnostic,
not a filter. No result may call the slot state nonverbal merely because the
model did not close naturally.

## 6. Seam selection

On 16 tasks, sample three paths each to cap 1,024. View the same 48 sampled paths
at caps 256, 512, and 1,024; these are 144 paired cap rows, not 144 independent
traces. Evaluate every control at each cap and select the smallest cap passing:

- constrained exact accuracy in `[0.20, 0.80]`;
- at least six tasks containing both correct and incorrect real-thought traces;
- 100% finite real slot rows;
- real-thought accuracy at least 0.05 above the task-level no-thought slot; and
- real-thought accuracy at least 0.03 above exact-length shuffled thought.

Chance under the mask is `1/12`; report the exact chance comparison but do not
treat trace rows sharing a task as independent. Full-vocabulary and free-form
metrics are diagnostics. The upper accuracy bound preserves causal headroom.
If no cap passes, terminal `COMMIT_SLOT_SEAM_FAIL`; confirmation, value, and
causal stages remain unopened. A ceiling result above 0.80 is reported as
constrained-interface saturation and may motivate a separate capability test,
but it does not open value fitting here.

## 7. Untouched seam confirmation

Open only the selected cap on 16 untouched tasks and three disjoint-seed traces
per task. The complete selection row files and summary are hash-locked before
model loading. Confirmation requires:

- constrained exact accuracy in `[0.20, 0.80]`;
- at least six mixed-correctness tasks;
- 100% finite real slot rows;
- at least +0.03 over the no-thought slot; and
- at least +0.02 over exact-length shuffled thought.

The smaller confirmation differences are frozen before outcomes and each still
requires at least one additional success among 48 rows. No other cap may rescue
confirmation. Passing yields `COMMIT_SLOT_SEAM_REPLICATED`; failure yields
`COMMIT_SLOT_SEAM_NOT_REPLICATED` and seals later stages.

## 8. Prefix value labels

Only a replicated slot seam opens the 32 value-fit tasks. Generate three native
paths per task to selected cap `B*`. Retain genuine open-thought prefixes at
exact fractions `0.5 B*` and `1.0 B*`; a path that naturally closes or reaches
EOS before a checkpoint is excluded from that checkpoint and counted.

For each retained prefix, run the exact fixed commit-slot prefill and define

`V(prefix) = constrained probability assigned to the gold alias`.

This is a deterministic, continuous, oracle label computed at that prefix. It
is not copied from whole-trace correctness and it is not estimated from later
free-form continuations. Also record the hard constrained argmax outcome,
margin, entropy, unmasked alias mass, and direct alias mentions.

## 9. J-space value model

Run one exact `prompt + thought prefix + </think> + fixed slot` prefill. Capture
the 24 frozen J coordinates at layers 4--8 at the final thought-token position,
which is causally before the supplied close and slot. Concatenate the five
24-coordinate vectors. The final slot logits supply `V`; suffix positions are
never used as J features.

Use four GroupKFold splits by task. Standardization parameters are fit on train
tasks only. Remove label-free feature means within each task/checkpoint group;
fit only on train labels. Primary evaluation compares out-of-fold scores for
higher- versus lower-`V` prefixes within the same held-out task and checkpoint
fraction, then macro-averages by task. Equal-label pairs are omitted and counted.

Required comparators are:

- frozen correct-alias J-coordinate activity at the thought endpoint;
- ordinary constrained slot margin and entropy;
- prefix length/checkpoint;
- a within-task/checkpoint shuffled-label J readout; and
- a full-residual linear probe, diagnostic only.

G1 requires at least 128 scored prefixes, at least 16 tasks with both correct
and incorrect hard slot choices, task-macro pairwise AUC at least 0.65, J minus
correct-alias activity at least 0.03, J minus ordinary slot margin at least 0.02,
and shuffled AUC within 0.05 of 0.50. The residual probe cannot rescue J. On a
pass, refit once on all value-fit tasks and freeze feature transforms, beta,
shuffled beta, direct-output comparator, and the training-score 0.90 quantile.

Because `V` uses the gold alias and is derived from the same decision logits,
even a pass is only evidence for a task-general internal ranker after the
identity/margin gates. It is not yet proof of an independent notion of
certainty.

## 10. Scalar J write

Let `z_l` be the 24 J coordinates at layer `l`, `beta_l` the frozen value-readout
slice, `s = sum_l beta_l^T z_l`, and `s*` the frozen value-fit 0.90 score
quantile. For a recipient with `s < s*`, define the minimum-coordinate-norm
global scalar update

`z'_l = z_l + (s* - s) beta_l / sum_j ||beta_j||^2`.

During the exact full commit-slot prefill, clamp only the final thought-token
coordinates at layers 4--8 to those clean desired values, then read the later
fixed-slot logits. Unit tests and live receipts must show the intended scalar
movement and preserve the orthogonal part of each layer's J coordinates before
bf16 realization. No donor coordinates or answer identity enter the primary
update.

## 11. Numeric control calibration

Before causal outcomes, construct controls on at least 16 eligible seam-prefix
states using the frozen beta and target quantile. For every live recipient and
layer, build two independent random residual deltas from 32 fixed starts using
the replicated quantization-aware correction and exact bf16 lattice solver.
Candidate selection sees only the current activation, frozen dictionary,
requested primary norm, norm error, and J-span projection.

Every applied random delta must satisfy, after bf16 application:

- relative norm error `<= 1e-5`; and
- realized projection into the full 24-dimensional J span `<= 0.01`.

Controls are constructed separately for each live prefix and layer and cannot
be reused across lengths. Any failure yields `CONTROL_UNREACHABLE`; thresholds
cannot be relaxed after outcomes.

## 12. Untouched causal pairs

Only G1 and numeric calibration passes open the 32 causal-confirmation tasks.
Generate three disjoint-seed native paths/task and measure the two registered
fractions with the frozen slot. Within each task and fraction, deterministically
choose the largest gold-probability gap satisfying:

- low hard slot choice is incorrect and `V_low <= 1/3`;
- high hard slot choice is correct and `V_high >= 2/3`; and
- the low recipient's frozen scalar score is below `s*`.

Ties use stable row IDs. Require at least 16 task-level pairs before any patched
slot outcomes are summarized. Pair choice is oracle and used only to test the
mechanism. Since the slot is deterministic, baseline and patched arms reuse the
same exact prefix; there is no continuation-resampling claim.

## 13. Causal arms

Evaluate all arms in one fixed order-independent harness at the same final
thought position and exact sequence:

1. unpatched low prefix;
2. scalar J value write to `s*` (primary);
3. full 24-J same-task high donor;
4. correct-alias versus baseline-choice J identity write;
5. within-task/checkpoint shuffled value beta;
6. exact random A;
7. exact random B;
8. direct current-slot correct-versus-baseline logit-Jacobian write;
9. full raw same-task high activation;
10. full J-span component of the high-minus-low donor delta;
11. per-layer norm-matched non-J donor remainder;
12. value-fit high-minus-low ActAdd frozen before causal tasks; and
13. full-J donor from a different task with a different gold alias.

All non-donor magnitude controls match the primary realized delta per layer.
Both random arms repeat both post-bf16 gates on every confirmation row. Direct
coordinate realization, final logits, constrained and unmasked probabilities,
hard choice, prefix length, and alias verbalization are reported. Identity,
full-state, direct-output-gradient, or ActAdd success cannot rescue the primary.

## 14. Frozen causal decision

Require at least 16 eligible task-level pairs. `ORACLE_COMMIT_VALUE_CAUSAL`
requires all of:

- scalar-write hard-success uplift at least 0.10 over paired baseline;
- a task-paired 10,000-resample 95% bootstrap lower bound above zero;
- scalar minus the better-performing exact random arm at least 0.08;
- scalar minus shuffled beta at least 0.08;
- scalar minus matched non-J at least 0.05; and
- every design, value, numeric, sequence-position, arm-completeness, and
  realization contract.

If the J/full-donor/direct-identity arms work while scalar value fails, use
`IDENTITY_NOT_VALUE`. Other terminal labels are `VALUE_NOT_DECODABLE`,
`CONTROL_UNREACHABLE`, and `DECODED_NOT_CAUSAL`. No secondary arm or subgroup
may rescue a failed primary.

## 15. Capability and matched-compute boundary

The fixed slot is label-free after its global cap is selected, but the value
label, beta training, high/low causal pair selection, and causal scoring use
gold answers. Thus even `ORACLE_COMMIT_VALUE_CAUSAL` is oracle mechanism
evidence. Report all generated and prefill tokens, full forward calls, and
matched additional sampling coverage, but do not call this a capability gain.

Only a successor may freeze a label-free score/controller and test it once on
fresh procedural tasks. That successor must beat the strongest fixed slot,
close-only/free-form output, no-thought slot, coherent-thought controls, and
matched-compute sampling on the same backend before making a capability claim.

## 16. Stage and artifact discipline

The README, this preregistration, adversarial review, and semantic config payload
are hash-anchored to an ancestor design commit. Every procedural split is also
checked against its CPU manifest at runtime. The order is immutable design ->
outcome-blind model smoke -> seam selection ->
untouched seam confirmation -> value fit -> numeric calibration -> frozen causal
pair receipt -> one causal confirmation. Every stage hash-unlocks the next.
Incomplete output rows are buffered and cannot produce a scientific summary.
Until separately implemented and audited, value, calibration, and causal stages
must fail closed rather than emit placeholders.

Preserve every negative and failed control. Update owning program ledgers and
shared synthesis at terminal gates. No claim ID is reserved while the repository
claim re-grade remains open.
