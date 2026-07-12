# Adversarial Design Review

Completed before implementation of model stages and before any model call.

## Verdict

Proceed as a staged oracle mechanism experiment. The replicated lookup result
licenses native thought, but the domain shift and answer-leakage risks are large;
every gate below can terminate the line without opening later outcomes.

## 1. The target first operation may not be identifiable

The generator's latent pipeline is not automatically the unique explanation of
I/O. CPU review already found `negate`-first compositions algebraically
reorderable.

**Hardening:** exhaustively enumerate every concrete depth-2 pipeline against the
visible examples and require a singleton matching first-operation type. Keep
`negate` only as a second-operation/distractor. Test and record this per item.

## 2. Alias tokens may leak the answer or change the task

The artificial alias menu could make output easy or encode operation frequency.

**Hardening:** use a fixed one-to-one mapping, balance 11 identifiable target
types per split, verify every alias is one frozen-lens token, and keep operation
descriptions fully visible. Scope the result to alias-based first-op
identification; held alias permutations belong in a successor.

## 3. Whole-trace correctness mislabeled as every token's value

A successful final answer does not prove each preceding token was good.

**Hardening:** label a prefix only by three new continuations sampled from that
exact prefix. Never broadcast the original trace outcome.

## 4. Value can be task identity

Different correct aliases and task difficulty could drive a global probe.

**Hardening:** GroupKFold by task, training-fold standardization, task and
checkpoint-fraction detrending, and within-task pairwise AUC. Alias identity is
constant inside each primary comparison.

## 5. Prefix length or position can masquerade as certainty

Later prefixes may simply be more committed or closer to the answer.

**Hardening:** use two frozen fractional checkpoints, fraction detrending, and
explicit length/entropy comparators. High/low causal pairs must share task and
checkpoint fraction.

## 6. Shuffled thinking/token presence can explain decodability

The answer-token separability experiment found shuffled thinking as good as real
thinking.

**Hardening:** this study intervenes inside a natural coherent trace and requires
exact outcome change. A label-shuffled value axis is both a G1 null and a G2
causal control. No claim about coherent reasoning follows from decodability.

## 7. Continuation seeds leak between selection and evaluation

Selecting a high prefix on the same stochastic completions used for the effect
would guarantee regression artifacts.

**Hardening:** initial, value, causal-selection, and causal-evaluation seeds are
separate. Pair selection artifacts may contain selection outcomes; evaluation
does not run until the pair list and model are frozen.

## 8. High donors copy the correct answer rather than certainty

Full donor coordinates can contain the target alias explicitly.

**Hardening:** primary patch is one scalar task-general value axis. Full-J and
correct-alias clamps are identity upper bounds, not substitutes. A scalar pass
must beat shuffled-axis, exact random, and non-J controls. Wrong-task full J
tests identity transport directly.

## 9. The learned value axis uses hidden labels at test time

Weights trained on value labels are allowed, but donor selection is also
outcome-aware.

**Hardening:** label all stages oracle. Do not call a pass capability gain. A
separate experiment must select/apply a fixed target using only prefix features
and beat matched sampling on untouched tasks.

## 10. Applying a patch once is not context-local through generation

A prefill-only edit may vanish, while cache reuse can freeze inconsistent state.

**Hardening:** recompute the complete prefix without cache at every token and
reapply the fixed clamp at the same historical thought token. Baseline and all
controls use the identical full-recompute loop.

## 11. Repeated clamping changes realized deltas

Upstream patches alter later-layer inputs; quantization and suffix growth could
change geometry.

**Hardening:** measure sequential realized deltas in the live hook, require exact
causal invariance of the historical token across suffixes, and audit first/last
generation-step geometry. Any drift invalidates the control.

## 12. Pre-cast random orthogonality repeats the parent failure

The parent reached 5.7% span leakage after bf16.

**Hardening:** reuse the replicated post-bf16 exact lattice solver, two random
arms, a numeric-only calibration split, and fatal 1e-5 norm/1% projection gates
at confirmation too.

## 13. Numeric optimizer selects an inert model outcome

Trying candidates after seeing logits could manufacture favorable controls.

**Hardening:** selection sees only current residual, frozen dictionary, target
norm, and the two geometry errors. Calibration outcome logits are discarded.

## 14. Forced close creates unreachable prefixes

Prior work showed forced `</think>` can score states the model would not reach.

**Hardening:** require natural think close and parse. Never inject close tokens,
lower the minimum span, or rescue cap-bound traces.

## 15. Batch/cache/backend artifacts

Qwen3.5 batch-two logits differed materially in the parent replication, and
vLLM/HF seeds are not comparable.

**Hardening:** every call is unpadded Transformers batch one, bf16 SDPA,
`use_cache=False`. No vLLM runner is included.

## 16. Layer, scale, checkpoint, or threshold search

The large design space could overfit a positive cell.

**Hardening:** inherit band 4--8; use fixed fractions 0.33/0.67 and alpha-one
set clamps; no layer/scale selection. All thresholds and tie rules are frozen
before the model smoke.

## 17. The value probe is just a flexible residual probe

A full-state probe may decode value even when J coordinates do not.

**Hardening:** concatenated J coordinates are primary. The full-residual probe is
diagnostic and cannot rescue G1. Require J to beat correct-alias coordinate
activity, with a within-task shuffled-label null.

## 18. Scalar coordinate math can smuggle identity

Concatenating layerwise beta slices and patching each independently could move
more than one scalar.

**Hardening:** use the preregistered minimum-norm coordinate update
`Delta z_l = Delta s beta_l / ||beta||^2`, freeze clean desired coordinates, and
unit-test that only the global beta score changes in float32 before bf16 audit.

## 19. Non-J remainder is an unfair weak control

Raw donor energy may be much larger or have different layer allocation.

**Hardening:** decompose with the same frozen projector, then norm-match each
layer to the primary realized scalar delta. Report requested and realized norms.

## 20. Extra forward compute is mistaken for capability

Full recomputation and geometry search cost more than one baseline sample.

**Hardening:** report forwards, generated tokens, and matched additional baseline
coverage. This oracle experiment has no capability endpoint regardless of cost.

## 21. Low-prefix selection guarantees improvement by regression

Selecting observed V=0 and resampling can raise baseline even without a patch.

**Hardening:** pair selection and evaluation seeds are disjoint; primary is
paired scalar minus the newly resampled low-prefix baseline, not selection V.

## 22. Multiple positive arms rescue a failed primary

Full J, correct alias, raw donor, or ActAdd might work when certainty does not.

**Hardening:** only scalar certainty satisfies `ORACLE_VALUE_CAUSAL`. Identity
arms yield `IDENTITY_NOT_CERTAINTY`; other arms are explanatory.

## 23. Confirmation can be opened before feasibility is known

Native generation is expensive and invites partial peeking.

**Hardening:** immutable design -> model smoke -> seam -> G1 -> numeric control
calibration -> frozen pair receipt -> one causal confirmation. Each receipt hash
unlocks the next stage; outcome rows are buffered to completion.

## 24. Claim pressure after a spectacular oracle result

The lookup replication was perfect, and this line is explicitly capability
motivated.

**Hardening:** no claim ID during the open repository re-grade. Even a scalar
causal pass only launches a non-oracle experiment with matched sampling.

## Required assertions before causal confirmation

1. exact model revision, lens hash/rank, band, token IDs, and batch-one backend;
2. 80 unique fresh task fingerprints and zero parent overlap;
3. visible first-operation identifiability and balanced target support;
4. natural close/parse seam pass without injected tokens;
5. task-held-out J value gate and shuffled null;
6. frozen beta, normalization, quantile, pairs, and disjoint seeds;
7. 100% numeric calibration and confirmation control validity;
8. exact arm/item/seed completeness and full-prefix causal invariance;
9. primary scalar gate cannot be rescued by identity arms; and
10. explicit oracle/non-capability label in every report.
