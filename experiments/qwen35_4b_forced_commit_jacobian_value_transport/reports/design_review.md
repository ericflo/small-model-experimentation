# Adversarial Design Review: Forced-Commit Jacobian Value Transport

Completed before implementation of value/causal stages and before any model
call. CPU-only generation, exhaustive enumeration, lens hashing, gate arithmetic,
and unit tests produced no model outcome.

## Verdict

Proceed as a staged counterfactual-policy mechanism study. Injected close is an
authorized test-time action only because the same action is defined as the
deployment policy. The seam must independently pass C51-shaped parsing and
headroom gates before any internal value result can be opened.

## 1. Forced close is rebranded as natural termination

The model did not autonomously choose this state; calling it natural would erase
the two terminal seam failures.

**Hardening:** every injected row stores
`counterfactual_to_natural_close=true`; all artifacts say forced policy. Natural
and forced branches are reported separately.

## 2. An artificial state is declared illegitimate by definition

Conversely, rejecting every controller action would also reject the repository's
deployed forced-budget baselines.

**Hardening:** legitimacy is operational: calibration, value labeling, causal
testing, and later deployment must perform the exact same append-close action.
No inference is made about autonomous close.

## 3. C51 already showed forced answers are non-actionable

C51 found only 13.2% fresh parse after injected close.

**Hardening:** force-only parse >=90%, policy parse >=90%, answer-cap contact
<=5%, success headroom, and mixed tasks are fatal first-stage gates. Value cannot
rescue a bad interface.

## 4. The selected cap overfits answer correctness

Success/headroom and mixed-task gates use gold labels.

**Hardening:** cap selection is training-side policy calibration only. Freeze the
smallest passing cap, then repeat every threshold on untouched tasks/seeds. A
later controller deploys only that fixed global cap/action without test labels.

## 5. A larger cap rescues failed confirmation

Trying another cap after confirmation would turn it into tuning.

**Hardening:** confirmation opens only the selected cap. Any miss is terminal and
requires a new experiment.

## 6. Paired cap rows are counted as independent traces

Three views of one sampled path are not three traces.

**Hardening:** selection reports 48 traces and 144 paired policy rows; no pooled
independence claim or between-cap p-value is permitted.

## 7. Natural and forced commits are pooled to hide forced failure

If natural rows parse but forced rows fail, overall policy parsing could pass.

**Hardening:** require both overall and forced-only parse >=90% and a forced
share >=50%. Report natural and forced counts at every cap.

## 8. EOS-before-cap is silently converted into a forced prefix

Replaying through an EOS token is not the registered policy.

**Hardening:** EOS-before-cap is `malformed_pre_cap`, parse failure, and never
force-replayed. The trace remains in all denominators.

## 9. An answer cap hides incomplete formatting

The model may emit a parseable alias and then ramble until truncation.

**Hardening:** freeze 16 answer tokens and require answer-cap contact <=5% in
both selection and confirmation, in addition to parsing.

## 10. The injected close token directly supplies the correct alias

The close token is identical across tasks and contains no answer identity, but
could still act as a generic motor cue.

**Hardening:** that generic cue is part of every baseline/control arm. Causal
success is scalar-minus-identical forced baseline, not forced versus natural.

## 11. The thought prefix already verbalizes the answer

J value may merely read a written alias near the endpoint.

**Hardening:** primary comparisons are within task/fraction; correct-alias
coordinate activity is a required comparator; correct-alias clamp, full-J, and
wrong-task donor are causal identity controls. Report first answer/alias mention
positions diagnostically.

## 12. “Exactly two” is behaviorally false

The prior generator admitted rare rows with a complete depth-one explanation,
which can induce endless rechecking.

**Hardening:** exhaustively reject every visible set with any matching concrete
depth-one operation before model work. Preserve the change as a scope boundary.

## 13. The first operation remains ambiguous among depth-two pipelines

A latent sampled pipeline is not automatically identifiable.

**Hardening:** enumerate every concrete depth-two pipeline and require all
visible-consistent candidates to share one first-operation type.

## 14. `negate` target identity is algebraically reorderable

The parent already found this target invalid.

**Hardening:** keep `negate` only as second operation/distractor and exclude it
from target support.

## 15. Aliases leak frequency or use multiple tokens

Output difficulty could differ by target.

**Hardening:** fixed balanced one-to-one aliases, model-smoke verification of 12
unique leading-space token IDs, and within-task primary metrics.

## 16. Fresh rows overlap prior observed tasks

Overlap would contaminate the new interface/value confirmation.

**Hardening:** require 96 unique fingerprints and zero overlap with all three
Jacobian/seam parents before model loading.

## 17. Hidden examples leak into the model prompt or training

The hidden half exists only for task fingerprints and generator audits.

**Hardening:** render visible examples only. No benchmark content or hidden
examples enter prompts, labels beyond the generated first-op gold, or training.

## 18. Trace and answer seeds leak across stages

Reusing a continuation for cap selection, value fitting, donor selection, and
causal evaluation would overstate stability.

**Hardening:** separate base seeds for every listed stage; stable IDs include
task, trace, cap/fraction, rollout, and arm. Hash stage artifacts before opening
the next split.

## 19. Whole-trace correctness is broadcast to every prefix

A final answer does not label earlier thought tokens.

**Hardening:** define `V(prefix)` only from three new forced-policy continuations
sampled from that exact prefix. Original trace outcomes are never copied.

## 20. Prefixes after natural close are treated as thoughts

If a trace closes before a frozen checkpoint, later tokens are answers.

**Hardening:** exclude and count any natural/EOS termination before each
checkpoint. Causal pairs require genuine open thought at the exact position.

## 21. Value is task identity or target alias

A global probe may rank easy tasks instead of better states.

**Hardening:** GroupKFold by task, train-fold-only standardization/detrending,
and within-task, same-fraction pairwise AUC. Alias is constant inside each
primary comparison.

## 22. Prefix fraction or length masquerades as certainty

Later prefixes are closer to forced answer and have different kernel geometry.

**Hardening:** exact fractions 0.5/1.0, fraction detrending, length comparator,
same-fraction causal pairs, and per-live-length controls.

## 23. Appending close changes endpoint activations numerically

The natural-seam parent measured suffix-length bf16 drift.

**Hardening:** fit features and perform every arm on the exact
`prompt+prefix+close` sequence. Do not compare a prefix-only activation to a
prefix-plus-close patch. Sequence length is matched within each row.

## 24. The close token semantically leaks backward through attention

A future token should not affect the endpoint under causal masking, but kernel
implementation bugs are possible.

**Hardening:** audit causal masks/model architecture in smoke and compare
same-length close-versus-neutral endpoint activations diagnostically before
causal confirmation. Any semantic backward dependence invalidates the study.

## 25. A prefill edit does not survive into answer generation

Changing one hidden vector might be discarded rather than written into cache.

**Hardening:** patch during the exact prefill before cache creation. Include
full-J/correct-alias positive controls and record direct coordinate realization;
their failure blocks scalar interpretation.

## 26. KV caching is silently disabled

Full-prefix recomputation would reintroduce suffix-dependent historical states.

**Hardening:** audit every forward input length: one complete prefill, then only
single-token calls. Any failing row invalidates its stage.

## 27. Batch/backend differences manufacture effects

Prior Qwen work observed batch and vLLM/HF differences despite equal seeds.

**Hardening:** unpadded Transformers batch one, bf16 SDPA, one backend/model
revision for every arm and baseline. No cross-backend sample comparisons.

## 28. A full residual probe rescues J-space failure

High-dimensional residual features may decode value when the 24 coordinates do
not.

**Hardening:** full residual is diagnostic only. G1 is J pairwise AUC plus a
required advantage over correct-alias activity and shuffled null.

## 29. Scalar beta smuggles answer identity

Even a single readout can align with the correct alias coordinate.

**Hardening:** minimum-norm global beta update, alias-activity comparator,
correct-alias causal arm, shuffled beta, and wrong-task/different-alias donor.
Only scalar-specific success satisfies the primary label.

## 30. Full donor state copies a completed solution

Same-task high prefixes may encode the correct answer explicitly.

**Hardening:** full-J/raw donor arms are oracle upper references. The primary
copies only one learned scalar score and must beat identity/shuffled/random/non-J
controls.

## 31. Pre-cast random orthogonality repeats the invalid parent

Bf16 rounding can reintroduce J-span leakage and norm error.

**Hardening:** reuse the replicated quantization-aware continuous correction and
exact bf16 lattice solver; require <=1e-5 relative norm and <=1% realized span
projection for every layer at calibration and confirmation.

## 32. One random control is reused across lengths

The live bf16 lattice depends on recipient activation and sequence length.

**Hardening:** construct two independent random arms separately for every live
recipient prefix and layer. Never reuse a basis across rows/lengths.

## 33. The random solver chooses controls by model outcome

Trying candidates against answer logits would manufacture inert controls.

**Hardening:** solver selection sees geometry only—current activation,
dictionary, target norm, and errors. Outcome logits are unopened during control
construction.

## 34. The non-J control is unfairly weak

A donor remainder can have different norm/layer allocation.

**Hardening:** decompose with the frozen projector and match each layer's
post-bf16 realized norm to the primary scalar delta.

## 35. ActAdd is tuned on confirmation

Scale/layer search could create a favorable secondary arm.

**Hardening:** freeze value-fit high-minus-low ActAdd, inherited band, and
alpha-one before causal tasks. It is explanatory and cannot rescue scalar
failure.

## 36. Low-prefix resampling improves by regression alone

Selecting `V<=1/3` guarantees some fresh baseline rebound.

**Hardening:** donor-selection and evaluation seeds are disjoint; primary is the
paired scalar-minus-new-baseline difference, not scalar minus selection `V`.

## 37. Pair selection is impossible at a saturated cap

If policy success is too high, low donors and +0.10 gain are unattainable.

**Hardening:** seam success must retain 5%--95% headroom; causal pairs require
low<=1/3/high>=2/3; run the lifecycle feasibility receipt after pairs and before
outcomes. No threshold may relax after baseline observation.

## 38. Multiple positive arms rescue a failed primary

Identity, full J, raw donor, or ActAdd could look spectacular.

**Hardening:** only scalar beta meets `FORCED_VALUE_CAUSAL`. Identity-only success
yields `IDENTITY_NOT_VALUE`; all other arms are explanatory.

## 39. Generated close is counted as model capability

The controller supplies a token and extra forward work.

**Hardening:** every baseline/control receives the identical close/action and
answer budget. Report token/forward costs. The oracle stage has no capability
endpoint regardless of point estimate.

## 40. Fixed forced commit itself beats sampling and is overclaimed

A globally selected cap may be deployable, but cap selection used training-side
gold and has no matched-compute comparison here.

**Hardening:** report seam accuracy only as interface feasibility. Any capability
claim requires a separate untouched comparison against frozen and matched
sampling with the cap frozen in advance.

## 41. “Sample more” is replaced by more labeled rollouts

Three labels and two evaluation rollouts consume extra compute.

**Hardening:** labels/donors are oracle mechanism instrumentation. Report matched
additional baseline coverage; only a later non-oracle controller can compete
with sampling at equal total tokens/forwards.

## 42. The stricter task generator changes the substrate

Removing depth-one collapses may improve behavior relative to parents.

**Hardening:** state this scope explicitly; never pool accuracies with parents.
All arms and all four new splits use the same hardened generator.

## 43. Decoded thought inspection tunes parsers or thresholds

Qualitative tails can invite exceptions.

**Hardening:** buffer full stage rows, compute the frozen decision automatically,
and inspect decoded content only after a complete summary. Parser and thresholds
are immutable.

## 44. Partial files become scientific evidence

Long trace generation can be interrupted.

**Hardening:** hold rows in memory until stage completion; only complete files
receive hashes and summaries. Progress output contains counts only.

## 45. Unimplemented value/causal code emits a placeholder pass

Design prose could outrun executable contracts.

**Hardening:** runner raises for `prefix-value`, `control-calibration`, and
`causal-confirmation` until audited implementations and tests are committed.

## 46. Oracle result is promoted under mission pressure

Gold labels train beta and choose high donors.

**Hardening:** no claim ID; all reports say oracle. A separate fixed non-oracle
controller must replicate a contamination-free held-out gain over strongest
controls and matched sampling.

## Required assertions before seam selection

1. immutable published design commit and matching README/prereg hashes;
2. exact model revision, lens hash/rank, layer band, token IDs, aliases, and
   cache-forward contract;
3. 96 unique fresh exact-depth fingerprints, zero parent overlap, balanced
   identifiable target support, and no benchmark content;
4. exact caps, seeds, policy action, prompt/context envelope, and answer budget;
5. mathematically reachable selection/confirmation gates; and
6. explicit counterfactual/oracle/no-claim scope.

## Required assertions before value and causal stages

1. smallest-cap selection and untouched same-cap seam replication;
2. complete hash-bound prefix/value rows with disjoint seeds;
3. task-held-out J G1, alias comparator, and shuffled null;
4. frozen beta/normalization/quantile/pairs and feasibility receipt;
5. 100% per-live-prefix post-bf16 random validity;
6. exact arm/item/seed/cache/position completeness; and
7. primary scalar gate cannot be rescued by identity or secondary arms.
