# Adversarial Design Review: Commit-Slot Jacobian Value Transport

Completed before any model call. CPU generation, exact task enumeration, prior
fingerprint comparison, lens hashing, gate arithmetic, static review, and unit
tests exposed no model outcome. This review treats a convenient positive result
as the failure mode to defend against.

## Verdict

Proceed only as a staged constrained-interface and oracle-mechanism study. The
slot repairs a real emission failure, but it also changes the task into forced
choice. It earns scientific use only if coherent real thought beats both the
same interface with no generated thought and an exact-length shuffled-token
control on untouched tasks. Value and causal code remain unauthorized until the
slot seam independently replicates and those stages receive another
implementation audit.

## 1. The alias mask guarantees a parse and is mistaken for capability

Every finite row must choose one of 12 aliases, so format success is true by
construction.

**Hardening:** never report slot parse rate. Report exact semantic accuracy,
chance `1/12`, no-thought and shuffled controls, unmasked alias mass, and the
unconstrained top token. Scope any positive to the registered closed-choice
interface.

## 2. Fixed syntax secretly supplies the answer

`First:` narrows what kind of object comes next and might be described as a
reasoning intervention.

**Hardening:** the literal suffix is identical across all tasks and arms and
contains no alias. Call it an output interface, not an internal edit. Answer
identity is forbidden outside oracle labels and named identity controls.

## 3. The mask selects among negligible logits

The highest alias can win after masking even if the model assigns essentially
all probability to more analysis.

**Hardening:** preserve full-vocabulary logits; report total alias probability
mass, correct-alias full-vocabulary probability, and top-is-alias rate. A masked
gain remains legitimate only as constrained choice, never autonomous commit.

## 4. “No thought” still computes in the slot prefill

The final slot token can attend to the full task prompt and perform substantial
depth-wise computation even without generated tokens.

**Hardening:** treat no-thought as an interface baseline, not a no-compute arm.
Add exact-length shuffled thought as the load-bearing coherent-content control.

## 5. Token count or token presence, not coherent thought, causes the gain

Prior repository evidence found shuffled thinking can mimic representational
effects.

**Hardening:** deterministically permute the exact thought token multiset and
length, then append the identical close, slot, and mask. Selection and untouched
confirmation both require real-over-shuffled accuracy.

## 6. Shuffling removes alias mentions and creates an easy straw control

If the real trace writes the answer token, a shuffle that deletes it would be
unfair.

**Hardening:** shuffle positions only; preserve the exact token multiset, hence
all alias mentions. Record moved-position rate and require exact multiset/length
equality in unit tests.

## 7. Shuffling creates nonsense unlike any reachable state

An incoherent sequence is out of distribution and cannot by itself prove the
real thought algorithm is necessary.

**Hardening:** make only the narrow content-over-bag-of-tokens inference. Keep
no-thought and paired cap views too. A later successor can add truncated or
foreign coherent controls; this experiment cannot overclaim necessity.

## 8. The trace directly verbalizes the correct alias

J value could simply read a written answer near the endpoint.

**Hardening:** record any/correct/last alias mentions in every real prefix,
preserve mention-positive and mention-negative slices, compare correct-alias J
activity, and retain shuffled token order where the same mention tokens remain.
No verbalization slice rescues a failed primary gate.

## 9. The prompt itself contains every alias

The public mapping is necessary to define the task, but alias frequency in the
prompt could dominate the next token.

**Hardening:** all tasks share the same balanced mapping and output grammar;
within-task trace comparisons hold it fixed. No-thought quantifies prompt-only
slot behavior.

## 10. Alias tokenization or lexical frequency biases targets

Some choices may be easier because they are common or split differently.

**Hardening:** model smoke requires 12 unique leading-space single token IDs;
target types are balanced; report per-alias outcomes. Alias permutation is a
successor robustness test, not a hidden post-hoc arm.

## 11. Near-chance accuracy is called a working seam

With 12 choices, a few successes can occur by chance.

**Hardening:** raise the frozen range floor to 0.20 (at least 10/48), require six
mixed tasks, +5pp over no-thought and +3pp over shuffled in selection, then
repeat on untouched tasks. Report task-level uncertainty without pretending 48
rows are independent.

## 12. High accuracy leaves no room for a causal improvement

A slot at 95% would look strong but make the later +10pp gate impossible.

**Hardening:** cap seam eligibility at 0.80. A saturated constrained interface
is preserved as an elicitation result and branched separately; it cannot open
the value-causal ladder.

## 13. Gold labels choose the most favorable thought cap

Cap selection explicitly uses correctness and control gaps.

**Hardening:** label it training-side policy calibration, select the smallest
passing cap, then reopen only that cap on untouched tasks/seeds. Deployment may
use the one frozen global cap without test labels.

## 14. A larger cap rescues failed confirmation

Trying the other registered rungs after seeing confirmation would convert the
holdout into tuning.

**Hardening:** hash-lock selection and open exactly one cap. Any miss is terminal
for this experiment.

## 15. Different selection and confirmation gaps are outcome tuning

Selection uses +5/+3pp while confirmation uses +3/+2pp.

**Hardening:** both sets are frozen now, before model loading, and integer gate
reachability is recorded. The confirmation thresholds cannot change after any
selection outcome.

## 16. Paired cap views are counted as independent traces

Caps 256, 512, and 1,024 are right-censored views of the same sampled path.

**Hardening:** report 48 traces and 144 paired views, never 144 samples. Choose a
cap descriptively; do not compute a pooled between-cap significance test.

## 17. Early natural closes create duplicated cap rows

One naturally closed thought may be replayed identically at all larger caps.

**Hardening:** label `natural_prefix_replayed`, report unique prefix counts, and
retain duplicates only as paired policy evaluations. Never count them as new
traces or force them to an artificial longer position.

## 18. EOS-before-cap is silently replayed as valid thought

Appending after EOS would create a state outside the registered policy.

**Hardening:** mark it `malformed_pre_cap`, score it incorrect/nonfinite in all
denominators, and never pass it to a slot or free-form replay.

## 19. Close-only free-form and constrained argmax are compared as if decoding matched

One arm samples up to 16 tokens; the other takes a masked deterministic token.

**Hardening:** use free-form only as an interface diagnostic. Primary mechanism
comparisons are real, shuffled, and no-thought under the identical slot argmax.

## 20. The exact-depth task is secretly solvable in one operation

Then longer thought or J value would be testing a misdescribed substrate.

**Hardening:** exhaustively reject every visible set with any concrete
depth-one fit and persist the zero-count receipt before model loading.

## 21. The latent first operation is not behaviorally identifiable

Several depth-two pipelines may fit the visible examples with different first
operations.

**Hardening:** enumerate all concrete depth-two matches and require a singleton
first-operation type for every item.

## 22. Algebraic reordering makes `negate` an invalid target

The parent CPU audit already found this failure.

**Hardening:** exclude `negate` from first-operation target support while keeping
it only second/distractor; assert the target support in tests.

## 23. Hidden examples leak into prompts or learning

The generator creates hidden rows that could accidentally influence labels or
training.

**Hardening:** render visible examples only. Hidden examples are used solely in
task construction/fingerprinting audits and never as model input or training
data.

## 24. Benchmark content contaminates the capability line

Reading held-out suite sources would permanently contaminate later work.

**Hardening:** use only the self-contained procedural generator. Never read or
import `benchmarks/`; repository validators and the data manifest record the
firewall.

## 25. Fresh tasks collide with a parent

The first CPU attempt in fact reused an exact generator seed from prior work.

**Hardening:** replace the entire seed block, compare all 96 fingerprints to all
four direct parents, require zero overlap, and preserve the caught collision in
the log rather than hiding it.

## 26. Seeds leak between scientific stages or controls

Reusing paths could make replication and null arms dependent.

**Hardening:** separate base seeds for selection trace/free-form/shuffle,
confirmation trace/free-form/shuffle, value, value shuffle, causal trace, two
random arms, and bootstrap. Stable IDs include task, trace, cap/fraction, and
arm.

## 27. A partial run is summarized after favorable early tasks

Long generation invites interruption and selective completion.

**Hardening:** write scientific summaries only after exact row-count and cache
contracts pass. Interrupted raw work is non-evidence and cannot unlock another
stage.

## 28. Cached versus full-prefill execution changes the result

Native generation uses caching while slot logits use full recomputation.

**Hardening:** this distinction is policy-defined: cache only samples the token
sequence; every real/shuffled/no-thought slot and every later intervention uses
the same exact full prefill. Audit cached generation lengths and never compare
HF with vLLM samples.

## 29. Batch effects manufacture a control gap

Qwen3.5 work in this repository has observed batch-sensitive logits.

**Hardening:** unpadded batch one for every arm, bf16 SDPA, same revision and
runner. No vectorized batch shortcut is authorized.

## 30. The continuous value label is a tautological output logit

`V` is the gold alias probability from the same slot whose outcome is later
edited.

**Hardening:** state this plainly. Capture J only at the causally earlier thought
position, require held-out-task within-group ranking, beat ordinary slot margin
and correct-alias activity, and call a pass a predictive internal coordinate—not
independent metacognition.

## 31. Gold answer identity is smuggled into a deployable score

Selecting the correct component of the probability vector requires labels.

**Hardening:** label value fitting and causal pair selection oracle. The frozen
beta can later be evaluated label-free, but this experiment cannot claim that
successor outcome in advance.

## 32. Value is task or alias identity

Easy aliases or task families could drive a global probe.

**Hardening:** GroupKFold by task; within-task, same-fraction pairwise metrics;
label-free group feature centering; balanced target support; and correct-alias
coordinate comparator. Test tasks never enter fitting.

## 33. Prefix length masquerades as certainty

The later checkpoint may have systematically larger values and different
activation geometry.

**Hardening:** compare only within the same task and fraction, group-center
features, report length/fraction comparators, and require causal high/low pairs
to share a fraction.

## 34. The frozen lens has no validity on this domain

It was built from context-local concept prompts, not long native reasoning.

**Hardening:** treat this as the hypothesis, not an assumption. Verify hash and
rank only; require held-out G1 and causal specificity. A residual probe cannot
rescue J-space failure.

## 35. The correct-alias J coordinate alone explains everything

Beta may merely detect or amplify the answer word.

**Hardening:** require J value AUC to exceed correct-alias activity by 0.03;
include alias-mention diagnostics and a correct-alias causal identity arm. If
identity works and scalar value does not, label `IDENTITY_NOT_VALUE`.

## 36. Ordinary output confidence explains everything

A margin/entropy heuristic could rank traces without any special J coordinate.

**Hardening:** require J AUC to beat constrained slot margin by 0.02 on held-out
tasks and include a direct current-slot logit-Jacobian causal control.

## 37. A flexible residual probe rescues J failure

High-dimensional state may decode the label even when the frozen coordinates do
not.

**Hardening:** full residual is diagnostic only. G1 and every later opening
depend on J metrics and frozen comparators.

## 38. Suffix activations leak backward into the thought feature

The prefill includes close and slot tokens after the captured thought position.

**Hardening:** audit causal masking and capture only the final thought position.
Compare prefix-only versus suffix-included thought activations in model smoke or
numeric calibration; any nonzero semantic backward effect invalidates the
sequence construction.

## 39. Patching the slot token directly changes the answer

That would be an ordinary output-logit intervention, not workspace transport.

**Hardening:** the primary hook may touch only the final thought-token position
at layers 4--8. Slot-position gradients are named direct-output controls only.

## 40. Five layerwise writes are falsely called one scalar

Independent per-layer manipulation could carry multiple degrees of freedom.

**Hardening:** use one frozen global beta and one scalar target. The unique
minimum-coordinate-norm update is determined by `s*-s`; unit-test the score
change and preservation of beta-orthogonal coordinate components.

## 41. Beta scale or regularization is tuned on confirmation

Choosing L2, standardization, layer slices, or target quantile after outcomes
could make an arbitrary axis look causal.

**Hardening:** freeze L2=1, train-only standardization, band 4--8, and the
training 0.90 score quantile before model loading; no causal scale sweep.

## 42. The shuffled-label null leaks group structure

Global shuffling can create an easy null while preserving task confounding in
the real model.

**Hardening:** shuffle labels only within task and checkpoint fraction, use the
same folds/features/regularization, and require null AUC within 0.05 of chance.

## 43. Pre-bf16 random orthogonality repeats the invalid parent

Rounding can restore J-span projection or change perturbation norm.

**Hardening:** use the independently replicated quantization-aware correction
and exact bf16 lattice repair. Enforce <=1e-5 realized norm error and <=1% full-J
projection for every layer in calibration and confirmation.

## 44. One random direction is reused across recipients or lengths

The live bf16 lattice depends on each activation and sequence.

**Hardening:** construct random A and B independently for each recipient, layer,
and live prefix. Never reuse a control basis across rows.

## 45. The random solver searches model outcomes

Selecting a candidate because it leaves the answer unchanged would manufacture
an inert null.

**Hardening:** candidate selection sees geometry only. Outcome logits remain
unopened during control construction, and the receipt says so explicitly.

## 46. The non-J arm is weaker by construction

Different layerwise norms could make a donor remainder an unfair comparator.

**Hardening:** project with the same frozen J dictionary and match each layer's
post-bf16 realized norm to the primary scalar delta. Report requested and
realized norms.

## 47. Full donor state copies a solved answer

The high prefix may contain task identity and a completed solution.

**Hardening:** full J/raw/J-component donor arms are oracle upper references.
The primary receives no donor state. A wrong-task/different-alias donor tests
identity transport and cannot count toward scalar success.

## 48. Direct output gradients make any answer writable

An oracle correct-versus-wrong slot gradient may trivially flip the next token.

**Hardening:** name it a direct-logit identity control and exclude it from the
primary decision. Its success only proves endpoint editability.

## 49. Pair selection guarantees a favorable baseline

Choosing incorrect low and correct high states uses gold outcomes.

**Hardening:** mark pair selection oracle, freeze pairs before patched summaries,
and compare every arm on the exact same low prefix. The deterministic baseline
is not replaced by a new stochastic draw.

## 50. Pair and absolute gates are mathematically infeasible

Too few mixed groups or a saturated baseline could make success impossible.

**Hardening:** seam headroom is frozen at 0.20--0.80; require 16 causal pairs and
write a pair-feasibility receipt before outcomes. Never lower a gate after
observing availability.

## 51. Trace rows are used as independent bootstrap units

Three paths and two fractions within one task share the task and answer.

**Hardening:** causal pairs are at most one registered pair per task, and the
bootstrap resamples tasks. Observational AUC macro-averages tasks.

## 52. Multiple positive arms rescue a failed scalar

Full J, correct alias, direct gradient, raw donor, or ActAdd may work when value
does not.

**Hardening:** only the frozen scalar arm can earn
`ORACLE_COMMIT_VALUE_CAUSAL`. Secondary arms explain a failure; none change the
terminal primary label.

## 53. A masked one-token win is called free-form capability

The interface removes format and vocabulary uncertainty.

**Hardening:** every report says constrained choice. A successor must compare
the deployed slot and natural/free-form interfaces separately on fresh tasks.

## 54. Oracle mechanism evidence is called installed capability

Gold labels fit beta and choose causal pairs.

**Hardening:** reserve capability language for a label-free successor that beats
frozen and matched-compute sampling. Even perfect causal transport here remains
oracle.

## 55. More compute, not a better policy, explains a later gain

Trace generation, full prefills, and control construction consume many forwards.

**Hardening:** report generated tokens, prefill tokens, forward calls, and the
matched additional-sampling coverage. The successor's endpoint must beat that
matched-compute baseline on the same backend.

## 56. Unimplemented later stages emit optimistic placeholders

Scaffolding can accidentally look like a pass in catalogs or reports.

**Hardening:** value, calibration, and causal commands raise a fatal unavailable
error until audited implementations are committed after seam replication.

## 57. Claim-number pressure causes a premature ledger entry

Concurrent agents and an open claim re-grade make collisions especially costly.

**Hardening:** reserve no claim ID. Commit and synchronize experiment-local work
first; update shared evidence only at terminal gates under the repository's
rebase/check/push protocol.

## 58. A spectacular result weakens the reporting boundary

This line is explicitly motivated by capability installation, inviting motivated
reasoning after a positive outcome.

**Hardening:** terminal labels, oracle wording, control advantages, headroom,
and successor requirements are immutable. Preserve negative controls and
subgroups even if the headline is positive.

## Required assertions before the first scientific run

1. immutable README, preregistration, adversarial-review, and semantic-config
   hashes plus ancestor commit;
2. exact model revision, token IDs, lens hash/rank, band, dtype, backend, and
   batch-one contracts;
3. 96 unique exact-depth fingerprints, zero overlap with four parents, and no
   benchmark content;
4. reachable integer seam gates and tested deterministic shuffle invariants;
5. model smoke records no task correctness, chosen alias, or trace text; and
6. later stages remain fail-closed.

## Additional assertions before causal confirmation

1. untouched slot-seam replication and hash-locked raw rows;
2. held-out-task G1, margin/identity advantages, and within-group shuffled null;
3. frozen transforms, beta, score quantile, direct comparator, and pair rules;
4. at least 16 task-level pairs before outcome summaries;
5. 100% post-bf16 numeric validity for both random arms;
6. exact thought-position and causal-suffix invariance;
7. complete arm/item rows and task-level bootstrap; and
8. explicit oracle/constrained/non-capability wording in every artifact.
