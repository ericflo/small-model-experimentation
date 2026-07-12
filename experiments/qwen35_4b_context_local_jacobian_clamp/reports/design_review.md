# Adversarial Design Review

Completed before implementation and before any result-bearing GPU call.

## Verdict

**Proceed only with the registered positive-control ladder.** The corrected
design can distinguish causal-site failure, donor-state transport, direct token
writing, and J-coordinate transport. It cannot establish deployable capability:
all counterfactual target donors are oracle interventions.

## Threat 1: direct answer-gradient injection

The most obvious “context-local Jacobian” is the gradient of the desired target
digit logit. Adding it would almost tautologically raise that digit and would
encode the answer in the intervention.

**Hardening:** no digit covector, digit unembedding vector, or consequence
target-minus-source gradient may fit, select, scale, or gate the J clamp. The
dictionary is pulled back only from future direct concept-token reports. Output
margin gradients are post-hoc diagnostics after all results are frozen.

## Threat 2: answer-position motor control masquerading as semantics

The parent patched the final prompt position, and its late layer profile was
consistent with imminent token control.

**Hardening:** patch only the selected concept's earlier occurrence in the
shared prefix. No answer-position or generated-token patch is allowed. Require a
separately tokenized digit consequence in addition to direct key repetition.

## Threat 3: repeated swaps undo one another

Swapping source and target independently at several layers can move toward the
target and then swap back. The parent's low band result was therefore not a
clean clamp test.

**Hardening:** capture clean target-donor coordinates before intervention and
set every patched layer to that fixed value. Alpha is fixed at one. The desired
state never depends on the already patched trajectory.

## Threat 4: perturbation magnitude explains apparent specificity

The parent's direct J delta was much larger than its random control, weakening
the J-versus-random contrast.

**Hardening:** first record the actual primary J delta at every item/layer. Then
construct a random vector orthogonal to the complete J dictionary and rescale it
to the exact same norm at that item/layer. Enforce relative error <=1e-5 in code
and invalidate rather than reinterpret failed controls.

## Threat 5: layer and coefficient garden of forking paths

A wide layer/alpha sweep could manufacture a positive late token effect.

**Hardening:** alpha is fixed at one. A disjoint selection split chooses one of
six registered five-layer bands using only the full-activation donor positive
control. J results are not run on selection. Confirmation is untouched.

## Threat 6: the token position is not causally sufficient

Later positions may already have consumed the literal token within the same
layer, or the model may use another occurrence in the table. A J null at a bad
site says little about the dictionary.

**Hardening:** the full target-donor activation clamp is mandatory and selects
the band. If it cannot redirect both direct and consequence outputs, label
`NO_CAUSAL_SITE`; do not interpret the J arm. The table intentionally retains
the earlier mapping occurrence so the test asks whether the selected-token
state, rather than deletion of all source evidence, controls reference.

## Threat 7: full donor is itself an oracle rewrite

Replacing an activation from an otherwise identical target-key prompt can carry
many features besides concept identity.

**Hardening:** treat it solely as a causal-site ceiling. The primary claim, if
any, belongs to the restricted J coordinate clamp. Include pair-only,
logit-lens, wrong-donor, and orthogonal controls to characterize what the
dictionary captures. Never call donor success a capability gain.

## Threat 8: wrong-donor corruption looks like target success

A large intervention could suppress the source, cause arbitrary output changes,
and occasionally hit the target.

**Hardening:** every item has distinct source, target, and wrong concepts/digits.
Measure both the registered target rate and the wrong donor's own expected digit.
Require target advantage over wrong-donor-to-target and a rise in wrong-donor's
own output.

## Threat 9: the model ignores the fresh lookup table

If mappings repeat or digits correlate with concepts, success may come from
pretraining rather than prompt-local reasoning.

**Hardening:** regenerate one-to-one random mappings per item, balance digits,
keep splits disjoint, and require >=0.80 clean consequence accuracy. Store exact
mapping rows and hashes. No benchmark task is read or trained on.

## Threat 10: direct and consequence prompts alter the antecedent state

If a non-causal implementation lets suffix tokens affect the earlier activation,
comparing the two tasks is invalid.

**Hardening:** prompts share a byte-identical prefix through the selected token.
Before results, compare captured selected-position activations across suffixes at
every fitted layer and require numerical equality within the registered dtype
tolerance. Run without cache.

## Threat 11: tokenizer and position mismatch

Leading spaces, table punctuation, or chat-template variation can split tokens
or move the target donor's selected position.

**Hardening:** CPU/model smoke audits all concept and digit token IDs, source and
donor token counts, final-occurrence indices, answer IDs, and shared-prefix
identity. Any mismatch is `INVALID_CONTROL`, not an item silently dropped.

## Threat 12: nonorthogonal or rank-deficient coordinates

Twenty-four pullbacks may be nearly collinear. A pseudoinverse could amplify
noise and make “coordinate setting” a poorly specified large perturbation.

**Hardening:** unit-normalize directions, use a fixed SVD cutoff, record the full
singular spectrum/effective rank/condition number by layer, and require full rank
24. Report realized norms. A failed rank gate stops before causal results.

## Threat 13: full-dictionary setting overclaims sparse J-space

Setting all 24 targeted coordinates is not the same intervention as swapping two
named coordinates, and the dictionary is not a complete residual basis.

**Hardening:** call the primary arm “all-24 targeted J clamp,” not full J-space.
Preserve the non-J remainder. Report the two-coordinate arm separately. Do not
claim the residual has been decomposed into a complete proper basis.

## Threat 14: Qwen hybrid cache, padding, and batching artifacts

Qwen3.5 mixes attention and recurrent layers; cache or left/right padding can
change exact internal computations.

**Hardening:** scientific interventions are cache-free, unpadded, batch-one full
forwards. Equal-length batch equivalence is a preflight only. Record versions,
device, dtype, sequence length, and forward-token counts.

## Threat 15: confirmation leakage through band selection

Using J effectiveness to choose the band and then reporting the same items would
be circular.

**Hardening:** only full-activation donor outcomes on the 24 selection items can
choose a band. Confirmation has 48 independent mappings and is opened once.

## Threat 16: multiple secondary arms dilute the primary result

Pair clamp, logit lens, donor, and gradients provide many possible favorable
comparisons.

**Hardening:** the primary endpoint is frozen: all-24 J target-digit rate versus
baseline and exact norm-matched orthogonal control on confirmation. Other arms
explain the result but cannot substitute for a failed primary gate.

## Threat 17: positive causal transport still does not install capability

The intervention is given the desired target concept, so even perfect transport
is unavailable at deployment.

**Hardening:** use only mechanism labels. A positive result licenses a new
experiment that must learn a non-oracle intervention rule from visible model
state and beat both frozen inference and matched-compute sampling on fresh
held-out tasks. No adapter or benchmark run occurs here.

## Threat 18: stopping rules are ignored after an intriguing partial result

The parent produced an enticing 75% direct effect despite failing the actual
consequence gate.

**Hardening:** full donor failure stops J confirmation; J direct-only success is
terminal `DIRECT_ONLY`; only `J_TRANSPORT` can advance to native thoughts. All
negative rows and failed controls remain in the repository.

## Required implementation assertions

1. model ID and revision exact;
2. no path under `benchmarks/` imported or read;
3. one-token concept/digit contracts;
4. equal source/donor token lengths and selected indices;
5. shared-prefix and causal-activation equality;
6. no cache, padding, or intervention batching;
7. J rank exactly 24 and finite coordinates;
8. J outcomes absent from band-selection code;
9. alpha exactly 1;
10. per-item/layer norm-match relative error <=1e-5;
11. target-digit gradients isolated to diagnostic stage;
12. confirmation cannot run unless the stored donor gate passes.
