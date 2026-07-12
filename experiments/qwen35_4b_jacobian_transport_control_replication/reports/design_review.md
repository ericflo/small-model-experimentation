# Adversarial Design Review

Completed before implementation and before any model call in this experiment.

## Verdict

Proceed as a strict replication. The parent point estimate is exceptionally
strong, which makes anti-confirmation-bias controls more important, not less.

## 1. Post-result tuning of the successful mechanism

Changing layers, lens prompts, dictionary size, alpha, or task grammar after a
48/48 result would turn replication into optimization.

**Hardening:** copy the parent lens byte-for-byte, fix band 4–8 and alpha one,
and use fresh mappings. No J outcome can select anything.

## 2. Quietly repairing the one failed parent row

Rerunning only `confirm-0046` or expanding its random search after seeing the
answer would not be independent evidence.

**Hardening:** parent remains terminal `INVALID_CONTROL`. This experiment uses
new calibration and confirmation IDs/seeds and its own immutable boundary.

## 3. Quantization makes pre-cast orthogonality misleading

The parent requested vectors were orthogonal to 2.34e-7, but realized bf16
deltas reached 5.7% J-span projection.

**Hardening:** gate both norm and projection on the actual post-addition delta in
the live hook. Requested geometry is recorded but cannot substitute.

## 4. An impossible orthogonality bar can censor the experiment

Exact zero projection is generally unattainable on a bf16 activation lattice.

**Hardening:** freeze a 1% norm-fraction ceiling before model calls. It limits
span energy to 1e-4. Retain wrong-donor J as a same-span semantic control, which
tests label specificity without relying on complement geometry.

## 5. Control optimizer secretly selects by model output

Trying many random candidates could choose the one least likely to change the
answer.

**Hardening:** candidate selection sees only current residual, J dictionary,
target norm, norm error, projection fraction, and candidate index. Calibration
logits are discarded and absent from artifacts. Confirmation uses the first
numeric passing candidate under fixed seeds.

## 6. Calibration leaks confirmation or mechanism outcomes

A calibration split could become a layer/algorithm tuning set.

**Hardening:** optimizer hyperparameters and thresholds are already frozen.
Calibration is pass/fail numeric feasibility only; no logits or answer flags are
stored. Failure stops rather than invites amendment.

## 7. One lucky random control

A single random direction might be unusually inert.

**Hardening:** require two independent controls on every item and compare J to
the worse target rate. Both receive paired bootstrap tests.

## 8. Random complement is not the strongest semantic control

Orthogonal noise tests magnitude specificity but not whether another J-space
state causes its own semantic consequence.

**Hardening:** wrong-donor J is required to increase its own mapped digit and not
the target. Pair-only J and logit lens distinguish sparse semantic coordinates
from broad state replacement and ordinary token directions.

## 9. Target digit injection

An output-margin gradient would trivially bias the desired digit.

**Hardening:** static intervention code may reference concept lens directions
and donor activations only. No digit direction/gradient is implemented. Result
receipts assert this boundary.

## 10. Fresh mappings accidentally overlap parent data

Reusing exact table/source/target tuples would not be independent replication.

**Hardening:** compare canonical SHA-256 fingerprints against both parent data
directories during CPU smoke. Any overlap is fatal.

## 11. Qwen hybrid batching/cache artifacts

The parent measured equal-length batch-two logit differences up to 0.21875.

**Hardening:** every model call remains unpadded batch one with cache disabled.
Direct/consequence antecedent activations must be exactly invariant.

## 12. Multiple arms enable a favorable substitute endpoint

Full donor or pair J could look good if primary all-24 J fails.

**Hardening:** only all-24 J mapped-digit transport versus the worse of two valid
random controls is primary. Other arms explain but cannot rescue failure.

## 13. Oracle mechanism is mislabeled capability

Target donor identity is supplied.

**Hardening:** even a replication uses only the label
`REPLICATED_J_TRANSPORT`, never “capability gain.” Native thoughts and a
non-oracle controller require a separate experiment and matched-sampling test.

## 14. Claim pressure from a perfect point estimate

The current ledger re-grade blocks new claims, and the parent is invalid.

**Hardening:** reserve no claim ID. Preserve every control failure and update
program strategy only after the frozen decision.

## Required assertions

1. exact model/revision and lens hash;
2. fixed band/alpha and fresh disjoint fingerprints;
3. one-token/position/causal contracts;
4. cache-free unpadded batch one;
5. calibration artifacts contain no logits/outcomes;
6. both random arms pass realized norm and projection at every layer;
7. numeric-only first-passing candidate selection;
8. no target digit direction or gradient;
9. confirmation inaccessible before calibration pass;
10. frozen endpoint and paired bootstrap rules.

## Post-smoke implementation audit (before calibration)

Three outcome-blind model-smoke attempts localized a discrete bf16 failure at
layer 8. No calibration or confirmation outcome was opened. Adding an exact
lattice repair after observing numeric smoke geometry creates a post-design
implementation risk, so it is constrained as follows:

- it runs only when the already-frozen continuous optimizer fails;
- it scores exact pairs of one-ULP bf16 coordinate moves against only the two
  frozen numeric constraints, never logits, labels, or answer identity;
- it preserves the original candidate identity, seeds, 32 draws, damping,
  continuous correction/search budgets, layer band, target norm, and thresholds;
- it reuses the frozen 512 bound and stops at the first passing lattice state or
  when no exact pair improves the joint objective;
- every applied pair count and final realized geometry is recorded; and
- calibration still requires 480/480 passing layer deltas and remains a fatal
  firewall before untouched confirmation.

This repair can only make the random control more stringently orthogonal to the
J span. It cannot select an inert model outcome. The adversarial verdict remains
proceed-to-calibration only if a fresh rerun of the original 20-row model smoke
passes in full.

## Pre-confirmation implementation audit

Completed after the committed 480/480 numeric calibration pass and before any
confirmation outcome was opened.

1. **Calibration could be overwritten or selectively rerun.** Confirmation is
   locked to commit `2bd6376c28283546a111e54ec2dd2e92a0fd6a64` and exact SHA-256
   hashes for both committed calibration files. The runner also reconstructs
   all 480 expected identities, enforces the outcome-blind schema, and recomputes
   both maxima.
2. **A missing arm could silently improve the endpoint.** Pure decision logic
   requires exactly 48 items x 2 prompt kinds x 8 registered arms = 768 unique
   outcome rows, plus exactly 960 unique random-control layer rows
   (`48 x 2 x 2 x 5`).
3. **The easier random draw could be reported.** Primary effect uses the larger
   target rate of `random_a` and `random_b`; paired 10,000-resample lower bounds
   must independently exceed zero against both arms.
4. **Wrong-donor J could merely act like target J.** The frozen gate separately
   requires target-vs-wrong specificity and an increase in the wrong donor's own
   mapped digit. Unit tests make a nonspecific wrong arm fail.
5. **Controls could drift between calibration and confirmation.** Both stages
   call the same quantization-aware constructor with the same frozen 32 draws,
   correction/search budgets, exact lattice solver, and post-bf16 checks. Fresh
   confirmation IDs produce disjoint fixed seeds.
6. **Raw logits or a target-digit gradient could construct the intervention.**
   Interventions use only target/wrong activations, the frozen concept lens, or
   the preregistered concept unembedding control. Serialized rows retain only
   top-token flags and two prespecified scalar margins; no digit direction or
   gradient exists.
7. **Backend or batching could recreate the known Qwen discrepancy.** Every arm
   is an unpadded, cache-free Transformers bf16 SDPA batch-one full forward. No
   vLLM/HF mixing or batch-two scoring is implemented.
8. **Implementation could be changed after looking.** The runner, unlock hashes,
   arm-completeness tests, decision tests, and this audit must be committed and
   pushed before the one confirmation run.

Verdict: proceed once the implementation commit is on `main` and local/remote
release gates are green. A passing result remains oracle mechanism evidence,
not a deployable capability claim.
