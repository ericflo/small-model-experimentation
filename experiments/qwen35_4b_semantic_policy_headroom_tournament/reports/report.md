# Report: semantic-policy headroom tournament

## Verdict

`INSTRUMENT_FAIL`. The exact transaction-trained parent was evaluated on both
frozen blocks, but answer-cap contacts reached 12.08% and 12.67% of turns,
above the preregistered 5% ceiling. No axis independently met the replicated
headroom rule. Training, checkpoint creation, and Menagerie were never
authorized.

## What ran

The immutable design at `391dadc1` and lock receipt at `9cc90db4` evaluated the
unchanged Qwen3.5-4B transaction checkpoint on two content-disjoint blocks.
Each block contained 36 fresh procedural repositories and two controlled
states per repository, for 72 trajectories. Nine inferred-contract cells
crossed negative-quantity, non-integer-quantity, and blank-resource policies
with bundle, record, and tuple representations. Three explicit-contract cells
served as instrument controls.

The model received one greedy six-turn trajectory per case with 512 thinking
and 512 answer tokens per turn. The host alone executed hidden tests. No
benchmark source, item, transcript, or result entered the run.

## Primary qualification

An axis required 15–80% failed-test success in both blocks and at least two of
three shape-family rates inside that band in each block.

| Axis | A failed-test success | A shapes in band | B failed-test success | B shapes in band | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: |
| negative quantity | 9/9 | 0/3 | 9/9 | 0/3 | no |
| non-integer quantity | 9/9 | 0/3 | 9/9 | 0/3 | no |
| blank resource | 8/9 | 1/3 | 7/9 | 1/3 | no |

The shape-level failed-test rates were:

| Axis / representation | Headroom A | Headroom B |
| --- | ---: | ---: |
| negative / bundle | 3/3 | 3/3 |
| negative / record | 3/3 | 3/3 |
| negative / tuple | 3/3 | 3/3 |
| non-integer / bundle | 3/3 | 3/3 |
| non-integer / record | 3/3 | 3/3 |
| non-integer / tuple | 3/3 | 3/3 |
| blank / bundle | 3/3 | 3/3 |
| blank / record | 2/3 | 3/3 |
| blank / tuple | 3/3 | 1/3 |

Negative and non-integer recovery were descriptively saturated. Blank-resource
recovery had nontrivial misses, but its supported representation changed from
record in A to tuple in B. This is not the replicated two-shape substrate the
frozen rule required.

## Global controls and interface

| Metric | Frozen gate | Headroom A | Headroom B | Result |
| --- | ---: | ---: | ---: | ---: |
| explicit-control failed-test success | ≥85% | 9/9 | 8/9 | pass |
| invalid actions / turn | ≤5% | 6/356 (1.69%) | 5/363 (1.38%) | pass |
| answer-cap contacts / turn | ≤5% | 43/356 (12.08%) | 46/363 (12.67%) | **fail** |
| content manifests differ | required | yes | yes | pass |
| eligible axes | ≥1 | 0 | 0 | fail |

Because the preregistration gives instrument failures precedence, the terminal
label is `INSTRUMENT_FAIL`, not the cleaner `NO_QUALIFIED_AXIS` label.

## Answer-cap forensics

Across both blocks, 89/719 turns reached the 512-token answer allowance. This
was usually not an unusable truncation: 78/89 capped turns contained a complete
parseable tool object, and 77/78 then continued with post-call run-on. Eleven
capped turns were invalid. Capped answers comprised 41 inspections, 35
patches, 11 invalid calls, one verification, and one commit.

At case level, 54/66 cap-contact trajectories succeeded versus 78/78 without a
cap contact. Every one of the 12 failures contacted the cap. Yet all targeted
local transitions still occurred among cap-contact cases: failed-test changed-
patch-within-two was 38/38 and rejected-patch valid-changed-within-two was
28/28. Failures instead exhausted six turns without a successful workspace or
submission.

This supports two bounded conclusions. First, the cap gate is partly measuring
valid-call run-on rather than parser failure. Second, cap contact remains
associated with every end-to-end miss, so it cannot be waived after seeing the
data. A fresh successor should use a parse-aware payload allowance and preserve
both raw cap and validity diagnostics.

## Earlier-proposal forensics

The raw trajectories expose a cleaner state contrast than the terminal score.
All 72 failed-test cases reached a fully correct executable patch at some point;
the four failed-test endpoint misses were destructive regressions after an
earlier correct workspace. First-patch correctness was already 29/36 in A and
30/36 in B.

The rejected-patch condition provided no test output. Across its 54 inferred-
contract cases, the first changed patch was fully correct 0 times. Across all
rejected cases, only 6/36 first patches per block were fully correct, all from
the explicit negative/non-integer controls. None of the 72 rejected
trajectories inspected visible tests before first patching; the usual action
was another source read. Nevertheless, 64/72 eventually reached a correct
workspace after later verifier evidence.

This localizes the capability gap to specification acquisition and evidence-
conditioned proposal formation. It is not a missing ability to write the
correct patch once the public failure states the distinction, and it is not a
deleted recovery transition.

## Interpretation

Direct failed-test evidence is no longer the best place to seek this
capability. The transaction-trained parent repaired negative and non-integer
semantic conflicts in every inferred-contract cell, while blank-resource
errors did not replicate across shapes. Training on these states would either
target already-mastered behavior or chase an unstable representation-specific
residual.

The unresolved seam lies earlier: deciding when public evidence must be
inspected and binding that evidence to an initial proposal before the verifier
states the discrepancy. The strongest next strategy is counterfactual evidence
binding. Hold issue and source nearly constant, flip visible tests between
opposed but equally plausible policies, and require the model's patch to flip
with the evidence. Train complete, transition-balanced trajectories only after
the exact parent shows replicated initial-proposal headroom on fresh families.
Compare against matched extra replay and equal-compute sampling, then require
locality, conditional transition retention, family-held-out transfer, and only
then paired Menagerie.

## Exposure ledger

- Training: not run.
- New checkpoint: none.
- Menagerie: not invoked; no benchmark seed was consumed.
- Claim ledger: unchanged.
- Compact qualification and result receipts: committed.
- Detailed trajectories: external and checksummed in the artifact manifest.
