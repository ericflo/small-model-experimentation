# Pre-Mechanics Adversarial Audit

Completed after `CONTROL_CALIBRATION_PASS` and before any mechanics probability
is retained. This audit cannot modify scientific gates or authorize continuation.

## Calibration verdict

Accept the full outcome-blind numeric firewall:

- exactly 880/880 unique numeric rows pass;
- non-J A: 440 rows, maximum norm error `9.9968e-6`, maximum span `0.0099951`;
- non-J B: 440 rows, maximum norm error `9.9789e-6`, maximum span `0.0099942`;
- all five layers contain 176 rows, and every layer passes;
- at most three exact lattice-pair steps were needed, all at layer 8;
- all 2,240/2,240 unique intervention rows pass exact-once/finite rules;
- all 440 donor-J, additive-J, wrong-J, and logit-lens deltas are nonzero;
- all 40 mean-donor-J deltas are nonzero;
- all 88 full-donor cases have a nonzero layer-4 overwrite; 352 later-layer
  full-donor deltas are exactly zero because the layer-4 full state already
  carries the clean donor trajectory forward, matching the parent mechanism;
- all four exact 512-token prefix ID lists are locked, with task zero byte-equal
  to model smoke; and
- causal difference is zero, donor tensors are immutable, and all token/position/
  length/config/model/lens/boundary contracts pass.

No logits, probabilities, task correctness, first operations, hidden examples,
target pipelines, or confirmation fields were loaded or written.

## 1. Maxima sit close to numeric thresholds

That is expected for exact bf16 geometry and is not rounded. Stored full-
precision values remain below `1e-5` and `0.01`; mechanics must reproduce every
numeric/intervention row exactly before probability files can be written. No
tolerance change or favorable rounding is allowed.

## 2. Full donor has many zero layer deltas

Every full-donor case has nonzero layer 4 and exact-once hooks at layers 4--8.
After the full state is replaced at layer 4, the recipient follows the donor so
later desired states are already reached. Treating those causal zeros as absent
hooks would incorrectly reject the replicated positive's characteristic.

## 3. Four tasks do not make 44 independent capability units

Mechanics is a hard supplied-target write gate, not a population estimate. The
task remains the statistical unit in later qualification/confirmation. No
confidence interval or capability claim is computed from 44 donor rows.

## 4. Direct identity could pass by copying

Direct is diagnostic only. Continuation opens only if donor J also computes the
prompt-local operation result and maps it to a task-randomized label with the
frozen breadth, wrong-donor, non-J, probability-lift, parse, and full-donor
gates.

## 5. Consequence could still exploit fixed token semantics

Alias-to-operation meaning rotates across tasks; result-to-label meaning rotates
independently. The target expectation is computed only from those public prompt
mappings. Fixed `cat=reverse` or a fixed result label cannot pass breadth.

## 6. Wrong donor might merely disrupt source

Wrong donors are a bijective non-source derangement. Mechanics requires the
wrong donor's own computed label >=50% and registered target <=15%, not merely a
change away from source.

## 7. Additive J can no longer be treated as a negative control

It receives its own frozen `ADDITIVE_ANCHOR_TRANSPORT` decision. Passing would
show explicit anchoring, not donor replacement, was the missing ingredient.

## 8. Mean donor/logit arms can expose simpler common shifts

Both are retained for every target. Neither may be omitted from stored outcomes
or later capability comparison. They cannot substitute for the two live non-J
or wrong-donor specificity controls.

## 9. Numeric outcomes could differ when logits are retained

Mechanics reruns the same full forwards with scoring and demands byte-equality
of sorted numeric and intervention rows to calibration. If scoring changes any
bf16 delta/candidate/lattice choice, it writes only
`mechanics_invalid_control.json` with no probabilities and stops.

## 10. Prefix or prompt could silently regenerate

Mechanics never samples. It loads the four committed token-ID lists, reconstructs
the public prompt, verifies prompt length and prefix SHA, and uses those exact
IDs. Any mismatch fails before outcome retention.

## 11. Mechanics might accidentally load gold

The sole task path is `mechanics_public.jsonl`, with exact schema
`task_id, visible, alias_to_operation, source_alias,
result_label_by_operation`. Summary flags for correct alias, first operation,
hidden data, and pipeline must remain false.

## 12. Constrained selection can hide invalid full outputs

Every row records both the constrained distribution and full-vocabulary top.
Global full-top parse must be >=95%. Full registered mass is retained; a
constrained-only success with invalid full output cannot pass.

## 13. Row duplication can inflate rates

Pure evaluation requires exactly `4*11*2*10 = 880` unique
`task,target,probe,arm` identities. Numeric controls require exactly 880 unique
rows. J success breadth spans target aliases, randomized labels, and all tasks.

## 14. Outcome write could be partially committed

All rows stay in memory through the numeric firewall. Writes use temporary files
and atomic replace. Any numeric mismatch/failure leaves no outcome/probability
artifact.

## 15. Decision routing must be fail-specific

Frozen ladder: invalid control; unreachable text probe; no full-state transport;
no direct J transport; direct-only J; or complete consequence transport. Only
the last opens continuation. Additive decision is orthogonal and explicit.

## Locked artifacts required before mechanics

After commit and push, `mechanics_boundary` must hash:

- passing model-smoke summary and exact prefix file;
- full calibration summary;
- all 880 numeric rows;
- all 88 position contracts;
- all four prefix token-ID rows; and
- all 2,240 intervention rows.

The boundary commit must be an ancestor and local/committed hashes must agree.

## Authorization

After these artifacts and this audit are pushed and the mechanics boundary is
separately committed/pushed, authorize one label-free mechanics run. No
qualification, confirmation, free continuation, or correctness load is
authorized unless the automatic decision is
`NATIVE_ANCHOR_J_CONSEQUENCE_TRANSPORT`.
