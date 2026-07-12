# Qwen3.5-4B Commit-Slot Jacobian Value Transport Report

## Status

Terminal `COMMIT_SLOT_SEAM_FAIL`; all later stages are ineligible.

## Purpose

Natural close and close-only free-form output both failed before J value. This
experiment supplies fixed answer syntax but not identity, constrains the next
choice to public aliases, requires real thought to beat both an immediate slot
and the same thought tokens in shuffled order, and then tests whether a scalar J
value coordinate improves the semantic choice.

## Plumbing

CPU smoke only: 96/96 unique fresh exact-depth tasks, zero overlap with four
parents, no visible depth-one fits, exact lens hash, and reachable gates. The
CPU check caught an initial generator-seed collision before design freeze; the
entire seed block was replaced and the final manifest has zero overlap.

After the immutable design commit, model smoke passed the pinned 32-layer,
2,560-wide revision; five rank-24 lens matrices; 12 distinct leading-space alias
tokens; fixed slot tokenization; finite constrained logits; and native/free-form
cache contracts. Peak allocated GPU memory was 8,515,461,632 bytes. The receipt
contains neither task correctness nor the chosen alias.

The pre-selection implementation audit then verified exact real/shuffled token
multisets, runtime data/config hashes, complete-row buffering, unmasked-logit
diagnostics, control finiteness, and fail-closed later stages. No scientific
outcome informed the audit.

## Selection results

The scientific run completed 48/48 native paths, 144 real slot rows, 144
exact-length shuffled-thought slot rows, 144 matched close-only free-form rows,
and 16 task-level no-thought slots. Every path contacted 1,024; there were no
natural closes or malformed rows. All cache, finite, exact-multiset, cardinality,
and observed-gate-feasibility contracts passed.

| cap | real slot | no-thought | shuffled | real−none | real−shuffle | mixed | pass |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 256 | 10/48 (20.8%) | 4/16 (25.0%) | 11/48 (22.9%) | −4.17pp | −2.08pp | 2 | no |
| 512 | 9/48 (18.8%) | 4/16 (25.0%) | 15/48 (31.2%) | −6.25pp | −12.5pp | 2 | no |
| 1024 | 15/48 (31.2%) | 4/16 (25.0%) | 11/48 (22.9%) | +6.25pp | +8.33pp | 5 | no |

The 1,024 rung passed the frozen 20%--80% accuracy range, +5pp no-thought gap,
+3pp shuffled gap, and 100% finite-row gate. It failed only the requirement for
six tasks with both correct and incorrect traces: observed five. Because the
gate was frozen and mathematically reachable, the one-task miss is terminal; it
does not authorize confirmation or a J-space value fit.

The slot itself repaired most of the prior emission wall. At 1,024, an alias was
already the unmasked full-vocabulary top token on 41/48 rows (85.4%), total alias
probability mass averaged 68.5%, and only 2/15 correct constrained choices
needed the mask because a non-alias was the unmasked top. By contrast, matched
close-only free-form parsing was 2/48 (4.2%), exact success 1/48, and 43/48
answers exhausted 16 tokens. Fixed syntax—not the close token—created answer
mode, but answer mode alone did not make the semantic effect stable enough.

The run sampled 49,152 thought tokens in 1,678.391 seconds. Peak allocated GPU
memory was 8,705,302,016 bytes. Both real and shuffled slot prefills processed
148,395 tokens; the free-form controls generated 2,171 answer tokens.

## Frozen decision

No cap passes, so `selected_cap=null` and the automatic decision is
`COMMIT_SLOT_SEAM_FAIL`. The untouched seam-confirmation, value-fit, numeric
control, and causal-confirmation task sets remain unopened. No J-space
observation or intervention result exists.

## Post-decision diagnostics

Diagnostics cannot rescue the gate. At 1,024, real thought beat shuffled on
seven paired traces and lost on three, a net four. Across 16 task units, the
real-minus-shuffled mean was +8.33pp with a 10,000-resample 95% interval
`[-6.25pp, +27.08pp]`; real-minus-no-thought was +6.25pp with
`[-14.58pp, +27.08pp]`. Both include zero.

The signal was concentrated. Real thought produced four correct `cat` rows
versus zero shuffled and two correct `ocean` rows versus zero, while all six
`tiger` rows were correct under both real and shuffled thought. Eight tasks were
0/3 real, four were 1/3, one was 2/3, and three were 3/3. The slot chose only six
of twelve aliases and selected `tiger` on 25/48 rows; shuffled thought selected
it on 31/48. This concentration explains why a modest pooled gain did not earn
the task-variation gate.

Direct verbalization does not explain the hint. Prefixes containing the correct
alias token were 2/10 correct (20%); prefixes without it were 13/38 (34.2%).
Rows with no alias mention at all were 10/24 correct. Ordered-content benefit,
where present, is not simply copying the written correct alias into the slot.

Three label-free but explicitly post-hoc residual decoders were checked only to
choose a fresh successor. At 1,024, subtracting the same-task no-thought logits
scored 13/48, subtracting exact shuffled-thought logits scored 8/48, and
subtracting a global no-thought log-bias scored 14/48, all below the registered
15/48 slot. Bias subtraction is therefore not the next branch.

## Interpretation

The fixed slot successfully separates **emission syntax** from **semantic
resolution**. The earlier close-only failure was mostly an answer-mode problem;
`First:` makes an alias naturally competitive even without masking. At the
semantic level, 1,024 ordered tokens show a small coherent-content hint, but it
is clustered and task-level uncertainty remains wide. The honest next step is a
fresh, powered, fixed-1,024 replication with more balanced task units and a
task-bootstrap gate—not confirmation of this failed selector, not a relaxed
mixed-task threshold, and not a larger cap yet.

## Boundary

The slot and alias mask are deployment scaffolds. Any positive is constrained
choice evidence, not natural/free-form capability. This result does not license
a value label, J coordinate, donor, controller, or capability claim.

## Artifact Manifest

See `artifact_manifest.yaml`.
