# Qwen3.5-4B Verified Macro Invention Report

## Summary

The first vLLM smoke failed its interface gate. Two task-independent, plan-given repairs then
separated answer-boundary failure from semantic alias failure. No-think made all 16 final-gate
outputs valid, macro-using, and non-truncated, but only 1/4 records reproduced the supplied plan
exactly. The registered stop rule fired before any fresh induction prompt. This experiment is
closed with the verified-macro hypothesis unresolved and no claim-ledger update.

## Research Program Fit

The question remains attached to `operator_and_skill_inventories`, with secondary relevance to
`structured_execution_and_compilers` and `benchmark_generalization`. Existing work studies fixed
operator banks; this experiment still targets the unresolved question of whether train-derived,
exact composite entries can change held-out proposal coverage beyond sample-more and matched
random entries.

## Method

The procedural list DSL, construction corpus, exact interpreter, true-depth-5 full tasks, mined and
placebo libraries, visible-only selector, and vLLM-only inference path were frozen before model
generation. Smoke v1 used a disjoint 12-task set. The base arm sampled K=24 and each macro arm K=12;
the registered interface gate pooled the matched base/design comparison and separately checked
each arm.

After v1, amendment 1 inserted a non-scored train-only plan-given gate before any new induction
task. Four prompts each supplied the verified primitive plan and asked for an optimal rendering
with designed aliases; each received four samples. This isolates whether the model can call the
surface when no program induction is required.

## Results

| Smoke-v1 gate metric | Result |
| --- | ---: |
| Pooled base/designed parse | 0.5972 |
| Base parse | 0.6111 |
| Designed-ceiling parse | 0.5694 |
| Pooled base/designed answer truncation | 0.40046 |
| Designed valid macro-using candidates | 0 |
| Designed oracle coverage | 0/12 = 0.0000 |
| Base oracle coverage | 1/12 = 0.0833 |

All 1,440 solver samples force-closed and 607 answer stages truncated across all generated arms.
The only base oracle solve was in the no-reuse split. The whole-answer proposal parser accepted
0/16 outputs. An exploratory line-local audit after failure found 18 behaviorally unique,
train-supported candidates, but that reparse is not part of the v1 result and cannot backfill a
Qwen-ranked arm.

Interface attempt 2 then produced:

| Plan-given interface metric | Result |
| --- | ---: |
| Records / samples | 4 / 16 |
| Successful records | 2 (`00`, `02`) |
| Strictly valid samples | 4 |
| Macro-using samples | 4 |
| Answer truncation | 12/16 = 0.75 |

It failed the required at-least-3/4 successful records and below-0.05 truncation conjunction. The
pipeline stopped before generating a fresh smoke prompt.

Interface attempt 3 then changed only the plan-given gate's thinking channel:

| No-think interface metric | Result |
| --- | ---: |
| Records / samples | 4 / 16 |
| Successful records | 1 (`00`) |
| Strictly valid samples | 16 |
| Macro-using samples | 16 |
| Answer truncation | 0/16 = 0.0000 |

Attempt 3 passed the truncation gate but failed the unchanged at-least-3/4 record requirement.
The committed raw-row audit found three exact samples, all on record `00`. Each of the 13 failed
samples used multiple macros and expanded beyond the five-primitive limit (depth 6--10); 10/13
included the correct designated alias but appended unrelated aliases, while 3/13 omitted it. No
fresh induction or full prompt was ever generated.

## Controls

No full control contrast was run. Construction/full overlap remained zero, full tasks and hidden
outputs were not used for repair, and every model call used the same experiment-local vLLM
backend. Smoke v1 is preserved rather than overwritten. Amendment 1 uses a new seed and ids,
matched K=12 base/designed arms, identical surface-first instructions, and a train-only plan-given
mechanical probe.

The failed plan-given attempts are also preserved separately. Amendment 2 changed only the final
gate's thinking mode; it did not alter prompts, targets, parser, thresholds, fresh smoke, proposal
ranking, or the full protocol.

Attempt 3 confirms why exact verification was non-negotiable: parse and raw macro use were both
16/16, yet exact record success was 1/4. Neither plan-given attempt is pooled with another or with
scientific-task evidence.

## Oracle Versus Deployable Evidence

Neither deployable nor oracle abstraction evidence exists yet. Designed-ceiling oracle coverage
was zero because the interface produced no valid macro-using candidate; this prevents attribution
to library quality. The base arm's single no-reuse solve is too small and on the wrong slice to
support a scientific comparison.

## Interpretation

Smoke v1 diagnosed a budget-and-surface failure. Attempt 2 established that designed aliases were
sometimes callable but budgeted thinking spilled into the answer. Attempt 3 showed that removing
thinking fixed syntax and termination completely without fixing exact alias substitution. A model
can call the intended leading abstraction, then over-compress unrelated suffix operations with
plausible but behaviorally wrong aliases.

The durable lesson is methodological: strict syntax and macro-use rates are insufficient. A
composite call is valid only when literal expansion preserves the intended plan. The scientific
macro-invention question itself remains unresolved because its fresh task was never attempted.

## Next Experiments

Do not add amendment 4. A further attempt would need a materially different exact-call interface,
such as a constrained representation or separately verified rewrite procedure, and is therefore a
new experiment. It should preserve the same one-model/vLLM boundary, establish plan-given expansion
fidelity before induction, and retain matched-compute sampling and no-reuse controls.

## Artifact Manifest

The adjacent `artifact_manifest.yaml` records the vLLM-only reproduction commands and the complete
versioned smoke-v1, interface-v2, and interface-v3 archives. There are no external model or adapter
artifacts.
