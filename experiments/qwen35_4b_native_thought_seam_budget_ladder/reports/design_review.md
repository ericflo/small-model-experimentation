# Adversarial Design Review: Native-Thought Seam Budget Ladder

Completed before any model call. CPU-only task construction, exhaustive DSL
enumeration, and gate arithmetic were allowed and produced no model outcome.

## Verdict

Proceed only as a two-stage interface calibration. The design is acceptable
because it creates a fresh result boundary, separates cap selection from
confirmation, preserves autonomous close-and-commit behavior, and cannot turn a
setup pass into a J-space or capability claim.

## 1. This is merely changing a failed parent's cap after looking

Raising 160 inside the parent would overwrite an observed stop gate.

**Hardening:** this is a new experiment with fresh task fingerprints and seeds.
The 160-token result remains terminal and unchanged.

## 2. Prompt tuning could masquerade as a budget repair

A shorter or more imperative prompt might produce closure without more natural
reasoning budget.

**Hardening:** inherit the parent's prompt function, system message, alias map,
task depth, and output grammar exactly. Only the preregistered cap/cache protocol
changes, and that domain change is reported.

## 3. A ladder over independent samples confounds cap and luck

Different stochastic traces at each rung could select a lucky budget.

**Hardening:** generate each selection trace once to 1024 and classify lower
rungs by the close token's exact generation step. Treat cells as paired nested
right-censoring, never independent samples.

## 4. Nested reuse inflates the sample size

Forty-eight traces viewed at three caps are still 48 traces, not 144.

**Hardening:** every summary labels the paired construction; no between-rung
p-value, pooled denominator, or pseudo-replication is allowed.

## 5. An off-by-one close rule can choose the wrong cap

Counting thought tokens rather than generation steps is ambiguous at the cap.

**Hardening:** freeze `close_step` as one-indexed among generated tokens and
declare a close reachable exactly when `close_step <= cap`. Unit-test the
boundary at cap and cap+1.

## 6. Selecting on correctness overfits the budget

Headroom and mixed-task gates use gold labels, so the selected cap is partly
label-selected.

**Hardening:** treat selection as setup-only, open a disjoint task/seed
confirmation at the selected cap, and make no accuracy comparison across caps.
Gold protects later value feasibility; it is not a deployable signal.

## 7. A larger cap can rescue failed confirmation

Opening another rung after seeing a miss would make confirmation another tuning
set.

**Hardening:** confirmation runs only the smallest selected cap. Any miss is
terminal `SEAM_NOT_REPLICATED`; a different cap requires a new experiment.

## 8. Repeatedly appending rungs guarantees eventual success

A 2048 or unbounded fallback would turn the ladder into optional stopping.

**Hardening:** the only rungs are 256/512/1024. No selection pass means terminal
`NO_BUDGET_SELECTED` and unopened confirmation.

## 9. Injecting `</think>` creates the answer seam being measured

Prior work showed teacher-forced close states can look informative but remain
undeployable.

**Hardening:** never inject a close token. A cap-bound trace has no answer and
cannot parse, score, or become usable.

## 10. EOS-before-close is silently treated as a natural close

The model may terminate malformed output without the close delimiter.

**Hardening:** record `eos_before_close` separately and count it as no natural
close, no parse, and no usable trace.

## 11. A generous answer budget hides an answer-format failure

Unlimited post-close decoding could eventually stumble into `First:`.

**Hardening:** freeze 16 answer tokens, stop earlier on EOS, and report answer-cap
contact. The grammar is unchanged from the parent.

## 12. Parse rate uses a favorable denominator

Conditional-on-close parsing can look high even if very few traces close.

**Hardening:** require close >=80%, report parse over all traces as well as
conditional parse, and require absolute usable counts.

## 13. Cap-bound rows manufacture mixed-success tasks

Calling every non-close a wrong answer would create artificial correct/incorrect
variation for the later value study.

**Hardening:** mixed tasks use only natural-close, parseable, minimum-length
usable traces. Cap contacts are excluded rather than labeled incorrect.

## 14. Very short thoughts pass closure but offer no prefix interior

A model could immediately close and answer, leaving no meaningful 0.33/0.67
thought checkpoints.

**Hardening:** usable traces require at least 16 thought tokens and absolute
usable-count gates of 32 selection / 48 confirmation.

## 15. Accuracy saturation leaves no value contrast

If every usable trace is correct or wrong, a seam exists but the next value
experiment remains infeasible.

**Hardening:** require usable success in [0.05,0.95] and mixed usable tasks
six/eight. These are feasibility gates, not capability endpoints.

## 16. The latent first operation is not behaviorally identifiable

Multiple pipelines can match visible I/O while starting with different
operations; `negate` is a known example.

**Hardening:** enumerate every concrete depth-two pipeline for every task and
require a singleton matching first-operation type. Exclude `negate` targets.

## 17. Fresh rows overlap the parents

Reusing observed tasks could make confirmation easier and undermine a clean
successor.

**Hardening:** hash visible+hidden task structure; require 40/40 uniqueness and
zero overlap with both Jacobian parents before any model load.

## 18. Alias frequency or tokenization changes the problem

Multi-token or duplicated aliases alter answer difficulty and future lens use.

**Hardening:** inherit the fixed one-to-one mapping, balance target types to
within one, and have model smoke verify every leading-space alias is one unique
token.

## 19. Cached decoding is silently not cached

`use_cache=True` can be ignored or broken by a model/backend path, changing cost
and the exact state contract.

**Hardening:** audit every top-level forward input length: one full prefill then
only single-token calls. Any failing scientific row invalidates the whole stage.

## 20. Cache mode differs from the cache-free parent

Numerical differences could be misreported as a replication of the 160 result.

**Hardening:** do not pool or statistically compare parent rows. The parent is
lineage only; all new ladder and confirmation cells use one Transformers cached
backend. The change is intentional because the next intervention must scale.

## 21. Batch or backend mixing changes samples

Earlier work observed material batch/backend differences despite equal seeds.

**Hardening:** unpadded batch one for every arm; no vLLM, no batch-two, no
cache-free comparator, and explicit sampling values throughout.

## 22. Model or special-token drift invalidates the seam

A floating revision or changed chat template could move close behavior.

**Hardening:** pin model revision and verify architecture plus exact open/close
IDs in a non-result-bearing model smoke after the design commit.

## 23. Prompt plus ladder exceeds the declared context envelope

Truncating a prompt or exceeding context would change tasks selectively.

**Hardening:** reject prompts above 768 and prove the worst case fits 2048 before
loading the model. Never truncate task prompts.

## 24. A hard gate is mathematically unreachable

Mixed-task and count requirements could contradict rate thresholds.

**Hardening:** freeze a machine-readable reachability receipt for both stages.
It verifies the minimum counts and a feasible 0.5 success construction.

## 25. Marginal close rate is mistaken for a replicated interface

At 72 traces, an 80% point estimate remains uncertain.

**Hardening:** confirmation additionally requires the 95% Wilson lower bound for
natural close >=0.70 and retains absolute usable/mixed counts.

## 26. Selection seeds leak into confirmation

Shared randomness could make a cap appear stable.

**Hardening:** separate base seeds, disjoint task fingerprints, and a hash-frozen
selection artifact before confirmation model loading.

## 27. Human inspection of thoughts tunes the rule

Qualitative examples could motivate a favorable parsing or cap exception.

**Hardening:** buffer all rows, write only after stage completion, and compute the
decision automatically before any decoded thought inspection. Parsing and gates
are code-frozen.

## 28. An interrupted run selects from complete-looking partial data

Incremental output files could be mistaken for a full stage.

**Hardening:** rows remain in memory until every task/trace completes; only a
complete file receives a hash and summary. Progress lines contain counts only.

## 29. A seam pass is promoted to a J-space result

No activations are read, so a positive closure result cannot validate value,
certainty, or transport.

**Hardening:** all terminal narratives label the result setup-only. Only a new
experiment may fit/patch J coordinates.

## 30. A seam pass is promoted to capability gain

The study has no matched-compute sampling control and uses gold for feasibility.

**Hardening:** no capability endpoint or claim ID. A future non-oracle method
must still beat frozen, strongest controls, and matched-compute sampling on
untouched procedural tasks.

## 31. Historical-token activation invariance is assumed again

The parent smoke found suffix-length-dependent bf16 activation drift of 0.0625.

**Hardening:** this seam study makes no activation-invariance claim. Its licensed
successor must replay each exact prefix as its own sequence and intervene at the
live prefix endpoint.

## 32. One fixed random control is reused across sequence lengths

Even live endpoint interventions can realize different post-bf16 geometry as
prefix lengths vary.

**Hardening:** the next experiment must construct and audit random/non-J controls
separately for every live prefix and sequence length. This requirement is part
of the positive branch contract, not deferred discretion.

## Required assertions before budget selection

1. published immutable design commit and matching README/prereg hashes;
2. exact model revision, architecture, token IDs, alias tokenization, and cache
   input-length contract;
3. 40 unique fresh fingerprints, zero parent overlap, exhaustive visible
   identifiability, and balanced target support;
4. exact rungs, seeds, prompt/context envelope, generation parameters, and
   answer allowance;
5. reachable selection and confirmation gates; and
6. explicit setup-only interpretation with no claim allocation.

## Required assertions before confirmation

1. complete 48-row selection file and matching hash;
2. exactly one smallest selected cap from the frozen gates;
3. untouched confirmation tasks and seed base;
4. no alternative confirmation cap; and
5. full 72-row completion before a terminal decision.
