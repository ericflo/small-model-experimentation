# Qwen3.5-4B Partial-Structure Recognition-Guided Search Report

## Summary

**G1 stop: the search state is useful to an exact oracle but unreadable by the frozen model.** A width-4
oracle beam preserved a hidden-solving path on all 12 dedicated depth-5 development tasks while reducing
completed type skeletons from 1,048,576 to 4 per task (262,144x). That made type-prefix viability a promising
*oracle* control signal. It did not make it a usable model signal. Across 7,200 depth-4 calibration children,
thinking P(viable) achieved task-macro, depth-stratified AUROC 0.506 (95% task-bootstrap CI 0.470--0.543) and
live-child recall@4 0.251. No-think P(viable), the strongest control on both metrics, reached 0.556 AUROC and
0.303 recall. Thinking therefore lost by 0.049 AUROC (CI -0.090 to -0.010) and 0.052 recall (CI -0.110 to
0.007); every preregistered gate check failed.

The apparent thinking signal was exactly the confound the design was built to catch: pooled AUROC was 0.557,
but within-task AUROC was chance. On an eight-task canary, replacing each task's visible examples with another
task's examples did not hurt (original 0.450, task-shuffled 0.476; difference -0.025, CI -0.146 to 0.090).
All thinking traces also hit the 256-token ceiling. The model spent serial compute, but the score did not track
which sibling could actually finish the shown task.

Per the frozen stop rule, no full depth-5 model-guided search was run and no banking follow-up was created.
The supported conclusion is not “recognition-guided search fails in general.” It is narrower: a parameter-free
operation-type prefix, visible I/O, and an existential completion question do not expose actionable partial
viability to this fixed Qwen3.5-4B at think@256.

## Research Program Fit

This experiment connects `structured_execution_and_compilers` with `evidence_conditioned_selection`. C25
showed weak next-operation proposal far from the goal; C47 showed thinking can discriminate some *completed*
computational candidates; C35 left depth 5 unmeasured. The experiment tested the seam between them: use the
model as a recognizer of externally supplied partial structures, then let exact execution do parameter filling
and final selection.

The seam does not hold for this representation. Completed-candidate verification does not automatically
extend to existential reachability of unfinished, parameter-free prefixes. Recognize -> Search -> Bank stops at
Recognize; banking remains unlicensed.

## Method

- Model: only `Qwen/Qwen3.5-4B`, revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, frozen.
- Backend: the experiment-local pinned vLLM runner for every model call.
- Substrate: a fresh procedural list DSL with 16 operation types and 32 concrete parameterized operations;
  no benchmark source or item was read.
- Frozen partitions: 48 exact-depth-4 calibration tasks, 12 exact-depth-5 oracle-development tasks, and 60
  exact-depth-5 primary tasks. Each has 8 visible, 6 label-probe, and 6 hidden examples.
- State: an ordered operation-*type* prefix with no parameters and no materialized intermediate state.
- Semantic live label: positive iff at least one exactly enumerated concrete parameterization and suffix solves
  all visible plus label-probe cases. Every alternative successful factorization contributes positive prefixes;
  the serialized target path is not the label.
- Judge: two-pass thinking. Pass 1 retains up to 256 reasoning tokens. Pass 2 appends the exact
  `</think>\n\nAnswer: ` prefix and reads targeted A/B log probabilities. No-think uses the same A/B readout;
  next-op uses targeted A--P probabilities.
- Calibration groups: all 16 children of selected live parents (highest and lowest completion counts where
  available) plus randomly sampled dead parents. There were 450 sibling groups, 306 mixed-live groups, 7,200
  children, and a 0.0664 live rate.
- Statistics: macro average of within-task, prefix-depth AUROCs; sibling live recall@4; task-cluster bootstrap
  with 10,000 replicates. Pooled AUROC is diagnostic only.

The integrity gate independently recomputed 120/120 uncapped minimum-depth receipts and validated all 120
task/oracle pairs. The three partitions had 120 distinct behaviors on a frozen 64-input common probe bank,
with zero within- or cross-partition collisions. Semantic labels use only visible plus label-probe cases;
hidden cases never enter model prompts, semantic labels, search, or selection.

## Results

### Oracle state: useful

On the dedicated depth-5 development split, the oracle-live width-4 beam achieved path coverage 12/12 and
visible-only consensus selection 12/12 on hidden grading. Mean completed-leaf compression was 262,144x. Live
prefix density fell from 0.1198 at length 1 to 0.00000779 at length 5, so the state is not trivially dense.
This gate loaded calibration and development artifacts only; it did not load primary tasks or hidden labels.

### Model recognition: unreadable and non-actionable

| method | macro within-task AUROC | pooled AUROC (diagnostic) | live recall@4 | mean best-live rank |
|---|---:|---:|---:|---:|
| thinking P(viable) | **0.506** | 0.557 | **0.251** | 7.102 |
| no-think P(viable) | 0.556 | 0.580 | 0.303 | 6.527 |
| next-operation likelihood | 0.504 | 0.521 | 0.228 | 6.983 |
| model-free surface prior | 0.519 | 0.536 | 0.278 | 7.370 |
| deterministic random | 0.506 | 0.498 | 0.263 | 6.953 |

Thinking's AUROC CI was [0.470, 0.543], below the preregistered 0.65 point threshold. Its paired AUROC delta
against no-think was -0.049 [ -0.090, -0.010 ], rather than the required +0.05 with positive lower bound.
Its recall@4 delta against no-think was -0.052 [ -0.110, +0.007 ], rather than the required +0.10 with
positive lower bound. All six gate predicates failed.

No-think's 0.556 AUROC is a small above-chance signal, not a substitute winner: it remains far below 0.65,
and its recall is only 0.040 above random in point estimate. No model arm licensed primary search.

### Model-free depth-5 reference: tractable here

Exact visible-only full brute enumeration covered a hidden solver on 60/60 primary tasks and selected a
hidden-correct program on 56/60. Across the 60 tasks it represented 62,914,560 logical type skeletons and
2,013,265,920 concrete leaves, while the exact behavioral quotient physically computed 227,198,208 vector
transitions. Eight CPU workers completed in 111.85 seconds wall time (833.36 summed task-seconds).

Depth 5 is therefore combinatorially large but not operationally intractable on this DSL and hardware. A model
controller would need to buy substantial token or wall efficiency—not merely make enumeration possible.

### Resources

The full calibration took 2,563.24 seconds of GPU/model wall time. Thinking scored 7,200 children with 14,400
logical requests, 10,030,528 prefill tokens, and 1,850,400 sampled tokens (11,880,928 total logical model
tokens). No-think used 4,118,864 total tokens; the 1,200-row task-shuffle canary used 1,973,376; next-op used
287,321. Every thinking row was forced closed at 256 tokens.

## Controls

- Exact depth is an uncapped exhaustion result. The audit rejected the inherited 60,000-state “not found”
  shortcut before task creation.
- The oracle gate uses a separate development partition. An audit caught an initial implementation that gated
  on primary hidden outcomes; that receipt was replaced before any primary model search.
- Within-task/depth macro metrics prevent task-difficulty discrimination from masquerading as child ranking.
- Task-shuffled visible examples test whether thinking score content is coupled to the shown problem.
- No-think, next-op likelihood, task-independent surface, and deterministic random controls distinguish
  semantic reachability from proposal and structural priors.
- A CPU smoke initially estimated thinking AUROC at 0.662 on two tasks. The 48-task result was 0.506. Smoke is
  infrastructure evidence, not scientific evidence.
- The primary harness was hardened before launch to require both sampled-token- and total-model-token-matched
  direct sampling, exact task alignment, relational artifact hashes, pool-exhaustion checks, and immutable
  gate receipts. Those arms remain unrun because calibration stopped them.

## Oracle Versus Deployable Evidence

Oracle evidence establishes that an exact live-prefix signal would compress search dramatically. It does not
establish that the model can compute that signal. Deployable evidence is the visible-only P(viable) ranking,
and it is null within tasks. The final hidden set is used only after a candidate pool and visible-only selector
are frozen; for this stopped experiment, hidden primary outcomes appear only in the independent full-brute
reference.

The full-brute split is also instructive: coverage was 1.0 while selected success was 0.933. Even exhaustive
visible consistency leaves four ambiguous/overfit selections. That is a downstream selection issue, separate
from the failed recognition controller.

## Interpretation

The most likely failure is representational, not merely statistical. “Can this prefix still be completed?” is
an existential inverse problem over missing parameters and missing operations. The prompt supplies neither
the parameter-domain constraints already induced nor per-example intermediate/residual states. An exact
oracle can recover reachability from the full transition system; the model sees a thin symbolic name sequence.
Completed-candidate verification (C47) is consequently not the right analogy: verifying a concrete finished
program is much closer to execution, whereas prefix viability requires constructing or ruling out a suffix.

More serial text did not bridge that gap. Thinking was worse than the one-token no-think readout, all traces
hit the budget, and wrong-task visible examples were no worse. The useful lesson is not “increase the thinking
budget.” It is “change what the state makes explicit.”

The gate saved the expensive part of the program. Without within-task and recall requirements, pooled AUROC
0.557 could have been narrated as promising and triggered millions more model tokens in depth-5 search. The
stop is a successful research outcome: it retires the type-only viability representation before search and
banking compounded it.

## Limitations and Protocol Deviations

- Calibration is depth 4; depth-5 model viability was deliberately not measured after the gate failed.
- Think@256 was always truncated. Longer thinking is untested, although the task-shuffle and negative lift make
  a budget-only rescue a weak next bet.
- Parent sampling implemented max/min-completion live parents and random dead parents. The design review's
  broader likelihood-frontier, uniform-frontier, one-edit, and hard-negative mixture was not implemented. The
  result therefore scopes to these sibling groups, not every possible on-policy frontier distribution.
- Exact-depth task construction uses all three frozen case partitions to reject shallow-equivalent targets;
  semantic live labels use visible plus label-probe only. This construction-time use is shared across arms and
  is distinct from model/search leakage, but “hidden grading-only” should be read as applying after tasks are
  frozen.
- The initial oracle-gate code consumed primary hidden grades. A pre-primary audit replaced it with the
  dedicated 12-task development gate. No primary model output was generated under either version.
- One model revision, one DSL, one thinking budget, and one independent-score prompt bound generalization.

## Learned Lessons

1. **Gate on control utility before model readability.** Sparse oracle reachability made the representation
   worth testing; this avoided confusing a bad state with a bad scorer.
2. **Pooled AUROC is unsafe for search controllers.** Here 0.557 pooled collapsed to 0.506 within task.
3. **Ranking action, not separability, is the launch criterion.** Recall@beam would have stopped the line even
   if AUROC had barely passed.
4. **Smoke can reverse the conclusion.** Two tasks suggested 0.662 AUROC; 48 tasks showed chance.
5. **Match prefills as well as decode tokens.** Repeated judge prompts are most of recognition's model work;
   a decode-only sample-more baseline is underfunded.
6. **Measure brute wall time.** The nominal million-leaf depth-5 problem took under two minutes in parallel
   with an exact behavioral quotient, changing the economic bar for model guidance.
7. **Keep raw traces without making Git the artifact store.** Detailed 178 MB calibration traces were archived
   with checksums; compact score rows and a sealed receipt remain in-repo.

## Next Experiments

1. **Locate the actual brute-force crossover (recommended).** Measure the same exact quotient at fresh
   exact-depth 6, including wall time, peak memory, physical transitions, coverage, and selector success. If it
   remains operationally cheap, retire model-guided pruning on this DSL. If it creates a real resource wall,
   that becomes the justified target for learned guidance.
2. **Residualized-state recognition, calibration only.** Expose feasible parameter domains, materialized prefix
   outputs where parameters are fixed, and per-example residual constraints; compare independent P(viable)
   with one listwise best-4-of-16 readout. Do not create a new primary split until an arm clears the same
   within-task and recall gate.
3. **Repair visible-only selection over exact pools.** Full brute covered 60/60 but selected 56/60. Compare the
   current consensus selector against leave-one-visible-example-out stability, behavioral simplicity, and
   support on fresh unlabeled probes, freezing the rule before a new primary evaluation.

Do not bank this result. A banking experiment is licensed only after a separately held-out search win.

## Artifact Manifest

The compact source of truth is `analysis/summary.json`; gate and resource receipts are under `runs/`. Detailed
raw calibration traces are external under `large_artifacts/qwen35_4b_partial_structure_search/calibration/`
and are checksum-indexed by `analysis/calibration_compaction.json` and `reports/artifact_manifest.yaml`.
