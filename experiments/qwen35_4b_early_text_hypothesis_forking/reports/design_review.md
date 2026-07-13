# Adversarial Design Review: Early Text Hypothesis Forking

Completed before any model-bearing call. Two independent reviewers attacked the
scientific design and implementation plan. The initial 12-type design is
withdrawn; the resolutions below define the only authorized experiment.

## Verdict

Proceed through model-free smoke, then a separately locked live mechanics run.
The experiment can test whether an early, concrete textual hypothesis reshapes
Qwen3.5-4B's proposal distribution. It cannot by itself establish J-space
transport, internal certainty, weight installation, or autonomous capability.
Because the public depth-two DSL is exhaustively searchable, even a replicated
positive is verifier-assisted proposal shaping on this substrate, not a win
over the strongest available symbolic tool.

## Attacks and resolutions

1. **The original “twelve concrete operations” were only operation types.**
   Parameter binding was still left to the model, despite prior evidence that
   this is the deployable bottleneck. The systematic bank now enumerates all 24
   bound operations: eight parameter-free, six `add_k`, three `mul_k`, four
   `take_k`, and three `rotate_k`. `negate` remains a distractor but is excluded
   as a generated first step because it creates many reorderable equivalents.

2. **A unique operation name does not identify a concrete first step.** Every
   generated visible set must uniquely identify the complete `(name,
   parameter)` first operation over all 576 two-step programs. Type-only
   uniqueness is insufficient.

3. **“Exact depth two” can hide many visible-consistent programs.** The CPU
   generator enumerates the full 24² grammar. It rejects depth-one fits and
   stores every visible-consistent two-step program. A task is admitted only if
   those programs agree on all hidden and unlabeled probe inputs; therefore a
   visible-only tie cannot be rescued by hidden labels.

4. **CPU brute force already solves the public search problem.** A deterministic
   exhaustive visible-DSL arm is mandatory. It is the scope ceiling and
   strongest tool control. If it solves the task, a model win over sampling
   means more efficient model proposal stratification only; it is not a new
   end-to-end capability unavailable to the system.

5. **The old bare `PROGRAM: a | b` ABI may repeat a known formatting failure.**
   Final answers use a strict, natural Python helper function. An AST parser
   accepts only `def transform(xs)`, exactly two allowed helper assignments,
   and `return xs`. It rejects imports, attributes, control flow, arithmetic,
   extra statements, and out-of-domain parameters. Parsed code is canonicalized
   and interpreted; generated Python is never executed.

6. **Putting early text in the user message confounds timing with chat role.**
   Early and late conditions append exactly the same provisional-hypothesis
   token IDs inside the open `<think>` channel. Prefix stitching is performed
   on token IDs, not decode-and-retokenize text. Hashes prove injection identity.

7. **One shared late prefix destroys diversity and favors the early arm.** Each
   late branch receives its own candidate-blind 512-token prefix. A shared-
   prefix result may be reported only as a historical diagnostic, never as the
   primary timing comparator.

8. **Early versus late still has unequal usable reasoning time.** Two late
   controls are mandatory: `late_equal_total` gets 512 blind plus 512
   post-hypothesis tokens; `late_equal_post` gets 512 blind plus the full 1,024
   post-hypothesis tokens. A timing claim requires early to exceed the first and
   not lose to the overmatched second.

9. **A special sentence may help independently of its semantic content.** The
   design includes 24 exact-scaffold placebo branches whose hypothesis span is
   token-accounted, 24 duplicate branches, a neutral-scaffold sample-more pool,
   and a plain sample-more pool. Candidate text and branch position are
   task-randomized independently.

10. **Systematic enumeration receives more useful diversity than “sample
    more.”** One frozen neutral master pool and one frozen plain master pool are
    ordered before outcomes. They supply sampled-token-matched,
    logical-model-token-matched, full-K, and first-over-budget controls. No
    favorable fractional sample or outcome-sorted prefix is allowed.

11. **Call counts do not match inference compute.** Resource receipts count
    prompt prefill, every independent late prefix, resumed-prefix re-prefill,
    injected tokens, sampled thought, forced close, answer continuation,
    invalid/duplicate outputs, and any verifier/model calls. Both sampled tokens
    and logical model tokens are matched; wall time is descriptive.

12. **A selector can quietly use the answer.** Public task objects have an exact
    schema. Prompts, branch plans, raw outputs, resource receipts, canonical
    candidates, and selected IDs are written and hashed before gold files are
    opened. Mutating hidden examples, the gold first operation, or the target
    pipeline must leave all pre-grade bytes and predictions unchanged.

13. **A loose selector can be tuned after seeing results.** All arms use the
    same frozen rule: strict parse, canonical deduplication, exact visible pass,
    grouping by unlabeled-probe behavior, cluster support, then a task-seeded
    canonical hash. No passer means abstention. Invalid outputs and abstentions
    are failures. Hidden correctness never breaks a tie or triggers a rerun.

14. **Slot order can reproduce the parent composed-map cancellation bug.** Task
    behavior, branch permutation, and decode streams use domain-separated
    seeds; cyclic shifts are prohibited. Smoke serializes the actual composed
    map `slot -> bound operation -> fixed-panel behavior`, requires taskwise
    variation, and checks balanced gold positions. Changing the branch seed
    must not change task behavior.

15. **Mechanics may only measure blind copying.** Four public diagnostic lists
    are crossed with all 24 bound operations. Correct list computation, not a
    copied name, is the endpoint. Mechanics requires broad operation support,
    an unrestricted parse gate, low cap contact, candidate adherence, and a
    large taskwise advantage over deranged/duplicate/placebo controls.

16. **A mechanics pass can coexist with an unusable full-program ceiling.** The
    correct-hypothesis branch must also produce a visible-pass full program on
    at least half of mechanics program cases before qualification is authorized.

17. **Natural close and forced-close behavior can differ by arm.** Every arm
    uses the same fixed-cap/force-close policy. Exact `<think>` placement,
    premature close/EOS, cap contacts, unresolved thinking, and loop indicators
    are recorded. Any primary arm below 90% unrestricted AST parse or above 5%
    cap contact invalidates mechanics.

18. **Fresh task IDs do not prove fresh behavior.** Complete visible/hidden
    behavior fingerprints are checked against readable procedural experiment
    artifacts. Benchmark contents are never opened. Qualification and
    confirmation are content-disjoint and never pooled.

19. **Branches are not independent statistical units.** The task is the unit.
    Qualification uses 48 tasks in two frozen blocks and confirmation uses 96
    untouched tasks. Paired task bootstrap intervals and Holm correction cover
    the mandatory comparator family; both qualification blocks must agree.

20. **A ten-point gate can be impossible near saturation.** Early selected
    accuracy must lie in the registered 15–85% range. Gate reachability is
    checked before interpreting gains. No ceiling arithmetic may be waived
    post hoc.

21. **One easy operation can carry the aggregate.** Report bound-operation and
    operation-type matrices, parameterized versus parameter-free cases, program
    diversity, candidate adherence, false visible passes, oracle coverage, and
    selector capture. A pass requires broad type support and limits any one
    type to 25% of the gain.

22. **Coverage is cheap when every first operation is supplied.** Oracle
    coverage is necessary but never sufficient. The deployed visible-only
    selection must beat every registered late, duplicate, placebo, neutral, and
    plain matched-sampling construction; selector capture must be at least 90%
    of the early pool's oracle coverage.

23. **A positive could be described as an installed or latent capability.** No
    weights change. The strongest allowed interpretation is externally
    stratified, verifier-assisted text elicitation. A separate clean experiment
    would be required for training, controller installation, higher-depth
    transfer, or verifier-free deployment.

24. **Depth two is too easy to establish a frontier result.** It is retained as
    a feasibility test because supplying the first bound operation leaves one
    plausible residual step. Depth three previously left an oracle-hinted
    residual that was too hard to diagnose. A replicated depth-two pass licenses
    a fresh successor at materially larger depth/inventory; it does not license
    silently extending this result-bearing directory.

## Required pre-GPU boundary

- deterministic 24-operation inventory and exhaustive 24² search tests;
- unique concrete-first-step, no-depth-one, and visible-equivalence checks;
- strict AST parser/executor round trips and malicious-syntax rejections;
- selector/generation/resource invariance under every gold mutation;
- independent branch permutations and composed-map receipts;
- exact early/late injection-token and prefix-stitching tests;
- outcome-blind matched-pool construction and exhaustion tests;
- fresh-data collision audit without benchmark reads;
- all later stages fail closed; and
- reviewed code, public data, tests, and smoke hashes committed and pushed
  before the first live model call.

## Fail-specific terminal outcomes

- `INVALID_INTERFACE_PARSE`
- `NO_HYPOTHESIS_ADHERENCE`
- `NO_CORRECT_HYPOTHESIS_CEILING`
- `EARLY_EQUALS_LATE`
- `NO_MATCHED_SAMPLING_GAIN`
- `PROPOSAL_SHIFT_SELECTOR_FAIL`
- `NO_PROPOSAL_SHIFT`
- `EARLY_HYPOTHESIS_FORKING_CONFIRMED`

No failure authorizes prompt, parser, layer, temperature, candidate, or budget
tuning in this experiment.
