# Adversarial design review: verified macro invention

Date: 2026-07-09, before model smoke. Verdict: **sound with must-fix changes**. The
preregistration and implementation incorporate the fixes below; deviations must be recorded in
the experiment log before any scored continuation.

## 1. Compression is guaranteed, so it is not evidence

The latent source grammar deliberately contains repeated motifs. Surface-depth reduction and
oracle compressibility are therefore substrate gates, not results. The only headline endpoint is
visible-only selected hidden accuracy. A positive result means a learned representation changed
Qwen's search distribution; it does not show deeper primitive-composition capability in the
weights.

**Resolution:** freeze the latent grammar, construction/smoke/eval files, and their hashes before
mining. Prohibit exact full-program and behavioral-signature overlap. Treat intended length-2/3
window overlap as the treatment and report it. Never regenerate evaluation because a library
compresses it poorly.

## 2. Callable macros are confounded with a highlighted prior

`mined > base` can be caused by repeated-sequence hints, extra context, a larger inventory, or
shorter callable output.

**Resolution:** add `mined_hint`, with identical neutral aliases, definitions, ordering, and
underlying demonstrations, but require expanded base-primitive output. Callable chunking requires
`mined` to beat both base and hint. All arms receive the same train-only demonstration tasks in
base-expanded form; no demo is selected because it contains an arm's motifs.

## 3. One random library is not a placebo distribution

Library-draw variance may exceed task-sampling variance.

**Resolution:** freeze at least five count/length/support-matched random libraries. Exclude exact
and behavioral duplicates, identities/cancellations, and expansions equivalent to shorter
programs. Give them the same syntax and definition detail. Analyze task and library-draw
variation. Construct a separate matched-placebo ensemble for the Qwen-selected library if its
support profile differs from the deterministic mined library.

## 4. Qwen selects candidates from a closed macro language

If legal macros are all supported contiguous length-2/3 subsequences, deterministic mining already
enumerates the complete candidate language. Qwen is proposing/ranking entries, not inventing a new
schema.

**Resolution:** call the arm `qwen_ranked`. A Qwen-specific verdict requires a full eight-entry
verified library, a matched placebo, performance within 0.05 of deterministic mining, and at least
two correct selections uniquely enabled by Qwen-exclusive entries. Otherwise report proposal
construction failure or ranking-only recovery; never silently pad the library.

## 5. Exact verification must stay exact

Finite probe agreement for model-written code would not verify a macro.

**Resolution:** accept only canonical lists of allowed base tokens. Macro semantics are literal
expansion through the committed interpreter. Names/rationales are recorded but Qwen-authored names
are not exposed to the solver.

## 6. Split difficulty and minimum depth

An iid-uniform no-reuse split would differ in primitive marginals and task difficulty. A historical
depth helper capped its search too shallowly for true depth-5 verification.

**Resolution:** create no-reuse tasks by shuffling the exact primitive multiset of paired reuse
tasks until no latent motif remains, preserving base depth, primitive marginals, and distinct-op
count. Report output-length/entropy diagnostics. Prove depth 5 by exhaustively searching every
shorter program through depth 4; a search-cap exit is unverified, never passed.

## 7. Confirmatory logic

The review recommends a conjunction rather than choosing a favorable comparison:

1. system benefit: `mined - base >= +0.10`, paired 95% lower bound above zero, surviving a base
   prefix with no-smaller measured compute;
2. callable representation: `mined - mined_hint >= +0.05`, paired lower bound above zero;
3. learned recurrence: `mined - matched_random >= +0.05`, paired/hierarchical lower bound above
   zero across frozen random draws.

Only all three support callable verified abstraction. Gate 1 alone supports the complete prompt
intervention; gates 1+3 without gate 2 support motif highlighting rather than callable chunking.

## 8. Compute and vLLM

All proposal and solver inference must use the copied vLLM runner. Report exact sampled tokens,
logical input/prefill tokens, interpreter work, and batched wall time. Unique prompt tokens are not
treated as actual compute with prefix caching disabled. Include Qwen proposal cost both cold and
amortized at fixed deployment horizons.

If a relevant full arm force-closes over 50%, run the preregistered larger-budget subset for every
corresponding comparison arm or label the contrast budget-confounded. Never patch one arm only.

## 9. Minimum gates

Before full generation:

- frozen split hashes and zero forbidden overlap;
- exact depth verification through `d-1`;
- matched, nondegenerate libraries;
- within-arm smoke parse rate >=0.50 for base and designed ceiling;
- at least two valid macro-using designed-arm candidates;
- answer truncation below 5%;
- forced-close rate measured and the contingency honored;
- no inspection-driven tuning on full outputs.

## Pre-registered outcome branches

- Designed ceiling has no oracle lift: interface failure; abstraction quality unresolved.
- Oracle lift without selected lift: selection bottleneck.
- Mined ties hint: highlighted prior sufficient; callable representation unsupported.
- Mined ties random: extra composite vocabulary/syntax sufficient; recurrence unsupported.
- Reuse and matched no-reuse move equally: recurring-motif mechanism rejected.
- Mined passes while Qwen fails: verified tool mining works; no model-invention evidence.

## Addendum: review of the v2 interface repair

Date: 2026-07-09, after failed smoke v1 and before any v2 GPU call. Verdict: **scoped repair is
sound if the amendment is followed exactly**.

Smoke v1 passed parser-rate gates but all 1,440 solver samples force-closed, 607 answer stages
truncated, and the designed arm never emitted a valid macro-using candidate. The designed oracle
was therefore zero. This identifies an unusable elicitation surface, not a negative result on
macro quality.

### Post-smoke tuning risk

Repair after observing outputs can silently become evaluation tuning.

**Resolution:** allow only v1 smoke, v1 train-only proposal output, CPU diagnostics, and a
train-only plan-given probe into the repair. Use fresh v2 seed `20260710` and new ids; prove v2 is
behaviorally and exactly disjoint from construction, v1, and full. Keep every full task and
decision rule frozen. If v2 fails, stop without inspecting full outputs.

### Budget and procedure could become new treatments

Increasing thinking or explicitly teaching alias use could manufacture a designed-arm advantage.

**Resolution:** use think@768 because it was already the registered full budget, not a value chosen
from a sweep. Apply the same surface-first infer/rewrite/check procedure and the same abstract
alias example to base and designed prompts. The placeholder example explains calling convention
only and contains no real primitive, macro, task, or answer. Use matched K=12. Treat smoke as a
go/no-go interface check, never a treatment estimate.

### A plan-given probe could leak the answer pattern

A probe chosen from evaluation motifs could train the prompt toward the target.

**Resolution:** use verified train-only plans and label the probe non-scored. It checks strict
rendering and exact alias expansion only. It cannot select macros, tasks, examples, thresholds, or
full-run prompts based on hidden behavior.

### Parser repair could salvage solver prose selectively

Line-local recovery would inflate parse and allow different amounts of cleanup across arms.

**Resolution:** keep the solver parser strict: one answer-region line, supplied inventory only, no
prose salvage. Proposal lines are independent records, so their predeclared parser may validate
lines independently and retain the first eight valid supported unique entries. Record every
rejection. The exploratory v1 line-local audit remains diagnostic and is never rescored as v1.

### Repeated macro samples could fake interface breadth

Two macro-using candidates from one easy task would meet the old count without demonstrating that
the surface works across tasks.

**Resolution:** require valid designed alias use on at least two distinct fresh reuse tasks, plus
the original parse, truncation, and oracle-not-below-base gates. Run only base and designed at
smoke; generate mined, hint, random, and Qwen-ranked full arms only after the interface passes.

All confirmatory full contrasts and the 1536-token force-close contingency remain unchanged.

## Addendum: review of no-think interface attempt 3

Date: 2026-07-09, after interface attempt 2 and before any attempt-3 GPU call. Verdict: **a
no-think retry is justified only for the plan-given transcription gate**.

Attempt 2 succeeded on 2/4 records, with 4 strict valid samples that all used macros, but 12/16
answers truncated. All 16 samples exhausted think@768 and force-closed. The run stopped before any
fresh induction prompt, so the fresh smoke tasks and all full tasks remain model-unseen.

### No-think could erase reasoning needed by the scientific task

Turning thinking off for induction would be a new treatment and would break comparability with the
registered full protocol.

**Resolution:** use vLLM `thinking: off` only for interface attempt 3. Its prompt supplies the exact
primitive plan, so the measured operation is alias substitution and strict transcription. If the
gate passes, restore the unchanged think@768 protocol before the first fresh induction prompt.

### A third attempt could become threshold shopping

Repeated prompt or gate changes could eventually manufacture a pass.

**Resolution:** freeze the same four prompt contents and targets, n=4, answer cap 128, parent seed
family, strict parser, exact expansion check, and original thresholds. The only intervention is the
thinking channel. Require at least 3/4 successful records and truncation below 0.05; do not combine
samples across attempts. A failure ends the experiment before fresh smoke.

### Partial attempt-2 success could be overstated

Four macro-using samples show that the surface is sometimes callable, but 2/4 record coverage and
0.75 truncation are below the registered reliability requirements.

**Resolution:** preserve attempt 2 as a failed mechanical diagnostic. Do not promote its valid
samples into the scored smoke, proposal ranking, or any macro claim. Attempt 3 starts a separate
gate artifact and cannot backfill attempt 2.

The construction corpus, libraries, fresh smoke, full evaluation, proposal protocol, scientific
sampling budgets, and all confirmatory decisions remain frozen.

## Final addendum: attempt-3 result and stop-rule enforcement

Date: 2026-07-09, after interface attempt 3. Verdict: **stop the experiment; do not add amendment
4**.

No-think cleanly isolated transcription mechanics. All 16 outputs parsed, all 16 called macros,
and none truncated, so format and answer-boundary defects no longer explain the gate. Only one of
four records had any macro-using optimal surface whose literal expansion equaled the supplied
plan. The exact 1/4 result fails the unchanged 3/4 requirement.

The subsequent committed raw-row audit made the failure pattern regenerable: all 13 failed
samples used multiple aliases and expanded beyond depth five; 10/13 included the designated alias
and then introduced unrelated aliases, while 3/13 omitted it. This audit does not alter the frozen
gate. The exact executor—not parse or raw macro use—correctly determined failure.

### Why another in-place repair is not allowed

Attempt 3 held prompts, targets, K, answer cap, parser, seed family, and thresholds fixed while
changing only the thinking channel. Its failure exhausts amendment 2's explicit branch. Further
prompt examples, output constraints, alias syntax, decoding changes, or tool mediation would be a
new interface treatment, not a mechanical retry. Continuing here would convert a protected gate
into iterative threshold shopping.

### Evidence boundary at closure

The fresh induction tasks were never prompted, and full generation never began. Therefore:

- do not infer that verified macros help or fail on held-out induction;
- do not promote plan-given interface metrics into a capability claim;
- do not add a claim-ledger entry; and
- preserve the unseen scientific tasks and all failed-interface artifacts.

A future material follow-up may redesign how exact composite calls are represented or constrained,
but it requires its own experiment scaffold, intake, design review, preregistration, and
matched-compute baseline. This stopped directory remains immutable evidence of the interface
sequence and its stop decision.
