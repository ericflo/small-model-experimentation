# Preregistration: verified macro invention long-context rerun

Status: frozen on 2026-07-09 before the first follow-up GPU call. Constants live in
`configs/default.yaml`; automated gates and the committed analyzer, not hand calculation, decide
branches and verdicts.

## Question and evidence boundary

Can reusable composite operators, derived only from prior verified programs, improve visible-only
selected accuracy on fresh behaviorally true-depth-5 tasks beyond matched-compute sampling over the
original primitives when Qwen3.5-4B receives a demonstrably nonbinding inference envelope?

The parent `qwen35_4b_verified_macro_invention` remains immutable. It stopped before showing the
model any v2 induction-smoke or full-evaluation prompt. Its budgeted plan-given gate was censored:
16/16 completions exhausted 768 thinking tokens and 12/16 exhausted the 128-token answer stage.
Those artifacts establish that the old setup was too small, not that verified macros fail. The
later no-thinking probe measured a distinct transcription condition and cannot substitute for
budgeted induction. This follow-up therefore carries no parent accuracy result forward.

## Frozen inheritance

The follow-up copies the parent's inputs byte-for-byte from commit
`1c8c5bbb81d2a67618891597205ceb2f40f498d8`. It does not regenerate, reshuffle, filter, or relabel
them. Exact provenance is in `data/source_provenance.json`.

- `tasks.json` SHA-256:
  `82fbbd57e26fd392aa8f30ec6f26d370dc08dd78b3279bed6ee2e2174aea5073`
- `libraries.json` SHA-256:
  `a2ae3663753a3a0d0c9614a5d7c1d250506c74fd7879e11e99b66f5c1e43f865`
- frozen full-evaluation subset hash:
  `16838c0b8b85b105023b72e3459035b78bcf20bf5f3e28f00ed7c93311680c24`
- frozen v2-smoke subset hash:
  `cf2467a05bd11d279e692c30a64a0a3508a9ccde7bc2979ba6f6addb4a3ff5db`

The corpus contains 800 construction programs, a 150-program train-only proposal view, 12 v2
smoke tasks, and 120 full tasks. The full split has 80 `reuse` and 40 `no_reuse` tasks. Every
scientific task has disjoint visible, hidden-grade, and unlabeled-probe inputs; exact CPU execution
verified behavioral minimum depth five by exhausting every shorter program through depth four.

The parent CPU gates remain required: zero construction/evaluation exact-program and behavioral
signature overlap; no macro equal to a full evaluation program; five distinct frequency-matched
placebo libraries; designed-macro median surface reduction of at least two on `reuse` and at most
0.5 on `no_reuse`; and exact preservation of the frozen smoke/full hashes.

## Experimental unit and splits

The scientific unit is one procedural list-transformation task in the parent's bounded,
parameter-free DSL.

- `reuse`: true-depth-5 programs formed from recurring motifs in combinations absent from the
  construction corpus. No macro equals a complete evaluation program.
- `no_reuse`: primitive-multiset permutations paired to the reuse substrate, at the same verified
  depth and excluding the three evaluation-recurrent motifs.
- `smoke`: a disjoint seed and 12 tasks, never included in the full estimate.

Subprogram overlap is the intended treatment. Full-program or behavioral-signature overlap is
forbidden leakage.

The interface calibration and gate are not scientific units. They use plan-given, verified
construction-only records. They contain no smoke/full I/O or target program. The heldout interface
records are disjoint from the calibration records; "heldout" here means held out from budget
calibration, not part of the scientific evaluation.

## Libraries and scientific arms

Every callable entry literally expands to two or three allowed base primitives and is exact by
construction.

1. `base`: base primitives only.
2. `mined`: the top supported, nonredundant subsequences mined deterministically from the frozen
   train-only proposal view.
3. `mined_hint`: identical subsequences and prompt information, but aliases are not legal output;
   the model must emit expanded primitives.
4. `qwen_ranked`: Qwen proposals from the same train-only view, accepted only by syntax, literal
   expansion verification, and train support, then frozen before evaluation.
5. `random_0` through `random_4`: five count/length/support-matched non-selected libraries. If the
   Qwen library profile differs, it receives its own independent five-draw matched ensemble.
6. `designed_ceiling`: generator-known recurring motifs, reported only as a non-discovery ceiling.

Aliases remain neutral (`M0`, `M1`, ...), definitions show literal expansions, and demonstrations
remain identical and base-expanded across arms. The scientific prompts and strict
`PROGRAM: OP | OP | ...` parser are inherited; only the inference envelope and protected
anti-censoring gates change.

## Model and vLLM inference

The only model is `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. Every calibration, proposal, interface, smoke, and
full completion uses this experiment's copied `src/vllm_runner.py` under `.venv-vllm`. Transformers
inference and mixed backends are forbidden. Each batch uses vLLM `n`, not a Python request loop.

The engine is frozen to `max_model_len=65536`. The runner must preflight every rendered prompt plus
the registered thinking allowance, exact injected close sequence, and 512-token answer allowance
against that context. A context-overflow or engine error is an infrastructure failure and produces
no score.

Sampling retains the parent's temperature, top-p, seed derivation, and K values. Macro arms use
K=12; base uses K=24 so a measured-token prefix can represent the sample-more baseline. Every
comparison arm at a given stage uses the same budget rung and answer allowance.

## Metadata-only budget calibration

The allowed budgeted-thinking ladder is fixed before generation:

`[2048, 4096, 8192, 16384, 32768]`

The answer allowance is 512 tokens at every rung. The calibration set is four deterministic,
plan-given, verified train-only records with 16 samples per record, yielding 64 completions per
rung. These records are disjoint from the 16-record interface gate and from all smoke/full tasks.

Run ladder rungs in ascending order and select the smallest rung satisfying all four conditions on
the complete 64-completion batch:

1. thinking-cap/contact rate is below 5%;
2. answer-truncation rate is below 5%;
3. nearest-rank p99 thinking-token count is at most 75% of that rung's allowance; and
4. nearest-rank p99 answer-token count is at most 50% of 512, i.e. 256 tokens.

Thinking-cap/contact conservatively means any completion with a forced close, a stage-one `length`
finish reason, or an exact thinking-token count that reaches the registered allowance. Answer
truncation means the second stage ended by its length limit rather than the registered terminal
token. p99 uses exact runner token counts, not retokenized text.

Only runner metadata fields needed for those four conditions may be read by the selector. Generated
text, parse status, task correctness, macro use, and all hidden/probe behavior are unavailable to
budget selection. Raw rows remain sealed artifacts until a rung has been selected. Zero cap and
truncation are preferred diagnostics, but the four numerical conditions—not output quality—choose
the rung. If no rung qualifies through 32,768, stop as **budget calibration inconclusive**. Do not
run the heldout interface, smoke, or full matrix.

## Heldout plan-given interface gate

At the selected rung, run 16 deterministic, construction-only plan-given records at n=4 (64
completions). Each target is macro-compressible using the supplied designed library and has an exact
known primitive expansion. A record is covered when at least one of its four completions:

- passes the strict one-line program parser;
- uses at least one supplied macro alias;
- has the target's optimal macro surface-call length; and
- literally expands to exactly the supplied primitive plan.

The gate passes only when at least 12/16 records are covered, the 64-completion thinking-cap rate is
below 5%, and the answer-truncation rate is below 5%. The p99 calibration headroom conditions do not
apply to this heldout gate.

If either censoring rate reaches 5%, advance to the next registered rung and rerun all 16 records
and all four samples. Preserve the lower-rung output as a censored diagnostic, but do not pool rows,
backfill record coverage, or select favorable samples across rungs. Exhausting 32,768 while
censored is setup-inconclusive. Coverage below 12/16 with both censoring rates below 5% is a clean
interface-gate failure and stops before any fresh induction prompt.

## Scientific stage escalation

The rung that clears the heldout interface becomes the initial rung for train-only proposal,
fresh smoke, and full generation. Censoring is checked automatically before task metrics are
released.

- A proposal batch with thinking-cap or answer-truncation rate at least 5% is rerun in full at the
  next rung before parsing proposals.
- A smoke comparison is eligible only if the pooled and each-arm thinking-cap and
  answer-truncation rates are below 5%. Otherwise rerun the complete base/designed smoke matrix at
  the next rung.
- A full result is eligible only if the pooled and every arm's thinking-cap and answer-truncation
  rates are below 5%. Otherwise rerun the entire full arm-by-task-by-sample matrix at the next rung.

No arm receives a private budget increase. Lower-rung rows are retained for diagnosis but never
pooled, substituted, or included in a scored estimate. Every published comparison comes from one
complete matrix at one rung. If a stage is still censored at 32,768, label that stage
setup-inconclusive and make no macro-mechanism claim.

## Fresh smoke gate

Only after the heldout interface passes may the unchanged v2 smoke prompts be rendered. Smoke runs
`base` and `designed_ceiling` at matched K=12 after any required whole-matrix escalation. It passes
only if:

- pooled parse rate across base and designed is at least 0.50;
- parse rate within each arm is at least 0.50;
- designed yields at least two valid macro-using candidates on at least two distinct reuse tasks;
- designed oracle coverage is not below base oracle coverage; and
- the anti-censoring conditions above hold for both arms and pooled.

Smoke is a go/no-go interface check, not hypothesis evidence. Its outputs cannot tune full prompts,
libraries, parsers, thresholds, or task selection. A clean smoke failure stops full generation and
localizes the failure as an induction/interface ceiling, not a test of mined versus random macro
quality.

## Metrics and selection

Primary deployable metric:

- selected hidden-all accuracy. For each task and arm, select the earliest syntactically valid
  candidate with maximal visible-example score; break ties by sample order. Abstention is failure.
  Hidden labels never influence selection.

Secondary metrics:

- hidden oracle coverage@K, explicitly non-deployable;
- parse, valid-program, visible-pass, abstention, and false-visible-pass rates;
- macro-use among selected and correct candidates;
- expanded primitive depth and surface-call depth;
- library support, compression, motif recall, and overlap audits;
- exact sampled thinking/answer tokens, injected tokens, unique prompts, logical model inputs,
  generation wall time, and interpreter calls. Qwen proposal cost is shown cold and amortized.

## Confirmatory contrasts and unchanged thresholds

The sole primary contrast is `mined - base` on the pooled `reuse` split at K=12. Report a paired
task-bootstrap 95% interval. The macro mechanism clears only if all of these inherited conditions
hold:

1. point improvement is at least +0.10 and the paired interval lower bound is above zero;
2. the improvement remains positive against the base prefix with a no-smaller sampled-plus-unique-
   prompt token budget;
3. at least half of treatment-only correct selections actually use a macro; and
4. `mined - base` on `no_reuse` is no larger than half the reuse improvement.

Callable chunking additionally requires `mined - mined_hint >= +0.05` with a paired 95% lower bound
above zero. If hint ties mined, highlighting a verified prior is sufficient and callable chunking
is not established.

Learned recurrence additionally requires `mined - random_mean >= +0.05` with a positive
task-and-library-draw interval. The complete callable-abstraction verdict is the conjunction of the
base, hint, and random gates; no favorable single contrast substitutes for another.

Qwen-specific value is secondary and unchanged. `qwen_ranked` must contain exactly eight unique
supported verified entries, beat its independently matched random ensemble by at least +0.05 with
a positive interval, finish within -0.05 of `mined`, and uniquely enable at least two correct
selections. Otherwise report ranking-only recovery or proposal-construction failure, not invention.

All other arm/slice contrasts are descriptive. Significant-versus-nonsignificant comparisons are
not evidence of a difference.

## Hidden-label and inspection boundary

- Budget calibration sees runner metadata only.
- Macro mining/proposal sees construction programs only.
- Interface records are verified train-only plans and cannot alter scientific inputs.
- Solver prompts see visible I/O only.
- Hidden-grade and unlabeled-probe outputs enter only the already committed analyzer.
- No smoke or full output may be inspected to change prompts, parsing, budgets outside the fixed
  ladder, libraries, task membership, or thresholds.

## Registered outcome branches

- Ladder exhausted: inference setup unresolved; no capability conclusion.
- Clean heldout interface failure: free-form alias transcription remains unreliable under the
  tested envelope; induction hypothesis still untested.
- Designed smoke has no oracle lift: interface/induction ceiling; do not interpret mined/Qwen
  differences as abstraction quality.
- Oracle lift without selected lift: selection bottleneck.
- Mined ties hint: highlighted prior sufficient; callable representation unsupported.
- Mined ties matched random: extra composite vocabulary/syntax sufficient; recurrence unsupported.
- Reuse and matched no-reuse move equally: recurring-motif mechanism rejected.
- Mined passes while Qwen fails: verified deterministic mining works; no model-invention evidence.

## Scope

This is a corrected test of reusable composite abstractions in one fresh procedural DSL, not a
universal concept-formation study and not an optimization of thinking-budget accuracy. The ladder
is an anti-censoring ceiling selected without correctness feedback. A positive result would be a
system capability from verified tools plus Qwen, not proof that new capability entered the model's
weights.
