# Mechanics implementation review

Status: attempt-1 implementation review passed, but live preflight exposed a
group-aware cache-validation bug before the first experimental request. The
append-only v2 repair is under independent adversarial review; model execution
remains withheld pending a committed/pushed v2 preoutcome and separate v2 lock.

No model was loaded and no model call was made during this review. Only the
published public mechanics and audit files were opened. No gold, qualification,
confirmation, or benchmark file was opened.

## Implemented boundary

The mechanics implementation is deliberately separate from qualification and
confirmation. It prepares nine immutable invocations totaling 1,984 requests:
four 52-row live-suffix arms, one 24-row direct ABI arm, three 576-row binary
viability arms, and one 24-row listwise arm. It cannot construct or run a
qualification or confirmation prompt.

Stable full-SHA256 record IDs encode disjoint domains for suffix, direct,
binary, and listwise requests. Arm names are absent from causal record IDs.
The four suffix arms share exact IDs, row order, seed keys, and stage-one seeds;
the three binary arms do likewise. Materialized and shuffled prompt token
multisets are checked pairwise after the exact chat template is rendered.

Every mechanics stage, including preparation and standalone analysis, requires
the repository's `.venv-vllm` interpreter. A stdlib-only bootstrap validates
all 189 exact/direct-URL distributions in `requirements-vllm.lock.txt` before
importing NumPy, Transformers, or any experiment-local module. It rejects
duplicate `--stage`/`--lock` spellings so later argument parsing cannot choose
a different stage or lock. For prepare/run/analyze it installs an audit hook
that denies every `benchmarks/` open and every procedural-data open except the
published construction manifest, mechanics public rows, and mechanics audit
rows.

The fixed runner uses only `Qwen/Qwen3.5-4B` at revision
`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bfloat16, vLLM
`0.24.0+cu129`, one device, synchronous scheduling, no adapter, no prefix
caching, Mamba cache mode `none`, `max_num_seqs=64`, and explicit full-decode
CUDA-graph sizes `[1,2,4,8,16,32,64]`. Imported-runner Mamba clamping is not
accepted. A live preflight recomputes every rendered token hash and proves
block-rounded KV capacity for every invocation before the first request. It
also freezes the full pre-import package inventory, Python/platform, GPU and
driver, CUDA toolkit, and git runtime. Every resumed invocation must match that
runtime exactly, preventing causal arms from being mixed across machines or
software states.

## Fail-closed preparation and transactions

Preparation reads explicit allowlisted paths and reconstructs every request,
surface fold, random rank, file table, and receipt before accepting stored
artifacts. It binds the published construction manifest, all source and test
hashes, exact prompt text/token hashes, row IDs, seed keys, and expected request
counts. The receipt declares zero model calls and empty hidden/future/benchmark
read lists. It directly matches the published construction receipt's plain and
leading-space alias IDs, think-boundary IDs, thinking/no-thinking prompt suffix
IDs, forced-close IDs, and exact 52-row echo prompt/token hashes.

The authoritative design allowlist—configuration, construction inputs,
preregistration, design review, protocol/interpreter/runner, and their tests—is
compared byte-for-byte with `git show` at design commit `ba08a732...`; a later
implementation lock cannot self-bless design drift. `README.md` is deliberately
excluded from both the authoritative design set and implementation critical
set because it is mutable status/outcome prose. The preregistration, design
review, configuration, and executable protocol are the authoritative science.

Live execution requires an exact critical-file implementation lock whose
implementation commit and current `HEAD` are ancestors of `origin/main`.
Current bytes must equal both the lock hashes and implementation-commit blobs.
The lock's current bytes must also equal its committed `HEAD` blob, independent
of git status flags such as skip-worktree. Only mechanics raw/scored/summary
paths may be dirty. One OS-level exclusive lock covers generation, recovery,
authentication, scoring, and standalone analysis.

Each invocation writes `STARTED`, raw JSONL, runner metadata, and then
`COMPLETE`. A crash after both raw data and metadata are durable is recovered
only by authenticating them and writing the missing completion receipt; it is
never resampled. Every other partial state is ambiguous and terminal. Complete
transactions are reconstructed and authenticated before they can be skipped.
Each completion binds its start receipt, raw and metadata hashes, live
preflight, implementation lock, and prior completion hash. Raw, scored, and
summary artifacts are immutable after creation; the completed chain is
committed and pushed immediately after the run to provide the external anchor.

## Raw authentication

Offline analysis re-authenticates the stored live preflight rather than merely
checking that it exists. For every invocation it verifies:

- exact model, revision, runner hash, engine config and engine arguments;
- exact CUDA-graph resolution and raw-logprob mode;
- sampling config, every pinned distribution, full preflight runtime identity,
  V1 multiprocessing mode, all termination/prompt-channel IDs, and restored
  global RNG receipt;
- request IDs/order, metadata, rendered prompt hash/count/channel, and absence
  of prompt logprobs;
- deterministic stage seeds, one output, text/token equality, terminal
  trimming, natural versus forced closure, injected close IDs, and all logical
  and sampled-token accounting; and
- aggregate runner counts recomputed directly from raw rows.

Ranking accepts only the first stage-one completion-logprob position. It
requires exactly one sampled token, no second stage, and every requested plain
token ID exactly once and finite. Binary scores are oriented LIVE-minus-DEAD
after an explicit float32 cast. Listwise scores use all plain A-X IDs. Across
the model rankers this authenticates 2,304 candidate scores backed by exactly
4,032 requested raw logprob values. No grammar, mask, bias, forced token, prompt
logprob, cumulative logprob, or decoded-token substitute is used.

## Frozen scientific interpretations

Mechanics A uses raw visible-only execution for its registered capability gate
and separately reports full public-probe selector eligibility. Answer-cap
contact is conservative: an answer token count at the cap or any final length
finish counts; a forced thinking close alone does not. Parameter-class support
uses generated suffix operations only, requires at least one successful row in
each class, and allows a mixed suffix to satisfy both.

All A rates are converted to exact integer boundaries before evaluation:
47/52 parses, at most 2/52 cap contacts, 22/24 direct parses, at most 1/24
direct cap contact, 47/52 echo execution, 9/24 materialized task successes, and
four-task gains over both name-only and shuffled. ABI failures return
`MECHANICS_INTERFACE_INVALID` before scientific failure is considered.

Mechanics B implements the exact 35-feature float64 leave-one-task-out surface
control locally: 24 candidate one-hots plus 11 registered relation features,
training-fold-only population z-scoring, zero-variance scale one, class weights
`n/(2*n_class)`, balanced weighted mean logistic loss, lambda-one L2 on
coefficients only, and an unpenalized intercept. Aligned L1 is the per-row sum
of absolute coordinate differences, averaged over the eight rows (zero for an
unequal-length row), not coordinate MAE. A deterministic damped Newton
solver fails closed on nonconvergence. Random and rank-tie salts are fixed.
Equal-task recall/hit/reciprocal-rank metrics prohibit pooled AUROC. Point-gate
comparisons accept mathematical equality within `1e-12` solely to prevent
binary floating-point representations of exact preregistered boundaries from
turning equality into failure.

Mechanics B failure seals only the future independently generated top-four
secondary. Mechanics A success alone licenses the primary all-24 qualification
implementation. Mechanics outputs are never reused as top-four outputs, and no
cost or capability claim is made at this stage.

## Adversarial findings and resolutions

Two independent current-code adversaries initially returned `NO-FREEZE`. Their
blockers and the implemented resolutions were:

- coordinate MAE had weakened the registered surface L1 control; it is now a
  per-row L1 sum with a nonzero length-three regression fixture;
- resumed arms could mix runtimes; the live preflight and every metadata file
  now share one exact full runtime fingerprint;
- the environment-lock SHA alone did not prove installed packages; all 189
  pins, including the direct-URL vLLM wheel, are verified before local imports
  and again from runner metadata;
- implementation hashes could self-bless design drift; authoritative files now
  equal their design-commit blobs independently;
- local code ran before the lock and sealed-input boundary; a stdlib bootstrap
  verifies import blobs and installs the deny hook first;
- tokenizer state was only partially pinned; all published IDs plus the exact
  echo receipt now match the construction lock;
- stored seed flags could forge natural versus forced continuation; branch
  identity, retained tokens, and forced-close state are derived from stage-one
  tokens and finish reason;
- analysis could mutate outside the generation lock; every mutating path now
  shares the same exclusive lock; and
- a mutable lock file could hide behind git index flags; its exact current bytes
  now equal `HEAD`.

## Model-free verification performed

The pure mechanics suite currently has 45 tests, and the complete experiment
suite has 71. It covers exact request
inventory, public-live and all-candidate coverage, cross-arm IDs/seeds, disjoint
seed domains, sampling/engine settings, hidden-data exclusion, hand-computed
surface features, all 24 LOOTO folds, training-only scaling, row-order
invariance, deterministic realized random ranks, raw-logprob orientation and
malformed cases, ranking coverage, every A and B gate at equality and just
below, public echo execution, conservative cap contact, ambiguous `STARTED`
transactions, receipt-only recovery, lock contention, committed-lock byte
identity, strict bootstrap parsing, wrong-interpreter rejection, full package
pinning, runtime drift, the expected clean-to-live-artifact Git-dirty
transition, authoritative hybrid-cache flooring and 11-block geometry, the
703-block false-pass boundary, persisted per-arm block corruption,
validation-before-write behavior, raw prompt/retained-token/branch/truncation/
count corruption, and unsafe raw inventories. Separate model-free subprocess
checks
validated all 189 installed pins and proved that mechanics-gold and benchmark
opens are denied before any scientific stage. Deterministic preparation
reconstructed all 1,984 requests, all 4,032 targeted raw values, 24 converged
LOOTO folds, exact token-multiset controls, and exact construction-tokenizer
receipts with zero model loads and calls.

Final authorization remains contingent on resolving every independent audit
finding, rerunning deterministic preparation byte-identically, passing the full
experiment and repository checks, pushing the implementation commit, then
publishing and pushing a separate implementation-lock receipt. Until that
sequence completes, further model execution remains prohibited.

## Preflight-only attempt and v2 repair

Attempt 1 initialized the engine but aborted before `_generate`, leaving no
invocation `STARTED` receipt and no sampled output. The exact incident and its
append-only recovery are recorded in `mechanics_preflight_incident.md`. The v2
repair replaces the invalid inverse-float check with pinned vLLM's authoritative
hybrid-cache identities, uses a conservative 11-block reservation per active
sequence, and validates in memory before writing a PASS receipt. It preserves
the old lock, preoutcome receipt, and failed preflight unchanged while moving
all active retry artifacts to versioned paths.
