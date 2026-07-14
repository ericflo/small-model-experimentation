# Qwen3.5-4B Tokenizer-EOS Answer Commit Factorial

**Status:** finished

Calibration qualified the tokenizer-EOS-only interface; mechanics transport
passed 24/24, but visible selection ended in terminal instrument failure after
all five transactions. Residual capability remains unadjudicated.

This fresh successor tests whether the prior strict answer-seam failure was
caused by waiting past Qwen3.5's tokenizer chat-end token. It registers the
first tokenizer EOS only during answer generation, preserves every pre-commit
byte under the strict grammar, and compares it with the existing HF-model-EOS
boundary on new calibration and residual-mechanics identities.

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: can a token-native output boundary expose a deployable
  structured compiler interface without supplying answer identity?
- Closest result-bearing predecessor:
  `qwen35_4b_materialized_residual_answer_seam_factorial`.
- Prior evidence: the predecessor was 0/48 strict in every registered arm, but
  suffix-only removal gave 48/48 frozen-parser exactness in both paired
  no-think cells; thinking remained 38/48 and 24/48 because of extra closes.

## Question

On fresh known-answer rows, does stopping the answer stage at the first
tokenizer EOS (248046) create a strict exact program interface that the matched
HF-model-EOS (248044) boundary does not—and, only if it qualifies, does that
interface permit a disjoint materialized-residual capability test?

## Hypothesis

Qwen3.5 naturally commits short structured answers with `<|im_end|>`. The
predecessor treated that token and the following newline as answer content
because it waited for `<|endoftext|>`. Registering first-248046 answer-stage
stopping should lift no-think exact/parse from the predecessor's 0/48 to at
least 44/48 without changing prompts, answer identity, or thought generation.
This is an interface hypothesis, not yet a capability hypothesis.

## Setup

- Model: only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Backend: pinned repository vLLM for every paired arm and matched-compute
  baseline.
- Dataset/task source: fresh procedural exact-depth-three list transforms;
  never `benchmarks/`.
- Frozen fresh splits: 48 known-answer calibration tasks and 24 disjoint
  residual-mechanics tasks, with new functions, task IDs, record IDs, token
  sequences, and seed domain.
- Factorial: answer boundary {first tokenizer EOS, HF model EOS control} x
  thinking {off, forced close at 512} x answer prefix {freeform, `PROGRAM:`}.
- All 192 boundary pairs must share prompts, answer seeds, persisted thought
  IDs, request adjacency, and sampled token prefix through the earliest stop or
  cap; one mismatch terminates the experiment before qualification.
- Every task has one persisted thought transaction reused by all four thinking
  answer cells. Natural post-close answer content is discarded and cannot
  bypass the answer-stage comparison.
- Answer-stage tokenizer EOS never applies to thought generation.
- Calibration gates: >=44/48 strict exact echoes, >=44/48 parses, <=2/48
  answer-cap contacts, plus >=22/24 exact/parse and <=1/24 cap contact in each
  arity.
- Parse is membership in a frozen token-ID grammar for any A-X program of the
  registered arity; exact is equality to the known answer. A stop on sampled
  token 24 still counts as a cap contact.
- Eligible winner: first qualifying tokenizer-EOS arm in the frozen no-think
  `PROGRAM:`, no-think freeform, think `PROGRAM:`, think freeform priority.
- Boundary controls: matched HF EOS, first-stop uniqueness, early/interior/
  missing tokenizer EOS, extra pre-commit bytes, and exact stop/token/cost
  receipt authentication.
- Conditional mechanics is fully frozen at 24 tasks, 24 candidates per suffix
  arm, and a 96-row-per-task direct ceiling. Controls are materialized state,
  name-only, shuffled materialized state/target, candidate-blind direct
  samples, exhaustive CPU ceiling, and taskwise sampled/logical-token matched-
  compute prefixes.
- Every suffix is bound to its semantic candidate first operation before any
  deduplication or scoring, yielding the same canonical full three-operation
  proposal type used by direct sampling.
- Primary capability metric if opened: hidden exact accuracy of a frozen
  visible-only selector versus all structured controls and matched-compute
  direct sampling.
- Hidden-label boundary: mechanics remains sealed until a committed-green
  calibration decision and second winner-bound lock; hidden labels remain
  sealed until a committed-green visible-selection receipt.

The prefix cells share tasks and seed derivation and are paired conditions, not
independent replications. Only tokenizer-pass/HF-fail in the selected matched
cell supports a causal termination-boundary claim. Dual qualification within
one matched thinking/prefix pair is impossible under the authenticated-prefix,
exactness, and cap gates: shared exact-cap overlap is <=2 globally and <=1 per
arity, yielding `44+44-2>48` and `22+22-1>24`. Any observed dual qualification
terminates as a scoring invariant violation.

## Run

Model-free smoke:

```bash
python3 -B experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run.py --smoke
```

Fresh construction:

```bash
python3 -B experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/construct.py
```

Full:

```bash
# Still sealed pending implementation review and the published-green lock.
# Once those gates pass, these are the only supported entry points:
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/calibration_launcher --stage lock
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/calibration_launcher --stage run
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/calibration_launcher --stage analyze
```

Winner-bound mechanics (still unauthorized until the implementation review,
review receipt, and mechanics lock have each been committed, pushed, and
green):

```bash
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher --stage lock
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher --stage run
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher --stage analyze-visible
# Only after visible_selection.json is committed, pushed, and green:
experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher --stage score-hidden
```

`calibration_launcher` is a 12.6-KiB static x86-64 ELF with no dynamic
interpreter. It retains a waiting static parent, opens its exact executable on
inherited descriptor 198, discards the inherited environment, and directly
`execve`s the pinned Python interpreter with `-I -B` in a child protected by a
parent-death signal. The Python bootstrap requires the live parent executable,
the inherited descriptor, and the tracked launcher path to be the same stable
inode with the reviewed SHA-256. This kernel-carried provenance cannot be
forged by an environment marker. Its source is
`scripts/calibration_launcher.S`; its exact reproducible build command is:

```bash
/usr/bin/gcc -nostdlib -static -no-pie -s -Wl,--build-id=none \
  -o experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/calibration_launcher \
  experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/calibration_launcher.S
```

The mechanics launcher applies the same kernel-carried provenance design to
the winner-bound runner. Its exact reproducible build command is:

```bash
/usr/bin/gcc -nostdlib -static -no-pie -s -Wl,--build-id=none \
  -o experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher \
  experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/mechanics_launcher.S
```

Direct Python invocation—even with a caller-supplied marker and an open copy of
the launcher on descriptor 198—fails because its live parent is not the static
launcher. Missing/malformed stages, non-isolated Python, altered launcher
bytes, and dynamic-loader variables also fail before local imports.

## Results

The model-free protocol smoke passes with zero model calls. It accepts only a
clean expected answer followed by one terminal registered stop token. The
matched HF boundary preserves tokenizer EOS/newline as strict answer content
and therefore fails exactness. Early stops, interior-plus-terminal stops,
missing claimed stops, and extra pre-commit bytes all fail exactness or their
registered contract as appropriate. A unique final early stop is authenticated
but scored; an exact-cap length trace is authenticated with all tokens retained
as content.

After six prospective holds, global independent rereview returned
`PASS_DESIGN` for exact pushed/green commit `abd2ffcd`. No model or capability
result exists; live execution remains sealed behind implementation release
review and a committed-green lock.

Model-free construction then passed with 48 calibration and 24 mechanics tasks,
the frozen 8/8/4/4 strata in each calibration arity and mechanics, every A-X
alias once per answer position, zero overlap with all 72 predecessor public
fingerprints, zero request/seed/prompt/derived-seed overlap, and a distinct
transport namespace. Mechanics gold exists only as tracked AES-256-GCM
ciphertext; its key is local and ignored. Model calls and sampled outputs remain
zero.

Before any model call, the prospective runner was security-hardened to pin
child executables/environment and preserve isolation across recovery re-exec.
The tokenizer receipt's sole runner binding was correspondingly refreshed from
`cbbfae3e...` to `4ce61e64...`; its grammar, prompts, termination IDs,
freshness inventories, and zero-call declarations were unchanged.

A later clean implementation rereview held live calls because an in-Python
`LD_PRELOAD` rejection occurs after interpreter startup and because several
zero counters still accepted JSON Boolean aliases. The prospective static
launcher now sanitizes before Python exists, and exact integer predicates plus
typed-canonical engine comparisons reject those aliases. These are model-free
repairs; no calibration request or sampled output has yet occurred.

The next clean rereview held again because the first static-launch repair used
a caller-controlled environment marker to assert entry. The parent-plus-open-
executable proof above replaces that assertion with kernel process/file state
that remains valid across the sanctioned Mamba recovery `execve`. The repair
is still model-free; live authorization remains absent.

The seventh clean review then returned `PASS_IMPLEMENTATION` for exact
pushed-green commit `d70756122bc768e82fa4d77a61e05522ef5bca79`: 95/95
permitted tests passed, the launcher rebuilt byte-identically, all protected
read inventories were empty, and model/GPU calls remained zero. The PASS does
not itself authorize generation; its canonical hash-bound receipt and the
subsequent implementation lock must each be committed, pushed, and green.

The independently committed receipt and implementation lock both passed their
two required workflows. The sealed live calibration then completed 48 shared-
thought requests plus 384 paired answer requests. All 192 boundary pairs and
the five-invocation transaction chain authenticated. Results were:

| Cell | Exact | Parse | Cap contacts | Arity-2 exact | Arity-3 exact |
| --- | ---: | ---: | ---: | ---: | ---: |
| tokenizer EOS · no-think · `PROGRAM:` | 48/48 | 48/48 | 0 | 24/24 | 24/24 |
| tokenizer EOS · no-think · freeform | 48/48 | 48/48 | 0 | 24/24 | 24/24 |
| tokenizer EOS · think512 · `PROGRAM:` | 38/48 | 38/48 | 0 | 18/24 | 20/24 |
| tokenizer EOS · think512 · freeform | 30/48 | 30/48 | 16 | 15/24 | 15/24 |
| HF model EOS · no-think · `PROGRAM:` | 0/48 | 0/48 | 0 | 0/24 | 0/24 |
| HF model EOS · no-think · freeform | 0/48 | 0/48 | 0 | 0/24 | 0/24 |
| HF model EOS · think512 · `PROGRAM:` | 0/48 | 0/48 | 0 | 0/24 | 0/24 |
| HF model EOS · think512 · freeform | 0/48 | 0/48 | 16 | 0/24 | 0/24 |

The frozen decision is `TOKENIZER_EOS_ONLY_INTERFACE_QUALIFIED`; the winner is
`tokenizer_eos_no_think_program_slot`, and its matched HF control is
`hf_model_eos_no_think_program_slot`. This opens the preregistered conditional
mechanics branch only after this calibration result and a second winner-bound
lock are committed, pushed, and green.

The first exact-commit conditional-mechanics review returned
`HOLD_IMPLEMENTATION` for `fd06b5053c9327c61775d07061c9a84e070cdcb6`.
It found four live-stage deadlocks (an incomplete path-audit allowlist, an
invalid calibration-verifier call, and two Python-tuple/JSON-list durable
comparison failures), partial type/schema authentication, a missing registered
direct-pool-exhaustion terminal, and incomplete resource receipts. The
prospective repairs add exact calibration/review support reads, a scoped and
tested immutable-verifier adapter, JSON-domain receipts, exact typed
engine/prompt/seed/token/terminal/cost authentication, omission of unrequested
likelihood diagnostics, a durable `DIRECT_RESOURCE_MATCH_POOL_EXHAUSTED`
receipt, and explicit overshoot/row-ID inventories. The model-free suite now
passes 124/124. These repairs do not authorize mechanics; a fresh pushed-green
exact-commit adversarial review and second lock are still required.

The round-two review of pushed-green commit
`3d2f051203c56456fad716e20950b55d5717afd5` verified all seven earlier repairs
and held on three new exact-type replay failures: the recorded mechanics
preflight authenticated only a runtime subset, visible authorization inherited
Python's Boolean/integer equality, and the calibration-era generic transaction
envelopes had the same alias at STARTED and bundle schema boundaries. The
prospective repair now reconstructs the entire clean preflight from the live
runner and exact-compares every JSON-domain field after authenticating recorded
CI. Visible authorization also uses recursive exact-type comparison. Because
the generic transaction source is an immutable calibration-critical file,
mechanics routes through a new additive exact-typed transaction layer rather
than altering the calibration anchor. Durable hostile tests cover all ten
preflight mutations, visible `true` versus `1`, STARTED/bundle aliases,
receipt aliases, and registered row-count aliases. The full model-free suite
passes 130/130; mechanics calls and protected reads remain zero. A third fresh
pushed-green exact-commit review is still required.

That round-three review of pushed-green `9b527cbfd6934f579b0ebcb07cb4b695c370798b`
returned `HOLD_IMPLEMENTATION` despite confirming all round-two closures. It
found that production tuple-valued `logprob_token_ids` would mismatch the
JSON-native STARTED sampling after the first call, that hidden scoring reread
the visible receipt after authorization, and that the low-level transaction
primitive authenticated only a predecessor's state before a fresh successor
call. The repairs normalize the complete fresh generated bundle before its
first validation/write, pass the exact in-memory authorized visible object
directly into scoring, and fully authenticate plus recheck the predecessor
chain inside the primitive itself. Tests now use the actual frozen mechanics
sampling plan, assert no second visible read, and corrupt a predecessor before
a fresh successor. The full model-free suite passes 134/134; mechanics calls
and protected reads remain zero. A fourth pushed-green exact-commit review is
required before any lock or generation.

Round-four probes confirmed all production-shape and static-corruption repairs,
then held on two narrower concurrent-mutation windows: a predecessor could
change inside generation or promotion after the pre-call recheck, and visible
authorization compared `HEAD:path` before separately resolving the commit
recorded in its receipt. The primitive now rechecks the authenticated
predecessor immediately after generation and again before returning from every
promotion/recovery path. Visible authorization resolves one commit exactly
once and uses that immutable commit ID for its blob comparison and receipt.
New callbacks mutate the predecessor during generation and at COMPLETE
publication; both are detected before bundle publication or successful return.
The full model-free suite passes 136/136 with zero mechanics calls or protected
reads. A fifth pushed-green exact-commit review is required.

Round five confirmed every earlier regression family but found that the
pre-return predecessor recheck covered only the terminal COMPLETE file. A
concurrent change to an already-authenticated predecessor STARTED, bundle, or
GENERATED artifact could therefore escape the primitive's successful return
until final chain authentication. The primitive now reruns exact authentication
over the entire predecessor prefix at each recheck and compares the resulting
receipt with the original authenticated snapshot. New regressions change each
non-terminal predecessor artifact during generation, publication, or recovery;
all fail closed without a successful successor return or recovery resample.
The model-free suite passes 139/139 with zero mechanics model/GPU calls or
protected reads. A sixth pushed-green exact-commit review is required.

Round six returned `PASS_IMPLEMENTATION` for exact pushed-green commit
`df096d330f09847ce844af6255b349b4f707f464`. It passed 139/139 tests, rebuilt
the static launcher byte-identically, rejected all full-prefix changes across
pre-call, callback, publication, and recovery checks, and reconfirmed the
production tuple path, hidden object/commit binding, exact typed gates,
resource receipts, immutable calibration files, and bootstrap routing. All
protected-read arrays and model/GPU counters were zero. The hash-bound review
report and canonical receipt must now be committed, pushed, and green before
the mechanics lock can be published; this PASS alone does not authorize a
model call.

After that receipt was published green, the first lock-only invocation failed
closed before creating a lock: calibration decision recomputation retained 32
integer dictionary keys while canonical JSON represented those object keys as
strings. There were zero actual tuple/list mismatches. The lock compared across
that serialization boundary without first entering the JSON domain. The
prospective mechanics
verifier now JSON-normalizes the recomputation and then applies recursive exact-
typed comparison, preserving rejection of Boolean/integer aliases. A direct
model-free recomputation now authenticates the frozen
`TOKENIZER_EOS_ONLY_INTERFACE_QUALIFIED` decision and winner. No mechanics data,
model, or GPU call occurred. Because this changes reviewed mechanics code, a
fresh exact-commit implementation review and replacement receipt are required
before retrying the lock. The model-free suite passes 140/140.

Round seven verified that real decision repair, rejecting 20/20 typed aliases
and four semantic mutations, and completed a read-only prospective lock build.
It nevertheless held on the next exact-type boundary: durable sampling-plan
validation used ordinary Python equality, accepting five `n=1` to `true`
aliases and 25 Boolean-to-integer aliases across the five arms. Sampling-plan
validation now uses recursive exact JSON equality. The real-plan regression
mutates both directions for all 30 fields and requires every case to fail.
No mechanics payload, model, or GPU call occurred; another exact-commit review
and replacement receipt remain required.

Round eight returned `PASS_IMPLEMENTATION` for exact pushed-green commit
`3e7b650a90ff1d65fe371552354895756efcf728`. It passed 140/140 tests; rejected
all 58 real-plan type/schema cases and all five semantic seed mutations;
reauthenticated the real calibration decision with the correct 32 key
conversions; and completed read-only lock build/validation over 29/29 current
critical files. Full-prefix, production durability, hidden binding, preflight,
resource, calibration, routing, launcher, and static checks also passed. All
mechanics/protected read inventories and model/GPU counters remained zero. The
replacement hash-bound receipt must be committed, pushed, and green before the
lock-only stage is retried.

After that replacement receipt was published green, the next lock-only attempt
failed closed while hashing reviewed critical evidence: the pre-import path
audit allowed all 22 runtime files and 11 support files but omitted the seven
reviewed mechanics test files in the 29-file critical inventory. The lock was
not written, and no mechanics payload, model, or GPU call occurred. A separate
exact seven-file critical-test allowlist now permits only those review inputs;
it does not widen runtime imports, prepared/mechanics data, or hidden access.
The bootstrap test proves that this tuple is exactly the critical-minus-runtime
set and is actually consumed by the path audit. Because bootstrap-reviewed code
changed, another exact-commit review and replacement receipt are required. The
model-free suite passes 141/141.

Round nine returned `PASS_IMPLEMENTATION` for exact pushed-green commit
`c0075a019fd0f202c3b0e6cf0be5528e08c61649`. It passed 141/141 tests; proved
the exact seven-file critical-test inventory; hashed all 29 critical files
through the active audit while denying neighboring files; preserved the 22/11
runtime/support inventories; and passed every prior typed, semantic,
full-prefix, durability, hidden-binding, calibration, launcher, and resource
regression. The prospective lock build/validation was read-only, and mechanics
payload reads, model/GPU calls, and protected-read inventories remained zero.
The replacement hash-bound receipt must be committed, pushed, and green before
the lock-only stage is retried.

The replacement receipt was committed, rebased, pushed, and green. An initial
lock retry then failed closed because the shared-main rebase advanced HEAD to a
commit whose `Validate Repository` workflow was still running; it wrote no
lock and made no mechanics/model/GPU call. After both workflows for that exact
HEAD passed, lock-only publication succeeded. The canonical implementation
lock has SHA-256
`d42e2db4b589e470f42d963b19e01a8b880fa7858a40b10966150c775c3d925b`,
binds all 29 critical and 22 runtime files, and records zero pre-lock
generation requests, sampled outputs, and protected reads. Live mechanics
remains unauthorized until this lock is itself committed, pushed, and green.

The lock was committed and both exact-commit workflows passed before live
mechanics. The selected interface then transported perfectly: 24/24 exact
echoes, 24/24 parses, and zero answer-cap contacts, balanced 12/12 in each
arity. All five durable transactions completed and preserved 4,056 sampled
outputs. Visible analysis nevertheless failed before selection because it
recomputed the already-recorded transport decision through the initial
authorization path, which correctly requires every later invocation to be
absent; at replay time `direct` and all suffix invocations were already
complete. No visible selection, resource decision, hidden score, hidden read,
or benchmark read occurred. This is terminal instrument failure, not evidence
for or against residual capability. The generated result is preserved, and a
fresh-identity successor must separate initial transport authorization from
post-chain replay authentication under a new review and lock.

## Interpretation

The fresh result isolates a real termination-boundary effect: with identical
paired prompts/seeds and authenticated shared prefixes, stopping on tokenizer
EOS yields 48/48 strict structured answers while waiting for HF model EOS yields
0/48 because tokenizer EOS/newline remains pre-commit content. This is an
interface qualification, not yet a capability gain. Thinking is actively worse
at this interface (38/48 and 30/48, including 16 freeform cap contacts), so the
frozen no-think `PROGRAM:` winner is the only branch that advances.

Any conditional mechanics pass is scoped to a 24-task contamination-free
large-effect pilot; it is not a confirmatory or general deployability claim.

## Knowledgebase Update

- Program evidence, backlog, scorecard, and shared synthesis now record the
  committed-green calibration result and advance only the frozen no-think
  `PROGRAM:` winner.
- Claim-ledger allocation is deferred: the repository's knowledgebase protocol
  forbids adding or promoting claims until the outstanding adversarial re-grade
  checklist is processed. No claim ID is consumed, avoiding a collision while
  preserving the result at program level.

## Artifacts

- `src/protocol.py`
- `scripts/run.py`
- `configs/default.yaml`
- `runs/smoke/summary.json`
- `reports/preregistration.md`
- `reports/design_review.md`
- `reports/artifact_manifest.yaml`
- `runs/calibration/implementation_lock.json`
- `runs/calibration/live_preflight.json`
- `runs/calibration/decision.json`
- `src/mechanics_runtime.py`
- `src/mechanics_stage.py`
- `src/mechanics_lock.py`
- `scripts/run_mechanics.py`
- `scripts/mechanics_launcher`
