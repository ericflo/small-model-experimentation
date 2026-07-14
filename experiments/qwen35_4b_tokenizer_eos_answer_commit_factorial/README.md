# Qwen3.5-4B Tokenizer-EOS Answer Commit Factorial

**Status:** in-progress · since 2026-07-14 · `PASS_DESIGN` + exact-commit `PASS_IMPLEMENTATION`; fresh calibration: tokenizer-EOS-only interface qualified; conditional mechanics pending winner-bound lock

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

- Program evidence update pending the committed-green calibration artifact.
- Program backlog: advance only the frozen no-think `PROGRAM:` winner.
- Claim ledger: add a narrowly scoped tokenizer-EOS interface claim after the
  result commit is published and synchronized to avoid claim-number collision.

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
