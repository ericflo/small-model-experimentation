# Qwen3.5-4B Tokenizer-EOS Answer Commit Factorial

**Status:** in-progress · since 2026-07-14 · adversarial `PASS_DESIGN`; fresh construction and preoutcome receipt pass; implementation review next; no model call

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
.venv-vllm/bin/python -I experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run_calibration.py --stage lock
.venv-vllm/bin/python -I experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run_calibration.py --stage run
.venv-vllm/bin/python -I experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run_calibration.py --stage analyze
```

The isolated-mode guard executes immediately after importing Python's built-in
`sys` module and before any shadowable import. Invocations without `-I`, with a
missing stage, or with a malformed stage fail before repository-local imports.

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

## Interpretation

The smoke proves that the proposed measurement is precise and falsifiable. It
does not show that live vLLM emits the expected first-stop receipts on fresh
rows, that a tokenizer-EOS arm qualifies, or that materialized residuals improve
capability. The branch stops permanently if no fresh tokenizer-EOS interface
qualifies.

Any conditional mechanics pass is scoped to a 24-task contamination-free
large-effect pilot; it is not a confirmatory or general deployability claim.

## Knowledgebase Update

- Program evidence updated: predecessor terminal result only.
- Program backlog updated: this fresh successor is now active.
- Claim ledger updated: no; no result.

## Artifacts

- `src/protocol.py`
- `scripts/run.py`
- `configs/default.yaml`
- `runs/smoke/summary.json`
- `reports/preregistration.md`
- `reports/design_review.md`
- `reports/artifact_manifest.yaml`
