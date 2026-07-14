# Qwen3.5-4B Tokenizer-EOS Answer Commit Factorial

**Status:** in-progress · since 2026-07-14 · first adversarial design review held; prospective remediation frozen; no construction or model call

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
- Eligible winner: first qualifying tokenizer-EOS arm in the frozen no-think
  `PROGRAM:`, no-think freeform, think `PROGRAM:`, think freeform priority.
- Boundary controls: matched HF EOS, first-stop uniqueness, early/interior/
  missing tokenizer EOS, extra pre-commit bytes, and exact stop/token/cost
  receipt authentication.
- Conditional mechanics is fully frozen at 24 tasks, 24 candidates per suffix
  arm, and a 96-row-per-task direct ceiling. Controls are materialized state,
  name-only, shuffled
  materialized state/target, candidate-blind direct samples, exhaustive CPU
  ceiling, and taskwise sampled/logical-token matched-compute prefixes.
- Primary capability metric if opened: hidden exact accuracy of a frozen
  visible-only selector versus all structured controls and matched-compute
  direct sampling.
- Hidden-label boundary: mechanics remains sealed until a committed-green
  calibration decision and second winner-bound lock; hidden labels remain
  sealed until a committed-green visible-selection receipt.

The prefix cells share tasks and seed derivation and are paired conditions, not
independent replications. Only tokenizer-pass/HF-fail in the selected matched
cell supports a causal termination-boundary claim; a both-pass result can
authorize interface use but not boundary causality.

## Run

Model-free smoke:

```bash
python3 -B experiments/qwen35_4b_tokenizer_eos_answer_commit_factorial/scripts/run.py --smoke
```

Full:

```bash
sealed pending fresh construction, adversarial design review, implementation lock, and green publication boundary
```

## Results

The model-free protocol smoke passes with zero model calls. It accepts only a
clean expected answer followed by one terminal registered stop token. The
matched HF boundary preserves tokenizer EOS/newline as strict answer content
and therefore fails exactness. Early stops, interior-plus-terminal stops,
missing stops, and extra pre-commit bytes all fail their registered contract.

The first independent review returned `HOLD_DESIGN`; its four blockers are now
addressed prospectively in the config and preregistration. The remediation has
not yet passed independent rereview. No model or capability result exists.

## Interpretation

The smoke proves that the proposed measurement is precise and falsifiable. It
does not show that live vLLM emits the expected first-stop receipts on fresh
rows, that a tokenizer-EOS arm qualifies, or that materialized residuals improve
capability. The branch stops permanently if no fresh tokenizer-EOS interface
qualifies.

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
