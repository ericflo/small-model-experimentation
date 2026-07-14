# Qwen3.5-4B Materialized Residual Answer-Seam Factorial

**Status:** finished

Terminal 2026-07-14: `NO_VALID_RESIDUAL_ANSWER_SEAM`; mechanics remained sealed.

The registered answer interface failed before residual mechanics opened. All
240 calibration outputs authenticated, but every arm scored 0/48 strict parses
and exact echoes. After removing only the terminal `<|im_end|>` plus newline,
the two no-think arms pass the frozen exact parser on 48/48 rows. Thinking arms
remain partly invalid because some emit another `</think>` boundary. This is
evidence for a fresh native answer-stage commit-boundary experiment, not
permission to repair this result in place.

## Research Program

- Program: `structured_execution_and_compilers`
- Program question: can externalized intermediate state become a deployable
  compositional interface rather than only local semantic routing?
- Prior anchors: `qwen35_4b_materialized_residual_sibling_search_fresh_replication`,
  `qwen35_4b_commit_slot_semantic_power_replication`, and
  `qwen35_4b_early_text_hypothesis_forking`.

## Question

Can a separately calibrated, autonomously scored short answer seam make
materialized residual generation measurable, and if so does it beat name-only,
token-preserving shuffled, and matched-compute direct sampling controls?

## Hypothesis

The prior model could copy a structured echo on 20/24 non-cap rows but every
thought hit its cap. A complete 2x2 crossing think@512/no-think with
freeform/literal-`PROGRAM:` prefill can isolate answer syntax from reasoning
policy without supplying answer identity. A separately calibrated policy may
expose residual completion that the invalid free-form interface hid.

## Setup

- Model: only `Qwen/Qwen3.5-4B` revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, bf16.
- Dataset/task source: fresh procedural exact-depth-three list transforms;
  never `benchmarks/`.
- Split: 48 known-answer interface-calibration tasks and 24 disjoint mechanics
  tasks, with new task/request/seed domains.
- Interface arms: `think512_freeform`, `think512_program_slot`,
  `no_think_freeform`, and `no_think_program_slot`; all answer aliases are
  sampled autonomously with the same 24-token tail cap.
- Baseline: taskwise matched-compute direct full-program sampling on the same
  backend and selected interface budget.
- Controls: name-only siblings, task-hash shuffled materialized states/targets,
  exact echo, candidate-blind direct sampling, and exhaustive CPU ceiling.
- Primary metric: hidden exact accuracy of the pre-hidden visible-only selector,
  gated behind the interface calibration.
- Oracle-only diagnostics: all-sibling hidden-correct proposal coverage, exact
  candidate viability, and hidden program success; none may affect interface
  choice, prompts, budgets, or selected IDs.
- Calibration gates: >=44/48 exact echoes, >=44/48 parses, <=2/48 answer-cap
  contacts, plus >=22/24 exact/parse and <=1/24 cap contacts in each arity.
- Winner: first qualifier in the fixed least-departure priority, never the
  best observed metric.
- Hidden-label boundary: mechanics remains inaccessible until a committed
  winner receipt and second lock; hidden scoring remains inaccessible until a
  committed visible-selection receipt. Qualification/confirmation and all
  benchmark content remain unread.

## Run

Smoke:

```bash
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run.py --smoke
```

Fresh construction and append-only v2 smoke:

```bash
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/construct.py
python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run.py --design-smoke
```

Full:

```bash
.venv-vllm/bin/python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_calibration.py --stage lock
.venv-vllm/bin/python experiments/qwen35_4b_materialized_residual_answer_seam_factorial/scripts/run_calibration.py --stage run
```

## Results

The exact reviewed implementation received `PASS_RELEASE_LIVE_CALLS`; its
calibration lock was committed and passed both workflows before the live run.
The five durable transactions then authenticated 240/240 outputs with exact
shared-thought and answer-seed pairing. The fixed gate returned
`NO_VALID_RESIDUAL_ANSWER_SEAM`:

- all four arms: 0/48 strict parses and 0/48 exact echoes;
- `think512_freeform`: 48/48 thought-cap and 18/48 answer-cap contacts;
- `think512_program_slot`: 48/48 thought-cap and 0/48 answer-cap contacts; and
- both no-think arms: zero thought/answer-cap contacts.

Mechanics, qualification, confirmation, hidden, and benchmark reads stayed
empty. A frozen post-decision diagnostic removed only the exact decoded
`<|im_end|>\n` terminal suffix. The frozen parser then accepted 24/48
think/freeform, 38/48 think/`PROGRAM:`, and 48/48 in each no-think arm. A looser
expected-answer-tail diagnostic was 29/48, 48/48, 48/48, and 48/48 respectively,
but it hides five/ten additional thinking-close boundaries and is not full-
string exactness. Neither diagnostic changes the registered zero-parse result.

## Interpretation

The tested strict HF-model-EOS interface is invalid, so residualization remains
unadjudicated. The negative is not evidence that the no-think model lacked the
requested program: both no-think arms became 48/48 full-string exact after one
post hoc terminal-suffix removal. Thinking has an additional close-boundary
failure. The only warranted continuation is a new experiment that registers
first tokenizer EOS as the answer-stage deployment commit event on fresh tasks,
keeps strict parsing, and reruns matched HF-EOS and malformed-terminator
controls. No mechanics stage from this experiment may be opened.

## Knowledgebase Update

- Program evidence updated: yes; strict interface failure and tokenizer-EOS
  diagnostic recorded.
- Program backlog updated: yes; fresh commit-boundary successor required.
- Claim ledger updated: no; this is a single terminal interface result, not a
  capability claim.

## Artifacts

- `src/`
- `scripts/`
- `configs/`
- `runs/`
- `runs/calibration/decision.json`
- `analysis/calibration_terminal_diagnostic.json`
- `runs/smoke/summary.json`
- `reports/`
- `reports/artifact_manifest.yaml`
