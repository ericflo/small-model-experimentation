# Natural-Language State-Table Universal Curriculum

**Status:** in-progress · since 2026-07-14 · design frozen; control training awaits a pushed-green checkpoint

This result-separated successor tests whether truth-audited, variable-depth
natural-language state tables plus independent hypothesis scoring and a short
verified commit install a reusable reasoning procedure better than an exact-token
replay continuation from the same parent.

## Research Program

- Program: `agentic_breadth_installation`
- Program question: can one cleanly installed procedure improve every held-out
  benchmark family rather than redistribute wins?
- Prior anchors: `qwen35_4b_universal_search_scaffold_token_match`,
  `qwen_trace_procedure_depth_stress`, `qwen_constrained_abi_parser`, C37, and C38.

## Question

Does matching the training interface to variable-depth natural-language execution
teach the model to maintain explicit state, compare independently simulated
hypotheses, and stop with a concise answer—without sacrificing broad replay behavior?

## Hypothesis

The failed predecessor often reached the correct final state but did not commit, and
it regressed hypothesis selection. A truth-audited table that records each natural-
language transition should make execution inspectable; separate rows that score each
hypothesis on every probe should preserve discrimination; an answer-only commit after
verification should train the missing emission seam. The mechanism is false if the
candidate cannot beat both its parent and an exact-token replay control on a fresh
unchanged local gate.

## Setup

- Model: only `Qwen/Qwen3.5-4B`, pinned revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Parent: authenticated `close_xi` adapter from
  `qwen35_4b_universal_close_weight_token_match`; the failed scaffold adapter is not
  inherited.
- Dataset/task source: fresh deterministic procedural synthesis owned by this
  experiment. No benchmark source, item, transcript, or result detail may be read.
- Candidate: variable-depth natural-language state execution, independent hypothesis
  scoring, verification/repair, and concise commit lessons.
- Mechanism-falsifying control: same-parent replay continuation with identical forward
  tokens, optimizer steps, backend, seed, and position-aligned shared replay.
- Frozen arms: 320 rows and exactly 286,814 forward tokens each, zero skips, 40
  optimizer steps, and 200 byte-identical replay rows at the same positions. Candidate
  contains 80 curriculum rows plus 40 replay filler; control contains 120 replay rows.
- Primary admission: the inherited absolute local capability gate, a new explicit
  probe ≥0.50 check, and strict paired wins over parent and active replay both overall
  and on execute/induct/probe combined.
- Conditional broad admission: aggregate-only same-backend evaluation only after the
  sole candidate passes every local check; all reported families must improve before
  higher-tier confirmation or matched-compute sample-more.
- Reserved seeds: construction `77112`, training `46`, fresh local `88008`, and
  conditional aggregate `78138`.

## Run

Frozen smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B experiments/qwen35_4b_universal_state_table_compiler_token_match/scripts/run.py --smoke
```

The adversarial review passed and the harness now exposes exactly one expensive stage
per invocation: `train-control`, `train-candidate`, `local`, `merge`, or `benchmark`.
Each stage requires a clean worktree; every predecessor receipt must already be
committed at `HEAD`. Follow `reports/preregistration.md` and publish/CI-verify every
stage before starting the next.

## Results

CPU construction produced 80 truth-audited rows: 20 each execute, score, repair, and
commit. All 80 answers recompute from executable state; all score rows evaluate three
hypotheses on five probes; correct hypothesis position is balanced 7/7/6. Exact-token
materialization succeeded at 320 rows, 286,814 tokens, zero skips, and 200 aligned
replay positions per arm. The frozen smoke passes 48 tests. No model result, merge,
local capability event, or benchmark event exists.

## Interpretation

This design changes the interface, not the dose or close-token weight. Its novelty is
an executable natural-language state table connected to independent hypothesis scores
and a verified answer seam. Any later gain belongs to that package unless a frozen
ablation separates the pieces.

## Knowledgebase Update

- Program evidence: unchanged until a result exists.
- Program backlog: this result-separated successor is design-frozen.
- Claim ledger: unchanged.

## Artifacts

- `idea_intake.md`
- `configs/default.yaml`
- `data/design_receipt.json`
- `data/stream_token_receipt.json`
- `scripts/run.py`
- `reports/design_review.md`
- `reports/preregistration.md`
- `reports/report.md`
- `reports/artifact_manifest.yaml`
